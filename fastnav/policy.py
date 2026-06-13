"""Tiny MLP policy with built-in observation normalization (checkpoint-portable).

Action heads are a strategy, like fastnav.kinematics: a head owns its output
parameterization end to end -- the deterministic action, the BC training
criterion, and the PPO sampling distribution. Trainers consume `bc_loss` /
`sample_logp` / `log_prob` / `entropy` and never branch on the head type, so a
new parameterization (mixture density, action chunks, ...) is one new class.

  continuous  tanh-squashed 2-dim mean; BC = MSE; PPO = diagonal Gaussian.
  discrete_w  continuous v + K-bin categorical omega (argmax at deploy);
              BC = MSE(v) + cross-entropy(omega); PPO = Gaussian x categorical.
              Fixes MSE mode-averaging on the left/right turn decision that
              cripples diff-drive BC (47% -> 73% held-out at equal budget).
  flow_chunk  H-step action chunk via flow matching; samples one coherent
              maneuver per chunk (temporal mode consistency). BC-only.
  waypoint_flow  flow matching over the next H ego-frame POSITIONS (the
              expert's future, sim.expert_waypoints) + built-in follower that
              tracks the plan as diffdrive_vel commands. The deployable
              trajectory head: latency-tolerant, safety-gateable. BC-only.

Memory cores are a strategy the same way: a core owns temporal processing in
both regimes -- `unroll` over training sequences (episode-boundary resets via
done_seq) and `step` at deployment, where the carried state is one flat
[N, core.state_size] array that consumers allocate with zeros and reset by
zeroing (both cores define the zero state as a cold start).

  gru          self-excited recurrence h' = f(h, x); state = the hidden itself,
               chunk-start hiddens stored (h0_size = width) as BPTT init.
  transformer  bounded sliding-window attention (pre-LN, ALiBi); state = step
               counter + per-layer KV ring. Nothing persists past the window,
               so the closed loop cannot hold a corrupted internal attractor
               (the near-goal limit-cycle failure family); h0_size = 0 --
               training chunks start cold and warm up inside the window.

PPO's state-independent exploration scale `log_std` covers the head's
continuous dims only (`n_continuous`); it is owned by PPONavPolicy and passed
into the distribution methods.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from fastnav import kinematics
from fastnav.sim import SimConfig

_LOG_2PI = math.log(2.0 * math.pi)


def _gaussian_logp(x: mx.array, mean: mx.array, log_std: mx.array) -> mx.array:
    """Diagonal Gaussian log-prob summed over the trailing dim."""
    z = (x - mean) * mx.exp(-log_std)
    return mx.sum(-0.5 * mx.square(z) - log_std - 0.5 * _LOG_2PI, axis=-1)


class ContinuousHead(nn.Linear):
    """tanh-squashed 2-dim action. Subclasses nn.Linear so parameters keep the
    original `head.weight`/`head.bias` layout -- existing checkpoints load
    unchanged."""

    n_continuous = 2  # dims covered by PPO's log_std
    state_size = 0    # deploy-time head state (chunk heads carry one; see step())
    label_kind = "action"  # what trainers store as the BC target (see label_dim)
    label_dim = 2     # per-step BC target width trainers allocate buffers for

    def __init__(self, hidden: int, act_scale: tuple[float, float]):
        super().__init__(hidden, 2)
        self.act_scale = tuple(act_scale)  # plain tuple: constant, not a parameter

    def mean(self, h: mx.array) -> mx.array:
        return mx.array(self.act_scale) * mx.tanh(super().__call__(h))

    def act(self, h: mx.array, state: mx.array) -> tuple[mx.array, mx.array]:
        return self.mean(h), state

    def bc_loss(self, h: mx.array, target: mx.array, done: mx.array | None = None) -> mx.array:
        """Per-step imitation loss [...]; target [..., 2]. `done` accepted for a
        uniform head interface (trajectory heads use it); ignored here."""
        return mx.mean(mx.square(self.mean(h) - target), axis=-1)

    def sample_logp(self, h: mx.array, log_std: mx.array) -> tuple[mx.array, mx.array]:
        mean = self.mean(h)
        a = mean + mx.exp(log_std) * mx.random.normal(mean.shape)
        return a, _gaussian_logp(a, mean, log_std)

    def log_prob(self, h: mx.array, act: mx.array, log_std: mx.array) -> mx.array:
        return _gaussian_logp(act, self.mean(h), log_std)

    def entropy(self, h: mx.array, log_std: mx.array) -> mx.array:
        return mx.sum(log_std) + 0.5 * self.n_continuous * (1.0 + _LOG_2PI)


class DiscreteOmegaHead(nn.Module):
    """Continuous first action dim (tanh mean) + categorical second dim over
    `bins` evenly spaced values in [-scale1, scale1]; argmax at deploy.

    Cross-entropy training keeps probability mass on the true turn modes
    instead of regressing toward their useless average."""

    n_continuous = 1
    state_size = 0
    label_kind = "action"
    label_dim = 2

    def __init__(self, hidden: int, act_scale: tuple[float, float], bins: int = 15):
        super().__init__()
        self.act_scale = tuple(act_scale)
        self.bins = bins
        self.vlin = nn.Linear(hidden, 1)
        self.wlin = nn.Linear(hidden, bins)

    def _bin_values(self) -> mx.array:
        return mx.linspace(-self.act_scale[1], self.act_scale[1], self.bins)

    def _bin_index(self, w: mx.array) -> mx.array:
        s = self.act_scale[1]
        return mx.clip(mx.round((w + s) / (2 * s) * (self.bins - 1)),
                       0, self.bins - 1).astype(mx.int32)

    def _v_mean(self, h: mx.array) -> mx.array:
        return self.act_scale[0] * mx.tanh(self.vlin(h)[..., 0])

    def act(self, h: mx.array, state: mx.array) -> tuple[mx.array, mx.array]:
        w = self._bin_values()[mx.argmax(self.wlin(h), axis=-1)]
        return mx.stack([self._v_mean(h), w], axis=-1), state

    def bc_loss(self, h: mx.array, target: mx.array, done: mx.array | None = None) -> mx.array:
        ce = nn.losses.cross_entropy(self.wlin(h), self._bin_index(target[..., 1]),
                                     reduction="none")
        return mx.square(self._v_mean(h) - target[..., 0]) + ce

    def sample_logp(self, h: mx.array, log_std: mx.array) -> tuple[mx.array, mx.array]:
        v_mean = self._v_mean(h)
        v = v_mean + mx.exp(log_std[0]) * mx.random.normal(v_mean.shape)
        logits = self.wlin(h)
        idx = mx.random.categorical(logits)
        a = mx.stack([v, self._bin_values()[idx]], axis=-1)
        return a, self.log_prob(h, a, log_std)

    def log_prob(self, h: mx.array, act: mx.array, log_std: mx.array) -> mx.array:
        logp_v = _gaussian_logp(act[..., :1], self._v_mean(h)[..., None], log_std)
        logp_w = mx.take_along_axis(nn.log_softmax(self.wlin(h), axis=-1),
                                    self._bin_index(act[..., 1])[..., None], axis=-1)[..., 0]
        return logp_v + logp_w

    def entropy(self, h: mx.array, log_std: mx.array) -> mx.array:
        logp = nn.log_softmax(self.wlin(h), axis=-1)
        ent_w = -mx.sum(mx.exp(logp) * logp, axis=-1)
        return ent_w + mx.sum(log_std) + 0.5 * (1.0 + _LOG_2PI)


class FlowChunkHead(nn.Module):
    """H-step action chunk via conditional flow matching (rectified flow).

    Targets the limit-cycle failure family at the policy-class level: the task
    has states where two maneuvers are both valid (geodesic ties), and any
    per-step readout mixes them in closed loop (dither). This head models the
    JOINT distribution over the next H actions and samples ONE coherent,
    time-varying maneuver -- mode selection happens once per chunk.

    BC: velocity-matching loss v(x_t, t | feat) -> (x1 - x0) on linear paths,
    where x1 is the expert's next-H action window (built in-head from the
    per-step label sequence trainers already store; incomplete tail windows
    are masked). Draws noise inside bc_loss -- trainers thread mx.random.state
    through their compiled updates to keep that sound.

    Deploy: Euler-integrate a chunk when none is in flight, then execute it
    open-loop via the head's deploy state [steps_left | chunk] (zeroed state =
    no chunk = resample, so episode resets Just Work). Stochastic by design:
    the x0 draw is the mode choice.

    PPO: chunk log-probs are intractable -- sample_logp/log_prob/entropy raise.
    The bc_loss path still works, so a BC-anchored-only PPO variant remains
    possible later; for now this head is BC-only.
    """

    n_continuous = 0  # no PPO log_std dims (BC-only head)
    label_kind = "action"  # per-step labels, windowed in-head into chunks
    label_dim = 2

    def __init__(self, hidden: int, act_scale: tuple[float, float], horizon: int = 8,
                 euler_steps: int = 8, width: int = 256, token_frac: float = 1.0,
                 exec_len: int | None = None):
        super().__init__()
        self.act_scale = tuple(act_scale)
        self.horizon = horizon
        self.euler_steps = euler_steps
        self.exec_len = exec_len or horizon  # execute first k of H before resampling:
        # the commitment/reactivity dial (1 = receding horizon, H = open loop).
        # Deploy-only -- sweepable on a trained policy by setting the attribute.
        self.token_frac = token_frac  # train the flow loss on this fraction of
        # sequence positions per update (unbiased; adjacent windows are highly
        # correlated, so subsampling buys ~1/frac update speed for little signal)
        d = 2 * horizon
        self.inp = nn.Linear(d + 1 + hidden, width)
        self.mid = nn.Linear(width, width)
        self.out = nn.Linear(width, d)

    @property
    def state_size(self) -> int:
        return 1 + 2 * self.horizon  # [steps_left | chunk (action units)]

    def _velocity(self, x: mx.array, t: mx.array, feat: mx.array) -> mx.array:
        z = nn.silu(self.inp(mx.concatenate([x, t, feat], axis=-1)))
        return self.out(nn.silu(self.mid(z)))

    def _sample_chunk(self, feat: mx.array) -> mx.array:
        """[N, F] -> [N, 2H] in action units (one Euler-integrated flow sample)."""
        n = feat.shape[0]
        x = mx.random.normal((n, 2 * self.horizon))
        for i in range(self.euler_steps):
            t = mx.full((n, 1), (i + 0.5) / self.euler_steps)
            x = x + self._velocity(x, t, feat) / self.euler_steps
        return (x.reshape(n, self.horizon, 2) * mx.array(self.act_scale)).reshape(n, -1)

    def act(self, feat: mx.array, state: mx.array) -> tuple[mx.array, mx.array]:
        n = feat.shape[0]
        left = state[:, :1].astype(mx.float32)
        chunk = state[:, 1:].astype(mx.float32)
        need = left <= 0.5  # no chunk in flight (fresh episode or chunk exhausted)
        chunk = mx.where(need, self._sample_chunk(feat), chunk)
        left = mx.where(need, float(self.exec_len), left)
        pos = (self.exec_len - left).astype(mx.int32)[:, :, None]  # [N,1,1] in [0, exec_len)
        a = mx.take_along_axis(chunk.reshape(n, self.horizon, 2),
                               mx.broadcast_to(pos, (n, 1, 2)), axis=1)[:, 0]
        return a, mx.concatenate([left - 1.0, chunk], axis=1)

    def bc_loss(self, feat: mx.array, target: mx.array, done: mx.array | None = None) -> mx.array:
        """Per-step flow-matching loss [B, T]; target [B, T, 2] per-step labels,
        windowed in-head into next-H chunks (incomplete tails weighted 0).
        `done` accepted for interface uniformity; unused (chunks are self-contained)."""
        h = self.horizon
        b, t_len, _ = target.shape
        idx = mx.arange(t_len)[:, None] + mx.arange(h)[None, :]
        w_full = (idx[:, -1] < t_len).astype(mx.float32)  # [T]
        chunks = mx.take(target, mx.minimum(idx, t_len - 1), axis=1)  # [B, T, H, 2]
        x1 = (chunks / mx.array(self.act_scale)).reshape(b, t_len, 2 * h)
        if self.token_frac < 1.0:
            # gather BEFORE the velocity net so the subsample actually buys
            # compute; shared positions per update, importance-corrected scatter
            k = max(1, int(t_len * self.token_frac))
            pos = mx.random.permutation(t_len)[:k]
            x1_s, feat_s, w_s = x1[:, pos], feat[:, pos], w_full[pos]
            x0 = mx.random.normal(x1_s.shape)
            t = mx.random.uniform(shape=(b, k, 1))
            v = self._velocity(t * x1_s + (1.0 - t) * x0, t, feat_s)
            loss_s = mx.mean(mx.square(v - (x1_s - x0)), axis=-1) * w_s[None, :]
            return mx.put_along_axis(mx.zeros((b, t_len)), mx.broadcast_to(pos[None], (b, k)),
                                     loss_s / self.token_frac, axis=1)
        x0 = mx.random.normal(x1.shape)
        t = mx.random.uniform(shape=(b, t_len, 1))
        v = self._velocity(t * x1 + (1.0 - t) * x0, t, feat)
        return mx.mean(mx.square(v - (x1 - x0)), axis=-1) * w_full[None, :]

    def sample_logp(self, *_):
        raise NotImplementedError("flow_chunk is BC-only: chunk log-probs are intractable")

    def log_prob(self, *_):
        raise NotImplementedError("flow_chunk is BC-only: chunk log-probs are intractable")

    def entropy(self, *_):
        raise NotImplementedError("flow_chunk is BC-only: chunk log-probs are intractable")


class WaypointFlowHead(nn.Module):
    """Ego-frame trajectory head: flow matching over the next `horizon`
    positions (sensor frame at plan time, `stride` sim steps apart), plus a
    built-in receding-horizon follower that turns the in-flight plan into
    diffdrive_vel body-velocity commands. Designed for the diffdrive_vel
    kinematics; the real robot runs the same follower against its own local
    odometry, with the predicted trajectory exposed for safety gating.

    vs flow_chunk: the target is WHERE the expert goes (sim.expert_waypoints),
    not what it outputs -- labels are complete per-state futures, so there is
    no in-head windowing, no incomplete-tail mask, and supervision lives in
    bounded position space instead of compounding action space.

    Deploy state [left | dead-reckoned pose x,y,th | chunk (waypoint units)]:
    a sampled plan executes open-loop for `replan` steps, tracked by
    integrating the issued commands through the same P-steering + unicycle
    model the sim applies (noise-free mirror of diffdrive_vel's kin_execute),
    aiming one waypoint interval ahead along the time-parameterized plan.
    Zeroed state = no plan = resample, so episode resets Just Work.

    BC-only, like flow_chunk (chunk log-probs are intractable)."""

    n_continuous = 0
    label_kind = "waypoint"  # trainers store sim.expert_waypoints(horizon, stride)

    # follower constants -- mirror _DIFFDRIVE_VEL_METAL's KIN_TGAIN and the
    # backward-cone turn-direction convention; keep in sync
    TGAIN = 4.0
    BACK_CONE = -2.618

    def __init__(self, hidden: int, act_scale: tuple[float, float], horizon: int = 8,
                 stride: int = 2, dt: float = 0.1, w_max: float = 2.5,
                 euler_steps: int = 8, width: int = 256, token_frac: float = 1.0,
                 replan: int | None = None, cond_dropout: float = 0.25,
                 cond_noise: float = 1.5, conditioned: bool = True,
                 kinematics: str = "diffdrive_vel"):
        super().__init__()
        self.conditioned = conditioned  # False = legacy unconditioned head (no prev-plan
        # input, narrower inp, no prev_cond in deploy state). Lets pre-conditioning
        # checkpoints load; new training defaults to True.
        # action-output convention: diffdrive_vel emits a body-velocity (vx, vy)
        # that the sim kernel's P-steering converts to (v, omega); diffdrive emits
        # (v, omega) DIRECTLY -- so the follower owns the steering controller and
        # converts the pursued-waypoint velocity to (v, omega) via _steer.
        self.kinematics = kinematics
        self.act_scale = tuple(act_scale)
        self.horizon = horizon
        self.stride = stride
        self.dt = dt
        self.w_max = w_max
        self.euler_steps = euler_steps
        self.token_frac = token_frac
        self.cond_dropout = cond_dropout  # prob of zeroing the prev-plan conditioning
        # in training, so the head can still cold-start / override a bad prior
        self.cond_noise = cond_noise  # std of Gaussian noise added to the (unit-var)
        # conditioning so it conveys coarse mode, not the exact answer (anti-leak)
        self.replan = replan or stride  # commitment dial: re-plan every `stride`
        # steps (one waypoint interval). NOTE the follower itself is best at
        # replan=horizon -- fed EXPERT plans, pure pursuit scores 100%/0%coll at
        # replan=8 but only 58%/0% at replan=1 (long lookahead helps when plans
        # are good). Short replan wins for a TRAINED net only because its plans
        # are imperfect and drift into walls over open-loop steps (BC ckpt:
        # replan=2 24.8%/53%coll vs replan=8 19.8%/77%). So this default trades
        # follower optimality for robustness to plan error -- and once PPO
        # cleans up plan quality, longer replan may become preferable. Re-sweep
        # after PPO. Deploy-only, sweepable by setting the attribute; frequent
        # replan also matches the real-robot need to react to the live scan.
        # the flow models per-step DELTAS, not absolute waypoints: absolute
        # supervision lets far-waypoint error compound freely (measured: wp1 err
        # 0.08m -> wp8 err 0.82m, plans veering into walls), while deltas all
        # share one scale. _sample_plan cumsums deltas to absolute waypoints;
        # bc_loss differences the label.
        #
        # CRITICAL: standardize deltas to ~UNIT VARIANCE so the flow target
        # matches the x0~N(0,1) prior. Dividing by the MAX step (v_max*dt*stride)
        # gives target std ~0.49 -- half the noise scale, so the velocity target
        # x1-x0 is noise-dominated and the model collapses to predicting ~-x0
        # (=> ~zero plans, the BC failure we hit at 35%). The empirical expert
        # delta std is ~0.49 of the max step, so divide by that to land at std~1.
        self._step_scale = float(act_scale[0] * dt * stride * 0.49)  # unit-variance target
        d = 2 * horizon
        # +d input channels (conditioned only) = the previous plan (delta space) as
        # conditioning, so the flow learns coherent continuations; zeros = cold start
        self.inp = nn.Linear(d + 1 + hidden + (d if conditioned else 0), width)
        self.mid = nn.Linear(width, width)
        self.out = nn.Linear(width, d)

    @property
    def label_dim(self) -> int:
        return 2 * self.horizon

    @property
    def state_size(self) -> int:
        # [left | est x,y,th | chunk (| prev-cond if conditioned)]
        return 4 + 2 * self.horizon + (2 * self.horizon if self.conditioned else 0)

    def _velocity(self, x: mx.array, t: mx.array, feat: mx.array,
                  cond: mx.array | None = None) -> mx.array:
        parts = [x, t, feat] + ([cond] if self.conditioned else [])
        z = nn.silu(self.inp(mx.concatenate(parts, axis=-1)))
        return self.out(nn.silu(self.mid(z)))

    def _to_waypoints(self, deltas_norm: mx.array) -> mx.array:
        """[N, 2H] normalized per-step deltas -> [N, 2H] absolute ego waypoints
        (meters), by un-scaling and cumulative-summing along the horizon."""
        n = deltas_norm.shape[0]
        d = (deltas_norm * self._step_scale).reshape(n, self.horizon, 2)
        return mx.cumsum(d, axis=1).reshape(n, 2 * self.horizon)

    def _to_deltas(self, wp: mx.array) -> mx.array:
        """[B..., 2H] absolute ego waypoints -> [B..., 2H] normalized per-step
        deltas (the flow-matching target). wp_0 is the first delta from origin."""
        shp = wp.shape
        w = wp.reshape(*shp[:-1], self.horizon, 2)
        prev = mx.concatenate([mx.zeros((*shp[:-1], 1, 2)), w[..., :-1, :]], axis=-2)
        return ((w - prev) / self._step_scale).reshape(shp)

    def _sample_plan(self, feat: mx.array, cond: mx.array | None = None) -> mx.array:
        """[N, F] -> [N, 2H] absolute ego waypoints in meters (one Euler flow
        sample in delta space, integrated to positions). `cond` [N, 2H] = the
        previous plan in delta space (None = cold start = zeros)."""
        n = feat.shape[0]
        if cond is None and self.conditioned:
            cond = mx.zeros((n, 2 * self.horizon))
        x = mx.random.normal((n, 2 * self.horizon))
        for i in range(self.euler_steps):
            t = mx.full((n, 1), (i + 0.5) / self.euler_steps)
            x = x + self._velocity(x, t, feat, cond) / self.euler_steps
        return self._to_waypoints(x)

    def _steer(self, a: mx.array) -> tuple[mx.array, mx.array]:
        """P-steering: body-velocity command `a` [N, 2] -> (v, omega). Mirrors
        _DIFFDRIVE_VEL_METAL kin_execute (P-gain TGAIN, cos^4 speed gate,
        backward-cone fix). On diffdrive this IS the follower's controller; on
        diffdrive_vel the sim kernel applies the identical conversion."""
        alpha = mx.arctan2(a[:, 1], a[:, 0])
        alpha = mx.where(alpha < self.BACK_CONE, alpha + 2.0 * math.pi, alpha)
        wz = mx.clip(self.TGAIN * alpha, -self.w_max, self.w_max)
        vn = mx.minimum(mx.sqrt(mx.sum(mx.square(a), axis=1)), self.act_scale[0])
        v = vn * mx.maximum(mx.cos(alpha), 0.0) ** 4
        return v, wz

    def _dead_reckon(self, est: mx.array, a: mx.array) -> mx.array:
        """Advance the plan-frame pose estimate by one noise-free step of body-
        velocity command `a` (P-steering via _steer + SUB=2 unicycle, no
        collision). Identical for both kinematics: diffdrive executes (v,omega)
        directly, diffdrive_vel's kernel produces the same (v,omega)."""
        v, wz = self._steer(a)
        x, y, th = est[:, 0], est[:, 1], est[:, 2]
        for _ in range(2):
            th = th + wz * self.dt / 2
            x = x + mx.cos(th) * v * self.dt / 2
            y = y + mx.sin(th) * v * self.dt / 2
        return mx.stack([x, y, th], axis=1)

    def act(self, feat: mx.array, state: mx.array) -> tuple[mx.array, mx.array]:
        n = feat.shape[0]
        h2 = 2 * self.horizon
        left = state[:, :1].astype(mx.float32)
        est = state[:, 1:4].astype(mx.float32)
        chunk = state[:, 4:4 + h2].astype(mx.float32)
        prev_cond = state[:, 4 + h2:].astype(mx.float32) if self.conditioned else None
        need = left <= 0.5  # no plan in flight (fresh episode or plan exhausted)
        # condition the resample on the previous plan (zeros after a reset, since
        # the state -- hence prev_cond -- was zeroed); learned coherent continuation
        new_chunk = self._sample_plan(feat, prev_cond)
        chunk = mx.where(need, new_chunk, chunk)
        if self.conditioned:
            # carry the (delta-space) committed plan as next frame's conditioning
            prev_cond = mx.where(need, self._to_deltas(new_chunk), prev_cond)
        est = mx.where(need, mx.zeros_like(est), est)
        left = mx.where(need, float(self.replan), left)
        # time-parameterized pursuit: aim one waypoint interval ahead of the
        # current plan time, linearly interpolating between waypoints
        j = (self.replan - left)[:, 0]  # steps already executed on this plan
        tt = mx.minimum(j + float(self.stride), float(self.horizon * self.stride))
        u = mx.clip(tt / self.stride - 1.0, 0.0, float(self.horizon - 1))
        lo = mx.floor(u)
        frac = (u - lo)[:, None]
        w = chunk.reshape(n, self.horizon, 2)
        lo_i = lo.astype(mx.int32)[:, None, None]
        w_lo = mx.take_along_axis(w, mx.broadcast_to(lo_i, (n, 1, 2)), axis=1)[:, 0]
        hi_i = mx.minimum(lo_i + 1, self.horizon - 1)
        w_hi = mx.take_along_axis(w, mx.broadcast_to(hi_i, (n, 1, 2)), axis=1)[:, 0]
        tgt = w_lo * (1.0 - frac) + w_hi * frac  # plan frame
        # rotate the offset into the current dead-reckoned body frame
        c, s = mx.cos(est[:, 2]), mx.sin(est[:, 2])
        rx, ry = tgt[:, 0] - est[:, 0], tgt[:, 1] - est[:, 1]
        rel = mx.stack([c * rx + s * ry, -s * rx + c * ry], axis=1)
        a = rel / mx.maximum((tt - j)[:, None] * self.dt, self.dt)
        nrm = mx.maximum(mx.sqrt(mx.sum(mx.square(a), axis=1, keepdims=True)), 1e-6)
        a = a * mx.minimum(1.0, self.act_scale[0] / nrm)
        est = self._dead_reckon(est, a)  # est always integrates the body-velocity command
        # emit the action in the kinematics' action space: diffdrive_vel takes the
        # body velocity directly (kernel P-steers it); diffdrive takes (v, omega),
        # so the follower converts here (it owns the steering controller)
        if self.kinematics == "diffdrive":
            v, wz = self._steer(a)
            out = mx.stack([v, wz], axis=1)
        else:
            out = a
        parts = [left - 1.0, est, chunk] + ([prev_cond] if self.conditioned else [])
        return out, mx.concatenate(parts, axis=1)

    def bc_loss(self, feat: mx.array, target: mx.array, done: mx.array | None = None) -> mx.array:
        """Per-step flow-matching loss [B, T]; target [B, T, 2H] = complete
        ego-waypoint labels (no windowing, no tail mask). Conditioning at step t
        is the PREVIOUS step's plan (delta space): cond[t] = to_deltas(target[t-1]),
        zeroed at t=0 and across episode boundaries (`done` [B, T, 1]), then
        cond_dropout'd so the head learns to cold-start too."""
        b, t_len, _ = target.shape
        x1 = self._to_deltas(target)  # flow target = normalized per-step deltas
        if not self.conditioned:
            x0 = mx.random.normal(x1.shape)
            t = mx.random.uniform(shape=(b, t_len, 1))
            if self.token_frac < 1.0:
                k = max(1, int(t_len * self.token_frac))
                pos = mx.random.permutation(t_len)[:k]
                x1_s, feat_s, x0_s = x1[:, pos], feat[:, pos], x0[:, pos]
                t_s = mx.random.uniform(shape=(b, k, 1))
                v = self._velocity(t_s * x1_s + (1.0 - t_s) * x0_s, t_s, feat_s)
                loss_s = mx.mean(mx.square(v - (x1_s - x0_s)), axis=-1)
                return mx.put_along_axis(mx.zeros((b, t_len)), mx.broadcast_to(pos[None], (b, k)),
                                         loss_s / self.token_frac, axis=1)
            v = self._velocity(t * x1 + (1.0 - t) * x0, t, feat)
            return mx.mean(mx.square(v - (x1 - x0)), axis=-1)
        # cond[t] = x1[t-1] shifted forward, cold-start zeros at t=0; an env that
        # was `done` at t-1 starts a fresh episode at t, so its cond is zeros too
        cond = mx.concatenate([mx.zeros((b, 1, 2 * self.horizon)), x1[:, :-1]], axis=1)
        if done is not None:
            prev_done = mx.concatenate([mx.ones((b, 1, 1)), done[:, :-1]], axis=1)
            cond = cond * (1.0 - prev_done)
        # conditioning CORRUPTION: the expert prev-plan overlaps the current plan
        # by ~0.07m, so a clean cond lets the head shortcut ("shift the cond") and
        # never learn to plan from feat -- which collapses at deploy where cond is
        # the head's OWN imperfect plan (measured: dropout 1.0->52%, 0.5->4%,
        # 0.1->0%, monotone). Heavy noise keeps the COARSE intent (left/right mode)
        # while destroying the precise answer, forcing feat-based planning.
        if self.cond_noise > 0:
            cond = cond + self.cond_noise * mx.random.normal(cond.shape)
        # conditioning dropout: random per (env, step), so cold-start is in-dist
        if self.cond_dropout > 0:
            keep = (mx.random.uniform(shape=(b, t_len, 1)) >= self.cond_dropout).astype(cond.dtype)
            cond = cond * keep
        if self.token_frac < 1.0:
            # gather BEFORE the velocity net so the subsample buys compute
            # (same scheme as flow_chunk)
            k = max(1, int(t_len * self.token_frac))
            pos = mx.random.permutation(t_len)[:k]
            x1_s, feat_s, cond_s = x1[:, pos], feat[:, pos], cond[:, pos]
            x0 = mx.random.normal(x1_s.shape)
            t = mx.random.uniform(shape=(b, k, 1))
            v = self._velocity(t * x1_s + (1.0 - t) * x0, t, feat_s, cond_s)
            loss_s = mx.mean(mx.square(v - (x1_s - x0)), axis=-1)
            return mx.put_along_axis(mx.zeros((b, t_len)), mx.broadcast_to(pos[None], (b, k)),
                                     loss_s / self.token_frac, axis=1)
        x0 = mx.random.normal(x1.shape)
        t = mx.random.uniform(shape=(b, t_len, 1))
        v = self._velocity(t * x1 + (1.0 - t) * x0, t, feat, cond)
        return mx.mean(mx.square(v - (x1 - x0)), axis=-1)

    def sample_logp(self, *_):
        raise NotImplementedError("waypoint_flow is BC-only: plan log-probs are intractable")

    def log_prob(self, *_):
        raise NotImplementedError("waypoint_flow is BC-only: plan log-probs are intractable")

    def entropy(self, *_):
        raise NotImplementedError("waypoint_flow is BC-only: plan log-probs are intractable")


HEADS = {"continuous": ContinuousHead, "discrete_w": DiscreteOmegaHead,
         "flow_chunk": FlowChunkHead, "waypoint_flow": WaypointFlowHead}


class GRUCore(nn.GRU):
    """GRU memory core. Subclasses nn.GRU (like ContinuousHead/nn.Linear) so
    parameters keep the historical `gru.*` checkpoint key layout."""

    attr = "gru"  # policy attribute name = checkpoint key prefix

    def __init__(self, in_dim: int, width: int):
        super().__init__(in_dim, width)
        self.width = width

    @property
    def state_size(self) -> int:
        return self.width

    @property
    def h0_size(self) -> int:
        return self.width

    state_dtype = mx.float32

    def step(self, x: mx.array, state: mx.array) -> tuple[mx.array, mx.array]:
        h = super().__call__(x[:, None, :], hidden=state)[:, -1, :]
        return h, h

    def unroll(self, x: mx.array, h0: mx.array, done_seq: mx.array) -> mx.array:
        """[B, T, in_dim] -> features [B, T, width]; hidden resets after done."""
        h = h0
        hs = []
        for t in range(x.shape[1]):
            if t > 0:
                h = h * (1.0 - done_seq[:, t - 1])  # done_seq [B, T, 1] -> [B, 1]
            h = super().__call__(x[:, t][:, None, :], hidden=h)[:, -1, :]
            hs.append(h)
        return mx.stack(hs, axis=1)


def _heads_split(x: mx.array, heads: int) -> mx.array:
    """[B, T, W] -> [B, H, T, W/H]"""
    b, t, w = x.shape
    return x.reshape(b, t, heads, w // heads).transpose(0, 2, 1, 3)


def _heads_merge(x: mx.array) -> mx.array:
    """[B, H, T, W/H] -> [B, T, W]"""
    b, h, t, dh = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, t, h * dh)


class _TfmBlock(nn.Module):
    """Pre-LN causal attention + SiLU MLP. `mask` is additive ([..., Tq, Tk]):
    ALiBi bias on attendable pairs, -1e9 elsewhere."""

    def __init__(self, width: int, heads: int, mlp_mult: int):
        super().__init__()
        self.heads = heads
        self.scale = (width // heads) ** -0.5
        self.ln1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)
        self.up = nn.Linear(width, mlp_mult * width)
        self.down = nn.Linear(mlp_mult * width, width)

    def _mlp(self, x: mx.array) -> mx.array:
        return x + self.down(nn.silu(self.up(self.ln2(x))))

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        q, k, v = (_heads_split(z, self.heads) for z in
                   mx.split(self.qkv(self.ln1(x)), 3, axis=-1))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self._mlp(x + self.out(_heads_merge(o)))

    def step(self, x: mx.array, k_cache: mx.array, v_cache: mx.array,
             mask: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """One token [N, W] attending over the UNMODIFIED ring [N, K, W] plus
        itself as an appended position (so the layer pass never rewrites the
        ring -- the core commits all layers' new rows in one scatter).
        Ring-sized tensors stay in the ring's dtype (fp16: the ring IS the
        step's memory traffic); token-sized tensors stay fp32, and the softmax
        runs fp32 via the mask add. Returns (x, new k row, new v row)."""
        rd = k_cache.dtype
        q, k, v = mx.split(self.qkv(self.ln1(x)), 3, axis=-1)
        qh = _heads_split(q[:, None, :], self.heads)  # [N, H, 1, dh]
        ring = qh.astype(rd) @ _heads_split(k_cache, self.heads).transpose(0, 1, 3, 2) * self.scale
        own = mx.sum(qh * _heads_split(k[:, None, :], self.heads), axis=-1,
                     keepdims=True) * self.scale  # ALiBi bias for self is 0
        wts = mx.softmax(mx.concatenate([ring + mask, own], axis=-1), axis=-1)
        o = ((wts[..., :-1].astype(rd) @ _heads_split(v_cache, self.heads)).astype(x.dtype)
             + wts[..., -1:] * _heads_split(v[:, None, :], self.heads))
        return self._mlp(x + self.out(_heads_merge(o)[:, 0])), k.astype(rd), v.astype(rd)


class TransformerCore(nn.Module):
    """Sliding-window attention core. The deploy state is a step counter plus a
    per-layer KV ring -- bounded context by construction, and `state * 0` is a
    correct cold reset because slot validity is derived from the counter.
    ALiBi supplies relative position (no embedding tables, window-size-free).

    The default window (64) matches the maneuver horizon the BPTT-64 result
    established. The deploy ring is fp16 -- the state is the policy's memory
    traffic, so halving it halves step cost; training unrolls stay fp32. fp16
    rather than bf16 because the ring carries the step counter, and fp16 is
    integer-exact to 2048 (bf16 only to 256 -- episodes are 512 steps)."""

    attr = "tfm"
    state_dtype = mx.float16

    def __init__(self, in_dim: int, width: int, layers: int = 3, heads: int = 4,
                 context: int = 64, mlp_mult: int = 4):
        super().__init__()
        if in_dim != width:
            raise ValueError("transformer core requires enc width == core width")
        self.width = width
        self.heads = heads
        self.context = context
        self.blocks = [_TfmBlock(width, heads, mlp_mult) for _ in range(layers)]
        self.ln_f = nn.LayerNorm(width)
        self._compiled_step = None  # built on first step(); not part of module state

    @property
    def state_size(self) -> int:
        return 1 + len(self.blocks) * 2 * self.context * self.width

    @property
    def h0_size(self) -> int:
        return 0

    def _slopes(self) -> mx.array:
        h = self.heads
        return mx.array([2.0 ** (-8.0 * (i + 1) / h) for i in range(h)])

    def unroll(self, x: mx.array, h0: mx.array, done_seq: mx.array) -> mx.array:
        """[B, T, W] -> features [B, T, W]; attention is causal, windowed, and
        never crosses episode boundaries. h0 is unused: chunks start cold and
        the leading (burn-in) steps warm the context instead of a hidden."""
        del h0
        b, t_len, _ = x.shape
        done = done_seq[..., 0]
        eid = mx.cumsum(done, axis=1)  # episode id per token (done resets BEFORE next step)
        eid = mx.concatenate([mx.zeros((b, 1)), eid[:, :-1]], axis=1)
        rel = mx.arange(t_len)[:, None] - mx.arange(t_len)[None, :]  # t - s
        allowed = ((rel >= 0) & (rel < self.context))[None, None]
        allowed = mx.logical_and(allowed, (eid[:, :, None] == eid[:, None, :])[:, None])
        bias = (-self._slopes()[:, None, None] * rel)[None]
        mask = mx.where(allowed, bias, -1e9)  # [B, H, T, T]
        for blk in self.blocks:
            x = blk(x, mask)
        return self.ln_f(x)

    def step(self, x: mx.array, state: mx.array) -> tuple[mx.array, mx.array]:
        # the eager step graph is ~20 ops/block over a [N, K, W] ring -- compiled
        # once (per shape), it replays without python graph-building and can
        # donate the state buffer instead of reallocating it every step
        if self._compiled_step is None:
            self._compiled_step = mx.compile(self._step, inputs=[self.state])
        return self._compiled_step(x, state)

    def _step(self, x: mx.array, state: mx.array) -> tuple[mx.array, mx.array]:
        n = x.shape[0]
        k, w = self.context, self.width
        t = state[:, :1]  # steps taken so far in this episode
        kv = state[:, 1:].reshape(n, len(self.blocks), 2, k, w)
        slot = (t % k).astype(mx.int32)  # ring slot this step will occupy
        # pre-write ring ages: slot j holds step t - a, a = ((slot-1-j) mod K) + 1;
        # usable iff within both the window (a <= K-1; the token itself is the
        # K'th position, appended in the block) and this episode (a <= t)
        a = ((slot - 1 - mx.arange(k)[None, :]) % k).astype(mx.float32) + 1.0
        ok = a <= mx.minimum(t, float(k - 1))
        mask = mx.where(ok[:, None, :], -self._slopes()[None, :, None] * a[:, None, :],
                        -1e9)[:, :, None, :]  # [N, H, 1, K]
        rows = [t + 1.0]
        for i, blk in enumerate(self.blocks):
            x, k_new, v_new = blk.step(x, kv[:, i, 0], kv[:, i, 1], mask)
            rows.extend([k_new, v_new])
        # one scatter commits the counter + every layer's new (k, v) ring row
        base = mx.arange(2 * len(self.blocks)).astype(mx.int32) * k  # row offsets in [L*2, K] units
        idx = 1 + (base[None, :, None] + slot[:, :, None]) * w + mx.arange(w)[None, None, :]
        idx = mx.concatenate([mx.zeros((n, 1), dtype=mx.int32), idx.reshape(n, -1)], axis=1)
        state = mx.put_along_axis(state, idx, mx.concatenate(rows, axis=1), axis=1)
        return self.ln_f(x), state


CORES = {"gru": GRUCore, "transformer": TransformerCore}


def _scales(cfg: SimConfig) -> tuple[tuple[float, float], list[float]]:
    """(per-dim action scale, per-dim obs scale w/o the prev-action tail).

    The action scale is kept as a plain tuple (not an mx.array attribute) so it
    stays a constant rather than a learnable parameter, and checkpoints keep the
    same key set across kinematics."""
    act_scale = tuple(float(s) for s in kinematics.get(cfg.kinematics).action_scale(cfg))
    return act_scale, [1.0 / cfg.max_range] * (cfg.n_rays + 2)


class NavPolicy(nn.Module):
    def __init__(self, cfg: SimConfig, hidden: int = 256, depth: int = 2, use_pos: bool = True):
        super().__init__()
        self.n_rays = cfg.n_rays
        self.act_scale, scale = _scales(cfg)
        # lidar in [0, max_range]; rel_goal up to ~scene diameter; pos in scene extent
        pos_scale = 0.1 if use_pos else 0.0  # 0 = ablate absolute position
        self._scale = mx.array(scale + [pos_scale] * 2)
        dims = [cfg.obs_dim] + [hidden] * depth + [2]
        self.layers = [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]

    def __call__(self, obs: mx.array) -> mx.array:
        x = obs * self._scale
        for layer in self.layers[:-1]:
            x = nn.silu(layer(x))
        return mx.array(self.act_scale) * mx.tanh(self.layers[-1](x))


class RecurrentNavPolicy(nn.Module):
    """Encoder MLP -> memory core (CORES) -> action head (+ cost-to-go value head).

    Input is obs concatenated with the previously executed action [N, obs_dim+2].
    The value head predicts the normalized geodesic cost-to-go, distilling the
    planner's field so the trunk must internalize scene topology.

    The deploy-time carried state is a flat [N, state_size] array owned by the
    core (GRU: the hidden itself; transformer: counter + KV ring); consumers
    allocate it with zeros and reset it by zeroing. The core lives under the
    attribute its class names (`gru`/`tfm`) so GRU checkpoints keep their
    historical `gru.*` key layout.
    """

    VAL_SCALE = 1.0 / 20.0  # geodesic meters -> ~[0, 1]

    def __init__(self, cfg: SimConfig, hidden: int = 256, enc: int | None = None,
                 use_pos: bool = True, head: str = "continuous", core: str = "gru",
                 core_opts: dict | None = None, head_opts: dict | None = None):
        super().__init__()
        self.hidden = hidden
        self.act_scale, scale = _scales(cfg)
        pos_scale = 0.1 if use_pos else 0.0
        self._scale = mx.array(scale + [pos_scale] * 2
                               + [1.0 / s for s in self.act_scale])  # last 2 = prev action
        enc = enc or hidden
        self.enc = nn.Linear(cfg.obs_dim + 2, enc)
        core_mod = CORES[core](enc, hidden, **(core_opts or {}))
        self._core_attr = core_mod.attr
        setattr(self, core_mod.attr, core_mod)
        self.head = HEADS[head](hidden, self.act_scale, **(head_opts or {}))
        self.vhead = nn.Linear(hidden, 1)

    @property
    def core(self):
        return getattr(self, self._core_attr)

    @property
    def state_size(self) -> int:
        """Width of the flat deploy-time state (allocate via new_state).
        Layout: [core state | head state] -- the head part exists for heads
        that carry execution state (an in-flight action chunk); it is empty
        for per-step heads. Zeroing resets both halves (cold core, no chunk)."""
        return self.core.state_size + self.head.state_size

    @property
    def h0_size(self) -> int:
        """Width of the chunk-start state trainers store as sequence init
        (core only: training never executes head chunks, it teacher-forces)."""
        return self.core.h0_size

    def new_state(self, n: int) -> mx.array:
        """Fresh (cold) deploy state for n envs. Size and dtype are the core's
        business -- consumers must not assume float32 (the transformer ring is
        fp16)."""
        return mx.zeros((n, self.state_size), dtype=self.core.state_dtype)

    def mask_state(self, state: mx.array, live: mx.array) -> mx.array:
        """Reset finished envs' state to cold (live [N, 1] in {0, 1}). The cast
        keeps the state's dtype: a bare `state * live` would silently promote
        an fp16 state to fp32."""
        return state * live.astype(state.dtype)

    def _encode(self, obs_prev: mx.array) -> mx.array:
        return nn.silu(self.enc(obs_prev * self._scale))

    def _split_state(self, state: mx.array) -> tuple[mx.array, mx.array]:
        return state[:, :self.core.state_size], state[:, self.core.state_size:]

    def _step_feature(self, obs_prev: mx.array, state: mx.array) -> tuple[mx.array, mx.array]:
        """Core-only step (analysis/probing); accepts the full state, returns
        the core part. Action-producing callers use step(), which also runs
        the head's state."""
        core_state, _ = self._split_state(state)
        return self.core.step(self._encode(obs_prev), core_state)

    def features(self, obs_seq: mx.array, h0: mx.array,
                 done_seq: mx.array) -> tuple[mx.array, mx.array]:
        """Core unroll over [B, T, obs_dim+2]; state resets after done steps.
        Returns (features [B, T, H], values [B, T])."""
        feats = self.core.unroll(self._encode(obs_seq), h0, done_seq)
        return feats, self.vhead(feats)[..., 0]

    def step(self, obs_prev: mx.array, state: mx.array) -> tuple[mx.array, mx.array]:
        """One deploy timestep for [N, obs_dim+2] with state [N, state_size]."""
        core_state, head_state = self._split_state(state)
        feat, core_state = self.core.step(self._encode(obs_prev), core_state)
        act, head_state = self.head.act(feat, head_state)
        return act, mx.concatenate([core_state, head_state.astype(core_state.dtype)], axis=1)

    def bc_loss(self, obs_seq: mx.array, h0: mx.array, done_seq: mx.array,
                target: mx.array) -> tuple[mx.array, mx.array]:
        """Per-step imitation loss [B, T] (head-defined) and values [B, T].
        done_seq is passed to the head so trajectory heads can zero their
        cross-episode conditioning; per-step heads ignore it."""
        feats, vals = self.features(obs_seq, h0, done_seq)
        return self.head.bc_loss(feats, target, done_seq), vals
