"""Hyper-online DAgger: PPO-shaped loop, everything GPU-resident in MLX.

Each iteration:
  1. rollout `chunk` steps with a beta-mixture of expert/policy actions,
     labeling every visited state with the expert kernel
  2. append (obs, expert_action) to a GPU ring buffer
  3. a few compiled minibatch Adam updates on buffer samples

No transition ever touches host memory during training.
"""

from __future__ import annotations

import dataclasses
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from fastnav.policy import NavPolicy, RecurrentNavPolicy
from fastnav.sim import Sim


@dataclasses.dataclass
class DaggerConfig:
    chunk: int = 16              # rollout steps per iteration
    buffer_size: int = 2_000_000
    batch_size: int = 32_768
    updates_per_iter: int = 4
    lr: float = 3e-4
    beta_decay_iters: int = 80   # expert-action mixture: 1 -> 0 linearly
    hidden: int = 256
    depth: int = 2
    augment: bool = False        # dihedral (rotation+reflection) obs/action augmentation
    lidar_noise: float = 0.0     # gaussian range noise sigma (m), train-time
    ray_dropout: float = 0.0     # per-ray prob of a missed return (-> max_range)
    use_pos: bool = True         # feed absolute position to the policy
    burn_in: int = 0             # BPTT steps that warm the hidden without loss (recurrent)
    value_weight: float = 0.5    # cost-to-go distillation loss weight (recurrent)


class DaggerTrainer:
    def __init__(self, sim: Sim, cfg: DaggerConfig | None = None, seed: int = 0):
        self.sim = sim
        self.cfg = cfg = cfg or DaggerConfig()
        mx.random.seed(seed)
        self.policy = NavPolicy(sim.cfg, hidden=cfg.hidden, depth=cfg.depth, use_pos=cfg.use_pos)
        self.opt = optim.Adam(learning_rate=cfg.lr)
        mx.eval(self.policy.parameters())

        d = sim.cfg.obs_dim
        self.buf_obs = mx.zeros((cfg.buffer_size, d), dtype=mx.float32)
        self.buf_act = mx.zeros((cfg.buffer_size, 2), dtype=mx.float32)
        self.buf_ptr = 0
        self.buf_full = False
        self.iter = 0

        def loss_fn(model, obs, act):
            return nn.losses.mse_loss(model(obs), act)

        loss_and_grad = nn.value_and_grad(self.policy, loss_fn)
        state = [self.policy.state, self.opt.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def update(obs, act):
            loss, grads = loss_and_grad(self.policy, obs, act)
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            self.opt.update(self.policy, grads)
            return loss

        self._update = update

    @property
    def beta(self) -> float:
        return max(0.0, 1.0 - self.iter / self.cfg.beta_decay_iters)

    def _append(self, obs: mx.array, act: mx.array) -> None:
        n = obs.shape[0]
        cap = self.cfg.buffer_size
        p = self.buf_ptr
        if p + n <= cap:
            self.buf_obs[p:p + n] = obs
            self.buf_act[p:p + n] = act
        else:
            k = cap - p
            self.buf_obs[p:] = obs[:k]
            self.buf_act[p:] = act[:k]
            self.buf_obs[: n - k] = obs[k:]
            self.buf_act[: n - k] = act[k:]
            self.buf_full = True
        self.buf_ptr = (p + n) % cap
        if self.buf_ptr < p and not self.buf_full:
            self.buf_full = True

    @property
    def buf_count(self) -> int:
        return self.cfg.buffer_size if self.buf_full else self.buf_ptr

    def rollout(self) -> None:
        sim, beta = self.sim, self.beta
        obs = sim.obs()
        for _ in range(self.cfg.chunk):
            expert = sim.expert_actions()
            self._append(obs, expert)
            if beta >= 1.0:
                act = expert
            else:
                act = self.policy(obs)
                if beta > 0.0:
                    pick = mx.random.uniform(shape=(act.shape[0], 1)) < beta
                    act = mx.where(pick, expert, act)
            obs, _, _ = sim.step(act)
        mx.eval(obs, self.buf_obs, self.buf_act)

    def _augment(self, obs: mx.array, act: mx.array) -> tuple[mx.array, mx.array]:
        """Dihedral-group augmentation: exact symmetry of the orientation-free
        holonomic robot with a uniform 360-degree lidar ring. Plus sensor DR."""
        import math

        cfg = self.cfg
        r = self.sim.cfg.n_rays
        b = obs.shape[0]
        lidar = obs[:, :r]
        rel_goal = obs[:, r:r + 2]
        pos = obs[:, r + 2:r + 4]

        # reflection across the x-axis (y -> -y), per-sample coin flip
        refl = mx.random.uniform(shape=(b, 1)) < 0.5
        ridx = (-mx.arange(r)) % r
        lidar = mx.where(refl, lidar[:, ridx], lidar)
        flip = mx.where(refl, -1.0, 1.0)[:, 0]

        # rotation by a random multiple of the ray spacing:
        # world rotated by phi -> lidar'[i] = lidar[(i - k) % r], vectors rotated by phi
        k = mx.random.randint(0, r, shape=(b,))
        idx = (mx.arange(r)[None, :] - k[:, None]) % r
        lidar = mx.take_along_axis(lidar, idx, axis=1)
        phi = k.astype(mx.float32) * (2.0 * math.pi / r)
        c, s = mx.cos(phi), mx.sin(phi)

        def xform(v):
            x, y = v[:, 0], v[:, 1] * flip
            return mx.stack([c * x - s * y, s * x + c * y], axis=1)

        rel_goal, pos, act = xform(rel_goal), xform(pos), xform(act)

        if cfg.lidar_noise > 0:
            lidar = lidar + cfg.lidar_noise * mx.random.normal(lidar.shape)
        if cfg.ray_dropout > 0:
            miss = mx.random.uniform(shape=lidar.shape) < cfg.ray_dropout
            lidar = mx.where(miss, self.sim.cfg.max_range, lidar)
        lidar = mx.clip(lidar, 0.0, self.sim.cfg.max_range)
        return mx.concatenate([lidar, rel_goal, pos], axis=1), act

    def train(self) -> float:
        n = self.buf_count
        loss = mx.array(0.0)
        for _ in range(self.cfg.updates_per_iter):
            idx = mx.random.randint(0, n, shape=(self.cfg.batch_size,))
            obs, act = self.buf_obs[idx], self.buf_act[idx]
            if self.cfg.augment:
                obs, act = self._augment(obs, act)
            loss = self._update(obs, act)
        mx.eval(loss, self.policy.state, self.opt.state)
        return float(loss)

    def step(self) -> float:
        self.rollout()
        loss = self.train()
        self.iter += 1
        return loss


class RecurrentDaggerTrainer:
    """DAgger with a GRU policy: 16-step rollout chunks become BPTT sequences.

    The hidden state at chunk start is stored with each sequence (R2D2-style)
    and used as the BPTT init; hidden resets at episode boundaries both during
    rollout and inside the BPTT unroll. With augmentation enabled, sequences are
    rotated/reflected consistently across time and h0 is zeroed (a rotated
    history has no matching hidden state).
    """

    def __init__(self, sim: Sim, cfg: DaggerConfig | None = None, seed: int = 0):
        self.sim = sim
        self.cfg = cfg = cfg or DaggerConfig()
        mx.random.seed(seed)
        self.policy = RecurrentNavPolicy(sim.cfg, hidden=cfg.hidden, use_pos=cfg.use_pos)
        self.opt = optim.Adam(learning_rate=cfg.lr)
        mx.eval(self.policy.parameters())

        d = sim.cfg.obs_dim + 2  # obs | prev action
        self.cap = cfg.buffer_size // cfg.chunk
        self.buf_obs = mx.zeros((self.cap, cfg.chunk, d), dtype=mx.float32)
        self.buf_act = mx.zeros((self.cap, cfg.chunk, 2), dtype=mx.float32)
        self.buf_val = mx.zeros((self.cap, cfg.chunk), dtype=mx.float32)
        self.buf_done = mx.zeros((self.cap, cfg.chunk, 1), dtype=mx.float32)
        self.buf_h0 = mx.zeros((self.cap, cfg.hidden), dtype=mx.float32)
        self.h = mx.zeros((sim.num_envs, cfg.hidden), dtype=mx.float32)
        self.prev_act = mx.zeros((sim.num_envs, 2), dtype=mx.float32)
        self.buf_ptr = 0
        self.buf_full = False
        self.iter = 0

        # loss masked over burn-in: those steps only warm the hidden state
        mask = mx.array([0.0 if t < cfg.burn_in else 1.0 for t in range(cfg.chunk)])
        mask = mask / mx.maximum(mx.sum(mask), 1.0)
        vs = type(self.policy).VAL_SCALE

        def loss_fn(model, obs, h0, done, act, val):
            pred_a, pred_v = model(obs, h0, done)
            loss_a = mx.sum(mx.mean(mx.square(pred_a - act), axis=(0, 2)) * mask)
            # clip + huber: padded/obstacle-filled geo regions produce rare huge
            # cost-to-go targets that otherwise poison whole batches
            val_t = mx.clip(val * vs, 0.0, 2.5)
            err = pred_v - val_t
            hub = mx.where(mx.abs(err) < 1.0, 0.5 * mx.square(err), mx.abs(err) - 0.5)
            loss_v = mx.sum(mx.mean(hub, axis=0) * mask)
            return loss_a + cfg.value_weight * loss_v

        loss_and_grad = nn.value_and_grad(self.policy, loss_fn)
        state = [self.policy.state, self.opt.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def update(obs, h0, done, act, val):
            loss, grads = loss_and_grad(self.policy, obs, h0, done, act, val)
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)  # long BPTT explodes without this
            self.opt.update(self.policy, grads)
            return loss

        self._update = update

    beta = DaggerTrainer.beta

    @property
    def buf_count(self) -> int:
        return self.cap if self.buf_full else self.buf_ptr

    def _append_rows(self, obs, act, val, done, h0) -> None:
        n = obs.shape[0]
        p = self.buf_ptr
        bufs = ((self.buf_obs, obs), (self.buf_act, act), (self.buf_val, val),
                (self.buf_done, done), (self.buf_h0, h0))
        if p + n <= self.cap:
            for buf, x in bufs:
                buf[p:p + n] = x
        else:
            k = self.cap - p
            for buf, x in bufs:
                buf[p:] = x[:k]
                buf[: n - k] = x[k:]
            self.buf_full = True
        self.buf_ptr = (p + n) % self.cap
        if self.buf_ptr < p and not self.buf_full:
            self.buf_full = True

    def rollout(self) -> None:
        sim, beta, cfg = self.sim, self.beta, self.cfg
        obs = sim.obs()
        h0 = self.h
        obs_l, act_l, val_l, done_l = [], [], [], []
        for _ in range(cfg.chunk):
            expert = sim.expert_actions()
            obs_in = mx.concatenate([obs, self.prev_act], axis=1)
            pol_act, self.h = self.policy.step(obs_in, self.h)
            obs_l.append(obs_in)
            act_l.append(expert)
            val_l.append(sim.expert_geo_val)
            if beta >= 1.0:
                act = expert
            elif beta > 0.0:
                pick = mx.random.uniform(shape=(pol_act.shape[0], 1)) < beta
                act = mx.where(pick, expert, pol_act)
            else:
                act = pol_act
            obs, term, trunc = sim.step(act)
            done = mx.maximum(term, trunc).astype(mx.float32)[:, None]
            done_l.append(done)
            self.h = self.h * (1.0 - done)
            self.prev_act = act * (1.0 - done)
        self._append_rows(mx.stack(obs_l, axis=1), mx.stack(act_l, axis=1),
                          mx.stack(val_l, axis=1), mx.stack(done_l, axis=1), h0)
        mx.eval(obs, self.h, self.buf_obs)

    def _augment_seq(self, obs, act):
        """Dihedral augmentation, consistent across each sequence's timesteps."""
        import math

        r = self.sim.cfg.n_rays
        b, t_len, d = obs.shape
        obs = obs.reshape(b * t_len, d)
        act = act.reshape(b * t_len, 2)
        lidar, rel_goal, pos = obs[:, :r], obs[:, r:r + 2], obs[:, r + 2:r + 4]
        prev = obs[:, r + 4:r + 6]

        refl = mx.repeat(mx.random.uniform(shape=(b, 1)) < 0.5, t_len, axis=0)
        ridx = (-mx.arange(r)) % r
        lidar = mx.where(refl, lidar[:, ridx], lidar)
        flip = mx.where(refl, -1.0, 1.0)[:, 0]

        k = mx.repeat(mx.random.randint(0, r, shape=(b,)), t_len, axis=0)
        idx = (mx.arange(r)[None, :] - k[:, None]) % r
        lidar = mx.take_along_axis(lidar, idx, axis=1)
        phi = k.astype(mx.float32) * (2.0 * math.pi / r)
        c, s = mx.cos(phi), mx.sin(phi)

        def xform(v):
            x, y = v[:, 0], v[:, 1] * flip
            return mx.stack([c * x - s * y, s * x + c * y], axis=1)

        rel_goal, pos, act = xform(rel_goal), xform(pos), xform(act)
        prev = xform(prev)
        if self.cfg.lidar_noise > 0:
            lidar = lidar + self.cfg.lidar_noise * mx.random.normal(lidar.shape)
        if self.cfg.ray_dropout > 0:
            miss = mx.random.uniform(shape=lidar.shape) < self.cfg.ray_dropout
            lidar = mx.where(miss, self.sim.cfg.max_range, lidar)
        lidar = mx.clip(lidar, 0.0, self.sim.cfg.max_range)
        obs = mx.concatenate([lidar, rel_goal, pos, prev], axis=1)
        return obs.reshape(b, t_len, d), act.reshape(b, t_len, 2)

    def train(self) -> float:
        n = self.buf_count
        b_seq = max(1, self.cfg.batch_size // self.cfg.chunk)
        loss = mx.array(0.0)
        for _ in range(self.cfg.updates_per_iter):
            idx = mx.random.randint(0, n, shape=(b_seq,))
            obs, act = self.buf_obs[idx], self.buf_act[idx]
            val, done, h0 = self.buf_val[idx], self.buf_done[idx], self.buf_h0[idx]
            if self.cfg.augment:
                obs, act = self._augment_seq(obs, act)
                h0 = mx.zeros_like(h0)
            loss = self._update(obs, h0, done, act, val)
        mx.eval(loss, self.policy.state, self.opt.state)
        return float(loss)

    def step(self) -> float:
        self.rollout()
        loss = self.train()
        self.iter += 1
        return loss


def evaluate(sim: Sim, policy) -> dict:
    """Greedy policy rollout; unbiased success over each env's FIRST episode.

    Runs a full max_steps horizon so slow failures count, not just fast successes.
    """
    sim.reset()
    n = sim.num_envs
    recurrent = isinstance(policy, RecurrentNavPolicy)
    h = mx.zeros((n, policy.hidden), dtype=mx.float32) if recurrent else None
    prev = mx.zeros((n, 2), dtype=mx.float32)
    succeeded = mx.zeros((n,), dtype=mx.bool_)
    finished = mx.zeros((n,), dtype=mx.bool_)
    steps_taken = mx.zeros((n,), dtype=mx.int32)
    min_clear = mx.full((n,), 9.0)
    for t in range(sim.cfg.max_steps + 1):
        obs = sim.obs()
        if recurrent:
            act, h = policy.step(mx.concatenate([obs, prev], axis=1), h)
        else:
            act = policy(obs)
        obs, term, trunc = sim.step(act)
        min_clear = mx.where(finished, min_clear, mx.minimum(min_clear, sim.clearance))
        if recurrent:
            live = 1.0 - mx.maximum(term, trunc).astype(mx.float32)[:, None]
            h = h * live
            prev = act * live
        done = mx.logical_or(term.astype(mx.bool_), trunc.astype(mx.bool_))
        first = mx.logical_and(done, mx.logical_not(finished))
        succeeded = mx.logical_or(succeeded, mx.logical_and(first, term.astype(mx.bool_)))
        steps_taken = mx.where(first, t + 1, steps_taken)
        finished = mx.logical_or(finished, done)
        if t % 64 == 0:
            mx.eval(finished)
            if bool(mx.all(finished)):
                break
    mx.eval(succeeded, finished, steps_taken, min_clear)
    n_fin = int(mx.sum(finished))
    n_suc = int(mx.sum(succeeded))
    safe = mx.logical_and(succeeded, min_clear > 0.03)  # never within 3cm of contact
    return {
        "success": n_suc / max(n_fin, 1),
        "safe_success": int(mx.sum(safe)) / max(n_fin, 1),
        "episodes": n_fin,
        "steps_per_episode": float(mx.sum(steps_taken)) / max(n_fin, 1),
    }
