"""Recurrent PPO fine-tuning on the geodesic-progress reward, fully GPU-resident.

Reward (training-time oracle, never observed by the policy):
    r_t = (geo(s_t) - geo(s_{t+1})) * VAL_SCALE          while the episode runs
    r_t = geo(s_t) * VAL_SCALE + success_bonus           on reaching the goal
    r_t = 0                                              on timeout
With this scaling V(s) ~= geo(s) * VAL_SCALE, so the BC value-distillation head
is already a near-correct critic at initialization.

Recurrent details: rollout chunks of T steps; the hidden at chunk start is
stored and reused as BPTT init during updates (same hidden the rollout actually
had, so old/new log-probs are consistent at epoch 0). Hidden and prev-action
reset at episode boundaries both in rollout and inside the BPTT unroll.
Timeouts are treated as terminal: under the shaped reward a stuck policy's
true remaining return is ~0, which matches the V target this induces.
"""

from __future__ import annotations

import dataclasses
import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from fastnav.policy import RecurrentNavPolicy
from fastnav.sim import Sim


class PPONavPolicy(RecurrentNavPolicy):
    """RecurrentNavPolicy + state-independent log-std over the head's
    continuous dims; the head supplies the sampling distribution."""

    def __init__(self, *args, init_std: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_std = mx.full((self.head.n_continuous,), math.log(init_std))

    def step_full(self, obs_prev: mx.array, h: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """One timestep: returns (deterministic action [N,2], value [N], new state)."""
        core_state, head_state = self._split_state(h)
        feat, core_state = self.core.step(self._encode(obs_prev), core_state)
        act, head_state = self.head.act(feat, head_state)
        return (act, self.vhead(feat)[:, 0],
                mx.concatenate([core_state, head_state.astype(core_state.dtype)], axis=1))

    def step_sample(self, obs_prev: mx.array,
                    h: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """One rollout timestep: (sampled action [N,2], logp [N], value [N], new state).
        Per-step sampling: the head state passes through untouched (chunk heads
        are BC-only and raise in sample_logp)."""
        core_state, head_state = self._split_state(h)
        feat, core_state = self.core.step(self._encode(obs_prev), core_state)
        a, logp = self.head.sample_logp(feat, self.log_std)
        return a, logp, self.vhead(feat)[:, 0], mx.concatenate([core_state, head_state], axis=1)


@dataclasses.dataclass
class PPOConfig:
    chunk: int = 16
    lr: float = 1e-4
    clip_eps: float = 0.2
    gamma: float = 0.995
    lam: float = 0.95
    epochs: int = 2
    minibatch_seqs: int = 2048
    value_coef: float = 0.5
    entropy_coef: float = 5e-4
    success_bonus: float = 0.5
    timeout_penalty: float = 0.0  # terminal penalty when the step budget expires
                                  # unreached: prices orbiting-near-goal as a loss
                                  # instead of a free outcome
    init_std: float = 0.3
    max_grad_norm: float = 1.0
    geo_clip: float = 50.0  # meters; padded/obstacle geo regions are huge outliers
    clear_margin: float = 0.10  # proximity = (margin - clearance)/margin below this (m)
    clear_coef: float = 0.012   # quadratic barrier: coef * proximity^2 per step
    speed_prox_coef: float = 0.012  # in-loop governor: coef * proximity * (speed/vmax)
    smooth_coef: float = 0.0    # control-smoothness penalty: coef * mean_dim((a - a_prev)
                                # / action_scale)^2 per step. Penalizes jerk in the commanded
                                # (v, omega) so the low-level controller tracks a smoother
                                # reference; normalized per-dim so v and omega weigh equally,
                                # and zeroed across episode boundaries (a reset is not a jerk).
    collision_penalty: float = 0.25  # terminal penalty when contact ends the episode
    bc_coef: float = 0.0  # DAgger anchor: the head's BC loss against expert labels
                          # added to the PPO loss. Stabilizes fine-tuning of weak
                          # inits whose rollouts PPO alone degrades; labels come from
                          # the expert kernel the rollout already evaluates anyway.
    head: str = "continuous"  # action head (fastnav.policy.HEADS)
    core: str = "gru"         # memory core (fastnav.policy.CORES)
    core_opts: dict = dataclasses.field(default_factory=dict)
    burn_in: int = 0          # chunk-leading steps excluded from the loss; they only
                              # warm the state. Required for cores with no stored h0
                              # (transformer): without it, cold-context logp_new is
                              # compared against full-context rollout logp_old and the
                              # early-token importance ratios are biased.
    hidden: int = 256
    use_pos: bool = False


class PPOTrainer:
    def __init__(self, sim: Sim, cfg: PPOConfig | None = None, seed: int = 0,
                 init_weights: str | None = None):
        self.sim = sim
        self.cfg = cfg = cfg or PPOConfig()
        mx.random.seed(seed)
        self.policy = PPONavPolicy(sim.cfg, hidden=cfg.hidden, use_pos=cfg.use_pos,
                                   init_std=cfg.init_std, head=cfg.head, core=cfg.core,
                                   core_opts=cfg.core_opts)
        if init_weights:
            self.policy.load_weights(init_weights, strict=False)  # BC ckpt has no log_std
        self.opt = optim.Adam(learning_rate=cfg.lr)
        mx.eval(self.policy.parameters())

        n = sim.num_envs
        self.h = self.policy.new_state(n)
        self.prev_act = mx.zeros((n, 2), dtype=mx.float32)
        # 1 where prev_act is a real predecessor (episode did not just reset);
        # carried across chunks like prev_act so the smoothness term never
        # charges jerk against a post-reset zero (see rollout). Cold start = 0:
        # the very first action of the run has no predecessor.
        self.prev_live = mx.zeros((n,), dtype=mx.float32)
        self.iter = 0

        # loss masked over burn-in: those steps only warm the state (see burn_in)
        lmask = mx.array([0.0 if t < cfg.burn_in else 1.0 for t in range(cfg.chunk)])
        lmask = lmask / mx.maximum(mx.sum(lmask), 1.0)

        def wmean(x):  # per-seq weighted step mean, then batch mean
            return mx.mean(mx.sum(x * lmask, axis=-1))

        def loss_fn(model, obs, h0, done, act, logp_old, adv, ret, exp):
            feats, vals = model.features(obs, h0, done)
            logp = model.head.log_prob(feats, act, model.log_std)
            ratio = mx.exp(logp - logp_old)
            clipped = mx.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
            pol_loss = -wmean(mx.minimum(ratio * adv, clipped * adv))
            verr = vals - ret
            v_loss = wmean(mx.where(mx.abs(verr) < 1.0, 0.5 * mx.square(verr),
                                    mx.abs(verr) - 0.5))
            entropy = wmean(model.head.entropy(feats, model.log_std)
                            * mx.ones(ratio.shape))  # continuous-head entropy is scalar
            bc_loss = wmean(model.head.bc_loss(feats, exp))
            return (pol_loss + cfg.value_coef * v_loss - cfg.entropy_coef * entropy
                    + cfg.bc_coef * bc_loss)

        loss_and_grad = nn.value_and_grad(self.policy, loss_fn)
        # mx.random.state: see dagger.py -- heads may sample inside bc_loss
        state = [self.policy.state, self.opt.state, mx.random.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def update(obs, h0, done, act, logp_old, adv, ret, exp):
            loss, grads = loss_and_grad(self.policy, obs, h0, done, act, logp_old, adv, ret, exp)
            grads, _ = optim.clip_grad_norm(grads, max_norm=cfg.max_grad_norm)
            self.opt.update(self.policy, grads)
            return loss

        self._update = update

    def swap_sim(self, sim: Sim) -> None:
        """Replace the rollout sim (pack rotation); recurrent state restarts."""
        self.sim = sim
        sim.reset()
        n = sim.num_envs
        self.h = self.policy.new_state(n)
        self.prev_act = mx.zeros((n, 2), dtype=mx.float32)
        self.prev_live = mx.zeros((n,), dtype=mx.float32)  # no predecessor after a reset
        mx.clear_cache()  # return the old pack's buffers to the OS (else swap ratchets)

    def _clamp(self, a: mx.array) -> mx.array:
        """Same action clamp the sim applies, so prev-action input matches reality."""
        return self.sim.kin.clamp(a, self.sim.cfg)

    def _geo(self) -> tuple[mx.array, mx.array]:
        """Oracle (cost-to-go, expert action) at the current states."""
        exp = self.sim.expert_actions()
        return mx.clip(self.sim.expert_geo_val, 0.0, self.cfg.geo_clip), exp

    def rollout(self) -> dict:
        sim, cfg = self.sim, self.cfg
        vs = type(self.policy).VAL_SCALE
        h0 = self.h[:, :self.policy.h0_size]  # chunk-start state the BPTT init needs
        obs_l, act_l, logp_l, val_l, geo_l, exp_l, done_l, reach_l, pen_l, hit_l = ([], [], [],
            [], [], [], [], [], [], [])
        obs = sim.obs()
        for _ in range(cfg.chunk):
            geo, exp = self._geo()
            geo_l.append(geo)
            exp_l.append(exp)
            obs_in = mx.concatenate([obs, self.prev_act], axis=1)
            a, logp, v, h_new = self.policy.step_sample(obs_in, self.h)
            logp_l.append(logp)
            obs_l.append(obs_in)
            act_l.append(a)
            val_l.append(v)
            obs, term, trunc = sim.step(a)
            spd = sim.kin.speed(a, sim.cfg)  # normalized linear speed
            prox = mx.maximum(cfg.clear_margin - sim.clearance, 0.0) / cfg.clear_margin
            # convex barrier punishes corner-skimming hard but wall-adjacent travel
            # mildly; the speed term is the governor learned in-loop
            pen = cfg.clear_coef * prox * prox + cfg.speed_prox_coef * prox * spd
            if cfg.smooth_coef:
                # jerk in normalized command units vs the clamped previous action,
                # charged only where that predecessor is real (prev_live); a reset
                # leaves prev_act at 0, which is not a jerk
                dj = (self._clamp(a) - self.prev_act) / mx.array(self.policy.act_scale)
                pen = pen + cfg.smooth_coef * mx.mean(dj * dj, axis=1) * self.prev_live
            pen_l.append(pen)
            done = mx.maximum(term, trunc).astype(mx.float32)
            done_l.append(done)
            reach_l.append(term.astype(mx.float32))
            hit_l.append(sim.hit.astype(mx.float32))
            live = (1.0 - done)[:, None]
            self.h = self.policy.mask_state(h_new, live)
            self.prev_act = self._clamp(a) * live
            self.prev_live = live[:, 0]

        # bootstrap value and next-geo at the chunk's final state
        geo_T, _ = self._geo()
        obs_in = mx.concatenate([obs, self.prev_act], axis=1)
        _, v_T, _ = self.policy.step_full(obs_in, self.h)

        # rewards and GAE (reversed scan over the chunk)
        adv_l = [None] * cfg.chunk
        ret_l = [None] * cfg.chunk
        r_l = [None] * cfg.chunk
        gae = mx.zeros_like(v_T)
        next_v = v_T
        next_geo = geo_T
        for t in reversed(range(cfg.chunk)):
            done, reach = done_l[t], reach_l[t]
            run_r = (geo_l[t] - next_geo) * vs
            term_r = geo_l[t] * vs + cfg.success_bonus
            fail_r = mx.where(hit_l[t] > 0.5, -cfg.collision_penalty, -cfg.timeout_penalty)
            r = mx.where(done > 0.5, mx.where(reach > 0.5, term_r, fail_r), run_r) - pen_l[t]
            r_l[t] = r
            live = 1.0 - done
            delta = r + cfg.gamma * next_v * live - val_l[t]
            gae = delta + cfg.gamma * cfg.lam * live * gae
            adv_l[t] = gae
            ret_l[t] = gae + val_l[t]
            next_v = val_l[t]
            next_geo = geo_l[t]

        batch = {
            "obs": mx.stack(obs_l, axis=1),
            "act": mx.stack(act_l, axis=1),
            "logp": mx.stack(logp_l, axis=1),
            "done": mx.stack(done_l, axis=1)[..., None],
            "adv": mx.stack(adv_l, axis=1),
            "ret": mx.stack(ret_l, axis=1),
            "exp": mx.stack(exp_l, axis=1),
            "h0": h0,
        }
        adv = batch["adv"]
        batch["adv"] = (adv - mx.mean(adv)) / (mx.std(adv) + 1e-6)
        stats = {
            "reward_mean": float(mx.mean(mx.stack(r_l))),
            "rollout_success": float(mx.sum(mx.stack(reach_l)) /
                                     mx.maximum(mx.sum(mx.stack(done_l)), 1.0)),
            "value_mean": float(mx.mean(mx.stack(val_l))),
        }
        # h0 included: a zero-width h0 (transformer) is a slice VIEW of the full
        # state; left lazy it pins the chunk-start KV ring alive (see dagger.py)
        mx.eval(batch["obs"], batch["adv"], batch["h0"])
        return batch | stats

    def train(self, batch: dict) -> dict:
        cfg = self.cfg
        n = batch["obs"].shape[0]
        mb = min(cfg.minibatch_seqs, n)
        loss = mx.array(0.0)
        for _ in range(cfg.epochs):
            perm = mx.random.permutation(n)
            for s in range(0, n - mb + 1, mb):
                idx = perm[s:s + mb]
                loss = self._update(batch["obs"][idx], batch["h0"][idx], batch["done"][idx],
                                    batch["act"][idx], batch["logp"][idx],
                                    batch["adv"][idx], batch["ret"][idx], batch["exp"][idx])
        mx.eval(loss, self.policy.state, self.opt.state)
        return {"loss": float(loss), "std": float(mx.mean(mx.exp(self.policy.log_std)))}

    def step(self) -> dict:
        batch = self.rollout()
        stats = self.train(batch)
        self.iter += 1
        return {"reward_mean": batch["reward_mean"],
                "rollout_success": batch["rollout_success"],
                "value_mean": batch["value_mean"], **stats}
