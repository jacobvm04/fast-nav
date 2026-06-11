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
        """One timestep: returns (deterministic action [N,2], value [N], new hidden)."""
        h = self._feature(obs_prev, h)
        return self.head.act(h), self.vhead(h)[:, 0], h

    def step_sample(self, obs_prev: mx.array,
                    h: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """One rollout timestep: (sampled action [N,2], logp [N], value [N], new hidden)."""
        h = self._feature(obs_prev, h)
        a, logp = self.head.sample_logp(h, self.log_std)
        return a, logp, self.vhead(h)[:, 0], h


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
    init_std: float = 0.3
    max_grad_norm: float = 1.0
    geo_clip: float = 50.0  # meters; padded/obstacle geo regions are huge outliers
    clear_margin: float = 0.10  # proximity = (margin - clearance)/margin below this (m)
    clear_coef: float = 0.012   # quadratic barrier: coef * proximity^2 per step
    speed_prox_coef: float = 0.012  # in-loop governor: coef * proximity * (speed/vmax)
    collision_penalty: float = 0.25  # terminal penalty when contact ends the episode
    bc_coef: float = 0.0  # DAgger anchor: the head's BC loss against expert labels
                          # added to the PPO loss. Stabilizes fine-tuning of weak
                          # inits whose rollouts PPO alone degrades; labels come from
                          # the expert kernel the rollout already evaluates anyway.
    head: str = "continuous"  # action head (fastnav.policy.HEADS)
    hidden: int = 256
    use_pos: bool = False


class PPOTrainer:
    def __init__(self, sim: Sim, cfg: PPOConfig | None = None, seed: int = 0,
                 init_weights: str | None = None):
        self.sim = sim
        self.cfg = cfg = cfg or PPOConfig()
        mx.random.seed(seed)
        self.policy = PPONavPolicy(sim.cfg, hidden=cfg.hidden, use_pos=cfg.use_pos,
                                   init_std=cfg.init_std, head=cfg.head)
        if init_weights:
            self.policy.load_weights(init_weights, strict=False)  # BC ckpt has no log_std
        self.opt = optim.Adam(learning_rate=cfg.lr)
        mx.eval(self.policy.parameters())

        n = sim.num_envs
        self.h = mx.zeros((n, cfg.hidden), dtype=mx.float32)
        self.prev_act = mx.zeros((n, 2), dtype=mx.float32)
        self.iter = 0

        def loss_fn(model, obs, h0, done, act, logp_old, adv, ret, exp):
            feats, vals = model.features(obs, h0, done)
            logp = model.head.log_prob(feats, act, model.log_std)
            ratio = mx.exp(logp - logp_old)
            clipped = mx.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
            pol_loss = -mx.mean(mx.minimum(ratio * adv, clipped * adv))
            verr = vals - ret
            v_loss = mx.mean(mx.where(mx.abs(verr) < 1.0, 0.5 * mx.square(verr),
                                      mx.abs(verr) - 0.5))
            entropy = mx.mean(model.head.entropy(feats, model.log_std))
            bc_loss = mx.mean(model.head.bc_loss(feats, exp))
            return (pol_loss + cfg.value_coef * v_loss - cfg.entropy_coef * entropy
                    + cfg.bc_coef * bc_loss)

        loss_and_grad = nn.value_and_grad(self.policy, loss_fn)
        state = [self.policy.state, self.opt.state]

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
        self.h = mx.zeros((n, self.cfg.hidden), dtype=mx.float32)
        self.prev_act = mx.zeros((n, 2), dtype=mx.float32)
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
        h0 = self.h
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
            pen_l.append(cfg.clear_coef * prox * prox + cfg.speed_prox_coef * prox * spd)
            done = mx.maximum(term, trunc).astype(mx.float32)
            done_l.append(done)
            reach_l.append(term.astype(mx.float32))
            hit_l.append(sim.hit.astype(mx.float32))
            live = (1.0 - done)[:, None]
            self.h = h_new * live
            self.prev_act = self._clamp(a) * live

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
            fail_r = mx.where(hit_l[t] > 0.5, -cfg.collision_penalty, 0.0)
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
        mx.eval(batch["obs"], batch["adv"])
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
