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

from fastnav.policy import NavPolicy
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


class DaggerTrainer:
    def __init__(self, sim: Sim, cfg: DaggerConfig | None = None, seed: int = 0):
        self.sim = sim
        self.cfg = cfg = cfg or DaggerConfig()
        mx.random.seed(seed)
        self.policy = NavPolicy(sim.cfg, hidden=cfg.hidden, depth=cfg.depth)
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

    def train(self) -> float:
        n = self.buf_count
        loss = mx.array(0.0)
        for _ in range(self.cfg.updates_per_iter):
            idx = mx.random.randint(0, n, shape=(self.cfg.batch_size,))
            loss = self._update(self.buf_obs[idx], self.buf_act[idx])
        mx.eval(loss, self.policy.state, self.opt.state)
        return float(loss)

    def step(self) -> float:
        self.rollout()
        loss = self.train()
        self.iter += 1
        return loss


def evaluate(sim: Sim, policy: NavPolicy) -> dict:
    """Greedy policy rollout; unbiased success over each env's FIRST episode.

    Runs a full max_steps horizon so slow failures count, not just fast successes.
    """
    sim.reset()
    n = sim.num_envs
    succeeded = mx.zeros((n,), dtype=mx.bool_)
    finished = mx.zeros((n,), dtype=mx.bool_)
    steps_taken = mx.zeros((n,), dtype=mx.int32)
    for t in range(sim.cfg.max_steps + 1):
        obs = sim.obs()
        obs, term, trunc = sim.step(policy(obs))
        done = mx.logical_or(term.astype(mx.bool_), trunc.astype(mx.bool_))
        first = mx.logical_and(done, mx.logical_not(finished))
        succeeded = mx.logical_or(succeeded, mx.logical_and(first, term.astype(mx.bool_)))
        steps_taken = mx.where(first, t + 1, steps_taken)
        finished = mx.logical_or(finished, done)
        if t % 64 == 0:
            mx.eval(finished)
            if bool(mx.all(finished)):
                break
    mx.eval(succeeded, finished, steps_taken)
    n_fin = int(mx.sum(finished))
    n_suc = int(mx.sum(succeeded))
    return {
        "success": n_suc / max(n_fin, 1),
        "episodes": n_fin,
        "steps_per_episode": float(mx.sum(steps_taken)) / max(n_fin, 1),
    }
