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

    def __init__(self, hidden: int, act_scale: tuple[float, float]):
        super().__init__(hidden, 2)
        self.act_scale = tuple(act_scale)  # plain tuple: constant, not a parameter

    def mean(self, h: mx.array) -> mx.array:
        return mx.array(self.act_scale) * mx.tanh(super().__call__(h))

    def act(self, h: mx.array) -> mx.array:
        return self.mean(h)

    def bc_loss(self, h: mx.array, target: mx.array) -> mx.array:
        """Per-step imitation loss [...]; target [..., 2]."""
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

    def act(self, h: mx.array) -> mx.array:
        w = self._bin_values()[mx.argmax(self.wlin(h), axis=-1)]
        return mx.stack([self._v_mean(h), w], axis=-1)

    def bc_loss(self, h: mx.array, target: mx.array) -> mx.array:
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


HEADS = {"continuous": ContinuousHead, "discrete_w": DiscreteOmegaHead}


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
    """Encoder MLP -> GRU -> action head (+ cost-to-go value head).

    Input is obs concatenated with the previously executed action [N, obs_dim+2].
    The value head predicts the normalized geodesic cost-to-go, distilling the
    planner's field so the trunk must internalize scene topology.
    """

    VAL_SCALE = 1.0 / 20.0  # geodesic meters -> ~[0, 1]

    def __init__(self, cfg: SimConfig, hidden: int = 256, enc: int = 256, use_pos: bool = True,
                 head: str = "continuous"):
        super().__init__()
        self.hidden = hidden
        self.act_scale, scale = _scales(cfg)
        pos_scale = 0.1 if use_pos else 0.0
        self._scale = mx.array(scale + [pos_scale] * 2
                               + [1.0 / s for s in self.act_scale])  # last 2 = prev action
        self.enc = nn.Linear(cfg.obs_dim + 2, enc)
        self.gru = nn.GRU(enc, hidden)
        self.head = HEADS[head](hidden, self.act_scale)
        self.vhead = nn.Linear(hidden, 1)

    def _feature(self, obs_prev: mx.array, h: mx.array) -> mx.array:
        x = nn.silu(self.enc(obs_prev * self._scale))
        return self.gru(x[:, None, :], hidden=h)[:, -1, :]

    def features(self, obs_seq: mx.array, h0: mx.array,
                 done_seq: mx.array) -> tuple[mx.array, mx.array]:
        """BPTT unroll over [B, T, obs_dim+2]; hidden resets after done steps.
        Returns (features [B, T, H], values [B, T])."""
        t_len = obs_seq.shape[1]
        h = h0
        hs = []
        for t in range(t_len):
            if t > 0:
                h = h * (1.0 - done_seq[:, t - 1])  # done_seq [B, T, 1] -> [B, 1]
            h = self._feature(obs_seq[:, t], h)
            hs.append(h)
        feats = mx.stack(hs, axis=1)
        return feats, self.vhead(feats)[..., 0]

    def step(self, obs_prev: mx.array, h: mx.array) -> tuple[mx.array, mx.array]:
        """One deterministic timestep for [N, obs_dim+2] with hidden [N, H]."""
        h = self._feature(obs_prev, h)
        return self.head.act(h), h

    def bc_loss(self, obs_seq: mx.array, h0: mx.array, done_seq: mx.array,
                target: mx.array) -> tuple[mx.array, mx.array]:
        """Per-step imitation loss [B, T] (head-defined) and values [B, T]."""
        feats, vals = self.features(obs_seq, h0, done_seq)
        return self.head.bc_loss(feats, target), vals
