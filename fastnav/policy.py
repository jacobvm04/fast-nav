"""Tiny MLP policy with built-in observation normalization (checkpoint-portable)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from fastnav.sim import SimConfig


class NavPolicy(nn.Module):
    def __init__(self, cfg: SimConfig, hidden: int = 256, depth: int = 2, use_pos: bool = True):
        super().__init__()
        self.n_rays = cfg.n_rays
        self.v_max = cfg.v_max
        # lidar in [0, max_range]; rel_goal up to ~scene diameter; pos in scene extent
        pos_scale = 0.1 if use_pos else 0.0  # 0 = ablate absolute position
        scale = [1.0 / cfg.max_range] * cfg.n_rays + [1.0 / cfg.max_range] * 2 + [pos_scale] * 2
        self._scale = mx.array(scale)
        dims = [cfg.obs_dim] + [hidden] * depth + [2]
        self.layers = [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]

    def __call__(self, obs: mx.array) -> mx.array:
        x = obs * self._scale
        for layer in self.layers[:-1]:
            x = nn.silu(layer(x))
        return self.v_max * mx.tanh(self.layers[-1](x))


class RecurrentNavPolicy(nn.Module):
    """Encoder MLP -> GRU -> tanh velocity head (+ cost-to-go value head).

    Input is obs concatenated with the previously executed action [N, obs_dim+2].
    The value head predicts the normalized geodesic cost-to-go, distilling the
    planner's field so the trunk must internalize scene topology.
    """

    VAL_SCALE = 1.0 / 20.0  # geodesic meters -> ~[0, 1]

    def __init__(self, cfg: SimConfig, hidden: int = 256, enc: int = 256, use_pos: bool = True):
        super().__init__()
        self.v_max = cfg.v_max
        self.hidden = hidden
        pos_scale = 0.1 if use_pos else 0.0
        scale = ([1.0 / cfg.max_range] * cfg.n_rays + [1.0 / cfg.max_range] * 2
                 + [pos_scale] * 2 + [1.0 / cfg.v_max] * 2)  # last 2 = prev action
        self._scale = mx.array(scale)
        self.enc = nn.Linear(cfg.obs_dim + 2, enc)
        self.gru = nn.GRU(enc, hidden)
        self.head = nn.Linear(hidden, 2)
        self.vhead = nn.Linear(hidden, 1)

    def step(self, obs_prev: mx.array, h: mx.array) -> tuple[mx.array, mx.array]:
        """One timestep for [N, obs_dim+2] (obs | prev action) with hidden [N, H]."""
        x = nn.silu(self.enc(obs_prev * self._scale))
        h = self.gru(x[:, None, :], hidden=h)[:, -1, :]
        return self.v_max * mx.tanh(self.head(h)), h

    def __call__(self, obs_seq: mx.array, h0: mx.array,
                 done_seq: mx.array) -> tuple[mx.array, mx.array]:
        """BPTT over [B, T, obs_dim+2]; hidden resets after done steps.

        Returns (actions [B, T, 2], values [B, T])."""
        t_len = obs_seq.shape[1]
        h = h0
        acts, hs = [], []
        for t in range(t_len):
            if t > 0:
                h = h * (1.0 - done_seq[:, t - 1])  # done_seq [B, T, 1] -> [B, 1]
            a, h = self.step(obs_seq[:, t], h)
            acts.append(a)
            hs.append(h)
        vals = self.vhead(mx.stack(hs, axis=1))[..., 0]
        return mx.stack(acts, axis=1), vals
