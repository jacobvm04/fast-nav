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
    """Encoder MLP -> GRU -> tanh velocity head. Memory for detour decisions."""

    def __init__(self, cfg: SimConfig, hidden: int = 256, enc: int = 256, use_pos: bool = True):
        super().__init__()
        self.v_max = cfg.v_max
        self.hidden = hidden
        pos_scale = 0.1 if use_pos else 0.0
        scale = [1.0 / cfg.max_range] * cfg.n_rays + [1.0 / cfg.max_range] * 2 + [pos_scale] * 2
        self._scale = mx.array(scale)
        self.enc = nn.Linear(cfg.obs_dim, enc)
        self.gru = nn.GRU(enc, hidden)
        self.head = nn.Linear(hidden, 2)

    def step(self, obs: mx.array, h: mx.array) -> tuple[mx.array, mx.array]:
        """One timestep for [N, D] obs with hidden [N, H]."""
        x = nn.silu(self.enc(obs * self._scale))
        h = self.gru(x[:, None, :], hidden=h)[:, -1, :]
        return self.v_max * mx.tanh(self.head(h)), h

    def __call__(self, obs_seq: mx.array, h0: mx.array, done_seq: mx.array) -> mx.array:
        """BPTT over [B, T, D] sequences; hidden resets after done steps."""
        t_len = obs_seq.shape[1]
        h = h0
        acts = []
        for t in range(t_len):
            if t > 0:
                h = h * (1.0 - done_seq[:, t - 1])  # done_seq [B, T, 1] -> [B, 1]
            a, h = self.step(obs_seq[:, t], h)
            acts.append(a)
        return mx.stack(acts, axis=1)  # [B, T, 2]
