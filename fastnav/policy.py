"""Tiny MLP policy with built-in observation normalization (checkpoint-portable)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from fastnav.sim import SimConfig


class NavPolicy(nn.Module):
    def __init__(self, cfg: SimConfig, hidden: int = 256, depth: int = 2):
        super().__init__()
        self.n_rays = cfg.n_rays
        self.v_max = cfg.v_max
        # lidar in [0, max_range]; rel_goal up to ~scene diameter; pos in scene extent
        scale = [1.0 / cfg.max_range] * cfg.n_rays + [1.0 / cfg.max_range] * 2 + [0.1] * 2
        self._scale = mx.array(scale)
        dims = [cfg.obs_dim] + [hidden] * depth + [2]
        self.layers = [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]

    def __call__(self, obs: mx.array) -> mx.array:
        x = obs * self._scale
        for layer in self.layers[:-1]:
            x = nn.silu(layer(x))
        return self.v_max * mx.tanh(self.layers[-1](x))
