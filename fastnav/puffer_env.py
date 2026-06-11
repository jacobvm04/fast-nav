"""PufferLib native vectorized env over the MLX sim.

One instance = the whole batch (num_agents = num_envs), stepped in a single
GPU dispatch. Rewards are a placeholder (zeros) until reward design starts;
sim.term / sim.dist_goal already expose everything shaping will need.
"""

from __future__ import annotations

import gymnasium
import numpy as np
import pufferlib

from fastnav import kinematics
from fastnav.scene import ScenePack
from fastnav.sim import Sim, SimConfig


class FastNavEnv(pufferlib.PufferEnv):
    def __init__(self, scene_dir: str = "data/scenes", num_envs: int = 4096,
                 sim_config: SimConfig | None = None, render_mode=None, buf=None, seed: int = 0):
        cfg = sim_config or SimConfig()
        high = np.full(cfg.obs_dim, np.inf, dtype=np.float32)
        high[: cfg.n_rays] = cfg.max_range
        self.single_observation_space = gymnasium.spaces.Box(low=-high, high=high, dtype=np.float32)
        act_scale = kinematics.get(cfg.kinematics).action_scale(cfg)
        self.single_action_space = gymnasium.spaces.Box(
            low=-act_scale, high=act_scale, dtype=np.float32)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.agents_per_batch = num_envs  # trainer RNN path reads the plural name
        super().__init__(buf)

        pack = ScenePack.load_dir(scene_dir)
        self.sim = Sim(pack, num_envs=num_envs, cfg=cfg, seed=seed)

    def reset(self, seed=None):
        obs = self.sim.reset()
        self.observations[:] = np.array(obs, copy=False)
        return self.observations, []

    def step(self, actions):
        import mlx.core as mx

        obs, term, trunc = self.sim.step(mx.array(np.asarray(actions, dtype=np.float32)))
        self.observations[:] = np.array(obs, copy=False)
        self.rewards[:] = 0.0  # reward design intentionally deferred
        self.terminals[:] = np.array(term, copy=False).astype(bool)
        self.truncations[:] = np.array(trunc, copy=False).astype(bool)
        return self.observations, self.rewards, self.terminals, self.truncations, []

    def expert_actions(self) -> np.ndarray:
        return np.array(self.sim.expert_actions(), copy=False)

    def close(self):
        pass
