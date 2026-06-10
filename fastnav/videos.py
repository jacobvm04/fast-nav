"""Short policy mosaic videos (random episodes / failure replays) for run logging.

Works with both feedforward and recurrent policies; failure mode hunts the
policy's first-episode failures on the given pack and replays them exactly
(deterministic policy + sim), tiles bordered red while the original failed
episode is still running.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np

from fastnav.policy import RecurrentNavPolicy
from fastnav.render import MosaicRenderer
from fastnav.scene import ScenePack
from fastnav.sim import Sim, SimConfig


def _policy_stepper(policy, n: int):
    recurrent = isinstance(policy, RecurrentNavPolicy)
    h = mx.zeros((n, policy.hidden), dtype=mx.float32) if recurrent else None
    prev = mx.zeros((n, 2), dtype=mx.float32)

    def step(sim: Sim):
        nonlocal h, prev
        obs = sim.obs()
        if recurrent:
            act, h_new = policy.step(mx.concatenate([obs, prev], axis=1), h)
        else:
            act, h_new = policy(obs), None
        _, term, trunc = sim.step(act)
        if recurrent:
            live = 1.0 - mx.maximum(term, trunc).astype(mx.float32)[:, None]
            h = h_new * live
            prev = act * live
        return term, trunc

    return step


def hunt_failures(pack: ScenePack, policy, cfg: SimConfig, n_envs: int = 2048, seed: int = 123):
    """First-episode failures: returns (start_pos, goal, goal_k, scene) arrays."""
    sim = Sim(pack, num_envs=n_envs, cfg=cfg, seed=seed)
    sim.reset()
    init_pos = np.array(sim.pos)
    init_goal = np.array(sim.goal)
    init_k = np.array(sim.goal_k)
    scenes = np.array(sim.scene)
    step = _policy_stepper(policy, n_envs)
    succeeded = np.zeros(n_envs, dtype=bool)
    finished = np.zeros(n_envs, dtype=bool)
    for _ in range(cfg.max_steps + 1):
        term, trunc = step(sim)
        term = np.array(term).astype(bool)
        trunc = np.array(trunc).astype(bool)
        first = (term | trunc) & ~finished
        succeeded |= first & term
        finished |= term | trunc
        if finished.all():
            break
    failed = np.nonzero(finished & ~succeeded)[0]
    return init_pos[failed], init_goal[failed], init_k[failed], scenes[failed]


def policy_mosaic_video(pack: ScenePack, policy, cfg: SimConfig | None = None,
                        failures: bool = False, n_tiles: int = 16, cols: int = 4,
                        frames: int = 240, seed: int = 7,
                        out_path: str | None = None) -> str | None:
    """Render a mosaic mp4; returns the path (None if failures requested but none found)."""
    cfg = cfg or SimConfig()
    if failures:
        pos, goal, gk, scenes = hunt_failures(pack, policy, cfg)
        if len(pos) == 0:
            return None
        k = min(n_tiles, len(pos))
        pick = np.random.default_rng(0).choice(len(pos), size=k, replace=False)
        sim = Sim(pack, num_envs=k, cfg=cfg, seed=seed, scene_assign=scenes[pick])
        sim.reset()
        sim.set_state(pos[pick], goal[pick], gk[pick])
        ids = list(range(k))
    else:
        sim = Sim(pack, num_envs=max(n_tiles, 64), cfg=cfg, seed=seed)
        sim.reset()
        ids = list(range(n_tiles))

    ren = MosaicRenderer(sim, ids, cols=cols, tile_h=220)
    stepper = _policy_stepper(policy, sim.num_envs)
    out_path = out_path or tempfile.mktemp(suffix=".mp4")
    raw = str(out_path) + ".raw.mp4"
    in_first = np.ones(sim.num_envs, dtype=bool) if failures else None
    writer = None
    for _ in range(frames):
        img = ren.frame(np.array(sim.pos), np.array(sim.goal), np.array(sim.lidar),
                        np.array(sim.scene), highlight=in_first)
        if writer is None:
            writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                     (img.shape[1], img.shape[0]))
        writer.write(img)
        term, trunc = stepper(sim)
        if in_first is not None:
            done = np.array(term).astype(bool) | np.array(trunc).astype(bool)
            in_first &= ~done
    writer.release()
    try:  # h264 for browser playback in wandb
        subprocess.run(["ffmpeg", "-y", "-i", raw, "-c:v", "libx264", "-crf", "28",
                        "-pix_fmt", "yuv420p", "-loglevel", "error", str(out_path)], check=True)
        Path(raw).unlink()
    except Exception:
        Path(raw).rename(out_path)
    return str(out_path)
