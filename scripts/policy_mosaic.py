"""Mosaic of the trained policy on held-out scenes; --failures replays failure cases.

Failure mode: run a full first-episode evaluation on the held-out pack, collect the
(start, goal) of every failed episode, then replay them exactly (policy is
deterministic) in a mosaic. Tiles keep a red border while their original failed
episode is still running; after it ends the env continues with random episodes.
"""

import argparse

import cv2
import mlx.core as mx
import numpy as np

from fastnav.policy import NavPolicy
from fastnav.render import MosaicRenderer
from fastnav.scene import ScenePack
from fastnav.sim import Sim, SimConfig


def load_policy(path: str, cfg: SimConfig) -> NavPolicy:
    policy = NavPolicy(cfg)
    policy.load_weights(path)
    mx.eval(policy.parameters())
    return policy


def hunt_failures(pack: ScenePack, policy: NavPolicy, cfg: SimConfig, n_envs: int = 4096,
                  seed: int = 123):
    sim = Sim(pack, num_envs=n_envs, cfg=cfg, seed=seed)
    sim.reset()
    init_pos = np.array(sim.pos)
    init_goal = np.array(sim.goal)
    init_goal_k = np.array(sim.goal_k)
    scenes = np.array(sim.scene)
    succeeded = np.zeros(n_envs, dtype=bool)
    finished = np.zeros(n_envs, dtype=bool)
    for _ in range(cfg.max_steps + 1):
        obs, term, trunc = sim.step(policy(sim.obs()))
        term = np.array(term).astype(bool)
        trunc = np.array(trunc).astype(bool)
        first = (term | trunc) & ~finished
        succeeded |= first & term
        finished |= term | trunc
        if finished.all():
            break
    failed = np.nonzero(finished & ~succeeded)[0]
    print(f"failures: {len(failed)}/{finished.sum()} episodes "
          f"({100 * (1 - succeeded.sum() / finished.sum()):.1f}% failure rate)")
    return init_pos[failed], init_goal[failed], init_goal_k[failed], scenes[failed]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="data/scenes")
    ap.add_argument("--include", nargs="+", default=["Baked_sc2_*", "Baked_sc3_*"])
    ap.add_argument("--checkpoint", default="checkpoints/dagger/policy.safetensors")
    ap.add_argument("--failures", action="store_true")
    ap.add_argument("--envs", type=int, default=24)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="policy_mosaic.mp4")
    args = ap.parse_args()

    pack = ScenePack.load_dir(args.scenes, include=args.include)
    cfg = SimConfig()
    policy = load_policy(args.checkpoint, cfg)

    if args.failures:
        pos, goal, goal_k, scenes = hunt_failures(pack, policy, cfg)
        if len(pos) == 0:
            print("no failures found")
            return
        k = min(args.envs, len(pos))
        pick = np.random.default_rng(0).choice(len(pos), size=k, replace=False)
        sim = Sim(pack, num_envs=k, cfg=cfg, seed=7, scene_assign=scenes[pick])
        sim.reset()
        sim.set_state(pos[pick], goal[pick], goal_k[pick])
        env_ids = list(range(k))
    else:
        sim = Sim(pack, num_envs=max(args.envs, 64), cfg=cfg, seed=7)
        sim.reset()
        env_ids = list(range(args.envs))

    ren = MosaicRenderer(sim, env_ids, cols=args.cols)
    in_first = np.ones(sim.num_envs, dtype=bool) if args.failures else None
    writer = None
    for _ in range(args.frames):
        img = ren.frame(np.array(sim.pos), np.array(sim.goal), np.array(sim.lidar),
                        np.array(sim.scene), highlight=in_first)
        if writer is None:
            four = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.out, four, args.fps, (img.shape[1], img.shape[0]))
        writer.write(img)
        obs, term, trunc = sim.step(policy(sim.obs()))
        if in_first is not None:
            done = np.array(term).astype(bool) | np.array(trunc).astype(bool)
            in_first &= ~done
    writer.release()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
