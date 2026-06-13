"""Evaluate checkpoints head-to-head on identical episode seeds.

Eval noise at 2048-4096 episodes is +/-2pts, which has produced wrong
conclusions twice -- checkpoints are only comparable when each sees the SAME
first episodes. A fresh Sim per (checkpoint, seed) pins the episode draw to
the seed, so every checkpoint faces an identical episode set.
"""

import argparse

import numpy as np

from failure_taxonomy import load_policy
from fastnav.dagger import evaluate
from fastnav.scene import ScenePack
from fastnav.sim import Sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--scenes", default="data/scenes")
    ap.add_argument("--include", nargs="+", default=["Baked_sc2_*", "Baked_sc3_*"])
    ap.add_argument("--include2", nargs="+", default=None, help="optional second held-out set")
    ap.add_argument("--kinematics", default="diffdrive",
                    choices=["holonomic", "diffdrive", "diffdrive_vel"])
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    args = ap.parse_args()

    packs = [("heldout", ScenePack.load_dir(args.scenes, include=args.include, max_cells=500000))]
    if args.include2:
        packs.append(("heldout2",
                      ScenePack.load_dir(args.scenes, include=args.include2, max_cells=500000)))

    for path in args.checkpoints:
        policy, cfg = load_policy(path, args.kinematics)
        cells = []
        for name, pack in packs:
            succ = [evaluate(Sim(pack, num_envs=args.envs, cfg=cfg, seed=s), policy)["success"]
                    for s in args.seeds]
            cells.append(f"{name} {np.mean(succ) * 100:5.1f}% "
                         f"(seeds: {' '.join(f'{x * 100:.1f}' for x in succ)})")
        print(f"{path}\n  " + "  |  ".join(cells))


if __name__ == "__main__":
    main()
