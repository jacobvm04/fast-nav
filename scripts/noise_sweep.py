"""Zero-shot sim2real brittleness sweep: evaluate a checkpoint across noise types.

Each condition isolates one error source at a realistic and a severe level,
plus combined stacks. Reports success on both held-out packs and at a relaxed
goal radius (drift puts a physical floor under final-approach precision).
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from fastnav.dagger import evaluate
from fastnav.scene import ScenePack
from fastnav.sim import Sim, SimConfig

sys.path.insert(0, str(Path(__file__).parent))
from policy_mosaic import load_policy

REALISTIC = {
    "lidar": dict(lidar_sigma=0.02, lidar_dropout=0.02),
    "odom": dict(odom_rw=0.03, odom_bias=0.02, odom_scale=0.02),
    "heading": dict(head_rw=0.005, head_bias=0.003),
    "actuation": dict(act_noise=0.1, act_scale=0.05),
}
SEVERE = {
    "lidar": dict(lidar_sigma=0.05, lidar_dropout=0.10),
    "odom": dict(odom_rw=0.08, odom_bias=0.05, odom_scale=0.05),
    "heading": dict(head_rw=0.015, head_bias=0.010),
    "actuation": dict(act_noise=0.25, act_scale=0.15),
}


def conditions() -> list[tuple[str, dict]]:
    conds = [("clean", {})]
    for k, v in REALISTIC.items():
        conds.append((f"{k}", v))
    for k, v in SEVERE.items():
        conds.append((f"{k}-severe", v))
    all_real = {k: v for d in REALISTIC.values() for k, v in d.items()}
    all_sev = {k: v for d in SEVERE.values() for k, v in d.items()}
    conds.append(("ALL-realistic", all_real))
    conds.append(("ALL-severe", all_sev))
    return conds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="data/scenes")
    ap.add_argument("--checkpoint", default="checkpoints/ppo/policy_best.safetensors")
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--max-cells", type=int, default=500000)
    ap.add_argument("--out", default="checkpoints/noise_sweep.json")
    args = ap.parse_args()

    base = SimConfig()
    policy = load_policy(args.checkpoint, base)
    rc = ScenePack.load_dir(args.scenes, include=["Baked_sc2_*", "Baked_sc3_*"])
    pt = ScenePack.load_dir(args.scenes, include=["ProcTHOR-*-Test-*", "ProcTHOR-*-Val-*"],
                            max_cells=args.max_cells)

    rows = []
    print(f"{'condition':>16} {'RC@0.25':>9} {'PT@0.25':>9} {'RC@0.5':>9}")
    for name, kw in conditions():
        cfg = dataclasses.replace(base, **kw)
        cfg_r5 = dataclasses.replace(cfg, goal_radius=0.5)
        ev_rc = evaluate(Sim(rc, num_envs=args.envs, cfg=cfg, seed=2), policy)
        ev_pt = evaluate(Sim(pt, num_envs=args.envs, cfg=cfg, seed=3), policy)
        ev_r5 = evaluate(Sim(rc, num_envs=args.envs, cfg=cfg_r5, seed=2), policy)
        row = {"condition": name, **kw,
               "rc": ev_rc["success"], "pt": ev_pt["success"], "rc_r05": ev_r5["success"]}
        rows.append(row)
        print(f"{name:>16} {ev_rc['success'] * 100:>8.1f}% {ev_pt['success'] * 100:>8.1f}% "
              f"{ev_r5['success'] * 100:>8.1f}%")

    Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
