"""Sweep network sizes / training params; rank by held-out success.

Runs configs through scripts/train_dagger.py as subprocesses, a few workers at
a time (each worker holds its own scene pack — keep an eye on RAM, not GPU).
Uses small-house ProcTHOR buckets (1-3 rooms) + ReplicaCAD sc0/sc1 for fast,
low-memory runs; eval on the matching held-out splits.
"""

import argparse
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TRAIN = ["ProcTHOR-r[123]-*-Train-*", "Baked_sc0_*", "Baked_sc1_*"]
EVAL = ["Baked_sc2_*", "Baked_sc3_*"]
EVAL2 = ["ProcTHOR-r[123]-*-Test-*", "ProcTHOR-r[123]-*-Val-*"]

CONFIGS = [
    {"name": "mlp128x2", "hidden": 128, "depth": 2},
    {"name": "mlp256x2", "hidden": 256, "depth": 2},
    {"name": "mlp512x2", "hidden": 512, "depth": 2},
    {"name": "mlp256x3", "hidden": 256, "depth": 3},
    {"name": "mlp512x3", "hidden": 512, "depth": 3},
    {"name": "mlp256x2_lr1e3", "hidden": 256, "depth": 2, "lr": 1e-3},
    {"name": "mlp256x2_upd8", "hidden": 256, "depth": 2, "updates-per-iter": 8},
    {"name": "mlp256x2_batch64k", "hidden": 256, "depth": 2, "batch-size": 65536},
    {"name": "gru128", "hidden": 128, "recurrent": True},
    {"name": "gru256", "hidden": 256, "recurrent": True},
]


def run_config(cfg: dict, args) -> dict:
    name = cfg["name"]
    out = Path(args.out) / name
    cmd = [sys.executable, "scripts/train_dagger.py",
           "--iters", str(args.iters), "--eval-every", str(args.iters // 2),
           "--envs", str(args.envs), "--eval-envs", "2048",
           "--augment", "--no-pos", "--max-cells", "250000",
           "--video-every", str(args.iters // 2),
           "--out", str(out)]
    for pats, flag in ((TRAIN, "--train-include"), (EVAL, "--eval-include"), (EVAL2, "--eval2-include")):
        cmd += [flag, *pats]
    for k, v in cfg.items():
        if k == "name":
            continue
        if k == "recurrent":
            cmd.append("--recurrent")
        else:
            cmd += [f"--{k}", str(v)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[{name}] FAILED:\n{res.stdout[-500:]}\n{res.stderr[-500:]}")
        return {"name": name, "error": True}
    hist = json.loads((out / "history.json").read_text())[-1]
    row = {"name": name, **{k: hist[k] for k in
           ("train_success", "heldout_success", "heldout2_success", "loss", "frames_per_s")}}
    print(f"[{name}] train {row['train_success']*100:.1f}%  RC-held-out {row['heldout_success']*100:.1f}%  "
          f"ProcTHOR-held-out {row['heldout2_success']*100:.1f}%")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1600)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="checkpoints/sweep")
    args = ap.parse_args()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda c: run_config(c, args), CONFIGS))

    rows = [r for r in rows if not r.get("error")]
    rows.sort(key=lambda r: -(r["heldout_success"] + r["heldout2_success"]))
    print(f"\n{'config':>18} {'train':>7} {'RC-out':>7} {'PT-out':>7} {'loss':>7} {'fps':>9}")
    for r in rows:
        print(f"{r['name']:>18} {r['train_success']*100:>6.1f}% {r['heldout_success']*100:>6.1f}% "
              f"{r['heldout2_success']*100:>6.1f}% {r['loss']:>7.3f} {r['frames_per_s']/1e6:>7.1f}M")
    Path(args.out, "summary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
