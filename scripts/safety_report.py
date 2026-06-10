"""Deployment scorecard: raw success, contact-free success, and contact rate
for each checkpoint, clean and under the realistic noise stack."""

import argparse
import sys
from pathlib import Path

import mlx.core as mx

from fastnav.dagger import evaluate
from fastnav.policy import RecurrentNavPolicy
from fastnav.scene import ScenePack
from fastnav.sim import Sim, SimConfig, noisy_config

sys.path.insert(0, str(Path(__file__).parent))
from policy_mosaic import load_policy

CKPTS = [
    ("clean-trained", "checkpoints/ppo/policy_best.safetensors"),
    ("noise-trained", "checkpoints/ppo_noisy/policy_best.safetensors"),
    ("safe-trained", "checkpoints/ppo_safe/policy_best.safetensors"),
]


def contact_rate(pack, policy, cfg, n=2048, steps=512, seed=5) -> float:
    sim = Sim(pack, num_envs=n, cfg=cfg, seed=seed)
    sim.reset()
    h = mx.zeros((n, policy.hidden))
    prev = mx.zeros((n, 2))
    contact = mx.zeros((n,))
    for _ in range(steps):
        obs = sim.obs()
        act, h = policy.step(mx.concatenate([obs, prev], axis=1), h)
        _, term, trunc = sim.step(act)
        contact = contact + (sim.clearance < 0.01)
        live = 1.0 - mx.maximum(term, trunc).astype(mx.float32)[:, None]
        h = h * live
        prev = act * live
    mx.eval(contact)
    return float(mx.sum(contact)) / (n * steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="data/scenes")
    ap.add_argument("--envs", type=int, default=4096)
    args = ap.parse_args()

    base = SimConfig()
    noisy = noisy_config(base, 1.0)
    rc = ScenePack.load_dir(args.scenes, include=["Baked_sc2_*", "Baked_sc3_*"])
    pt = ScenePack.load_dir(args.scenes, include=["ProcTHOR-*-Test-*", "ProcTHOR-*-Val-*"],
                            max_cells=500000)

    print(f"{'checkpoint':>14} {'env':>6} | {'RC succ':>8} {'RC safe':>8} {'RC contact':>10} | "
          f"{'PT succ':>8} {'PT safe':>8}")
    for name, path in CKPTS:
        if not Path(path).exists():
            print(f"{name:>14}  (missing: {path})")
            continue
        policy = load_policy(path, base)
        for env_name, cfg in (("clean", base), ("noisy", noisy)):
            ev_rc = evaluate(Sim(rc, num_envs=args.envs, cfg=cfg, seed=2), policy)
            ev_pt = evaluate(Sim(pt, num_envs=args.envs, cfg=cfg, seed=3), policy)
            cr = contact_rate(rc, policy, cfg)
            print(f"{name:>14} {env_name:>6} | {ev_rc['success'] * 100:>7.1f}% "
                  f"{ev_rc['safe_success'] * 100:>7.1f}% {cr * 100:>9.2f}% | "
                  f"{ev_pt['success'] * 100:>7.1f}% {ev_pt['safe_success'] * 100:>7.1f}%")


if __name__ == "__main__":
    main()
