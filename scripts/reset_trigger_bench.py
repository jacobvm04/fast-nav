"""Benchmark the deploy-time reset trigger against plain deployment.

The trigger is the treatment the continuation tests validated offline: an env
that has been within `dist` of its goal for `patience` consecutive steps
without finishing is (with high probability) in a carried-state limit cycle;
wiping its policy state + prev action moves it to the decisive cold conditional
(measured 72-85% conversion from branch states). This measures the same
intervention in-episode: net success/collision/timeout vs an untouched run on
identical seeds. Healthy episodes rarely linger near the goal, so the trigger's
false-positive cost should be ~0 (unlike the falsified periodic resets).
"""

import argparse

import mlx.core as mx

from failure_taxonomy import load_policy
from fastnav.scene import ScenePack
from fastnav.sim import Sim


def evaluate_reset(sim: Sim, policy, dist: float = 0.0, patience: int = 64) -> dict:
    """First-episode eval; dist=0 disables the trigger (plain deploy)."""
    sim.reset()
    n = sim.num_envs
    h = policy.new_state(n)
    prev = mx.zeros((n, 2), dtype=mx.float32)
    near_ct = mx.zeros((n,), dtype=mx.int32)
    succeeded = mx.zeros((n,), dtype=mx.bool_)
    finished = mx.zeros((n,), dtype=mx.bool_)
    collided = mx.zeros((n,), dtype=mx.bool_)
    fired = mx.zeros((n,), dtype=mx.bool_)
    for t in range(sim.cfg.max_steps + 1):
        act, h = policy.step(mx.concatenate([sim.obs(), prev], axis=1), h)
        _, term, trunc = sim.step(act)
        done = mx.maximum(term, trunc)
        live = (1.0 - done.astype(mx.float32))[:, None]
        h = policy.mask_state(h, live)
        prev = act * live
        if dist > 0:
            near = (sim.dist_goal < dist).astype(mx.int32) * (1 - done.astype(mx.int32))
            near_ct = (near_ct + near) * near
            wipe = near_ct >= patience
            keep = (1.0 - wipe.astype(mx.float32))[:, None]
            h = policy.mask_state(h, keep)
            prev = prev * keep
            near_ct = near_ct * (1 - wipe.astype(mx.int32))
            fired = mx.logical_or(fired, mx.logical_and(wipe, mx.logical_not(finished)))
        done_b = done.astype(mx.bool_)
        first = mx.logical_and(done_b, mx.logical_not(finished))
        succeeded = mx.logical_or(succeeded, mx.logical_and(first, term.astype(mx.bool_)))
        collided = mx.logical_or(collided, mx.logical_and(first, sim.hit.astype(mx.bool_)))
        finished = mx.logical_or(finished, done_b)
        if t % 64 == 0:
            mx.eval(finished)
            if bool(mx.all(finished)):
                break
    mx.eval(succeeded, collided, fired)
    n_suc, n_col = int(mx.sum(succeeded)), int(mx.sum(collided))
    return {"success": n_suc / n, "collision": n_col / n,
            "timeout": (n - n_suc - n_col) / n, "fired": int(mx.sum(fired)) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--scenes", default="data/scenes")
    ap.add_argument("--include", nargs="+", default=["Baked_sc2_*", "Baked_sc3_*"])
    ap.add_argument("--kinematics", default="diffdrive",
                    choices=["holonomic", "diffdrive", "diffdrive_vel"])
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 11])
    ap.add_argument("--dist", type=float, default=1.5)
    ap.add_argument("--patience", type=int, nargs="+", default=[64, 128])
    args = ap.parse_args()

    pack = ScenePack.load_dir(args.scenes, include=args.include, max_cells=500000)
    for path in args.checkpoints:
        policy, cfg = load_policy(path, args.kinematics)
        print(path)
        configs = [(0.0, 0)] + [(args.dist, p) for p in args.patience]
        for dist, patience in configs:
            rows = [evaluate_reset(Sim(pack, args.envs, cfg=cfg, seed=s), policy,
                                   dist=dist, patience=patience) for s in args.seeds]
            mean = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
            tag = "plain deploy " if dist == 0 else f"wipe @{dist}m/{patience}st"
            print(f"  {tag:>18}  success {mean['success']*100:5.1f}%  "
                  f"collision {mean['collision']*100:4.2f}%  timeout {mean['timeout']*100:4.2f}%  "
                  f"fired {mean['fired']*100:4.1f}%", flush=True)


if __name__ == "__main__":
    main()
