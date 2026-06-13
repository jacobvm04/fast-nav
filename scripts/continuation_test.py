"""Hidden-state continuation test on the policy's residual timeout failures.

Hunts first-episode timeouts (failure_taxonomy machinery), replays each failed
episode exactly (deterministic policy + clean sim) up to a late stuck step,
then continues with a FRESH step budget under three arms that differ only in
carried state:

  keep       GRU hidden + prev action carried over (control: the failure loop)
  wipe_h     hidden zeroed, prev action kept (isolates the recurrent state)
  wipe_both  hidden and prev action zeroed (full cold restart in place)

If wipe arms convert far above keep, the failures are still limit cycles of
the (policy, hidden, env) closed loop (gradient-horizon problem); if wiping no
longer rescues, the endgame policy itself is the bottleneck. Reference
(pre-BPTT64 champion, ppo_dd_disc): keep 19% vs wipe 67%.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from failure_taxonomy import CLASSES, classify, episode_features, load_policy, rollout_record
from fastnav.scene import ScenePack
from fastnav.sim import Sim, SimConfig


def continue_episode(sim: Sim, policy, h0: mx.array, prev0: mx.array) -> np.ndarray:
    """First-episode success mask over a fresh max_steps budget from the sim's
    current state (set_state already re-zeroed the step counter)."""
    n = sim.num_envs
    h, prev = h0, prev0
    succeeded = np.zeros(n, dtype=bool)
    finished = np.zeros(n, dtype=bool)
    for _ in range(sim.cfg.max_steps + 1):
        obs = sim.obs()
        act, h_new = policy.step(mx.concatenate([obs, prev], axis=1), h)
        _, term, trunc = sim.step(act)
        live = 1.0 - mx.maximum(term, trunc).astype(mx.float32)[:, None]
        h = policy.mask_state(h_new, live)
        prev = act * live
        term_np = np.array(term).astype(bool)
        first = (term_np | np.array(trunc).astype(bool)) & ~finished
        succeeded |= first & term_np
        finished |= first
        if finished.all():
            break
    return succeeded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="data/scenes")
    ap.add_argument("--include", nargs="+", default=["Baked_sc2_*", "Baked_sc3_*"])
    ap.add_argument("--checkpoint", default="checkpoints/ppo_dd_safe/policy_best.safetensors")
    ap.add_argument("--kinematics", default="diffdrive",
                    choices=["holonomic", "diffdrive", "diffdrive_vel"])
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--t-stuck", type=int, default=448,
                    help="replay step at which to branch (late = inside the terminal loop)")
    ap.add_argument("--tag", default="heldout")
    ap.add_argument("--out-dir", default="runs/failure_taxonomy")
    args = ap.parse_args()

    pack = ScenePack.load_dir(args.scenes, include=args.include, max_cells=500000)
    policy, cfg = load_policy(args.checkpoint, args.kinematics)
    print(f"{args.tag}: {len(pack.scenes)} scenes, {args.envs} envs, branch at t={args.t_stuck}")
    sim = Sim(pack, num_envs=args.envs, cfg=cfg, seed=args.seed)
    rec = rollout_record(sim, policy)

    fails = []
    for i in np.nonzero(~rec["succeeded"] & ~rec["collided"])[0]:
        n_steps = int(rec["steps"][i])
        if n_steps <= args.t_stuck:  # needs to still be running at the branch point
            continue
        f = episode_features(rec["pos"][:n_steps, i], rec["geo"][:n_steps, i], False)
        f["env"] = int(i)
        f["class"] = classify(f)
        fails.append(f)
    idx = np.array([f["env"] for f in fails])
    cls = np.array([f["class"] for f in fails])
    print(f"timeout failures: {len(idx)} "
          f"({', '.join(f'{c} {int((cls == c).sum())}' for c in CLASSES if (cls == c).any())})")

    # exact replay of the failed episodes up to the branch point
    rsim = Sim(pack, num_envs=len(idx), cfg=cfg, seed=args.seed + 1,
               scene_assign=rec["scene"][idx])
    rsim.reset()
    rsim.set_state(rec["start"][idx], rec["goal"][idx], rec["goal_k"][idx],
                   heading=rec["head"][idx])
    h = policy.new_state(len(idx))
    prev = mx.zeros((len(idx), 2), dtype=mx.float32)
    done_replay = np.zeros(len(idx), dtype=bool)
    for _ in range(args.t_stuck):
        obs = rsim.obs()
        act, h = policy.step(mx.concatenate([obs, prev], axis=1), h)
        _, term, trunc = rsim.step(act)
        prev = act
        done_replay |= np.array(term).astype(bool) | np.array(trunc).astype(bool)
    drift = np.linalg.norm(np.array(rsim.pos) - rec["pos"][args.t_stuck, idx], axis=1)
    print(f"replay drift: max {drift.max():.4f}m  diverged (done early): {done_replay.sum()}")
    valid = ~done_replay & (drift < 0.05)

    pose = np.array(rsim.pose)
    arms = {"keep": (h, prev),
            "wipe_h": (mx.zeros_like(h), prev),
            "wipe_both": (mx.zeros_like(h), mx.zeros_like(prev))}
    results = {}
    for name, (h0, prev0) in arms.items():
        rsim.set_state(pose[:, :2], rec["goal"][idx], rec["goal_k"][idx], heading=pose[:, 2])
        results[name] = continue_episode(rsim, policy, h0, prev0)
        conv = results[name] & valid
        line = f"  {name:10s} convert {conv.sum() / max(valid.sum(), 1) * 100:5.1f}%"
        for c in CLASSES:
            m = valid & (cls == c)
            if m.any():
                line += f"  {c} {(results[name] & m).sum()}/{m.sum()}"
        print(line)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}_continuation.json").write_text(json.dumps(
        {"checkpoint": args.checkpoint, "t_stuck": args.t_stuck, "n_valid": int(valid.sum()),
         "classes": cls.tolist(), "valid": valid.tolist(),
         "converted": {k: v.tolist() for k, v in results.items()}}, indent=1))
    print(f"wrote {out / f'{args.tag}_continuation.json'}")


if __name__ == "__main__":
    main()
