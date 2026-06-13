"""Deploy-time commitment probe: does temporal mode-consistency kill the
boundary-surfing limit cycles, with zero retraining?

The residual timeout failures are the closed loop sitting ON the policy's
left/right decision boundary (argmax of a ~50/50 omega distribution flickers
each step, and the flicker holds the system at the boundary). If that story is
right, making the per-step readout hysteretic -- a logit bonus on the
previously chosen bin, or re-deciding only every k steps -- should collapse
the near-goal/stuck classes. This is the cheap discriminating test for whether
an action-chunk head (ACT / diffusion-policy / flow-matching style) is worth
building: chunking is the same commitment mechanism, baked in.
"""

import argparse

import mlx.core as mx
import numpy as np

from failure_taxonomy import load_policy
from fastnav.scene import ScenePack
from fastnav.sim import Sim


def evaluate_committed(sim: Sim, policy, bonus: float = 0.0, hold: int = 1) -> dict:
    """First-episode eval with a hysteretic omega readout: previous bin gets a
    logit bonus, and the decision is only revisited every `hold` steps.
    bonus=0, hold=1 reproduces plain argmax (parity-checked against evaluate)."""
    sim.reset()
    n = sim.num_envs
    head = policy.head
    h = policy.new_state(n)
    prev = mx.zeros((n, 2), dtype=mx.float32)
    prev_bin = mx.zeros((n,), dtype=mx.int32)
    age = mx.zeros((n,), dtype=mx.int32)  # steps since this env last re-decided
    bins = head._bin_values()
    succeeded = mx.zeros((n,), dtype=mx.bool_)
    finished = mx.zeros((n,), dtype=mx.bool_)
    collided = mx.zeros((n,), dtype=mx.bool_)
    for t in range(sim.cfg.max_steps + 1):
        obs = sim.obs()
        feat, h = policy._step_feature(mx.concatenate([obs, prev], axis=1), h)
        logits = head.wlin(feat)
        if bonus > 0.0:
            logits = logits + bonus * (mx.arange(head.bins)[None, :] == prev_bin[:, None])
        new_bin = mx.argmax(logits, axis=-1).astype(mx.int32)
        # first step of an episode always decides; afterwards only when age wraps
        decide = (age % hold) == 0
        w_bin = mx.where(decide, new_bin, prev_bin)
        act = mx.stack([head._v_mean(feat), bins[w_bin]], axis=-1)
        _, term, trunc = sim.step(act)
        done = mx.maximum(term, trunc)
        live = 1.0 - done.astype(mx.float32)
        h = policy.mask_state(h, live[:, None])
        prev = act * live[:, None]
        live_i = (1 - done).astype(mx.int32)
        prev_bin = w_bin * live_i
        age = (age + 1) * live_i
        done_b = done.astype(mx.bool_)
        first = mx.logical_and(done_b, mx.logical_not(finished))
        succeeded = mx.logical_or(succeeded, mx.logical_and(first, term.astype(mx.bool_)))
        collided = mx.logical_or(collided, mx.logical_and(first, sim.hit.astype(mx.bool_)))
        finished = mx.logical_or(finished, done_b)
        if t % 64 == 0:
            mx.eval(finished)
            if bool(mx.all(finished)):
                break
    mx.eval(succeeded, collided)
    n_suc, n_col = int(mx.sum(succeeded)), int(mx.sum(collided))
    return {"success": n_suc / n, "collision": n_col / n,
            "timeout": (n - n_suc - n_col) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="data/scenes")
    ap.add_argument("--include", nargs="+", default=["Baked_sc2_*", "Baked_sc3_*"])
    ap.add_argument("--checkpoint", default="checkpoints/tfm_ppo_dd/policy_best.safetensors")
    ap.add_argument("--kinematics", default="diffdrive",
                    choices=["holonomic", "diffdrive", "diffdrive_vel"])
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--bonus", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--hold", type=int, nargs="+", default=[1, 4, 8])
    args = ap.parse_args()

    pack = ScenePack.load_dir(args.scenes, include=args.include, max_cells=500000)
    policy, cfg = load_policy(args.checkpoint, args.kinematics)
    print(f"{args.checkpoint}  {len(pack.scenes)} scenes  {args.envs} envs  seed {args.seed}")
    print(f"{'config':>16}  {'success':>8}  {'collision':>9}  {'timeout':>8}")
    for bonus in args.bonus:
        ev = evaluate_committed(Sim(pack, args.envs, cfg=cfg, seed=args.seed), policy,
                                bonus=bonus, hold=1)
        print(f"{'bonus ' + format(bonus, '.1f'):>16}  {ev['success']*100:7.1f}%  "
              f"{ev['collision']*100:8.2f}%  {ev['timeout']*100:7.2f}%", flush=True)
    for hold in args.hold[1:]:
        ev = evaluate_committed(Sim(pack, args.envs, cfg=cfg, seed=args.seed), policy,
                                bonus=0.0, hold=hold)
        print(f"{'hold ' + str(hold):>16}  {ev['success']*100:7.1f}%  "
              f"{ev['collision']*100:8.2f}%  {ev['timeout']*100:7.2f}%", flush=True)


if __name__ == "__main__":
    main()
