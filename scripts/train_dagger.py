"""Train a nav policy with hyper-online DAgger; eval on held-out scene layouts.

Scene split (same apartment shell, different furniture layouts):
  train: Baked_sc0_*, Baked_sc1_*
  eval:  Baked_sc2_*, Baked_sc3_*   (never seen by the policy)
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from fastnav.dagger import DaggerConfig, DaggerTrainer, evaluate
from fastnav.scene import ScenePack
from fastnav.sim import Sim, SimConfig

TRAIN_PATTERNS = ["Baked_sc0_*", "Baked_sc1_*"]
EVAL_PATTERNS = ["Baked_sc2_*", "Baked_sc3_*"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="data/scenes")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--envs", type=int, default=8192)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-envs", type=int, default=4096)
    ap.add_argument("--out", default="checkpoints/dagger")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train_pack = ScenePack.load_dir(args.scenes, include=TRAIN_PATTERNS)
    eval_pack = ScenePack.load_dir(args.scenes, include=EVAL_PATTERNS)
    print(f"train scenes: {len(train_pack.scenes)}  eval scenes (held out): {len(eval_pack.scenes)}")

    scfg = SimConfig()
    sim = Sim(train_pack, num_envs=args.envs, cfg=scfg, seed=args.seed)
    sim.reset()
    trainer = DaggerTrainer(sim, DaggerConfig(), seed=args.seed)
    n_params = sum(v.size for _, v in tree_flatten(trainer.policy.parameters()))
    print(f"policy params: {n_params:,}")

    sim_train_eval = Sim(train_pack, num_envs=args.eval_envs, cfg=scfg, seed=args.seed + 1)
    sim_eval = Sim(eval_pack, num_envs=args.eval_envs, cfg=scfg, seed=args.seed + 2)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    frames = 0
    t0 = time.perf_counter()
    for it in range(1, args.iters + 1):
        loss = trainer.step()
        frames += args.envs * trainer.cfg.chunk
        if it % args.eval_every == 0 or it == args.iters:
            el = time.perf_counter() - t0
            ev_tr = evaluate(sim_train_eval, trainer.policy)
            ev_he = evaluate(sim_eval, trainer.policy)
            row = {
                "iter": it, "frames": frames, "loss": loss, "beta": trainer.beta,
                "train_success": ev_tr["success"], "heldout_success": ev_he["success"],
                "elapsed_s": el, "frames_per_s": frames / el,
            }
            history.append(row)
            print(f"it {it:4d}  frames {frames / 1e6:7.1f}M  loss {loss:.4f}  beta {trainer.beta:.2f}  "
                  f"train {ev_tr['success'] * 100:5.1f}%  HELD-OUT {ev_he['success'] * 100:5.1f}%  "
                  f"({frames / el / 1e6:.1f}M fps)")

    weights = dict(tree_flatten(trainer.policy.parameters()))
    mx.save_safetensors(str(out / "policy.safetensors"), weights)
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print(f"saved {out}/policy.safetensors")


if __name__ == "__main__":
    main()
