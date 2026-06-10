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

from fastnav.dagger import DaggerConfig, DaggerTrainer, RecurrentDaggerTrainer, evaluate
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
    ap.add_argument("--train-include", nargs="+", default=TRAIN_PATTERNS)
    ap.add_argument("--eval-include", nargs="+", default=EVAL_PATTERNS)
    ap.add_argument("--eval2-include", nargs="+", default=None, help="optional second held-out set")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--lidar-noise", type=float, default=0.0)
    ap.add_argument("--ray-dropout", type=float, default=0.0)
    ap.add_argument("--no-pos", action="store_true")
    ap.add_argument("--recurrent", action="store_true")
    ap.add_argument("--max-cells", type=int, default=None, help="drop scenes larger than H*W cells")
    args = ap.parse_args()

    train_pack = ScenePack.load_dir(args.scenes, include=args.train_include, max_cells=args.max_cells)
    eval_pack = ScenePack.load_dir(args.scenes, include=args.eval_include, max_cells=args.max_cells)
    print(f"train scenes: {len(train_pack.scenes)}  eval scenes (held out): {len(eval_pack.scenes)}")

    scfg = SimConfig()
    sim = Sim(train_pack, num_envs=args.envs, cfg=scfg, seed=args.seed)
    sim.reset()
    dcfg = DaggerConfig(hidden=args.hidden, depth=args.depth, augment=args.augment,
                        lidar_noise=args.lidar_noise, ray_dropout=args.ray_dropout,
                        use_pos=not args.no_pos)
    cls = RecurrentDaggerTrainer if args.recurrent else DaggerTrainer
    trainer = cls(sim, dcfg, seed=args.seed)
    n_params = sum(v.size for _, v in tree_flatten(trainer.policy.parameters()))
    print(f"policy params: {n_params:,}")

    sim_train_eval = Sim(train_pack, num_envs=args.eval_envs, cfg=scfg, seed=args.seed + 1)
    sim_eval = Sim(eval_pack, num_envs=args.eval_envs, cfg=scfg, seed=args.seed + 2)
    sim_eval2 = None
    if args.eval2_include:
        eval2_pack = ScenePack.load_dir(args.scenes, include=args.eval2_include, max_cells=args.max_cells)
        print(f"second held-out set: {len(eval2_pack.scenes)} scenes")
        sim_eval2 = Sim(eval2_pack, num_envs=args.eval_envs, cfg=scfg, seed=args.seed + 3)

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
            extra = ""
            if sim_eval2 is not None:
                ev2 = evaluate(sim_eval2, trainer.policy)
                row["heldout2_success"] = ev2["success"]
                extra = f"  HELD-OUT2 {ev2['success'] * 100:5.1f}%"
            history.append(row)
            print(f"it {it:4d}  frames {frames / 1e6:7.1f}M  loss {loss:.4f}  beta {trainer.beta:.2f}  "
                  f"train {ev_tr['success'] * 100:5.1f}%  HELD-OUT {ev_he['success'] * 100:5.1f}%{extra}  "
                  f"({frames / el / 1e6:.1f}M fps)")

    weights = dict(tree_flatten(trainer.policy.parameters()))
    mx.save_safetensors(str(out / "policy.safetensors"), weights)
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print(f"saved {out}/policy.safetensors")


if __name__ == "__main__":
    main()
