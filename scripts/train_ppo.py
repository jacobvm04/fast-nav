"""PPO fine-tune from a BC checkpoint on the geodesic-progress reward."""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from fastnav.dagger import evaluate
from fastnav.ppo import PPOConfig, PPOTrainer
from fastnav.scene import ScenePack
from fastnav.sim import Sim, SimConfig, noisy_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="data/scenes")
    ap.add_argument("--init", default="checkpoints/gru256_bc/policy.safetensors")
    ap.add_argument("--iters", type=int, default=3200)
    ap.add_argument("--envs", type=int, default=8192)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-envs", type=int, default=4096)
    ap.add_argument("--out", default="checkpoints/ppo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-include", nargs="+",
                    default=["ProcTHOR-*-Train-*", "Baked_sc0_*", "Baked_sc1_*"])
    ap.add_argument("--eval-include", nargs="+", default=["Baked_sc2_*", "Baked_sc3_*"])
    ap.add_argument("--eval2-include", nargs="+",
                    default=["ProcTHOR-*-Test-*", "ProcTHOR-*-Val-*"])
    ap.add_argument("--max-cells", type=int, default=500000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--entropy-coef", type=float, default=5e-4)
    ap.add_argument("--init-std", type=float, default=0.3)
    ap.add_argument("--wandb", dest="use_wandb", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--wandb-project", default="fast-nav")
    ap.add_argument("--video-every", type=int, default=800)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="sim2real noise level in training rollouts (1.0 = realistic)")
    ap.add_argument("--collision-penalty", type=float, default=0.25)
    ap.add_argument("--clear-coef", type=float, default=0.012)
    ap.add_argument("--speed-prox-coef", type=float, default=0.012)
    ap.add_argument("--clear-margin", type=float, default=0.10)
    ap.add_argument("--train-contact-margin", type=float, default=None,
                    help="inflated contact-terminal margin during training (eval stays at default)")
    ap.add_argument("--rotate-every", type=int, default=0,
                    help="resample a fresh train-scene subset every N iters (0 = off)")
    ap.add_argument("--rotate-size", type=int, default=800, help="scenes per rotated pack")
    args = ap.parse_args()

    import fnmatch
    import random

    import numpy as np

    from fastnav.scene import Scene

    def sample_pack(pool: list, k: int, seed: int) -> ScenePack:
        rng = random.Random(seed)
        return ScenePack([Scene.load(f) for f in rng.sample(pool, min(k, len(pool)))])

    # train pack may use a tighter size cap (GPU memory); evals stay at the
    # established 500k protocol for comparability across runs
    def edf_cells(path: Path) -> int:
        import zipfile

        from numpy.lib import format as npfmt
        with zipfile.ZipFile(path) as z, z.open("edf.npy") as fh:
            shape, _, _ = npfmt._read_array_header(fh, npfmt.read_magic(fh))
        return int(np.prod(shape))

    pool = None
    if args.rotate_every:
        cand = [f for f in sorted(Path(args.scenes).glob("*.npz"))
                if any(fnmatch.fnmatch(f.stem, p) for p in args.train_include)]
        pool = [f for f in cand if edf_cells(f) <= args.max_cells]
        print(f"rotation pool: {len(pool)} of {len(cand)} scenes (size cap {args.max_cells})")
        train_pack = sample_pack(pool, args.rotate_size, args.seed)
    else:
        train_pack = ScenePack.load_dir(args.scenes, include=args.train_include, max_cells=args.max_cells)
    eval_pack = ScenePack.load_dir(args.scenes, include=args.eval_include, max_cells=500000)
    eval2_pack = ScenePack.load_dir(args.scenes, include=args.eval2_include, max_cells=500000)
    print(f"train {len(train_pack.scenes)} / eval {len(eval_pack.scenes)} / eval2 {len(eval2_pack.scenes)} scenes")

    import dataclasses

    scfg = SimConfig()
    train_cfg = noisy_config(scfg, args.noise) if args.noise > 0 else scfg
    if args.train_contact_margin is not None:
        train_cfg = dataclasses.replace(train_cfg, contact_margin=args.train_contact_margin)
    sim = Sim(train_pack, num_envs=args.envs, cfg=train_cfg, seed=args.seed)
    sim.reset()
    pcfg = PPOConfig(lr=args.lr, entropy_coef=args.entropy_coef, init_std=args.init_std,
                     collision_penalty=args.collision_penalty, clear_coef=args.clear_coef,
                     clear_margin=args.clear_margin, speed_prox_coef=args.speed_prox_coef)
    init = args.init if args.init and Path(args.init).exists() else None
    trainer = PPOTrainer(sim, pcfg, seed=args.seed, init_weights=init)
    n_params = sum(v.size for _, v in tree_flatten(trainer.policy.parameters()))
    print(f"init: {init}  params: {n_params:,}")

    sim_tr = Sim(train_pack, num_envs=args.eval_envs, cfg=scfg, seed=args.seed + 1)
    sim_he = Sim(eval_pack, num_envs=args.eval_envs, cfg=scfg, seed=args.seed + 2)
    sim_h2 = Sim(eval2_pack, num_envs=args.eval_envs, cfg=scfg, seed=args.seed + 3)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = None
    if args.use_wandb:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, name=out.name,
                             config={**vars(args), "policy_params": n_params})
        except Exception as e:
            print(f"wandb disabled: {e}")

    history = []
    frames = 0
    best = 0.0
    t0 = time.perf_counter()
    for it in range(1, args.iters + 1):
        if pool and it > 1 and it % args.rotate_every == 1:
            trainer.sim = None  # free the old pack before loading the next (memory spike)
            train_pack = sample_pack(pool, args.rotate_size, args.seed + it)
            trainer.swap_sim(Sim(train_pack, num_envs=args.envs, cfg=train_cfg, seed=args.seed + it))
            print(f"rotated train pack at it {it} ({len(train_pack.scenes)} scenes)")
        stats = trainer.step()
        frames += args.envs * trainer.cfg.chunk
        if run:
            run.log({f"ppo/{k}": v for k, v in stats.items()}
                    | {"frames": frames, "time/fps_avg": frames / (time.perf_counter() - t0)},
                    step=it)
        if it % args.eval_every == 0 or it == args.iters:
            ev_tr = evaluate(sim_tr, trainer.policy)
            ev_he = evaluate(sim_he, trainer.policy)
            ev_h2 = evaluate(sim_h2, trainer.policy)
            row = {"iter": it, "frames": frames, **stats,
                   "train_success": ev_tr["success"], "heldout_success": ev_he["success"],
                   "heldout2_success": ev_h2["success"]}
            history.append(row)
            if run:
                run.log({"eval/train_success": ev_tr["success"],
                         "eval/heldout_success": ev_he["success"],
                         "eval/heldout2_success": ev_h2["success"]}, step=it)
            print(f"it {it:4d}  frames {frames / 1e6:7.1f}M  R {stats['reward_mean']:+.4f}  "
                  f"std {stats['std']:.3f}  roll-succ {stats['rollout_success'] * 100:5.1f}%  "
                  f"train {ev_tr['success'] * 100:5.1f}%  HELD-OUT {ev_he['success'] * 100:5.1f}%  "
                  f"HELD-OUT2 {ev_h2['success'] * 100:5.1f}%  "
                  f"safe {ev_he['safe_success'] * 100:5.1f}%/{ev_h2['safe_success'] * 100:5.1f}%")
            combined = ev_he["success"] + ev_h2["success"]
            if combined > best:
                best = combined
                weights = dict(tree_flatten(trainer.policy.parameters()))
                mx.save_safetensors(str(out / "policy_best.safetensors"), weights)
        if run and args.video_every and (it % args.video_every == 0 or it == args.iters):
            import wandb

            from fastnav.videos import policy_mosaic_video
            for fail, tag in ((False, "video/random"), (True, "video/failures")):
                path = policy_mosaic_video(eval2_pack, trainer.policy, cfg=scfg, failures=fail)
                if path:
                    run.log({tag: wandb.Video(path, format="mp4")}, step=it)

    weights = dict(tree_flatten(trainer.policy.parameters()))
    mx.save_safetensors(str(out / "policy_final.safetensors"), weights)
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print(f"saved {out}/policy_final.safetensors (best combined held-out: {best / 2 * 100:.1f}%)")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
