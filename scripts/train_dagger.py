"""Train a nav policy with hyper-online DAgger; eval on held-out scene layouts.

Scene split (same apartment shell, different furniture layouts):
  train: Baked_sc0_*, Baked_sc1_*
  eval:  Baked_sc2_*, Baked_sc3_*   (never seen by the policy)
"""

import argparse
import dataclasses
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
    ap.add_argument("--init", default=None,
                    help="warm-start weights (extend/resume a BC run; beta schedule restarts)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-include", nargs="+", default=TRAIN_PATTERNS)
    ap.add_argument("--eval-include", nargs="+", default=EVAL_PATTERNS)
    ap.add_argument("--rotate-every", type=int, default=0,
                    help="resample a fresh train-scene subset every N iters (0 = off); "
                         "needed when train_include spans more scenes than fit in GPU memory "
                         "(e.g. ProcTHOR-Train). Mirrors train_ppo.")
    ap.add_argument("--rotate-size", type=int, default=400, help="scenes per rotated pack")
    ap.add_argument("--rotate-pin", nargs="+", default=["Baked_*"],
                    help="patterns for scenes pinned into EVERY rotated pack (small sets that "
                         "random sampling would otherwise miss, e.g. ReplicaCAD)")
    ap.add_argument("--eval2-include", nargs="+", default=None, help="optional second held-out set")
    ap.add_argument("--eval2-max-scenes", type=int, default=128,
                    help="cap eval2 scene count: the stacked geo field is one mx buffer "
                         "and Metal limits a single allocation to ~20GB")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--lidar-noise", type=float, default=0.0)
    ap.add_argument("--ray-dropout", type=float, default=0.0)
    ap.add_argument("--no-pos", action="store_true")
    ap.add_argument("--recurrent", action="store_true")
    ap.add_argument("--max-cells", type=int, default=None, help="drop scenes larger than H*W cells")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=32768)
    ap.add_argument("--updates-per-iter", type=int, default=4)
    ap.add_argument("--wandb", dest="use_wandb", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--wandb-project", default="fast-nav")
    ap.add_argument("--video-every", type=int, default=0,
                    help="log random + failure mosaic videos to wandb every N iters (0=off)")
    ap.add_argument("--chunk", type=int, default=16, help="rollout chunk / BPTT length")
    ap.add_argument("--burn-in", type=int, default=0, help="BPTT warmup steps without loss")
    ap.add_argument("--value-weight", type=float, default=0.5)
    ap.add_argument("--detour-min", type=float, default=0.0,
                    help="min geodesic/euclidean ratio for episode starts")
    ap.add_argument("--kinematics", default="holonomic", choices=["holonomic", "diffdrive", "diffdrive_vel"])
    ap.add_argument("--odometry", action=argparse.BooleanOptionalAction, default=True,
                    help="--no-odometry: believed pose pinned at the episode start, so the "
                         "goal observation is a start-frame constant (see SimConfig.odometry)")
    ap.add_argument("--dt", type=float, default=0.1, help="control timestep (s); evals match")
    ap.add_argument("--lidar-offset", type=float, default=0.0,
                    help="lidar mount fwd of robot center (m); sim2real lever arm, evals match")
    ap.add_argument("--lidar-offset-sigma", type=float, default=0.0,
                    help="per-episode lidar mount randomization sigma (m); train only")
    ap.add_argument("--act-latency", type=int, default=0,
                    help="max command delay (steps), per-episode ~U{0..max}; evals match")
    ap.add_argument("--max-steps", type=int, default=512,
                    help="episode step budget (scale with dt to keep wall-time)")
    ap.add_argument("--head", default="continuous",
                    choices=["continuous", "discrete_w", "flow_chunk", "waypoint_flow"],
                    help="policy action head (fastnav.policy.HEADS)")
    ap.add_argument("--wp-horizon", type=int, default=8,
                    help="waypoint_flow: predicted waypoints per plan")
    ap.add_argument("--wp-stride", type=int, default=2,
                    help="waypoint_flow: sim steps between waypoints")
    ap.add_argument("--wp-replan", type=int, default=None,
                    help="waypoint_flow: steps a plan executes before resampling "
                         "(default horizon*stride/2)")
    ap.add_argument("--wp-cond-dropout", type=float, default=0.25,
                    help="waypoint_flow: prob of zeroing the previous-plan conditioning "
                         "in training (temporal-coherence head)")
    ap.add_argument("--wp-cond-noise", type=float, default=1.5,
                    help="waypoint_flow: std of noise on the conditioning (anti-leak; the "
                         "expert prev-plan overlaps the answer, so it must be corrupted)")
    ap.add_argument("--wp-unconditioned", action="store_true",
                    help="waypoint_flow: disable prev-plan conditioning entirely")
    ap.add_argument("--core", default="gru", choices=["gru", "transformer"],
                    help="memory core (fastnav.policy.CORES), recurrent only")
    ap.add_argument("--context", type=int, default=64, help="transformer sliding window")
    ap.add_argument("--core-layers", type=int, default=3, help="transformer blocks")
    ap.add_argument("--core-heads", type=int, default=4, help="transformer attention heads")
    ap.add_argument("--log-every", type=int, default=50,
                    help="stdout progress line every N iters (0 = eval rows only)")
    ap.add_argument("--token-frac", type=float, default=1.0,
                    help="flow_chunk only: fraction of sequence positions per update")
    ap.add_argument("--state-dropout", type=float, default=0.0,
                    help="per-step prob of zeroing an env's carried state mid-episode")
    ap.add_argument("--takeover-dist", type=float, default=0.0,
                    help="expert-takeover trigger radius (m); 0 = off (see DaggerConfig)")
    ap.add_argument("--takeover-patience", type=int, default=64)
    ap.add_argument("--takeover-len", type=int, default=64)
    ap.add_argument("--max-goal-dist", type=float, default=None,
                    help="cap train-episode geodesic length (m); evals keep the standard range")
    args = ap.parse_args()

    # large carried states (transformer KV rings) churn GB-scale buffers; an
    # uncapped MLX buffer cache ratchets into swap and gets the run OOM-killed
    mx.set_cache_limit(8 << 30)

    # scene-pack rotation: when train_include spans more scenes than fit in GPU
    # memory (ProcTHOR-Train is ~7k), sample a fresh rotate_size subset every
    # rotate_every iters instead of loading all at once (mirrors train_ppo).
    import fnmatch
    import random as _random

    from fastnav.scene import Scene

    def edf_cells(path: Path) -> int:
        import zipfile

        import numpy as _np
        from numpy.lib import format as npfmt
        with zipfile.ZipFile(path) as z, z.open("edf.npy") as fh:
            shape, _, _ = npfmt._read_array_header(fh, npfmt.read_magic(fh))
        return int(_np.prod(shape))

    # `pin` scenes (small sets that must stay in EVERY rotated pack, e.g. the
    # ReplicaCAD Baked layouts) are always present; the rest of the budget is a
    # fresh sample from the large pool. Without pinning, a 400-sample of the
    # ~3.5k-scene pool draws 0 of the 19 Baked scenes, so the policy trains on
    # pure ProcTHOR and ReplicaCAD-heldout collapses.
    def sample_pack(pool: list, pin: list, k: int, seed: int) -> ScenePack:
        rng = _random.Random(seed)
        rest = [f for f in pool if f not in pin]
        chosen = pin + rng.sample(rest, min(max(k - len(pin), 0), len(rest)))
        return ScenePack([Scene.load(f) for f in chosen])

    rot_pool = rot_pin = None
    if args.rotate_every:
        cand = [f for f in sorted(Path(args.scenes).glob("*.npz"))
                if any(fnmatch.fnmatch(f.stem, p) for p in args.train_include)]
        cap = args.max_cells or 10**12
        rot_pool = [f for f in cand if edf_cells(f) <= cap]
        rot_pin = [f for f in rot_pool
                   if any(fnmatch.fnmatch(f.stem, p) for p in args.rotate_pin)]
        print(f"rotation pool: {len(rot_pool)} of {len(cand)} scenes (size cap {args.max_cells}); "
              f"pinning {len(rot_pin)} every pack")
        train_pack = sample_pack(rot_pool, rot_pin, args.rotate_size, args.seed)
    else:
        train_pack = ScenePack.load_dir(args.scenes, include=args.train_include, max_cells=args.max_cells)
    eval_pack = ScenePack.load_dir(args.scenes, include=args.eval_include, max_cells=args.max_cells)
    print(f"train scenes: {len(train_pack.scenes)}  eval scenes (held out): {len(eval_pack.scenes)}")

    scfg = SimConfig(detour_min=args.detour_min, kinematics=args.kinematics,
                     dt=args.dt, max_steps=args.max_steps, odometry=args.odometry,
                     lidar_offset=args.lidar_offset, lidar_offset_sigma=args.lidar_offset_sigma,
                     act_latency=args.act_latency)
    if args.max_goal_dist is not None:
        scfg = dataclasses.replace(scfg, max_goal_dist=args.max_goal_dist)
    sim = Sim(train_pack, num_envs=args.envs, cfg=scfg, seed=args.seed)
    sim.reset()
    core_opts = ({"context": args.context, "layers": args.core_layers, "heads": args.core_heads}
                 if args.core == "transformer" else {})
    head_opts = {"token_frac": args.token_frac} if args.head == "flow_chunk" else {}
    if args.head == "waypoint_flow":
        # the head's follower mirrors the sim's diffdrive_vel dynamics, so it
        # needs the physics constants alongside the plan shape
        head_opts = {"horizon": args.wp_horizon, "stride": args.wp_stride,
                     "dt": args.dt, "w_max": scfg.w_max, "token_frac": args.token_frac,
                     "cond_dropout": args.wp_cond_dropout, "cond_noise": args.wp_cond_noise,
                     "conditioned": not args.wp_unconditioned, "kinematics": args.kinematics,
                     **({"replan": args.wp_replan} if args.wp_replan else {})}
    dcfg = DaggerConfig(hidden=args.hidden, depth=args.depth, augment=args.augment,
                        lidar_noise=args.lidar_noise, ray_dropout=args.ray_dropout,
                        use_pos=not args.no_pos, lr=args.lr, batch_size=args.batch_size,
                        updates_per_iter=args.updates_per_iter, chunk=args.chunk,
                        burn_in=args.burn_in, value_weight=args.value_weight, head=args.head,
                        core=args.core, core_opts=core_opts, head_opts=head_opts,
                        state_dropout=args.state_dropout, takeover_dist=args.takeover_dist,
                        takeover_patience=args.takeover_patience, takeover_len=args.takeover_len)
    cls = RecurrentDaggerTrainer if args.recurrent else DaggerTrainer
    trainer = cls(sim, dcfg, seed=args.seed)
    if args.init:
        # strict=False mirrors train_ppo's init load: a PPO checkpoint warm-starts
        # a BC run by dropping its extra log_std
        trainer.policy.load_weights(args.init, strict=False)
        mx.eval(trainer.policy.parameters())
    n_params = sum(v.size for _, v in tree_flatten(trainer.policy.parameters()))
    print(f"init: {args.init}  params: {n_params:,}")

    # evals keep the standard episode distribution (no curriculum) but must
    # match the training physics (dt / step budget)
    eval_cfg = SimConfig(kinematics=args.kinematics, dt=args.dt, max_steps=args.max_steps,
                         odometry=args.odometry, lidar_offset=args.lidar_offset,
                         act_latency=args.act_latency)
    sim_train_eval = Sim(train_pack, num_envs=args.eval_envs, cfg=eval_cfg, seed=args.seed + 1)
    sim_eval = Sim(eval_pack, num_envs=args.eval_envs, cfg=eval_cfg, seed=args.seed + 2)
    sim_eval2 = None
    video_pack = eval_pack
    if args.eval2_include:
        eval2_pack = ScenePack.load_dir(args.scenes, include=args.eval2_include, max_cells=args.max_cells)
        # the stacked geo field is ONE mx buffer; Metal caps a single allocation
        # at ~20GB, so a few hundred medium scenes (e.g. all of ProcTHOR-Test)
        # OOM at construction regardless of per-scene max_cells. Cap the count
        # and report the drop (silent truncation would read as full coverage).
        if len(eval2_pack.scenes) > args.eval2_max_scenes:
            kept = eval2_pack.scenes[:args.eval2_max_scenes]
            print(f"second held-out set: capping {len(eval2_pack.scenes)} -> "
                  f"{len(kept)} scenes (--eval2-max-scenes, avoids >20GB geo buffer)")
            eval2_pack = ScenePack(kept)
        print(f"second held-out set: {len(eval2_pack.scenes)} scenes")
        sim_eval2 = Sim(eval2_pack, num_envs=args.eval_envs, cfg=eval_cfg, seed=args.seed + 3)
        video_pack = eval2_pack

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # the recipe next to the weights: load_policy needs non-inferable core opts
    (out / "config.json").write_text(json.dumps(
        {**vars(args), "core_opts": core_opts, "head_opts": head_opts}, indent=1))

    run = None
    if args.use_wandb:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, name=out.name,
                             config={**vars(args), "policy_params": n_params})
        except Exception as e:
            print(f"wandb disabled: {e}")

    def gpu_util() -> float | None:
        import re
        import subprocess
        out_ = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
                              capture_output=True, text=True).stdout
        m = re.search(r'"Device Utilization %"=(\d+)', out_)
        return float(m.group(1)) if m else None

    history = []
    frames = 0
    t0 = time.perf_counter()
    for it in range(1, args.iters + 1):
        if rot_pool and it > 1 and it % args.rotate_every == 1:
            train_pack = sample_pack(rot_pool, rot_pin, args.rotate_size, args.seed + it)
            trainer.swap_sim(Sim(train_pack, num_envs=args.envs, cfg=scfg, seed=args.seed + it))
            print(f"rotated train pack at it {it} ({len(train_pack.scenes)} scenes)", flush=True)
        t_a = time.perf_counter()
        trainer.rollout()
        t_b = time.perf_counter()
        loss = trainer.train()
        trainer.iter += 1
        t_c = time.perf_counter()
        frames += args.envs * trainer.cfg.chunk
        if run:
            log = {"loss": loss, "beta": trainer.beta, "frames": frames,
                   "time/rollout_ms": (t_b - t_a) * 1e3, "time/train_ms": (t_c - t_b) * 1e3,
                   "time/fps_inst": args.envs * trainer.cfg.chunk / (t_c - t_a)}
            if it % 50 == 0:
                log["sys/gpu_util"] = gpu_util()
                log["sys/mlx_peak_gb"] = mx.get_peak_memory() / 1024**3
            run.log(log, step=it)
        if args.log_every and it % args.log_every == 0:
            take = (f"  takeover {trainer.takeover_frac * 100:.1f}%"
                    if getattr(trainer, "takeover_frac", 0) else "")
            print(f"it {it:4d}  loss {loss:.4f}  beta {trainer.beta:.2f}  "
                  f"roll {(t_b - t_a) * 1e3:4.0f}ms  train {(t_c - t_b) * 1e3:4.0f}ms  "
                  f"fps {args.envs * trainer.cfg.chunk / (t_c - t_a) / 1e3:.0f}k  "
                  f"mlx peak {mx.get_peak_memory() / 1024**3:.1f}GB{take}", flush=True)
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
            if run:
                run.log({"eval/train_success": ev_tr["success"],
                         "eval/heldout_success": ev_he["success"],
                         **({"eval/heldout2_success": row["heldout2_success"]}
                            if "heldout2_success" in row else {}),
                         "eval/steps_per_episode": ev_he["steps_per_episode"]}, step=it)
            print(f"it {it:4d}  frames {frames / 1e6:7.1f}M  loss {loss:.4f}  beta {trainer.beta:.2f}  "
                  f"train {ev_tr['success'] * 100:5.1f}%  HELD-OUT {ev_he['success'] * 100:5.1f}%{extra}  "
                  f"({frames / el / 1e6:.1f}M fps)", flush=True)
            mx.save_safetensors(str(out / "policy.safetensors"),
                                dict(tree_flatten(trainer.policy.parameters())))
            (out / "history.json").write_text(json.dumps(history, indent=1))
            mx.clear_cache()  # return ballooned eval/rollout buffers to the OS (else swap ratchets)
        if run and args.video_every and (it % args.video_every == 0 or it == args.iters):
            import wandb

            from fastnav.videos import policy_mosaic_video
            for fail, tag in ((False, "video/random"), (True, "video/failures")):
                path = policy_mosaic_video(video_pack, trainer.policy, cfg=eval_cfg, failures=fail)
                if path:
                    run.log({tag: wandb.Video(path, format="mp4")}, step=it)

    weights = dict(tree_flatten(trainer.policy.parameters()))
    mx.save_safetensors(str(out / "policy.safetensors"), weights)
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print(f"saved {out}/policy.safetensors")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
