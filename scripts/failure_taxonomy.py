"""Classify a policy's failure episodes from trajectory features and render a
labeled trajectory mosaic per class so the taxonomy's semantics can be checked
by eye.

Rolls out each env's FIRST episode (like dagger.evaluate), recording the full
trajectory and the oracle geodesic cost-to-go each step, then buckets failures:

  collision        contact terminal
  near_goal        got geodesically within reach of the goal but never converted
                   (the wander-loop / limit-cycle family)
  stuck            ended parked or orbiting an oscillation pocket (<1.25 m
                   extent over the last 10 s) far from the goal
  loop_revisit     kept moving but re-traversed the same ground (looping)
  slow_progress    still making clear progress when the step budget ran out
  wander           none of the above: kept moving over fresh ground without
                   converting it into geodesic progress

Outputs: stdout summary, per-failure features JSON, and a mosaic PNG with one
row per class (trajectory colored blue early -> red late, x = closest approach).
"""

import argparse
import json
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np

from fastnav.policy import HEADS, NavPolicy, RecurrentNavPolicy
from fastnav.render import MosaicRenderer
from fastnav.scene import ScenePack
from fastnav.sim import Sim, SimConfig, noisy_config

GEO_CLIP = 50.0  # padded/obstacle geo regions are huge outliers (see PPOConfig)

CLASSES = ["collision", "near_goal", "stuck", "loop_revisit", "slow_progress", "wander"]
CLASS_COLORS = {  # BGR tile borders
    "collision": (40, 40, 220), "near_goal": (200, 60, 200), "stuck": (60, 140, 230),
    "loop_revisit": (40, 170, 90), "slow_progress": (190, 120, 40), "wander": (120, 120, 120),
}


def load_policy(path: str, kin: str):
    """Build the matching policy from checkpoint keys (head type, core type,
    width, ray count, omega bin count -- all inferred from weight shapes) plus
    the config.json the train scripts write next to the checkpoint.

    config.json is REQUIRED knowledge for use_pos: MLX excludes underscore
    attributes from parameters(), so `_scale` (which encodes it) is not in any
    checkpoint, and a --no-pos policy loaded with use_pos=True feeds scaled
    positions through enc weights that never saw gradient -- silent noise
    injection (~1pt GRU, ~15pts transformer)."""
    w = mx.load(path)
    keys = set(w.keys())
    enc_key = "enc.weight" if "enc.weight" in keys else None
    n_rays = (w[enc_key].shape[1] - 6) if enc_key else (w["layers.0.weight"].shape[1] - 4)
    cfg_path = Path(path).parent / "config.json"
    train_cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    # physics must match training (a dt-0.25 policy evaluated at dt 0.1 is garbage);
    # so must the goal-observation frame (a no-odometry policy expects it constant)
    cfg = SimConfig(kinematics=kin, n_rays=n_rays, dt=train_cfg.get("dt", 0.1),
                    max_steps=train_cfg.get("max_steps", 512),
                    odometry=train_cfg.get("odometry", True))
    head = train_cfg.get("head") or (
        "discrete_w" if any(k.startswith("head.wlin") for k in keys) else "continuous")
    kwargs = {"use_pos": train_cfg.get("use_pos", not train_cfg.get("no_pos", False)),
              "head_opts": train_cfg.get("head_opts", {})}
    if any(k.startswith("tfm.") for k in keys):
        opts = train_cfg.get("core_opts", {})
        opts["layers"] = 1 + max(int(k.split(".")[2]) for k in keys if k.startswith("tfm.blocks."))
        kwargs |= {"core": "transformer", "core_opts": opts, "hidden": w[enc_key].shape[0]}
    if "log_std" in keys:
        from fastnav.ppo import PPONavPolicy
        policy = PPONavPolicy(cfg, head=head, **kwargs)
    elif enc_key:
        policy = RecurrentNavPolicy(cfg, head=head, **kwargs)
    else:
        policy = NavPolicy(cfg)
    if head == "discrete_w":
        bins = w["head.wlin.bias"].shape[0]
        if bins != policy.head.bins:
            policy.head = HEADS[head](policy.hidden, policy.act_scale, bins=bins)
    if head == "waypoint_flow" and "head.inp.weight" in w:
        # conditioned heads add 2H input channels; a legacy (pre-conditioning)
        # checkpoint's narrower inp means conditioned=False. Rebuild to match.
        ho = dict(train_cfg.get("head_opts", {}))
        h = ho.get("horizon", policy.head.horizon)
        legacy_in = 2 * h + 1 + policy.hidden  # no +2H cond channels
        if w["head.inp.weight"].shape[1] == legacy_in:
            policy.head = HEADS[head](policy.hidden, policy.act_scale,
                                      **{**ho, "conditioned": False})
    policy.load_weights(path)
    mx.eval(policy.parameters())
    return policy, cfg


def rollout_record(sim: Sim, policy) -> dict:
    """First-episode rollout for every env, recording positions and the oracle
    geodesic cost-to-go each step (history past an env's first done is ignored
    downstream via steps[i])."""
    n = sim.num_envs
    t_max = sim.cfg.max_steps + 1
    sim.reset()
    recurrent = isinstance(policy, RecurrentNavPolicy)
    h = policy.new_state(n) if recurrent else None
    prev = mx.zeros((n, 2), dtype=mx.float32)

    pos_hist = np.zeros((t_max + 1, n, 2), dtype=np.float32)
    geo_hist = np.zeros((t_max + 1, n), dtype=np.float32)
    pos_hist[0] = np.array(sim.pos)
    sim.expert_actions()
    geo_hist[0] = np.clip(np.array(sim.expert_geo_val), 0.0, GEO_CLIP)
    init = {"start": pos_hist[0].copy(), "head": np.array(sim.heading),
            "goal": np.array(sim.goal), "goal_k": np.array(sim.goal_k),
            "scene": np.array(sim.scene)}

    # NOTE: the kernel auto-resets within the done step, so pos/geo at index
    # steps[i] already belong to the env's NEXT episode -- consumers must slice
    # [:steps[i]] (the true terminal position is lost; its predecessor is one
    # dt away, negligible for trajectory features).
    succeeded = np.zeros(n, dtype=bool)
    finished = np.zeros(n, dtype=bool)
    collided = np.zeros(n, dtype=bool)
    steps = np.full(n, t_max, dtype=np.int32)
    min_clear = np.full(n, 9.0, dtype=np.float32)
    for t in range(t_max):
        obs = sim.obs()
        if recurrent:
            act, h_new = policy.step(mx.concatenate([obs, prev], axis=1), h)
        else:
            act, h_new = policy(obs), None
        _, term, trunc = sim.step(act)
        if recurrent:
            live = 1.0 - mx.maximum(term, trunc).astype(mx.float32)[:, None]
            h = policy.mask_state(h_new, live)
            prev = act * live
        pos_hist[t + 1] = np.array(sim.pos)
        sim.expert_actions()
        geo_hist[t + 1] = np.clip(np.array(sim.expert_geo_val), 0.0, GEO_CLIP)
        min_clear = np.where(finished, min_clear, np.minimum(min_clear, np.array(sim.clearance)))
        term_np = np.array(term).astype(bool)
        trunc_np = np.array(trunc).astype(bool)
        first = (term_np | trunc_np) & ~finished
        steps[first] = t + 1
        succeeded |= first & term_np
        collided |= first & np.array(sim.hit).astype(bool)
        finished |= term_np | trunc_np
        if finished.all():
            break
    return {"pos": pos_hist, "geo": geo_hist, "steps": steps, "succeeded": succeeded,
            "collided": collided, "min_clear": min_clear, **init}


def episode_features(traj: np.ndarray, geo: np.ndarray, collided: bool) -> dict:
    """Trajectory features for classification (traj/geo already sliced to the
    first episode, length steps+1)."""
    seg = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    path_len = float(seg.sum())
    cells = np.unique(np.floor(traj / 0.25).astype(np.int64), axis=0)
    coverage = len(cells) * 0.25  # meters of unique corridor visited
    tail = traj[-min(100, len(traj)):]
    tail_extent = float(np.linalg.norm(tail.max(axis=0) - tail.min(axis=0)))
    t_back = max(0, len(geo) - 129)
    return {
        "g0": float(geo[0]), "g_min": float(geo.min()), "g_end": float(geo[-1]),
        "t_min": int(geo.argmin()), "path_len": path_len,
        "revisit_ratio": path_len / max(coverage, 1e-6),
        "tail_extent": tail_extent,
        "recent_progress": float(geo[t_back] - geo[-1]),
        "collided": bool(collided), "steps": len(traj),
    }


def classify(f: dict) -> str:
    if f["collided"]:
        return "collision"
    if f["g_min"] < 1.5:
        return "near_goal"
    if f["tail_extent"] < 1.25:
        return "stuck"
    if f["revisit_ratio"] > 2.5 and f["path_len"] > 5.0:
        return "loop_revisit"
    if f["recent_progress"] > 0.5:
        return "slow_progress"
    return "wander"


def representatives(fails: list[dict], cls: str, k: int) -> list[dict]:
    """Spread picks across the class-defining feature so the row shows the
    mild-to-severe range, not k near-duplicates."""
    rows = [f for f in fails if f["class"] == cls]
    key = {"collision": "g_end", "near_goal": "g_min", "stuck": "g_end",
           "loop_revisit": "revisit_ratio", "slow_progress": "g_end", "wander": "g_end"}[cls]
    rows.sort(key=lambda f: f[key])
    if len(rows) <= k:
        return rows
    return [rows[int(round(q * (len(rows) - 1)))] for q in np.linspace(0, 1, k)]


def draw_tile(ren: MosaicRenderer, rec: dict, f: dict, names: list[str],
              goal_radius: float) -> np.ndarray:
    i, s = f["env"], f["scene"]
    traj = rec["pos"][: f["steps"], i]
    img = ren.bgs[s].copy()
    pts = ren._to_px(s, traj)
    gpx = ren._to_px(s, rec["goal"][i])
    cv2.circle(img, gpx, max(3, int(goal_radius / ren.cell * ren.scales[s])), (60, 60, 230), 1,
               cv2.LINE_AA)
    cv2.circle(img, gpx, 3, (60, 60, 230), -1, cv2.LINE_AA)
    for j in range(len(pts) - 1):
        a = j / max(len(pts) - 2, 1)  # blue early -> red late
        cv2.line(img, pts[j], pts[j + 1],
                 (int(220 * (1 - a)), int(90 * (1 - abs(2 * a - 1))), int(230 * a)), 1, cv2.LINE_AA)
    cv2.circle(img, pts[0], 4, (70, 170, 40), -1, cv2.LINE_AA)
    m = ren._to_px(s, traj[f["t_min"]])
    cv2.drawMarker(img, m, (200, 40, 160), cv2.MARKER_TILTED_CROSS, 9, 2, cv2.LINE_AA)
    cv2.drawMarker(img, pts[-1], (30, 30, 30), cv2.MARKER_CROSS, 9, 2, cv2.LINE_AA)
    name = names[s].replace("Baked_", "").replace("_staging", "").replace("ProcTHOR-10k-", "")
    for li, txt in enumerate([f"{f['class']}  {name}",
                              f"g {f['g0']:.1f}>{f['g_end']:.1f} min {f['g_min']:.1f}  "
                              f"te{f['tail_extent']:.1f} rr{f['revisit_ratio']:.1f}"]):
        cv2.putText(img, txt, (6, 16 + 15 * li), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (30, 30, 30), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), CLASS_COLORS[f["class"]], 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="data/scenes")
    ap.add_argument("--include", nargs="+", default=["Baked_sc2_*", "Baked_sc3_*"])
    ap.add_argument("--checkpoint", default="checkpoints/ppo_dd_safe/policy_best.safetensors")
    ap.add_argument("--kinematics", default="diffdrive",
                    choices=["holonomic", "diffdrive", "diffdrive_vel"])
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--cols", type=int, default=6, help="examples per class row")
    ap.add_argument("--tag", default="heldout")
    ap.add_argument("--out-dir", default="runs/failure_taxonomy")
    args = ap.parse_args()

    pack = ScenePack.load_dir(args.scenes, include=args.include, max_cells=500000)
    policy, cfg = load_policy(args.checkpoint, args.kinematics)
    if args.noise > 0:
        cfg = noisy_config(cfg, args.noise)
    print(f"{args.tag}: {len(pack.scenes)} scenes, {args.envs} envs, "
          f"{cfg.n_rays} rays, noise {args.noise}")
    sim = Sim(pack, num_envs=args.envs, cfg=cfg, seed=args.seed)
    rec = rollout_record(sim, policy)

    n = args.envs
    fails = []
    for i in np.nonzero(~rec["succeeded"])[0]:
        n_steps = int(rec["steps"][i])
        f = episode_features(rec["pos"][:n_steps, i], rec["geo"][:n_steps, i],
                             rec["collided"][i])
        f.update(env=int(i), scene=int(rec["scene"][i]), min_clear=float(rec["min_clear"][i]))
        f["class"] = classify(f)
        fails.append(f)

    n_fail = len(fails)
    print(f"episodes {n}  success {(n - n_fail) / n * 100:.1f}%  failures {n_fail}")
    for cls in CLASSES:
        rows = [f for f in fails if f["class"] == cls]
        if not rows:
            continue
        med = {k: float(np.median([r[k] for r in rows]))
               for k in ["g0", "g_min", "g_end", "revisit_ratio", "tail_extent", "steps"]}
        print(f"  {cls:14s} {len(rows):4d}  ({len(rows) / max(n_fail, 1) * 100:4.1f}% of fail, "
              f"{len(rows) / n * 100:4.2f}% of eps)  median g {med['g0']:.1f}>{med['g_end']:.1f} "
              f"min {med['g_min']:.1f}  rr {med['revisit_ratio']:.1f}  "
              f"tail {med['tail_extent']:.2f}m  {med['steps']:.0f}st")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps(
        {"checkpoint": args.checkpoint, "include": args.include, "episodes": n,
         "noise": args.noise, "failures": fails}, indent=1))

    ren = MosaicRenderer(sim, [], cols=args.cols, tile_h=300)
    rows = []
    for cls in CLASSES:
        tiles = [draw_tile(ren, rec, f, pack.names, cfg.goal_radius)
                 for f in representatives(fails, cls, args.cols)]
        if not tiles:
            continue
        while len(tiles) < args.cols:
            tiles.append(np.full_like(tiles[0], 250))
        rows.append(np.concatenate(tiles, axis=1))
    if rows:
        mosaic = np.concatenate(rows, axis=0)
        header = np.full((34, mosaic.shape[1], 3), 250, dtype=np.uint8)
        cv2.putText(header, f"{Path(args.checkpoint).parent.name}  {args.tag}  "
                    f"noise {args.noise}  success {(n - n_fail) / n * 100:.1f}%  "
                    f"({n_fail}/{n} failures; one row per class, mild left -> severe right)",
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        png = out / f"{args.tag}.png"
        cv2.imwrite(str(png), np.concatenate([header, mosaic], axis=0))
        print(f"wrote {png}")
    else:
        print("no failures - nothing to render")


if __name__ == "__main__":
    main()
