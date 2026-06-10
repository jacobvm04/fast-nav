"""Export held-out scenes + the PPO policy for the browser demo in web/.

Scenes go out as 1-bit occupancy PNGs (the browser recomputes the signed EDF
with an exact Euclidean distance transform); the policy goes out as raw
little-endian float32 with a JSON manifest of tensor offsets and sim constants.

  uv run python scripts/export_web.py            # scenes + policy
  uv run python scripts/export_web.py --fixture  # also dump an MLX trajectory
                                                 # fixture for web/test/parity.mjs
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import json
from pathlib import Path

import numpy as np
from PIL import Image

HELDOUT = {
    "ReplicaCAD": ["Baked_sc2_*", "Baked_sc3_*"],
    "ProcTHOR-Val": ["ProcTHOR-*-Val-*"],
    "ProcTHOR-Test": ["ProcTHOR-*-Test-*"],
}
MAX_CELLS = 500_000  # same filter as training eval packs

POLICY_TENSORS = ["enc.weight", "enc.bias", "gru.Wx", "gru.Wh", "gru.b", "gru.bhn",
                  "head.weight", "head.bias", "vhead.weight", "vhead.bias"]


def export_scenes(scenes_dir: Path, out: Path) -> None:
    out_scenes = out / "scenes"
    out_scenes.mkdir(parents=True, exist_ok=True)
    index = []
    for group, patterns in HELDOUT.items():
        files = sorted(f for f in scenes_dir.glob("*.npz")
                       if any(fnmatch.fnmatch(f.stem, p) for p in patterns))
        for f in files:
            d = np.load(f, allow_pickle=False)
            occ = d["occupancy"]
            h, w = occ.shape
            if h * w > MAX_CELLS:
                continue
            img = Image.fromarray((occ == 0)).convert("1")  # free=white, occupied=black
            img.save(out_scenes / f"{f.stem}.png", optimize=True)
            index.append({
                "name": f.stem,
                "group": group,
                "file": f"scenes/{f.stem}.png",
                "h": int(h), "w": int(w),
                "cell": float(d["cell"]),
                "origin": [float(d["origin"][0]), float(d["origin"][1])],
            })
    (out / "scene_index.json").write_text(json.dumps(index, indent=1))
    print(f"exported {len(index)} scenes -> {out_scenes}")


def export_policy(ckpt: Path, out: Path) -> None:
    import mlx.core as mx

    from fastnav.sim import SimConfig

    weights = mx.load(str(ckpt))
    cfg = SimConfig()
    blob = bytearray()
    tensors = {}
    for name in POLICY_TENSORS:
        a = np.array(weights[name], dtype=np.float32)
        tensors[name] = {"shape": list(a.shape), "offset": len(blob) // 4}
        blob += a.tobytes()  # little-endian on every platform we care about
    (out / "policy.bin").write_bytes(bytes(blob))
    manifest = {
        "checkpoint": str(ckpt),
        "tensors": tensors,
        "arch": {"hidden": 256, "enc": 256, "use_pos": False},
        "sim": {
            "n_rays": cfg.n_rays, "max_range": cfg.max_range, "dt": cfg.dt,
            "v_max": cfg.v_max, "robot_radius": cfg.robot_radius,
            "goal_radius": cfg.goal_radius, "max_steps": cfg.max_steps,
        },
        "val_scale": 20.0,  # value head output * this = est. cost-to-go in meters
    }
    (out / "policy.json").write_text(json.dumps(manifest, indent=1))
    print(f"exported policy ({len(blob) // 4} floats, {len(blob) / 1e6:.1f} MB) -> {out / 'policy.bin'}")


def export_fixture(scenes_dir: Path, ckpt: Path, out: Path, scene_name: str, steps: int) -> None:
    """Run the MLX sim + policy from a fixed state; dump everything the JS port
    needs to replay it (occupancy, state, per-step pos/action/lidar)."""
    import mlx.core as mx

    from fastnav.policy import RecurrentNavPolicy
    from fastnav.scene import Scene, ScenePack
    from fastnav.sim import Sim, SimConfig

    scene = Scene.load(scenes_dir / f"{scene_name}.npz")
    pack = ScenePack([scene])
    cfg = SimConfig()
    sim = Sim(pack, num_envs=1, cfg=cfg, seed=0)
    sim.reset()

    policy = RecurrentNavPolicy(cfg, hidden=256, enc=256, use_pos=False)
    policy.load_weights(str(ckpt), strict=False)  # ignore log_std from PPO wrapper
    mx.eval(policy.parameters())

    # pick a far start/goal pair from the precomputed tables (goal 0, hardest start)
    k = 0
    n_starts = int(scene.start_counts[k])
    start = scene.starts_xy[k, n_starts - 1]
    goal = scene.goals_xy[k]
    sim.set_state(start[None].astype(np.float32), goal[None].astype(np.float32),
                  np.array([k], dtype=np.int32))

    h = mx.zeros((1, 256))
    prev = mx.zeros((1, 2))
    traj = []
    for _ in range(steps):
        obs = sim.obs()
        act, h = policy.step(mx.concatenate([obs, prev], axis=1), h)
        mx.eval(act, h)
        traj.append({
            "pos": np.array(sim.pos)[0].tolist(),
            "lidar": np.array(sim.lidar)[0].tolist(),
            "act": np.array(act)[0].tolist(),
        })
        _, term, trunc = sim.step(act)
        mx.eval(term)
        live = 1.0 - mx.maximum(term, trunc).astype(mx.float32)[:, None]
        h = h * live
        prev = act * live
        if bool(term[0]):
            print(f"fixture: goal reached at step {len(traj)}")
            break

    out_test = out / "test"
    out_test.mkdir(parents=True, exist_ok=True)
    fixture = {
        "scene": scene_name,
        "h": scene.occupancy.shape[0], "w": scene.occupancy.shape[1],
        "cell": float(scene.cell),
        "origin": [float(scene.origin[0]), float(scene.origin[1])],
        "occupancy_b64": base64.b64encode(scene.occupancy.tobytes()).decode(),
        "edf_row128_b64": base64.b64encode(
            scene.edf[min(128, scene.edf.shape[0] - 1)].astype(np.float32).tobytes()).decode(),
        "start": start.tolist(),
        "goal": goal.tolist(),
        "steps": len(traj),
        "traj": traj,
    }
    (out_test / "fixture.json").write_text(json.dumps(fixture))
    print(f"fixture: {len(traj)} steps -> {out_test / 'fixture.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="data/scenes")
    ap.add_argument("--ckpt", default="checkpoints/ppo/policy_best.safetensors")
    ap.add_argument("--out", default="web")
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--fixture-scene", default="Baked_sc3_staging_01")
    ap.add_argument("--fixture-steps", type=int, default=300)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    export_scenes(Path(args.scenes), out)
    export_policy(Path(args.ckpt), out)
    if args.fixture:
        export_fixture(Path(args.scenes), Path(args.ckpt), out,
                       args.fixture_scene, args.fixture_steps)


if __name__ == "__main__":
    main()
