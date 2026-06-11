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

# per-head tensor manifests (names match fastnav.policy parameter paths)
_TRUNK = ["enc.weight", "enc.bias", "gru.Wx", "gru.Wh", "gru.b", "gru.bhn"]
POLICY_TENSORS = {
    "continuous": _TRUNK + ["head.weight", "head.bias", "vhead.weight", "vhead.bias"],
    "discrete_w": _TRUNK + ["head.vlin.weight", "head.vlin.bias", "head.wlin.weight",
                            "head.wlin.bias", "vhead.weight", "vhead.bias"],
}

# (id, label, checkpoint[, kinematics[, head]]) — exported in order; first
# existing one is the app default. kinematics/head default to holonomic/continuous.
POLICIES = [
    ("ppo-128", "PPO · 128-beam, noise-robust", "checkpoints/ppo_big_128init.safetensors"),
    ("ppo-diffdrive", "PPO · diff-drive (v, ω)",
     "checkpoints/ppo_dd_disc64/policy_best.safetensors", "diffdrive", "discrete_w"),
    ("ppo-big", "PPO · 3k-scene contact-safe", "checkpoints/ppo_big/policy_best.safetensors"),
    ("ppo-careful", "PPO · contact-safe (careful)", "checkpoints/ppo_careful2/policy_best.safetensors"),
    ("ppo-contact", "PPO · contact-safe", "checkpoints/ppo_contact2/policy_best.safetensors"),
    ("ppo-noisy", "PPO · noise-trained (DR 1.5)", "checkpoints/ppo_noisy/policy_best.safetensors"),
    ("ppo-clean", "PPO · clean-trained", "checkpoints/ppo/policy_best.safetensors"),
]


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


def export_policies(out: Path) -> None:
    import dataclasses

    import mlx.core as mx

    from fastnav.sim import SimConfig, noisy_config

    out_pol = out / "policies"
    out_pol.mkdir(parents=True, exist_ok=True)
    cfg = SimConfig()
    entries = []
    for pid, label, ckpt, *rest in POLICIES:
        kin = rest[0] if rest else "holonomic"
        head = rest[1] if len(rest) > 1 else "continuous"
        if not Path(ckpt).exists():
            print(f"skipping {pid}: {ckpt} not found")
            continue
        weights = mx.load(ckpt)
        blob = bytearray()
        tensors = {}
        for name in POLICY_TENSORS[head]:
            a = np.array(weights[name], dtype=np.float32)
            tensors[name] = {"shape": list(a.shape), "offset": len(blob) // 4}
            blob += a.tobytes()  # little-endian on every platform we care about
        (out_pol / f"{pid}.bin").write_bytes(bytes(blob))
        n_rays = tensors["enc.weight"]["shape"][1] - 6  # in = rays | rel_goal 2 | pos 2 | prev 2
        entries.append({
            "id": pid, "label": label, "checkpoint": ckpt, "n_rays": n_rays,
            "kinematics": kin, "head": head, "file": f"policies/{pid}.bin",
            "tensors": tensors,
            "arch": {"hidden": 256, "enc": 256, "use_pos": False},
        })
        print(f"exported {pid} ({len(blob) / 1e6:.1f} MB) <- {ckpt}")

    # per-step/per-episode noise sigmas at level 1.0 (the sim2real stack)
    noise = {f.name: getattr(noisy_config(cfg, 1.0), f.name)
             for f in dataclasses.fields(cfg)
             if f.name.startswith(("lidar_", "odom_", "head_", "act_")) and "goal" not in f.name}
    manifest = {
        "policies": entries,
        "sim": {
            "n_rays": cfg.n_rays, "max_range": cfg.max_range, "dt": cfg.dt,
            "v_max": cfg.v_max, "w_max": cfg.w_max, "robot_radius": cfg.robot_radius,
            "goal_radius": cfg.goal_radius, "max_steps": cfg.max_steps,
        },
        "noise_stack": noise,  # multiply by UI level (1.0 = realistic)
        "val_scale": 20.0,  # value head output * this = est. cost-to-go in meters
    }
    (out / "policies.json").write_text(json.dumps(manifest, indent=1))


def export_fixture(scenes_dir: Path, ckpt: Path, out: Path, scene_name: str, steps: int,
                   n_rays: int = 64) -> None:
    """Run the MLX sim + policy from a fixed state; dump everything the JS port
    needs to replay it (occupancy, state, per-step pos/action/lidar)."""
    import mlx.core as mx

    from fastnav.policy import RecurrentNavPolicy
    from fastnav.scene import Scene, ScenePack
    from fastnav.sim import Sim, SimConfig

    scene = Scene.load(scenes_dir / f"{scene_name}.npz")
    pack = ScenePack([scene])
    cfg = SimConfig(n_rays=n_rays)
    sim = Sim(pack, num_envs=1, cfg=cfg, seed=0)
    sim.reset()

    policy = RecurrentNavPolicy(cfg, hidden=256, enc=256, use_pos=False)
    policy.load_weights(str(ckpt), strict=False)  # ignore log_std from PPO wrapper
    mx.eval(policy.parameters())

    # pick a far start/goal pair from the precomputed tables (goal 0). The episode
    # must end in a clean goal-reach: any trunc (timeout OR contact-terminal) would
    # auto-reset the MLX sim mid-fixture and corrupt the reference trajectory.
    k = 0
    n_starts = int(scene.start_counts[k])
    goal = scene.goals_xy[k]
    traj = []
    for attempt in range(20):
        start = scene.starts_xy[k, n_starts - 1 - attempt * 37]
        sim.set_state(start[None].astype(np.float32), goal[None].astype(np.float32),
                      np.array([k], dtype=np.int32))
        h = mx.zeros((1, 256))
        prev = mx.zeros((1, 2))
        traj = []
        clean = False
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
            mx.eval(term, trunc)
            if bool(trunc[0]):
                break  # contact or timeout: discard, try another start
            if bool(term[0]):
                clean = True
                break
            prev = act
        if clean:
            print(f"fixture: goal reached at step {len(traj)} (start attempt {attempt})")
            break
    else:
        raise RuntimeError("no clean fixture episode found in 20 attempts")

    out_test = out / "test"
    out_test.mkdir(parents=True, exist_ok=True)
    fixture = {
        "scene": scene_name,
        "n_rays": cfg.n_rays,
        "ckpt": str(ckpt),
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
    ap.add_argument("--fixture-rays", type=int, default=64)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    export_scenes(Path(args.scenes), out)
    export_policies(out)
    if args.fixture:
        export_fixture(Path(args.scenes), Path(args.ckpt), out,
                       args.fixture_scene, args.fixture_steps, n_rays=args.fixture_rays)


if __name__ == "__main__":
    main()
