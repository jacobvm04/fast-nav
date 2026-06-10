"""ProcTHOR (ai2thor-hab) scenes -> preprocessed Scene npz files.

Doors are skipped (treated as open) to preserve room connectivity, mirroring
the ReplicaCAD preprocessing.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from fastnav.compose import compose_scene, rasterize_band
from fastnav.scene import FieldConfig, ScenePack, build_scene

SKIP_NAMES = ("door",)


def debug_png(scene, path: Path):
    occ = scene.occupancy
    img = np.full((*occ.shape, 3), 255, dtype=np.uint8)
    img[occ > 0] = (40, 40, 40)
    for gx, gy in scene.goals_xy:
        ix = int((gx - scene.origin[0]) / scene.cell)
        iy = int((gy - scene.origin[1]) / scene.cell)
        cv2.circle(img, (ix, iy), 5, (0, 0, 255), -1)
    cv2.imwrite(str(path), img)


def process_one(args_tuple) -> tuple[str, bool, str]:
    cfg_path, root, out, seed, skip_existing = args_tuple
    root, out = Path(root), Path(out)
    cfg = Path(cfg_path)
    rooms = cfg.parent.name
    name = f"ProcTHOR-r{rooms}-{cfg.stem.replace('.scene_instance', '')}"
    if skip_existing and (out / f"{name}.npz").exists():
        return name, True, "cached"
    fcfg = FieldConfig()
    try:
        tris = compose_scene(root, cfg, skip_names=SKIP_NAMES)
        occ, origin = rasterize_band(tris, fcfg.cell)
        scene = build_scene(name, occ, origin, fcfg, seed=seed)
        nav = (scene.edf > fcfg.robot_radius).mean() * occ.size * fcfg.cell**2
        if nav < 6.0:
            raise ValueError(f"navigable area only {nav:.1f} m^2 (connectivity issue?)")
        scene.save(out / f"{name}.npz")
        debug_png(scene, out / f"{name}.png")
        return name, True, f"grid {occ.shape}, nav ~{nav:.0f} m^2"
    except Exception as e:
        return name, False, str(e)[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ai2thor-hab/ai2thor-hab")
    ap.add_argument("--out", default="data/scenes_procthor")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    root = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfgs = sorted(root.glob("configs/scenes/ProcTHOR/*/*.scene_instance.json"))
    print(f"{len(cfgs)} scene configs, {args.workers} workers")
    jobs = [(str(c), str(root), str(out), i, args.skip_existing) for i, c in enumerate(cfgs)]
    ok, failed = 0, []
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = pool.map(process_one, jobs, chunksize=4)
            for i, (name, good, msg) in enumerate(results):
                if good:
                    ok += 1
                    if msg != "cached":
                        print(f"[{i + 1}/{len(jobs)}] {name}: {msg}", flush=True)
                else:
                    failed.append((name, msg))
                    print(f"[{i + 1}/{len(jobs)}] {name}: FAILED ({msg})", flush=True)
    else:
        for i, job in enumerate(jobs):
            name, good, msg = process_one(job)
            ok += good
            if not good:
                failed.append((name, msg))
            print(f"[{i + 1}/{len(jobs)}] {name}: {msg if good else 'FAILED ' + msg}", flush=True)

    print(f"\n{ok} scenes ok, {len(failed)} failed")
    for n, e in failed:
        print(f"  {n}: {e}")


if __name__ == "__main__":
    main()
