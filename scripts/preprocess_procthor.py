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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ai2thor-hab/ai2thor-hab")
    ap.add_argument("--out", default="data/scenes_procthor")
    args = ap.parse_args()

    root = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fcfg = FieldConfig()

    cfgs = sorted(root.glob("configs/scenes/ProcTHOR/*/*.scene_instance.json"))
    print(f"{len(cfgs)} scene configs")
    ok, failed = 0, []
    for i, cfg in enumerate(cfgs):
        rooms = cfg.parent.name
        name = f"ProcTHOR-r{rooms}-{cfg.stem.replace('.scene_instance', '')}"
        try:
            tris = compose_scene(root, cfg, skip_names=SKIP_NAMES)
            occ, origin = rasterize_band(tris, fcfg.cell)
            scene = build_scene(name, occ, origin, fcfg, seed=i)
            nav = (scene.edf > fcfg.robot_radius).mean() * occ.size * fcfg.cell**2
            if nav < 6.0:
                raise ValueError(f"navigable area only {nav:.1f} m^2 (connectivity issue?)")
            scene.save(out / f"{name}.npz")
            debug_png(scene, out / f"{name}.png")
            print(f"[{i + 1}/{len(cfgs)}] {name}: grid {occ.shape}, nav ~{nav:.0f} m^2, "
                  f"starts {scene.start_counts.min()}-{scene.start_counts.max()}")
            ok += 1
        except Exception as e:
            failed.append((name, str(e)[:100]))
            print(f"[{i + 1}/{len(cfgs)}] {name}: FAILED ({e})")

    print(f"\n{ok} scenes ok, {len(failed)} failed")
    for n, e in failed:
        print(f"  {n}: {e}")
    if ok:
        pack = ScenePack.load_dir(out)
        print(f"pack: {len(pack.scenes)} scenes, grid {pack.grid_hw}, geo {pack.geo_hw}")


if __name__ == "__main__":
    main()
