"""ReplicaCAD baked scenes -> preprocessed Scene npz files.

Loads each baked stage GLB (furniture included), composites articulated-object
base links from the scene_instance.json (doors skipped = treated as open),
slices the robot height band, rasterizes the XZ projection to an occupancy
grid, then runs the field pipeline. World (x, y) := mesh (X, Z); height = Y.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

from fastnav.scene import FieldConfig, ScenePack, build_scene

# articulated template_name -> urdf (relative to urdf_uncompressed/)
URDF_FILES = {
    "fridge": "fridge/fridge.urdf",
    "kitchen_counter": "kitchen_counter/kitchen_counter.urdf",
    "kitchenCupboard_01": "kitchen_cupboards/kitchenCupboard_01.urdf",
    "cabinet": "cabinet/cabinet.urdf",
    "chestOfDrawers_01": "chest_of_drawers/chestOfDrawers_01.urdf",
}
SKIP_PREFIXES = ("door",)  # doors treated as open (no leaf geometry)

BAND_LO, BAND_HI = 0.10, 1.30  # robot height band (m, Y-up)


def load_triangles(path: Path) -> np.ndarray:
    """All world-space triangles of a GLB as [T, 3, 3]."""
    scene = trimesh.load(path, force="scene", process=False)
    mesh = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
    return mesh.vertices[mesh.faces]


def instance_transform(inst: dict) -> np.ndarray:
    t = np.array(inst.get("translation", [0, 0, 0]), dtype=np.float64)
    w, x, y, z = inst.get("rotation", [1, 0, 0, 0])
    m = trimesh.transformations.quaternion_matrix([w, x, y, z])
    m[:3, 3] = t
    s = inst.get("uniform_scale", 1.0)
    m[:3, :3] *= s
    return m


def urdf_triangles(urdf_path: Path) -> np.ndarray:
    """All visual-mesh triangles of a URDF at zero joint angles, in base frame."""
    import xml.etree.ElementTree as ET

    root = ET.parse(urdf_path).getroot()

    def origin_matrix(el) -> np.ndarray:
        o = el.find("origin")
        if o is None:
            return np.eye(4)
        rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
        xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
        m = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")
        m[:3, 3] = xyz
        return m

    # link world transforms via joint chain (zero joint positions)
    link_tf = {}
    joints = root.findall("joint")
    children = {j.find("child").get("link") for j in joints}
    for link in root.findall("link"):
        if link.get("name") not in children:
            link_tf[link.get("name")] = np.eye(4)
    while True:
        progressed = False
        for j in joints:
            p, c = j.find("parent").get("link"), j.find("child").get("link")
            if p in link_tf and c not in link_tf:
                link_tf[c] = link_tf[p] @ origin_matrix(j)
                progressed = True
        if not progressed:
            break

    tris = []
    for link in root.findall("link"):
        tf = link_tf.get(link.get("name"))
        if tf is None:
            continue
        for vis in link.findall("visual"):
            mesh_el = vis.find("geometry/mesh")
            if mesh_el is None:
                continue
            m = tf @ origin_matrix(vis)
            scale = np.array([float(v) for v in mesh_el.get("scale", "1 1 1").replace(",", " ").split()])
            t = load_triangles(urdf_path.parent / mesh_el.get("filename")) * scale
            tris.append(t @ m[:3, :3].T + m[:3, 3])
    return np.concatenate(tris, axis=0)


def scene_triangles(stage_glb: Path, instance_json: Path, urdf_dir: Path) -> np.ndarray:
    tris = [load_triangles(stage_glb)]
    cfg = json.loads(instance_json.read_text())
    for inst in cfg.get("articulated_object_instances", []):
        name = inst["template_name"]
        if name.lower().startswith(SKIP_PREFIXES):
            continue
        rel = URDF_FILES.get(name)
        if rel is None:
            print(f"  ! unknown articulated template {name!r}, skipping")
            continue
        t = urdf_triangles(urdf_dir / rel)
        m = instance_transform(inst)
        t = t @ m[:3, :3].T + m[:3, 3]
        tris.append(t)
    return np.concatenate(tris, axis=0)


def rasterize_band(tris: np.ndarray, cell: float, pad: float = 0.15):
    """Triangles intersecting the height band -> 2D occupancy grid in XZ."""
    ymin = tris[:, :, 1].min(axis=1)
    ymax = tris[:, :, 1].max(axis=1)
    band = tris[(ymax > BAND_LO) & (ymin < BAND_HI)]
    pts2 = band[:, :, [0, 2]]  # XZ projection

    lo = pts2.reshape(-1, 2).min(axis=0) - pad
    hi = pts2.reshape(-1, 2).max(axis=0) + pad
    w = int(np.ceil((hi[0] - lo[0]) / cell))
    h = int(np.ceil((hi[1] - lo[1]) / cell))
    occ = np.zeros((h, w), dtype=np.uint8)

    shift = 4
    pix = np.round((pts2 - lo) / cell * (1 << shift)).astype(np.int32)
    polys = list(pix)  # one [3, 2] int array per triangle
    cv2.fillPoly(occ, polys, 1, lineType=cv2.LINE_8, shift=shift)
    # vertical faces project to zero-area polys; catch them with edges
    cv2.polylines(occ, polys, isClosed=True, color=1, thickness=1, lineType=cv2.LINE_8, shift=shift)
    origin = (lo + cell / 2).astype(np.float32)
    return occ, origin


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
    ap.add_argument("--data", default="data/replica_cad_baked")
    ap.add_argument("--out", default="data/scenes")
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fcfg = FieldConfig()

    stages = sorted((data / "stages_uncompressed").glob("*.glb"))
    for i, stage in enumerate(stages):
        name = stage.stem
        inst = data / "configs" / "scenes" / f"{name}.scene_instance.json"
        print(f"[{i + 1}/{len(stages)}] {name}")
        tris = scene_triangles(stage, inst, data / "urdf_uncompressed")
        print(f"  {len(tris)} triangles")
        occ, origin = rasterize_band(tris, fcfg.cell)
        print(f"  grid {occ.shape}, occupied {occ.mean() * 100:.1f}%")
        scene = build_scene(name, occ, origin, fcfg, seed=i)
        scene.save(out / f"{name}.npz")
        debug_png(scene, out / f"{name}.png")
        nav_area = (scene.edf > fcfg.robot_radius).mean() * occ.size * fcfg.cell**2
        print(f"  navigable area ~{nav_area:.1f} m^2, starts/goal {scene.start_counts.min()}-{scene.start_counts.max()}")

    pack = ScenePack.load_dir(out)
    print(f"\npack: {len(pack.scenes)} scenes, grid {pack.grid_hw}, geo {pack.geo_hw}")


if __name__ == "__main__":
    main()
