"""Compose Habitat scene_instance.json scenes into triangle soups (no habitat-sim).

Handles the scheme shared by ai2thor-hab / hssd-hab / ReplicaCAD: a stage GLB
plus object instances placed by translation + wxyz quaternion + (non_)uniform
scale, with template names resolved through *.object_config.json indirection.
Also rasterizes a robot-height band to a 2D occupancy grid (world (x,y) = mesh
(X,Z), height = Y-up).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import trimesh

BAND_LO, BAND_HI = 0.10, 1.30


@lru_cache(maxsize=8192)
def load_triangles(path: str) -> np.ndarray:
    """All world-space triangles of a mesh file as [T, 3, 3] (cached per asset)."""
    scene = trimesh.load(path, force="scene", process=False)
    mesh = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
    return mesh.vertices[mesh.faces]


def instance_transform(inst: dict) -> np.ndarray:
    t = np.array(inst.get("translation", [0, 0, 0]), dtype=np.float64)
    w, x, y, z = inst.get("rotation", [1, 0, 0, 0])
    m = trimesh.transformations.quaternion_matrix([w, x, y, z])
    m[:3, 3] = t
    if "non_uniform_scale" in inst:
        m[:3, :3] = m[:3, :3] @ np.diag(inst["non_uniform_scale"])
    else:
        m[:3, :3] *= inst.get("uniform_scale", 1.0)
    return m


_WORLD_FRONT = np.array([0.0, 0.0, -1.0])
_WORLD_UP = np.array([0.0, 1.0, 0.0])


def _orientation_correction(cfg: dict) -> np.ndarray:
    """Habitat re-orients assets so their declared up/front match the world frame."""
    up = np.array(cfg.get("up", [0, 1, 0]), dtype=np.float64)
    front = np.array(cfg.get("front", [0, 0, -1]), dtype=np.float64)

    def basis(f, u):
        return np.stack([f, u, np.cross(f, u)], axis=1)

    return basis(_WORLD_FRONT, _WORLD_UP) @ np.linalg.inv(basis(front, up))


def _resolve_render_asset(root: Path, template_name: str, kind: str) -> tuple[Path, np.ndarray]:
    """template 'objects/Box_30' or 'stages/ProcTHOR/1/X' -> (mesh path, 3x3 asset-frame
    correction: orientation from config up/front, plus optional config scale)."""
    name = template_name.split("/")[-1]
    sub = "/".join(template_name.split("/")[1:-1])  # e.g. 'ProcTHOR/1' for stages
    cfg_dir = root / "configs" / kind / sub if sub else root / "configs" / kind
    cfg_path = cfg_dir / f"{name}.{ 'stage_config' if kind == 'stages' else 'object_config'}.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        corr = _orientation_correction(cfg)
        if "scale" in cfg:
            corr = corr @ np.diag(cfg["scale"])
        ra = cfg.get("render_asset")
        if ra:
            return (cfg_path.parent / ra).resolve(), corr
        return root / "assets" / kind / (f"{sub}/" if sub else "") / f"{name}.glb", corr
    return root / "assets" / kind / (f"{sub}/" if sub else "") / f"{name}.glb", np.eye(3)


def compose_scene(root: Path, scene_instance_json: Path,
                  skip_names: tuple[str, ...] = ()) -> np.ndarray:
    """Triangle soup [T, 3, 3] for a habitat scene instance.

    skip_names: lowercase substrings of object template names to omit
    (e.g. ('door',) to treat doors as open).
    """
    cfg = json.loads(scene_instance_json.read_text())
    tris = []

    stage = cfg["stage_instance"]
    stage_path, corr = _resolve_render_asset(root, stage["template_name"], "stages")
    t = load_triangles(str(stage_path)) @ corr.T
    m = instance_transform(stage)
    tris.append(t @ m[:3, :3].T + m[:3, 3])

    missing = 0
    skipped: dict[str, int] = {}
    for inst in cfg.get("object_instances", []):
        name = inst["template_name"].split("/")[-1]
        low = name.lower()
        if any(s in low for s in skip_names):
            skipped[name] = skipped.get(name, 0) + 1
            continue
        path, corr = _resolve_render_asset(root, inst["template_name"], "objects")
        if not path.exists():
            missing += 1
            continue
        t = load_triangles(str(path)) @ corr.T
        m = instance_transform(inst)
        tris.append(t @ m[:3, :3].T + m[:3, 3])
    if missing:
        print(f"  ! {missing} object assets missing, skipped")
    if skipped:
        print(f"  skipped by name: {dict(list(skipped.items())[:6])}")
    return np.concatenate(tris, axis=0)


def rasterize_band(tris: np.ndarray, cell: float, pad: float = 0.15,
                   band: tuple[float, float] | None = None):
    """Triangles intersecting the height band -> 2D occupancy grid in XZ."""
    lo_y, hi_y = band or (BAND_LO, BAND_HI)
    ymin = tris[:, :, 1].min(axis=1)
    ymax = tris[:, :, 1].max(axis=1)
    sel = tris[(ymax > lo_y) & (ymin < hi_y)]
    pts2 = sel[:, :, [0, 2]]

    lo = pts2.reshape(-1, 2).min(axis=0) - pad
    hi = pts2.reshape(-1, 2).max(axis=0) + pad
    w = int(np.ceil((hi[0] - lo[0]) / cell))
    h = int(np.ceil((hi[1] - lo[1]) / cell))
    occ = np.zeros((h, w), dtype=np.uint8)

    shift = 4
    pix = np.round((pts2 - lo) / cell * (1 << shift)).astype(np.int32)
    polys = list(pix)
    cv2.fillPoly(occ, polys, 1, lineType=cv2.LINE_8, shift=shift)
    cv2.polylines(occ, polys, isClosed=True, color=1, thickness=1, lineType=cv2.LINE_8, shift=shift)
    origin = (lo + cell / 2).astype(np.float32)
    return occ, origin
