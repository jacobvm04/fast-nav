"""Scene preprocessing: occupancy grid -> distance/geodesic fields -> episode tables.

World frame is 2D (x, y) in meters. Grids are [H, W] indexed [iy, ix] with
world_x = origin_x + ix * cell, world_y = origin_y + iy * cell (cell centers).
For ReplicaCAD (Y-up), world (x, y) here corresponds to mesh (X, Z).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph


@dataclasses.dataclass
class FieldConfig:
    cell: float = 0.025          # occupancy/EDF resolution (m)
    geo_cell: float = 0.05       # geodesic field resolution (m)
    robot_radius: float = 0.18   # m
    start_clearance: float = 0.05  # extra clearance beyond radius for spawn points
    n_goals: int = 16            # geodesic fields per scene
    clearance_pref: float = 0.45   # distance (m) at which wall-proximity penalty fades to 0
    clearance_weight: float = 3.0  # max multiplicative path-cost penalty next to walls
    obstacle_pen_slope: float = 4.0  # geo field growth rate (per m) inside obstacles
    max_starts: int = 4096       # eligible start points stored per goal
    geo_smooth_sigma: float = 1.0  # gaussian blur (in geo cells) applied to geo field


@dataclasses.dataclass
class Scene:
    name: str
    cell: float
    origin: np.ndarray        # [2] world coords of occupancy cell (0,0) center
    occupancy: np.ndarray     # [H, W] uint8, 1 = occupied (geometry in robot height band)
    edf: np.ndarray           # [H, W] float32, signed distance to obstacles (m), <0 inside
    geo_cell: float
    geo_origin: np.ndarray    # [2]
    geo: np.ndarray           # [K, Hg, Wg] float32 cost-to-go (m-ish), repulsive inside obstacles
    goals_xy: np.ndarray      # [K, 2] float32 world coords
    starts_xy: np.ndarray     # [K, M, 2] float32, sorted by geodesic distance to goal
    starts_geo: np.ndarray    # [K, M] float32 geodesic distance of each start (sorted asc)
    start_counts: np.ndarray  # [K] int32 valid entries in starts_xy

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **{f.name: getattr(self, f.name) for f in dataclasses.fields(self)})

    @staticmethod
    def load(path: str | Path) -> "Scene":
        d = np.load(path, allow_pickle=False)
        kw = {k: d[k] for k in d.files}
        kw["name"] = str(kw["name"])
        kw["cell"] = float(kw["cell"])
        kw["geo_cell"] = float(kw["geo_cell"])
        return Scene(**kw)


def signed_edf(occupancy: np.ndarray, cell: float) -> np.ndarray:
    """Signed distance (m): positive in free space, negative inside obstacles."""
    free = occupancy == 0
    d_out = ndi.distance_transform_edt(free, sampling=cell)
    d_in = ndi.distance_transform_edt(~free, sampling=cell)
    return (d_out - d_in).astype(np.float32)


def _downsample_max(occ: np.ndarray, factor: int) -> np.ndarray:
    h, w = occ.shape
    hp, wp = -(-h // factor) * factor, -(-w // factor) * factor
    pad = np.ones((hp, wp), dtype=occ.dtype)  # pad with occupied
    pad[:h, :w] = occ
    return pad.reshape(hp // factor, factor, wp // factor, factor).max(axis=(1, 3))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n = ndi.label(mask)
    if n == 0:
        raise ValueError("no traversable space found")
    sizes = ndi.sum(mask, labels, index=np.arange(1, n + 1))
    return labels == (1 + int(np.argmax(sizes)))


def _grid_graph(trav: np.ndarray, weight: np.ndarray, cell: float) -> sp.csr_matrix:
    """8-connected graph over traversable cells; edge cost = dist * mean(node weights)."""
    h, w = trav.shape
    idx = -np.ones((h, w), dtype=np.int64)
    ys, xs = np.nonzero(trav)
    idx[ys, xs] = np.arange(len(ys))
    rows, cols, vals = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        dist = cell * np.hypot(dy, dx)
        src = trav[max(0, -dy):h - max(0, dy) or h, max(0, -dx):w - max(0, dx) or w]
        # shifted neighbor mask aligned with src
        ys0, xs0 = np.nonzero(src)
        ys0 = ys0 + max(0, -dy)
        xs0 = xs0 + max(0, -dx)
        ys1, xs1 = ys0 + dy, xs0 + dx
        ok = (ys1 >= 0) & (ys1 < h) & (xs1 >= 0) & (xs1 < w)
        ys0, xs0, ys1, xs1 = ys0[ok], xs0[ok], ys1[ok], xs1[ok]
        ok = trav[ys1, xs1]
        ys0, xs0, ys1, xs1 = ys0[ok], xs0[ok], ys1[ok], xs1[ok]
        rows.append(idx[ys0, xs0])
        cols.append(idx[ys1, xs1])
        vals.append(dist * 0.5 * (weight[ys0, xs0] + weight[ys1, xs1]))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    n = len(ys)
    g = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return g + g.T


def _farthest_point_sample(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    chosen = [rng.integers(len(points))]
    d = np.linalg.norm(points - points[chosen[0]], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(points - points[nxt], axis=1))
    return np.array(chosen)


def build_scene(name: str, occupancy: np.ndarray, origin: np.ndarray, cfg: FieldConfig,
                seed: int = 0) -> Scene:
    """From a fine occupancy grid, compute all fields and episode tables."""
    rng = np.random.default_rng(seed)
    occupancy = occupancy.astype(np.uint8)
    edf = signed_edf(occupancy, cfg.cell)

    # --- geodesic resolution grid ---
    factor = max(1, round(cfg.geo_cell / cfg.cell))
    geo_cell = cfg.cell * factor
    occ_g = _downsample_max(occupancy, factor)
    geo_origin = origin + (factor - 1) * cfg.cell / 2.0
    edf_g = signed_edf(occ_g, geo_cell)

    trav = _largest_component(edf_g > cfg.robot_radius)
    hg, wg = trav.shape

    # wall-proximity cost multiplier in [1, 1+clearance_weight]
    prox = np.clip((cfg.clearance_pref - edf_g) / cfg.clearance_pref, 0.0, 1.0)
    weight = 1.0 + cfg.clearance_weight * prox**2

    graph = _grid_graph(trav, weight, geo_cell)
    ys, xs = np.nonzero(trav)
    cells_xy = np.stack([geo_origin[0] + xs * geo_cell, geo_origin[1] + ys * geo_cell], axis=1)

    # goals: spread-out traversable cells with generous clearance
    roomy = edf_g[ys, xs] > cfg.robot_radius + 0.12
    cand = np.nonzero(roomy)[0] if roomy.sum() >= cfg.n_goals else np.arange(len(ys))
    goal_nodes = cand[_farthest_point_sample(cells_xy[cand], cfg.n_goals, rng)]
    goals_xy = cells_xy[goal_nodes].astype(np.float32)

    dist = csgraph.dijkstra(graph, directed=False, indices=goal_nodes)  # [K, n_nodes]

    # --- geo fields: scatter to grid, fill non-traversable with repulsive values ---
    k = cfg.n_goals
    geo = np.full((k, hg, wg), np.inf, dtype=np.float32)
    geo[:, ys, xs] = dist.astype(np.float32)
    reach = np.isfinite(geo[0])  # reachability identical across goals (one component)
    fill_d, (iy, ix) = ndi.distance_transform_edt(~reach, sampling=geo_cell, return_indices=True)
    pen = (fill_d * cfg.obstacle_pen_slope).astype(np.float32)
    geo = geo[:, iy, ix] + pen[None]
    if cfg.geo_smooth_sigma > 0:
        geo = ndi.gaussian_filter(geo, sigma=(0, cfg.geo_smooth_sigma, cfg.geo_smooth_sigma))

    # --- start tables: eligible spawns per goal, sorted by geodesic distance ---
    spawn_ok = edf_g[ys, xs] > cfg.robot_radius + cfg.start_clearance
    m = cfg.max_starts
    starts_xy = np.zeros((k, m, 2), dtype=np.float32)
    starts_geo = np.full((k, m), np.inf, dtype=np.float32)
    start_counts = np.zeros(k, dtype=np.int32)
    for i in range(k):
        ok = spawn_ok & np.isfinite(dist[i]) & (dist[i] > 0.5)
        nodes = np.nonzero(ok)[0]
        if len(nodes) > m:
            nodes = rng.choice(nodes, size=m, replace=False)
        order = np.argsort(dist[i][nodes])
        nodes = nodes[order]
        starts_xy[i, :len(nodes)] = cells_xy[nodes]
        starts_geo[i, :len(nodes)] = dist[i][nodes]
        start_counts[i] = len(nodes)

    return Scene(
        name=name, cell=cfg.cell, origin=origin.astype(np.float32),
        occupancy=occupancy, edf=edf,
        geo_cell=geo_cell, geo_origin=geo_origin.astype(np.float32), geo=geo,
        goals_xy=goals_xy, starts_xy=starts_xy, starts_geo=starts_geo,
        start_counts=start_counts,
    )


def make_synthetic_occupancy(size_m: tuple[float, float] = (12.0, 9.0), cell: float = 0.025,
                             n_obstacles: int = 24, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random cluttered room for testing without real assets."""
    rng = np.random.default_rng(seed)
    w = int(size_m[0] / cell)
    h = int(size_m[1] / cell)
    occ = np.zeros((h, w), dtype=np.uint8)
    t = max(2, int(0.1 / cell))
    occ[:t], occ[-t:], occ[:, :t], occ[:, -t:] = 1, 1, 1, 1
    for _ in range(n_obstacles):
        if rng.random() < 0.6:  # box
            bw = int(rng.uniform(0.3, 1.6) / cell)
            bh = int(rng.uniform(0.3, 1.6) / cell)
            y = rng.integers(t, h - t - bh)
            x = rng.integers(t, w - t - bw)
            occ[y:y + bh, x:x + bw] = 1
        else:  # disk
            r = rng.uniform(0.15, 0.5) / cell
            cy = rng.uniform(t + r, h - t - r)
            cx = rng.uniform(t + r, w - t - r)
            yy, xx = np.ogrid[:h, :w]
            occ[(yy - cy) ** 2 + (xx - cx) ** 2 < r**2] = 1
    origin = np.array([cell / 2, cell / 2], dtype=np.float32)
    return occ, origin


class ScenePack:
    """Scenes stacked into padded arrays ready to upload to the GPU.

    Grids are padded to common [H, W] (occupied / +inf outside), per-scene
    origins kept. All arrays returned as numpy; the sim converts to mx.
    """

    def __init__(self, scenes: list[Scene]):
        if not scenes:
            raise ValueError("need at least one scene")
        c0 = scenes[0]
        for s in scenes:
            if abs(s.cell - c0.cell) > 1e-9 or abs(s.geo_cell - c0.geo_cell) > 1e-9:
                raise ValueError("all scenes must share cell sizes")
            if s.geo.shape[0] != c0.geo.shape[0] or s.starts_xy.shape[1] != c0.starts_xy.shape[1]:
                raise ValueError("all scenes must share n_goals / max_starts")
        self.scenes = scenes
        self.names = [s.name for s in scenes]
        self.cell = c0.cell
        self.geo_cell = c0.geo_cell
        self.n_goals = c0.geo.shape[0]

        h = max(s.edf.shape[0] for s in scenes)
        w = max(s.edf.shape[1] for s in scenes)
        hg = max(s.geo.shape[1] for s in scenes)
        wg = max(s.geo.shape[2] for s in scenes)
        n = len(scenes)

        # pad EDF with deep "inside obstacle" so out-of-bounds reads as solid
        self.edf = np.full((n, h, w), -10.0, dtype=np.float32)
        self.geo = np.full((n, self.n_goals, hg, wg), 1e6, dtype=np.float32)
        self.occupancy = np.ones((n, h, w), dtype=np.uint8)
        self.origin = np.zeros((n, 2), dtype=np.float32)
        self.geo_origin = np.zeros((n, 2), dtype=np.float32)
        self.goals_xy = np.zeros((n, self.n_goals, 2), dtype=np.float32)
        self.starts_xy = np.zeros((n,) + c0.starts_xy.shape, dtype=np.float32)
        self.starts_geo = np.full((n,) + c0.starts_geo.shape, np.inf, dtype=np.float32)
        self.start_counts = np.zeros((n, self.n_goals), dtype=np.int32)
        for i, s in enumerate(scenes):
            sh, sw = s.edf.shape
            self.edf[i, :sh, :sw] = s.edf
            self.occupancy[i, :sh, :sw] = s.occupancy
            gh, gw = s.geo.shape[1:]
            self.geo[i, :, :gh, :gw] = s.geo
            self.origin[i] = s.origin
            self.geo_origin[i] = s.geo_origin
            self.goals_xy[i] = s.goals_xy
            self.starts_xy[i] = s.starts_xy
            self.starts_geo[i] = s.starts_geo
            self.start_counts[i] = s.start_counts
        self.grid_hw = (h, w)
        self.geo_hw = (hg, wg)

    @staticmethod
    def load_dir(path: str | Path) -> "ScenePack":
        files = sorted(Path(path).glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"no scene .npz files in {path}")
        return ScenePack([Scene.load(f) for f in files])

    def start_range_for(self, min_geo: float, max_geo: float) -> np.ndarray:
        """Per (scene, goal) index range [lo, hi) into the sorted start table whose
        geodesic distance lies within [min_geo, max_geo]. int32 [S, K, 2]."""
        s, k, m = self.starts_geo.shape
        lo = np.zeros((s, k), dtype=np.int32)
        hi = np.zeros((s, k), dtype=np.int32)
        for i in range(s):
            for j in range(k):
                g = self.starts_geo[i, j, : self.start_counts[i, j]]
                lo[i, j] = np.searchsorted(g, min_geo, side="left")
                hi[i, j] = np.searchsorted(g, max_geo, side="right")
                if hi[i, j] <= lo[i, j]:  # nothing in range: fall back to whole table
                    lo[i, j] = 0
                    hi[i, j] = max(1, len(g))
        return np.stack([lo, hi], axis=-1).astype(np.int32)
