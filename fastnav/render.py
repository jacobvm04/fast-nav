"""Top-down mosaic rendering of a subset of envs (viz only, sim-rate independent)."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from fastnav.scene import ScenePack
from fastnav.sim import Sim


class MosaicRenderer:
    def __init__(self, sim: Sim, env_ids: list[int], cols: int = 6, tile_h: int = 290,
                 tile_w: int | None = None):
        self.sim = sim
        self.env_ids = env_ids
        self.cols = cols
        pack = sim.pack
        self.tile_h = tile_h
        self.tile_w = tile_w or tile_h
        self.origin = pack.origin
        self.cell = pack.cell
        # scenes share a padded grid; crop each to its own free-space extent so
        # small scenes fill their tile instead of floating in padding
        self.bgs, self.scales, self.crop0, self.paste0 = [], [], [], []
        for occ in pack.occupancy:
            ys, xs = np.nonzero(occ == 0)
            m = 4
            y0, y1 = max(ys.min() - m, 0), min(ys.max() + m, occ.shape[0])
            x0, x1 = max(xs.min() - m, 0), min(xs.max() + m, occ.shape[1])
            crop = occ[y0:y1, x0:x1]
            scale = min(self.tile_w / crop.shape[1], self.tile_h / crop.shape[0])
            w_px = max(int(crop.shape[1] * scale), 1)
            h_px = max(int(crop.shape[0] * scale), 1)
            sub = np.full((*crop.shape, 3), 245, dtype=np.uint8)
            sub[crop > 0] = (60, 50, 45)
            sub = cv2.resize(sub, (w_px, h_px), interpolation=cv2.INTER_AREA)
            tile = np.full((self.tile_h, self.tile_w, 3), 235, dtype=np.uint8)
            py = (self.tile_h - h_px) // 2
            px = (self.tile_w - w_px) // 2
            tile[py:py + h_px, px:px + w_px] = sub
            self.bgs.append(tile)
            self.scales.append(scale)
            self.crop0.append((x0, y0))
            self.paste0.append((px, py))
        self.trails: dict[int, deque] = {i: deque(maxlen=50) for i in env_ids}
        r = sim.cfg.n_rays
        self.ray_dirs = np.stack([np.cos(2 * np.pi * np.arange(r) / r),
                                  np.sin(2 * np.pi * np.arange(r) / r)], axis=1).astype(np.float32)

    def _to_px(self, s: int, xy: np.ndarray) -> np.ndarray:
        g = (xy - self.origin[s]) / self.cell
        g = g - np.array(self.crop0[s])
        return (g * self.scales[s] + np.array(self.paste0[s])).astype(np.int32)

    def ego_tile(self, lidar_row: np.ndarray, rel_goal: np.ndarray, size: int | None = None) -> np.ndarray:
        """What the policy sees: lidar ranges + relative goal, agent-centered.

        World-aligned (the obs carries no orientation), so it translates with the
        agent but never rotates."""
        size = size or self.tile_h
        max_range = self.sim.cfg.max_range
        c = size // 2
        scale = (c - 8) / max_range
        img = np.full((size, size, 3), 250, dtype=np.uint8)
        for ring in range(1, int(max_range) + 1):
            cv2.circle(img, (c, c), int(ring * scale), (228, 228, 228), 1, cv2.LINE_AA)
        pts = (lidar_row[:, None] * self.ray_dirs * scale + c).astype(np.int32)
        cv2.fillPoly(img, [pts], (255, 245, 235))
        cv2.polylines(img, [pts], True, (200, 150, 90), 1, cv2.LINE_AA)
        hit = lidar_row < max_range - 1e-3
        for p, h in zip(pts, hit):
            cv2.circle(img, p, 2, (160, 90, 30) if h else (220, 200, 180), -1, cv2.LINE_AA)
        gd = np.linalg.norm(rel_goal)
        gdir = rel_goal / max(gd, 1e-6)
        gclip = gdir * min(gd, max_range) * scale + c
        cv2.arrowedLine(img, (c, c), gclip.astype(np.int32), (50, 50, 230), 2,
                        cv2.LINE_AA, tipLength=0.12)
        if gd > max_range:  # goal beyond lidar horizon: hollow marker at the edge
            cv2.circle(img, gclip.astype(np.int32), 6, (50, 50, 230), 2, cv2.LINE_AA)
        cv2.circle(img, (c, c), max(3, int(self.sim.cfg.robot_radius * scale)), (60, 160, 30), -1)
        cv2.putText(img, f"goal {gd:.1f}m", (8, size - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (size - 1, size - 1), (180, 180, 180), 1)
        return img

    def frame(self, pos: np.ndarray, goal: np.ndarray, lidar: np.ndarray, scene: np.ndarray,
              ego: bool = False, highlight: np.ndarray | None = None,
              waypoints: np.ndarray | None = None) -> np.ndarray:
        """waypoints [N, H, 2]: the policy's sampled trajectory plan in WORLD
        coords (None = no plan, e.g. non-trajectory heads); drawn as a magenta
        polyline so flow-matching plans are visible per frame."""
        tiles = []
        for i in self.env_ids:
            s = int(scene[i])
            img = self.bgs[s].copy()
            p = pos[i]
            self.trails[i].append(p.copy())
            hits = p[None, :] + lidar[i][:, None] * self.ray_dirs
            ppx = self._to_px(s, p)
            for hpx in self._to_px(s, hits):
                cv2.line(img, ppx, hpx, (235, 215, 170), 1, cv2.LINE_AA)
            for j, tp in enumerate(self.trails[i]):
                a = j / len(self.trails[i])
                cv2.circle(img, self._to_px(s, tp), 1, (140 + int(60 * a), 190, 140), -1)
            if waypoints is not None:
                wp_px = self._to_px(s, np.concatenate([p[None], waypoints[i]], axis=0))
                cv2.polylines(img, [wp_px], False, (220, 60, 200), 2, cv2.LINE_AA)
                for wpx in wp_px[1:]:
                    cv2.circle(img, wpx, 3, (220, 60, 200), -1, cv2.LINE_AA)
            cv2.circle(img, self._to_px(s, goal[i]), 5, (50, 50, 230), -1)
            cv2.circle(img, ppx, max(2, int(self.sim.cfg.robot_radius / self.cell * self.scales[s])),
                       (60, 160, 30), -1)
            if highlight is not None and highlight[i]:
                cv2.rectangle(img, (0, 0), (self.tile_w - 1, self.tile_h - 1), (60, 60, 220), 3)
            else:
                cv2.rectangle(img, (0, 0), (self.tile_w - 1, self.tile_h - 1), (180, 180, 180), 1)
            if ego:
                img = np.concatenate([img, self.ego_tile(lidar[i], goal[i] - pos[i])], axis=1)
            tiles.append(img)
        rows = []
        for r0 in range(0, len(tiles), self.cols):
            row = tiles[r0:r0 + self.cols]
            while len(row) < self.cols:
                row.append(np.zeros_like(tiles[0]))
            rows.append(np.concatenate(row, axis=1))
        return np.concatenate(rows, axis=0)
