"""Verify Metal kernels against numpy references on a synthetic scene."""

import mlx.core as mx
import numpy as np

from fastnav.scene import FieldConfig, ScenePack, build_scene, make_synthetic_occupancy
from fastnav.sim import Sim, SimConfig


def bilin_np(f, gx, gy):
    h, w = f.shape
    gx = np.clip(gx, 0.0, w - 1.001)
    gy = np.clip(gy, 0.0, h - 1.001)
    x0 = gx.astype(int)
    y0 = gy.astype(int)
    fx, fy = gx - x0, gy - y0
    v00 = f[y0, x0]
    v01 = f[y0, x0 + 1]
    v10 = f[y0 + 1, x0]
    v11 = f[y0 + 1, x0 + 1]
    return (v00 * (1 - fx) + v01 * fx) * (1 - fy) + (v10 * (1 - fx) + v11 * fx) * fy


def lidar_np(scene, pos, n_rays, max_range):
    cell = scene.cell
    eps, minstep = 0.5 * cell, 0.3 * cell
    out = np.zeros((len(pos), n_rays), dtype=np.float32)
    for i, p in enumerate(pos):
        for r in range(n_rays):
            th = 2 * np.pi * r / n_rays
            d_vec = np.array([np.cos(th), np.sin(th)])
            tt = 0.0
            for _ in range(96):
                q = p + tt * d_vec
                gx = (q[0] - scene.origin[0]) / cell
                gy = (q[1] - scene.origin[1]) / cell
                d = bilin_np(scene.edf, np.array(gx), np.array(gy))
                if d < eps:
                    break
                tt += max(float(d), minstep)
                if tt >= max_range:
                    tt = max_range
                    break
            out[i, r] = min(tt, max_range)
    return out


def main():
    occ, origin = make_synthetic_occupancy(seed=3)
    scene = build_scene("synth0", occ, origin, FieldConfig())
    pack = ScenePack([scene])
    cfg = SimConfig(n_rays=64)
    sim = Sim(pack, num_envs=256, cfg=cfg, seed=1)
    sim.reset()

    # --- lidar matches numpy reference ---
    pos = np.array(sim.pos)
    ref = lidar_np(scene, pos[:32], cfg.n_rays, cfg.max_range)
    got = np.array(sim.lidar)[:32]
    err = np.abs(ref - got)
    print(f"lidar max err {err.max():.5f} m, mean {err.mean():.6f} m (cell={scene.cell})")
    assert err.max() < 2 * scene.cell, "lidar mismatch vs numpy reference"

    # --- collision invariant under adversarial random actions ---
    mx.random.seed(7)
    min_edf = np.inf
    for _ in range(500):
        a = mx.random.uniform(low=-2.0, high=2.0, shape=(sim.num_envs, 2))
        sim.step(a)
        p = np.array(sim.pos)
        gx = (p[:, 0] - scene.origin[0]) / scene.cell
        gy = (p[:, 1] - scene.origin[1]) / scene.cell
        d = bilin_np(scene.edf, gx, gy)
        min_edf = min(min_edf, float(d.min()))
    print(f"min edf at agent position over 500 random steps: {min_edf:.4f} (radius {cfg.robot_radius})")
    assert min_edf > cfg.robot_radius - 0.02, "agent penetrated obstacles"

    # --- expert reaches goals, episodes recycle ---
    sim2 = Sim(pack, num_envs=512, cfg=cfg, seed=2)
    sim2.reset()
    terms, truncs = 0, 0
    for _ in range(1500):
        sim2.step(sim2.expert_actions())
        terms += int(np.array(sim2.term).sum())
        truncs += int(np.array(sim2.trunc).sum())
    print(f"expert: {terms} episodes reached, {truncs} timeouts -> success {terms/(terms+truncs+1e-9)*100:.2f}%")
    assert terms > 1000 and truncs / max(terms, 1) < 0.01

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
