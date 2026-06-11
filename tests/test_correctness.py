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


def lidar_np(scene, pos, n_rays, max_range, heading=None):
    """Numpy sphere-tracing reference; rays at 2*pi*r/R + heading (sensor frame)."""
    cell = scene.cell
    eps, minstep = 0.5 * cell, 0.3 * cell
    heading = np.zeros(len(pos)) if heading is None else heading
    out = np.zeros((len(pos), n_rays), dtype=np.float32)
    for i, p in enumerate(pos):
        for r in range(n_rays):
            th = 2 * np.pi * r / n_rays + heading[i]
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


def check_collision_invariant(pack, scene, kinematics):
    """No agent position may penetrate geometry under adversarial random actions."""
    cfg = SimConfig(n_rays=64, kinematics=kinematics)
    sim = Sim(pack, num_envs=256, cfg=cfg, seed=1)
    sim.reset()
    mx.random.seed(7)
    min_edf = np.inf
    for _ in range(500):
        a = mx.random.uniform(low=-3.0, high=3.0, shape=(sim.num_envs, 2))
        sim.step(a)
        p = np.array(sim.pos)
        gx = (p[:, 0] - scene.origin[0]) / scene.cell
        gy = (p[:, 1] - scene.origin[1]) / scene.cell
        d = bilin_np(scene.edf, gx, gy)
        min_edf = min(min_edf, float(d.min()))
    print(f"[{kinematics}] min edf over 500 random steps: {min_edf:.4f} (radius {cfg.robot_radius})")
    assert min_edf > cfg.robot_radius - 0.02, f"{kinematics}: agent penetrated obstacles"


def check_expert(pack, kinematics):
    """Expert reaches goals; timeouts stay rare (contact-terminals reported)."""
    cfg = SimConfig(n_rays=64, kinematics=kinematics)
    sim = Sim(pack, num_envs=512, cfg=cfg, seed=2)
    sim.reset()
    terms, truncs, hits = 0, 0, 0
    for _ in range(1500):
        sim.step(sim.expert_actions())
        terms += int(np.array(sim.term).sum())
        truncs += int(np.array(sim.trunc).sum())
        hits += int(np.array(sim.hit).sum())
    timeouts = truncs - hits
    success = terms / (terms + truncs + 1e-9)
    print(f"[{kinematics}] expert: {terms} reached, {timeouts} timeouts, {hits} contacts "
          f"-> success {success * 100:.2f}%")
    assert terms > 1000 and timeouts / max(terms, 1) < 0.01, f"{kinematics}: expert times out"
    assert success > 0.85, f"{kinematics}: expert success too low"


def check_diffdrive_kinematics(pack, scene):
    """Unit checks of the unicycle model from a forced free-space pose."""
    cfg = SimConfig(n_rays=64, kinematics="diffdrive")
    n = 8
    sim = Sim(pack, num_envs=n, cfg=cfg, seed=1)
    sim.reset()
    iy, ix = np.unravel_index(scene.edf.argmax(), scene.edf.shape)
    free = np.array([scene.origin[0] + ix * scene.cell, scene.origin[1] + iy * scene.cell])
    pos = np.tile(free, (n, 1))
    goal = pos + 3.0
    goal_k = np.zeros(n, np.int32)
    head = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2] * 2)

    # v > 0, w = 0: moves along the body heading, heading unchanged
    sim.set_state(pos, goal, goal_k, heading=head)
    sim.step(mx.array(np.tile([1.0, 0.0], (n, 1)).astype(np.float32)))
    expect = pos + cfg.dt * np.stack([np.cos(head), np.sin(head)], 1)
    err = np.abs(np.array(sim.pos) - expect).max()
    dth = np.abs(np.array(sim.heading) - head).max()
    print(f"[diffdrive] straight-drive err {err:.2e}, heading drift {dth:.2e}")
    assert err < 1e-5 and dth < 1e-6

    # v = 0, w != 0: rotates in place
    sim.set_state(pos, goal, goal_k, heading=head)
    sim.step(mx.array(np.tile([0.0, 1.5], (n, 1)).astype(np.float32)))
    derr = np.abs(np.array(sim.heading) - (head + 1.5 * cfg.dt)).max()
    perr = np.abs(np.array(sim.pos) - pos).max()
    print(f"[diffdrive] turn-in-place: pos err {perr:.2e}, heading err {derr:.2e}")
    assert perr < 1e-5 and derr < 1e-5

    # yaw rate clamps at w_max
    sim.set_state(pos, goal, goal_k, heading=head)
    sim.step(mx.array(np.tile([0.0, 99.0], (n, 1)).astype(np.float32)))
    cerr = np.abs(np.array(sim.heading) - (head + cfg.w_max * cfg.dt)).max()
    print(f"[diffdrive] w_max clamp err {cerr:.2e}")
    assert cerr < 1e-5

    # lidar is body-frame: matches the numpy reference traced at ray + heading
    sim.set_state(pos, goal, goal_k, heading=head)
    ref = lidar_np(scene, pos[:4], cfg.n_rays, cfg.max_range, heading=head[:4])
    err = np.abs(ref - np.array(sim.lidar)[:4]).max()
    print(f"[diffdrive] body-frame lidar max err {err:.5f} m (cell={scene.cell})")
    assert err < 2 * scene.cell, "diffdrive lidar mismatch vs numpy reference"

    # rel_goal observation is rotated into the believed body frame
    obs = np.array(sim.obs())
    rg = obs[:, cfg.n_rays:cfg.n_rays + 2]
    world = goal - pos
    expect = np.stack([np.cos(head) * world[:, 0] + np.sin(head) * world[:, 1],
                       -np.sin(head) * world[:, 0] + np.cos(head) * world[:, 1]], 1)
    err = np.abs(rg - expect).max()
    print(f"[diffdrive] body-frame rel_goal err {err:.2e}")
    assert err < 1e-5


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

    # --- collision invariant + expert, per kinematics ---
    for kin in ("holonomic", "diffdrive", "diffdrive_vel"):
        check_collision_invariant(pack, scene, kin)
        check_expert(pack, kin)

    # --- diffdrive unicycle model unit checks ---
    check_diffdrive_kinematics(pack, scene)

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
