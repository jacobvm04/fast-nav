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


def _free_pose(scene, n):
    """n copies of the most open point in the scene + the 4 cardinal headings."""
    iy, ix = np.unravel_index(scene.edf.argmax(), scene.edf.shape)
    free = np.array([scene.origin[0] + ix * scene.cell, scene.origin[1] + iy * scene.cell])
    pos = np.tile(free, (n, 1))
    head = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2] * (n // 4))
    return pos, head


def check_lidar_lever_arm(pack, scene):
    """With a mount offset, rays originate at pose + R(heading) * (offset, 0)."""
    off = 0.1
    cfg = SimConfig(n_rays=64, kinematics="diffdrive", lidar_offset=off)
    n = 8
    sim = Sim(pack, num_envs=n, cfg=cfg, seed=1)
    sim.reset()
    pos, head = _free_pose(scene, n)
    sim.set_state(pos, pos + 3.0, np.zeros(n, np.int32), heading=head)
    mount = pos + off * np.stack([np.cos(head), np.sin(head)], 1)
    ref = lidar_np(scene, mount, cfg.n_rays, cfg.max_range, heading=head)
    err = np.abs(ref - np.array(sim.lidar)).max()
    print(f"[lever-arm] lidar-from-mount max err {err:.5f} m (cell={scene.cell})")
    assert err < 2 * scene.cell, "lever-arm lidar mismatch vs numpy reference"


def check_act_latency(pack, scene):
    """Commands execute exactly act_delay steps late; fresh episodes start at rest."""
    cfg = SimConfig(n_rays=64, kinematics="holonomic", act_latency=2)
    n = 64
    sim = Sim(pack, num_envs=n, cfg=cfg, seed=1)
    sim.reset()
    pos, _ = _free_pose(scene, n)
    sim.set_state(pos, pos + 3.0, np.zeros(n, np.int32))
    delay = np.array(sim.act_delay)
    assert delay.min() == 0 and delay.max() == cfg.act_latency, "delay not spanning U{0..max}"
    a = mx.array(np.tile([0.5, 0.0], (n, 1)).astype(np.float32))
    moved_at = np.full(n, -1)
    for t in range(cfg.act_latency + 2):
        sim.step(a)
        m = np.abs(np.array(sim.pos) - pos).max(axis=1) > 1e-6
        moved_at[(moved_at < 0) & m] = t
    assert np.array_equal(moved_at, delay), "first motion step != per-env command delay"
    print(f"[latency] first motion matches per-env delay; counts {np.bincount(delay)} at 0/1/2")


def check_expert_waypoints(pack, kinematics):
    """The future kernel must reproduce live expert stepping: waypoints ==
    the positions an expert-driven noise-free sim visits over the next
    horizon*stride steps, rotated into the start sensor frame. Envs whose
    episode terminates inside the window are excluded (the kernel integrates
    through the goal; the live sim auto-resets)."""
    h, s = 6, 2
    cfg = SimConfig(n_rays=32, kinematics=kinematics)
    n = 128
    sim = Sim(pack, num_envs=n, cfg=cfg, seed=3)
    sim.reset()
    for _ in range(25):  # varied mid-episode states (turns, wall-adjacent)
        sim.step(sim.expert_actions())
    wp = np.array(sim.expert_waypoints(h, s)).reshape(n, h, 2)
    pose0 = np.array(sim.pose).astype(np.float64)
    alive = np.ones(n, bool)
    ref = np.zeros((n, h, 2))
    for t in range(h * s):
        _, term, trunc = sim.step(sim.expert_actions())
        alive &= ~(np.maximum(np.array(term), np.array(trunc)) > 0)
        if (t + 1) % s == 0:
            p = np.array(sim.pose).astype(np.float64)
            dx, dy = p[:, 0] - pose0[:, 0], p[:, 1] - pose0[:, 1]
            c0, s0 = np.cos(pose0[:, 2]), np.sin(pose0[:, 2])
            k = (t + 1) // s - 1
            ref[:, k, 0] = c0 * dx + s0 * dy
            ref[:, k, 1] = -s0 * dx + c0 * dy
    assert alive.sum() > n // 2, "too few full-window episodes to compare"
    err = np.abs(wp[alive] - ref[alive]).max()
    print(f"[{kinematics}] expert_waypoints vs live replay: max err {err:.2e} "
          f"({int(alive.sum())}/{n} envs)")
    assert err < 1e-4, f"{kinematics}: future kernel diverges from live expert stepping"


def check_no_odometry(pack):
    """cfg.odometry=False: the believed pose holds its episode-start anchor (so
    rel_goal/pos observations are per-episode constants), while diffdrive
    dynamics and lidar -- which use the true heading -- are bit-identical to
    the odometry run under the same seeds and the full noise stack."""
    from fastnav.sim import noisy_config

    def rollout(odometry: bool):
        cfg = noisy_config(SimConfig(n_rays=32, kinematics="diffdrive", odometry=odometry), 1.5)
        sim = Sim(pack, num_envs=64, cfg=cfg, seed=11)
        sim.reset()
        mx.random.seed(5)
        traj = []
        for _ in range(200):
            a = mx.random.uniform(low=-2.0, high=2.0, shape=(64, 2))
            obs, term, trunc = sim.step(a)
            done = np.maximum(np.array(term), np.array(trunc))
            traj.append((np.array(sim.pose), np.array(sim.lidar), np.array(sim.odom),
                         np.array(obs), done))
        return traj

    ref, noo = rollout(True), rollout(False)
    for (pose_r, lidar_r, *_), (pose_n, lidar_n, *_) in zip(ref, noo):
        assert np.array_equal(pose_r, pose_n), "odometry flag leaked into dynamics"
        assert np.array_equal(lidar_r, lidar_n), "odometry flag leaked into lidar"
    moved = np.abs(np.diff([p[:, :2] for p, *_ in noo], axis=0)).max()
    assert moved > 0.01, "robot did not move"
    for (_, _, od0, ob0, _), (_, _, od1, ob1, done1) in zip(noo, noo[1:]):
        # auto-reset is fused into the step that reports done, so done1 envs
        # already carry the next episode's anchor at the second sample
        live = done1 == 0
        assert np.array_equal(od0[live], od1[live]), "believed pose drifted without odometry"
        assert np.array_equal(ob0[live, 32:], ob1[live, 32:]), \
            "rel_goal/pos observation changed mid-episode without odometry"
    print("[no-odometry] dynamics/lidar bit-identical; believed pose pinned per episode")


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
        check_expert_waypoints(pack, kin)

    # --- diffdrive unicycle model unit checks ---
    check_diffdrive_kinematics(pack, scene)

    # --- sim2real: lidar lever arm + command latency ---
    check_lidar_lever_arm(pack, scene)
    check_act_latency(pack, scene)

    # --- odometry ablation (cfg.odometry=False) ---
    check_no_odometry(pack)

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
