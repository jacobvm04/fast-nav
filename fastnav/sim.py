"""Batched 2D-lidar navigation sim on Apple GPU via MLX custom Metal kernels.

Three kernels per step, all operating on the whole env batch:
  step_kernel   [N threads]    twist integration, EDF collision projection,
                               termination, fused auto-reset (start/goal resample)
  lidar_kernel  [N*R threads]  sphere tracing through the signed EDF
  expert_kernel [N threads]    gradient descent on precomputed geodesic fields

The kernel bodies are kinematics-agnostic: everything drive-type-specific
(action clamp + actuation noise, command/sensor frame, episode-start heading,
expert command conversion) is an inline `kin_*` function supplied by
fastnav.kinematics and compiled into a per-kinematics kernel set. With the
holonomic kinematics and a noise-free config this reproduces the original
holonomic sim bit-exactly.

Throughput comes from batch size: Python/dispatch overhead is paid once per
batched step, so ~8k envs at ~150 batched steps/s > 1M env steps/s.
"""

from __future__ import annotations

import dataclasses
import functools

import mlx.core as mx
import numpy as np

from fastnav import kinematics
from fastnav.scene import ScenePack

_HEADER = """
inline float bilin(const device float* f, long base, int H, int W, float gx, float gy) {
    gx = metal::clamp(gx, 0.0f, (float)W - 1.001f);
    gy = metal::clamp(gy, 0.0f, (float)H - 1.001f);
    int x0 = (int)gx;
    int y0 = (int)gy;
    float fx = gx - (float)x0;
    float fy = gy - (float)y0;
    long i00 = base + (long)y0 * W + x0;
    float v00 = f[i00];
    float v01 = f[i00 + 1];
    float v10 = f[i00 + W];
    float v11 = f[i00 + W + 1];
    return metal::mix(metal::mix(v00, v01, fx), metal::mix(v10, v11, fx), fy);
}
"""

# p_i: [H, W, K, M, max_steps, N, R]
# p_f: [dt, vmax, radius, goal_radius, cell, inv_cell, max_range, ..., wmax(17)]
_STEP_SRC = """
    uint i = thread_position_in_grid.x;
    int N = p_i[5];
    if (i >= (uint)N) return;
    int H = p_i[0], W = p_i[1], K = p_i[2], M = p_i[3], max_steps = p_i[4];
    float dt = p_f[0], vmax = p_f[1], radius = p_f[2], goal_r = p_f[3];
    float cell = p_f[4], inv_cell = p_f[5];

    int s = scene[i];
    long ebase = (long)s * H * W;
    float ox = origin[s * 2], oy = origin[s * 2 + 1];
    float px = pose[i * 3], py = pose[i * 3 + 1], th = pose[i * 3 + 2];
    float gx = goal[i * 2], gy = goal[i * 2 + 1];
    int gk = goal_k[i];
    int st = step_ct[i];

    bool freset = force_reset[0] > 0.5f;
    bool reached = false;
    bool trunc = false;
    bool hit = false;
    float dist_pre = 0.0f;

    float odx = odom_st[i * 3], ody = odom_st[i * 3 + 1], oth = odom_st[i * 3 + 2];
    float ep0 = ep_n[i * 5], ep1 = ep_n[i * 5 + 1], ep2 = ep_n[i * 5 + 2];
    float ep3 = ep_n[i * 5 + 3], ep4 = ep_n[i * 5 + 4];

    if (!freset) {
        // clamp + actuation noise -> executed twist in the command frame;
        // the frame's true world orientation distorts/steers what executes
        float e0, e1, wz;
        kin_execute(vel[i * 2], vel[i * 2 + 1], vmax, p_f[17], 1.0f + ep4, p_f[12],
                    rnd_n[i * 10], rnd_n[i * 10 + 1], e0, e1, wz);
        float f0 = kin_frame(th, oth);
        float ct = metal::cos(f0), sn = metal::sin(f0);
        float px0 = px, py0 = py;
        float thq = f0;
        const int SUB = 2;
        for (int sub = 0; sub < SUB; sub++) {
            thq += wz * dt / SUB;  // rotate, then translate along the new heading
            float cq = metal::cos(thq), sq = metal::sin(thq);
            float vx = cq * e0 - sq * e1;
            float vy = sq * e0 + cq * e1;
            float sx = px, sy = py;
            px += vx * dt / SUB;
            py += vy * dt / SUB;
            float d = 0.0f;
            for (int it = 0; it < 5; it++) {
                float cgx = (px - ox) * inv_cell;
                float cgy = (py - oy) * inv_cell;
                d = bilin(edf, ebase, H, W, cgx, cgy);
                if (d >= radius || it == 4) break;  // last pass only re-checks, no push
                float dxp = bilin(edf, ebase, H, W, cgx + 1.0f, cgy) - bilin(edf, ebase, H, W, cgx - 1.0f, cgy);
                float dyp = bilin(edf, ebase, H, W, cgx, cgy + 1.0f) - bilin(edf, ebase, H, W, cgx, cgy - 1.0f);
                float gl = metal::sqrt(dxp * dxp + dyp * dyp);
                if (gl < 1e-6f) break;
                float push = (radius - d) + 0.25f * cell;
                px += dxp / gl * push;
                py += dyp / gl * push;
            }
            if (d < radius) { px = sx; py = sy; }  // projection failed: stay put (start was valid)
        }
        float dth = wz * dt;
        th += dth;  // rotation is never blocked by contact (disk robot)
        // integrate odometry: measured displacement = R(-frame) * true, plus
        // scale error, per-episode bias, and distance-scaled random walk;
        // heading drift also grows with rotation (0.5 m-per-rad equivalence)
        float tdx = px - px0, tdy = py - py0;
        float dl = metal::sqrt(tdx * tdx + tdy * tdy);
        float mdx = ct * tdx + sn * tdy;
        float mdy = -sn * tdx + ct * tdy;
        float oscale = 1.0f + ep2;
        float derr = dl + 0.5f * metal::abs(dth);
        odx += mdx * oscale + (ep0 + p_f[7] * rnd_n[i * 10 + 2]) * dl;
        ody += mdy * oscale + (ep1 + p_f[7] * rnd_n[i * 10 + 3]) * dl;
        oth += dth;
        oth += (ep3 + p_f[10] * rnd_n[i * 10 + 4]) * derr;

        // contact check: touching geometry ends the episode as a failure
        float cnow = bilin(edf, ebase, H, W, (px - ox) * inv_cell, (py - oy) * inv_cell) - radius;
        hit = (p_f[16] > 0.0f) && (cnow < p_f[16]);

        float ddx = gx - px, ddy = gy - py;
        dist_pre = metal::sqrt(ddx * ddx + ddy * ddy);
        reached = (dist_pre < goal_r) && !hit;
        st += 1;
        trunc = (st >= max_steps) && !reached && !hit;
    }

    bool done = freset || reached || trunc || hit;
    if (done) {
        float u0 = rnd[i * KIN_NU], u1 = rnd[i * KIN_NU + 1];
        float u2 = rnd[i * KIN_NU + 2], u3 = rnd[i * KIN_NU + 3];
        int nk = metal::min(K - 1, (int)(u0 * (float)K));
        int cnt = metal::max(pool_cnt[(long)s * K + nk], 1);
        int ii = metal::min(cnt - 1, (int)(u1 * (float)cnt));
        int si = pool[((long)s * K + nk) * M + ii];
        long sb = (((long)s * K + nk) * M + si) * 2;
        px = starts[sb] + (u2 - 0.5f) * 0.04f;
        py = starts[sb + 1] + (u3 - 0.5f) * 0.04f;
        gx = goals_all[((long)s * K + nk) * 2];
        gy = goals_all[((long)s * K + nk) * 2 + 1];
        gk = nk;
        st = 0;
        // re-anchor odometry to the new episode start, resample per-episode errors
        odx = px; ody = py;
        float u4 = (KIN_NU > 4) ? rnd[i * KIN_NU + 4] : 0.0f;
        kin_reset(u4, th, oth);
        ep0 = p_f[8] * rnd_n[i * 10 + 5];
        ep1 = p_f[8] * rnd_n[i * 10 + 6];
        ep2 = p_f[9] * rnd_n[i * 10 + 7];
        ep3 = p_f[11] * rnd_n[i * 10 + 8];
        ep4 = p_f[13] * rnd_n[i * 10 + 9];
    }

    float fgx = (px - ox) * inv_cell, fgy = (py - oy) * inv_cell;
    clear_out[i] = bilin(edf, ebase, H, W, fgx, fgy) - radius;

    odom_out[i * 3] = odx;
    odom_out[i * 3 + 1] = ody;
    odom_out[i * 3 + 2] = oth;
    ep_out[i * 5] = ep0;
    ep_out[i * 5 + 1] = ep1;
    ep_out[i * 5 + 2] = ep2;
    ep_out[i * 5 + 3] = ep3;
    ep_out[i * 5 + 4] = ep4;

    pose_out[i * 3] = px;
    pose_out[i * 3 + 1] = py;
    pose_out[i * 3 + 2] = th;
    goal_out[i * 2] = gx;
    goal_out[i * 2 + 1] = gy;
    goal_k_out[i] = gk;
    step_out[i] = st;
    term_out[i] = reached ? (uint8_t)1 : (uint8_t)0;
    trunc_out[i] = (trunc || hit) ? (uint8_t)1 : (uint8_t)0;  // any non-success terminal
    hit_out[i] = hit ? (uint8_t)1 : (uint8_t)0;
    dist_out[i] = dist_pre;
"""

_LIDAR_SRC = """
    uint t = thread_position_in_grid.x;
    int N = p_i[5], R = p_i[6];
    if (t >= (uint)(N * R)) return;
    int i = t / R, r = t % R;
    int H = p_i[0], W = p_i[1];
    float cell = p_f[4], inv_cell = p_f[5], max_range = p_f[6];

    int s = scene[i];
    long ebase = (long)s * H * W;
    float ox = origin[s * 2], oy = origin[s * 2 + 1];
    // rays are indexed in the sensor frame; its true world orientation
    // (heading error / body heading) rotates them in the true frame
    float theta = 6.283185307f * (float)r / (float)R + kin_frame(pose[i * 3 + 2], odom_st[i * 3 + 2]);
    float dx = metal::cos(theta), dy = metal::sin(theta);
    float px = pose[i * 3], py = pose[i * 3 + 1];

    float eps = 0.5f * cell;
    float minstep = 0.3f * cell;
    float tt = 0.0f;
    for (int it = 0; it < 96; it++) {
        float gx = (px + tt * dx - ox) * inv_cell;
        float gy = (py + tt * dy - oy) * inv_cell;
        float d = bilin(edf, ebase, H, W, gx, gy);
        if (d < eps) break;
        tt += metal::max(d, minstep);
        if (tt >= max_range) { tt = max_range; break; }
    }
    tt = tt + p_f[14] * nse[(long)i * R + r];
    if (unif[(long)i * R + r] < p_f[15]) tt = max_range;
    lidar[(long)i * R + r] = metal::clamp(tt, 0.0f, max_range);
"""

# pe_i: [Hg, Wg, K, N]
# pe_f: [geo_cell, inv_geo_cell, vmax, slow_radius, beta, blend_radius, wmax, turn_gain]
_EXPERT_SRC = """
    uint i = thread_position_in_grid.x;
    int N = pe_i[3];
    if (i >= (uint)N) return;
    int Hg = pe_i[0], Wg = pe_i[1], K = pe_i[2];
    float inv_gc = pe_f[1], vmax = pe_f[2], slow_r = pe_f[3], beta = pe_f[4], blend_r = pe_f[5];

    int s = scene[i];
    int k = goal_k[i];
    long base = ((long)s * K + k) * Hg * Wg;
    float ox = geo_origin[s * 2], oy = geo_origin[s * 2 + 1];
    float px = pose[i * 3], py = pose[i * 3 + 1];
    float gx = (px - ox) * inv_gc;
    float gy = (py - oy) * inv_gc;

    float dgx = bilin(geo, base, Hg, Wg, gx + 1.0f, gy) - bilin(geo, base, Hg, Wg, gx - 1.0f, gy);
    float dgy = bilin(geo, base, Hg, Wg, gx, gy + 1.0f) - bilin(geo, base, Hg, Wg, gx, gy - 1.0f);
    float gl = metal::sqrt(dgx * dgx + dgy * dgy);
    val_out[i] = bilin(geo, base, Hg, Wg, gx, gy);

    float tx = goal[i * 2] - px, ty = goal[i * 2 + 1] - py;
    float dist = metal::sqrt(tx * tx + ty * ty);
    float invd = 1.0f / metal::max(dist, 1e-6f);
    float dirx, diry;
    if (gl > 1e-9f) {
        dirx = -dgx / gl;
        diry = -dgy / gl;
    } else {
        dirx = tx * invd;
        diry = ty * invd;
    }
    float b = metal::clamp(1.0f - dist / blend_r, 0.0f, 1.0f);
    dirx = metal::mix(dirx, tx * invd, b);
    diry = metal::mix(diry, ty * invd, b);
    float dl = metal::max(metal::sqrt(dirx * dirx + diry * diry), 1e-6f);
    float speed = vmax * metal::clamp(dist / slow_r, 0.0f, 1.0f);
    float ax = dirx / dl * speed;
    float ay = diry / dl * speed;
    dir_out[i * 2] = metal::atan2(diry, dirx);  // desired world angle (rad)
    dir_out[i * 2 + 1] = speed;                 // desired speed (m/s)
    // shared planner output (desired world velocity) -> drive command
    float a0, a1;
    kin_expert(ax, ay, speed, pose[i * 3 + 2], beta, pe_f[6], pe_f[7],
               prev[i * 2], prev[i * 2 + 1], a0, a1);
    act[i * 2] = a0;
    act[i * 2 + 1] = a1;
"""

@functools.lru_cache(maxsize=None)
def _kernels(kin_name: str):
    """(step, lidar, expert) kernel set with the kinematics' inline functions
    compiled in. Cached: one compilation per kinematics per process."""
    header = _HEADER + kinematics.get(kin_name).metal
    step = mx.fast.metal_kernel(
        name=f"nav_step_{kin_name}",
        input_names=["pose", "vel", "goal", "goal_k", "step_ct", "scene", "edf", "origin",
                     "starts", "goals_all", "pool", "pool_cnt", "odom_st", "ep_n", "rnd",
                     "rnd_n", "force_reset", "p_f", "p_i"],
        output_names=["pose_out", "goal_out", "goal_k_out", "step_out", "term_out", "trunc_out",
                      "hit_out", "dist_out", "odom_out", "ep_out", "clear_out"],
        source=_STEP_SRC,
        header=header,
    )
    lidar = mx.fast.metal_kernel(
        name=f"nav_lidar_{kin_name}",
        input_names=["pose", "scene", "odom_st", "edf", "origin", "nse", "unif", "p_f", "p_i"],
        output_names=["lidar"],
        source=_LIDAR_SRC,
        header=header,
    )
    expert = mx.fast.metal_kernel(
        name=f"nav_expert_{kin_name}",
        input_names=["pose", "goal", "goal_k", "scene", "prev", "geo", "geo_origin",
                     "pe_f", "pe_i"],
        output_names=["act", "val_out", "dir_out"],
        source=_EXPERT_SRC,
        header=header,
    )
    return step, lidar, expert


def noisy_config(cfg: "SimConfig", level: float) -> "SimConfig":
    """Scale the realistic sim2real noise stack by `level` (1.0 = realistic)."""
    return dataclasses.replace(
        cfg, lidar_sigma=0.02 * level, lidar_dropout=0.02 * level,
        odom_rw=0.03 * level, odom_bias=0.02 * level, odom_scale=0.02 * level,
        head_rw=0.005 * level, head_bias=0.003 * level,
        act_noise=0.1 * level, act_scale=0.05 * level)


@dataclasses.dataclass
class SimConfig:
    kinematics: str = "holonomic"  # drive type, see fastnav/kinematics.py
    n_rays: int = 64
    max_range: float = 6.0
    dt: float = 0.1
    v_max: float = 1.5
    w_max: float = 2.5           # yaw-rate limit (rad/s); diffdrive only
    robot_radius: float = 0.18
    goal_radius: float = 0.25
    max_steps: int = 512
    min_goal_dist: float = 2.0   # episode geodesic length range
    max_goal_dist: float = 14.0
    detour_min: float = 0.0      # min geodesic/euclidean ratio for episode starts (curriculum)

    # --- sim2real noise model (all default 0 = ideal sensors/actuators) ---
    lidar_sigma: float = 0.0     # per-ray range noise sigma (m)
    lidar_dropout: float = 0.0   # per-ray prob of no-return (reads max_range)
    odom_rw: float = 0.0         # odometry random-walk sigma, fraction of distance moved
    odom_bias: float = 0.0       # per-episode systematic drift sigma (fraction of distance)
    odom_scale: float = 0.0      # per-episode odometry scale-factor sigma
    head_rw: float = 0.0         # heading random-walk sigma (rad per meter moved)
    head_bias: float = 0.0       # per-episode heading drift sigma (rad per meter)
    act_noise: float = 0.0       # additive actuation noise sigma (m/s)
    act_scale: float = 0.0       # per-episode actuation scale-factor sigma
    contact_margin: float = 0.01  # clearance below this = contact -> terminal FAILURE (0 disables)
    expert_slow_radius: float = 0.6
    expert_beta: float = 0.35
    expert_blend_radius: float = 0.5
    expert_turn_gain: float = 4.0  # diffdrive expert: omega = gain * heading error

    @property
    def obs_dim(self) -> int:
        return self.n_rays + 4  # lidar | rel_goal(2) | pos(2)


class Sim:
    """Fully batched sim. State lives in MLX arrays; step() is 2 kernel dispatches."""

    def __init__(self, pack: ScenePack, num_envs: int, cfg: SimConfig | None = None, seed: int = 0,
                 scene_assign: np.ndarray | None = None):
        self.pack = pack
        self.cfg = cfg = cfg or SimConfig()
        self.kin = kinematics.get(cfg.kinematics)
        self._step_kernel, self._lidar_kernel, self._expert_kernel = _kernels(cfg.kinematics)
        self.num_envs = n = num_envs
        mx.random.seed(seed)

        h, w = pack.grid_hw
        hg, wg = pack.geo_hw
        k = pack.n_goals
        m = pack.starts_xy.shape[2]

        self.edf = mx.array(pack.edf)
        self.origin = mx.array(pack.origin)
        self.geo = mx.array(pack.geo)
        self.geo_origin = mx.array(pack.geo_origin)
        self.starts = mx.array(pack.starts_xy)
        self.goals_all = mx.array(pack.goals_xy)
        self.pool, self.pool_cnt = self._build_start_pools(pack, cfg)

        self.p_f = mx.array([cfg.dt, cfg.v_max, cfg.robot_radius, cfg.goal_radius,
                             pack.cell, 1.0 / pack.cell, cfg.max_range,
                             cfg.odom_rw, cfg.odom_bias, cfg.odom_scale,
                             cfg.head_rw, cfg.head_bias, cfg.act_noise, cfg.act_scale,
                             cfg.lidar_sigma, cfg.lidar_dropout, cfg.contact_margin,
                             cfg.w_max], dtype=mx.float32)
        self.p_i = mx.array([h, w, k, m, cfg.max_steps, n, cfg.n_rays], dtype=mx.int32)
        self.pe_f = mx.array([pack.geo_cell, 1.0 / pack.geo_cell, cfg.v_max,
                              cfg.expert_slow_radius, cfg.expert_beta,
                              cfg.expert_blend_radius, cfg.w_max,
                              cfg.expert_turn_gain], dtype=mx.float32)
        self.pe_i = mx.array([hg, wg, k, n], dtype=mx.int32)

        if scene_assign is None:
            scene_assign = np.arange(n, dtype=np.int32) % len(pack.scenes)
        self.scene = mx.array(scene_assign.astype(np.int32))
        self.pose = mx.zeros((n, 3), dtype=mx.float32)   # true (x, y, heading)
        self.odom = mx.zeros((n, 3), dtype=mx.float32)   # believed (x, y) + frame angle
        self.ep_noise = mx.zeros((n, 5), dtype=mx.float32)
        self.goal = mx.zeros((n, 2), dtype=mx.float32)
        self.goal_k = mx.zeros((n,), dtype=mx.int32)
        self.step_ct = mx.zeros((n,), dtype=mx.int32)
        self.expert_prev = mx.zeros((n, 2), dtype=mx.float32)
        self.last_done = mx.zeros((n, 1), dtype=mx.float32)
        self._zero = mx.zeros((1,), dtype=mx.float32)
        self._one = mx.ones((1,), dtype=mx.float32)
        self.lidar = mx.zeros((n, cfg.n_rays), dtype=mx.float32)
        self.term = mx.zeros((n,), dtype=mx.uint8)
        self.trunc = mx.zeros((n,), dtype=mx.uint8)
        self.dist_goal = mx.zeros((n,), dtype=mx.float32)

    @property
    def pos(self) -> mx.array:
        """True position [N, 2] (read-only view of the pose)."""
        return self.pose[:, :2]

    @property
    def heading(self) -> mx.array:
        """True heading [N] (holonomic: identically 0)."""
        return self.pose[:, 2]

    @staticmethod
    def _build_start_pools(pack: ScenePack, cfg: SimConfig) -> tuple[mx.array, mx.array]:
        """Per (scene, goal): indices into the start table satisfying the geodesic
        range and detour-ratio filter. Falls back to range-only, then to all."""
        s, k, m = pack.starts_geo.shape
        pool = np.zeros((s, k, m), dtype=np.int32)
        cnt = np.zeros((s, k), dtype=np.int32)
        euclid = np.linalg.norm(pack.starts_xy - pack.goals_xy[:, :, None, :], axis=-1)
        ratio = pack.starts_geo / np.maximum(euclid, 1e-6)
        for i in range(s):
            for j in range(k):
                n = pack.start_counts[i, j]
                geo = pack.starts_geo[i, j, :n]
                in_range = (geo >= cfg.min_goal_dist) & (geo <= cfg.max_goal_dist)
                ok = in_range & (ratio[i, j, :n] >= cfg.detour_min)
                idx = np.nonzero(ok)[0]
                if len(idx) < 16:
                    idx = np.nonzero(in_range)[0]
                if len(idx) == 0:
                    idx = np.arange(max(n, 1))
                pool[i, j, : len(idx)] = idx
                cnt[i, j] = len(idx)
        return mx.array(pool), mx.array(cnt)

    def _step_raw(self, actions: mx.array, force_reset: mx.array) -> None:
        n = self.num_envs
        r = self.cfg.n_rays
        rnd = mx.random.uniform(shape=(n, self.kin.n_uniform))
        rnd_n = mx.random.normal(shape=(n, 10))
        outs = self._step_kernel(
            inputs=[self.pose, actions, self.goal, self.goal_k, self.step_ct, self.scene,
                    self.edf, self.origin, self.starts, self.goals_all, self.pool,
                    self.pool_cnt, self.odom, self.ep_noise, rnd, rnd_n, force_reset,
                    self.p_f, self.p_i],
            output_shapes=[(n, 3), (n, 2), (n,), (n,), (n,), (n,), (n,), (n,), (n, 3), (n, 5), (n,)],
            output_dtypes=[mx.float32, mx.float32, mx.int32, mx.int32, mx.uint8, mx.uint8,
                           mx.uint8, mx.float32, mx.float32, mx.float32, mx.float32],
            grid=(n, 1, 1),
            threadgroup=(256, 1, 1),
        )
        (self.pose, self.goal, self.goal_k, self.step_ct, self.term, self.trunc,
         self.hit, self.dist_goal, self.odom, self.ep_noise, self.clearance) = outs
        self.lidar = self._lidar_kernel(
            inputs=[self.pose, self.scene, self.odom, self.edf, self.origin,
                    mx.random.normal(shape=(n, r)), mx.random.uniform(shape=(n, r)),
                    self.p_f, self.p_i],
            output_shapes=[(n, r)],
            output_dtypes=[mx.float32],
            grid=(n * r, 1, 1),
            threadgroup=(256, 1, 1),
        )[0]
        self.last_done = mx.maximum(self.term, self.trunc).astype(mx.float32)[:, None]

    def obs(self) -> mx.array:
        """Observation uses the believed (odometry) pose, never the true pose.
        The goal vector is expressed in the kinematics' observation frame."""
        odom_xy = self.odom[:, :2]
        return mx.concatenate([self.lidar, self.kin.rel_goal(self.goal, self.odom),
                               odom_xy], axis=1)

    def reset(self) -> mx.array:
        self._step_raw(mx.zeros((self.num_envs, 2), dtype=mx.float32), self._one)
        self.expert_prev = mx.zeros((self.num_envs, 2), dtype=mx.float32)
        self.last_done = mx.zeros((self.num_envs, 1), dtype=mx.float32)
        o = self.obs()
        mx.eval(o, self.pos)
        return o

    def step(self, actions: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Returns (obs, terminated, truncated); auto-resets internally."""
        self._step_raw(actions, self._zero)
        return self.obs(), self.term, self.trunc

    def set_state(self, pos: np.ndarray, goal: np.ndarray, goal_k: np.ndarray,
                  heading: np.ndarray | None = None) -> None:
        """Force exact episode states (e.g. to replay failures). Resets counters.
        The believed pose is anchored to the true pose (odometry starts exact)."""
        n = self.num_envs
        head = (heading if heading is not None else np.zeros(len(pos))).astype(np.float32)
        self.pose = mx.array(np.concatenate([pos.astype(np.float32), head[:, None]], axis=1))
        self.goal = mx.array(goal.astype(np.float32))
        self.goal_k = mx.array(goal_k.astype(np.int32))
        self.step_ct = mx.zeros((n,), dtype=mx.int32)
        self.term = mx.zeros((n,), dtype=mx.uint8)
        self.trunc = mx.zeros((n,), dtype=mx.uint8)
        self.expert_prev = mx.zeros((n, 2), dtype=mx.float32)
        self.last_done = mx.zeros((n, 1), dtype=mx.float32)
        self.dist_goal = mx.sqrt(mx.sum(mx.square(self.goal - self.pos), axis=1))
        self.odom = mx.array(self.pose)  # believed pose anchored to the true pose
        self.ep_noise = mx.zeros((n, 5), dtype=mx.float32)
        r = self.cfg.n_rays
        self.lidar = self._lidar_kernel(
            inputs=[self.pose, self.scene, self.odom, self.edf, self.origin,
                    mx.random.normal(shape=(n, r)), mx.random.uniform(shape=(n, r)),
                    self.p_f, self.p_i],
            output_shapes=[(n, r)],
            output_dtypes=[mx.float32],
            grid=(n * r, 1, 1),
            threadgroup=(256, 1, 1),
        )[0]
        mx.eval(self.pose, self.lidar)

    def expert_actions(self) -> mx.array:
        """Expert drive command (in the kinematics' action space). Also stores the
        geodesic cost-to-go (m) in self.expert_geo_val (oracle for value
        distillation) and the planner's desired (world angle, speed) in
        self.expert_dir (enables heading-relabel augmentation)."""
        prev = self.expert_prev * (1.0 - self.last_done)
        act, val, dirs = self._expert_kernel(
            inputs=[self.pose, self.goal, self.goal_k, self.scene, prev,
                    self.geo, self.geo_origin, self.pe_f, self.pe_i],
            output_shapes=[(self.num_envs, 2), (self.num_envs,), (self.num_envs, 2)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
            grid=(self.num_envs, 1, 1),
            threadgroup=(256, 1, 1),
        )
        self.expert_prev = act
        self.expert_geo_val = val
        self.expert_dir = dirs
        return act
