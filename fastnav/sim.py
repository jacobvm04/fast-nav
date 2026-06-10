"""Batched 2D-lidar navigation sim on Apple GPU via MLX custom Metal kernels.

Three kernels per step, all operating on the whole env batch:
  step_kernel   [N threads]    holonomic integration, EDF collision projection,
                               termination, fused auto-reset (start/goal resample)
  lidar_kernel  [N*R threads]  sphere tracing through the signed EDF
  expert_kernel [N threads]    gradient descent on precomputed geodesic fields

Throughput comes from batch size: Python/dispatch overhead is paid once per
batched step, so ~8k envs at ~150 batched steps/s > 1M env steps/s.
"""

from __future__ import annotations

import dataclasses

import mlx.core as mx
import numpy as np

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

# p_i: [H, W, K, M, max_steps, N, R]   p_f: [dt, vmax, radius, goal_radius, cell, inv_cell, max_range]
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
    float px = pos[i * 2], py = pos[i * 2 + 1];
    float gx = goal[i * 2], gy = goal[i * 2 + 1];
    int gk = goal_k[i];
    int st = step_ct[i];

    bool freset = force_reset[0] > 0.5f;
    bool reached = false;
    bool trunc = false;
    float dist_pre = 0.0f;

    if (!freset) {
        float vx = vel[i * 2], vy = vel[i * 2 + 1];
        float vn = metal::sqrt(vx * vx + vy * vy);
        if (vn > vmax) { vx *= vmax / vn; vy *= vmax / vn; }
        const int SUB = 2;
        for (int sub = 0; sub < SUB; sub++) {
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
        float ddx = gx - px, ddy = gy - py;
        dist_pre = metal::sqrt(ddx * ddx + ddy * ddy);
        reached = dist_pre < goal_r;
        st += 1;
        trunc = (st >= max_steps) && !reached;
    }

    bool done = freset || reached || trunc;
    if (done) {
        float u0 = rnd[i * 4], u1 = rnd[i * 4 + 1], u2 = rnd[i * 4 + 2], u3 = rnd[i * 4 + 3];
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
    }

    pos_out[i * 2] = px;
    pos_out[i * 2 + 1] = py;
    goal_out[i * 2] = gx;
    goal_out[i * 2 + 1] = gy;
    goal_k_out[i] = gk;
    step_out[i] = st;
    term_out[i] = reached ? (uint8_t)1 : (uint8_t)0;
    trunc_out[i] = trunc ? (uint8_t)1 : (uint8_t)0;
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
    float theta = 6.283185307f * (float)r / (float)R;
    float dx = metal::cos(theta), dy = metal::sin(theta);
    float px = pos[i * 2], py = pos[i * 2 + 1];

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
    lidar[(long)i * R + r] = metal::min(tt, max_range);
"""

# pe_i: [Hg, Wg, K, N]   pe_f: [geo_cell, inv_geo_cell, vmax, slow_radius, beta, blend_radius]
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
    float px = pos[i * 2], py = pos[i * 2 + 1];
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
    act[i * 2] = prev[i * 2] + beta * (ax - prev[i * 2]);
    act[i * 2 + 1] = prev[i * 2 + 1] + beta * (ay - prev[i * 2 + 1]);
"""

_step_kernel = mx.fast.metal_kernel(
    name="nav_step",
    input_names=["pos", "vel", "goal", "goal_k", "step_ct", "scene", "edf", "origin",
                 "starts", "goals_all", "pool", "pool_cnt", "rnd", "force_reset", "p_f", "p_i"],
    output_names=["pos_out", "goal_out", "goal_k_out", "step_out", "term_out", "trunc_out", "dist_out"],
    source=_STEP_SRC,
    header=_HEADER,
)

_lidar_kernel = mx.fast.metal_kernel(
    name="nav_lidar",
    input_names=["pos", "scene", "edf", "origin", "p_f", "p_i"],
    output_names=["lidar"],
    source=_LIDAR_SRC,
    header=_HEADER,
)

_expert_kernel = mx.fast.metal_kernel(
    name="nav_expert",
    input_names=["pos", "goal", "goal_k", "scene", "prev", "geo", "geo_origin", "pe_f", "pe_i"],
    output_names=["act", "val_out"],
    source=_EXPERT_SRC,
    header=_HEADER,
)


@dataclasses.dataclass
class SimConfig:
    n_rays: int = 64
    max_range: float = 6.0
    dt: float = 0.1
    v_max: float = 1.5
    robot_radius: float = 0.18
    goal_radius: float = 0.25
    max_steps: int = 512
    min_goal_dist: float = 2.0   # episode geodesic length range
    max_goal_dist: float = 14.0
    detour_min: float = 0.0      # min geodesic/euclidean ratio for episode starts (curriculum)
    expert_slow_radius: float = 0.6
    expert_beta: float = 0.35
    expert_blend_radius: float = 0.5

    @property
    def obs_dim(self) -> int:
        return self.n_rays + 4  # lidar | rel_goal(2) | pos(2)


class Sim:
    """Fully batched sim. State lives in MLX arrays; step() is 2 kernel dispatches."""

    def __init__(self, pack: ScenePack, num_envs: int, cfg: SimConfig | None = None, seed: int = 0,
                 scene_assign: np.ndarray | None = None):
        self.pack = pack
        self.cfg = cfg = cfg or SimConfig()
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
                             pack.cell, 1.0 / pack.cell, cfg.max_range], dtype=mx.float32)
        self.p_i = mx.array([h, w, k, m, cfg.max_steps, n, cfg.n_rays], dtype=mx.int32)
        self.pe_f = mx.array([pack.geo_cell, 1.0 / pack.geo_cell, cfg.v_max,
                              cfg.expert_slow_radius, cfg.expert_beta,
                              cfg.expert_blend_radius], dtype=mx.float32)
        self.pe_i = mx.array([hg, wg, k, n], dtype=mx.int32)

        if scene_assign is None:
            scene_assign = np.arange(n, dtype=np.int32) % len(pack.scenes)
        self.scene = mx.array(scene_assign.astype(np.int32))
        self.pos = mx.zeros((n, 2), dtype=mx.float32)
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
        rnd = mx.random.uniform(shape=(n, 4))
        outs = _step_kernel(
            inputs=[self.pos, actions, self.goal, self.goal_k, self.step_ct, self.scene,
                    self.edf, self.origin, self.starts, self.goals_all, self.pool,
                    self.pool_cnt, rnd, force_reset, self.p_f, self.p_i],
            output_shapes=[(n, 2), (n, 2), (n,), (n,), (n,), (n,), (n,)],
            output_dtypes=[mx.float32, mx.float32, mx.int32, mx.int32, mx.uint8, mx.uint8, mx.float32],
            grid=(n, 1, 1),
            threadgroup=(256, 1, 1),
        )
        self.pos, self.goal, self.goal_k, self.step_ct, self.term, self.trunc, self.dist_goal = outs
        self.lidar = _lidar_kernel(
            inputs=[self.pos, self.scene, self.edf, self.origin, self.p_f, self.p_i],
            output_shapes=[(n, self.cfg.n_rays)],
            output_dtypes=[mx.float32],
            grid=(n * self.cfg.n_rays, 1, 1),
            threadgroup=(256, 1, 1),
        )[0]
        self.last_done = mx.maximum(self.term, self.trunc).astype(mx.float32)[:, None]

    def obs(self) -> mx.array:
        return mx.concatenate([self.lidar, self.goal - self.pos, self.pos], axis=1)

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

    def set_state(self, pos: np.ndarray, goal: np.ndarray, goal_k: np.ndarray) -> None:
        """Force exact episode states (e.g. to replay failures). Resets counters."""
        n = self.num_envs
        self.pos = mx.array(pos.astype(np.float32))
        self.goal = mx.array(goal.astype(np.float32))
        self.goal_k = mx.array(goal_k.astype(np.int32))
        self.step_ct = mx.zeros((n,), dtype=mx.int32)
        self.term = mx.zeros((n,), dtype=mx.uint8)
        self.trunc = mx.zeros((n,), dtype=mx.uint8)
        self.expert_prev = mx.zeros((n, 2), dtype=mx.float32)
        self.last_done = mx.zeros((n, 1), dtype=mx.float32)
        self.dist_goal = mx.sqrt(mx.sum(mx.square(self.goal - self.pos), axis=1))
        self.lidar = _lidar_kernel(
            inputs=[self.pos, self.scene, self.edf, self.origin, self.p_f, self.p_i],
            output_shapes=[(n, self.cfg.n_rays)],
            output_dtypes=[mx.float32],
            grid=(n * self.cfg.n_rays, 1, 1),
            threadgroup=(256, 1, 1),
        )[0]
        mx.eval(self.pos, self.lidar)

    def expert_actions(self) -> mx.array:
        """Expert velocity command; also stores the geodesic cost-to-go (m) at the
        current positions in self.expert_geo_val (oracle for value distillation)."""
        prev = self.expert_prev * (1.0 - self.last_done)
        act, val = _expert_kernel(
            inputs=[self.pos, self.goal, self.goal_k, self.scene, prev,
                    self.geo, self.geo_origin, self.pe_f, self.pe_i],
            output_shapes=[(self.num_envs, 2), (self.num_envs,)],
            output_dtypes=[mx.float32, mx.float32],
            grid=(self.num_envs, 1, 1),
            threadgroup=(256, 1, 1),
        )
        self.expert_prev = act
        self.expert_geo_val = val
        return act
