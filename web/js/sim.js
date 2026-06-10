// 1:1 JavaScript port of the Metal step/lidar kernels in fastnav/sim.py,
// for a single robot. Same bilinear EDF sampling, same collision projection,
// same sphere-traced lidar, same constants.

import { signedEDF } from './edt.js';

export class Sim {
  // occ: Uint8Array [h*w] (1 = occupied), origin: [x, y] world coords of cell (0,0)
  constructor(occ, h, w, cell, origin, cfg) {
    this.h = h;
    this.w = w;
    this.cell = cell;
    this.ox = origin[0];
    this.oy = origin[1];
    this.cfg = cfg; // {n_rays, max_range, dt, v_max, robot_radius, goal_radius, max_steps}
    this.occ = occ;
    this.edf = signedEDF(occ, h, w, cell);
    this.lidar = new Float32Array(cfg.n_rays);
    this.pos = [0, 0];
    this.goal = [0, 0];
    this.stepCount = 0;
  }

  // bilin() from the Metal header, with the same edge clamping.
  sampleEDF(gx, gy) {
    const { h: H, w: W, edf } = this;
    gx = Math.min(Math.max(gx, 0), W - 1.001);
    gy = Math.min(Math.max(gy, 0), H - 1.001);
    const x0 = Math.floor(gx);
    const y0 = Math.floor(gy);
    const fx = gx - x0;
    const fy = gy - y0;
    const i00 = y0 * W + x0;
    const v00 = edf[i00];
    const v01 = edf[i00 + 1];
    const v10 = edf[i00 + W];
    const v11 = edf[i00 + W + 1];
    return (v00 + (v01 - v00) * fx) + ((v10 + (v11 - v10) * fx) - (v00 + (v01 - v00) * fx)) * fy;
  }

  edfAt(x, y) {
    return this.sampleEDF((x - this.ox) / this.cell, (y - this.oy) / this.cell);
  }

  // _LIDAR_SRC: sphere tracing through the EDF, rays fixed in the world frame.
  updateLidar() {
    const { cfg, cell } = this;
    const eps = 0.5 * cell;
    const minstep = 0.3 * cell;
    const [px, py] = this.pos;
    for (let r = 0; r < cfg.n_rays; r++) {
      const theta = (2 * Math.PI * r) / cfg.n_rays;
      const dx = Math.cos(theta);
      const dy = Math.sin(theta);
      let tt = 0;
      for (let it = 0; it < 96; it++) {
        const gx = (px + tt * dx - this.ox) / cell;
        const gy = (py + tt * dy - this.oy) / cell;
        const d = this.sampleEDF(gx, gy);
        if (d < eps) break;
        tt += Math.max(d, minstep);
        if (tt >= cfg.max_range) { tt = cfg.max_range; break; }
      }
      this.lidar[r] = Math.min(tt, cfg.max_range);
    }
  }

  // _STEP_SRC integration: velocity clamp, 2 substeps, 5-iter EDF gradient
  // projection, revert on failure. Returns {reached, truncated}.
  step(vx, vy) {
    const { cfg, cell } = this;
    const inv_cell = 1 / cell;
    const vn = Math.sqrt(vx * vx + vy * vy);
    if (vn > cfg.v_max) {
      vx *= cfg.v_max / vn;
      vy *= cfg.v_max / vn;
    }
    let [px, py] = this.pos;
    const SUB = 2;
    for (let sub = 0; sub < SUB; sub++) {
      const sx = px, sy = py;
      px += (vx * cfg.dt) / SUB;
      py += (vy * cfg.dt) / SUB;
      let d = 0;
      for (let it = 0; it < 5; it++) {
        const cgx = (px - this.ox) * inv_cell;
        const cgy = (py - this.oy) * inv_cell;
        d = this.sampleEDF(cgx, cgy);
        if (d >= cfg.robot_radius || it === 4) break; // last pass only re-checks
        const dxp = this.sampleEDF(cgx + 1, cgy) - this.sampleEDF(cgx - 1, cgy);
        const dyp = this.sampleEDF(cgx, cgy + 1) - this.sampleEDF(cgx, cgy - 1);
        const gl = Math.sqrt(dxp * dxp + dyp * dyp);
        if (gl < 1e-6) break;
        const push = (cfg.robot_radius - d) + 0.25 * cell;
        px += (dxp / gl) * push;
        py += (dyp / gl) * push;
      }
      if (d < cfg.robot_radius) { px = sx; py = sy; } // projection failed: stay put
    }
    this.pos = [px, py];
    const ddx = this.goal[0] - px;
    const ddy = this.goal[1] - py;
    const dist = Math.sqrt(ddx * ddx + ddy * ddy);
    const reached = dist < cfg.goal_radius;
    this.stepCount += 1;
    const truncated = this.stepCount >= cfg.max_steps && !reached;
    this.updateLidar();
    return { reached, truncated, dist };
  }

  // obs = [lidar (R) | goal - pos (2) | pos (2)], as in Sim.obs().
  obs(out) {
    const R = this.cfg.n_rays;
    out.set(this.lidar, 0);
    out[R] = this.goal[0] - this.pos[0];
    out[R + 1] = this.goal[1] - this.pos[1];
    out[R + 2] = this.pos[0];
    out[R + 3] = this.pos[1];
    return out;
  }

  setState(pos, goal) {
    this.pos = [pos[0], pos[1]];
    this.goal = [goal[0], goal[1]];
    this.stepCount = 0;
    this.updateLidar();
  }
}
