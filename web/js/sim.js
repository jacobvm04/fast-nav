// 1:1 JavaScript port of the Metal step/lidar kernels in fastnav/sim.py,
// for a single robot. Same bilinear EDF sampling, same collision projection,
// same sphere-traced lidar, same constants — including the sim2real noise
// stack: SE(2) odometry drift, heading error, lidar + actuation noise.
//
// Kinematics mirror fastnav/kinematics.py: each entry owns the action clamp +
// actuation noise (execute -> body twist), the command/sensor frame, the
// episode-start heading, and the goal rotation into the observation frame.
// The step/lidar/obs code below is kinematics-agnostic, like the kernels.

import { signedEDF } from './edt.js';

const NOISE_KEYS = ['lidar_sigma', 'lidar_dropout', 'odom_rw', 'odom_bias', 'odom_scale',
  'head_rw', 'head_bias', 'act_noise', 'act_scale'];

const TWO_PI = 2 * Math.PI;

export const KINEMATICS = {
  holonomic: {
    bodyOriented: false, // actions/sensing in the (believed) world frame
    // (vx, vy) in the believed world frame, norm-clamped
    execute(sim, a0, a1, ascale, g0, g1) {
      const n = Math.hypot(a0, a1);
      if (n > sim.cfg.v_max) {
        a0 *= sim.cfg.v_max / n;
        a1 *= sim.cfg.v_max / n;
      }
      return [a0 * ascale + sim.noise.act_noise * g0,
        a1 * ascale + sim.noise.act_noise * g1, 0];
    },
    // the believed frame is meant to be world-aligned; odom[2] is its error
    frame: (sim) => sim.odom[2],
    resetTheta(sim) {
      sim.heading = 0;
      sim.odom[2] = 0;
    },
    relGoal: (sim) => [sim.goal[0] - sim.odom[0], sim.goal[1] - sim.odom[1]],
  },
  diffdrive: {
    // (v, omega): forward velocity + yaw rate, executed in the true body frame
    execute(sim, a0, a1, ascale, g0, g1) {
      const { v_max, w_max } = sim.cfg;
      const v = Math.min(Math.max(a0, -v_max), v_max);
      const w = Math.min(Math.max(a1, -w_max), w_max);
      return [v * ascale + sim.noise.act_noise * g0, 0,
        w * ascale + sim.noise.act_noise * (w_max / v_max) * g1];
    },
    frame: (sim) => sim.heading,
    resetTheta(sim) {
      sim.heading = TWO_PI * Math.random() - Math.PI;
      sim.odom[2] = sim.heading; // believed heading starts exact
    },
    relGoal(sim) {
      const rx = sim.goal[0] - sim.odom[0];
      const ry = sim.goal[1] - sim.odom[1];
      const c = Math.cos(sim.odom[2]);
      const s = Math.sin(sim.odom[2]);
      return [c * rx + s * ry, -s * rx + c * ry];
    },
  },
};

export class Sim {
  // occ: Uint8Array [h*w] (1 = occupied), origin: [x, y] world coords of cell (0,0)
  constructor(occ, h, w, cell, origin, cfg) {
    this.h = h;
    this.w = w;
    this.cell = cell;
    this.ox = origin[0];
    this.oy = origin[1];
    this.cfg = cfg; // {kinematics?, n_rays, max_range, dt, v_max, w_max?, robot_radius, goal_radius, max_steps}
    this.kin = KINEMATICS[cfg.kinematics || 'holonomic'];
    this.occ = occ;
    this.edf = signedEDF(occ, h, w, cell);
    this.lidar = new Float32Array(cfg.n_rays);
    this.pos = [0, 0];
    this.heading = 0; // true heading (holonomic: 0, unused)
    this.goal = [0, 0];
    this.stepCount = 0;
    this.noise = Object.fromEntries(NOISE_KEYS.map((k) => [k, 0]));
    this.odom = [0, 0, 0]; // believed (x, y) + frame angle (see kinematics.py)
    this.ep = [0, 0, 0, 0, 0]; // per-episode: drift bias x/y, odom scale, head bias, act scale
    this._spare = null; // Box-Muller cache
  }

  gauss() {
    if (this._spare !== null) {
      const s = this._spare;
      this._spare = null;
      return s;
    }
    let u, v, s;
    do {
      u = Math.random() * 2 - 1;
      v = Math.random() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const m = Math.sqrt((-2 * Math.log(s)) / s);
    this._spare = v * m;
    return u * m;
  }

  // noise_stack: per-field sigmas at level 1.0 (manifest); level: UI multiplier.
  setNoise(stack, level) {
    for (const k of NOISE_KEYS) this.noise[k] = (stack[k] || 0) * level;
    this.resampleEpisodeNoise();
  }

  // Matches the auto-reset branch of the step kernel: re-anchor odometry to the
  // true pose and resample the per-episode systematic errors.
  resampleEpisodeNoise() {
    const n = this.noise;
    this.ep = [n.odom_bias * this.gauss(), n.odom_bias * this.gauss(),
      n.odom_scale * this.gauss(), n.head_bias * this.gauss(),
      n.act_scale * this.gauss()];
  }

  // New episode at the current pose: believed pose re-anchors to the true pose
  // (the heading is kept — the web robot is persistent across episodes).
  newEpisode() {
    this.odom = [this.pos[0], this.pos[1], this.kin === KINEMATICS.holonomic ? 0 : this.heading];
    this.resampleEpisodeNoise();
    this.stepCount = 0;
    this.updateLidar();
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

  // Recompute the EDF after this.occ has been edited (obstacle painting).
  rebuildEDF() {
    this.edf = signedEDF(this.occ, this.h, this.w, this.cell);
  }

  // True world orientation of the sensor frame (kin_frame in the kernels).
  sensorAngle() {
    return this.kin.frame(this);
  }

  // _LIDAR_SRC: sphere tracing through the EDF. Rays are indexed in the sensor
  // frame; its true world orientation rotates them in the true frame. Range
  // noise and per-ray dropout applied to the traced distance.
  updateLidar() {
    const { cfg, cell, noise } = this;
    const eps = 0.5 * cell;
    const minstep = 0.3 * cell;
    const [px, py] = this.pos;
    const frame = this.sensorAngle();
    for (let r = 0; r < cfg.n_rays; r++) {
      const theta = (TWO_PI * r) / cfg.n_rays + frame;
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
      if (noise.lidar_sigma > 0) tt += noise.lidar_sigma * this.gauss();
      if (noise.lidar_dropout > 0 && Math.random() < noise.lidar_dropout) tt = cfg.max_range;
      this.lidar[r] = Math.min(Math.max(tt, 0), cfg.max_range);
    }
  }

  // _STEP_SRC integration: kinematics execute (clamp + actuation noise -> body
  // twist), 2 substeps of rotate-then-translate, 5-iter EDF gradient projection
  // with revert on failure, then shared odometry. Returns {reached, truncated}.
  step(a0, a1) {
    const { cfg, cell, noise } = this;
    const inv_cell = 1 / cell;
    const ascale = 1 + this.ep[4];
    const [e0, e1, wz] = this.kin.execute(this, a0, a1, ascale, this.gauss(), this.gauss());
    const f0 = this.kin.frame(this);
    const ct = Math.cos(f0);
    const sn = Math.sin(f0);
    let [px, py] = this.pos;
    const px0 = px, py0 = py;
    let thq = f0;
    const SUB = 2;
    for (let sub = 0; sub < SUB; sub++) {
      thq += (wz * cfg.dt) / SUB; // rotate, then translate along the new heading
      const cq = Math.cos(thq);
      const sq = Math.sin(thq);
      const vx = cq * e0 - sq * e1;
      const vy = sq * e0 + cq * e1;
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
    const dth = wz * cfg.dt;
    this.heading += dth; // rotation is never blocked by contact (disk robot)
    this.pos = [px, py];
    // integrate odometry: measured displacement = R(-frame) * true, plus scale
    // error, per-episode bias, and distance-scaled random walk; heading drift
    // also grows with rotation (0.5 m-per-rad equivalence)
    const tdx = px - px0;
    const tdy = py - py0;
    const dl = Math.sqrt(tdx * tdx + tdy * tdy);
    const derr = dl + 0.5 * Math.abs(dth);
    const mdx = ct * tdx + sn * tdy;
    const mdy = -sn * tdx + ct * tdy;
    const oscale = 1 + this.ep[2];
    this.odom[0] += mdx * oscale + (this.ep[0] + noise.odom_rw * this.gauss()) * dl;
    this.odom[1] += mdy * oscale + (this.ep[1] + noise.odom_rw * this.gauss()) * dl;
    this.odom[2] += dth + (this.ep[3] + noise.head_rw * this.gauss()) * derr;
    const ddx = this.goal[0] - px;
    const ddy = this.goal[1] - py;
    const dist = Math.sqrt(ddx * ddx + ddy * ddy);
    const reached = dist < cfg.goal_radius;
    this.stepCount += 1;
    const truncated = this.stepCount >= cfg.max_steps && !reached;
    this.updateLidar();
    return { reached, truncated, dist };
  }

  // Lidar resolution follows the selected policy. The cfg object is copied,
  // not mutated — the manifest's sim config is shared across sims.
  setRays(n) {
    if (n === this.cfg.n_rays) return;
    this.cfg = { ...this.cfg, n_rays: n };
    this.lidar = new Float32Array(n);
    this.updateLidar();
  }

  // Drive type follows the selected policy (like setRays).
  setKinematics(name) {
    const kin = KINEMATICS[name || 'holonomic'];
    if (kin === this.kin) return;
    this.kin = kin;
    this.cfg = { ...this.cfg, kinematics: name };
    kin.resetTheta(this);
    this.newEpisode();
  }

  // obs = [lidar (R) | rel_goal (2) | odom (2)]: the policy sees the believed
  // (odometry) pose, never the true pose — as in Sim.obs(). The goal vector is
  // expressed in the kinematics' observation frame.
  obs(out) {
    const R = this.cfg.n_rays;
    out.set(this.lidar, 0);
    const [gx, gy] = this.kin.relGoal(this);
    out[R] = gx;
    out[R + 1] = gy;
    out[R + 2] = this.odom[0];
    out[R + 3] = this.odom[1];
    return out;
  }

  setState(pos, goal) {
    this.pos = [pos[0], pos[1]];
    this.goal = [goal[0], goal[1]];
    this.kin.resetTheta(this);
    this.newEpisode();
  }
}
