// Interactive demo: pick a held-out scene, click a goal, watch the trained
// policy navigate. Physics is the exact training sim (sim.js) at dt = 0.1 s;
// rendering interpolates between physics states at display refresh rate.

import { loadManifest, loadPolicy } from './policy.js';
import { Sim } from './sim.js';

const $ = (id) => document.getElementById(id);
const canvas = $('canvas');
const ctx = canvas.getContext('2d');

const state = {
  manifest: null,
  policy: null,
  policyBins: {},      // id -> Policy (loaded lazily)
  noiseLevel: 0,
  index: [],
  sim: null,
  comp: null,          // Int32Array component label per cell (-1 = blocked)
  mainComp: -1,
  mapCanvas: null,     // prerendered scene bitmap
  view: { s: 1, x0: 0, y0: 0 }, // world -> screen
  mode: 'idle',        // idle | running | success | blocked-flash
  paused: false,
  speed: 1,
  acc: 0,
  lastT: 0,
  prevPos: [0, 0],
  trail: [],
  heading: [1, 0],
  succ: 0,
  tries: 0,
  stepMs: 0,
  ripple: 0,           // success animation clock (s)
  obsBuf: null,
  tool: 'nav',         // nav | wall | erase
  brushR: 0.3,         // meters
  hover: null,         // [wx, wy] for brush preview
  painting: false,
  lastPaint: null,
  occBase: null,       // pristine occupancy for "reset map"
  rebuildMs: 10,       // EMA of derived-state rebuild cost
  lastRebuild: 0,
  sealed: false,
};

// ---------------------------------------------------------------- scene load

async function loadOccupancy(meta) {
  const img = new Image();
  img.src = meta.file;
  await img.decode();
  const off = new OffscreenCanvas(meta.w, meta.h);
  const c = off.getContext('2d', { willReadFrequently: true });
  c.drawImage(img, 0, 0);
  const data = c.getImageData(0, 0, meta.w, meta.h).data;
  const occ = new Uint8Array(meta.w * meta.h);
  for (let i = 0; i < occ.length; i++) occ[i] = data[i * 4] < 128 ? 1 : 0;
  return occ;
}

// Connected components of {edf > robot_radius} for click validation + spawning.
function labelComponents(sim) {
  const { h, w, edf, cfg } = sim;
  const comp = new Int32Array(h * w).fill(-1);
  const stack = new Int32Array(h * w);
  let nComp = 0;
  let bestComp = -1;
  let bestSize = 0;
  for (let seed = 0; seed < h * w; seed++) {
    if (comp[seed] !== -1 || edf[seed] <= cfg.robot_radius) continue;
    let top = 0;
    stack[top++] = seed;
    comp[seed] = nComp;
    let size = 0;
    while (top > 0) {
      const i = stack[--top];
      size++;
      const y = (i / w) | 0;
      const x = i - y * w;
      if (x > 0 && comp[i - 1] === -1 && edf[i - 1] > cfg.robot_radius) { comp[i - 1] = nComp; stack[top++] = i - 1; }
      if (x < w - 1 && comp[i + 1] === -1 && edf[i + 1] > cfg.robot_radius) { comp[i + 1] = nComp; stack[top++] = i + 1; }
      if (y > 0 && comp[i - w] === -1 && edf[i - w] > cfg.robot_radius) { comp[i - w] = nComp; stack[top++] = i - w; }
      if (y < h - 1 && comp[i + w] === -1 && edf[i + w] > cfg.robot_radius) { comp[i + w] = nComp; stack[top++] = i + w; }
    }
    if (size > bestSize) { bestSize = size; bestComp = nComp; }
    nComp++;
  }
  return { comp, mainComp: bestComp };
}

function compAt(x, y) {
  const { sim, comp } = state;
  const ix = Math.round((x - sim.ox) / sim.cell);
  const iy = Math.round((y - sim.oy) / sim.cell);
  if (ix < 0 || ix >= sim.w || iy < 0 || iy >= sim.h) return -1;
  return comp[iy * sim.w + ix];
}

function renderMapBitmap(sim) {
  const { h, w, edf } = sim;
  const img = new ImageData(w, h);
  const d = img.data;
  for (let i = 0; i < h * w; i++) {
    const e = edf[i];
    let r, g, b;
    if (e <= 0) { r = 56; g = 65; b = 94; }       // wall
    else {                                          // floor, lighter near walls
      const t = Math.min(e / 0.7, 1);
      r = 22 - 8 * t; g = 28 - 10 * t; b = 46 - 16 * t;
    }
    const o = i * 4;
    d[o] = r; d[o + 1] = g; d[o + 2] = b; d[o + 3] = 255;
  }
  const off = new OffscreenCanvas(w, h);
  off.getContext('2d').putImageData(img, 0, 0);
  return off;
}

function randomSpawn() {
  const { sim, comp, mainComp } = state;
  const { h, w, edf, cfg } = sim;
  for (let tries = 0; tries < 10000; tries++) {
    const i = (Math.random() * h * w) | 0;
    if (comp[i] === mainComp && edf[i] > cfg.robot_radius + 0.1) {
      const y = (i / w) | 0;
      const x = i - y * w;
      return [sim.ox + x * sim.cell, sim.oy + y * sim.cell];
    }
  }
  return [sim.ox + (w / 2) * sim.cell, sim.oy + (h / 2) * sim.cell];
}

async function setScene(name) {
  const meta = state.index.find((m) => m.name === name);
  $('status').textContent = `loading ${meta.name}…`;
  const occ = await loadOccupancy(meta);
  const sim = new Sim(occ, meta.h, meta.w, meta.cell, meta.origin, state.manifest.sim);
  sim.setNoise(state.manifest.noise_stack, state.noiseLevel);
  state.sim = sim;
  state.occBase = occ.slice();
  state.sealed = false;
  state.painting = false;
  state.hover = null;
  Object.assign(state, labelComponents(sim));
  state.mapCanvas = renderMapBitmap(sim);
  const spawn = randomSpawn();
  sim.setState(spawn, spawn);
  state.prevPos = [...spawn];
  state.trail = [];
  state.heading = [1, 0];
  state.mode = 'idle';
  state.policy.reset();
  fitView();
  const sizeM = `${(meta.w * meta.cell).toFixed(0)}×${(meta.h * meta.cell).toFixed(0)} m`;
  setStatus(`${meta.name} (${meta.group}, ${sizeM}) — click anywhere to set a goal`);
  $('s-steps').textContent = '–';
  $('s-val').textContent = '–';
}

// --------------------------------------------------------- obstacle brushes

// Paint a disk of occupancy. Wall mode protects the robot and the goal point;
// erase mode preserves a 2-cell solid ring at the grid border (the EDF sampler
// clamps at the edges, so a breached border would read as open world).
function paintDisk(wx, wy, add) {
  const { sim } = state;
  const rc = state.brushR / sim.cell;
  const gx = (wx - sim.ox) / sim.cell;
  const gy = (wy - sim.oy) / sim.cell;
  const x0 = Math.max(add ? 0 : 2, Math.floor(gx - rc));
  const x1 = Math.min(add ? sim.w - 1 : sim.w - 3, Math.ceil(gx + rc));
  const y0 = Math.max(add ? 0 : 2, Math.floor(gy - rc));
  const y1 = Math.min(add ? sim.h - 1 : sim.h - 3, Math.ceil(gy + rc));
  const protect = add ? [
    [sim.pos[0], sim.pos[1], sim.cfg.robot_radius + 0.08],
    ...(state.mode === 'running' ? [[sim.goal[0], sim.goal[1], sim.cfg.goal_radius]] : []),
  ] : [];
  const v = add ? 1 : 0;
  let changed = false;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      if ((x - gx) ** 2 + (y - gy) ** 2 > rc * rc) continue;
      const cwx = sim.ox + x * sim.cell;
      const cwy = sim.oy + y * sim.cell;
      if (protect.some(([px, py, pr]) => (cwx - px) ** 2 + (cwy - py) ** 2 < pr * pr)) continue;
      const i = y * sim.w + x;
      if (sim.occ[i] !== v) { sim.occ[i] = v; changed = true; }
    }
  }
  if (changed) {
    // instant visual feedback on the map bitmap; full rebuild restores shading
    const mctx = state.mapCanvas.getContext('2d');
    mctx.fillStyle = add ? 'rgb(56,65,94)' : 'rgb(14,18,30)';
    mctx.beginPath();
    mctx.arc(gx, gy, rc, 0, 7);
    mctx.fill();
  }
  return changed;
}

function paintSegment(from, to, add) {
  const stepLen = Math.max(state.brushR * 0.5, state.sim.cell);
  const d = Math.hypot(to[0] - from[0], to[1] - from[1]);
  const n = Math.max(1, Math.ceil(d / stepLen));
  let changed = false;
  for (let i = 1; i <= n; i++) {
    changed = paintDisk(from[0] + ((to[0] - from[0]) * i) / n,
      from[1] + ((to[1] - from[1]) * i) / n, add) || changed;
  }
  return changed;
}

// If an erase stroke removed the floor assumptions the robot relied on (or a
// fast drag painted over it), push it back to free space along the EDF gradient.
function unstickRobot() {
  const { sim } = state;
  let [x, y] = sim.pos;
  let d = sim.edfAt(x, y);
  for (let i = 0; i < 50 && d < sim.cfg.robot_radius; i++) {
    const h = sim.cell;
    const dx = sim.edfAt(x + h, y) - sim.edfAt(x - h, y);
    const dy = sim.edfAt(x, y + h) - sim.edfAt(x, y - h);
    const gl = Math.hypot(dx, dy);
    if (gl < 1e-9) break;
    const push = (sim.cfg.robot_radius - d) + 0.5 * sim.cell;
    x += (dx / gl) * push;
    y += (dy / gl) * push;
    d = sim.edfAt(x, y);
  }
  if (d < sim.cfg.robot_radius) {
    const spawn = randomSpawn();
    sim.setState(spawn, spawn);
    state.prevPos = [...spawn];
    state.trail = [];
    state.mode = 'idle';
    state.policy.reset();
    setStatus('robot was buried — respawned. click to set a goal', 'warn');
  } else {
    // shift belief by the same amount so painting doesn't fake odometry info
    sim.odom[0] += x - sim.pos[0];
    sim.odom[1] += y - sim.pos[1];
    sim.pos = [x, y];
  }
}

// Recompute everything derived from occupancy: EDF, reachability, map colors.
function rebuildDerived() {
  const { sim } = state;
  const t0 = performance.now();
  sim.rebuildEDF();
  Object.assign(state, labelComponents(sim));
  state.mapCanvas = renderMapBitmap(sim);
  unstickRobot();
  sim.updateLidar();
  if (state.mode === 'running') {
    const sealed = compAt(sim.goal[0], sim.goal[1]) !== compAt(sim.pos[0], sim.pos[1]);
    if (sealed && !state.sealed) setStatus('goal sealed off — the robot can’t reach it now', 'warn');
    if (!sealed && state.sealed) setStatus('path reopened — navigating…');
    state.sealed = sealed;
  }
  state.rebuildMs = 0.5 * state.rebuildMs + 0.5 * (performance.now() - t0);
  state.lastRebuild = performance.now();
}

// ------------------------------------------------------------------- view

function fitView() {
  const { sim } = state;
  if (!sim) return;
  const dpr = devicePixelRatio || 1;
  const cw = canvas.clientWidth * dpr;
  const ch = canvas.clientHeight * dpr;
  canvas.width = cw;
  canvas.height = ch;
  const wm = sim.w * sim.cell;
  const hm = sim.h * sim.cell;
  const s = Math.min(cw / wm, ch / hm) * 0.94;
  state.view = {
    s,
    x0: (cw - wm * s) / 2 - (sim.ox - sim.cell / 2) * s,
    y0: (ch - hm * s) / 2 - (sim.oy - sim.cell / 2) * s,
  };
}

const W2S = (x, y) => [state.view.x0 + x * state.view.s, state.view.y0 + y * state.view.s];
const S2W = (sx, sy) => {
  const dpr = devicePixelRatio || 1;
  return [(sx * dpr - state.view.x0) / state.view.s, (sy * dpr - state.view.y0) / state.view.s];
};

// The PPO critic learned V(s) ≈ geo(s)/20 + 0.5·γ^(geo/0.15) (progress reward
// plus discounted success bonus; fastnav/ppo.py). Invert it to read the
// policy's own distance-to-go estimate in meters.
function criticToMeters(v) {
  const f = (g) => g / 20 + 0.5 * Math.pow(0.995, g / 0.15);
  if (v <= f(0)) return 0;
  let lo = 0, hi = 200;
  for (let i = 0; i < 50; i++) {
    const mid = (lo + hi) / 2;
    if (f(mid) < v) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

// ------------------------------------------------------------------ physics

function physicsStep() {
  const { sim, policy } = state;
  const t0 = performance.now();
  state.prevPos = [...sim.pos];
  sim.obs(state.obsBuf);
  const act = policy.step(state.obsBuf);
  const { reached } = sim.step(act[0], act[1]);
  state.stepMs = 0.9 * state.stepMs + 0.1 * (performance.now() - t0);
  const sp = Math.hypot(act[0], act[1]);
  if (sp > 0.05) state.heading = [act[0] / sp, act[1] / sp];
  state.trail.push([...sim.pos]);
  if (state.trail.length > 8192) state.trail.shift();
  $('s-steps').textContent = `${sim.stepCount}`;
  $('s-drift').textContent = state.noiseLevel > 0
    ? `${Math.hypot(sim.pos[0] - sim.odom[0], sim.pos[1] - sim.odom[1]).toFixed(2)} m` : '–';
  $('s-val').textContent = `${criticToMeters(policy.value).toFixed(1)} m to go`;
  $('s-ms').textContent = `${state.stepMs.toFixed(2)} ms/step`;
  if (reached) {
    state.mode = 'success';
    state.ripple = 0;
    state.succ++;
    setStatus(`goal reached in ${sim.stepCount} steps (${(sim.stepCount * sim.cfg.dt).toFixed(1)} s sim time) — click for a new goal`, 'success');
    updateScore();
  } else if (sim.stepCount === sim.cfg.max_steps) {
    setStatus(`past the ${sim.cfg.max_steps}-step training horizon — still trying`, 'warn');
  }
}

// --------------------------------------------------------------- rendering

function draw(now) {
  requestAnimationFrame(draw);
  const dt = Math.min((now - state.lastT) / 1000, 0.1);
  state.lastT = now;
  const { sim } = state;
  if (!sim) return;

  if (state.mode === 'running' && !state.paused) {
    state.acc += dt * state.speed;
    const stepDt = sim.cfg.dt;
    let n = 0;
    while (state.acc >= stepDt && n < 64 && state.mode === 'running') {
      state.acc -= stepDt;
      physicsStep();
      n++;
    }
  }
  state.ripple += dt;

  const { s, x0, y0 } = state.view;
  ctx.fillStyle = '#07090f';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(state.mapCanvas,
    x0 + (sim.ox - sim.cell / 2) * s, y0 + (sim.oy - sim.cell / 2) * s,
    sim.w * sim.cell * s, sim.h * sim.cell * s);

  // interpolated robot position for smooth motion between physics steps
  const f = state.mode === 'running' && !state.paused
    ? Math.min(state.acc / sim.cfg.dt, 1) : 1;
  const rx = state.prevPos[0] + (sim.pos[0] - state.prevPos[0]) * f;
  const ry = state.prevPos[1] + (sim.pos[1] - state.prevPos[1]) * f;
  const [prx, pry] = W2S(rx, ry);

  // lidar
  if (state.mode === 'running' || state.mode === 'success') {
    ctx.lineWidth = 1;
    for (let r = 0; r < sim.cfg.n_rays; r++) {
      const th = (2 * Math.PI * r) / sim.cfg.n_rays + sim.odom[2];
      const d = sim.lidar[r];
      const [hx, hy] = W2S(rx + Math.cos(th) * d, ry + Math.sin(th) * d);
      ctx.strokeStyle = 'rgba(76, 201, 240, 0.07)';
      ctx.beginPath();
      ctx.moveTo(prx, pry);
      ctx.lineTo(hx, hy);
      ctx.stroke();
      if (d < sim.cfg.max_range - 1e-3) {
        ctx.fillStyle = 'rgba(76, 201, 240, 0.45)';
        ctx.fillRect(hx - 1, hy - 1, 2, 2);
      }
    }
  }

  // trail
  if (state.trail.length > 1) {
    ctx.strokeStyle = 'rgba(76, 201, 240, 0.35)';
    ctx.lineWidth = Math.max(1.5, 0.05 * s);
    ctx.lineJoin = 'round';
    ctx.beginPath();
    const t0 = Math.max(0, state.trail.length - 2048);
    ctx.moveTo(...W2S(state.trail[t0][0], state.trail[t0][1]));
    for (let i = t0 + 1; i < state.trail.length; i++) {
      ctx.lineTo(...W2S(state.trail[i][0], state.trail[i][1]));
    }
    ctx.lineTo(prx, pry);
    ctx.stroke();
  }

  // goal
  if (state.mode === 'running' || state.mode === 'success') {
    const [gx, gy] = W2S(sim.goal[0], sim.goal[1]);
    const pulse = 1 + 0.15 * Math.sin(now / 300);
    const gr = sim.cfg.goal_radius * s * pulse;
    ctx.strokeStyle = '#ffd166';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(gx, gy, gr, 0, 7);
    ctx.stroke();
    ctx.fillStyle = '#ffd166';
    ctx.beginPath();
    ctx.arc(gx, gy, 3, 0, 7);
    ctx.fill();
    if (state.mode === 'success' && state.ripple < 1) {
      ctx.strokeStyle = `rgba(122, 229, 130, ${1 - state.ripple})`;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(gx, gy, gr + state.ripple * 40, 0, 7);
      ctx.stroke();
    }
  }

  // believed pose (odometry) ghost — only meaningful with noise on
  const rr = Math.max(sim.cfg.robot_radius * s, 4);
  if (state.noiseLevel > 0 && state.mode !== 'idle') {
    const [bx, by] = W2S(sim.odom[0], sim.odom[1]);
    if (Math.hypot(bx - prx, by - pry) > 2) {
      ctx.strokeStyle = 'rgba(200, 160, 255, 0.35)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(prx, pry);
      ctx.lineTo(bx, by);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.strokeStyle = 'rgba(200, 160, 255, 0.85)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.arc(bx, by, rr, 0, 7);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // robot
  ctx.fillStyle = 'rgba(76, 201, 240, 0.18)';
  ctx.beginPath();
  ctx.arc(prx, pry, rr * 1.8, 0, 7);
  ctx.fill();
  ctx.fillStyle = '#4cc9f0';
  ctx.strokeStyle = '#aee8fa';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(prx, pry, rr, 0, 7);
  ctx.fill();
  ctx.stroke();
  ctx.strokeStyle = '#06222e';
  ctx.lineWidth = Math.max(2, rr * 0.3);
  ctx.beginPath();
  ctx.moveTo(prx, pry);
  ctx.lineTo(prx + state.heading[0] * rr * 0.9, pry + state.heading[1] * rr * 0.9);
  ctx.stroke();

  if (state.mode === 'idle') {
    const pulse = 1 + 0.25 * Math.sin(now / 400);
    ctx.strokeStyle = 'rgba(76, 201, 240, 0.5)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(prx, pry, rr * 2.2 * pulse, 0, 7);
    ctx.stroke();
  }

  // brush preview
  if (state.tool !== 'nav' && state.hover) {
    const [hx, hy] = W2S(state.hover[0], state.hover[1]);
    ctx.strokeStyle = state.tool === 'wall' ? 'rgba(255, 209, 102, 0.8)' : 'rgba(255, 107, 107, 0.8)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(hx, hy, state.brushR * s, 0, 7);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  drawScan();
}

// ------------------------------------------------------- robot's-eye panel

const scanCanvas = $('scan');
const sctx = scanCanvas.getContext('2d');

function arrow(c, x0, y0, x1, y1, color, width) {
  const a = Math.atan2(y1 - y0, x1 - x0);
  const hl = Math.max(5, width * 2.5);
  c.strokeStyle = color;
  c.fillStyle = color;
  c.lineWidth = width;
  c.beginPath();
  c.moveTo(x0, y0);
  c.lineTo(x1, y1);
  c.stroke();
  c.beginPath();
  c.moveTo(x1, y1);
  c.lineTo(x1 - hl * Math.cos(a - 0.45), y1 - hl * Math.sin(a - 0.45));
  c.lineTo(x1 - hl * Math.cos(a + 0.45), y1 - hl * Math.sin(a + 0.45));
  c.closePath();
  c.fill();
}

// The policy input, drawn geometrically in the believed frame: ray r sits at
// its indexed angle 2πr/R (heading error is invisible to the robot), the goal
// arrow is goal − odom. Orientation matches the map for easy comparison.
function drawScan() {
  const { sim, policy } = state;
  if (!sim) return;
  const dpr = devicePixelRatio || 1;
  const cw = Math.round(scanCanvas.clientWidth * dpr);
  const ch = Math.round(scanCanvas.clientHeight * dpr);
  if (scanCanvas.width !== cw || scanCanvas.height !== ch) {
    scanCanvas.width = cw;
    scanCanvas.height = ch;
  }
  sctx.clearRect(0, 0, cw, ch);
  const cx = cw / 2;
  const cy = ch / 2;
  const rad = Math.min(cx, cy) - 6 * dpr;
  const s = rad / sim.cfg.max_range;
  const R = sim.cfg.n_rays;

  // range rings every 2 m
  sctx.strokeStyle = 'rgba(107, 119, 153, 0.25)';
  sctx.lineWidth = 1;
  for (let m = 2; m <= sim.cfg.max_range; m += 2) {
    sctx.beginPath();
    sctx.arc(cx, cy, m * s, 0, 7);
    sctx.stroke();
  }

  // scan polygon + return dots
  sctx.beginPath();
  for (let r = 0; r < R; r++) {
    const th = (2 * Math.PI * r) / R;
    const x = cx + Math.cos(th) * sim.lidar[r] * s;
    const y = cy + Math.sin(th) * sim.lidar[r] * s;
    if (r === 0) sctx.moveTo(x, y);
    else sctx.lineTo(x, y);
  }
  sctx.closePath();
  sctx.fillStyle = 'rgba(76, 201, 240, 0.10)';
  sctx.fill();
  sctx.strokeStyle = 'rgba(76, 201, 240, 0.45)';
  sctx.lineWidth = 1;
  sctx.stroke();
  sctx.fillStyle = 'rgba(76, 201, 240, 0.85)';
  for (let r = 0; r < R; r++) {
    if (sim.lidar[r] >= sim.cfg.max_range - 1e-3) continue; // no return
    const th = (2 * Math.PI * r) / R;
    const x = cx + Math.cos(th) * sim.lidar[r] * s;
    const y = cy + Math.sin(th) * sim.lidar[r] * s;
    sctx.fillRect(x - dpr, y - dpr, 2 * dpr, 2 * dpr);
  }

  // believed goal vector (clipped to panel), with distance readout
  const gx = sim.goal[0] - sim.odom[0];
  const gy = sim.goal[1] - sim.odom[1];
  const gd = Math.hypot(gx, gy);
  if (state.mode !== 'idle' && gd > 1e-6) {
    const cl = Math.min(gd, sim.cfg.max_range) * s;
    arrow(sctx, cx, cy, cx + (gx / gd) * cl, cy + (gy / gd) * cl, '#ffd166', 1.5 * dpr);
    sctx.fillStyle = '#ffd166';
    sctx.font = `${10 * dpr}px ui-monospace, monospace`;
    const tx = cx + (gx / gd) * cl * 0.72;
    const ty = cy + (gy / gd) * cl * 0.72;
    sctx.fillText(`${gd.toFixed(1)}m`, tx + 4 * dpr, ty - 4 * dpr);
  }

  // previous executed action (part of the input), scaled to half-radius at v_max
  const [ax, ay] = policy.prev;
  const an = Math.hypot(ax, ay);
  if (an > 0.02) {
    const as = (an / sim.cfg.v_max) * rad * 0.5;
    arrow(sctx, cx, cy, cx + (ax / an) * as, cy + (ay / an) * as, '#4cc9f0', 2 * dpr);
  }

  // robot
  sctx.fillStyle = '#aee8fa';
  sctx.beginPath();
  sctx.arc(cx, cy, 2.5 * dpr, 0, 7);
  sctx.fill();
}

// ------------------------------------------------------------------ UI

let statusTimer = null;
function setStatus(msg, cls = '') {
  const el = $('status');
  el.textContent = msg;
  el.className = cls;
  el.style.opacity = 1;
  clearTimeout(statusTimer);
  if (cls === 'error') {
    statusTimer = setTimeout(() => { el.style.opacity = 0.25; }, 2500);
  }
}

function updateScore() {
  $('s-succ').textContent = `${state.succ}/${state.tries}`;
}

function eventWorld(ev) {
  const rect = canvas.getBoundingClientRect();
  return S2W(ev.clientX - rect.left, ev.clientY - rect.top);
}

function onPointerDown(ev) {
  const { sim } = state;
  if (!sim) return;
  const [wx, wy] = eventWorld(ev);

  if (state.tool !== 'nav' && !ev.shiftKey) {
    state.painting = true;
    state.lastPaint = [wx, wy];
    canvas.setPointerCapture(ev.pointerId);
    paintDisk(wx, wy, state.tool === 'wall');
    return;
  }

  const c = compAt(wx, wy);
  const clear = sim.edfAt(wx, wy);

  if (ev.shiftKey) {
    if (c === -1 || clear < sim.cfg.robot_radius + 0.02) {
      setStatus('cannot teleport there — blocked', 'error');
      return;
    }
    sim.setState([wx, wy], sim.pos);
    state.prevPos = [wx, wy];
    state.trail = [];
    state.policy.reset();
    state.mode = 'idle';
    setStatus('robot moved — click to set a goal');
    return;
  }

  const robotComp = compAt(sim.pos[0], sim.pos[1]);
  if (c !== robotComp || clear < sim.cfg.robot_radius + 0.02) {
    setStatus(c !== robotComp && c !== -1
      ? 'that spot is sealed off from the robot' : 'blocked — pick open floor', 'error');
    return;
  }
  sim.goal = [wx, wy];
  sim.newEpisode();       // re-anchor odometry, resample per-episode noise
  state.policy.reset();   // new episode: hidden state + prev action reset
  state.acc = 0;
  state.mode = 'running';
  state.sealed = false;
  state.tries++;
  updateScore();
  setStatus('navigating…');
}

function onPointerMove(ev) {
  if (!state.sim) return;
  const [wx, wy] = eventWorld(ev);
  state.hover = [wx, wy];
  if (!state.painting) return;
  paintSegment(state.lastPaint, [wx, wy], state.tool === 'wall');
  state.lastPaint = [wx, wy];
  // keep physics honest during long strokes without re-running EDT every event
  if (performance.now() - state.lastRebuild > Math.max(120, 3 * state.rebuildMs)) {
    rebuildDerived();
  }
}

function onPointerUp() {
  if (!state.painting) return;
  state.painting = false;
  state.lastPaint = null;
  rebuildDerived();
}

async function main() {
  const [manifest, index] = await Promise.all([
    loadManifest('.'),
    fetch('scene_index.json').then((r) => r.json()),
  ]);
  state.manifest = manifest;
  state.policy = await loadPolicy(manifest, manifest.policies[0].id, '.');
  state.policyBins[manifest.policies[0].id] = state.policy;
  state.index = index;
  state.obsBuf = new Float32Array(state.policy.obsDim);

  const psel = $('policy');
  for (const p of manifest.policies) {
    const o = document.createElement('option');
    o.value = p.id;
    o.textContent = p.label;
    psel.appendChild(o);
  }
  psel.onchange = async () => {
    if (!state.policyBins[psel.value]) {
      state.policyBins[psel.value] = await loadPolicy(manifest, psel.value, '.');
    }
    state.policy = state.policyBins[psel.value];
    state.policy.reset();
    if (state.sim && state.mode === 'running') state.sim.newEpisode(); // restart episode cleanly
  };

  for (const b of $('noise').querySelectorAll('button')) {
    b.onclick = () => {
      state.noiseLevel = parseFloat(b.dataset.x);
      $('noise').querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
      if (state.sim) state.sim.setNoise(manifest.noise_stack, state.noiseLevel);
      $('s-drift').textContent = state.noiseLevel > 0 ? '0.00 m' : '–';
    };
  }

  const sel = $('scene');
  for (const group of [...new Set(index.map((m) => m.group))]) {
    const og = document.createElement('optgroup');
    og.label = group;
    for (const m of index.filter((x) => x.group === group)) {
      const o = document.createElement('option');
      o.value = m.name;
      o.textContent = m.name.replace('ProcTHOR-r1-ProcTHOR-', '').replace('ProcTHOR-rc-ProcTHOR-', '').replace('Baked_', '');
      og.appendChild(o);
    }
    sel.appendChild(og);
  }
  sel.onchange = () => setScene(sel.value);
  $('shuffle').onclick = () => {
    const m = index[(Math.random() * index.length) | 0];
    sel.value = m.name;
    setScene(m.name);
  };
  $('pause').onclick = () => {
    state.paused = !state.paused;
    $('pause').textContent = state.paused ? '▶ resume' : '⏸ pause';
  };
  for (const b of $('speed').querySelectorAll('button')) {
    b.onclick = () => {
      state.speed = parseFloat(b.dataset.x);
      $('speed').querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
    };
  }
  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointercancel', onPointerUp);
  canvas.addEventListener('pointerleave', () => { state.hover = null; });

  for (const b of $('tool').querySelectorAll('button')) {
    b.onclick = () => {
      state.tool = b.dataset.t;
      $('tool').querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
    };
  }
  $('brush').oninput = () => { state.brushR = parseFloat($('brush').value); };
  $('resetmap').onclick = () => {
    if (!state.sim) return;
    state.sim.occ.set(state.occBase);
    rebuildDerived();
    setStatus('scene restored');
  };

  new ResizeObserver(fitView).observe($('stage'));

  window.fastnav = state; // debug/test hook
  $('loading').remove();
  const first = index[(Math.random() * index.length) | 0];
  sel.value = first.name;
  await setScene(first.name);
  requestAnimationFrame((t) => { state.lastT = t; requestAnimationFrame(draw); });
}

main().catch((e) => {
  $('status').textContent = `failed to load: ${e.message}`;
  console.error(e);
});
