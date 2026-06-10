// Replays the MLX-generated fixture trajectory through the JS sim + policy and
// reports divergence. Run from repo root:  node web/test/parity.mjs
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { Policy } from '../js/policy.js';
import { Sim } from '../js/sim.js';

const here = dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(readFileSync(join(here, 'fixture.json'), 'utf8'));
const manifest = JSON.parse(readFileSync(join(here, '../policy.json'), 'utf8'));
const bin = readFileSync(join(here, '../policy.bin'));

const occ = new Uint8Array(Buffer.from(fixture.occupancy_b64, 'base64'));
const sim = new Sim(occ, fixture.h, fixture.w, fixture.cell, fixture.origin, manifest.sim);
const policy = new Policy(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));

// 1. EDF parity vs scipy (row 128 of the training-time field)
const refBuf = Buffer.from(fixture.edf_row128_b64, 'base64');
const refRow = new Float32Array(refBuf.buffer, refBuf.byteOffset, refBuf.byteLength / 4);
const y = Math.min(128, fixture.h - 1);
let edfErr = 0;
for (let x = 0; x < fixture.w; x++) {
  edfErr = Math.max(edfErr, Math.abs(sim.edf[y * fixture.w + x] - refRow[x]));
}
console.log(`EDF max |err| on row ${y}: ${edfErr.toExponential(2)} m`);

// 2. Closed-loop rollout from the same state
sim.setState(fixture.start, fixture.goal);
sim.goal = [fixture.goal[0], fixture.goal[1]];
policy.reset();

const obs = new Float32Array(manifest.sim.n_rays + 4);
let maxLidarErr0 = 0;
let maxPosErr = 0;
let reachedAt = -1;
for (let t = 0; t < fixture.steps; t++) {
  const ref = fixture.traj[t];
  const posErr = Math.hypot(sim.pos[0] - ref.pos[0], sim.pos[1] - ref.pos[1]);
  maxPosErr = Math.max(maxPosErr, posErr);
  if (t === 0) {
    for (let r = 0; r < manifest.sim.n_rays; r++) {
      maxLidarErr0 = Math.max(maxLidarErr0, Math.abs(sim.lidar[r] - ref.lidar[r]));
    }
  }
  sim.obs(obs);
  const act = policy.step(obs);
  if (t === 0) {
    console.log(`step 0: lidar max |err| ${maxLidarErr0.toExponential(2)} m, ` +
      `action err [${Math.abs(act[0] - ref.act[0]).toExponential(2)}, ` +
      `${Math.abs(act[1] - ref.act[1]).toExponential(2)}] m/s`);
  }
  const { reached } = sim.step(act[0], act[1]);
  if (reached) { reachedAt = t + 1; break; }
}
console.log(`MLX reached goal at step ${fixture.steps}; JS at step ${reachedAt}`);
console.log(`max position divergence over rollout: ${maxPosErr.toFixed(4)} m`);

const ok = edfErr < 1e-4 && maxLidarErr0 < 1e-3 && reachedAt > 0 &&
  Math.abs(reachedAt - fixture.steps) <= 5 && maxPosErr < 0.25;
console.log(ok ? 'PARITY OK' : 'PARITY FAILED');
process.exit(ok ? 0 : 1);
