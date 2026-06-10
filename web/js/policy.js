// JavaScript port of fastnav.policy.RecurrentNavPolicy.step (MLX):
//   x = silu(enc(obs_prev * scale)); h = GRU(x, h); act = v_max * tanh(head(h))
// GRU gate order in the stacked [3H] dim is (r, z, n), with the extra bhn bias
// applied inside the reset gate, exactly as mlx.nn.GRU computes it.

function matvec(out, W, x, nOut, nIn, bias) {
  for (let i = 0; i < nOut; i++) {
    let s = bias ? bias[i] : 0;
    const row = i * nIn;
    for (let j = 0; j < nIn; j++) s += W[row + j] * x[j];
    out[i] = s;
  }
}

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

export class Policy {
  // manifest: parsed policy.json; buf: ArrayBuffer of policy.bin
  constructor(manifest, buf) {
    const f32 = new Float32Array(buf);
    const t = {};
    for (const [name, { shape, offset }] of Object.entries(manifest.tensors)) {
      const size = shape.reduce((a, b) => a * b, 1);
      t[name] = f32.subarray(offset, offset + size);
    }
    this.t = t;
    this.simCfg = manifest.sim;
    this.valScale = manifest.val_scale;
    const cfg = manifest.sim;
    this.H = manifest.arch.hidden;
    this.E = manifest.arch.enc;
    this.nRays = manifest.n_rays ?? cfg.n_rays; // per-policy lidar resolution
    this.obsDim = this.nRays + 4;
    this.inDim = this.obsDim + 2; // obs | prev action

    // observation normalization baked into the policy (policy.py)
    const scale = new Float32Array(this.inDim);
    for (let i = 0; i < this.nRays + 2; i++) scale[i] = 1 / cfg.max_range;
    const posScale = manifest.arch.use_pos ? 0.1 : 0.0;
    scale[this.nRays + 2] = posScale;
    scale[this.nRays + 3] = posScale;
    scale[this.obsDim] = 1 / cfg.v_max;
    scale[this.obsDim + 1] = 1 / cfg.v_max;
    this.scale = scale;

    this.h = new Float32Array(this.H);
    this.prev = new Float32Array(2);
    this.x = new Float32Array(this.inDim);
    this.xe = new Float32Array(this.E);
    this.xg = new Float32Array(3 * this.H);
    this.hg = new Float32Array(3 * this.H);
    this.value = 0; // raw critic output after each step()
  }

  reset() {
    this.h.fill(0);
    this.prev.fill(0);
    this.value = 0;
  }

  // obs: Float32Array [obsDim]. Returns [vx, vy]; updates hidden + prev action.
  step(obs) {
    const { t, H, E, x, xe, xg, hg, h } = this;
    for (let i = 0; i < this.obsDim; i++) x[i] = obs[i] * this.scale[i];
    x[this.obsDim] = this.prev[0] * this.scale[this.obsDim];
    x[this.obsDim + 1] = this.prev[1] * this.scale[this.obsDim + 1];

    matvec(xe, t['enc.weight'], x, E, this.inDim, t['enc.bias']);
    for (let i = 0; i < E; i++) xe[i] = xe[i] * sigmoid(xe[i]); // silu

    matvec(xg, t['gru.Wx'], xe, 3 * H, E, t['gru.b']);
    matvec(hg, t['gru.Wh'], h, 3 * H, H, null);
    const bhn = t['gru.bhn'];
    for (let i = 0; i < H; i++) {
      const r = sigmoid(xg[i] + hg[i]);
      const z = sigmoid(xg[H + i] + hg[H + i]);
      const n = Math.tanh(xg[2 * H + i] + r * (hg[2 * H + i] + bhn[i]));
      h[i] = (1 - z) * n + z * h[i];
    }

    const vmax = this.simCfg.v_max;
    const hw = t['head.weight'];
    const hb = t['head.bias'];
    let a0 = hb[0], a1 = hb[1];
    let v = t['vhead.bias'][0];
    const vw = t['vhead.weight'];
    for (let i = 0; i < H; i++) {
      a0 += hw[i] * h[i];
      a1 += hw[H + i] * h[i];
      v += vw[i] * h[i];
    }
    const act = [vmax * Math.tanh(a0), vmax * Math.tanh(a1)];
    this.prev[0] = act[0];
    this.prev[1] = act[1];
    this.value = v;
    return act;
  }
}

// policies.json holds shared sim config + per-policy tensor manifests.
export async function loadManifest(baseUrl = '.') {
  return fetch(`${baseUrl}/policies.json`).then((r) => r.json());
}

export async function loadPolicy(manifest, id, baseUrl = '.') {
  const entry = manifest.policies.find((p) => p.id === id) || manifest.policies[0];
  const bin = await fetch(`${baseUrl}/${entry.file}`).then((r) => r.arrayBuffer());
  return new Policy({ ...entry, sim: manifest.sim, val_scale: manifest.val_scale }, bin);
}
