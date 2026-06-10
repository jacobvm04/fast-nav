// Exact signed Euclidean distance field from a binary occupancy grid.
// Felzenszwalb & Huttenlocher 2-pass squared EDT — matches
// scipy.ndimage.distance_transform_edt, which produced the training EDFs.

const INF = 1e20;

// 1D lower-envelope distance transform on squared distances, in place into d.
function dt1d(f, d, v, z, n) {
  let k = 0;
  v[0] = 0;
  z[0] = -INF;
  z[1] = INF;
  for (let q = 1; q < n; q++) {
    let s;
    for (;;) {
      const p = v[k];
      s = (f[q] + q * q - (f[p] + p * p)) / (2 * q - 2 * p);
      if (s <= z[k]) k--;
      else break;
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = INF;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    const p = v[k];
    d[q] = (q - p) * (q - p) + f[p];
  }
}

// Squared distance from every cell to the nearest cell where mask[i] != 0.
function edt2dSq(mask, h, w) {
  const g = new Float64Array(h * w);
  for (let i = 0; i < h * w; i++) g[i] = mask[i] ? 0 : INF;
  const n = Math.max(h, w);
  const f = new Float64Array(n);
  const d = new Float64Array(n);
  const v = new Int32Array(n);
  const z = new Float64Array(n + 1);
  for (let x = 0; x < w; x++) {           // columns
    for (let y = 0; y < h; y++) f[y] = g[y * w + x];
    dt1d(f, d, v, z, h);
    for (let y = 0; y < h; y++) g[y * w + x] = d[y];
  }
  for (let y = 0; y < h; y++) {           // rows
    for (let x = 0; x < w; x++) f[x] = g[y * w + x];
    dt1d(f, d, v, z, w);
    for (let x = 0; x < w; x++) g[y * w + x] = d[x];
  }
  return g;
}

// occ: Uint8Array [h*w], 1 = occupied. Returns Float32Array signed distance in
// meters: positive in free space, negative inside obstacles.
export function signedEDF(occ, h, w, cell) {
  const free = new Uint8Array(h * w);
  for (let i = 0; i < h * w; i++) free[i] = occ[i] ? 0 : 1;
  const dOut = edt2dSq(occ, h, w);   // dist to occupied (0 inside obstacles)
  const dIn = edt2dSq(free, h, w);   // dist to free (0 in free space)
  const edf = new Float32Array(h * w);
  for (let i = 0; i < h * w; i++) {
    edf[i] = (Math.sqrt(dOut[i]) - Math.sqrt(dIn[i])) * cell;
  }
  return edf;
}
