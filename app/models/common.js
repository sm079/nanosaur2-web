// Shared helpers: rotary tables (cos block followed by sin block, each [L][D/2]) for the
// split-half RoPE convention, where pair i is (x[i], x[i + D/2]).

function table(gpu, angles, L, half) {
  const cs = new Float32Array(2 * L * half);
  for (let i = 0; i < L * half; i++) {
    cs[i] = Math.cos(angles[i]);
    cs[L * half + i] = Math.sin(angles[i]);
  }
  const t = gpu.fromArray(cs, [cs.length]);
  t.sinOff = L * half;
  return t;
}

const invFreq = (D, theta) => Array.from({ length: D / 2 }, (_, i) => Math.fround(1 / Math.pow(theta, Math.fround((2 * i) / D))));

// Standard 1D rope: angle(l, i) = l / theta^(2i/D)
export function ropeTable(gpu, L, D, theta) {
  const half = D / 2;
  const f = invFreq(D, theta);
  const a = new Float32Array(L * half);
  for (let l = 0; l < L; l++) for (let i = 0; i < half; i++) a[l * half + i] = l * f[i];
  return table(gpu, a, L, half);
}

// Nanosaur2 2D rope over centered token coordinates: the first D/4 pairs rotate by the column,
// the next D/4 by the row, each with the frequencies of a D/2-dim axis (theta 10000).
export function rope2dTable(gpu, H, W, D) {
  const half = D / 2;
  const f = invFreq(half, 10000);
  const a = new Float32Array(H * W * half);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const o = (y * W + x) * half;
      const cx = x - (W - 1) / 2, cy = y - (H - 1) / 2;
      for (let i = 0; i < half / 2; i++) {
        a[o + i] = cx * f[i];
        a[o + half / 2 + i] = cy * f[i];
      }
    }
  }
  return table(gpu, a, H * W, half);
}
