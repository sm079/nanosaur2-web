// WGSL kernel generators. Every kernel binds its parameters as a uniform at binding 0.

const cache = new Map();
const memo = (key, fn) => {
  if (!cache.has(key)) cache.set(key, fn());
  return cache.get(key);
};

const ACT = /* wgsl */ `
fn silu(x: f32) -> f32 { return x / (1.0 + exp(-x)); }
// tanh-approximated GELU (PyTorch approximate="tanh"); tanh written out so large |x| can't overflow
fn gelu_tanh(x: f32) -> f32 {
  let u = 0.7978845608028654 * (x + 0.044715 * x * x * x);
  let e = exp(-2.0 * abs(u));
  let t = sign(u) * (1.0 - e) / (1.0 + e);
  return 0.5 * x * (1.0 + t);
}
`;

// GEMM lives in gemm.js
export { matmulShader } from "./gemm.js";

// ----------------------------------------------------------------------------- row kernels

// Row-wise softmax in place: S[row][0..cols) * scale, optional causal mask (col > row % rowsPerHead).
export const softmaxShader = () => memo("softmax", () => /* wgsl */ `
struct Params { rows: u32, cols: u32, causal: u32, rowsPerHead: u32, nx: u32, scale: f32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> S: array<f32>;
var<workgroup> red: array<f32, 256>;
fn reduceMax(t: u32, v: f32) -> f32 {
  red[t] = v; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { red[t] = max(red[t], red[t + s]); } workgroupBarrier(); }
  let r = red[0]; workgroupBarrier(); return r;
}
fn reduceSum(t: u32, v: f32) -> f32 {
  red[t] = v; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { red[t] = red[t] + red[t + s]; } workgroupBarrier(); }
  let r = red[0]; workgroupBarrier(); return r;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let row = wg.y * P.nx + wg.x;
  if (row >= P.rows) { return; }
  let base = row * P.cols;
  var limit = P.cols;
  if (P.causal == 1u) { limit = min(P.cols, row % P.rowsPerHead + 1u); }
  var mx = -3.0e38;
  for (var c = t; c < limit; c += 256u) { mx = max(mx, S[base + c] * P.scale); }
  mx = reduceMax(t, mx);
  var sm = 0.0;
  for (var c = t; c < limit; c += 256u) { let e = exp(S[base + c] * P.scale - mx); S[base + c] = e; sm += e; }
  sm = reduceSum(t, sm);
  let inv = 1.0 / sm;
  for (var c = t; c < P.cols; c += 256u) {
    if (c < limit) { S[base + c] = S[base + c] * inv; } else { S[base + c] = 0.0; }
  }
}`);

// RMSNorm over rows of length `cols` with a weight vector of length `cols`:
//   y = x * rsqrt(mean(x^2) + eps) * w
export const rmsnormShader = () => memo("rms", () => /* wgsl */ `
struct Params { rows: u32, cols: u32, nx: u32, eps: f32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> Wt: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let row = wg.y * P.nx + wg.x;
  if (row >= P.rows) { return; }
  let base = row * P.cols;
  var ss = 0.0;
  for (var c = t; c < P.cols; c += 64u) { let v = X[base + c]; ss += v * v; }
  red[t] = ss; workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) { if (t < s) { red[t] = red[t] + red[t + s]; } workgroupBarrier(); }
  let r = inverseSqrt(red[0] / f32(P.cols) + P.eps);
  for (var c = t; c < P.cols; c += 64u) { Y[base + c] = X[base + c] * r * Wt[c]; }
}`);

// Norm followed by adaLN modulation: y = norm(x) * (1 + scale) + shift, with scale and shift
// read from MOD at the given element offsets.
//   rms = false: LayerNorm without affine
//   rms = true:  RMSNorm with a weight vector, y = x * rsqrt(mean(x^2) + eps) * w
export const normModShader = (rms) => memo("nmod" + rms, () => /* wgsl */ `
struct Params { rows: u32, cols: u32, shiftOff: u32, scaleOff: u32, nx: u32, eps: f32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> MOD: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
${rms ? "@group(0) @binding(4) var<storage, read> Wt: array<f32>;" : ""}
var<workgroup> red: array<f32, 256>;
fn rsum(t: u32, v: f32) -> f32 {
  red[t] = v; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { red[t] = red[t] + red[t + s]; } workgroupBarrier(); }
  let r = red[0]; workgroupBarrier(); return r;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let row = wg.y * P.nx + wg.x;
  if (row >= P.rows) { return; }
  let base = row * P.cols;
  ${rms ? `
  let mean = 0.0;
  var v = 0.0;
  for (var c = t; c < P.cols; c += 256u) { let x = X[base + c]; v += x * x; }` : `
  var s = 0.0;
  for (var c = t; c < P.cols; c += 256u) { s += X[base + c]; }
  let mean = rsum(t, s) / f32(P.cols);
  var v = 0.0;
  for (var c = t; c < P.cols; c += 256u) { let d = X[base + c] - mean; v += d * d; }`}
  let r = inverseSqrt(rsum(t, v) / f32(P.cols) + P.eps);
  for (var c = t; c < P.cols; c += 256u) {
    Y[base + c] = (X[base + c] - mean) * r${rms ? " * Wt[c]" : ""} * (1.0 + MOD[P.scaleOff + c]) + MOD[P.shiftOff + c];
  }
}`);

// Fused ("flash") attention, one head x 64 queries per workgroup of 128 threads, keys streamed in
// blocks of 32 with an online softmax; nothing proportional to Lq*Lk touches memory.
// Head dim D: a multiple of 32 up to 128 (64, 96, 128). 16 KB of workgroup memory, as a vec4
// array SH[1024]:
//   [0, 512)    P    probabilities of the current key block, [32 keys][64 rows]
//   [512, 768)  Qs   Q chunk [16 dims][64 rows]        (phase 1)
//   [768, 896)  Ks   K chunk [16 dims][32 keys]        (phase 1), then row-sum partials
//   [896, 1024) row-max partials [64 rows][8]
//   [512, 1024) Vs   V chunk [32 keys][64 or 32 dims]  (phase 3, reuses the above)
// Thread (sr = t / 8, sc = t % 8) owns rows sr*4..+3, keys sc*4..+3 of the score tile and
// output columns c*32 + sc*4..+3 for c < D/32; all register arrays use constant indices only.
// bias: an additive score bias per key, KB[key - kbStart] for keys >= kbStart (keys before
// kbStart get none), e.g. log prompt weights on the text keys of joint image+text attention.
export const flashAttentionShader = (D, bias = false) => memo(`flash${D}${bias}`, () => {
  if (D % 32 || D > 128) throw new Error(`flash attention: unsupported head dim ${D}`);
  const DC = D / 16; // phase-1 dim chunks
  const NO = D / 32; // output vec4 columns per row
  // phase-3 value chunks: 64 dims (two output columns) while they fit, then one of 32
  const chunks = [];
  for (let c = 0; c < NO; c += 2) chunks.push(c + 1 < NO ? { col: c, w: 64 } : { col: c, w: 32 });
  const r4 = [0, 1, 2, 3];
  const o = [];
  for (let i = 0; i < 4; i++) for (let c = 0; c < NO; c++) o.push(`var o${i}_${c} = vec4<f32>();`);
  const rescale = [];
  for (let i = 0; i < 4; i++) for (let c = 0; c < NO; c++) rescale.push(`o${i}_${c} = o${i}_${c} * alpha.${"xyzw"[i]};`);
  const phase3 = (ch) => {
    const fma = [];
    for (let i = 0; i < 4; i++) {
      fma.push(`o${i}_${ch.col} += pv.${"xyzw"[i]} * v0;`);
      if (ch.w === 64) fma.push(`o${i}_${ch.col + 1} += pv.${"xyzw"[i]} * v1;`);
    }
    const dimOff = ch.col * 32;
    if (ch.w === 64) {
      return `{
      let kr = t / 4u;
      let d16 = (t % 4u) * 16u;
      var a = vec4<f32>(); var b = vec4<f32>(); var c = vec4<f32>(); var e = vec4<f32>();
      if (k0 + kr < P.Lk) {
        let gi = ((k0 + kr) * P.ldv + P.vOff + h * ${D}u + ${dimOff}u + d16) >> 2u;
        a = V[gi]; b = V[gi + 1u]; c = V[gi + 2u]; e = V[gi + 3u];
      }
      let base = 512u + kr * 16u + d16 / 4u;
      SH[base] = a; SH[base + 1u] = b; SH[base + 2u] = c; SH[base + 3u] = e;
      workgroupBarrier();
      for (var k = 0u; k < 32u; k++) {
        let pv = SH[k * 16u + sr];
        let v0 = SH[512u + k * 16u + sc];
        let v1 = SH[512u + k * 16u + 8u + sc];
        ${fma.join("\n        ")}
      }
      workgroupBarrier();
    }`;
    }
    return `{
      let kr = t / 4u;
      let d8 = (t % 4u) * 8u;
      var a = vec4<f32>(); var b = vec4<f32>();
      if (k0 + kr < P.Lk) {
        let gi = ((k0 + kr) * P.ldv + P.vOff + h * ${D}u + ${dimOff}u + d8) >> 2u;
        a = V[gi]; b = V[gi + 1u];
      }
      let base = 512u + kr * 8u + d8 / 4u;
      SH[base] = a; SH[base + 1u] = b;
      workgroupBarrier();
      for (var k = 0u; k < 32u; k++) {
        let pv = SH[k * 16u + sr];
        let v0 = SH[512u + k * 8u + sc];
        ${fma.join("\n        ")}
      }
      workgroupBarrier();
    }`;
  };
  const store = [];
  for (let i = 0; i < 4; i++) {
    store.push(`{ let q = q0 + sr * 4u + ${i}u; if (q < P.Lq) { let inv = 1.0 / l.${"xyzw"[i]}; let ob = q * P.ldo + P.oOff + h * ${D}u;`);
    for (let c = 0; c < NO; c++) {
      store.push(`  { let v = o${i}_${c} * inv; let b = ob + ${c * 32}u + sc * 4u; O[b] = v.x; O[b + 1u] = v.y; O[b + 2u] = v.z; O[b + 3u] = v.w; }`);
    }
    store.push("} }");
  }
  const keyBias = bias ? `
    {
      let kk = vec4<u32>(k0 + sc * 4u) + vec4<u32>(0u, 1u, 2u, 3u);
      var kb = vec4<f32>(0.0);
      for (var j = 0u; j < 4u; j++) {
        if (kk[j] >= P.kbStart && kk[j] < P.Lk) { kb[j] = KB[kk[j] - P.kbStart]; }
      }
      s0 += kb; s1 += kb; s2 += kb; s3 += kb;
    }` : "";
  return /* wgsl */ `
struct Params { Lq: u32, Lk: u32, ldq: u32, ldk: u32, ldv: u32, ldo: u32, qOff: u32, kOff: u32, vOff: u32, oOff: u32, scale: f32, kbStart: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> K: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
${bias ? "@group(0) @binding(5) var<storage, read> KB: array<f32>;" : ""}
var<workgroup> SH: array<vec4<f32>, 1024>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let h = wg.y;
  let q0 = wg.x * 64u;
  let sr = t / 8u;
  let sc = t % 8u;
  ${o.join("\n  ")}
  var m = vec4<f32>(-1e30);
  var l = vec4<f32>(0.0);

  for (var k0 = 0u; k0 < P.Lk; k0 += 32u) {
    // ---- phase 1: S = Q K^T for the 64 x 32 tile
    var s0 = vec4<f32>(); var s1 = vec4<f32>(); var s2 = vec4<f32>(); var s3 = vec4<f32>();
    for (var dc = 0u; dc < ${DC}u; dc++) {
      {
        // Q chunk: row t/2, dims (t%2)*8..+7 of this chunk
        let qr = t / 2u;
        let d8 = (t % 2u) * 8u;
        var a = vec4<f32>(); var b = vec4<f32>();
        if (q0 + qr < P.Lq) {
          let gi = ((q0 + qr) * P.ldq + P.qOff + h * ${D}u + dc * 16u + d8) >> 2u;
          a = Q[gi]; b = Q[gi + 1u];
        }
        let lane = qr & 3u;
        let col = qr >> 2u;
        SH[512u + (d8 + 0u) * 16u + col][lane] = a.x; SH[512u + (d8 + 1u) * 16u + col][lane] = a.y;
        SH[512u + (d8 + 2u) * 16u + col][lane] = a.z; SH[512u + (d8 + 3u) * 16u + col][lane] = a.w;
        SH[512u + (d8 + 4u) * 16u + col][lane] = b.x; SH[512u + (d8 + 5u) * 16u + col][lane] = b.y;
        SH[512u + (d8 + 6u) * 16u + col][lane] = b.z; SH[512u + (d8 + 7u) * 16u + col][lane] = b.w;
      }
      {
        // K chunk: key t/4, dims (t%4)*4..+3
        let kr = t / 4u;
        let d4 = (t % 4u) * 4u;
        var a = vec4<f32>();
        if (k0 + kr < P.Lk) { a = K[((k0 + kr) * P.ldk + P.kOff + h * ${D}u + dc * 16u + d4) >> 2u]; }
        let lane = kr & 3u;
        let col = kr >> 2u;
        SH[768u + (d4 + 0u) * 8u + col][lane] = a.x; SH[768u + (d4 + 1u) * 8u + col][lane] = a.y;
        SH[768u + (d4 + 2u) * 8u + col][lane] = a.z; SH[768u + (d4 + 3u) * 8u + col][lane] = a.w;
      }
      workgroupBarrier();
      for (var d = 0u; d < 16u; d++) {
        let qa = SH[512u + d * 16u + sr];
        let kb = SH[768u + d * 8u + sc];
        s0 += qa.x * kb; s1 += qa.y * kb; s2 += qa.z * kb; s3 += qa.w * kb;
      }
      workgroupBarrier();
    }

    // ---- phase 2: online softmax
    let kmask = vec4<f32>(select(vec4<f32>(0.0), vec4<f32>(-1e30),
      vec4<u32>(k0 + sc * 4u) + vec4<u32>(0u, 1u, 2u, 3u) >= vec4<u32>(P.Lk)));
    s0 = s0 * P.scale + kmask; s1 = s1 * P.scale + kmask; s2 = s2 * P.scale + kmask; s3 = s3 * P.scale + kmask;${keyBias}
    let pm = vec4<f32>(max(max(s0.x, s0.y), max(s0.z, s0.w)), max(max(s1.x, s1.y), max(s1.z, s1.w)),
                       max(max(s2.x, s2.y), max(s2.z, s2.w)), max(max(s3.x, s3.y), max(s3.z, s3.w)));
    // row-max partials: rows sr*4+i, slot sc  -> float index (sr*4+i)*8 + sc
    ${r4.map((i) => `SH[896u + ((sr * 4u + ${i}u) * 8u + sc) / 4u][sc % 4u] = pm.${"xyzw"[i]};`).join("\n    ")}
    workgroupBarrier();
    var bm = vec4<f32>(-1e30);
    ${r4.map((i) => `{ let a = SH[896u + (sr * 4u + ${i}u) * 2u]; let b = SH[896u + (sr * 4u + ${i}u) * 2u + 1u];
      bm.${"xyzw"[i]} = max(max(max(a.x, a.y), max(a.z, a.w)), max(max(b.x, b.y), max(b.z, b.w))); }`).join("\n    ")}
    let mn = max(m, bm);
    let e0 = exp(s0 - mn.x); let e1 = exp(s1 - mn.y); let e2 = exp(s2 - mn.z); let e3 = exp(s3 - mn.w);
    let ps = vec4<f32>(dot(e0, vec4<f32>(1.0)), dot(e1, vec4<f32>(1.0)), dot(e2, vec4<f32>(1.0)), dot(e3, vec4<f32>(1.0)));
    ${r4.map((i) => `SH[768u + ((sr * 4u + ${i}u) * 8u + sc) / 4u][sc % 4u] = ps.${"xyzw"[i]};`).join("\n    ")}
    // probabilities, stored [key][row] so phase 3 reads 4 rows as one vec4
    SH[(sc * 4u + 0u) * 16u + sr] = vec4<f32>(e0.x, e1.x, e2.x, e3.x);
    SH[(sc * 4u + 1u) * 16u + sr] = vec4<f32>(e0.y, e1.y, e2.y, e3.y);
    SH[(sc * 4u + 2u) * 16u + sr] = vec4<f32>(e0.z, e1.z, e2.z, e3.z);
    SH[(sc * 4u + 3u) * 16u + sr] = vec4<f32>(e0.w, e1.w, e2.w, e3.w);
    workgroupBarrier();
    var bs = vec4<f32>(0.0);
    ${r4.map((i) => `{ let a = SH[768u + (sr * 4u + ${i}u) * 2u]; let b = SH[768u + (sr * 4u + ${i}u) * 2u + 1u];
      bs.${"xyzw"[i]} = dot(a, vec4<f32>(1.0)) + dot(b, vec4<f32>(1.0)); }`).join("\n    ")}
    let alpha = exp(m - mn);
    l = l * alpha + bs;
    m = mn;
    ${rescale.join("\n    ")}
    workgroupBarrier(); // partials consumed before the V chunk overwrites them

    // ---- phase 3: O += P V, V streamed in 64- or 32-wide chunks
    ${chunks.map(phase3).join("\n    ")}
  }

  ${store.join("\n  ")}
}`;
});

// Per-head RMSNorm (+ split-half RoPE), in place on heads stored inside wider rows: head h of
// row l is X[(rowOff + l) * ld + colOff + h * D ..][D]. One workgroup of 64 threads per
// (row, head). With rope, thread i owns the rope pairs (i, i + D/2) and the rotation for row l
// is CS[l][D/2] cos, then sin at sinOff.
export const headNormShader = (D, rope) => memo(`hn${D}${rope}`, () => /* wgsl */ `
struct Params { L: u32, H: u32, ld: u32, colOff: u32, rowOff: u32, sinOff: u32, nx: u32, eps: f32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
@group(0) @binding(2) var<storage, read> Wt: array<f32>;
${rope ? "@group(0) @binding(3) var<storage, read> CS: array<f32>;" : ""}
const D = ${D}u;
const HALF = ${D / 2}u;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let id = wg.y * P.nx + wg.x;
  if (id >= P.L * P.H) { return; }
  let l = id / P.H;
  let h = id % P.H;
  let base = (P.rowOff + l) * P.ld + P.colOff + h * D;
  var ss = 0.0;
  for (var i = t; i < D; i += 64u) { let v = X[base + i]; ss += v * v; }
  red[t] = ss; workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) { if (t < s) { red[t] = red[t] + red[t + s]; } workgroupBarrier(); }
  let r = inverseSqrt(red[0] / f32(D) + P.eps);
  ${rope ? `
  for (var i = t; i < HALF; i += 64u) {
    let y1 = X[base + i] * r * Wt[i];
    let y2 = X[base + i + HALF] * r * Wt[i + HALF];
    let c = CS[l * HALF + i];
    let s = CS[P.sinOff + l * HALF + i];
    X[base + i] = y1 * c - y2 * s;
    X[base + i + HALF] = y2 * c + y1 * s;
  }` : `
  for (var i = t; i < D; i += 64u) { X[base + i] = X[base + i] * r * Wt[i]; }`}
}`);

// Gated linear unit over rows of a stacked [rows, 2I] projection: Y[r][j] = act(X[r][j]) * X[r][I + j]
// (SwiGLU with act = silu, Gemma's MLP with act = gelu_tanh).
export const gluShader = (act) => memo("glu" + act, () => /* wgsl */ `
struct Params { n: u32, I: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
${ACT}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i >= P.n) { return; }
  let r = i / P.I;
  let j = i % P.I;
  let base = r * 2u * P.I + j;
  Y[i] = ${act}(X[base]) * X[base + P.I];
}`);

// Weighted sum over rows: Y[n] = scale * sum_r w[r] * X[r][n] (prompt-weighted text pooling).
export const rowSumShader = () => memo("rowsum", () => /* wgsl */ `
struct Params { rows: u32, N: u32, scale: f32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> Wr: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.N) { return; }
  var s = 0.0;
  for (var r = 0u; r < P.rows; r++) { s += Wr[r] * X[r * P.N + n]; }
  Y[n] = s * P.scale;
}`);

// ConvRot: y = x @ H per contiguous group of 256 features, H = kron(H4,H4,H4,H4)/16 (symmetric).
export const hadamardShader = () => memo("had", () => /* wgsl */ `
struct Params { groups: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
var<workgroup> s: array<f32, 256>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let g = wg.y * P.nx + wg.x;
  if (g >= P.groups) { return; }
  let base = g * 256u;
  for (var j = 0u; j < 4u; j++) { s[t + 64u * j] = X[base + t + 64u * j]; }
  workgroupBarrier();
  var stride = 1u;
  for (var d = 0u; d < 4u; d++) {
    let lo = t % stride;
    let i0 = (t / stride) * stride * 4u + lo;
    let a = s[i0]; let b = s[i0 + stride]; let c = s[i0 + 2u * stride]; let e = s[i0 + 3u * stride];
    // regular H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]
    workgroupBarrier();
    s[i0] = a + b + c - e;
    s[i0 + stride] = a + b - c + e;
    s[i0 + 2u * stride] = a - b + c + e;
    s[i0 + 3u * stride] = -a + b + c + e;
    workgroupBarrier();
    stride *= 4u;
  }
  for (var j = 0u; j < 4u; j++) { Y[base + t + 64u * j] = s[t + 64u * j] * 0.0625; }
}`);

// ----------------------------------------------------------------------------- GroupNorm
// 32 groups over the channels of an NHWC image X[P][C] (group g = channels g*C/32 ..). Two
// passes for the statistics (mean, then the centered sum of squares) so large spatial sizes
// keep full precision: per-chunk partial sums, a per-group reduction into STATS (mean at
// [0, 32), 1/std at [32, 64)), then the normalization.

// Partial sums for one chunk of pixels per workgroup: PART[chunk][32]. pass 0 sums x, pass 1
// sums (x - mean)^2. C must divide 256 or be a multiple of 256.
export const groupNormPartialShader = (C) => memo("gnp" + C, () => {
  const lanes = Math.min(C, 256); // threads that read distinct channels of one pixel
  const nc = C / lanes; // channels per thread
  const pStride = 256 / lanes; // pixels read side by side per iteration
  const j = [...Array(nc).keys()];
  return /* wgsl */ `
struct Params { P: u32, chunk: u32, phase: u32, cpg: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> STATS: array<f32>;
@group(0) @binding(3) var<storage, read_write> PART: array<f32>;
var<workgroup> sh: array<f32, ${256 * nc}>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let p0 = wg.x * P.chunk;
  let p1 = min(p0 + P.chunk, P.P);
  let c0 = t % ${lanes}u;
  ${j.map((k) => `var acc${k} = 0.0;`).join(" ")}
  ${j.map((k) => `let m${k} = select(0.0, STATS[(c0 + ${lanes * k}u) / P.cpg], P.phase == 1u);`).join("\n  ")}
  for (var p = p0 + t / ${lanes}u; p < p1; p += ${pStride}u) {
    ${j.map((k) => `{ let v = X[p * ${C}u + c0 + ${lanes * k}u] - m${k}; acc${k} += select(v, v * v, P.phase == 1u); }`).join("\n    ")}
  }
  ${j.map((k) => `sh[t * ${nc}u + ${k}u] = acc${k};`).join(" ")}
  workgroupBarrier();
  if (t < 32u) {
    var s = 0.0;
    for (var tt = 0u; tt < 256u; tt++) {
      ${j.map((k) => `if ((tt % ${lanes}u + ${lanes * k}u) / P.cpg == t) { s += sh[tt * ${nc}u + ${k}u]; }`).join("\n      ")}
    }
    PART[wg.x * 32u + t] = s;
  }
}`;
});

export const groupNormFinalizeShader = () => memo("gnf", () => /* wgsl */ `
struct Params { chunks: u32, phase: u32, count: f32, eps: f32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> PART: array<f32>;
@group(0) @binding(2) var<storage, read_write> STATS: array<f32>;
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) g: u32) {
  var s = 0.0;
  for (var k = 0u; k < P.chunks; k++) { s += PART[k * 32u + g]; }
  if (P.phase == 0u) { STATS[g] = s / P.count; } else { STATS[32u + g] = inverseSqrt(s / P.count + P.eps); }
}`);

// y = (x - mean) / std * gamma + beta (+ SiLU)
export const groupNormApplyShader = (silu) => memo("gna" + silu, () => /* wgsl */ `
struct Params { n: u32, C: u32, cpg: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> STATS: array<f32>;
@group(0) @binding(3) var<storage, read> GB: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
${ACT}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i >= P.n) { return; }
  let c = i % P.C;
  let g = c / P.cpg;
  let y = (X[i] - STATS[g]) * STATS[32u + g] * GB[c] + GB[P.C + c];
  Y[i] = ${silu ? "silu(y)" : "y"};
}`);

// ----------------------------------------------------------------------------- elementwise

// Y += A, in place (a buffer can't be bound both read-only and writable in one dispatch)
export const accumulateShader = () => memo("acc", () => /* wgsl */ `
struct Params { n: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i < P.n) { Y[i] = Y[i] + A[i]; }
}`);

export const elementwiseShader = (op) => memo("ew" + op, () => {
  const body = {
    add: "Y[i] = A[i] + B[i];",
    sub: "Y[i] = A[i] - B[i];",
    silu: "Y[i] = silu(A[i]);",
    copy: "Y[i] = A[i];",
  }[op];
  // "auto" layouts drop unreferenced bindings, so B is only declared for binary ops
  const binary = op === "add" || op === "sub";
  return /* wgsl */ `
struct Params { n: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> A: array<f32>;
${binary ? "@group(0) @binding(2) var<storage, read> B: array<f32>;" : ""}
@group(0) @binding(${binary ? 3 : 2}) var<storage, read_write> Y: array<f32>;
${ACT}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i >= P.n) { return; }
  ${body}
}`;
});

export const zeroShader = () => memo("zero", () => /* wgsl */ `
struct Params { n: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i < P.n) { Y[i] = 0.0; }
}`);
