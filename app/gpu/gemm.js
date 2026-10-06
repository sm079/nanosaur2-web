// Tiled GEMM generator.
//
// C[z][m][n] = epilogue( alpha * sum_k A(z,m,k) * B(z,n,k) )
//
// A modes : "rows"  A[aOff + z*aBatch + m*lda + k]
//           "conv3" implicit im2col of an NHWC image, 3x3 kernel, pad 1, optional fused
//                   2x nearest upsample of the input (K = 9*Cin, k = tap*Cin + ci)
// B fmts  : "f32"   B[bOff + z/bDiv*bBatch + n*ldb + k]    ("f32t": B[... + k*ldb + n])
//           "bf16"  [N,K] bf16 pairs packed in u32
//           "i8"    [N,K] int8 packed 4 per u32            (per-row scale in epilogue)
// Epilogue: *rowScale[n], +bias[n], +add[m][n], act (silu), and either store or
//           C = C + gate[gOff+n] * v (gated residual, in place).
//
// Tiles (256 threads, 16 KB of workgroup memory in both):
//   R = 4: 64x64,   TK = 32, 4x4 outputs per thread  (small problems)
//   R = 8: 128x128, TK = 16, 8x8 outputs per thread  (everything large)
// Every thread stages 8 consecutive k of one A row and one B row per k-tile, and issues the
// next tile's global loads before computing on the current one (register prefetch).
//
// All register arrays are indexed with constants only (the generator unrolls every loop that
// touches them): shader compilers spill dynamically indexed arrays to local memory.

const ACT = /* wgsl */ `
fn silu(x: f32) -> f32 { return x / (1.0 + exp(-x)); }
`;

const range = (n) => [...Array(n).keys()];
const cache = new Map();

// Loaders return V8 { lo, hi }: 8 consecutive k values.
function loaderA(a, vec) {
  if (a === "rows") {
    if (vec) {
      return `
fn loadA8(z: u32, m: u32, k: u32) -> V8 {
  if (m >= P.M || k >= P.K) { return V8(); }
  let q = (P.aOff + z * P.aBatch + m * P.lda + k) >> 2u;
  return V8(A[q], A[q + 1u]);
}`;
    }
    const e = range(8).map((j) => `select(0.0, A[base + ${j}u], k + ${j}u < P.K)`);
    return `
fn loadA8(z: u32, m: u32, k: u32) -> V8 {
  if (m >= P.M) { return V8(); }
  let base = P.aOff + z * P.aBatch + m * P.lda + k;  // out-of-range lanes are masked (robust access keeps reads safe)
  return V8(vec4<f32>(${e.slice(0, 4).join(", ")}), vec4<f32>(${e.slice(4).join(", ")}));
}`;
  }
  // conv3: NHWC input; output pixel m = oy*W + ox; the 8 k share one tap (Cin % 8 == 0)
  const body = vec
    ? "let q = base >> 2u;\n  return V8(A[q], A[q + 1u]);"
    : `return V8(vec4<f32>(${range(4).map((j) => `A[base + ${j}u]`).join(", ")}), vec4<f32>(${range(4).map((j) => `A[base + ${j + 4}u]`).join(", ")}));`;
  return `
fn loadA8(z: u32, m: u32, k: u32) -> V8 {
  if (m >= P.M || k >= P.K) { return V8(); }
  let ox = i32(m % P.cw);
  let oy = i32(m / P.cw);
  let tap = k / P.cin;
  let ci = k % P.cin;
  let iy = oy + i32(tap / 3u) - 1;
  let ix = ox + i32(tap % 3u) - 1;
  if (iy < 0 || ix < 0 || iy >= i32(P.ch) || ix >= i32(P.cw)) { return V8(); }
  var sy = u32(iy); var sx = u32(ix); var sw = P.cw;
  if (P.up == 1u) { sy = sy >> 1u; sx = sx >> 1u; sw = P.cw >> 1u; }
  let base = (sy * sw + sx) * P.cin + ci;
  ${body}
}`;
}

function loaderB(b, vec) {
  if (b === "f32") {
    if (vec) {
      return `
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N || k >= P.K) { return V8(); }
  let q = (P.bOff + (z / P.bDiv) * P.bBatch + n * P.ldb + k) >> 2u;
  return V8(B[q], B[q + 1u]);
}`;
    }
    const e = range(8).map((j) => `select(0.0, B[base + ${j}u], k + ${j}u < P.K)`);
    return `
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N) { return V8(); }
  let base = P.bOff + (z / P.bDiv) * P.bBatch + n * P.ldb + k;
  return V8(vec4<f32>(${e.slice(0, 4).join(", ")}), vec4<f32>(${e.slice(4).join(", ")}));
}`;
  }
  if (b === "f32t") {
    const e = range(8).map((j) => `select(0.0, B[min(base + (k + ${j}u) * P.ldb, last)], k + ${j}u < P.K)`);
    return `
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N) { return V8(); }
  let base = P.bOff + (z / P.bDiv) * P.bBatch + n;
  let last = arrayLength(&B) - 1u;
  return V8(vec4<f32>(${e.slice(0, 4).join(", ")}), vec4<f32>(${e.slice(4).join(", ")}));
}`;
  }
  if (b === "bf16") {
    const lo = (w) => `bitcast<f32>(${w} << 16u)`;
    const hi = (w) => `bitcast<f32>(${w} & 0xffff0000u)`;
    if (vec) {
      return `
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N || k >= P.K) { return V8(); }
  let w = B[(n * P.K + k) >> 3u];
  return V8(vec4<f32>(${lo("w.x")}, ${hi("w.x")}, ${lo("w.y")}, ${hi("w.y")}), vec4<f32>(${lo("w.z")}, ${hi("w.z")}, ${lo("w.w")}, ${hi("w.w")}));
}`;
    }
    // K even; a partial last chunk is zero-masked
    const w = range(4).map((j) => `let w${j} = select(0u, B[base + ${j}u], k + ${2 * j}u < P.K);`);
    return `
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N || k >= P.K) { return V8(); }
  let base = (n * P.K + k) >> 1u;
  ${w.join("\n  ")}
  return V8(vec4<f32>(${lo("w0")}, ${hi("w0")}, ${lo("w1")}, ${hi("w1")}), vec4<f32>(${lo("w2")}, ${hi("w2")}, ${lo("w3")}, ${hi("w3")}));
}`;
  }
  if (b === "i8") {
    const load = vec
      ? `let w = B[(n * P.K + k) >> 3u];
  let w0 = i32(w.x);
  let w1 = i32(w.y);`
      : `let base = (n * P.K + k) >> 2u;
  let w0 = i32(B[base]);
  let w1 = i32(B[base + 1u]);`;
    const x = (w) => range(4).map((j) => `f32(extractBits(${w}, ${8 * j}u, 8u))`).join(", ");
    return `
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N || k >= P.K) { return V8(); }
  ${load}
  return V8(vec4<f32>(${x("w0")}), vec4<f32>(${x("w1")}));
}`;
  }
  throw new Error(`unknown B format ${b}`);
}

// vecA / vecB: vectorized global loads; the caller guarantees K % 8 == 0 and 4-aligned offsets
// and strides (for bf16 / int8 weights: K % 8 == 0).
// add: an extra [M, N] input added before the activation (LoRA deltas), ADD[m*N + n].
export function matmulShader({ a = "rows", b = "f32", bias = false, act = "none", resid = false, gate = false, R = 8, vecA = false, vecB = false, add = false }) {
  const key = JSON.stringify([a, b, bias, act, resid, gate, R, vecA, vecB, add]);
  if (cache.has(key)) return cache.get(key);

  const bindings = [];
  const bind = (decl) => bindings.push(`@group(0) @binding(${bindings.length + 1}) ${decl};`);
  bind(`var<storage, read> A: array<${vecA ? "vec4<f32>" : "f32"}>`);
  const bType = b === "f32" ? (vecB ? "vec4<f32>" : "f32")
    : b === "f32t" ? "f32"
      : b === "i8" ? (vecB ? "vec2<u32>" : "u32")
        : vecB ? "vec4<u32>" : "u32"; // bf16
  bind(`var<storage, read> B: array<${bType}>`);
  bind("var<storage, read_write> C: array<f32>");
  if (b === "i8") bind("var<storage, read> RS: array<f32>");
  if (bias) bind("var<storage, read> BIAS: array<f32>");
  if (gate) bind("var<storage, read> G: array<f32>");
  if (add) bind("var<storage, read> ADD: array<f32>");

  let epi = "";
  if (b === "i8") epi += " r = r * RS[n];";
  if (bias) epi += " r = r + BIAS[n];";
  if (add) epi += " r = r + ADD[m * P.N + n];";
  if (act === "silu") epi += " r = silu(r);";
  const store = resid ? (gate ? "C[ci] = C[ci] + G[P.gOff + n] * r;" : "C[ci] = C[ci] + r;") : "C[ci] = r;";

  const T = 16 * R; // tile edge
  const TK = R === 8 ? 16 : 32;
  const V = T / 4; // vec4 per k-row of a tile
  const G = R / 4; // 64-wide row/column groups per thread
  const perRow = TK / 8; // threads per staged row

  // staging: 8 k values of row lr -> As[(lk + j) * V + (lr >> 2)][lr & 3]
  const stage = range(8).map((j) => {
    const src = j < 4 ? `lo.${"xyzw"[j]}` : `hi.${"xyzw"[j - 4]}`;
    return `As[(lk + ${j}u) * ${V}u + sIdx][sLane] = av.${src};\n      Bs[(lk + ${j}u) * ${V}u + sIdx][sLane] = bv.${src};`;
  });

  const fma = [];
  for (let g = 0; g < G; g++) fma.push(`let a${g} = As[kk * ${V}u + ${g * 16}u + tm];`);
  for (let h = 0; h < G; h++) fma.push(`let b${h} = Bs[kk * ${V}u + ${h * 16}u + tn];`);
  for (let g = 0; g < G; g++) {
    ["x", "y", "z", "w"].forEach((c, i) => {
      for (let h = 0; h < G; h++) fma.push(`acc${(g * 4 + i) * G + h} += a${g}.${c} * b${h};`);
    });
  }

  // epilogue: fully unrolled over the thread's R x R outputs
  const out = [];
  for (let g = 0; g < G; g++) {
    for (let i = 0; i < 4; i++) {
      out.push(`{ let m = m0 + ${g * 64 + i}u + tm * 4u;\n    if (m < P.M) {`);
      for (let h = 0; h < G; h++) {
        for (let j = 0; j < 4; j++) {
          out.push(`      { let n = n0 + ${h * 64 + j}u + tn * 4u; if (n < P.N) { let ci = P.cOff + z * P.cBatch + m * P.ldc + n; var r = acc${(g * 4 + i) * G + h}.${"xyzw"[j]} * P.alpha;${epi} ${store} } }`);
        }
      }
      out.push("    } }");
    }
  }

  const code = /* wgsl */ `
struct Params {
  M: u32, N: u32, K: u32, alpha: f32,
  lda: u32, aBatch: u32, aOff: u32, ldb: u32,
  bBatch: u32, bDiv: u32, bOff: u32, ldc: u32,
  cBatch: u32, cOff: u32, gOff: u32, cin: u32,
  ch: u32, cw: u32, up: u32, pad0: u32,
};
struct V8 { lo: vec4<f32>, hi: vec4<f32> };
@group(0) @binding(0) var<uniform> P: Params;
${bindings.join("\n")}
${act !== "none" ? ACT : ""}
${loaderA(a, vecA)}
${loaderB(b, vecB)}

const TK = ${TK}u;
var<workgroup> As: array<vec4<f32>, ${V * TK}>;  // [TK][T/4]
var<workgroup> Bs: array<vec4<f32>, ${V * TK}>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let z = wg.z;
  let m0 = wg.y * ${T}u;
  let n0 = wg.x * ${T}u;
  let tm = t & 15u;
  let tn = t >> 4u;
  let lr = t / ${perRow}u;
  let lk = (t % ${perRow}u) * 8u;
  let sIdx = lr >> 2u;
  let sLane = lr & 3u;
  ${range(R * G).map((i) => `var acc${i} = vec4<f32>();`).join("\n  ")}
  var av = loadA8(z, m0 + lr, lk);
  var bv = loadB8(z, n0 + lr, lk);

  for (var k0 = 0u; k0 < P.K; k0 += TK) {
    ${stage.join("\n    ")}
    workgroupBarrier();
    if (k0 + TK < P.K) {
      av = loadA8(z, m0 + lr, k0 + TK + lk);
      bv = loadB8(z, n0 + lr, k0 + TK + lk);
    }
    for (var kk = 0u; kk < TK; kk++) {
      ${fma.join("\n      ")}
    }
    workgroupBarrier();
  }

  ${out.join("\n  ")}
}`;
  cache.set(key, code);
  return code;
}
