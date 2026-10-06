// Nanosaur2 diffusion transformer (670M): adaLN-single DiT on 64-channel 1/16-scale latents
// (one token per latent pixel), 2D RoPE, QK-norm, SwiGLU, x-prediction.
//
//  - Text: Gemma hidden states -> y_embedder (linear + RMSNorm) -> 2 text-refine blocks
//    conditioned on the timestep. Prompt weights pool the refined text into the global
//    condition and become log-biases on the text keys.
//  - Image blocks: even blocks attend jointly over [image keys; text keys], odd blocks over the
//    image only. Block modulation = shared_encoder_adaLN(condition) + a learned per-block offset.
//  - SPRINT: blocks 2..15 form a residual "sparse path", x + sprint_out_proj(path(x) - x).
//    Path-drop guidance skips it for the unconditional pass.
// See nanosaur2_support/model.py for the reference.

import * as ops from "../gpu/ops.js";
import { rope2dTable } from "./common.js";

const D = 1536;
const HEADS = 16;
const HD = D / HEADS; // 96
const BLOCKS = 18;
const SPRINT_F = 2; // dense blocks before the sparse path
const SPRINT_H = 2; // dense blocks after it
const CH = 64; // latent channels

// Sinusoidal embedding of timestep * 1000 (cos half, then sin half), 256 dims
export function timestepEmbedding(sigma) {
  const dim = 256, half = dim / 2;
  const t = Math.fround(sigma * 1000);
  const out = new Float32Array(dim);
  for (let i = 0; i < half; i++) {
    const f = Math.fround(Math.exp(Math.fround((-Math.log(10000) * i) / half)));
    const a = Math.fround(t * f);
    out[i] = Math.cos(a);
    out[half + i] = Math.sin(a);
  }
  return out;
}

export class Nanosaur2DiT {
  static async load(gpu, st, onProgress) {
    const m = new Nanosaur2DiT(gpu);
    const lin = (n) => st.linear(gpu, n);
    const vec = (n) => st.vector(gpu, n);
    m.sEmbed = await lin("s_embedder.proj.");
    m.t1 = await lin("t_embedder.mlp.0.");
    m.t2 = await lin("t_embedder.mlp.2.");
    m.yEmbed = await lin("y_embedder.proj.");
    m.yNorm = await vec("y_embedder.norm.weight");
    m.yPool = await lin("y_pool_proj.");
    m.sharedAda = await lin("shared_encoder_adaLN.0.");
    m.sprintOut = await lin("sprint_out_proj.");
    m.text = [];
    for (let i = 0; st.has(`text_refine_blocks.${i}.norm1.weight`); i++) {
      const p = `text_refine_blocks.${i}.`;
      m.text.push({
        ada: await lin(p + "adaLN_modulation.0."),
        n1: await vec(p + "norm1.weight"),
        n2: await vec(p + "norm2.weight"),
        qkv: await lin(p + "attn.qkv."),
        qn: await vec(p + "attn.q_norm.weight"),
        kn: await vec(p + "attn.k_norm.weight"),
        proj: await lin(p + "attn.proj."),
        w12: await lin(p + "mlp.w12."),
        w3: await lin(p + "mlp.w3."),
      });
    }
    m.blocks = [];
    for (let i = 0; i < BLOCKS; i++) {
      const p = `blocks.${i}.`;
      m.blocks.push({
        offset: await vec(`encoder_adaLN_offsets.${i}`),
        n1: await vec(p + "norm1.weight"),
        n2: await vec(p + "norm2.weight"),
        qkv: await lin(p + "attn.qkv_x."),
        kvY: st.has(p + "attn.kv_y.weight") ? await lin(p + "attn.kv_y.") : null,
        qn: await vec(p + "attn.q_norm.weight"),
        kn: await vec(p + "attn.k_norm.weight"),
        proj: await lin(p + "attn.proj."),
        w12: await lin(p + "mlp.w12."),
        w3: await lin(p + "mlp.w3."),
      });
      onProgress?.((i + 1) / BLOCKS);
    }
    m.finalAda = await lin("final_layer.adaLN_modulation.");
    m.finalLinear = await lin("final_layer.linear.");
    return m;
  }

  constructor(gpu) {
    this.gpu = gpu;
  }

  // ------------------------------------------------------------------ text

  // Per-prompt text input. hidden: Float32Array [L, 640] from the text encoder; weights: per token.
  // Tokens with weight <= 0 are fully masked in the reference (no key, zeroed text row, no pooling
  // contribution), so they are dropped here; they still count in the pooling denominator.
  prepareText(hidden, weights) {
    const gpu = this.gpu;
    const keep = weights.map((w, i) => (w > 0 ? i : -1)).filter((i) => i >= 0);
    const L = keep.length;
    const denom = Math.max(1, weights.reduce((a, b) => a + b, 0));
    if (!L) return { L: 0, release() {} };
    const ctx = new Float32Array(L * 640);
    keep.forEach((src, i) => ctx.set(hidden.subarray(src * 640, (src + 1) * 640), i * 640));
    const x = gpu.fromArray(ctx, [L, 640]);
    const h = ops.linear(gpu, x, this.yEmbed);
    x.release();
    const y = ops.rmsnorm(gpu, h, this.yNorm, D);
    h.release();
    const w = keep.map((i) => weights[i]);
    const pool = gpu.fromArray(new Float32Array(w), [L]);
    // log prompt weights as attention biases on the text keys (all zeros: no bias needed)
    const bias = w.some((v) => v !== 1) ? gpu.fromArray(new Float32Array(w.map((v) => Math.log(Math.max(v, 1e-4)))), [L]) : null;
    return {
      L, y, pool, poolScale: 1 / denom, bias,
      release() { y.release(); pool.release(); bias?.release(); },
    };
  }

  // Text refine blocks for one timestep: Tensor [L, D]
  refineText(text, tc) {
    const gpu = this.gpu;
    const L = text.L;
    const txt = ops.elementwise(gpu, "copy", text.y);
    for (const b of this.text) {
      const mod = ops.linear(gpu, tc, b.ada); // [1, 6D]
      const n = ops.normMod(gpu, txt, mod, 0, D, D, b.n1);
      const qkv = ops.linear(gpu, n, b.qkv, { name: "dit.text.qkv" });
      n.release();
      ops.headNorm(gpu, qkv, b.qn, { L, H: HEADS, D: HD, ld: 3 * D, colOff: 0 });
      ops.headNorm(gpu, qkv, b.kn, { L, H: HEADS, D: HD, ld: 3 * D, colOff: D });
      const a = ops.attention(gpu, { q: qkv, k: qkv, v: qkv, Lq: L, Lk: L, H: HEADS, D: HD, ldq: 3 * D, ldk: 3 * D, ldv: 3 * D, qOff: 0, kOff: D, vOff: 2 * D });
      qkv.release();
      ops.linear(gpu, a, b.proj, { out: txt, resid: true, gate: mod, gOff: 2 * D });
      a.release();
      this.mlp(txt, b, mod);
      mod.release();
    }
    return txt;
  }

  // x += gate_mlp * w3(silu(x1) * x2), [x1 x2] = w12(modulate(norm2(x)))
  mlp(x, b, mod) {
    const gpu = this.gpu;
    const n = ops.normMod(gpu, x, mod, 3 * D, 4 * D, D, b.n2);
    const h = ops.linear(gpu, n, b.w12, { name: "dit.mlp.w12" });
    n.release();
    const g = ops.glu(gpu, h, "silu");
    h.release();
    ops.linear(gpu, g, b.w3, { out: x, resid: true, gate: mod, gOff: 5 * D, name: "dit.mlp.w3" });
    g.release();
  }

  // ------------------------------------------------------------------ conditioning

  // Everything that depends on sigma and the prompt but not on the latent.
  condition(text, sigma) {
    const gpu = this.gpu;
    const temb = gpu.fromArray(timestepEmbedding(sigma), [1, 256]);
    const h = ops.linear(gpu, temb, this.t1, { act: "silu" });
    temb.release();
    const t = ops.linear(gpu, h, this.t2); // [1, D]
    h.release();
    const tc = ops.elementwise(gpu, "silu", t);
    let txt = null, txtR = null;
    if (text.L) {
      txt = this.refineText(text, tc);
      // int8 kv_y projections take rotated inputs: rotate the text once for all blocks
      if (this.blocks.some((b) => b.kvY && ops.needsRotation(b.kvY))) txtR = ops.rotate(gpu, txt);
      const pooled = ops.rowSum(gpu, txt, text.pool, text.poolScale);
      ops.linear(gpu, pooled, this.yPool, { out: t, resid: true });
      pooled.release();
    }
    tc.release();
    const cond = ops.elementwise(gpu, "silu", t);
    t.release();
    const mod = ops.linear(gpu, cond, this.sharedAda); // [1, 6D]
    const final = ops.linear(gpu, cond, this.finalAda); // [1, 2D]: shift, scale
    cond.release();
    return {
      txt, txtR, mod, final,
      release() { txt?.release(); txtR?.release(); mod.release(); final.release(); },
    };
  }

  // ------------------------------------------------------------------ forward

  block(i, x, L, text, c, cs) {
    const gpu = this.gpu;
    const b = this.blocks[i];
    const mod = ops.elementwise(gpu, "add", c.mod, b.offset); // [1, 6D]
    const n = ops.normMod(gpu, x, mod, 0, D, D, b.n1);
    const cross = !!(b.kvY && text.L);
    // rows [0, L): image q|k|v; rows [L, L + text): text k|v (q columns unused)
    const Lk = L + (cross ? text.L : 0);
    const qkv = gpu.empty([Lk, 3 * D]);
    ops.linear(gpu, n, b.qkv, { out: qkv, name: "dit.qkv" });
    n.release();
    ops.headNorm(gpu, qkv, b.qn, { L, H: HEADS, D: HD, ld: 3 * D, colOff: 0, cs });
    ops.headNorm(gpu, qkv, b.kn, { L, H: HEADS, D: HD, ld: 3 * D, colOff: D, cs });
    if (cross) {
      ops.linear(gpu, c.txt, b.kvY, { xr: c.txtR, out: qkv, ldc: 3 * D, cOff: L * 3 * D + D, name: "dit.kv_y" });
      ops.headNorm(gpu, qkv, b.kn, { L: text.L, H: HEADS, D: HD, ld: 3 * D, colOff: D, rowOff: L });
    }
    const a = ops.attention(gpu, {
      q: qkv, k: qkv, v: qkv, Lq: L, Lk, H: HEADS, D: HD, ldq: 3 * D, ldk: 3 * D, ldv: 3 * D,
      qOff: 0, kOff: D, vOff: 2 * D, bias: cross ? text.bias : null, biasStart: L,
    });
    qkv.release();
    ops.linear(gpu, a, b.proj, { out: x, resid: true, gate: mod, gOff: 2 * D, name: "dit.proj" });
    a.release();
    this.mlp(x, b, mod);
    mod.release();
  }

  // latent: Float32Array [64, h, w]; text: prepareText(...). pathDrop skips the sparse middle
  // blocks (the unconditional pass of path-drop guidance). Returns the x0 prediction [64, h, w].
  async forward(latent, h, w, text, sigma, { pathDrop = false, onBlock } = {}) {
    const gpu = this.gpu;
    const L = h * w;
    const tokens = new Float32Array(L * CH);
    for (let c = 0; c < CH; c++) for (let l = 0; l < L; l++) tokens[l * CH + c] = latent[c * L + l];
    const xin = gpu.fromArray(tokens, [L, CH]);
    const x = ops.linear(gpu, xin, this.sEmbed);
    xin.release();
    const cs = rope2dTable(gpu, h, w, HD);
    const c = this.condition(text, sigma);
    const total = pathDrop ? SPRINT_F + SPRINT_H : BLOCKS;
    let done = 0;
    const run = async (i, s) => {
      this.block(i, s, L, text, c, cs);
      gpu.flush();
      if (onBlock) await onBlock(++done / total);
    };

    for (let i = 0; i < SPRINT_F; i++) await run(i, x);
    if (!pathDrop) {
      const g = ops.elementwise(gpu, "copy", x);
      for (let i = SPRINT_F; i < BLOCKS - SPRINT_H; i++) await run(i, g);
      const d = ops.elementwise(gpu, "sub", g, x);
      g.release();
      ops.linear(gpu, d, this.sprintOut, { out: x, resid: true });
      d.release();
    }
    for (let i = BLOCKS - SPRINT_H; i < BLOCKS; i++) await run(i, x);
    cs.release();

    const n = ops.normMod(gpu, x, c.final, 0, D, D);
    x.release();
    c.release();
    const out = ops.linear(gpu, n, this.finalLinear); // [L, 64]
    n.release();
    const o = await gpu.read(out);
    out.release();
    const x0 = new Float32Array(CH * L);
    for (let l = 0; l < L; l++) for (let ch = 0; ch < CH; ch++) x0[ch * L + l] = o[l * CH + ch];
    return x0;
  }
}
