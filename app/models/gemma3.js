// Gemma3-270M text encoder as used by Nanosaur2: the hidden state after the second-to-last layer,
// with the final norm applied (ComfyUI layer="hidden", layer_idx=-2, layer_norm_hidden_state).
//
// Gemma specifics: embeddings scaled by sqrt(hidden), RMSNorm weights stored as (w - 1),
// norms before and after both attention and MLP, q/k norm, GQA with one kv head,
// GELU(tanh) gated MLP. Every 6th layer is global (rope theta 1e6), the others local
// (theta 1e4, sliding window 512 > the 256-token limit, so the window never applies).

import * as ops from "../gpu/ops.js";
import { concatLinears } from "../weights.js";
import { ropeTable } from "./common.js";

const CFG = { layers: 18, hidden: 640, heads: 4, kvHeads: 1, headDim: 256, globalEvery: 6 };
const RUN = CFG.layers - 1; // layers needed for the penultimate hidden state

export class Gemma3 {
  static async load(gpu, st, onProgress) {
    const m = new Gemma3(gpu, st);
    const L = [];
    for (let i = 0; i < RUN; i++) {
      const p = `model.layers.${i}.`;
      const norm = (n) => st.vector(gpu, p + n + ".weight", 1);
      L.push({
        global: (i + 1) % CFG.globalEvery === 0,
        inNorm: await norm("input_layernorm"),
        postAttnNorm: await norm("post_attention_layernorm"),
        preMlpNorm: await norm("pre_feedforward_layernorm"),
        postMlpNorm: await norm("post_feedforward_layernorm"),
        qn: await norm("self_attn.q_norm"),
        kn: await norm("self_attn.k_norm"),
        qkv: concatLinears(gpu, [await st.linear(gpu, p + "self_attn.q_proj."), await st.linear(gpu, p + "self_attn.k_proj."), await st.linear(gpu, p + "self_attn.v_proj.")]),
        o: await st.linear(gpu, p + "self_attn.o_proj."),
        gateUp: concatLinears(gpu, [await st.linear(gpu, p + "mlp.gate_proj."), await st.linear(gpu, p + "mlp.up_proj.")]),
        down: await st.linear(gpu, p + "mlp.down_proj."),
      });
      onProgress?.((i + 1) / RUN);
    }
    m.layers = L;
    m.norm = await st.vector(gpu, "model.norm.weight", 1);
    return m;
  }

  constructor(gpu, st) {
    this.gpu = gpu;
    this.st = st; // embedding rows are gathered from the file on demand
  }

  // ids -> Tensor [L, 640]
  async encode(ids) {
    const gpu = this.gpu;
    const { hidden: D, heads: H, kvHeads: KH, headDim: HD } = CFG;
    const Lt = ids.length;
    const qkvW = (H + 2 * KH) * HD;
    const emb = await this.st.rows("model.embed_tokens.weight", ids);
    const scale = Math.sqrt(D);
    for (let i = 0; i < emb.length; i++) emb[i] *= scale;
    const x = gpu.fromArray(emb, [Lt, D]);
    const csLocal = ropeTable(gpu, Lt, HD, 1e4);
    const csGlobal = ropeTable(gpu, Lt, HD, 1e6);
    for (const l of this.layers) {
      const cs = l.global ? csGlobal : csLocal;
      const h = ops.rmsnorm(gpu, x, l.inNorm, D);
      const qkv = ops.linear(gpu, h, l.qkv, { name: "te.qkv" }); // [Lt, q 1024 | k 256 | v 256]
      h.release();
      ops.headNorm(gpu, qkv, l.qn, { L: Lt, H, D: HD, ld: qkvW, colOff: 0, cs });
      ops.headNorm(gpu, qkv, l.kn, { L: Lt, H: KH, D: HD, ld: qkvW, colOff: H * HD, cs });
      const a = ops.attention(gpu, {
        q: qkv, k: qkv, v: qkv, Lq: Lt, Lk: Lt, H, D: HD, ldq: qkvW, ldk: qkvW, ldv: qkvW,
        qOff: 0, kOff: H * HD, vOff: (H + KH) * HD, group: H / KH, causal: true,
      });
      qkv.release();
      const o = ops.linear(gpu, a, l.o, { name: "te.o" });
      a.release();
      const on = ops.rmsnorm(gpu, o, l.postAttnNorm, D);
      o.release();
      ops.accumulate(gpu, x, on);
      on.release();
      const h2 = ops.rmsnorm(gpu, x, l.preMlpNorm, D);
      const gu = ops.linear(gpu, h2, l.gateUp, { name: "te.mlp" });
      h2.release();
      const g = ops.glu(gpu, gu, "gelu_tanh");
      gu.release();
      const d = ops.linear(gpu, g, l.down, { name: "te.mlp" });
      g.release();
      const dn = ops.rmsnorm(gpu, d, l.postMlpNorm, D);
      d.release();
      ops.accumulate(gpu, x, dn);
      dn.release();
    }
    csLocal.release();
    csGlobal.release();
    const out = ops.rmsnorm(gpu, x, this.norm, D);
    x.release();
    return out;
  }
}
