// Nanosaur2 VAE decoder (the "semantic" DINOv2 VAE's convolutional decoder), NHWC on the GPU.
// Same structure as ComfyUI's ldm Decoder: ch 128, ch_mult (1, 1, 2, 2, 4), 2 res blocks (+1 in
// the decoder) per level, GroupNorm(32), single-head attention at the latent resolution,
// nearest 2x upsampling (x16 total) and a tanh output. Only the decoder is loaded; the DINOv2
// encoder in the same file is for image-to-latent and not needed here.

import * as ops from "../gpu/ops.js";
import { concatLinears } from "../weights.js";

const LEVELS = 5;
const BLOCKS = 3;

export class VAEDecoder {
  static async load(gpu, st) {
    const v = new VAEDecoder(gpu);
    const gn = async (p) => {
      const g = await st.f32(p + ".weight");
      const b = await st.f32(p + ".bias");
      const gb = new Float32Array(g.length * 2);
      gb.set(g);
      gb.set(b, g.length);
      return gpu.upload(gb);
    };
    const res = async (p) => ({
      n1: await gn(p + ".norm1"),
      c1: await st.conv3x3(gpu, p + ".conv1."),
      n2: await gn(p + ".norm2"),
      c2: await st.conv3x3(gpu, p + ".conv2."),
      sc: st.has(p + ".nin_shortcut.weight") ? await st.linear(gpu, p + ".nin_shortcut.") : null,
    });
    const attn = async (p) => ({
      norm: await gn(p + ".norm"),
      qkv: concatLinears(gpu, [await st.linear(gpu, p + ".q."), await st.linear(gpu, p + ".k."), await st.linear(gpu, p + ".v.")]),
      proj: await st.linear(gpu, p + ".proj_out."),
    });
    v.convIn = await st.conv3x3(gpu, "decoder.conv_in.");
    v.mid = { b1: await res("decoder.mid.block_1"), attn: await attn("decoder.mid.attn_1"), b2: await res("decoder.mid.block_2") };
    v.up = [];
    for (let l = 0; l < LEVELS; l++) {
      const p = `decoder.up.${l}`;
      const level = { blocks: [], attn: [], upsample: null };
      for (let i = 0; i < BLOCKS; i++) {
        level.blocks.push(await res(`${p}.block.${i}`));
        if (st.has(`${p}.attn.${i}.q.weight`)) level.attn.push(await attn(`${p}.attn.${i}`));
      }
      if (st.has(`${p}.upsample.conv.weight`)) level.upsample = await st.conv3x3(gpu, `${p}.upsample.conv.`);
      v.up.push(level);
    }
    v.normOut = await gn("decoder.norm_out");
    v.convOut = await st.conv3x3(gpu, "decoder.conv_out.");
    v.mean = await st.f32("latent_mean");
    v.std = await st.f32("latent_std");
    v.channels = v.mean.length;
    return v;
  }

  constructor(gpu) {
    this.gpu = gpu;
  }

  resblock(x, r, h, w) {
    const gpu = this.gpu;
    const cin = x.shape[1];
    let t = ops.groupNorm(gpu, x, r.n1, cin, true);
    const a = ops.conv3x3(gpu, t, r.c1, h, w);
    t.release();
    t = ops.groupNorm(gpu, a, r.n2, r.c1.N, true);
    a.release();
    const out = r.sc ? ops.linear(gpu, x, r.sc) : x;
    ops.conv3x3Into(gpu, t, r.c2, h, w, out);
    t.release();
    if (out !== x) x.release();
    return out;
  }

  // single-head attention over all pixels, residual
  attention(x, at, hw) {
    const gpu = this.gpu;
    const C = x.shape[1];
    const n = ops.groupNorm(gpu, x, at.norm, C, false);
    const qkv = ops.linear(gpu, n, at.qkv);
    n.release();
    const a = ops.attention(gpu, { q: qkv, k: qkv, v: qkv, Lq: hw, Lk: hw, H: 1, D: C, ldq: 3 * C, ldk: 3 * C, ldv: 3 * C, qOff: 0, kOff: C, vOff: 2 * C });
    qkv.release();
    ops.linear(gpu, a, at.proj, { out: x, resid: true });
    a.release();
    return x;
  }

  // latent: Float32Array [64, h, w] (the model's normalized space) -> RGBA Uint8ClampedArray
  async decode(latent, h, w, onStage) {
    const gpu = this.gpu;
    const C = this.channels;
    const hw = h * w;
    const z = new Float32Array(hw * C);
    for (let c = 0; c < C; c++) for (let i = 0; i < hw; i++) z[i * C + c] = latent[c * hw + i] * this.std[c] + this.mean[c];
    const zt = gpu.fromArray(z, [hw, C]);
    let x = ops.conv3x3(gpu, zt, this.convIn, h, w);
    zt.release();
    x = this.resblock(x, this.mid.b1, h, w);
    x = this.attention(x, this.mid.attn, hw);
    x = this.resblock(x, this.mid.b2, h, w);

    let H = h, W = w;
    const stages = LEVELS * BLOCKS;
    let stage = 0;
    for (let l = LEVELS - 1; l >= 0; l--) {
      const level = this.up[l];
      for (let i = 0; i < BLOCKS; i++) {
        x = this.resblock(x, level.blocks[i], H, W);
        if (level.attn[i]) x = this.attention(x, level.attn[i], H * W);
        await gpu.sync();
        onStage?.(++stage / stages);
      }
      if (level.upsample) {
        const y = ops.conv3x3(gpu, x, level.upsample, H, W, true);
        x.release();
        x = y;
        H *= 2; W *= 2;
      }
    }
    const t = ops.groupNorm(gpu, x, this.normOut, x.shape[1], true);
    x.release();
    const rgb = ops.conv3x3(gpu, t, this.convOut, H, W);
    t.release();
    const px = await gpu.read(rgb);
    rgb.release();
    const img = new Uint8ClampedArray(H * W * 4);
    for (let i = 0; i < H * W; i++) {
      for (let c = 0; c < 3; c++) img[i * 4 + c] = Math.round(((Math.tanh(px[i * 3 + c]) + 1) / 2) * 255);
      img[i * 4 + 3] = 255;
    }
    return { data: img, width: W, height: H };
  }
}
