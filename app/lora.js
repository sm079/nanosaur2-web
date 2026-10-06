// LoRA support without touching the base weights: each targeted linear gets a low-rank side
// path, y = W x + scale * B (A x), evaluated in ops.linear (see loraDelta). Strength changes are
// free and nothing has to be re-uploaded.
//
// Supported files: plain LoRA in PEFT naming ("diffusion_model.<module>.lora_A/B.weight" +
// ".alpha", as written by nanosaur2_support/train_lora.py) or kohya naming
// ("lora_unet_<module_with_underscores>.lora_down/up.weight" + ".alpha"), for the DiT and the
// Gemma3 text encoder. LoKr/LoHa/DoRA are reported as unsupported.

import { SafeTensors } from "./weights.js";

const norm = (name) => name.replace(/\./g, "_");

const SUFFIXES = [
  [".lora_A.weight", "A"], [".lora_B.weight", "B"],
  [".lora_down.weight", "A"], [".lora_up.weight", "B"],
  [".alpha", "alpha"],
];
const TE_PREFIXES = ["lora_te_", "lora_te1_", "text_encoders.gemma3_270m.transformer.", "text_encoder.", "te."];
const DIT_PREFIXES = ["lora_unet_", "model.diffusion_model.", "diffusion_model.", "base_model.model.", "transformer.", "unet."];
const UNSUPPORTED = /(lokr_|hada_|\.dora_scale|\.diff$|\.diff_b$|\.w_norm|\.b_norm)/;

// Quick look at a LoRA's tensor names (from the file header alone, before downloading the rest):
// is it a LoRA we can apply, and do its modules have Nanosaur2's names? The engine does the
// exact per-layer match after loading; this only catches files made for other models early.
const NANOSAUR_MODULE = new RegExp("^(" + [
  "blocks_\\d+_(attn_(qkv_x|kv_y|proj)|mlp_w(12|3))",
  "text_refine_blocks_\\d+_(attn_(qkv|proj)|mlp_w(12|3)|adaLN_modulation_0)",
  "(s|y)_embedder_proj", "t_embedder_mlp_[02]", "shared_encoder_adaLN_0", "y_pool_proj", "sprint_out_proj",
  "final_layer_(linear|adaLN_modulation)",
  "(model_)?layers_\\d+_(self_attn_[qkvo]_proj|mlp_(gate|up|down)_proj)",
].join("|") + ")$");

export function inspectLoraKeys(keys) {
  const modules = new Set();
  let unsupported = 0;
  for (const key of keys) {
    if (UNSUPPORTED.test(key)) { unsupported++; continue; }
    const suf = SUFFIXES.find(([s]) => key.endsWith(s));
    if (suf && suf[1] !== "alpha") modules.add(key.slice(0, -suf[0].length));
  }
  let fits = 0;
  for (const m of modules) {
    let name = m;
    const p = [...TE_PREFIXES, ...DIT_PREFIXES].find((x) => name.startsWith(x));
    if (p) name = name.slice(p.length);
    if (NANOSAUR_MODULE.test(norm(name))) fits++;
  }
  return { lora: modules.size > 0, modules: modules.size, fits, unsupported };
}

// name -> { W, off, n } for every linear reachable from a model object
export function linearRegistry(root) {
  const reg = new Map();
  const seen = new Set();
  const walk = (o) => {
    if (!o || typeof o !== "object" || seen.has(o) || ArrayBuffer.isView(o)) return;
    if (typeof GPUBuffer !== "undefined" && o instanceof GPUBuffer) return;
    if (o instanceof SafeTensors) return;
    seen.add(o);
    if (o.kind && o.buf && o.N && o.K) {
      if (o.parts) for (const p of o.parts) reg.set(norm(p.name), { W: o, off: p.off, n: p.n });
      else if (o.name) reg.set(norm(o.name), { W: o, off: 0, n: o.N });
      return;
    }
    for (const [k, v] of Object.entries(o)) if (k !== "gpu" && k !== "st") walk(v);
  };
  walk(root);
  return reg;
}

// int8 (ConvRot) layers receive Hadamard-rotated inputs x_r = x H, and x A^T = x_r (A H)^T, so A
// is rotated once at load time (per 256-feature group, the same transform as the shader).
export function hadamardRows(data, rows, K) {
  for (let r = 0; r < rows; r++) {
    for (let g = 0; g < K; g += 256) {
      const base = r * K + g;
      for (let stride = 1; stride < 256; stride *= 4) {
        for (let t = 0; t < 64; t++) {
          const i0 = base + Math.floor(t / stride) * stride * 4 + (t % stride);
          const a = data[i0], b = data[i0 + stride], c = data[i0 + 2 * stride], e = data[i0 + 3 * stride];
          data[i0] = a + b + c - e;
          data[i0 + stride] = a + b - c + e;
          data[i0 + 2 * stride] = a - b + c + e;
          data[i0 + 3 * stride] = -a + b + c + e;
        }
      }
      for (let i = 0; i < 256; i++) data[base + i] *= 0.0625;
    }
  }
}

function findTarget(module, regs) {
  let te = false;
  let name = module;
  for (const p of TE_PREFIXES) if (name.startsWith(p)) { name = name.slice(p.length); te = true; break; }
  if (!te) for (const p of DIT_PREFIXES) if (name.startsWith(p)) { name = name.slice(p.length); break; }
  const key = norm(name);
  const order = te ? [regs.te] : [regs.dit, regs.te];
  for (const reg of order) {
    if (!reg) continue;
    if (reg.has(key)) return reg.get(key);
    if (reg.has("model_" + key)) return reg.get("model_" + key); // kohya TE names omit "model."
  }
  return null;
}

// Parses a LoRA file and uploads its matrices for the given model. Returns
// { modules: [{ target, A, B, r, alphaScale }], matched, total, unsupported, skipped }.
export async function loadLora(gpu, blob, regs) {
  const st = await SafeTensors.open(blob);
  const groups = new Map();
  let unsupported = 0;
  for (const key of Object.keys(st.header)) {
    if (UNSUPPORTED.test(key)) { unsupported++; continue; }
    const suf = SUFFIXES.find(([s]) => key.endsWith(s));
    if (!suf) continue;
    const module = key.slice(0, -suf[0].length);
    if (!groups.has(module)) groups.set(module, {});
    groups.get(module)[suf[1]] = key;
  }
  const modules = [];
  let skipped = 0;
  for (const [module, keys] of groups) {
    if (!keys.A || !keys.B) { skipped++; continue; }
    const target = findTarget(module, regs);
    const aInfo = st.info(keys.A), bInfo = st.info(keys.B);
    const r = aInfo.shape[0];
    if (!target || aInfo.shape[1] !== target.W.K || bInfo.shape[0] !== target.n || bInfo.shape[1] !== r) { skipped++; continue; }
    const A = new Float32Array(await st.f32(keys.A)); // own copy: rotated in place for int8 layers
    if (target.W.kind === "i8") hadamardRows(A, r, target.W.K);
    const B = await st.f32(keys.B);
    const alpha = keys.alpha ? (await st.f32(keys.alpha))[0] : null;
    modules.push({ target, A: gpu.upload(A), B: gpu.upload(B), r, alphaScale: alpha ? alpha / r : 1 });
  }
  return { modules, matched: modules.length, total: groups.size, unsupported, skipped };
}

export function destroyLora(L) {
  for (const m of L.modules) { m.A.destroy(); m.B.destroy(); }
}

// Replace every LoRA side path in the registries with the given [{ lora, strength }].
export function attachLoras(regs, active) {
  for (const reg of [regs.dit, regs.te]) if (reg) for (const { W } of reg.values()) W.lora = [];
  for (const { lora, strength } of active) {
    if (!strength) continue;
    for (const m of lora.modules) {
      m.target.W.lora.push({ A: m.A, B: m.B, r: m.r, scale: strength * m.alphaScale, off: m.target.off, n: m.target.n });
    }
  }
}
