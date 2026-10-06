// Nanosaur2 text-to-image pipeline: download-once model files -> WebGPU models -> sampling -> VAE.

import { GPU } from "./gpu/device.js";
import { SafeTensors } from "./weights.js";
import { cachedFile, requestPersistence } from "./store.js";
import { Nanosaur2Tokenizer } from "./tokenizer.js";
import { Gemma3 } from "./models/gemma3.js";
import { Nanosaur2DiT } from "./models/nanosaur2.js";
import { VAEDecoder } from "./models/vae.js";
import { TorchGenerator } from "./rng.js";
import { sample, SCHEDULERS } from "./samplers.js";
import { linearRegistry, loadLora, attachLoras, destroyLora } from "./lora.js";
import { PREVIEW_FACTORS, PREVIEW_BIAS } from "./preview.js";

// Model files: the DiT in two precisions. bf16: the original checkpoints, read as they are from the
// model repo ("upstream"). int8: DiTs with int8 block linears, built by tools/build_assets.py and
// hosted separately ("builds"). The text encoder is the original; the VAE is the original's decoder
// half (a lossless subset, also from the builds: the DINOv2 encoder is never used).
const up = (path, size) => ({ path, size, src: "upstream" });
const built = (path, size) => ({ path, size, src: "builds" });
export const MODELS = [
  {
    id: "4step", label: "4-step", family: "fast",
    blurb: "Distilled for 4 steps without guidance: an image in a few seconds. The best place to start.",
    dit: {
      bf16: up("nanosaur2_dmad_4step_diffusion_model.safetensors", 1329693704),
      int8: built("nanosaur2_dmad_4step_diffusion_model.int8.safetensors", 722398936),
    },
  },
  {
    id: "base", label: "Normal", family: "full",
    blurb: "The base model with guidance and a negative prompt, 30–50 steps: more varied and controllable, many times slower.",
    dit: {
      bf16: up("nanosaur2_diffusion_model.safetensors", 1329693600),
      int8: built("nanosaur2_diffusion_model.int8.safetensors", 722398832),
    },
  },
];
export const SHARED_FILES = {
  te: up("nanosaur2_text_encoder.safetensors", 541023274),
  vae: built("nanosaur2_vae_decoder.safetensors", 83150422),
};
export const PRECISIONS = ["int8", "bf16"];
export const LATENT_SCALE = 16; // image pixels per latent pixel
const CH = 64;
const PATH_DROP_COST = 4 / 18; // a path-drop pass runs 4 of the 18 blocks

export function resolveFiles(modelId, precision = "bf16") {
  const model = MODELS.find((m) => m.id === modelId);
  if (!model) throw new Error(`unknown model ${modelId}`);
  if (!PRECISIONS.includes(precision)) throw new Error(`unknown precision ${precision}`);
  return { model, files: { dit: model.dit[precision], ...SHARED_FILES } };
}

// Fast preview: a linear projection of the x0 prediction to RGB at latent resolution.
function latentPreview(latent, h, w) {
  const img = new Uint8ClampedArray(h * w * 4);
  const hw = h * w;
  for (let i = 0; i < hw; i++) {
    for (let c = 0; c < 3; c++) {
      let v = PREVIEW_BIAS[c];
      for (let k = 0; k < CH; k++) v += latent[k * hw + i] * PREVIEW_FACTORS[k * 3 + c];
      img[i * 4 + c] = ((v + 1) / 2) * 255;
    }
    img[i * 4 + 3] = 255;
  }
  return { data: img, width: w, height: h };
}

export class Nanosaur2Pipeline {
  // gpuOptions: passed to GPU.create (e.g. { profile: true } for profiling)
  constructor(gpuOptions = {}) {
    this.gpuOptions = gpuOptions;
    this.gpu = null;
    this.loaded = {}; // component -> path of the file currently on the GPU
    this.model = null;
    this.textCache = new Map();
    this.loraSpec = [];
    this.loraCache = new Map(); // key -> uploaded LoRA for the current models
  }

  isLoaded(files) {
    return ["dit", "te", "vae"].every((k) => this.loaded[k] === files[k].path);
  }

  // Downloads (first time only) and loads the selected model ({ model, precision }); components
  // that are already on the GPU with the same file are kept. bases: { upstream, builds } URLs.
  // token: optional Hugging Face access token, sent only when the files come from huggingface.co
  async load(bases, selection, { onStatus = () => {}, signal, token = "" } = {}) {
    await requestPersistence();
    this.gpu = this.gpu || (await GPU.create(this.gpuOptions));
    const gpu = this.gpu;
    const { model, files: want } = resolveFiles(selection.model, selection.precision);

    const parts = ["te", "vae", "dit"].filter((k) => this.loaded[k] !== want[k].path);
    const total = parts.reduce((a, k) => a + want[k].size, 0);
    const got = Object.fromEntries(parts.map((k) => [k, 0]));
    const files = {};
    for (const k of parts) {
      const f = want[k];
      const base = new URL(bases[f.src]);
      const headers = token && base.hostname === "huggingface.co" ? { Authorization: `Bearer ${token}` } : {};
      files[k] = await cachedFile(new URL(f.path, base).href, f.path, f.size, (done) => {
        got[k] = done;
        onStatus({ phase: "download", file: f.path, done: parts.reduce((a, p) => a + got[p], 0), total });
      }, signal, headers);
    }

    for (const k of parts) this.unloadPart(k);
    // cached prompt encodings depend on the text encoder and the DiT's text embedder
    if (parts.includes("te") || parts.includes("dit")) this.clearText();
    const labels = { te: "text encoder", dit: "diffusion model", vae: "VAE" };
    for (const k of parts) {
      onStatus({ phase: "load", what: labels[k], frac: 0 });
      const progress = (f) => onStatus({ phase: "load", what: labels[k], frac: f });
      const st = await SafeTensors.open(files[k]);
      if (k === "te") {
        this.tokenizer = new Nanosaur2Tokenizer(await st.bytes("spiece_model"));
        this.te = await Gemma3.load(gpu, st, progress);
      }
      if (k === "dit") this.dit = await Nanosaur2DiT.load(gpu, st, progress);
      if (k === "vae") this.vae = await VAEDecoder.load(gpu, st);
      this.loaded[k] = want[k].path;
    }
    // LoRA side paths hang off the model's linears: re-attach after a model change
    if (parts.includes("te") || parts.includes("dit")) {
      for (const L of this.loraCache.values()) destroyLora(L);
      this.loraCache.clear();
      if (this.loraSpec.length) {
        onStatus({ phase: "load", what: "LoRAs", frac: 1 });
        await this.applyLoras();
      }
    }
    await gpu.sync();
    this.model = model;
    this.selection = { ...selection };
    onStatus({ phase: "ready" });
  }

  // list: [{ key, blob, strength }] (blob: File/Blob of the .safetensors). Returns a report per
  // LoRA: { key, matched, total, unsupported } so the UI can flag files that don't fit.
  async setLoras(list) {
    this.loraSpec = list;
    return this.applyLoras();
  }

  async applyLoras() {
    if (!this.dit || !this.te) return [];
    const regs = { dit: linearRegistry(this.dit), te: linearRegistry(this.te) };
    const active = [];
    const report = [];
    for (const spec of this.loraSpec) {
      let L = this.loraCache.get(spec.key);
      if (!L) {
        L = await loadLora(this.gpu, spec.blob, regs);
        this.loraCache.set(spec.key, L);
      }
      active.push({ lora: L, strength: spec.strength });
      report.push({ key: spec.key, matched: L.matched, total: L.total, unsupported: L.unsupported });
    }
    // drop files that are no longer used
    for (const [k, L] of this.loraCache) {
      if (!this.loraSpec.some((s) => s.key === k)) { destroyLora(L); this.loraCache.delete(k); }
    }
    attachLoras(regs, active);
    this.clearText(); // the text encoder and the text embedder may carry LoRAs
    return report;
  }

  unloadPart(k) {
    // weights are plain GPU buffers; destroy eagerly so a switch doesn't briefly need memory for both
    const destroy = (o) => {
      if (!o || typeof o !== "object") return;
      if (o instanceof GPUBuffer) { o.destroy(); return; }
      if (ArrayBuffer.isView(o)) return;
      for (const v of Array.isArray(o) ? o : Object.values(o)) if (v && typeof v === "object" && v !== this.gpu && !(v instanceof SafeTensors)) destroy(v);
    };
    if (this[k]) destroy(this[k]);
    this[k] = null;
    delete this.loaded[k];
  }

  clearText() {
    for (const t of this.textCache.values()) t.release();
    this.textCache.clear();
  }

  unload() {
    for (const k of ["te", "dit", "vae"]) this.unloadPart(k);
    for (const L of this.loraCache.values()) destroyLora(L);
    this.loraCache.clear();
    this.clearText();
    this.gpu?.pool.trim();
    this.model = null;
  }

  // Prompt -> text input for the DiT. Only the prompts of the current image are kept.
  async encode(text, keep = [text]) {
    for (const [k, t] of this.textCache) if (!keep.includes(k)) { t.release(); this.textCache.delete(k); }
    if (this.textCache.has(text)) return this.textCache.get(text);
    const { ids, weights } = this.tokenizer.encode(text);
    const hidden = await this.te.encode(ids);
    const hv = await this.gpu.read(hidden);
    hidden.release();
    const t = this.dit.prepareText(hv, weights);
    this.textCache.set(text, t);
    return t;
  }

  // cfg 1: one DiT pass per step (the 4-step model). cfg != 1: classifier-free guidance,
  // uncond + cfg * (cond - uncond), with the unconditional pass chosen by `guidance`:
  //   "cfg"        the full model on the negative prompt
  //   "path_drop"  the negative prompt without the sparse middle blocks (SPRINT path drop)
  //   "alternate"  cfg on even steps, path drop on odd steps (the model's recommendation)
  async generate(opts) {
    const { prompt, negative = "", cfg = 1, guidance = "alternate", width, height, sampler, seed, shift = 3, scheduler = "simple", onProgress = () => {}, onPreview, signal } = opts;
    const gpu = this.gpu;
    const h = height / LATENT_SCALE, w = width / LATENT_SCALE;
    const t0 = performance.now();
    const guided = cfg !== 1;
    onProgress({ phase: "encode" });
    const keep = guided ? [prompt, negative] : [prompt];
    const cond = await this.encode(prompt, keep);
    const uncond = guided ? await this.encode(negative, keep) : null;

    const sigmas = (SCHEDULERS[scheduler] || SCHEDULERS.simple)(opts.steps, shift);
    const steps = sigmas.length - 1; // the beta scheduler can merge duplicate timesteps
    const noise = new TorchGenerator(seed).randn(CH * h * w);
    let x = noise.map((v) => v * sigmas[0]);
    const tSample = performance.now();
    let passes = 0, passUnits = 0;

    const check = () => { if (signal?.aborted) throw new DOMException("Generation cancelled", "AbortError"); };
    const pathDropAt = (i) => guidance === "path_drop" || (guidance === "alternate" && i % 2 === 1);
    const denoise = async (xin, sigma, i) => {
      // progress within the step, weighting each pass by the blocks it runs
      const units = 1 + (guided ? (pathDropAt(i) ? PATH_DROP_COST : 1) : 0);
      let base = 0;
      const pass = async (text, pathDrop) => {
        const cost = pathDrop ? PATH_DROP_COST : 1;
        const x0 = await this.dit.forward(xin, h, w, text, sigma, {
          pathDrop,
          onBlock: async (f) => {
            check();
            onProgress({ phase: "sample", step: i, steps, frac: (i + (base + f * cost) / units) / steps });
            await gpu.sync(); // keep the queue short so cancel/progress stay responsive
          },
        });
        base += cost;
        passes++;
        passUnits += cost;
        return x0;
      };
      const den = await pass(cond, false);
      if (guided) {
        const un = await pass(uncond, pathDropAt(i));
        for (let j = 0; j < den.length; j++) den[j] = un[j] + cfg * (den[j] - un[j]);
      }
      return den;
    };

    let image, tVae;
    try {
      x = await sample(sampler, denoise, x, sigmas, {
        seed, shift,
        onStep: (i, xi, den) => { if (onPreview) onPreview(latentPreview(den, h, w), i); },
      });
      check();
      tVae = performance.now();
      onProgress({ phase: "decode", frac: 0 });
      image = await this.vae.decode(x, h, w, (f) => onProgress({ phase: "decode", frac: f }));
    } finally {
      // free pooled activations (the VAE's full-resolution tensors dominate) so idle VRAM is
      // just the weights; also runs after a cancel
      await gpu.sync().catch(() => {});
      gpu.pool.trim();
    }
    const t1 = performance.now();
    const sampleMs = tVae - tSample;
    return {
      image,
      latent: x,
      timings: { encode: tSample - t0, sample: sampleMs, decode: t1 - tVae, total: t1 - t0, perStep: sampleMs / steps, perPass: sampleMs / Math.max(passUnits, 1), passes },
    };
  }
}
