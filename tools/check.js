import { Nanosaur2Pipeline } from "../app/pipeline.js";
import { simpleSigmas } from "../app/samplers.js";

const q = new URLSearchParams(location.search);
const base = new URL(q.get("models") || "../models/", location.href);
const dumpDir = new URL(q.get("dump") || "../out/dump/", location.href);
const only = (q.get("only") || "tok,te,dit,full,vae").split(",");
const logEl = document.getElementById("log");
const log = (s, cls) => {
  const span = document.createElement("span");
  if (cls) span.className = cls;
  span.textContent = s + "\n";
  logEl.append(span);
  console.log(s);
};
window.__checkDone = false;
window.__checkResults = {};

async function loadDump(name) {
  const r = await fetch(new URL(name + ".bin", dumpDir));
  if (!r.ok) throw new Error(`missing dump ${name}`);
  return new Float32Array(await r.arrayBuffer());
}

function compare(name, got, ref, tol) {
  let maxAbs = 0, num = 0, den = 0;
  for (let i = 0; i < ref.length; i++) {
    const d = got[i] - ref[i];
    maxAbs = Math.max(maxAbs, Math.abs(d));
    num += d * d;
    den += ref[i] * ref[i];
  }
  const rel = Math.sqrt(num / Math.max(den, 1e-30));
  const ok = rel < tol && Number.isFinite(rel) && got.length >= ref.length;
  window.__checkResults[name] = { rel, maxAbs, ok };
  log(`${ok ? "PASS" : "FAIL"} ${name}: rel L2 ${rel.toExponential(3)}  max|d| ${maxAbs.toExponential(3)}  (n=${ref.length}, tol ${tol})`, ok ? "ok" : "bad");
  return ok;
}

try {
  const meta = await (await fetch(new URL("index.json", dumpDir))).json();
  log(`model ${meta.model} (${meta.precision || "bf16"}), ${meta.width}x${meta.height}, ${meta.steps} steps, ${meta.sampler}, cfg ${meta.cfg}${meta.cfg !== 1 ? ` (${meta.guidance})` : ""}`);
  const pipe = new Nanosaur2Pipeline();
  let lastPct = -1;
  let t = performance.now();
  await pipe.load({ upstream: base, builds: base }, { model: meta.model, precision: meta.precision || "bf16" }, {
    onStatus: (s) => {
      if (s.phase === "download") {
        const pct = Math.floor((100 * s.done) / s.total);
        if (pct !== lastPct) { lastPct = pct; if (pct % 10 === 0) log(`download ${pct}%`); }
      } else if (s.phase === "load" && s.frac === 0) log(`loading ${s.what}`);
    },
  });
  const gpu = pipe.gpu;
  log(`loaded in ${(performance.now() - t).toFixed(0)} ms; adapter: ${gpu.info.vendor || ""} ${gpu.info.architecture || ""} ${gpu.info.description || ""}; maxBinding ${(gpu.maxBinding / 2 ** 20) | 0} MiB`);

  if (meta.loras?.length) {
    const specs = await Promise.all(meta.loras.map(async (l) => ({ key: l.path, strength: l.strength, blob: await (await fetch(new URL("../../" + l.path, dumpDir))).blob() })));
    const report = await pipe.setLoras(specs);
    for (const r of report) log(`lora ${r.key}: ${r.matched}/${r.total} modules applied${r.unsupported ? `, ${r.unsupported} unsupported tensors` : ""}`);
  }

  if (only.includes("tok")) {
    const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
    const tok = pipe.tokenizer.encode(meta.prompt);
    const wOk = tok.weights.every((w, i) => Math.abs(w - meta.weights[i]) < 1e-6);
    log(`tokens ${same(tok.ids, meta.ids) && wOk ? "match" : "MISMATCH"} (${tok.ids.length})`, same(tok.ids, meta.ids) && wOk ? "ok" : "bad");
  }

  const h = meta.height / 16, w = meta.width / 16;
  if (only.includes("te")) {
    t = performance.now();
    const hidden = await pipe.te.encode(meta.ids);
    const hv = await gpu.read(hidden);
    hidden.release();
    log(`text encoder ${(performance.now() - t).toFixed(0)} ms`);
    compare("te_hidden", hv, await loadDump("te_hidden"), 1e-4);
  }

  if (only.includes("dit")) {
    const text = pipe.dit.prepareText(await loadDump("te_hidden"), meta.weights);
    const noise = await loadDump("noise");
    const sigma = simpleSigmas(meta.steps, 3)[0];
    const x = noise.map((v) => v * sigma);
    for (let rep = 0; rep < 2; rep++) {
      t = performance.now();
      const x0 = await pipe.dit.forward(x, h, w, text, sigma);
      log(`dit forward ${meta.width}x${meta.height}: ${(performance.now() - t).toFixed(0)} ms${rep === 0 ? " (includes shader compile)" : ""}`);
      if (rep === 0) compare("x0_cond0", x0, await loadDump("x0_cond0"), 1e-4);
    }
    text.release();
    if (meta.cfg !== 1) {
      const neg = pipe.dit.prepareText(await loadDump("te_hidden_neg"), meta.neg_weights);
      compare("x0_uncond0", await pipe.dit.forward(x, h, w, neg, sigma), await loadDump("x0_uncond0"), 1e-4);
      t = performance.now();
      const pd = await pipe.dit.forward(x, h, w, neg, sigma, { pathDrop: true });
      log(`path-drop forward: ${(performance.now() - t).toFixed(0)} ms`);
      compare("x0_pathdrop0", pd, await loadDump("x0_pathdrop0"), 1e-4);
      neg.release();
    }
  }

  if (only.includes("full")) {
    // the whole sampling loop through the real pipeline vs the reference
    const res = await pipe.generate({
      prompt: meta.prompt, negative: meta.negative, cfg: meta.cfg, guidance: meta.guidance,
      width: meta.width, height: meta.height, steps: meta.steps, sampler: meta.sampler, seed: meta.seed,
    });
    log(`full generate ${(res.timings.total / 1000).toFixed(1)} s (${(res.timings.perStep / 1000).toFixed(2)} s/step, decode ${(res.timings.decode / 1000).toFixed(2)} s)`);
    compare("latent_final", res.latent, await loadDump("latent_final"), 2e-2);
  }

  if (only.includes("vae")) {
    const lat = await loadDump("latent_final");
    for (let rep = 0; rep < 2; rep++) {
      t = performance.now();
      const img = await pipe.vae.decode(lat, h, w);
      log(`vae decode ${(performance.now() - t).toFixed(0)} ms${rep === 0 ? " (includes shader compile)" : ""}`);
      if (rep) continue;
      const ref = await loadDump("image"); // [3, H, W] in [0,1]
      const got = new Float32Array(ref.length);
      const HW = img.width * img.height;
      for (let i = 0; i < HW; i++) for (let c = 0; c < 3; c++) got[c * HW + i] = img.data[i * 4 + c] / 255;
      compare("image", got, ref, 1e-2);
      const cv = document.getElementById("c");
      cv.width = img.width; cv.height = img.height;
      cv.getContext("2d").putImageData(new ImageData(img.data, img.width, img.height), 0, 0);
    }
  }
  log("done");
} catch (e) {
  log("ERROR " + (e.stack || e), "bad");
  window.__checkError = String(e.stack || e);
}
window.__checkDone = true;
