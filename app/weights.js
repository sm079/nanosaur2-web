// Reads safetensors files (from a Blob/File, e.g. the OPFS cache) and uploads weights.
//
// Linear weight objects: { kind: "bf16" | "i8", N, K, buf, scale? (i8), bias?, name }. Weights go to
// the GPU as stored (the original bf16 checkpoints, or the int8 builds from
// tools/build_assets.py) and are decoded in the GEMM.

const BYTES = { F32: 4, BF16: 2, F16: 2, I8: 1, U8: 1 };

export function bf16ToF32(u8) {
  const u16 = new Uint16Array(u8.buffer, u8.byteOffset, u8.byteLength / 2);
  const out = new Float32Array(u16.length);
  const o32 = new Uint32Array(out.buffer);
  for (let i = 0; i < u16.length; i++) o32[i] = u16[i] << 16;
  return out;
}

function f16ToF32(u8) {
  const u16 = new Uint16Array(u8.buffer, u8.byteOffset, u8.byteLength / 2);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 31;
    const m = h & 1023;
    out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}

// f32 -> bf16 (round to nearest even), for weights stored in another float format
function f32ToBf16(f) {
  const u32 = new Uint32Array(f.buffer, f.byteOffset, f.length);
  const out = new Uint16Array(f.length);
  for (let i = 0; i < f.length; i++) {
    const x = u32[i];
    out[i] = (x + 0x7fff + ((x >>> 16) & 1)) >>> 16;
  }
  return new Uint8Array(out.buffer);
}

export class SafeTensors {
  static async open(blob, prefix = "") {
    const head = new DataView(await blob.slice(0, 8).arrayBuffer());
    const n = Number(head.getBigUint64(0, true));
    const header = JSON.parse(new TextDecoder().decode(await blob.slice(8, 8 + n).arrayBuffer()));
    delete header.__metadata__;
    return new SafeTensors(blob, header, 8 + n, prefix);
  }

  constructor(blob, header, dataStart, prefix) {
    this.blob = blob;
    this.header = header;
    this.dataStart = dataStart;
    this.prefix = prefix;
  }

  has(name) {
    return this.prefix + name in this.header;
  }

  info(name) {
    const t = this.header[this.prefix + name];
    if (!t) throw new Error(`missing tensor ${this.prefix + name}`);
    return t;
  }

  async bytes(name) {
    const t = this.info(name);
    const [a, b] = t.data_offsets;
    return new Uint8Array(await this.blob.slice(this.dataStart + a, this.dataStart + b).arrayBuffer());
  }

  async f32(name) {
    const t = this.info(name);
    const u8 = await this.bytes(name);
    if (t.dtype === "F32") return new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4);
    if (t.dtype === "BF16") return bf16ToF32(u8);
    if (t.dtype === "F16") return f16ToF32(u8);
    // other types show up in small tensors, e.g. LoRA alphas saved as integers
    const buf = u8.slice().buffer; // copy: typed-array views need an aligned offset
    const other = { F64: Float64Array, I64: BigInt64Array, I32: Int32Array, I16: Int16Array, I8: Int8Array, U8: Uint8Array, BOOL: Uint8Array }[t.dtype];
    if (other) return Float32Array.from(new other(buf), Number);
    throw new Error(`unsupported dtype ${t.dtype} for ${name}`);
  }

  // weight as bf16 bytes, whatever float format it is stored in
  async bf16(name) {
    const t = this.info(name);
    if (t.dtype === "BF16") return this.bytes(name);
    return f32ToBf16(await this.f32(name));
  }

  quantMeta(base) {
    return this.has(base + "comfy_quant") ? this.bytes(base + "comfy_quant").then((b) => JSON.parse(new TextDecoder().decode(b))) : null;
  }

  // Gather rows of a 2D table (embeddings) without reading the whole tensor.
  async rows(name, ids) {
    const t = this.info(name);
    const [, dim] = t.shape;
    const rowBytes = dim * BYTES[t.dtype];
    const out = new Float32Array(ids.length * dim);
    await Promise.all(ids.map(async (id, i) => {
      const off = this.dataStart + t.data_offsets[0] + id * rowBytes;
      const u8 = new Uint8Array(await this.blob.slice(off, off + rowBytes).arrayBuffer());
      out.set(t.dtype === "BF16" ? bf16ToF32(u8) : t.dtype === "F16" ? f16ToF32(u8) : new Float32Array(u8.buffer), i * dim);
    }));
    return out;
  }

  // f32 vector on the GPU; add: a constant added to every element (Gemma stores norm weights as w - 1)
  async vector(gpu, name, add = 0) {
    const v = await this.f32(name);
    return gpu.upload(add ? v.map((x) => x + add) : v);
  }

  // Loads "<base>weight" and optional "<base>bias". The result carries its module name (base
  // without the trailing dot) so LoRAs can find it. Conv weights [N, ...] are flattened to [N, K].
  // int8 builds (tools/build_assets.py) store ConvRot-quantized linears in ComfyUI's
  // comfy_quant layout: int8 weight + per-row weight_scale, decoded in the GEMM.
  async linear(gpu, base) {
    const w = this.info(base + "weight");
    const N = w.shape[0];
    const K = w.shape.slice(1).reduce((a, b) => a * b, 1);
    const bias = this.has(base + "bias") ? await this.vector(gpu, base + "bias") : null;
    const name = base.replace(/\.$/, "");
    const meta = await this.quantMeta(base);
    if (meta) {
      if (meta.format !== "int8_tensorwise" || !meta.convrot) throw new Error(`${base}: unsupported quantization ${JSON.stringify(meta)}`);
      return { kind: "i8", N, K, buf: gpu.upload(await this.bytes(base + "weight")), scale: await this.vector(gpu, base + "weight_scale"), bias, name };
    }
    return { kind: "bf16", N, K, buf: gpu.upload(await this.bf16(base + "weight")), bias, name };
  }

  // 3x3 conv weight [N, Cin, 3, 3] reordered to the GEMM's implicit im2col layout [N][tap][Cin].
  async conv3x3(gpu, base) {
    const w = this.info(base + "weight");
    const [N, C, kh, kw] = w.shape;
    if (kh !== 3 || kw !== 3) throw new Error(`${base}: expected a 3x3 conv`);
    const src = new Uint16Array((await this.bf16(base + "weight")).slice().buffer);
    const dst = new Uint16Array(src.length);
    for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) for (let t = 0; t < 9; t++) dst[(n * 9 + t) * C + c] = src[(n * C + c) * 9 + t];
    const bias = this.has(base + "bias") ? await this.vector(gpu, base + "bias") : null;
    return { kind: "bf16", N, K: 9 * C, buf: gpu.upload(new Uint8Array(dst.buffer)), bias, name: base.replace(/\.$/, "") };
  }
}

// Stack bf16 linears that share an input into one [sum N, K] weight (e.g. q, k, v -> qkv), so
// one GEMM replaces several. Biases are stacked too (zeros for parts without one).
export function concatLinears(gpu, Ws) {
  const { K } = Ws[0];
  if (!Ws.every((w) => w.kind === "bf16" && w.K === K)) throw new Error("concatLinears: mismatched weights");
  const out = { kind: "bf16", K, N: Ws.reduce((a, w) => a + w.N, 0), parts: [] };
  let colOff = 0;
  for (const w of Ws) { out.parts.push({ name: w.name, off: colOff, n: w.N }); colOff += w.N; }
  const enc = gpu.device.createCommandEncoder();
  const stack = (field, bytes) => {
    const dst = gpu.device.createBuffer({ size: Math.ceil(Ws.reduce((a, w) => a + bytes(w), 0) / 16) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    let off = 0;
    for (const w of Ws) {
      if (w[field]) enc.copyBufferToBuffer(w[field], 0, dst, off, bytes(w));
      off += bytes(w);
    }
    return dst;
  };
  out.buf = stack("buf", (w) => w.N * K * 2);
  if (Ws.some((w) => w.bias)) out.bias = stack("bias", (w) => w.N * 4); // new buffers are zeroed
  gpu.flush();
  gpu.device.queue.submit([enc.finish()]);
  // the sources are no longer needed once the copies have run (destroy waits for the queue)
  for (const w of Ws) { w.buf.destroy(); w.bias?.destroy(); }
  return out;
}
