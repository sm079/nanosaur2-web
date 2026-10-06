"""PyTorch reference implementation of the Nanosaur2 pipeline, in fp32, reading the original
model files.

It exists to (1) validate the port end to end by producing an image and (2) dump intermediate
tensors that the WebGPU engine is checked against (tools/check.html).

  python tools/reference.py --model 4step --prompt "..." --seed 42 --out out/ref.png [--dump out/dump]

Needs torch, safetensors and sentencepiece. The model files go in models/ (see README).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FILES = {
    "bf16": {
        "4step": "nanosaur2_dmad_4step_diffusion_model.safetensors",
        "base": "nanosaur2_diffusion_model.safetensors",
        "te": "nanosaur2_text_encoder.safetensors",
        "vae": "nanosaur2_vae.safetensors",
    },
    # tools/build_assets.py
    "int8": {
        "4step": "nanosaur2_dmad_4step_diffusion_model.int8.safetensors",
        "base": "nanosaur2_diffusion_model.int8.safetensors",
        "te": "nanosaur2_text_encoder.safetensors",
        "vae": "nanosaur2_vae_decoder.safetensors",
    },
}


class Weights:
    def __init__(self, path: str, loras=()):
        self.f = safe_open(path, "pt", device="cpu")
        self.keys = set(self.f.keys())
        self.deltas = lora_deltas(loras)

    def __call__(self, name: str) -> torch.Tensor:
        base = name[: -len("weight")] if name.endswith("weight") else None
        if base is not None and base + "comfy_quant" in self.keys:
            # int8 builds: dequantize on access
            w = quant.dequantize({s: self.f.get_tensor(base + s).to(DEV) for s in ("weight", "weight_scale", "comfy_quant")})
        else:
            w = self.f.get_tensor(name).to(DEV).float()
        if name.endswith(".weight"):
            for scale, B, A in self.deltas.get(name[: -len(".weight")].replace(".", "_"), []):
                w = w + scale * (B @ A)
        return w

    def has(self, name: str) -> bool:
        return name in self.keys


def lora_deltas(loras):
    """name (dots as underscores) -> [(scale, B, A)], merged into the weights on access.
    Same key conventions as app/lora.js (PEFT lora_A/B or kohya lora_down/up + alpha)."""
    suffixes = [(".lora_A.weight", "A"), (".lora_B.weight", "B"), (".lora_down.weight", "A"), (".lora_up.weight", "B"), (".alpha", "alpha")]
    prefixes = ["lora_unet_", "model.diffusion_model.", "diffusion_model.", "lora_te_", "text_encoder."]
    out = {}
    for path, strength in loras:
        f = safe_open(path, "pt", device="cpu")
        groups = {}
        for k in f.keys():
            for suf, role in suffixes:
                if k.endswith(suf):
                    groups.setdefault(k[: -len(suf)], {})[role] = k
        for module, ks in groups.items():
            name = module
            for p in prefixes:
                if name.startswith(p):
                    name = name[len(p):]
                    break
            A = f.get_tensor(ks["A"]).float().to(DEV)
            B = f.get_tensor(ks["B"]).float().to(DEV)
            scale = strength * (f.get_tensor(ks["alpha"]).float().item() / A.shape[0] if "alpha" in ks else 1.0)
            out.setdefault(name.replace(".", "_"), []).append((scale, B, A))
    return out


def rms_norm(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope_split_half(x, cos, sin):
    """x [..., L, D]; pair i is (x[i], x[i + D/2]); cos/sin [L, D/2]."""
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ----------------------------------------------------------------------------- tokenizer

def parse_prompt_emphasis(caption):
    """Copy of nanosaur2_support/text_encoder.py parse_prompt_emphasis."""
    weight_pattern = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")
    spans, parts = [], []
    cursor = output_len = idx = 0
    while idx < len(caption):
        if caption[idx] != "(":
            idx += 1
            continue
        depth, end = 1, idx + 1
        while end < len(caption) and depth > 0:
            depth += {"(": 1, ")": -1}.get(caption[end], 0)
            end += 1
        if depth != 0:
            idx += 1
            continue
        inner = caption[idx + 1:end - 1]
        inner_depth, colon_idx = 0, -1
        for i, ch in enumerate(inner):
            if ch == "(":
                inner_depth += 1
            elif ch == ")":
                inner_depth -= 1
            elif ch == ":" and inner_depth == 0:
                colon_idx = i
        text, weight_text = inner[:colon_idx], inner[colon_idx + 1:].strip()
        if colon_idx == -1 or not text or not weight_pattern.fullmatch(weight_text):
            idx += 1
            continue
        parts.append(caption[cursor:idx])
        output_len += idx - cursor
        parts.append(text)
        spans.append((output_len, output_len + len(text), float(weight_text)))
        output_len += len(text)
        cursor = idx = end
    parts.append(caption[cursor:])
    return "".join(parts), spans


class Tokenizer:
    def __init__(self, te_path):
        import sentencepiece as spm
        data = safe_open(te_path, "pt", device="cpu").get_tensor("spiece_model").numpy().tobytes()
        self.sp = spm.SentencePieceProcessor(model_proto=data)

    def encode(self, text):
        plain, spans = parse_prompt_emphasis(text)
        ids, weights = [2], [1.0]
        for piece in self.sp.encode(plain, out_type="immutable_proto").pieces:
            w = 1.0
            for b, e, sw in spans:
                if piece.begin < e and piece.end > b:
                    w *= sw
            ids.append(piece.id)
            weights.append(w)
        return ids[:256], weights[:256]


# ----------------------------------------------------------------------------- Gemma3-270M

def gemma_rope(L, theta, D=256):
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, device=DEV).float() / D))
    a = torch.arange(L, device=DEV).float()[:, None] * inv[None]
    return a.cos(), a.sin()


@torch.no_grad()
def gemma_encode(W: Weights, ids):
    """Hidden state after layer 16 of 18, with the final norm (layer_idx -2)."""
    H, KH, HD = 4, 1, 256
    x = W("model.embed_tokens.weight")[torch.tensor(ids, device=DEV)] * math.sqrt(640)
    L = x.shape[0]
    ropes = {t: gemma_rope(L, t) for t in (1e4, 1e6)}
    mask = torch.full((L, L), float("-inf"), device=DEV).triu(1)
    for i in range(17):
        p = f"model.layers.{i}."
        n = lambda k: W(p + k + ".weight") + 1.0  # noqa: E731
        cos, sin = ropes[1e6 if (i + 1) % 6 == 0 else 1e4]
        h = rms_norm(x, n("input_layernorm"))
        q = (h @ W(p + "self_attn.q_proj.weight").T).view(L, H, HD).transpose(0, 1)
        k = (h @ W(p + "self_attn.k_proj.weight").T).view(L, KH, HD).transpose(0, 1)
        v = (h @ W(p + "self_attn.v_proj.weight").T).view(L, KH, HD).transpose(0, 1)
        q = rope_split_half(rms_norm(q, n("self_attn.q_norm")), cos, sin)
        k = rope_split_half(rms_norm(k, n("self_attn.k_norm")), cos, sin)
        k, v = k.repeat_interleave(H // KH, 0), v.repeat_interleave(H // KH, 0)
        a = F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=mask)[0]
        a = a.transpose(0, 1).reshape(L, H * HD) @ W(p + "self_attn.o_proj.weight").T
        x = x + rms_norm(a, n("post_attention_layernorm"))
        h = rms_norm(x, n("pre_feedforward_layernorm"))
        g = F.gelu(h @ W(p + "mlp.gate_proj.weight").T, approximate="tanh") * (h @ W(p + "mlp.up_proj.weight").T)
        x = x + rms_norm(g @ W(p + "mlp.down_proj.weight").T, n("post_feedforward_layernorm"))
    return rms_norm(x, W("model.norm.weight") + 1.0)


# ----------------------------------------------------------------------------- DiT

D, HEADS, HD = 1536, 16, 96


def lin(W, name, x):
    y = x @ W(name + ".weight").T
    return y + W(name + ".bias") if W.has(name + ".bias") else y


def timestep_embedding(t, dim=256):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=DEV) / half)
    a = t * freqs
    return torch.cat([a.cos(), a.sin()])[None]


def rope_2d(h, w):
    axis = HD // 2
    inv = 1.0 / (10000.0 ** (torch.arange(0, axis, 2, dtype=torch.float32, device=DEV) / axis))
    y = torch.arange(h, dtype=torch.float32, device=DEV) - (h - 1) / 2
    x = torch.arange(w, dtype=torch.float32, device=DEV) - (w - 1) / 2
    y, x = torch.meshgrid(y, x, indexing="ij")
    a = torch.cat([torch.outer(x.flatten(), inv), torch.outer(y.flatten(), inv)], dim=-1)
    return a.cos(), a.sin()


def attention(q, k, v, bias=None):
    """q [Lq, D], k/v [Lk, D] -> [Lq, D]; heads split off; bias [Lk] additive."""
    qh = q.view(q.shape[0], HEADS, HD).transpose(0, 1)
    kh = k.view(k.shape[0], HEADS, HD).transpose(0, 1)
    vh = v.view(v.shape[0], HEADS, HD).transpose(0, 1)
    mask = None if bias is None else bias[None, None, :].expand(1, q.shape[0], -1)
    o = F.scaled_dot_product_attention(qh[None], kh[None], vh[None], attn_mask=mask)[0]
    return o.transpose(0, 1).reshape(q.shape[0], D)


def head_norm(x, w):
    return rms_norm(x.view(x.shape[0], HEADS, HD), w).view(x.shape[0], D)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class DiT:
    def __init__(self, W: Weights):
        self.W = W

    def text_refine(self, txt, tc, keep_bias, keep):
        W = self.W
        for i in range(2):
            p = f"text_refine_blocks.{i}."
            m = lin(W, p + "adaLN_modulation.0", tc).chunk(6, -1)
            h = modulate(rms_norm(txt, W(p + "norm1.weight")), m[0], m[1])
            q, k, v = lin(W, p + "attn.qkv", h).chunk(3, -1)
            a = attention(head_norm(q, W(p + "attn.q_norm.weight")), head_norm(k, W(p + "attn.k_norm.weight")), v, keep_bias)
            txt = txt + m[2] * lin(W, p + "attn.proj", a)
            h = modulate(rms_norm(txt, W(p + "norm2.weight")), m[3], m[4])
            x1, x2 = lin(W, p + "mlp.w12", h).chunk(2, -1)
            txt = txt + m[5] * lin(W, p + "mlp.w3", F.silu(x1) * x2)
            txt = txt * keep[:, None]
        return txt

    def block(self, i, x, txt, cos, sin, mod, txt_bias):
        W = self.W
        p = f"blocks.{i}."
        m = (mod + W(f"encoder_adaLN_offsets.{i}")).chunk(6, -1)
        h = modulate(rms_norm(x, W(p + "norm1.weight")), m[0], m[1])
        q, k, v = lin(W, p + "attn.qkv_x", h).chunk(3, -1)
        L = x.shape[0]

        def rope(t, wn):
            th = rms_norm(t.view(L, HEADS, HD), W(wn)).transpose(0, 1)
            return rope_split_half(th, cos, sin).transpose(0, 1).reshape(L, D)

        q, k = rope(q, p + "attn.q_norm.weight"), rope(k, p + "attn.k_norm.weight")
        bias = None
        if W.has(p + "attn.kv_y.weight"):
            ky, vy = lin(W, p + "attn.kv_y", txt).chunk(2, -1)
            k = torch.cat([k, head_norm(ky, W(p + "attn.k_norm.weight"))])
            v = torch.cat([v, vy])
            bias = torch.cat([torch.zeros(L, device=DEV), txt_bias])
        x = x + m[2] * lin(W, p + "attn.proj", attention(q, k, v, bias))
        h = modulate(rms_norm(x, W(p + "norm2.weight")), m[3], m[4])
        x1, x2 = lin(W, p + "mlp.w12", h).chunk(2, -1)
        return x + m[5] * lin(W, p + "mlp.w3", F.silu(x1) * x2)

    @torch.no_grad()
    def forward(self, latent, sigma, context, weights, path_drop=False, dump=None):
        """latent [64, h, w], context [Lt, 640], weights [Lt] -> x0 [64, h, w]"""
        W = self.W
        _, h, w = latent.shape
        keep = (weights > 0).float()
        txt_bias = torch.log(weights.clamp(min=1e-4)).masked_fill(weights <= 0, float("-inf"))
        refine_keep = weights > 0
        if not refine_keep.any():
            refine_keep[0] = True
        keep_bias = torch.zeros_like(weights).masked_fill(~refine_keep, float("-inf"))

        t = lin(W, "t_embedder.mlp.2", F.silu(lin(W, "t_embedder.mlp.0", timestep_embedding(torch.tensor(sigma * 1000.0, device=DEV)))))
        tc = F.silu(t)
        txt = rms_norm(lin(W, "y_embedder.proj", context), W("y_embedder.norm.weight"))
        txt = self.text_refine(txt, tc, keep_bias, keep)
        pooled = (txt * weights[:, None]).sum(0) / weights.sum().clamp(min=1.0)
        cond = F.silu(t + lin(W, "y_pool_proj", pooled[None]))
        mod = lin(W, "shared_encoder_adaLN.0", cond)
        if dump is not None:
            dump["txt_refined"] = txt
            dump["cond"] = cond

        cos, sin = rope_2d(h, w)
        x = lin(W, "s_embedder.proj", latent.flatten(1).T)
        for i in range(2):
            x = self.block(i, x, txt, cos, sin, mod, txt_bias)
        if dump is not None:
            dump["block1"] = x
        if not path_drop:
            g = x
            for i in range(2, 16):
                g = self.block(i, g, txt, cos, sin, mod, txt_bias)
            x = x + lin(W, "sprint_out_proj", g - x)
        for i in range(16, 18):
            x = self.block(i, x, txt, cos, sin, mod, txt_bias)
        shift, scale = lin(W, "final_layer.adaLN_modulation", cond).chunk(2, -1)
        x = modulate(F.layer_norm(x, (D,), eps=1e-6), shift, scale)
        return lin(W, "final_layer.linear", x).T.reshape(64, h, w)


# ----------------------------------------------------------------------------- VAE decoder

class VAE:
    def __init__(self, W: Weights):
        self.W = W

    def gn(self, x, p, silu):
        x = F.group_norm(x, 32, self.W(p + ".weight"), self.W(p + ".bias"), eps=1e-6)
        return F.silu(x) if silu else x

    def conv(self, x, p, pad=1):
        return F.conv2d(x, self.W(p + ".weight"), self.W(p + ".bias"), padding=pad)

    def res(self, x, p):
        h = self.conv(self.gn(x, p + ".norm1", True), p + ".conv1")
        h = self.conv(self.gn(h, p + ".norm2", True), p + ".conv2")
        if self.W.has(p + ".nin_shortcut.weight"):
            x = self.conv(x, p + ".nin_shortcut", 0)
        return x + h

    def attn(self, x, p):
        _, c, h, w = x.shape
        n = self.gn(x, p + ".norm", False)
        q, k, v = (self.conv(n, p + s, 0).flatten(2).transpose(1, 2) for s in (".q", ".k", ".v"))
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, c, h, w)
        return x + self.conv(a, p + ".proj_out", 0)

    @torch.no_grad()
    def decode(self, latent):
        W = self.W
        z = latent[None] * W("latent_std") + W("latent_mean")
        x = self.conv(z, "decoder.conv_in")
        x = self.res(x, "decoder.mid.block_1")
        x = self.attn(x, "decoder.mid.attn_1")
        x = self.res(x, "decoder.mid.block_2")
        for level in reversed(range(5)):
            p = f"decoder.up.{level}"
            for i in range(3):
                x = self.res(x, f"{p}.block.{i}")
                if W.has(f"{p}.attn.{i}.q.weight"):
                    x = self.attn(x, f"{p}.attn.{i}")
            if W.has(f"{p}.upsample.conv.weight"):
                x = self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"), f"{p}.upsample.conv")
        x = self.conv(self.gn(x, "decoder.norm_out", True), "decoder.conv_out")
        return ((torch.tanh(x[0]) + 1) / 2).clamp(0, 1)  # [3, H, W]


# ----------------------------------------------------------------------------- sampling

def simple_sigmas(steps, shift=3.0):
    t = torch.arange(1, 1001, dtype=torch.float32) / 1000
    table = shift * t / (1 + (shift - 1) * t)
    ss = 1000 / steps
    return [float(table[-(1 + int(x * ss))]) for x in range(steps)] + [0.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--model", default="4step", choices=["4step", "base"])
    ap.add_argument("--precision", default="bf16", choices=["bf16", "int8"], help="model files to read (int8: tools/build_assets.py)")
    ap.add_argument("--prompt", default="newest, masterpiece, 1girl, solo, (fennec ears:1.3), long blonde wavy hair, blue eyes, big fluffy tail, smile, forest, sunlight")
    ap.add_argument("--negative", default="oldest, low quality, lowres, blurry, out of focus, jpeg artifacts, watermark, signature, text, bad anatomy, deformed, extra limbs, missing fingers, cropped")
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=1216)
    ap.add_argument("--steps", type=int, default=None, help="default: 4 (4step) / 30 (base)")
    ap.add_argument("--cfg", type=float, default=None, help="default: 1 (4step) / 4 (base)")
    ap.add_argument("--guidance", default="alternate", choices=["alternate", "cfg", "path_drop"])
    ap.add_argument("--sampler", default=None, choices=["euler", "lcm"], help="default: lcm (4step) / euler (base)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora", action="append", default=[], help="path[:strength], repeatable")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "ref.png"))
    ap.add_argument("--dump", default=None, help="directory for tensors compared by tools/check.html")
    args = ap.parse_args()
    fast = args.model == "4step"
    steps = args.steps or (4 if fast else 30)
    cfg = args.cfg if args.cfg is not None else (1.0 if fast else 4.0)
    sampler = args.sampler or ("lcm" if fast else "euler")
    loras = [(p.rsplit(":", 1)[0], float(p.rsplit(":", 1)[1])) if re.search(r":-?[\d.]+$", p) else (p, 1.0) for p in args.lora]

    files = FILES[args.precision]
    te_path = os.path.join(args.models, files["te"])
    tok = Tokenizer(te_path)
    te = Weights(te_path, loras)
    dit = DiT(Weights(os.path.join(args.models, files[args.model]), loras))
    vae = VAE(Weights(os.path.join(args.models, files["vae"])))

    dump = {}
    def text(prompt):
        ids, weights = tok.encode(prompt)
        return ids, weights, gemma_encode(te, ids), torch.tensor(weights, device=DEV)

    ids, weights, ctx, wts = text(args.prompt)
    guided = cfg != 1
    if guided:
        n_ids, n_weights, n_ctx, n_wts = text(args.negative)
    h, w = args.height // 16, args.width // 16
    sigmas = simple_sigmas(steps)
    noise = torch.randn((1, 64, h, w), generator=torch.Generator().manual_seed(args.seed))[0].to(DEV)
    renoise = torch.Generator().manual_seed(args.seed)  # the samplers' own generator (app/samplers.js)
    x = noise * sigmas[0]
    for i in range(steps):
        s, s1 = sigmas[i], sigmas[i + 1]
        x0 = dit.forward(x, s, ctx, wts, dump=dump if i == 0 else None)
        if i == 0:
            dump["x0_cond0"] = x0
        if guided:
            path_drop = args.guidance == "path_drop" or (args.guidance == "alternate" and i % 2 == 1)
            x0u = dit.forward(x, s, n_ctx, n_wts, path_drop=path_drop)
            if i == 0:
                dump["x0_uncond0"] = x0u if not path_drop else dit.forward(x, s, n_ctx, n_wts)
                dump["x0_pathdrop0"] = x0u if path_drop else dit.forward(x, s, n_ctx, n_wts, path_drop=True)
            x0 = x0u + cfg * (x0 - x0u)
        if sampler == "lcm":
            x = x0 if s1 == 0 else (1 - s1) * x0 + s1 * torch.randn((1, 64, h, w), generator=renoise)[0].to(DEV)
        else:
            x = x0 if s1 == 0 else (s1 / s) * x + (1 - s1 / s) * x0
        print(f"step {i + 1}/{steps}")
    image = vae.decode(x)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    from PIL import Image
    Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(args.out)
    print("wrote", args.out)

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        tensors = {"te_hidden": ctx, "noise": noise, "latent_final": x, "image": image, **dump}
        if guided:
            tensors["te_hidden_neg"] = n_ctx
        for name, t in tensors.items():
            t.detach().float().cpu().contiguous().numpy().tofile(os.path.join(args.dump, name + ".bin"))
        meta = {
            "model": args.model, "precision": args.precision, "prompt": args.prompt, "negative": args.negative if guided else "", "ids": ids, "weights": weights,
            "neg_ids": n_ids if guided else None, "neg_weights": n_weights if guided else None,
            "width": args.width, "height": args.height, "steps": steps, "cfg": cfg, "guidance": args.guidance, "sampler": sampler, "seed": args.seed,
            "loras": [{"path": os.path.relpath(p, ROOT).replace(os.sep, "/"), "strength": s} for p, s in loras],
        }
        json.dump(meta, open(os.path.join(args.dump, "index.json"), "w"), indent=1)
        print("dumped", sorted(tensors))


if __name__ == "__main__":
    main()
