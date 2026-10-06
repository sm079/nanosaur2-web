"""Builds the hosted files for the compressed download.

The app loads the original files straight from the model repo; this writes the alternatives next
to them in models/ (which must hold the originals, see README):

  nanosaur2_dmad_4step_diffusion_model.int8.safetensors  DiT, block linears int8 + ConvRot
  nanosaur2_diffusion_model.int8.safetensors
  nanosaur2_vae_decoder.safetensors                       the VAE's decoder only (bf16, lossless)

Quantized: the DiT's attention and MLP linears in every block and text refine block (ComfyUI
int8_tensorwise + ConvRot, per-output-channel scales). Everything else stays bf16: adaLN,
embedders, final layer, norms (small and the most precision-sensitive parts). The text encoder is
used as released.

  python tools/build_assets.py [--force]
  python tools/measure_quality.py      # fidelity vs the bf16 files
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DITS = ["nanosaur2_dmad_4step_diffusion_model", "nanosaur2_diffusion_model"]
DIT_QUANT_RE = re.compile(r"^(blocks|text_refine_blocks)\.\d+\.(attn\.(qkv_x|kv_y|qkv|proj)|mlp\.w(12|3))\.weight$")


def fmt_size(n: int) -> str:
    return f"{n / 2**30:.2f} GiB" if n > 2**30 else f"{n / 2**20:.1f} MiB"


def build_dit(src: str, out: str, device: str) -> None:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(src, "pt") as f:
        meta = f.metadata() or {}
        keys = list(f.keys())
        for i, k in enumerate(keys):
            t = f.get_tensor(k)
            if DIT_QUANT_RE.search(k) and quant.can_convrot(t):
                for suffix, v in quant.quantize_int8_convrot(t.to(device)).items():
                    tensors[k[: -len("weight")] + suffix] = v.cpu()
            else:
                tensors[k] = t
            if i % 50 == 0:
                print(f"  {i}/{len(keys)}", flush=True)
    save_file(tensors, out, metadata={**meta, "quant": "int8"})


def build_vae_decoder(src: str, out: str) -> None:
    with safe_open(src, "pt") as f:
        meta = f.metadata() or {}
        tensors = {k: f.get_tensor(k) for k in f.keys() if k.startswith("decoder.") or k in ("latent_mean", "latent_std")}
    save_file(tensors, out, metadata=meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    d = args.models

    def build(name: str, fn) -> None:
        path = os.path.join(d, name)
        if args.force or not os.path.exists(path):
            print(f"{name} ...", flush=True)
            fn(path)
        print(f"  {name}: {os.path.getsize(path)} bytes ({fmt_size(os.path.getsize(path))})")

    for dit in DITS:
        build(f"{dit}.int8.safetensors", lambda out, dit=dit: build_dit(os.path.join(d, dit + ".safetensors"), out, args.device))
    build("nanosaur2_vae_decoder.safetensors", lambda out: build_vae_decoder(os.path.join(d, "nanosaur2_vae.safetensors"), out))


if __name__ == "__main__":
    main()
