"""Fits the latent -> RGB projection used for step previews (app/preview.js).

Generates a few images with the reference pipeline (4-step model), then solves least squares
from each latent pixel (64 channels + bias) to the mean color of its 16x16 image block, in [-1, 1].

  python tools/fit_preview.py [--size 768]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as R  # noqa: E402

PROMPTS = [
    "newest, masterpiece, 1girl, solo, long blonde wavy hair, blue eyes, smile, forest, sunlight",
    "newest, masterpiece, 1boy, knight, silver armor, castle ruins, sunset, dramatic lighting",
    "newest, masterpiece, no humans, scenery, cozy cafe interior, rain on window, warm lighting, plants",
    "newest, masterpiece, 1girl, witch hat, purple cape, night sky, full moon, city lights, stars",
    "newest, masterpiece, chibi, 2girls, eating ice cream, beach, bright colors, simple background",
    "newest, masterpiece, mountain village in autumn, red maple leaves, stone path, morning mist",
    "newest, masterpiece, 1girl, black dress, red background, monochrome, high contrast",
    "newest, masterpiece, underwater, coral reef, fish, light rays, blue and green",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(R.ROOT, "models"))
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--out", default=os.path.join(R.ROOT, "app", "preview.js"))
    args = ap.parse_args()
    te_path = os.path.join(args.models, R.FILES["bf16"]["te"])
    tok = R.Tokenizer(te_path)
    te = R.Weights(te_path)
    dit = R.DiT(R.Weights(os.path.join(args.models, R.FILES["bf16"]["4step"])))
    vae = R.VAE(R.Weights(os.path.join(args.models, R.FILES["bf16"]["vae"])))
    h = w = args.size // 16
    sigmas = R.simple_sigmas(4)
    X, Y = [], []
    for k, prompt in enumerate(PROMPTS):
        ids, weights = tok.encode(prompt)
        ctx, wts = R.gemma_encode(te, ids), torch.tensor(weights, device=R.DEV)
        gen = torch.Generator().manual_seed(1000 + k)
        x = torch.randn((64, h, w), generator=gen).to(R.DEV)
        for i in range(4):
            x0 = dit.forward(x, sigmas[i], ctx, wts)
            s1 = sigmas[i + 1]
            x = x0 if s1 == 0 else (1 - s1) * x0 + s1 * torch.randn((64, h, w), generator=gen).to(R.DEV)
        rgb = vae.decode(x) * 2 - 1  # [3, H, W] in [-1, 1]
        X.append(x.flatten(1).T)
        Y.append(F.avg_pool2d(rgb[None], 16)[0].flatten(1).T)
        print(f"{k + 1}/{len(PROMPTS)}")
    X = torch.cat(X).double()
    Y = torch.cat(Y).double()
    A = torch.cat([X, torch.ones(len(X), 1, device=X.device, dtype=X.dtype)], 1)
    sol = torch.linalg.lstsq(A.cpu(), Y.cpu()).solution  # [65, 3]
    err = (A.cpu() @ sol - Y.cpu()).pow(2).mean().sqrt().item()
    print(f"rms error {err:.4f} (colors in [-1, 1])")
    factors = ", ".join(f"{v:.5f}" for v in sol[:64].flatten().tolist())
    bias = ", ".join(f"{v:.5f}" for v in sol[64].tolist())
    with open(args.out, "w", newline="\n") as f:
        f.write("// Latent -> RGB projection for step previews: rgb = bias + sum_k latent[k] * factors[k], in\n")
        f.write("// [-1, 1]. Fitted by tools/fit_preview.py (least squares against decoded 16x16 blocks).\n")
        f.write(f"export const PREVIEW_FACTORS = new Float32Array([{factors}]);\n")
        f.write(f"export const PREVIEW_BIAS = [{bias}];\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
