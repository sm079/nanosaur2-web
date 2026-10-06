"""Measures how closely the int8 DiTs (tools/build_assets.py) reproduce the bf16 originals:
image PSNR (dB) vs the bf16 image for a few fixed prompts and seeds, with each model's default
sampling.

  python tools/measure_quality.py [--size 768] [--out out/quality]
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as R  # noqa: E402

PROMPTS = [
    ("newest, masterpiece, 1girl, solo, silver hair, blue eyes, school uniform, cherry blossoms, smile, looking at viewer", 42),
    ("newest, masterpiece, 1boy, knight, full armor, holding sword, castle, sunset, dramatic lighting, wide shot", 7),
    ("newest, masterpiece, no humans, scenery, cozy cafe interior, rain on window, warm lighting, plants, bookshelf", 1234),
]
NEGATIVE = "oldest, low quality, lowres, blurry, out of focus, jpeg artifacts, watermark, signature, text, bad anatomy, deformed, extra limbs, missing fingers, cropped"
SAMPLING = {"4step": {"steps": 4, "cfg": 1.0, "sampler": "lcm"}, "base": {"steps": 30, "cfg": 4.0, "sampler": "euler"}}


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a - b).pow(2).mean().item()
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


@torch.no_grad()
def generate(tok, te, dit, vae, model, prompt, seed, size):
    s = SAMPLING[model]
    ids, weights = tok.encode(prompt)
    ctx, wts = R.gemma_encode(te, ids), torch.tensor(weights, device=R.DEV)
    if s["cfg"] != 1:
        n_ids, n_weights = tok.encode(NEGATIVE)
        n_ctx, n_wts = R.gemma_encode(te, n_ids), torch.tensor(n_weights, device=R.DEV)
    h = w = size // 16
    sigmas = R.simple_sigmas(s["steps"])
    x = torch.randn((1, 64, h, w), generator=torch.Generator().manual_seed(seed))[0].to(R.DEV) * sigmas[0]
    renoise = torch.Generator().manual_seed(seed)
    for i in range(s["steps"]):
        sg, s1 = sigmas[i], sigmas[i + 1]
        x0 = dit.forward(x, sg, ctx, wts)
        if s["cfg"] != 1:
            x0u = dit.forward(x, sg, n_ctx, n_wts, path_drop=i % 2 == 1)
            x0 = x0u + s["cfg"] * (x0 - x0u)
        if s["sampler"] == "lcm":
            x = x0 if s1 == 0 else (1 - s1) * x0 + s1 * torch.randn((1, 64, h, w), generator=renoise)[0].to(R.DEV)
        else:
            x = x0 if s1 == 0 else (s1 / sg) * x + (1 - s1 / sg) * x0
    return vae.decode(x), ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(R.ROOT, "models"))
    ap.add_argument("--model", nargs="+", default=["4step", "base"], choices=["4step", "base"])
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--out", default=None, help="directory for comparison PNGs")
    args = ap.parse_args()
    tok = R.Tokenizer(os.path.join(args.models, R.FILES["bf16"]["te"]))
    weights = lambda prec, part: R.Weights(os.path.join(args.models, R.FILES[prec][part]))  # noqa: E731
    tes = {"bf16": weights("bf16", "te")}
    vae = R.VAE(weights("bf16", "vae"))
    configs = [("int8", "bf16")]
    for model in args.model:
        dits = {p: R.DiT(weights(p, model)) for p in ("bf16", "int8")}
        res = {c: {"psnr": [], "ctx": []} for c in configs}
        for pi, (prompt, seed) in enumerate(PROMPTS):
            ref_img, ref_ctx = generate(tok, tes["bf16"], dits["bf16"], vae, model, prompt, seed, args.size)
            for dit, te in configs:
                img, ctx = generate(tok, tes[te], dits[dit], vae, model, prompt, seed, args.size)
                res[(dit, te)]["psnr"].append(psnr(img, ref_img))
                res[(dit, te)]["ctx"].append(((ctx - ref_ctx).norm() / ref_ctx.norm()).item())
                if args.out:
                    from PIL import Image
                    os.makedirs(args.out, exist_ok=True)
                    for name, im in ((f"{model}_p{pi}_bf16.png", ref_img), (f"{model}_p{pi}_dit-{dit}_te-{te}.png", img)):
                        Image.fromarray((im.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(os.path.join(args.out, name))
                print(f"{model} prompt {pi} dit {dit} te {te}: PSNR {res[(dit, te)]['psnr'][-1]:.2f} dB, text error {res[(dit, te)]['ctx'][-1]:.2e}", flush=True)
        for (dit, te), r in res.items():
            print(f"== {model} dit {dit} te {te}: mean PSNR {sum(r['psnr']) / len(r['psnr']):.2f} dB, text error {sum(r['ctx']) / len(r['ctx']):.2e}")
        del dits


if __name__ == "__main__":
    main()
