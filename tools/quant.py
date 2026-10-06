"""Offline int8 weight quantizers producing ComfyUI / comfy_kitchen compatible tensors.

  * int8_tensorwise + convrot: per-output-channel int8 of the Hadamard-rotated weight, for
    linears whose input size is a multiple of 256. At runtime the input gets the same rotation,
    so x @ W^T = (x @ H) @ W_rot^T.

The math follows comfy_kitchen/backends/eager/quantization.py; the tensors use ComfyUI's
comfy_quant layout.
"""

from __future__ import annotations

import json
import math

import torch

CONVROT_GROUP = 256


def hadamard(size: int, device, dtype=torch.float32) -> torch.Tensor:
    """Normalized regular (symmetric) Hadamard matrix, size must be a power of 4."""
    if size < 4 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"size must be a power of 4, got {size}")
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype, device=device)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return h / math.sqrt(size)


def rotate(weight: torch.Tensor, group: int = CONVROT_GROUP) -> torch.Tensor:
    """W_rot = W @ H^T per contiguous group of `group` input features (H is symmetric)."""
    n, k = weight.shape
    h = hadamard(group, weight.device, torch.float32)
    return (weight.float().reshape(n, k // group, group) @ h.T).reshape(n, k)


def can_convrot(weight: torch.Tensor) -> bool:
    return weight.dim() == 2 and weight.shape[1] % CONVROT_GROUP == 0


def quant_meta(fmt: str, **extra) -> torch.Tensor:
    meta = {"format": fmt, **extra}
    return torch.tensor(list(json.dumps(meta).encode()), dtype=torch.uint8)


def quantize_int8_convrot(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    rot = rotate(weight)
    scale = (rot.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
    q = (rot / scale).round().clamp(-128, 127).to(torch.int8)
    return {
        "weight": q.contiguous(),
        "weight_scale": scale.float().contiguous(),
        "comfy_quant": quant_meta("int8_tensorwise", convrot=True, convrot_groupsize=CONVROT_GROUP),
    }


def dequantize(t: dict[str, torch.Tensor]) -> torch.Tensor:
    meta = json.loads(bytes(t["comfy_quant"].tolist()).decode())
    if meta["format"] != "int8_tensorwise":
        raise ValueError(f"unsupported format {meta['format']}")
    w = t["weight"].float() * t["weight_scale"].float()
    return rotate(w) if meta.get("convrot") else w  # H is symmetric and orthogonal: rotating twice is the identity
