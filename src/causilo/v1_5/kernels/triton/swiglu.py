"""Compute the SwiGLU activation in FP32 and store it for a BF16 projection."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _gated_product(value, gate, output, elements, block: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    active = index < elements
    v = tl.load(value + index, active, other=0).to(tl.float32)
    g = tl.load(gate + index, active, other=0).to(tl.float32)
    silu = tl.div_rn(g, 1.0 + libdevice.exp(-g))
    tl.store(output + index, (v * silu).to(tl.bfloat16), active)


def gated_product(value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Evaluate contiguous FP16 projections with a single BF16 output write."""
    output = torch.empty_like(value, dtype=torch.bfloat16)
    with torch.cuda.device(value.device):
        _gated_product[(triton.cdiv(value.numel(), 256),)](value, gate, output, value.numel(), 256)
    return output
