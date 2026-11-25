"""Apply the bounded content gain and the two FP16 query-scale products."""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _apply(
    query, length, content, output, elements,
    heads: tl.constexpr, rows, width: tl.constexpr,
    query_batch_stride, query_head_stride, query_row_stride,
    length_batch_stride: tl.constexpr, length_head_stride: tl.constexpr,
    radius: tl.constexpr, block: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    active = index < elements
    channel = index % width
    row = index // width % rows
    head = index // width // rows % heads
    batch = index // width // rows // heads
    position = batch * query_batch_stride + head * query_head_stride + row * query_row_stride + channel
    gain_position = batch * length_batch_stride + head * length_head_stride + channel
    q = tl.load(query + position, active, other=0).to(tl.float32)
    length_gain = tl.load(length + gain_position, active, other=1).to(tl.float32)
    raw = tl.load(content + index, active, other=0).to(tl.float32)
    # Preserve the cast after bounded_multiplier and the rounding of the gain product.
    bounded = libdevice.exp(libdevice.tanh(raw * (1.0 / radius)) * radius).to(tl.float16)
    combined = (length_gain * bounded.to(tl.float32)).to(tl.float16)
    tl.store(output + position, q * combined.to(tl.float32), active)


def apply_query_gain(query: torch.Tensor, length: torch.Tensor, content: torch.Tensor, limit: float) -> torch.Tensor:
    """The query and its output share a dense layout. Content is contiguous in heads-first order."""
    output = torch.empty_like(query)
    _, heads, rows, width = query.shape
    batch_stride = 0 if length.shape[0] == 1 else length.stride(0)
    with torch.cuda.device(query.device):
        _apply[(triton.cdiv(query.numel(), 256),)](
            query, length, content, output, query.numel(), heads, rows, width,
            *query.stride()[:3], batch_stride, length.stride(1), math.log(limit), 256,
            enable_fp_fusion=False,
        )
    return output
