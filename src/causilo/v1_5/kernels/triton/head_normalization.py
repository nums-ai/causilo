"""Unit-RMS normalization of packed FP16 attention heads."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sum_halves(values, heads: tl.constexpr, width: tl.constexpr):
    halves = tl.permute(tl.reshape(values, (heads, 2, width // 2)), (0, 2, 1))
    first, second = tl.split(halves)
    return first + second


@triton.jit
def _mean_squares(values, HEADS: tl.constexpr, WIDTH: tl.constexpr):
    square = values * values
    if WIDTH == 128:
        # PyTorch's vectorized input stage reduces four adjacent values per lane.
        evens, odds = tl.split(tl.reshape(square, (HEADS, 32, 2, 2)))
        a, c = tl.split(evens)
        b, d = tl.split(odds)
        square = ((a + b) + c) + d
    elif WIDTH == 64:
        square = _sum_halves(square, HEADS, 64)
    if WIDTH >= 32:
        square = _sum_halves(square, HEADS, 32)
    if WIDTH >= 16:
        square = _sum_halves(square, HEADS, 16)
    for width in tl.static_range(3, 0, -1):
        square = _sum_halves(square, HEADS, 1 << width)
    return tl.reshape(square, (HEADS,)) * (1.0 / WIDTH)


@triton.jit(do_not_specialize=["head_count"])
def _normalize(
    PACKED, head_count, WIDTH: tl.constexpr, BLOCK_HEADS: tl.constexpr, EPS: tl.constexpr,
):
    # The flattened head grid can exceed two billion elements.
    heads = tl.program_id(0).to(tl.int64) * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
    columns = tl.arange(0, WIDTH)
    offsets = heads[:, None] * WIDTH + columns[None, :]
    active = heads[:, None] < head_count
    values = tl.load(PACKED + offsets, active, other=0).to(tl.float32)
    squares = _mean_squares(values, BLOCK_HEADS, WIDTH)
    rms = tl.sqrt_rn(tl.maximum(squares, EPS, propagate_nan=tl.PropagateNan.ALL))
    tl.store(PACKED + offsets, tl.div_rn(values, rms[:, None]), active)


@torch.library.custom_op("causilo_v15::normalize_packed_heads_", mutates_args={"packed"})
def normalize_packed_heads_(packed: torch.Tensor) -> None:
    """Normalize a fresh Q or K projection in its own storage."""
    width = packed.shape[-1]
    head_count = packed.numel() // width
    _normalize[(triton.cdiv(head_count, 32),)](
        packed, head_count, width, 32, torch.finfo(torch.float32).eps, enable_fp_fusion=False,
    )


@normalize_packed_heads_.register_fake
def _normalize_inplace_fake(packed):
    return None
