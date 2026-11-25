"""Prepare a residual and its FP16 feedforward input in one pass."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _inverse_rms(values, rows: tl.constexpr, width: tl.constexpr, epsilon: tl.constexpr):
    tl.static_assert(width == 256)
    # Match the FP32 sum-of-squares order of PyTorch's vectorized CUDA RMSNorm.
    even, odd = tl.split(tl.reshape(values, (rows, width // 2, 2)))
    first, third = tl.split(tl.reshape(even, (rows, width // 4, 2)))
    second, fourth = tl.split(tl.reshape(odd, (rows, width // 4, 2)))
    sums = tl.fma(fourth, fourth, tl.fma(third, third, tl.fma(second, second, first * first)))
    lanes = tl.arange(0, width // 4)[None, :] + tl.zeros((rows, 1), tl.int32)
    for step in tl.static_range(5):
        sums += tl.gather(sums, lanes ^ (16 >> step), 1)
    sums += tl.gather(sums, lanes ^ 32, 1)
    columns = tl.arange(0, width)[None, :] + tl.zeros((rows, 1), tl.int32)
    total = tl.gather(sums, columns // 4, 1)
    return libdevice.rsqrt(total * (1.0 / width) + epsilon)


@triton.jit(do_not_specialize=["period", "tokens", "normalized_rows", "suffix_elements"])
def _prepare(
    rows, update, weight, residual, normalized, period, tokens, normalized_rows, suffix_elements,
    repeat_rows: tl.constexpr, prefix: tl.constexpr, width: tl.constexpr, epsilon: tl.constexpr,
    block_rows: tl.constexpr, block_elements: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    normalized_blocks = tl.cdiv(normalized_rows.to(tl.int64), block_rows)
    if not prefix or program < normalized_blocks:
        index = program * block_rows + tl.arange(0, block_rows)
        row = index // prefix * tokens + index % prefix if prefix else index
        source = row % period if repeat_rows else row
        col = tl.arange(0, width)
        pos = row[:, None] * width + col[None, :]
        valid = index[:, None] < normalized_rows
        a = tl.load(rows + source[:, None] * width + col[None, :], valid, other=0.).to(tl.float32)
        b = tl.load(update + pos, valid, other=0.).to(tl.float32)
        combined = (a + b).to(residual.dtype.element_ty)
        tl.store(residual + pos, combined, valid)
        values = combined.to(tl.float32)
        scale = _inverse_rms(values, block_rows, width, epsilon)
        hidden = (scale * values) * tl.load(weight + col)[None, :]
        tl.store(normalized + index[:, None] * width + col[None, :], hidden, valid)
    else:
        # The feature suffix is saved, but only the latent prefix enters this FFN.
        offset = (program - normalized_blocks) * block_elements + tl.arange(0, block_elements)
        suffix_row = offset // width
        full_row = suffix_row // (tokens - prefix) * tokens + prefix + suffix_row % (tokens - prefix)
        suffix_pos = full_row * width + offset % width
        suffix_src = (full_row % period) * width + offset % width if repeat_rows else suffix_pos
        base_value = tl.load(rows + suffix_src, offset < suffix_elements, other=0.).to(tl.float32)
        delta = tl.load(update + suffix_pos, offset < suffix_elements, other=0.).to(tl.float32)
        tl.store(residual + suffix_pos, base_value + delta, offset < suffix_elements)


def prepare_feedforward(
    rows: torch.Tensor, update: torch.Tensor, weight: torch.Tensor, epsilon: float, period: int, prefix: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = rows.shape[-1]
    residual = torch.empty_like(update, dtype=rows.dtype)
    hidden = update.new_empty((*update.shape[:-2], prefix, width)) if prefix else torch.empty_like(update)
    normalized_rows = hidden.numel() // width
    suffix_elements = residual.numel() - hidden.numel()
    block_rows = 4
    block_elements = 1024
    grid = triton.cdiv(normalized_rows, block_rows) + triton.cdiv(suffix_elements, block_elements)
    with torch.cuda.device(rows.device):
        _prepare[(grid,)](
            rows, update, weight, residual, hidden, period, update.shape[-2], normalized_rows, suffix_elements,
            bool(period), prefix, width, epsilon, block_rows, block_elements, num_warps=4, enable_fp_fusion=False,
        )
    return residual, hidden
