"""Fuse target alignment and the trimmed E8 mean for resident CUDA outputs."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["member_stride"], do_not_specialize_on_alignment=["member_stride"])
def trimmed_mean(
    quantiles, forward, reverse, signs, weights, parameters, result,
    member_stride, STRIDE_ROW: tl.constexpr, STRIDE_LEVEL: tl.constexpr,
    LEVELS: tl.constexpr, BLOCK_LEVELS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    level = tl.arange(0, BLOCK_LEVELS)
    member = tl.arange(0, 8)
    active = level < LEVELS
    forward_level = tl.load(forward + level, active, other=0)
    reverse_level = tl.load(reverse + level, active, other=0)
    negative = tl.load(signs + member) < 0
    index = tl.where(negative[None, :], reverse_level[:, None], forward_level[:, None])
    # Member stride follows the query chunk length. Keep it out of the compilation key.
    values = tl.load(
        quantiles + member_stride * member[None, :].to(tl.int64) + STRIDE_ROW * row + STRIDE_LEVEL * index,
        active[:, None],
        other=0,
    )
    values = tl.where(negative[None, :], -values, values)
    missing = values != values
    missing_count = tl.sum(missing.to(tl.int32), axis=1)
    values = tl.sort(tl.where(missing, float("inf"), values), descending=False)
    kept = (member >= 2) & (member < 6)
    combined = tl.sum(tl.where(kept[None, :], values.to(tl.float64), 0.0), axis=1) * 0.25
    combined = tl.where(missing_count > 2, float("nan"), combined)

    scale, offset = tl.load(parameters), tl.load(parameters + 1)
    restored = combined * scale + offset
    weight = tl.load(weights + level, active, other=0.0)
    mean = tl.sum(tl.where(active, restored * weight, 0.0), axis=0)
    lower = tl.sum(tl.where(level == 0, restored, 0.0), axis=0)
    upper = tl.sum(tl.where(level == LEVELS - 1, restored, 0.0), axis=0)
    lower = tl.minimum(lower, tl.load(parameters + 2))
    upper = tl.maximum(upper, tl.load(parameters + 3))
    result_mean = mean + lower * tl.load(parameters + 4) + upper * tl.load(parameters + 5)
    tl.store(result + row, result_mean)
