"""Inference attention for a small set of CLS queries.

Each program owns a row/head and reuses its K/V tile across the CLS queries.
Floating-point rounding can differ from SDPA.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(
    Q,
    K,
    V,
    MASK,
    OUT,
    H: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    qb: tl.constexpr,
    qh: tl.constexpr,
    qc: tl.constexpr,
    qd: tl.constexpr,
    kb: tl.constexpr,
    kh: tl.constexpr,
    kn: tl.constexpr,
    kd: tl.constexpr,
    vb: tl.constexpr,
    vh: tl.constexpr,
    vn: tl.constexpr,
    vd: tl.constexpr,
    mb: tl.constexpr,
    mh: tl.constexpr,
    mc: tl.constexpr,
    mn: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    # Strided K/V views can span more than 2**31 elements on long tables.
    bh = tl.program_id(0).to(tl.int64)
    b, h = bh // H, bh % H
    n, d = tl.arange(0, BN), tl.arange(0, BD)
    valid = (n[:, None] < N) & (d[None, :] < D)
    k = tl.load(K + b * kb + h * kh + n[:, None] * kn + d[None, :] * kd, valid, other=0).to(tl.float32)
    v = tl.load(V + b * vb + h * vh + n[:, None] * vn + d[None, :] * vd, valid, other=0).to(tl.float32)
    for c in tl.static_range(C):
        q = tl.load(Q + b * qb + h * qh + c * qc + d * qd, d < D, other=0).to(tl.float32)
        scores = tl.sum(k * q[None, :], 1) * (D**-0.5)
        if HAS_MASK:
            mask = tl.load(MASK + b * mb + h * mh + c * mc + n * mn, n < N, other=0)
            scores += mask.to(tl.float32)
        scores = tl.where(n < N, scores, -float("inf"))
        maximum = tl.max(scores, 0)
        maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
        numerator = tl.exp(scores - maximum)
        denominator = tl.sum(numerator, 0)
        denominator = tl.where(denominator == 0.0, 1.0, denominator)
        # Match the low-precision fused-attention PV convention.
        p_dot = numerator.to(V.dtype.element_ty).to(tl.float32)
        out = tl.sum(p_dot[:, None] * v, 0) / denominator
        tl.store(OUT + ((b * C + c) * H + h) * D + d, out, d < D)


# Kernel launch constants specialize only these opaque operations. They must not
# specialize the entire model graph to each feature width or tensor stride.
@torch.library.custom_op("causilo_v15::cls_readout_forward", mutates_args=())
def cls_readout(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None,
) -> torch.Tensor:
    """Read feature values into each CLS query."""
    b, h, c, d = q.shape
    n = k.shape[-2]
    out = torch.empty((b, c, h, d), dtype=q.dtype, device=q.device).transpose(1, 2)
    strides = (
        tuple(0 if size == 1 else stride for size, stride in zip(mask.shape, mask.stride(), strict=True))
        if mask is not None
        else (0, 0, 0, 0)
    )
    _forward[(b * h,)](
        q,
        k,
        v,
        mask if mask is not None else q,
        out,
        h,
        c,
        n,
        d,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *strides,
        mask is not None,
        triton.next_power_of_2(n),
        triton.next_power_of_2(d),
        num_warps=4,
    )
    return out


@cls_readout.register_fake
def _forward_fake(q, k, v, mask):
    b, h, c, d = q.shape
    return torch.empty((b, c, h, d), device=q.device, dtype=q.dtype).transpose(1, 2)
