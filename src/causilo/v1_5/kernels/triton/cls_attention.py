"""Inference attention over a small CLS key set.

Independent per-head programs handle batches of queries with FP32 reductions.
Floating-point rounding can differ from SDPA.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd(
    Q,
    K,
    V,
    OUT,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    qb: tl.constexpr,
    qh: tl.constexpr,
    qn: tl.constexpr,
    qd: tl.constexpr,
    kb: tl.constexpr,
    kh: tl.constexpr,
    kn: tl.constexpr,
    kd: tl.constexpr,
    vb: tl.constexpr,
    vh: tl.constexpr,
    vn: tl.constexpr,
    vd: tl.constexpr,
    M: tl.constexpr,
    BD: tl.constexpr,
    BC: tl.constexpr,
):
    # Query and output tiles can span more than 2**31 elements.
    bh = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    b, h = bh // H, bh % H
    n = tile * M + tl.arange(0, M)
    d = tl.arange(0, BD)
    key = tl.arange(0, BC)
    q = tl.load(
        Q + b * qb + h * qh + n[:, None] * qn + d[None, :] * qd, (n[:, None] < N) & (d[None, :] < D), other=0
    ).to(tl.float32)
    key_mask = (key[:, None] < C) & (d[None, :] < D)
    k = tl.load(K + b * kb + h * kh + key[:, None] * kn + d[None, :] * kd, key_mask, other=0).to(tl.float32)
    v = tl.load(V + b * vb + h * vh + key[:, None] * vn + d[None, :] * vd, key_mask, other=0).to(tl.float32)
    scores = tl.sum(q[:, None, :] * k[None, :, :], 2) * (D**-0.5)
    scores = tl.where(key[None, :] < C, scores, -float("inf"))
    numerator = tl.exp(scores - tl.max(scores, 1)[:, None])
    denominator = tl.sum(numerator, 1)
    # Fused attention rounds the unnormalized probabilities to V's dtype
    # before PV, then divides the FP32 accumulator by the softmax denominator.
    p_dot = numerator.to(V.dtype.element_ty).to(tl.float32)
    # Do not express this as a rank-3 reduction. Triton 3.7 rewrites it
    # to a K=4 TF32 dot, below Ampere MMA's K=8 tile, duplicating operands.
    # Explicit terms retain IEEE products for small key counts. C is a compile
    # time constant, so C=4 retains the original four-term specialization.
    out = tl.full((M, BD), 0, tl.float32)
    for index in tl.static_range(C):
        pi = tl.sum(tl.where(key[None, :] == index, p_dot, 0), 1)
        vi = tl.sum(tl.where(key[:, None] == index, v, 0), 0)
        out = out + pi[:, None] * vi[None, :]
    out = out / denominator[:, None]
    # Query-major storage makes the downstream (N,H,D)->(N,H*D) view free.
    tl.store(OUT + ((b * N + n[:, None]) * H + h) * D + d[None, :], out, (n[:, None] < N) & (d[None, :] < D))


# Keep launch specialization outside the encompassing model graph.
@torch.library.custom_op("causilo_v15::cls_attention_forward", mutates_args=())
def cls_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Attend to the CLS keys of each row."""
    b, h, n, d = q.shape
    c = k.shape[-2]
    assert c > 0 and k.shape == v.shape == (b, h, c, d)
    out = torch.empty((b, n, h, d), device=q.device, dtype=q.dtype).transpose(1, 2)
    # Put flattened batch/head on grid.x to support long row tiles.
    _fwd[(b * h, triton.cdiv(n, 32))](
        q,
        k,
        v,
        out,
        n,
        h,
        d,
        c,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        32,
        triton.next_power_of_2(d),
        triton.next_power_of_2(c),
    )
    return out


@cls_attention.register_fake
def _forward_fake(q, k, v):
    b, h, n, d = q.shape
    return torch.empty((b, n, h, d), device=q.device, dtype=q.dtype).transpose(1, 2)
