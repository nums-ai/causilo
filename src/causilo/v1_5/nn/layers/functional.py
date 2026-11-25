from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from ...kernels import HAS_TRITON

# Bound unrolled code size and per-program register use. Larger CLS sets keep
# the general attention backend; changing the model's count needs no new kernel.
_MAX_CLS_KERNEL_KEYS = 16


def _can_use_triton_attention(q: Tensor, k: Tensor, v: Tensor) -> bool:
    """Common input requirements for the small-key and small-query kernels."""
    return (
        HAS_TRITON
        and torch.version.hip is None
        and not torch.is_grad_enabled()
        and q.is_cuda
        and q.device == k.device == v.device
        and q.dtype == k.dtype == v.dtype
        and q.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and q.shape[0] > 0
        and q.shape[1] > 0
        and k.shape == v.shape == (q.shape[0], q.shape[1], k.shape[2], q.shape[3])
    )


def _can_use_cls_kernel(
    q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None, query_count: int,
) -> bool:
    """Bound the specialization to unmasked NVIDIA CUDA attention over few keys."""
    return (
        _can_use_triton_attention(q, k, v)
        and attn_mask is None
        and 0 < query_count <= 4096
        and 0 < q.shape[3] <= 128
        and 0 < k.shape[2] <= _MAX_CLS_KERNEL_KEYS
        # Keep the padded key-by-head tile no larger than the original C=4,
        # D=128 specialization; wider heads retain the general backend.
        and (k.shape[2] <= 4 or (k.shape[2] <= 8 and q.shape[3] <= 64) or q.shape[3] <= 32)
    )


def _can_use_cls_readout_kernel(
    q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None, query_count: int,
) -> bool:
    """Guarded small-query readout; unsupported inputs retain SDPA."""
    return (
        _can_use_triton_attention(q, k, v)
        and 0 < query_count <= 16
        and 0 < q.shape[3] <= 32
        and 0 < k.shape[2] <= 1024
        and (
            attn_mask is None
            or (
                attn_mask.device == q.device
                and attn_mask.dtype in (torch.float32, q.dtype)
                and attn_mask.ndim == 4
                and attn_mask.shape[0] in (1, q.shape[0])
                and attn_mask.shape[1] in (1, q.shape[1])
                and attn_mask.shape[2] in (1, q.shape[2])
                and attn_mask.shape[3] == k.shape[2]
            )
        )
    )


def unit_rms(tensor: Tensor) -> Tensor:
    """Normalize fresh Q/K heads in float32, reusing contiguous FP16 projections on CUDA."""
    dtype = tensor.dtype
    if (
        HAS_TRITON
        and tensor.is_cuda
        and torch.version.hip is None
        and not torch.is_grad_enabled()
        and dtype == torch.float16
        and tensor.ndim >= 3
        and tensor.shape[-1] in (8, 16, 32, 64, 128)
        and tensor.numel()
    ):
        packed = tensor.transpose(-3, -2)
        if packed.is_contiguous():
            from causilo.v1_5.kernels.triton.head_normalization import normalize_packed_heads_

            normalize_packed_heads_(packed)
            return tensor
    tensor_float = tensor.float()
    rms = tensor_float.square().mean(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps).sqrt()
    return (tensor_float / rms).to(dtype=dtype)


def scaled_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None = None,
    *,
    full_query_count: int | None = None,
) -> Tensor:
    """Select the same backend when replay only consumes a prefix of the original queries."""
    if q.shape[-2] == 0:
        return q

    query_count = q.shape[-2] if full_query_count is None else full_query_count
    if _can_use_cls_readout_kernel(q, k, v, attn_mask, query_count):
        from causilo.v1_5.kernels.triton.cls_readout import cls_readout

        return cls_readout(q, k, v, attn_mask)

    if _can_use_cls_kernel(q, k, v, attn_mask, query_count):
        # CPU-only installations never import Triton. Keep projections, Q/K
        # normalization, query scaling and output projection in the shared path.
        from causilo.v1_5.kernels.triton.cls_attention import cls_attention

        return cls_attention(q, k, v)

    return F.scaled_dot_product_attention(q, k, v, attn_mask)
