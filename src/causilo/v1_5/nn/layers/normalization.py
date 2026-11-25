"""RMS normalization with a fixed epsilon across inference dtypes."""

import torch
from torch import Tensor, nn

from ...kernels import HAS_TRITON

# Smaller residuals finish in PyTorch before a specialized CUDA launch saves wall time.
_MIN_FUSED_ELEMENTS = 8 * 1024 * 1024


def create_norm(width: int) -> nn.RMSNorm:
    return nn.RMSNorm(width, eps=1e-5)


def large_residual(rows: Tensor, *, extra_elements: int = 0) -> bool:
    return rows.numel() + extra_elements >= _MIN_FUSED_ELEMENTS and rows.shape[-1] == 256


def _row_period(rows: Tensor) -> int | None:
    if rows.is_contiguous():
        return 0
    if rows.stride()[-2:] == (rows.shape[-1], 1) and all(
        size == 1 or stride == 0 for size, stride in zip(rows.shape[:-2], rows.stride()[:-2])
    ):
        return rows.shape[-2]
    return None


def feedforward_residual(
    rows: Tensor, update: Tensor, norm: nn.RMSNorm, *, prefix: int = 0,
) -> tuple[Tensor, Tensor]:
    """Return the full residual and the normalized rows used by the feedforward projections."""
    if (
        HAS_TRITON
        and rows.is_cuda
        and torch.version.hip is None
        and not torch.is_grad_enabled()
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.float16
        and update.dtype == torch.float16
        and rows.dtype in (torch.float16, torch.float32)
        and rows.shape == update.shape
        and rows.device == update.device
        and update.is_contiguous()
        and rows.shape[-1] == 256
        and norm.eps is not None
        and norm.normalized_shape == (rows.shape[-1],)
        and norm.weight is not None
        and norm.weight.dtype == torch.float32
        and norm.weight.is_contiguous()
        and norm.weight.device == rows.device
        and not torch.compiler.is_compiling()
    ):
        # Column latents can share one row template across all batch dimensions.
        period = _row_period(rows)
        if period is not None:
            from ...kernels.triton.residual_normalization import prepare_feedforward

            return prepare_feedforward(rows, update, norm.weight, norm.eps, period, prefix)
    residual = rows + update
    return residual, norm(residual[..., :prefix, :] if prefix else residual)
