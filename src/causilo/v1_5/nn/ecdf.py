"""Context-fitted empirical ranks for feature values, independent of model state."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _validate_features(features: Tensor) -> None:
    if features.ndim < 2:
        raise ValueError("ECDF features must have shape (..., rows, features).")


def _validate_context(features: Tensor, context: Tensor) -> None:
    _validate_features(features)
    if context.ndim != features.ndim or context.shape[:-2] != features.shape[:-2]:
        raise ValueError("ECDF context and features must have matching batch dimensions.")
    if context.shape[-1] != features.shape[-1]:
        raise ValueError("ECDF context and features must have matching feature widths.")
    if context.device != features.device:
        raise ValueError("ECDF context and features must be on the same device.")


@torch.no_grad()
def fit_ecdf_context(features: Tensor, context_rows: int) -> Tensor:
    """Sort finite context values per feature, padding non-observations with infinity.

    Inputs and the returned detached state have shape ``(..., rows, features)``;
    only the first ``context_rows`` rows enter the state. Sorting uses float32,
    preserving float64 inputs when supplied.
    """
    _validate_features(features)
    if not 0 <= context_rows <= features.shape[-2]:
        raise ValueError("ECDF context_rows must lie within the input row count.")
    dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
    context = features[..., :context_rows, :].detach().to(dtype=dtype)
    context = context.masked_fill(~torch.isfinite(context), torch.inf)
    context = context.transpose(-2, -1).contiguous()
    return context.sort(dim=-1).values.transpose(-2, -1).contiguous()


@dataclass(frozen=True)
class WeightedECDF:
    """Sorted observed values and their cumulative repetition counts."""

    values: Tensor
    cumulative: Tensor


def fit_weighted_ecdf(features: Tensor, counts: Tensor) -> WeightedECDF:
    values = features[..., :len(counts), :].transpose(-2, -1).float()
    finite = torch.isfinite(values)
    ordered, indices = values.masked_fill(~finite, torch.inf).sort(dim=-1)
    frequencies = counts.to(device=values.device)[indices].masked_fill(~torch.isfinite(ordered), 0)
    cumulative = torch.nn.functional.pad(frequencies.cumsum(dim=-1), (1, 0))
    return WeightedECDF(ordered.transpose(-2, -1).contiguous(), cumulative)


def ecdf_ranks(features: Tensor, context: Tensor | WeightedECDF) -> Tensor:
    """Return context midranks ``(# less + 0.5 * # equal) / # finite``.

    ``context`` is the sorted state from :func:`fit_ecdf_context`. Missing or
    non-finite queries and features without observed context receive NaN.
    Finite queries outside the observed range saturate at zero or one.
    """
    values = context.values if isinstance(context, WeightedECDF) else context
    _validate_context(features, values)
    dtype = torch.float64 if torch.float64 in (features.dtype, values.dtype) else torch.float32
    boundaries = values.detach().to(dtype=dtype).transpose(-2, -1).contiguous()
    queries = features.detach().to(dtype=dtype).transpose(-2, -1).contiguous()
    finite = torch.isfinite(queries)
    queries = queries.masked_fill(~finite, 0.0)
    left = torch.searchsorted(boundaries, queries, right=False)
    right = torch.searchsorted(boundaries, queries, right=True)
    if isinstance(context, WeightedECDF):
        counts = context.cumulative[..., -1:]
        left = context.cumulative.gather(-1, left)
        right = context.cumulative.gather(-1, right)
        # Finite values precede the infinity padding in a sorted boundary.
        observed = torch.isfinite(boundaries[..., :1]).any(dim=-1, keepdim=True)
    else:
        # Missing context entries sort to +infinity after all finite observations.
        padding = boundaries.new_full((*boundaries.shape[:-1], 1), torch.inf)
        counts = torch.searchsorted(boundaries, padding)
        observed = counts != 0
    ranks = (left.to(dtype=dtype) + right.to(dtype=dtype)) * 0.5 / counts.clamp_min(1)
    ranks = ranks.masked_fill(~finite | ~observed, float("nan"))
    return ranks.transpose(-2, -1).contiguous()
