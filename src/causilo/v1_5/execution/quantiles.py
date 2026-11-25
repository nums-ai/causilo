"""Interpolation of sorted 1.5 head quantiles with exponential tails."""

import torch
from torch import Tensor

_TOL = 1e-6


def interpolate_quantiles(quantiles: Tensor, levels: list[float]) -> Tensor:
    """Evaluate one list of probabilities for every row, preserving their order."""
    count = quantiles.shape[-1]
    if count == 1:
        return quantiles.expand(*quantiles.shape[:-1], len(levels))

    # Construct the native grid in the default dtype before converting to the
    # head's dtype, as in the checkpoint's original prediction path.
    grid = torch.linspace(0.0, 1.0, count + 2, device="cpu")[1:-1].to(quantiles.dtype)
    levels_cpu = grid.new_tensor(levels)
    interior = bool(((grid[0] <= levels_cpu) & (levels_cpu <= grid[-1])).all())
    native, requested = grid.to(quantiles.device), levels_cpu.to(quantiles.device)
    lower = (torch.searchsorted(native[:-1], requested, right=True) - 1).clamp(0, count - 2)
    upper = lower + 1
    fraction = (requested - native[lower]) / (native[upper] - native[lower]).clamp_min(_TOL)
    low, high = quantiles[..., lower], quantiles[..., upper]
    middle = low + fraction.clamp(0.0, 1.0) * (high - low)

    left, right = quantiles[..., :1], quantiles[..., -1:]
    middle = torch.where(requested >= native[-1], right, middle)
    if interior:
        return middle
    # Match 1.0's adjacent-point slopes, using the 1.5 head's native grid and dtype.
    log_left = native[:2].clamp_min(_TOL).log()
    log_right = (1 - native[-2:]).clamp_min(_TOL).log()
    left_scale = (quantiles[..., 1:2] - left) / (log_left[1] - log_left[0]).clamp_min(_TOL)
    right_scale = (right - quantiles[..., -2:-1]) / (log_right[0] - log_right[1]).clamp_min(_TOL)
    left_gap = requested.clamp_min(_TOL).log() - log_left[0]
    right_gap = (1 - requested).clamp_min(_TOL).log() - log_right[1]
    left_tail = left + left_scale * left_gap
    right_tail = right - right_scale * right_gap
    return torch.where(
        requested < native[0], left_tail, torch.where(requested > native[-1], right_tail, middle),
    )


def regression_stats(raw_quantiles, output_type, levels):
    """Reduce one batch of raw regression heads in the model's output dtype."""
    quantiles = raw_quantiles.sort(dim=-1).values
    if output_type == "mean":
        return quantiles.mean(dim=-1)
    if output_type == "raw":
        return quantiles
    if output_type == "median":
        return interpolate_quantiles(quantiles, [0.5]).squeeze(-1)
    if output_type == "quantiles":
        return interpolate_quantiles(quantiles, levels)
    raise ValueError(f"Unsupported regression output {output_type!r}")
