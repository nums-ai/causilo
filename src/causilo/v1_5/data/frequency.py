"""Preprocessing helpers for integer row multiplicities."""

import math

import numpy as np


def validate_frequencies(sample_count, rows):
    """Return owned integer counts; reject importance weights and unsafe totals."""
    weights = np.asarray(sample_count)
    if weights.shape != (rows,) or weights.dtype.kind not in "iuf":
        raise ValueError("sample_count must contain one integer repetition count per training row")
    if not np.isfinite(weights).all() or (weights < 0).any() or (weights != np.floor(weights)).any():
        raise ValueError("sample_count must contain finite nonnegative integers")
    # Float64 preprocessing must be able to represent the total exactly. Check
    # before casting so unsigned integers and large floats cannot wrap around.
    if not 0 < sum(map(int, weights)) <= 2**53:
        raise ValueError("Total sample_count must be between 1 and 2**53")
    return weights.astype(np.int64, copy=True)


def _weighted_moments(column, frequency, correction):
    total = int(frequency.sum())
    center = np.average(column, weights=frequency) if total else 0.0
    variance = np.dot(frequency, (column - center) ** 2) / (total - correction) if total > correction else 0.0
    return center, max(math.sqrt(variance), 1e-6)


def weighted_outlier_bounds(values, weights):
    """Compute both observed-only bounds without expanding repeated observations."""
    lower, upper = np.empty(values.shape[1]), np.empty(values.shape[1])
    correction = int(weights.sum() > 1)
    for index, column in enumerate(values.T):
        observed = (~np.isnan(column)) & (weights > 0)
        column, frequency = column[observed], weights[observed]
        mean, preliminary = _weighted_moments(column, frequency, correction)
        central = (column >= mean - 4 * preliminary) & (column <= mean + 4 * preliminary)
        center, spread = _weighted_moments(column[central], frequency[central], correction)
        lower[index], upper[index] = center - 4 * spread, center + 4 * spread
    return lower, upper
