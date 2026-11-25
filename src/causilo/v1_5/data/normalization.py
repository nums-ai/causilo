# Copyright 2026 Nums AI Inc. Licensed under Apache-2.0. See LICENSE.

"""Standardize numeric features from context statistics."""

import numpy as np


def _fit_standardization(table, sample_count):
    # Overflow can occur while summing otherwise finite observations. Recompute
    # only columns whose fitted moments exceeded the working dtype.
    offsets = []
    with np.errstate(over="ignore", invalid="ignore"):
        center, scale, centered = _moments(table, sample_count)
        for column in np.flatnonzero(~np.isfinite(center) | ~np.isfinite(scale)):
            center[column], scale[column], offset = _scaled_moments(table[:, column], sample_count)
            offsets.append((column, offset))
    scaled = _standardize(table, center, scale, offsets, centered)
    return center, scale, tuple(offsets), scaled


def _moments(table, sample_count):
    missing = np.isnan(table)
    observed = np.where(missing, 0, table)
    if sample_count is None:
        counts = (~missing).sum(axis=0)
    else:
        counts = ((~missing) * sample_count[:, None]).sum(axis=0)
        observed = observed * sample_count[:, None]
    center = np.divide(
        observed.sum(axis=0), counts, out=np.zeros(table.shape[1], dtype=table.dtype), where=counts > 0,
    )
    del observed
    # Missing cells contribute neither a squared deviation nor a count.
    scaled = table - center
    squared = np.square(scaled)
    squared[missing] = 0
    if sample_count is not None:
        squared = squared * sample_count[:, None]
    variance = np.divide(
        squared.sum(axis=0), counts, out=np.zeros_like(center), where=counts > 0,
    )
    del squared, missing
    scale = np.sqrt(variance) + 1e-6
    return center, scale, scaled


def _scaled_moments(column, weights):
    """Fit the population moments in power-of-two units when the direct sums overflow."""
    observed = ~np.isnan(column)
    values = column[observed].astype(np.float64, copy=False)
    counts = None if weights is None else weights[observed]
    maximum = np.abs(values).max()
    unit = np.ldexp(1.0, int(np.frexp(maximum)[1]) - 1)
    values = values / unit
    # The theoretical mean and deviation are bounded by the largest absolute observation.
    bound = maximum / unit
    mean = column.dtype.type(np.clip(np.average(values, weights=counts), -bound, bound) * unit)
    deviations = values - float(mean) / unit
    # Preserve a mean between adjacent representable observations near the dtype maximum.
    residual = np.average(deviations, weights=counts)
    variance = np.average((deviations - residual) ** 2, weights=counts)
    scale = column.dtype.type(min(np.sqrt(variance), bound) * unit + 1e-6)
    return mean, scale, residual * unit / float(scale)


def _standardize(table, center, scale, offsets, centered=None):
    with np.errstate(over="ignore"):
        if centered is None:
            centered = table - center
        np.divide(centered, scale, out=centered)
        # Halving both operands and the scale keeps the fitted division finite
        # when the centered difference alone overflows.
        for column, offset in offsets:
            centered[:, column] = (
                (table[:, column] * 0.5 - center[column] * 0.5) / (scale[column] * 0.5) - offset
            )
        return centered


def outlier_bounds(values):
    """Fit four-sigma bounds, recomputing moments after excluding extreme values."""
    correction = int(len(values) > 1)

    def moments(missing, order):
        centered = values.copy(order=order)
        centered[missing] = 0
        counts = len(values) - missing.sum(axis=0)
        mean = centered.sum(axis=0)
        np.divide(mean, np.maximum(counts, 1), out=mean)
        centered -= mean
        centered[missing] = 0
        np.square(centered, out=centered)
        variance = centered.sum(axis=0)
        np.divide(variance, np.maximum(counts - correction, 1), out=variance)
        return mean, np.maximum(np.sqrt(variance), 1e-6)

    missing = np.isnan(values)
    mean, preliminary = moments(missing, "K")
    missing |= values < mean - 4 * preliminary
    missing |= values > mean + 4 * preliminary
    # The outlier-filtered table is reduced in row-major order.
    center, spread = moments(missing, "C")
    return center - 4 * spread, center + 4 * spread
