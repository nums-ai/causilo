# Copyright 2026 Nums AI Inc. Licensed under Apache-2.0. See LICENSE.

"""Context-fitted numerical views shared by the ensemble."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...v1_0.data.normalization import Normalizer as RankNormalizer
from .frequency import validate_frequencies, weighted_outlier_bounds
from .normalization import _fit_standardization, _standardize, outlier_bounds
from .schema import check_feature_count

_BLOCK_CELLS = 1 << 20


def _distinct_counts(column, weights):
    finite = np.isfinite(column)
    values = column[finite]
    if weights is None or not len(values):
        return np.unique(values, return_counts=True)
    order = np.argsort(values)
    values = values[order]
    start = np.r_[0, np.flatnonzero(values[:-1] != values[1:]) + 1]
    return values[start], np.add.reduceat(weights[finite][order], start)


def _quartiles(values, cumulative):
    ranks = (int(cumulative[-1]) - 1) * np.array([0.25, 0.5, 0.75])
    lower_rank = ranks.astype(np.int64)
    upper_rank = np.minimum(lower_rank + 1, cumulative[-1] - 1)
    lower = values[np.searchsorted(cumulative, lower_rank, side="right")]
    upper = values[np.searchsorted(cumulative, upper_rank, side="right")]
    fraction = ranks - lower_rank
    difference = upper - lower
    return np.where(fraction < 0.5, lower + fraction * difference, upper - (1 - fraction) * difference)


def _fit_robust(table, weights):
    median = np.full(table.shape[1], np.nan)
    scale = np.ones_like(median)
    for index, column in enumerate(table.T):
        values, counts = _distinct_counts(column, weights)
        if len(values):
            cumulative = counts.cumsum()
            lower, median[index], upper = _quartiles(values, cumulative)
            if upper != lower:
                scale[index] = upper - lower
    return median, scale


def _robust(table, median, scale):
    # Median/IQR scaling with soft clipping at ±3 is inspired by the Kumo Tabular
    # preprocessing recipe in NVIDIA Structured Data Models.
    # https://github.com/NVIDIA/structured-data-models/blob/5d663369ff8ead4a78c910dbb8483d558e46627b/sdm/models/kumo/tabular/recipe.py#L19-L24
    with np.errstate(over="ignore", invalid="ignore"):
        values = (table - median) / scale
        result = values / np.hypot(1, values / 3)
    np.copyto(result, 3 * np.sign(values), where=np.isinf(values))
    return result


def _compress(values, lower, upper):
    result = np.clip(values, lower, upper)
    excess = values - result
    outside = excess != 0
    np.arcsinh(excess, out=excess, where=outside)
    np.add(result, excess, out=result, where=outside)
    return result


def _fit_bounds(values, weights, output):
    lower, upper = outlier_bounds(values) if weights is None else weighted_outlier_bounds(values, weights)
    if output is not None:
        output[:] = _compress(values, lower, upper)
    return lower, upper


@dataclass(frozen=True)
class _FittedBlock:
    center: np.ndarray | None
    scale: np.ndarray | None
    extreme_offsets: tuple[tuple[int, float], ...]
    robust_center: np.ndarray | None
    robust_scale: np.ndarray | None
    rank_normalizer: RankNormalizer | None
    bounds: tuple[tuple[np.ndarray, np.ndarray] | None, ...]

    @classmethod
    def fit(cls, table, weights, outputs, selected_view):
        rank_normalizer = None
        if len(outputs) == 3:
            # The validated 1.0 path fits weighted rows in C order.
            rank_input = np.ascontiguousarray(table) if weights is not None else table
            rank_normalizer = RankNormalizer._fit(rank_input, "rank2gaussian", weights, output=outputs[2])
        if selected_view == 2:
            return cls(None, None, (), None, None, rank_normalizer, ())

        center, scale, offsets, scaled = _fit_standardization(table, weights)
        bounds = [_fit_bounds(scaled, weights, outputs[0]) if selected_view in (None, 0) else None]
        robust_center = robust_scale = None
        if len(outputs) > 1:
            robust_center, robust_scale = _fit_robust(scaled, weights)
            bounds.append(_fit_bounds(_robust(scaled, robust_center, robust_scale), weights, outputs[1]))
        return cls(center, scale, offsets, robust_center, robust_scale, rank_normalizer, tuple(bounds))

    @property
    def width(self) -> int:
        return len(self.center) if self.center is not None else len(self.rank_normalizer.fill)

    def transform(self, table, outputs):
        if len(outputs) == 3 and outputs[2] is not None:
            self.rank_normalizer._transform(table, output=outputs[2])
        if self.center is None:
            return

        scaled = _standardize(table, self.center, self.scale, self.extreme_offsets)
        if outputs[0] is not None:
            lower, upper = self.bounds[0]
            identity = _compress(scaled, lower, upper)
            rows, columns = np.nonzero(np.isinf(scaled))
            if len(rows):
                # If the standardized query cannot be represented as float64,
                # compute its logarithmic tail directly from the finite input.
                half_difference = np.abs(table[rows, columns] * 0.5 - self.center[columns] * 0.5)
                tail = np.log(half_difference) - np.log(self.scale[columns]) + np.log(4)
                negative = scaled[rows, columns] < 0
                identity[rows, columns] = np.where(negative, lower[columns] - tail, upper[columns] + tail)
            outputs[0][:] = identity
        if len(outputs) > 1 and outputs[1] is not None:
            outputs[1][:] = _compress(_robust(scaled, self.robust_center, self.robust_scale), *self.bounds[1])


@dataclass(frozen=True)
class FeatureViews:
    """Fit and share identity, robust, and standardized rank-Gaussian views."""

    width: int
    view_count: int
    blocks: tuple[_FittedBlock, ...]
    selected_view: int | None = None

    @classmethod
    def fit_transform(cls, table, sample_count=None, *, view_count=3, retain=True, selected_view=None):
        """Fit context statistics once and optionally retain the requested views."""
        table = np.asarray(table, dtype=np.float64)
        weights = active = None
        if sample_count is not None:
            weights = validate_frequencies(sample_count, len(table))
            if (weights == 0).any():
                active = weights > 0
                weights = weights[active]
            if (weights == 1).all():
                weights = None
        return cls._fit_transform(
            table, weights, view_count=view_count, retain=retain, selected_view=selected_view, active=active,
        )

    @classmethod
    def _fit_transform(cls, table, weights, *, view_count, retain, selected_view, active=None):
        """Fit an encoded float64 context with already validated positive counts."""
        rows, width = table.shape
        step = max(1, _BLOCK_CELLS // rows)
        views = _allocate_views(table, view_count) if retain else None
        prefix = () if selected_view is None else (None,) * selected_view
        blocks = []
        for start in range(0, width, step):
            columns = slice(start, start + step)
            part = table[:, columns]
            fitted = part if active is None else part[active]
            outputs = None if views is None else prefix + tuple(view[:, columns] for view in views)
            fit_outputs = outputs if active is None and outputs is not None else prefix + (None,) * view_count
            block = _FittedBlock.fit(fitted, weights, fit_outputs, selected_view)
            if active is not None and outputs is not None:
                block.transform(part, outputs)
            blocks.append(block)
        return cls(width, view_count, tuple(blocks), selected_view), views

    def transform(self, table):
        """Transform a query chunk using the unchanged context statistics."""
        table = np.asarray(table, dtype=np.float64)
        check_feature_count(table.shape[1], self.width)
        views = _allocate_views(table, self.view_count)
        prefix = () if self.selected_view is None else (None,) * self.selected_view
        start = 0
        for block in self.blocks:
            stop = start + block.width
            columns = slice(start, stop)
            block.transform(table[:, columns], prefix + tuple(view[:, columns] for view in views))
            start = stop
        return views


def _allocate_views(table, count):
    order = "F" if table.flags.f_contiguous else "C"
    return tuple(np.empty(table.shape, dtype=np.float32, order=order) for _ in range(count))
