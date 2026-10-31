"""Training-fitted transforms with explicit missing masks and compressed outliers."""

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

from .frequency import (
    frequency_quantile,
    validate_frequencies,
    weighted_mean_var,
    weighted_outlier_bounds,
    weighted_yeojohnson_lambda,
)


@dataclass
class Rank2Gaussian:
    """Interpolate training mid-ranks and map them through the Gaussian quantile function."""

    knots: tuple[tuple[np.ndarray, np.ndarray], ...]

    @classmethod
    def fit(cls, table, sample_weight=None):
        """Store unique values and tie-aware probabilities per feature column."""
        knots = []
        for column in table.T:
            if sample_weight is None:
                values, counts = np.unique(column, return_counts=True)
                total = len(column)
            else:
                values, inverse = np.unique(column, return_inverse=True)
                counts = np.bincount(inverse, weights=sample_weight)
                total = sample_weight.sum()
            probabilities = (np.cumsum(counts) - counts / 2) / total
            if sample_weight is not None:
                # Very large totals can round the highest mid-rank to 1.
                # Keep Gaussian quantiles finite at the supported count limit.
                probabilities = np.minimum(probabilities, np.nextafter(1.0, 0.0))
            knots.append((values, probabilities))
        return cls(tuple(knots))

    def transform(self, table):
        """Reuse fitted ranks; interpolation clamps unseen extremes to endpoint ranks."""
        return np.column_stack([
            ndtri(np.interp(column, values, probabilities))
            for column, (values, probabilities) in zip(table.T, self.knots)
        ])


@dataclass
class PowerNormalizer:
    """Keep undefined power outputs missing before and after standard scaling."""

    power: PowerTransformer
    scale: StandardScaler

    @staticmethod
    def _power_values(power, table):
        with np.errstate(over="ignore", invalid="ignore"):
            values = power.transform(table)
        values[~np.isfinite(values)] = np.nan
        return values

    @classmethod
    def fit(cls, table, sample_weight=None):
        power = PowerTransformer(standardize=False)
        if sample_weight is None:
            power.fit(table)
        else:
            power.n_features_in_ = table.shape[1]
            power.lambdas_ = np.array([
                weighted_yeojohnson_lambda(column, sample_weight) for column in table.T
            ])
        values = cls._power_values(power, table)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            scale = StandardScaler(copy=False).fit(values, sample_weight=sample_weight)
        return cls(power, scale)

    def transform(self, table):
        values = self._power_values(self.power, table)
        with np.errstate(over="ignore", invalid="ignore"):
            values = self.scale.transform(values)
        values[~np.isfinite(values)] = np.nan
        return values


@dataclass
class Normalizer:
    """Normalize encoded features using statistics fitted only on training rows.

    Missing entries are temporarily mean-filled to fit/apply transforms, then
    restored to NaN for the model's missing-value embedding. Method ``none``
    skips the optional distribution transform, but still standardizes columns
    and compresses outlier tails. Rank-to-Gaussian uses its own final scaling.
    """

    fill: np.ndarray
    center: np.ndarray
    spread: np.ndarray
    normalizer: object | None
    quantile_scale: StandardScaler | None
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def fit(cls, table: np.ndarray, method: str, sample_weight=None) -> "Normalizer":
        """Fit transforms using optional integer row multiplicities.

        Zero-count rows are excluded; unit counts use the original unweighted
        path. All statistics are computed without expanding repeated rows.
        """
        if sample_weight is not None:
            sample_weight = validate_frequencies(sample_weight, len(table))
            present = sample_weight > 0
            table, sample_weight = table[present], sample_weight[present]
            if np.all(sample_weight == 1):
                sample_weight = None
        missing = np.isnan(table)
        observed = ~missing
        values = np.where(missing, 0, table)
        rank_missing = np.flatnonzero(missing.any(axis=0)) if method == "rank2gaussian" else []
        # These columns use the grouped mean below; skip their ordinary sum,
        # which may overflow even when the resulting mean is finite.
        values[:, rank_missing] = 0
        if sample_weight is not None:
            observed = observed * sample_weight[:, None]
            values = values * sample_weight[:, None]
        counts = observed.sum(axis=0)
        fill = np.divide(values.sum(axis=0), counts, out=np.zeros(table.shape[1]), where=counts > 0)
        if method == "rank2gaussian":
            # Mean-filled entries participate in ranks. A one-ULP difference
            # can split a tie, so use the same grouped sum for compact and
            # expanded inputs, independent of row order and missing-row count.
            for index in rank_missing:
                present = ~missing[:, index]
                unique, inverse, frequencies = np.unique(
                    table[present, index], return_inverse=True, return_counts=True
                )
                if sample_weight is not None:
                    frequencies = np.bincount(inverse, weights=sample_weight[present])
                if unique.size:
                    scale = float(np.max(np.abs(unique)))
                    total = int(frequencies.sum())
                    if scale:
                        mean = math.fsum(
                            (float(value) / scale) * (int(count) / total)
                            for value, count in zip(unique, frequencies)
                        )
                        fill[index] = np.clip(mean, -1.0, 1.0) * scale
        filled = np.where(missing, fill, table)
        if method == "rank2gaussian":
            center = np.zeros(filled.shape[1], dtype=filled.dtype)
            spread = np.ones_like(center)
        else:
            if sample_weight is None:
                center, spread = filled.mean(axis=0), filled.std(axis=0)
            else:
                center, variance = weighted_mean_var(filled, sample_weight)
                spread = np.sqrt(variance)
            tolerance = np.finfo(filled.dtype).eps * np.maximum(np.abs(center), 1)
            spread = np.where(spread <= tolerance, 1.0, spread)
        scaled = (filled - center) / spread
        normalizer, quantile_scale = None, None
        if method == "power":
            normalizer = PowerNormalizer.fit(scaled, sample_weight)
        elif method == "robust":
            normalizer = fit_robust(scaled, sample_weight)
        elif method == "rank2gaussian":
            normalizer = Rank2Gaussian.fit(scaled, sample_weight)
        elif method != "none":
            raise ValueError(f"Unknown internal normalization: {method}")
        normalized = scaled if normalizer is None else normalizer.transform(scaled)
        if method == "rank2gaussian":
            quantile_scale = StandardScaler().fit(normalized, sample_weight=sample_weight)
            normalized = quantile_scale.transform(normalized)
        lower, upper = outlier_bounds(normalized, sample_weight)
        return cls(
            fill=fill,
            center=center,
            spread=spread,
            normalizer=normalizer,
            quantile_scale=quantile_scale,
            lower=lower,
            upper=upper,
        )

    def transform(self, table: np.ndarray) -> np.ndarray:
        """Return float32 features with the original NaN mask and unchanged fitted state."""
        missing = np.isnan(table)
        values = (np.where(missing, self.fill, table) - self.center) / self.spread
        if self.normalizer is not None:
            values = self.normalizer.transform(values)
        if self.quantile_scale is not None:
            values = self.quantile_scale.transform(values)
        values = compress_tails(values, self.lower, self.upper)
        return np.where(missing, np.nan, values).astype(np.float32)


def fit_robust(table, sample_weight=None):
    """Fit the same interpolated quartiles as RobustScaler on repeated rows."""
    normalizer = RobustScaler(unit_variance=True)
    if sample_weight is None:
        return normalizer.fit(table)
    normalizer.n_features_in_ = table.shape[1]
    quantiles = np.array([
        frequency_quantile(column, sample_weight, [0.25, 0.5, 0.75]) for column in table.T
    ])
    normalizer.center_ = quantiles[:, 1]
    spread = quantiles[:, 2] - quantiles[:, 0]
    spread[spread < 10 * np.finfo(spread.dtype).eps] = 1.0
    normalizer.scale_ = spread / (ndtri(0.75) - ndtri(0.25))
    return normalizer


def outlier_bounds(values, sample_weight=None):
    """Estimate four-sigma bounds after excluding preliminary extreme values.

    These thresholds are part of the fixed preprocessing policy. A second pass
    limits how much extreme training values can inflate the final bounds.
    """
    if sample_weight is not None:
        return weighted_outlier_bounds(values, sample_weight)
    if np.isnan(values).any():
        lower, upper = np.zeros(values.shape[1]), np.zeros(values.shape[1])
        for index, column in enumerate(values.T):
            observed = column[~np.isnan(column)]
            if observed.size:
                bounds = outlier_bounds(observed[:, None])
                lower[index], upper[index] = bounds[0][0], bounds[1][0]
        return lower, upper
    correction = int(len(values) > 1)
    preliminary = np.maximum(values.std(axis=0, ddof=correction), 1e-6)
    mean = values.mean(axis=0)
    central = np.where(np.abs(values - mean) > 4 * preliminary, np.nan, values)
    center = np.nan_to_num(np.nanmean(central, axis=0), nan=0)
    spread = np.maximum(np.nan_to_num(np.nanstd(central, axis=0, ddof=correction), nan=1), 1e-6)
    return center - 4 * spread, center + 4 * spread


def compress_tails(values, lower, upper):
    """Leave central values unchanged and smoothly compress both tails with arcsinh."""
    values = np.where(values < lower, lower - np.arcsinh(lower - values), values)
    return np.where(values > upper, upper + np.arcsinh(values - upper), values)
