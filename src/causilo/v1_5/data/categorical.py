"""Encode context categories and add frequency features for high cardinality."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

_COUNT_FEATURE_THRESHOLD = 50


def _category_value(value):
    """Map nulls and ambiguous object comparisons to NaN."""
    if value is None:
        return np.nan
    try:
        return np.nan if value != value else value
    except (TypeError, ValueError):
        return np.nan


_category_values = np.frompyfunc(_category_value, 1, 1)


def _typed_categories(dtype):
    return (
        isinstance(dtype, pd.CategoricalDtype)
        and isinstance(dtype.categories.dtype, np.dtype)
        and dtype.categories.dtype.kind in "biufSU"
    )


def _compact_categories(dtype, rows):
    if not isinstance(dtype, pd.CategoricalDtype) or 2 * len(dtype.categories) > rows:
        return False
    return _typed_categories(dtype) or all(
        type(level) in (str, bytes, np.str_, np.bytes_) for level in dtype.categories
    )


def _normalize_categorical_missing(X):
    """Normalize container-specific categorical nulls without mutating the input."""
    if isinstance(X, pd.DataFrame) and all(
        _typed_categories(dtype) or _compact_categories(dtype, len(X)) for dtype in X.dtypes
    ):
        # Typed dictionaries carry missing codes at any size. Only scan compact object dictionaries.
        return X.to_numpy(dtype=object, na_value=np.nan)
    if hasattr(X, "to_numpy"):
        try:
            X = X.to_numpy(dtype=object, na_value=np.nan)
        except TypeError:
            pass
    return _category_values(np.asarray(X, dtype=object))


def _fit_ordinal(ordinal, table):
    # Each observed category is packed and looked up. Use row values directly
    # when a dictionary would occupy more than half the table height.
    if not (
        isinstance(table, pd.DataFrame) and table.size
        and all(_compact_categories(dtype, len(table)) for dtype in table.dtypes)
    ):
        return ordinal.fit_transform(_normalize_categorical_missing(table))

    # Fit on one real value per observed category and expand by the original row codes.
    values = table.to_numpy(dtype=object, na_value=np.nan)
    indices = [
        np.unique(col.cat.codes.to_numpy(), return_index=True, return_inverse=True)[1:]
        for _, col in table.items()
    ]
    height = max(len(first) for first, _ in indices)
    packed = np.column_stack([
        np.resize(values[first, index], height) for index, (first, _) in enumerate(indices)
    ])
    lookup = ordinal.fit_transform(packed)
    return np.column_stack([lookup[inverse, index] for index, (_, inverse) in enumerate(indices)])


def _finite_category_number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return np.nan
    if isinstance(value, (int, np.integer)) and abs(int(value)) > 2**53:
        return np.nan
    try:
        number = float(value)
    except OverflowError:
        return np.nan
    return number if np.isfinite(number) else np.nan


def _ordered_numeric_levels(categories):
    """Find safely representable numeric levels and their original ordinal codes."""
    levels, codes = [], []
    for code, category in enumerate(categories):
        if isinstance(category, (float, np.floating)) and not np.isfinite(category):
            continue
        level = _finite_category_number(category)
        if np.isnan(level):
            return None
        levels.append(level)
        codes.append(code)
    if len(levels) < 2:
        return None
    order = np.argsort(levels)
    levels = np.asarray(levels)[order]
    if np.any(levels[1:] <= levels[:-1]):
        return None
    return levels, np.asarray(codes, dtype=np.intp)[order]


def _interval_fraction(query, low, high):
    with np.errstate(over="ignore", invalid="ignore"):
        width = high - low
        fraction = (query - low) / width
    wide = np.isinf(width)
    if wide.any():
        start, stop = low[wide] / 2, high[wide] / 2
        fraction[wide] = (query[wide] / 2 - start) / (stop - start)
    return fraction


@dataclass(frozen=True)
class CategoricalEncoder:
    ordinal: OrdinalEncoder
    log_counts: dict[int, np.ndarray]
    numeric_levels: tuple[tuple[int, np.ndarray, np.ndarray], ...] = ()

    @classmethod
    def fit(
        cls, values, sample_count: np.ndarray | None, *, count_columns: tuple[int, ...] | None = None,
    ) -> tuple[CategoricalEncoder, np.ndarray]:
        ordinal = OrdinalEncoder(
            dtype=np.float64,
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
        )
        codes = _fit_ordinal(ordinal, values)
        count_columns = None if count_columns is None else frozenset(count_columns)
        log_counts = {}
        numeric_levels = []
        for index, column in enumerate(codes.T):
            categories = ordinal.categories_[index]
            ordered = _ordered_numeric_levels(categories)
            if ordered is not None:
                numeric_levels.append((index, *ordered))
            n_categories = len(categories)
            if count_columns is None:
                if n_categories <= _COUNT_FEATURE_THRESHOLD:
                    continue
            elif index not in count_columns:
                continue
            observed = ~np.isnan(column)
            categories = column[observed].astype(np.intp)
            if sample_count is None:
                counts = np.bincount(categories)
            else:
                counts = np.zeros(n_categories, dtype=np.int64)
                np.add.at(counts, categories, sample_count[observed])
            if count_columns is not None or np.count_nonzero(counts) > _COUNT_FEATURE_THRESHOLD:
                log_counts[index] = np.log1p(counts)
        encoder = cls(ordinal, log_counts, tuple(numeric_levels))
        return encoder, encoder._append_counts(codes)

    def _append_counts(self, codes: np.ndarray, values: np.ndarray | None = None) -> np.ndarray:
        if not self.log_counts:
            return codes

        width = codes.shape[1]
        result = np.empty((len(codes), width + len(self.log_counts)), dtype=np.float64)
        result[:, :width] = codes
        for output, (index, log_counts) in enumerate(self.log_counts.items(), start=width):
            column = codes[:, index]
            missing = np.isnan(column)
            observed = ~missing
            result[observed, output] = log_counts[column[observed].astype(np.intp)]
            if values is None:
                result[missing, output] = np.nan
            else:
                # Unknown query categories have zero frequency; missing values stay missing.
                unknown = values[missing, index]
                result[missing, output] = np.where(unknown != unknown, np.nan, 0.0)
        return result

    def transform(self, values) -> np.ndarray:
        values = _normalize_categorical_missing(values)
        codes = self.ordinal.transform(values)
        encoded = self._append_counts(codes, values)
        if self.numeric_levels:
            self._interpolate_unknown(values, codes, encoded)
        return encoded

    def _interpolate_unknown(self, values, codes, encoded):
        count_columns = {index: column for column, index in enumerate(self.log_counts, start=codes.shape[1])}
        for index, levels, ordinals in self.numeric_levels:
            rows = np.flatnonzero(np.isnan(codes[:, index]))
            if not len(rows):
                continue
            query = np.fromiter((_finite_category_number(v) for v in values[rows, index]), dtype=np.float64)
            inside = (query > levels[0]) & (query < levels[-1])
            rows, query = rows[inside], query[inside]
            upper = np.searchsorted(levels, query)
            distinct = query < levels[upper]
            rows, query, upper = rows[distinct], query[distinct], upper[distinct]
            if not len(rows):
                continue
            fraction = _interval_fraction(query, levels[upper - 1], levels[upper])
            lower_codes, upper_codes = ordinals[upper - 1], ordinals[upper]
            encoded[rows, index] = (1 - fraction) * lower_codes + fraction * upper_codes
            if index in count_columns:
                counts = self.log_counts[index]
                encoded[rows, count_columns[index]] = (
                    (1 - fraction) * counts[lower_codes] + fraction * counts[upper_codes]
                )
