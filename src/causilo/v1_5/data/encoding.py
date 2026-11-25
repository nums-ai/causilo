"""Numerical and categorical input encoding."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype, is_bool_dtype, is_numeric_dtype
from sklearn.utils.validation import check_array

from .categorical import CategoricalEncoder
from .columns import CanonicalColumns
from .schema import check_feature_count, check_feature_types, named_column_positions, table_shape


def _to_float_array(table):
    """Convert numerical containers to float while preserving missing cells."""
    if hasattr(table, "to_numpy"):
        try:
            return table.to_numpy(dtype=np.float64, na_value=np.nan)
        except TypeError:
            pass
    values = np.asarray(table)
    if values.dtype.kind == "O":
        values = np.where(pd.isna(values), np.nan, values)
    return np.asarray(values, dtype=np.float64)


def _categorical_columns(table) -> tuple[int, ...]:
    if isinstance(table, pd.DataFrame):
        return tuple(
            index for index, dtype in enumerate(table.dtypes) if is_bool_dtype(dtype) or not is_numeric_dtype(dtype)
        )
    values = np.asarray(table)
    if values.dtype.kind == "O":
        numeric_types = {"integer", "floating", "mixed-integer-float", "decimal"}
        return tuple(
            index for index, column in enumerate(values.T) if infer_dtype(column, skipna=True) not in numeric_types
        )
    return tuple(range(values.shape[1])) if values.dtype.kind in {"b", "S", "U"} else ()


def _select_columns(table, indices: tuple[int, ...], width: int):
    if len(indices) == width:
        return table
    return table.iloc[:, list(indices)] if isinstance(table, pd.DataFrame) else np.asarray(table)[:, indices]


def _encoded_table(categories, numerical) -> np.ndarray:
    values = numerical if categories is None else np.concatenate((categories, numerical), axis=1)
    return check_array(values, dtype=np.float64, ensure_all_finite="allow-nan")


@dataclass(frozen=True)
class InputLayout:
    """Column types and the optional fixed set of categorical frequency features."""

    names: tuple | None
    width: int
    categorical_columns: tuple[int, ...]
    numerical_columns: tuple[int, ...]
    count_columns: tuple[int, ...] | None = None

    @classmethod
    def from_table(cls, table) -> InputLayout:
        _, width = table_shape(table)
        check_feature_types(table)
        names = tuple(table.columns) if isinstance(table, pd.DataFrame) else None
        if names is not None and not table.columns.is_unique:
            raise ValueError("DataFrame columns must have unique names")
        categorical = _categorical_columns(table)
        categorical_indices = set(categorical)
        numerical = tuple(index for index in range(width) if index not in categorical_indices)
        return cls(names, width, categorical, numerical)


def fit_sampling_columns(table, sample_count) -> tuple[InputLayout, CanonicalColumns]:
    """Scan the eligible context for feature structure, one input column at a time."""
    layout = InputLayout.from_table(table)
    count_columns = []
    category_width = len(layout.categorical_columns)

    def read_input(index):
        section = slice(index, index + 1)
        return table.iloc[:, section] if isinstance(table, pd.DataFrame) else table[:, section]

    def encode_category(index):
        values = read_input(layout.categorical_columns[index])
        return CategoricalEncoder.fit(values, sample_count)[1]

    def encode_number(index):
        values = read_input(layout.numerical_columns[index])
        return _encoded_table(None, _to_float_array(values))[:, 0]

    def encoded_columns():
        for index in range(category_width):
            encoded = encode_category(index)
            yield index, encoded[:, 0]
            if encoded.shape[1] == 2:
                yield category_width + len(count_columns), encoded[:, 1]
                count_columns.append(index)
        start = category_width + len(count_columns)
        for index in range(len(layout.numerical_columns)):
            yield start + index, encode_number(index)

    def read_column(index):
        if index < category_width:
            return encode_category(index)[:, 0]
        index -= category_width
        if index < len(count_columns):
            return encode_category(count_columns[index])[:, 1]
        return encode_number(index - len(count_columns))

    columns = CanonicalColumns.fit_streamed(encoded_columns(), read_column)
    return replace(layout, count_columns=tuple(count_columns)), columns


@dataclass(frozen=True)
class FeatureEncoder:
    """Fitted input schema and category features."""

    names: tuple | None
    width: int
    categorical_columns: tuple[int, ...]
    numerical_columns: tuple[int, ...]
    categorical: CategoricalEncoder | None

    @classmethod
    def fit(
        cls, table, *, sample_count: np.ndarray | None = None, layout: InputLayout | None = None,
    ) -> tuple[FeatureEncoder, np.ndarray]:
        """Fit context values, using a supplied input layout when sampling."""
        if layout is None:
            layout = InputLayout.from_table(table)
        categorical_columns, numerical_columns = layout.categorical_columns, layout.numerical_columns
        categorical, categories = None, None
        if categorical_columns:
            values = _select_columns(table, categorical_columns, layout.width)
            categorical, categories = CategoricalEncoder.fit(values, sample_count, count_columns=layout.count_columns)
        numerical = _to_float_array(_select_columns(table, numerical_columns, layout.width))
        encoded = _encoded_table(categories, numerical)
        return cls(layout.names, layout.width, categorical_columns, numerical_columns, categorical), encoded

    @property
    def feature_names(self) -> np.ndarray | None:
        if self.names is not None and all(type(name) is str for name in self.names):
            return np.asarray(self.names, dtype=object)
        return None

    def query_layout(self, table) -> tuple[int, np.ndarray | None]:
        """Validate the whole query and locate fitted columns for its slices."""
        rows, width = table_shape(table)
        check_feature_count(width, self.width)
        positions = named_column_positions(table, self.names)
        check_feature_types(table)
        return rows, positions

    def transform(self, table, *, layout: tuple[int, np.ndarray | None] | None = None) -> np.ndarray:
        """Encode a query or a slice of a query whose layout is already known."""
        _, positions = self.query_layout(table) if layout is None else layout
        if positions is not None:
            table = table.iloc[:, positions]
        categories = None
        if self.categorical is not None:
            values = _select_columns(table, self.categorical_columns, self.width)
            categories = self.categorical.transform(values)
        numerical = _to_float_array(_select_columns(table, self.numerical_columns, self.width))
        return _encoded_table(categories, numerical)
