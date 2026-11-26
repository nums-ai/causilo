"""Checks shared by fitted input schemas."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
    is_timedelta64_dtype,
)

from ...validation import describe_column_mismatch


def table_shape(table) -> tuple[int, int]:
    """Return the row and feature counts of a nonempty table."""
    shape = table.shape if hasattr(table, "shape") else np.asarray(table).shape
    if shape and not shape[0]:
        raise ValueError("Feature tables must contain at least one row")
    if len(shape) != 2:
        raise ValueError("Expected a two-dimensional feature table")
    rows, columns = shape
    if not columns:
        raise ValueError("Feature tables must contain at least one column")
    return rows, columns


def named_column_positions(table, expected: Iterable | None) -> np.ndarray | None:
    """Locate fitted columns by name when the query has named columns."""
    if expected is None or not isinstance(table, pd.DataFrame):
        return None
    actual = pd.Index(tuple(table.columns), dtype=object, tupleize_cols=False)
    fitted = pd.Index(tuple(expected), dtype=object, tupleize_cols=False)
    if not actual.is_unique:
        raise ValueError("DataFrame columns must have unique names")
    positions = actual.get_indexer(fitted)
    if len(actual) != len(fitted) or np.any(positions < 0):
        raise ValueError(
            "Prediction columns must match the names supplied during fit. "
            + describe_column_mismatch(fitted, actual)
        )
    return None if np.array_equal(positions, np.arange(len(actual))) else positions


def check_feature_types(table) -> None:
    """Reject unsupported raw feature dtypes before converting or selecting columns."""
    dtypes = table.dtypes if isinstance(table, pd.DataFrame) else (np.asarray(table).dtype,)
    for dtype in dtypes:
        if is_datetime64_any_dtype(dtype) or is_timedelta64_dtype(dtype) or isinstance(dtype, pd.PeriodDtype):
            raise TypeError("Date and duration features must be encoded before fitting or predicting")
        if is_complex_dtype(dtype):
            raise TypeError("Complex feature values are not supported")
        if not (
            isinstance(dtype, pd.CategoricalDtype)
            or is_bool_dtype(dtype)
            or is_numeric_dtype(dtype)
            or is_object_dtype(dtype)
            or is_string_dtype(dtype)
        ):
            raise TypeError(f"Unsupported feature dtype {dtype}. Encode it as numeric or categorical first")


def check_feature_count(actual: int, expected: int) -> None:
    """Reject a table whose input width differs from its fitted width."""
    if actual != expected:
        raise ValueError(f"Expected {expected} feature columns, got {actual}")
