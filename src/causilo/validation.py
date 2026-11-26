"""Validate inputs and output options shared by the estimators and model versions."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

_NAME_PREVIEW_LIMIT = 5


def describe_column_mismatch(expected: Iterable, actual: Iterable) -> str:
    """Name the first missing and unexpected columns so wide tables are easy to fix.

    Returns an empty string when both sides hold the same set of names.
    """
    expected = pd.Index(tuple(expected), dtype=object, tupleize_cols=False)
    actual = pd.Index(tuple(actual), dtype=object, tupleize_cols=False)
    details = [
        f"{label} columns: {_preview_names(names)}"
        for label, names in (
            ("Missing", expected[~expected.isin(actual)]),
            ("unexpected", actual[~actual.isin(expected)]),
        )
        if len(names)
    ]
    return "; ".join(details)


def _preview_names(names: pd.Index) -> str:
    shown = repr(list(names[:_NAME_PREVIEW_LIMIT]))
    hidden = len(names) - _NAME_PREVIEW_LIMIT
    return f"{shown} and {hidden} more" if hidden > 0 else shown


def validate_quantiles(quantiles) -> np.ndarray:
    """Require a nonempty vector of finite probabilities strictly between zero and one."""
    levels = np.asarray(quantiles)
    if levels.ndim != 1 or not levels.size or levels.dtype.kind not in "fiu":
        raise ValueError("quantiles must be a nonempty one-dimensional numeric sequence")
    levels = levels.astype(np.float64)
    if not np.isfinite(levels).all() or np.any((levels <= 0) | (levels >= 1)):
        raise ValueError("quantiles must be finite probabilities strictly between 0 and 1")
    return levels


def validate_regression_output(output_type, quantiles):
    if not isinstance(output_type, str) or output_type not in ("mean", "median", "raw", "quantiles"):
        raise ValueError("output_type must be 'mean', 'median', 'raw', or 'quantiles'")
    if output_type != "quantiles" and quantiles is not None:
        raise ValueError("quantiles is only valid with output_type='quantiles'")
    return validate_quantiles(quantiles) if output_type == "quantiles" else None
