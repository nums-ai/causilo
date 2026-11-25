"""Validate regression output options shared by the estimators."""

import numpy as np


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
