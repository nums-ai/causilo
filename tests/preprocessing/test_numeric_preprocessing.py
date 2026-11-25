import warnings

import numpy as np
import pytest

from causilo.v1_0.data.normalization import Normalizer as Normalizer10
from causilo.v1_5.data.normalization import outlier_bounds


@pytest.mark.parametrize(("dtype", "order"), [(np.float32, "C"), (np.float64, "F")])
def test_outlier_bounds_match_nan_reductions(dtype, order):
    rng = np.random.default_rng(719)
    values = np.array(rng.normal(size=(129, 7)), dtype=dtype, order=order)
    values[::5, 2:] = np.nan
    values[:, :2] = np.nan
    values[-1, 1] = 8
    values[0, -1] = 100

    def bounds(table):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nan_to_num(np.nanmean(table, axis=0))
            spread = np.maximum(np.nan_to_num(np.nanstd(table, axis=0, ddof=1)), 1e-6)
        return mean - 4 * spread, mean + 4 * spread

    lower, upper = bounds(values)
    central = values.copy()
    central[(central < lower) | (central > upper)] = np.nan
    original = values.copy()
    for actual, expected in zip(outlier_bounds(values), bounds(central)):
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
    np.testing.assert_array_equal(original, values)


@pytest.mark.parametrize("method", ["none", "power", "rank2gaussian", "robust"])
def test_normalization_keeps_missing_mask_and_fitted_state(method):
    table = np.random.default_rng(12).normal(size=(60, 4))
    table[::5, 1] = np.nan
    transform = Normalizer10.fit(table, method)
    first = transform.transform(table)
    transform.transform(table[:3] * 100)
    np.testing.assert_array_equal(first, transform.transform(table))
    np.testing.assert_array_equal(np.isnan(first), np.isnan(table))
    assert np.isfinite(first[~np.isnan(first)]).all()
