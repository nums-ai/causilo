"""Oracle comparisons materialize repeats only on the reference side."""

import warnings

import numpy as np
import pytest

from causilo.data.frequency import frequency_quantile, validate_frequencies
from causilo.data.normalization import Normalizer


@pytest.mark.parametrize("method", ["none", "rank2gaussian", "robust", "power"])
@pytest.mark.parametrize("case", ["ties_missing", "outliers", "all_ones", "zero_rows", "float32", "large_frequency"])
def test_matches_expanded_data(method, case):
    rng = np.random.default_rng(93)
    table = rng.normal(size=(17, 5))
    weights = rng.integers(1, 20, len(table))
    query = np.vstack((table, rng.normal(size=(8, 5)) * 4))
    if case == "ties_missing":
        table[:6, 1] = 3
        table[::3, 2] = np.nan
        table[:, 3] = 5
        table[:, 4] = np.nan
        query[::3, 2] = np.nan
        query[:, 3] = 5
        query[:, 4] = np.nan
    elif case == "outliers":
        table[0, 0] = 1e5
        table[1, 1] = -1e4
    elif case == "all_ones":
        weights[:] = 1
    elif case == "zero_rows":
        weights[[1, 4, 7]] = 0
        table[[1, 4, 7]] = 1e5
    elif case == "float32":
        table, query = table.astype(np.float32), query.astype(np.float32)
    elif case == "large_frequency":
        weights[0] = 10000
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = Normalizer.fit(np.repeat(table, weights, axis=0), method).transform(query)
        actual = Normalizer.fit(table, method, weights).transform(query)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5, equal_nan=True)


@pytest.mark.parametrize("weights", [[1, 1, 1], [1, 2, 4], [2, 2, 2], [20, 1, 3]])
def test_quantiles_match_numpy_linear_interpolation(weights):
    values = np.array([3., 1., 1.])
    quantiles = np.linspace(0, 1, 21)
    expected = np.quantile(np.repeat(values, weights), quantiles)
    np.testing.assert_allclose(frequency_quantile(values, np.array(weights), quantiles), expected)


@pytest.mark.parametrize("weights", [[1, -1], [0, 0], [1, .5], [1, np.nan], [1]])
def test_invalid_frequency(weights):
    with pytest.raises(ValueError):
        Normalizer.fit(np.array([[1.], [2.]]), "none", weights)


def test_large_frequencies_do_not_expand_rows(monkeypatch):
    def no_repeat(*args, **kwargs):
        pytest.fail("Weighted normalization expanded rows")
    monkeypatch.setattr(np, "repeat", no_repeat)
    table = np.array([[1., 3.], [2., 5.], [7., np.nan]])
    for method in ("none", "rank2gaussian", "robust", "power"):
        result = Normalizer.fit(table, method, np.array([1_000_000, 2, 7]))
        assert result.transform(table).shape == table.shape


@pytest.mark.parametrize("weights", [[True, False], [1, np.inf], [2**53, 1], [2**64 - 1, 1], [[1], [2]]])
def test_rejects_invalid_or_inexact_counts(weights):
    with pytest.raises(ValueError, match="sample_weight"):
        validate_frequencies(weights, 2)


def test_validation_copies_input_and_accepts_integral_floats():
    original = np.array([1., 0., 3.])
    validated = validate_frequencies(original, 3)
    original[0] = 10
    np.testing.assert_array_equal(validated, [1, 0, 3])
    assert validated.dtype == np.int64


@pytest.mark.parametrize("method", ["none", "rank2gaussian", "robust", "power"])
def test_unit_weights_preserve_exact_unweighted_results(method):
    table = np.random.default_rng(65).normal(size=(31, 4))
    table[::3, 2] = np.nan
    query = np.random.default_rng(66).normal(size=(13, 4))
    old = Normalizer.fit(table, method)
    weighted = Normalizer.fit(table, method, np.ones(len(table)))
    np.testing.assert_array_equal(weighted.transform(query), old.transform(query))


@pytest.mark.parametrize("method", ["none", "rank2gaussian", "robust", "power"])
def test_single_repeated_value_matches_expansion(method):
    table = np.array([[5., 1., np.nan]])
    expected = Normalizer.fit(np.repeat(table, 5, axis=0), method)
    actual = Normalizer.fit(table, method, [5])
    np.testing.assert_allclose(actual.transform(table), expected.transform(table), equal_nan=True)


def test_maximum_accepted_total_keeps_gaussian_ranks_finite():
    table = np.array([[0.], [1.]])
    counts = np.array([2**53 - 1, 1], dtype=np.int64)
    normalizer = Normalizer.fit(table, "rank2gaussian", counts)
    assert np.isfinite(normalizer.normalizer.transform(table)).all()
    assert np.isfinite(normalizer.transform(table)).all()


@pytest.mark.parametrize("tie", [0.3, np.nextafter(0.3, 1.0), np.nextafter(0.3, 0.0)])
@pytest.mark.parametrize("layout", ["original", "reversed", "split", "zero_row"])
def test_missing_mean_ranks_do_not_depend_on_row_expansion(tie, layout):
    table = np.array([np.nan, .5, 0., tie, tie, np.nan])[:, None]
    weights = np.array([8, 6, 4, 1, 5, 1])
    expanded = np.repeat(table, weights, axis=0)
    if layout == "reversed":
        table, weights = table[::-1], weights[::-1]
    elif layout == "split":
        table = np.concatenate([table, table[[0]]])
        weights = np.array([3, 6, 4, 1, 5, 1, 5])
    elif layout == "zero_row":
        table = np.concatenate([table, [[1e10]]])
        weights = np.append(weights, 0)
    compressed = Normalizer.fit(table, "rank2gaussian", weights)
    reference = Normalizer.fit(expanded, "rank2gaussian")
    permuted = Normalizer.fit(expanded[::-1], "rank2gaussian")
    np.testing.assert_array_equal(compressed.fill, reference.fill)
    np.testing.assert_array_equal(permuted.fill, reference.fill)
    for actual in (compressed, permuted):
        for left, right in zip(actual.normalizer.knots[0], reference.normalizer.knots[0]):
            np.testing.assert_array_equal(left, right)
        query = np.array([[0.], [.2], [tie], [.5], [np.nan]])
        np.testing.assert_allclose(actual.transform(query), reference.transform(query), atol=2e-6, rtol=2e-6)


def test_missing_mean_does_not_merge_distinct_neighboring_observations():
    lower, upper = .3, np.nextafter(.3, 1.)
    table = np.array([[lower], [upper], [np.nan], [.5], [0.]])
    result = Normalizer.fit(table, "rank2gaussian", [1, 2, 3, 4, 5])
    knots = result.normalizer.knots[0][0]
    assert lower in knots and upper in knots
    assert np.count_nonzero((knots >= lower) & (knots <= upper)) == 2


@pytest.mark.parametrize("values", [[-1e307, 1e307], [1e308, 1.7e308], [0., 0.]])
def test_missing_rank_mean_remains_finite_for_large_values(values):
    table = np.array([*values, np.nan])[:, None]
    with np.errstate(over="raise", invalid="raise"):
        compact = Normalizer.fit(table, "rank2gaussian", [100, 100, 7])
        expanded = Normalizer.fit(np.repeat(table, [100, 100, 7], axis=0), "rank2gaussian")
    assert np.isfinite(compact.fill).all()
    np.testing.assert_array_equal(compact.fill, expanded.fill)
    np.testing.assert_allclose(compact.transform(table), expanded.transform(table), atol=2e-6, rtol=2e-6)
