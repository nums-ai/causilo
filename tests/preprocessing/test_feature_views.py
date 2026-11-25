"""Exercise the fitted feature views before applying member signs."""

import numpy as np
import pytest
from scipy.special import ndtri

from causilo.v1_0.data.normalization import Normalizer, Rank2Gaussian
from causilo.v1_5.data import feature_views
from causilo.v1_5.data.feature_views import FeatureViews


def test_view_transforms_and_fitted_rank_endpoints():
    context = np.array([[0.0], [0.0], [1.0], [3.0]])
    query = np.array([[0.5], [3.0], [4.0], [np.nan]])
    processor, training = FeatureViews.fit_transform(context)
    views = processor.transform(query)

    initial = (context[:, 0] - context.mean()) / (context.std() + 1e-6)
    query_initial = (query[:, 0] - context.mean()) / (context.std() + 1e-6)
    np.testing.assert_allclose(training[0][:, 0], initial, rtol=1e-6)
    np.testing.assert_allclose(views[0][:, 0], query_initial, rtol=1e-6)
    quartiles = np.quantile(initial, [0.25, 0.5, 0.75])
    robust = (query_initial - quartiles[1]) / (quartiles[2] - quartiles[0])
    np.testing.assert_allclose(views[1][:, 0], robust / np.sqrt(1 + (robust / 3) ** 2), rtol=1e-6)
    fitted_ranks = ndtri([0.25, 0.25, 0.625, 0.875])
    query_ranks = (ndtri([0.4375, 0.875, 0.875, np.nan]) - fitted_ranks.mean()) / fitted_ranks.std()
    np.testing.assert_allclose(views[2][:, 0], query_ranks, rtol=1e-6)
    assert all(view.dtype == np.float32 for view in training + views)
    for actual, expected in zip(processor.transform(context), training, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_observed_only_fit_and_arcsinh_outside_four_sigma():
    context = np.array([[0.0], [1.0], [2.0], [np.nan]])
    query = np.array([[1.0], [10_000.0], [-10_000.0], [np.nan]])
    processor, views = FeatureViews.fit_transform(context)
    assert all(np.isnan(view[-1, 0]) for view in views)
    spread = np.std(context[:3], ddof=0) + 1e-6
    lower = -4 / spread
    upper = 4 / spread
    expected = [0.0, upper + np.arcsinh(9_999 / spread - upper),
                lower - np.arcsinh(lower + 10_001 / spread), np.nan]
    np.testing.assert_allclose(processor.transform(query)[0][:, 0], expected, rtol=1e-6)


def test_frequency_weighted_views_match_materialized_repeats():
    rng = np.random.default_rng(451)
    table = np.array(rng.integers(-5, 6, (90, 8)), dtype=np.float64, order="F")
    table[::3, 0] = np.nan
    table[:, 2] = 0.5
    table[:, 3] = np.nan
    table[0, 4] = 10_000
    query = np.vstack((table[:17], rng.normal(size=(21, table.shape[1]))))
    table.setflags(write=False)
    query.setflags(write=False)
    counts = rng.integers(0, 9, len(table))
    expected, _ = FeatureViews.fit_transform(np.repeat(table, counts, axis=0), retain=False)
    weighted, retained = FeatureViews.fit_transform(table, counts)

    query_views = weighted.transform(query)
    for left, right in zip(query_views, expected.transform(query), strict=True):
        np.testing.assert_allclose(left, right, rtol=2e-6, atol=2e-6)
    for left, right in zip(weighted.transform(table), retained, strict=True):
        np.testing.assert_array_equal(left, right)
    assert np.isnan(query_views[1][:, 3]).all()
    np.testing.assert_array_equal(np.isnan(query_views[2]), np.isnan(query))


def test_rank_view_matches_1_0_with_missing_values_and_full_ranks(monkeypatch):
    context = np.arange(20_001.0)[:, None]
    context[::19] = np.nan
    query = np.array([[-1.0], [1.0], [951.3], [20000.0], [30000.0], [np.nan]])
    reference = Normalizer.fit(context, "rank2gaussian")
    complete, _ = FeatureViews.fit_transform(context, retain=False)
    expected = reference.transform(query)
    np.testing.assert_array_equal(complete.transform(query)[2], expected)
    np.testing.assert_array_equal(expected[[0, 3]], expected[[1, 4]])

    def unexpected_standardization(*args, **kwargs):
        pytest.fail("A rank-only context does not use Base or Robust statistics")

    monkeypatch.setattr(feature_views, "_fit_standardization", unexpected_standardization)
    selected, (ranked,) = FeatureViews.fit_transform(context, view_count=1, selected_view=2)
    np.testing.assert_array_equal(ranked, reference.transform(context))
    np.testing.assert_array_equal(selected.transform(query)[0], expected)


def test_finite_values_near_float64_limit():
    maximum = np.finfo(np.float64).max
    context = np.array([[-0.9 * maximum, 0], [0.8 * maximum, 0], [maximum, 0]])
    processor, views = FeatureViews.fit_transform(context)
    assert all(np.isfinite(view).all() for view in views)
    outputs = processor.transform(np.array([[-maximum, -maximum], [maximum, maximum]]))
    assert all(np.isfinite(view).all() for view in outputs)
    assert outputs[0][1, 1] > 700
    np.testing.assert_allclose(outputs[0][0, 1], -outputs[0][1, 1], rtol=1e-6)


@pytest.mark.parametrize(("magnitude", "weighted"), [(1e155, False), (np.finfo(np.float64).max, True)])
def test_finite_extreme_feature_retains_its_scale_and_other_columns(magnitude, weighted):
    units = np.array([1., 1., 1., -1., -.5, np.nan])
    ordinary = np.array([3., 0., 1., -2., np.nan, 9.])
    table = np.column_stack([magnitude * units, ordinary])
    counts = np.array([2, 3, 2, 1, 1, 1]) if weighted else None
    present = ~np.isnan(units)
    observed = units[present]
    weights = counts[present] if weighted else None
    center = np.average(observed, weights=weights)
    spread = np.sqrt(np.average((observed - center) ** 2, weights=weights))
    expected = (units - center) / spread
    _, (reference,) = FeatureViews.fit_transform(np.column_stack((units, ordinary)), counts, view_count=1)

    with np.errstate(over="raise", invalid="raise"):
        processor, (context,) = FeatureViews.fit_transform(table, counts, view_count=1)
        (query,) = processor.transform(table[::-1])

    np.testing.assert_allclose(context[:, 0], expected, rtol=1e-6)
    np.testing.assert_array_equal(query, context[::-1])
    np.testing.assert_array_equal(context[:, 1].view(np.uint32), reference[:, 1].view(np.uint32))


def test_largest_constant_with_missing_cells_has_zero_variation():
    largest = np.finfo(np.float64).max
    table = np.array([[largest], [np.nan], [largest]])
    with np.errstate(over="raise", invalid="raise"):
        processor, (context,) = FeatureViews.fit_transform(table, view_count=1)
        (query,) = processor.transform([[-largest], [largest]])

    np.testing.assert_array_equal(context[:, 0], [0., np.nan, 0.])
    assert np.isfinite(query).all() and query[0, 0] < -700 and query[1, 0] == 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_adjacent_extreme_values_keep_their_centered_difference(dtype):
    upper = dtype(np.finfo(dtype).max)
    lower = np.nextafter(upper, dtype(0))
    table = np.array([[upper], [lower], [upper], [lower]], dtype=dtype)
    with np.errstate(over="raise", invalid="raise"):
        processor, (context,) = FeatureViews.fit_transform(table, view_count=1)
        (query,) = processor.transform(table[::-1])

    np.testing.assert_array_equal(context[:, 0], [1., -1., 1., -1.])
    np.testing.assert_array_equal(query, context[::-1])


def test_maximum_row_count_does_not_overflow_rank_probabilities():
    context = np.array([[0.0], [1.0]])
    processor, training = FeatureViews.fit_transform(context, [2**53 - 1, 1])
    assert np.isfinite(training[2]).all()
    assert training[2].tobytes() == processor.transform(context)[2].tobytes()


@pytest.mark.parametrize("retain", [False, True])
def test_rank_fit_reuses_transformed_training_rows(monkeypatch, retain):
    rng = np.random.default_rng(330)
    table = np.array(rng.integers(-3, 5, (12, 6)), dtype=np.float64, order="F")
    table[::3, 0] = np.nan
    table[:, 2] = np.nan
    counts = np.arange(len(table)) % 4 + 1
    ranges = (slice(0, 3), slice(3, 6))
    expected = np.column_stack([
        Normalizer.fit(table[:, columns], "rank2gaussian", counts).transform(table[:, columns])
        for columns in ranges
    ])
    original_transform, original_restore = Rank2Gaussian.transform, Normalizer._restore
    transforms, outputs = [], []

    def transform(self, rows):
        transforms.append(rows.shape)
        return original_transform(self, rows)

    def restore(self, values, missing, output):
        outputs.append(output)
        return original_restore(self, values, missing, output)

    monkeypatch.setattr(feature_views, "_BLOCK_CELLS", 36)
    monkeypatch.setattr(Rank2Gaussian, "transform", transform)
    monkeypatch.setattr(Normalizer, "_restore", restore)
    processor, training = FeatureViews.fit_transform(table, counts, retain=retain)
    assert transforms == [(12, 3)] * 2
    if retain:
        assert len(outputs) == 2 and all(np.shares_memory(output, training[2]) for output in outputs)
        assert training[2].tobytes() == expected.tobytes()
    else:
        assert training is None and not outputs
    assert processor.transform(table)[2].tobytes() == expected.tobytes()


@pytest.mark.parametrize("counts", [[0, 0], [2**53, 1], [0.5, 1], [1, np.nan]])
def test_direct_view_fit_rejects_invalid_counts(counts):
    with pytest.raises(ValueError, match="sample_count"):
        FeatureViews.fit_transform([[0], [1]], counts)


@pytest.mark.parametrize("count", [1, 2])
def test_small_ensembles_skip_unused_view_fits(monkeypatch, count):
    rng = np.random.default_rng(67)
    table, query = rng.normal(size=(40, 5)), rng.normal(size=(11, 5))
    complete, training = FeatureViews.fit_transform(table)
    expected = complete.transform(query)
    def unexpected_fit(*args, **kwargs):
        pytest.fail("Fitted an unused view")

    monkeypatch.setattr(feature_views.RankNormalizer, "_fit", unexpected_fit)
    if count == 1:
        monkeypatch.setattr(feature_views, "_fit_robust", unexpected_fit)

    processor, retained = FeatureViews.fit_transform(table, view_count=count)
    for actual, reference in zip(retained, training[:count], strict=True):
        np.testing.assert_array_equal(actual, reference)
    for actual, reference in zip(processor.transform(query), expected[:count], strict=True):
        np.testing.assert_array_equal(actual, reference)
