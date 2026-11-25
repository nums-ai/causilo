import numpy as np
import pandas as pd
import pytest
import torch

from causilo import CausiloClassifier, CausiloRegressor
from causilo.v1_5.data.columns import CanonicalColumns
from causilo.v1_5.data.dataset import PreparedDataset, TableViews, join_context
from causilo.v1_5.data.ensemble import feature_view_order, make_ensemble_plan
from causilo.v1_5.data.target import TargetPreprocessor
from causilo.v1_5.execution.direct import EnsembleRunner
from causilo.v1_5.execution.outputs import (
    MEAN_LEVELS,
    trimmed_member_quantiles,
    trimmed_regression_predictions,
)
from causilo.v1_5.execution.quantiles import interpolate_quantiles


def test_constant_columns_keep_mixed_missing_and_observed_values():
    table = np.array([
        [np.nan, np.nan, 1.0, -0.0],
        [np.nan, 1.0, np.nan, 0.0],
        [np.nan, 1.0, 1.0, -0.0],
    ])
    columns = CanonicalColumns.fit(table, effective_rows=len(table))
    np.testing.assert_array_equal(np.sort(columns.order), [1, 2])
    np.testing.assert_array_equal(columns.transform(table), table[:, columns.order])


def test_query_columns_follow_the_positive_weight_context():
    # The zero-weight row breaks the fitted tie and changes an otherwise constant column.
    table = np.array([
        [4, 7, 5, np.nan, 1],
        [1, 8, 0, np.nan, 0],
        [2, 8, 1, np.nan, 1],
        [3, 8, 2, np.nan, 2],
    ], dtype=np.float64, order="F")
    dataset = PreparedDataset.prepare(
        table, np.arange(4), task="regression", n_estimators=8, random_state=42, sample_count=[0, 1, 2, 1],
    )
    np.testing.assert_array_equal(dataset.columns.order, [2, 4, 0])
    assert dataset.columns.ties == ((0, 2),)
    query = np.array([[0.5, 99, 3, np.nan, -2], [8, 0, np.nan, 0, 1]], dtype=np.float64)
    expected = dataset.normalizer.transform(np.array([[-2, 3, 0.5], [1, np.nan, 8]], dtype=np.float64, order="F"))
    actual = dataset.query_table(query, slice(None)).values
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(observed.view(np.uint32), reference.view(np.uint32))


def test_constant_columns_are_kept_only_for_one_effective_row():
    table = np.array([[3., 7.], [2., 2.]])
    options = dict(task="regression", n_estimators=1, random_state=42)
    single = PreparedDataset.prepare(table, [0., 1.], sample_count=[0, 1], **options)
    assert single.training.shape == (1, 2)
    with pytest.raises(ValueError, match="At least one retained feature"):
        PreparedDataset.prepare(table, [0., 1.], sample_count=[0, 2], **options)


def test_canonical_ensemble_agrees_across_column_order_and_compact_repetitions():
    left = np.r_[np.zeros(8), np.arange(1, 53)]
    right = np.r_[np.zeros(8), np.arange(52, 0, -1)]
    table = pd.DataFrame({
        "left": left, "right": right, "twin": left, "kind": [f"c{index:02d}" for index in range(60)],
    })
    labels = np.arange(len(table)) % 3
    counts = np.resize([1, 4, 2], len(table))
    repeated = np.repeat(np.arange(len(table)), counts)
    query = pd.DataFrame({"left": [7.0, np.nan], "right": [1.0, 2.0], "twin": [0.0, 4.0], "kind": ["c01", None]})
    options = dict(task="classification", n_estimators=8, random_state=12)

    original = PreparedDataset.prepare(table, labels, sample_count=counts, **options)
    permuted = PreparedDataset.prepare(table.iloc[:, ::-1], labels, sample_count=counts, **options)
    expanded = PreparedDataset.prepare(table.iloc[repeated], labels[repeated], **options)
    assert original.training.shape[1] == 5
    assert original.columns.n_groups == 4
    np.testing.assert_array_equal((original.ensemble.feature_signs < 0).sum(axis=0), np.full(5, 4))
    for other in (permuted, expanded):
        np.testing.assert_array_equal(original.ensemble.feature_signs, other.ensemble.feature_signs)
        np.testing.assert_array_equal(original.ensemble.class_permutations, other.ensemble.class_permutations)

    expected = original.query_table(query, slice(None)).values
    for actual, reference in zip(permuted.query_table(query, slice(None)).values, expected, strict=True):
        np.testing.assert_array_equal(actual, reference)
    for actual, reference in zip(expanded.query_table(query, slice(None)).values, expected, strict=True):
        np.testing.assert_allclose(actual, reference, atol=1e-7, rtol=1e-7)
    for actual, reference in zip(expanded.training_table().values, original.training_table().values, strict=True):
        np.testing.assert_allclose(actual, reference[repeated], atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize(("width", "classes", "requested", "actual", "views", "trim"), [
    (1, None, 8, 4, (0, 0, 1, 2), 1),
    (1, 3, 8, 6, (0, 0, 0, 1, 1, 2), 0),
    (5, 4, 8, 8, (0, 0, 0, 0, 1, 1, 2, 2), 0),
])
def test_unique_ensemble_views_start_with_identity_and_balance_signs(width, classes, requested, actual, views, trim):
    plan = make_ensemble_plan(n_features=width, n_classes=classes, n_estimators=requested, random_state=17)
    signs = plan.feature_signs
    assert plan.feature_views == views and plan.trim_count == trim
    assert signs.shape == (actual, width)
    np.testing.assert_array_equal(signs[0], np.ones(width))
    np.testing.assert_array_equal((signs < 0).sum(axis=0), np.full(width, actual // 2))
    if classes is None:
        assert plan.target_signs[0] == 1
        targets = plan.target_signs[:, None]
    else:
        np.testing.assert_array_equal(plan.class_permutations[0], np.arange(classes))
        targets = plan.class_permutations
    assert len(np.unique(np.concatenate((signs, targets), axis=1), axis=0)) == actual


@pytest.mark.parametrize(("levels", "members", "trim"), [
    (None, 8, 2),
    ([0.9, 0.01, 0.45, 0.01], 4, 1),
    (MEAN_LEVELS, 8, 2),
], ids=["raw", "unordered-repeated-quantiles", "mean"])
def test_trimming_uses_complementary_quantiles_at_each_probability(levels, members, trim):
    rng = np.random.default_rng(36)
    raw = torch.from_numpy(np.sort(rng.normal(size=(members, 5, 17)).astype(np.float32), axis=-1))
    signs = np.resize([1, -1], members)
    target = TargetPreprocessor(classification=False)
    target.fit_transform([2.0, 8.0, 21.0])
    levels = None if levels is None else np.asarray(levels)
    output_type = "mean" if levels is MEAN_LEVELS else "raw" if levels is None else "quantiles"
    levels, requested, alignment = trimmed_member_quantiles(output_type, levels)
    if levels is None:
        outputs = raw.numpy()
        reference = np.where(signs[:, None, None] > 0, outputs, -outputs[..., ::-1])
    else:
        both_tails = interpolate_quantiles(raw, np.r_[levels, 1 - levels].tolist()).numpy()
        reference = np.where(
            signs[:, None, None] > 0, both_tails[..., :len(levels)], -both_tails[..., len(levels):],
        )
        outputs = interpolate_quantiles(raw, requested).numpy()
    actual = trimmed_regression_predictions(
        outputs, target, signs, output_type="raw" if levels is None else "quantiles",
        levels=levels, alignment=alignment, target_limits=(2.0, 21.0), trim=trim,
    )
    expected = target.inverse_transform(np.sort(reference, axis=0)[trim:-trim].mean(axis=0, dtype=np.float64))
    np.testing.assert_array_equal(actual, expected)
    if levels is MEAN_LEVELS:
        assert len(requested) == len(levels)
        mean = trimmed_regression_predictions(
            outputs, target, signs, output_type="mean", levels=levels, alignment=alignment,
            target_limits=(2.0, 21.0), trim=trim,
        )
        edges = np.column_stack([np.minimum(expected[:, 0], 2.0), expected, np.maximum(expected[:, -1], 21.0)])
        integral = ((edges[:, :-1] + edges[:, 1:]) * 0.5 * np.diff(np.r_[0, levels, 1])).sum(axis=1)
        np.testing.assert_allclose(mean, integral, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(("members", "slots", "trim"), [
    (1, (0,), 0),
    (4, (0, 0, 1, 1), 1),
    (8, (0, 0, 0, 0, 1, 1, 2, 2), 2),
])
def test_context_and_query_use_the_assigned_member_views(members, slots, trim):
    rng = np.random.default_rng(44)
    context, query = rng.lognormal(size=(40, 5)), rng.lognormal(size=(11, 5))
    labels = np.arange(len(context))
    options = dict(n_estimators=members, random_state=42)
    cls = PreparedDataset.prepare(context, labels % 3, task="classification", **options)
    reg = PreparedDataset.prepare(context, labels, task="regression", **options)
    for data in (cls, reg):
        assert data.ensemble.feature_views == slots
        assert data.view_count == len(set(slots))
        table = join_context(data.training_table(), data.query_table(query, slice(None)))
        signed = data.signed_views(table, slice(None))
        for index, view in enumerate(slots):
            unsigned = table
            if isinstance(table, TableViews):
                unsigned = np.concatenate((table.context[view], table.values[view]))
            np.testing.assert_array_equal(signed[index], unsigned * data.ensemble.feature_signs[index])
    assert cls.ensemble.trim_count == 0
    assert reg.ensemble.trim_count == trim
    reg_views = reg.normalizer.transform(query)
    for left, right in zip(cls.normalizer.transform(query), reg_views, strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(reg.ensemble.target_signs, np.resize([1, -1], members))


@pytest.mark.parametrize(("members", "views"), [(2, (0, 1)), (3, (0, 0, 1)), (5, (0, 0, 0, 1, 1))])
def test_small_ensembles_round_the_base_and_robust_view_counts(members, views):
    assert feature_view_order(members, members) == views


def test_classifier_combines_aligned_logits_before_softmax(checkpoint15, mixed_data, monkeypatch):
    from scipy.special import softmax

    context, query = mixed_data
    fitted = CausiloClassifier(
        checkpoint_path=checkpoint15("classification"), device="cpu", n_estimators=8,
    ).fit(context, np.resize(["zulu", "bravo", "alpha"], len(context)))
    state = fitted._engine.state
    classes = state.dataset.ensemble.class_permutations
    aligned = np.random.default_rng(75).normal(size=(len(classes), len(query), 3)).astype(np.float32)
    outputs = np.stack([member[:, np.argsort(order)] for member, order in zip(aligned, classes, strict=True)])
    monkeypatch.setattr(state.runner, "predict_table", lambda *args, **kwargs: outputs)
    expected = aligned.mean(axis=0)
    np.testing.assert_array_equal(fitted.classes_, ["alpha", "bravo", "zulu"])
    np.testing.assert_array_equal(fitted.predict_logits(query), expected)
    np.testing.assert_allclose(fitted.predict_proba(query), softmax(expected, axis=1), rtol=1e-6)


@pytest.mark.parametrize(("cache", "retain", "members"), [(True, True, 4), (False, False, 8)])
def test_trimmed_weighted_queries_and_chunking(checkpoint15, mixed_data, monkeypatch, cache, retain, members):
    context, query = mixed_data
    counts = np.arange(len(context)) % 4
    repeated = np.repeat(np.arange(len(context)), counts)
    y = np.arange(len(context), dtype=float) ** 1.5
    opt = dict(
        device="cpu", checkpoint_path=checkpoint15("regression"), use_kv_cache=cache,
        retain_preprocessing=retain, n_estimators=members,
    )
    weighted = CausiloRegressor(**opt).fit(context, y, sample_count=counts)
    expanded = CausiloRegressor(**opt).fit(context.iloc[repeated], y[repeated])
    settings = dict(output_type="quantiles", quantiles=[0.01, 0.5, 0.95])
    expected = expanded.predict(query, **settings)
    monkeypatch.setattr(EnsembleRunner, "query_batch_size", lambda *_, **__: 3)
    np.testing.assert_allclose(weighted.predict(query, **settings), expected, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(weighted.predict(query, output_type="median"), expected[:, 1], rtol=3e-5, atol=3e-5)
