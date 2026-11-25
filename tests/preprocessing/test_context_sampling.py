import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from causilo import CausiloClassifier, CausiloRegressor
from causilo.v1_5.data.columns import CanonicalColumns
from causilo.v1_5.data.context_sampling import sample_context_rows, validate_context_limit
from causilo.v1_5.data.encoding import FeatureEncoder, fit_sampling_columns
from causilo.v1_5.execution import fit
from causilo.v1_5.execution.direct import EnsembleRunner


@pytest.mark.parametrize("limit", [None, 20])
def test_full_context_skips_sampling(limit):
    assert sample_context_rows(20, limit, seed=42, member_index=0) is None


@pytest.mark.parametrize("limit", [True, np.bool_(False), 0, 1.0])
def test_context_limit_rejects_nonpositive_or_noninteger_values(limit):
    with pytest.raises(ValueError, match="positive integer or None"):
        validate_context_limit(limit)


def test_context_limit_accepts_numpy_integer():
    limit = validate_context_limit(np.int64(7))
    assert type(limit) is int and limit == 7


def test_member_samples_are_reproducible_and_keep_input_order():
    groups = [sample_context_rows(257, 32, seed=42, member_index=i) for i in reversed(range(8))]
    assert len({rows.tobytes() for rows in groups}) == 8
    for i, rows in enumerate(reversed(groups)):
        assert rows.dtype == np.intp and rows.shape == (32,)
        assert 0 <= rows[0] < rows[-1] < 257
        assert np.all(np.diff(rows) > 0)
        np.testing.assert_array_equal(rows, sample_context_rows(257, 32, seed=42, member_index=i))
        assert not np.array_equal(rows, sample_context_rows(257, 32, seed=43, member_index=i))


def test_class_repair_changes_only_as_many_rows_as_needed():
    rows = sample_context_rows(50, 6, seed=42, member_index=2)
    rare_rows = np.setdiff1d(np.arange(50), rows)[:5]
    labels = np.zeros(50, dtype=np.int64)
    labels[rare_rows[:3]] = 3
    labels[rare_rows[3:]] = 7

    repaired = sample_context_rows(50, 6, seed=42, member_index=2, labels=labels)
    assert len(repaired) == 6
    assert np.all(np.diff(repaired) > 0)
    assert len(np.intersect1d(rows, repaired)) == 4
    np.testing.assert_array_equal(np.unique(labels[repaired]), [0, 3, 7])
    np.testing.assert_array_equal(repaired, sample_context_rows(50, 6, seed=42, member_index=2, labels=labels))


def test_class_repair_preserves_an_already_covered_draw():
    rows = sample_context_rows(20, 4, seed=42, member_index=0)
    labels = np.zeros(20, dtype=np.int64)
    labels[rows[0]] = 1
    np.testing.assert_array_equal(rows, sample_context_rows(20, 4, seed=42, member_index=0, labels=labels))


def test_class_count_can_fill_but_not_exceed_the_budget():
    labels = np.repeat(np.arange(3), [12, 2, 1])
    rows = sample_context_rows(15, 3, seed=7, member_index=0, labels=labels)
    np.testing.assert_array_equal(np.sort(labels[rows]), [0, 1, 2])
    with pytest.raises(ValueError, match="at least the number of observed classes"):
        sample_context_rows(15, 2, seed=7, member_index=0, labels=labels)


@pytest.mark.parametrize("labels", [np.zeros(19, dtype=int), np.zeros(20, dtype=float)])
def test_active_sampler_checks_label_alignment_and_encoding(labels):
    with pytest.raises(ValueError, match="one encoded integer per eligible context row"):
        sample_context_rows(20, 4, seed=42, member_index=0, labels=labels)


@pytest.mark.parametrize(("constructor", "task", "mode"), [
    (CausiloClassifier, "classification", "light"),
    (CausiloRegressor, "regression", "default"),
])
def test_default_keeps_context_above_previous_cap(constructor, task, mode, checkpoint15, monkeypatch):
    table = np.random.default_rng(42).normal(size=(200_001, 2))
    targets = np.arange(len(table)) % 7
    model = constructor(mode=mode, device="cpu", checkpoint_path=checkpoint15(task))
    assert model.max_context is None
    monkeypatch.setattr(fit, "sample_context_rows", lambda *a, **kw: pytest.fail("Sampled without a cap"))

    model.fit(table, targets)
    assert len(model._engine.state.dataset.training) == len(table)

    model.fit_batched({"table": (table, targets)})
    assert len(model._engine.batch_states["table"].dataset.training) == len(table)


@pytest.mark.parametrize(("cache", "members"), [(True, 1), (False, 8)])
def test_weighted_sampling_keeps_eligible_rows_and_classes(checkpoint15, mixed_data, monkeypatch, cache, members):
    train, query = mixed_data
    labels = np.resize(["red", "blue"], len(train))
    labels[-1] = "rare"
    counts = np.arange(len(train)) % 4
    labels[0] = "drop"
    opt = dict(n_estimators=members, use_kv_cache=cache, device="cpu", checkpoint_path=checkpoint15("classification"))
    seen = []

    def record(*args, **kwargs):
        rows = sample_context_rows(*args, **kwargs)
        seen.append(rows)
        return rows

    monkeypatch.setattr(fit, "sample_context_rows", record)
    small = clone(CausiloClassifier(**opt, max_context=12)).fit(train, labels, sample_count=counts)
    assert len(seen) == len({rows.tobytes() for rows in seen}) == members
    state = small._engine.state
    member_states = state.members if members > 1 else (state,)
    for rows, member in zip(seen, member_states, strict=True):
        assert len(member.dataset.training) == 12
        np.testing.assert_array_equal(member.dataset.frequencies, counts[counts > 0][rows])
        np.testing.assert_array_equal(member.dataset.fitted_attributes()["classes_"], ["blue", "rare", "red"])
    eligible = counts > 0
    _, full = FeatureEncoder.fit(train.iloc[eligible], sample_count=counts[eligible])
    columns = CanonicalColumns.fit(full, effective_rows=counts.sum())
    for member in member_states:
        np.testing.assert_array_equal(member.dataset.columns.order, columns.order)
    assert np.isfinite(small.predict_proba(query)).all()


@pytest.mark.parametrize(("cache", "members", "outputs"), [
    (True, 8, ("mean", "quantiles")),
    (False, 3, ("median", "raw")),
])
def test_sampled_and_full_contexts_round_trip_together(
    checkpoint15, mixed_data, monkeypatch, tmp_path, cache, members, outputs,
):
    train, query = mixed_data
    y = 1e6 + np.arange(len(train), dtype=np.float64) ** 2
    model = CausiloRegressor(
        n_estimators=members, max_context=17, use_kv_cache=cache, retain_preprocessing=False,
        checkpoint_path=checkpoint15("regression"), device="cpu",
    ).fit_batched({"sampled": (train, y), "shared": (train.iloc[:12], y[:12])})
    path = tmp_path / "context"
    model.save_context(path)
    restored = CausiloRegressor.load_context(path, device="cpu")
    assert restored.max_context == 17
    for output_type in outputs:
        options = {"X": query, "output_type": output_type}
        if output_type == "quantiles":
            options["quantiles"] = [0.1, 0.5, 0.9]
        queries = {"sampled": options, "shared": options}
        old = model.predict_batched(queries)
        new = restored.predict_batched(queries)
        for key in old:
            np.testing.assert_array_equal(old[key], new[key])
        if output_type == "mean":
            with monkeypatch.context() as patch:
                patch.setattr(EnsembleRunner, "query_batch_size", lambda *args, **kwargs: 3)
                split = restored.predict_batched(queries)
            np.testing.assert_allclose(split["sampled"], old["sampled"], rtol=0, atol=1e-4)


def test_version_1_0_keeps_the_full_context(checkpoint_factory, mixed_data):
    train, query = mixed_data
    kw = {"checkpoint_path": checkpoint_factory("classification"), "device": "cpu", "n_estimators": 1}
    labels = np.arange(len(train)) % 2
    full = CausiloClassifier(**kw, max_context=None).fit(train, labels)
    capped = CausiloClassifier(**kw, max_context=2).fit(train, labels)
    np.testing.assert_array_equal(capped.predict_proba(query), full.predict_proba(query))


def test_sampled_ecoc_uses_one_context_and_keeps_all_classes(checkpoint15, mixed_data):
    train, query = mixed_data
    labels = np.arange(len(train)) % 11
    model = CausiloClassifier(checkpoint_path=checkpoint15("classification"), device="cpu", max_context=16)
    model.fit(train, labels)
    state = model._engine.state
    assert isinstance(state, fit.FitState) and state.dataset.codec is not None
    assert len(state.dataset.training) == 16
    np.testing.assert_array_equal(model.classes_, np.arange(11))
    np.testing.assert_allclose(model.predict_proba(query).sum(axis=1), 1, rtol=1e-6)
    with pytest.raises(ValueError, match="max_context must be at least"):
        model.set_params(max_context=10).fit(train, labels)


@pytest.mark.parametrize(("constructor", "task", "expected", "trim"), [
    (CausiloClassifier, "classification", (0, 0, 1, 1), 0),
    (CausiloRegressor, "regression", (0, 0, 0, 0, 1, 1, 2, 2), 2),
])
def test_sampled_contexts_keep_member_views_and_trim(checkpoint15, mixed_data, constructor, task, expected, trim):
    train, _ = mixed_data
    labels = np.arange(len(train)) % 3 if task == "classification" else np.arange(len(train), dtype=float)
    model = constructor(
        checkpoint_path=checkpoint15(task), device="cpu", n_estimators=len(expected), max_context=17,
    ).fit(train, labels)
    state = model._engine.require_state()
    assert tuple(member.dataset.normalizer.selected_view for member in state.members) == expected
    assert state.trim_count == trim



@pytest.mark.parametrize("array", [False, True])
def test_sampled_layout_matches_full_encoding(array):
    count = 1100
    index = np.arange(count)
    sparse = np.where(index < 1050, np.nan, index)
    category = np.resize(["b", "a", "c", None], count)
    many = np.where(index < 1049, 0, index)
    train = pd.DataFrame({
        "sparse": sparse, "duplicate": sparse, "varied_suffix": np.where(index < 1050, sparse, sparse + 1),
        "category": pd.Categorical(category), "category_duplicate": category,
        "many": pd.Categorical(many), "many_duplicate": pd.Categorical(many),
        "constant": 1, "missing": np.nan,
    })
    if array:
        train = train.to_numpy(dtype=object)
    weights = 1 + index % 4
    encoder, encoded = FeatureEncoder.fit(train, sample_count=weights)
    expected = CanonicalColumns.fit(encoded, effective_rows=weights.sum())

    layout, actual = fit_sampling_columns(train, weights)
    assert layout.categorical_columns == encoder.categorical_columns
    assert layout.numerical_columns == encoder.numerical_columns
    assert layout.count_columns == tuple(encoder.categorical.log_counts)
    np.testing.assert_array_equal(actual.order, expected.order)
    np.testing.assert_array_equal(actual.groups, expected.groups)
    assert actual.ties == expected.ties


@pytest.mark.parametrize(("cache", "classes", "estimators"), [(False, 3, 8), (True, 3, 1), (True, 12, 8)])
def test_fixed_sampled_width_and_local_category_frequencies(
    checkpoint15, monkeypatch, tmp_path, cache, classes, estimators,
):
    index = np.arange(101)
    levels = np.where(index < 56, index.astype(str), "shared")
    table = pd.DataFrame({
        "kind": pd.Categorical(levels),
        "rare": np.where(index == 0, 1.0, 0.0),
        "duplicate": np.where(index == 0, 1.0, 0.0),
        "dropped": np.where(index == 100, 2.0, 0.0),
        "missing": np.nan,
    })
    weights = 1 + index % 2
    weights[-1] = 0
    masks = [np.arange(45, 100), np.arange(55)]
    masks += [np.arange(start, start + 55) for start in range(1, 7)]

    def draw(n_rows, limit, *, seed, member_index, labels):
        assert n_rows == 100 and limit == 55 and seed == 42
        rows = masks[member_index]
        assert len(np.unique(labels[rows])) == classes
        return rows

    monkeypatch.setattr(fit, "sample_context_rows", draw)
    model = CausiloClassifier(
        max_context=55, n_estimators=estimators, random_state=42, use_kv_cache=cache,
        device="cpu", checkpoint_path=checkpoint15("classification"), retain_preprocessing=False,
    ).fit(table, index % classes, sample_count=weights)
    state = model._engine.require_state()
    members = state.members if isinstance(state, fit.SampledFitState) else (state,)
    assert len(members) == (1 if classes > 10 else estimators)
    full = members[0].dataset.columns
    assert set(full.order) == {0, 1, 2, 3}
    duplicate_groups = full.groups[np.isin(full.order, (2, 3))]
    assert duplicate_groups[0] == duplicate_groups[1]
    for rows, member in zip(masks[:len(members)], members, strict=True):
        dataset = member.dataset
        assert dataset.columns is full and dataset.training.shape == (55, 4)
        np.testing.assert_array_equal(dataset.frequencies, weights[rows])
        encoder = dataset.encoder
        assert list(encoder.categorical.log_counts) == [0]
        known = np.unique(levels[rows])
        np.testing.assert_array_equal(encoder.categorical.ordinal.categories_[0], known)
        expected_counts = [weights[rows][levels[rows] == level].sum() for level in known]
        np.testing.assert_array_equal(encoder.categorical.log_counts[0], np.log1p(expected_counts))

    query = table.iloc[[0, 48, 75]].copy()
    query.loc[48, "kind"] = np.nan
    encoded = members[0].dataset.encoder.transform(query)
    assert np.isnan(encoded[0, 0]) and encoded[0, 1] == 0
    assert np.isnan(encoded[1, :2]).all()
    assert encoded[2, 1] > 0
    predicted = model.predict_proba(query)
    assert np.isfinite(predicted).all()
    model.save_context(tmp_path / "fitted")
    restored = CausiloClassifier.load_context(tmp_path / "fitted", device="cpu")
    np.testing.assert_array_equal(restored.predict_proba(query), predicted)


def test_sample_with_no_variation_keeps_full_numeric_schema(checkpoint15, monkeypatch):
    table = np.full((13, 3), None, dtype=object)
    table[12, 0] = 2.5
    table[:, 1] = 1
    table[12, 1] = 10
    table[:, 2] = 0
    monkeypatch.setattr(fit, "sample_context_rows", lambda *args, **kwargs: np.arange(6))
    model = CausiloRegressor(
        max_context=6, n_estimators=1, device="cpu", checkpoint_path=checkpoint15("regression"),
    ).fit(table, np.arange(len(table)))
    dataset = model._engine.state.dataset
    assert dataset.encoder.categorical_columns == ()
    assert dataset.training.shape == (6, 2) and set(dataset.columns.order) == {0, 1}
    assert np.isfinite(model.predict(table[11:])).all()
