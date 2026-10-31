"""Frequency counts must behave like expanded observations through the public API."""

import copy
import io
import pickle

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn import config_context
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from causilo import CausiloClassifier, CausiloRegressor, checkpoints
from causilo.execution.direct import BatchPlan, BatchPlanner
from causilo.execution.runner import ModelRunner
from causilo.model import Model, ModelConfig
from causilo.serialization import FREQUENCY_STATE_SCHEMA, STATE_SCHEMA


@pytest.fixture(autouse=True)
def models(monkeypatch):
    """Exercise real model execution with small deterministic checkpoint fixtures."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    models = {}
    with torch.random.fork_rng():
        for task in ("classification", "regression"):
            torch.manual_seed(48)
            model = Model(ModelConfig(
                task=task, width=16, expansion=2, group_size=3, frequencies=4,
                column_latents=5, column_heads=2, column_depths=(2, 2),
                row_heads=2, row_latents=2, row_depths=(3, 2),
                prediction_heads=2, prediction_depth=3, outputs=3 if task == "classification" else 9,
            ))
            for name, parameter in model.named_parameters():
                if "latents" in name or name.endswith("frequencies") or name.endswith("missing"):
                    torch.nn.init.normal_(parameter, std=0.1)
            models[task] = model.eval()
    monkeypatch.setattr(checkpoints, "load_pretrained_model", lambda task: copy.deepcopy(models[task]))
    yield
    torch.set_num_threads(previous)


def data(task="classification"):
    rng = np.random.default_rng(124)
    table = pd.DataFrame(rng.normal(size=(14, 5)), columns=list("abcde"))
    table.loc[2, "a"] = np.nan
    table["kind"] = ["a", "b", None, "b", "c", "a", "a"] * 2
    targets = np.arange(len(table)) % 3 if task == "classification" else rng.normal(size=len(table)) * 3 + 7
    weights = np.array([1, 3, 2, 0, 5, 1, 2, 4, 1, 3, 1, 2, 1, 3])
    query = table.iloc[[2, 5, 8, 10]].copy()
    query.loc[5, "kind"] = "unseen"
    return table, targets, weights, query


def predict(model, query):
    if isinstance(model, CausiloClassifier):
        return model.predict_proba(query)
    return model.predict(query, output_type="raw")


@pytest.mark.parametrize("kind", [CausiloClassifier, CausiloRegressor])
@pytest.mark.parametrize("members", [1, 4, 8])
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("retain", [False, True])
def test_public_counts_match_expanded_rows(kind, members, cache, retain):
    task = "classification" if kind is CausiloClassifier else "regression"
    table, targets, weights, query = data(task)
    indices = np.repeat(np.arange(len(table)), weights)
    options = dict(n_estimators=members, device="cpu", use_kv_cache=cache, retain_preprocessing=retain)
    expanded = kind(**options).fit(table.iloc[indices], targets[indices])
    weighted = kind(**options).fit(table, targets, sample_weight=weights)
    np.testing.assert_allclose(predict(weighted, query), predict(expanded, query), atol=3e-5, rtol=3e-5)
    state = weighted._engine.state
    assert len(state.dataset.features) == np.count_nonzero(weights)
    assert bool(state.dataset.normalized_cache) == retain
    np.testing.assert_array_equal(state.dataset.frequencies, weights[weights > 0])
    if cache:
        assert state.caches[0].prediction.layers[0].key.shape[-2] == np.count_nonzero(weights)


@pytest.mark.parametrize("kind", [CausiloClassifier, CausiloRegressor])
@pytest.mark.parametrize("cache", [False, True])
def test_omitted_and_unit_counts_preserve_original_path(kind, cache):
    table, targets, _, query = data()
    model = kind(n_estimators=4, device="cpu", use_kv_cache=cache).fit(table, targets)
    expected = predict(model, query)
    parameters = model.get_params()
    assert "sample_weight" not in parameters
    assert clone(model).get_params() == parameters
    assert model.__getstate__()["schema"] == STATE_SCHEMA
    model.fit(table, targets, sample_weight=np.ones(len(table)))
    assert model._engine.state.dataset.frequencies is None
    np.testing.assert_array_equal(predict(model, query), expected)


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("serializer", [pickle, joblib], ids=["pickle", "joblib"])
@pytest.mark.parametrize("task", ["classification", "regression", "ecoc"])
def test_saved_counts_restore_without_refitting_or_rebuilding(task, cache, serializer, monkeypatch):
    table, targets, weights, query = data(task)
    kind = CausiloRegressor if task == "regression" else CausiloClassifier
    if task == "ecoc":
        targets = np.arange(len(table)) % 5
    model = kind(n_estimators=4, device="cpu", use_kv_cache=cache).fit(table, targets, weights)
    expected = predict(model, query)
    assert model.__getstate__()["schema"] == FREQUENCY_STATE_SCHEMA
    stream = io.BytesIO()
    serializer.dump(model, stream)
    stream.seek(0)
    monkeypatch.setattr(ModelRunner, "build_cache", lambda *a, **kw: pytest.fail("Rebuilt saved cache"))
    restored = serializer.load(stream)
    np.testing.assert_array_equal(predict(restored, query), expected)
    np.testing.assert_array_equal(restored._engine.state.dataset.frequencies, weights[weights > 0])
    if task == "ecoc":
        assert all(d.frequencies is restored._engine.state.dataset.frequencies
                   for d in restored._engine.state.code_datasets)


@pytest.mark.parametrize("cache", [False, True])
def test_ecoc_counts_match_expansion(cache):
    table, _, weights, query = data()
    targets = np.arange(len(table)) % 5
    rows = np.repeat(np.arange(len(table)), weights)
    options = dict(n_estimators=4, device="cpu", use_kv_cache=cache)
    expanded = CausiloClassifier(**options).fit(table.iloc[rows], targets[rows])
    weighted = CausiloClassifier(**options).fit(table, targets, weights)
    np.testing.assert_allclose(weighted.predict_proba(query), expanded.predict_proba(query), atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("mode", ["direct", "recompute"])
@pytest.mark.parametrize("kind", [CausiloClassifier, CausiloRegressor])
def test_weighted_partitioning_and_recompute(kind, mode, monkeypatch):
    table, targets, weights, query = data()
    model = kind(n_estimators=4, device="cpu").fit(table, targets, weights)
    expected = predict(model, query)
    monkeypatch.setattr(BatchPlanner, "choose", lambda *a: BatchPlan(1, 2, mode=mode))
    np.testing.assert_allclose(predict(model, query), expected, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("cache", [False, True])
def test_weighted_oom_retry_keeps_counts(cache, monkeypatch):
    table, targets, weights, query = data()
    model = CausiloClassifier(n_estimators=4, device="cpu", use_kv_cache=cache).fit(table, targets, weights)
    expected = model.predict_proba(query)
    method = "predict_cached" if cache else "predict"
    original = getattr(ModelRunner, method)
    attempts = []

    def fail_once(runner, *args, **kwargs):
        attempts.append(kwargs["frequencies"])
        if len(attempts) == 1:
            raise torch.cuda.OutOfMemoryError("Simulated frequency prediction OOM")
        return original(runner, *args, **kwargs)

    monkeypatch.setattr(ModelRunner, method, fail_once)
    np.testing.assert_allclose(model.predict_proba(query), expected, atol=3e-5, rtol=3e-5)
    assert len(attempts) > 1
    assert all(values is attempts[0] for values in attempts[1:])
    assert attempts[0].count == int(weights.sum())
    torch.testing.assert_close(attempts[0].log_bias.flatten(), torch.tensor(weights[weights > 0]).float().log())


def test_zero_count_rows_do_not_affect_schema_classes_or_validation():
    table, targets, weights, query = data()
    targets = targets.astype(float)
    table.loc[3, "a"] = np.inf
    table.loc[3, "kind"] = "zero-only"
    targets[3] = np.nan
    weighted = CausiloClassifier(n_estimators=4, device="cpu").fit(table, targets, weights)
    dataset = weighted._engine.state.dataset
    assert "zero-only" not in dataset.encoder.encoder.categories_[0]
    assert np.isfinite(weighted.predict_proba(query)).all()
    weights[:] = 0
    assert (dataset.frequencies > 0).all(), "Caller-owned weights must not alias fitted state"


@pytest.mark.parametrize("counts", [[0] * 14, [-1] * 14, [1.5] * 14, [np.inf] * 14,
                                    [np.nan] * 14, [1] * 13, [[1]] * 14, [True] * 14])
def test_invalid_counts_clear_previous_fit(counts):
    table, targets, _, query = data()
    model = CausiloClassifier(n_estimators=1, device="cpu").fit(table, targets)
    with pytest.raises(ValueError, match="sample_weight|frequenc|count"):
        model.fit(table, targets, counts)
    with pytest.raises(NotFittedError):
        model.predict(query)
    assert not hasattr(model, "classes_")


def test_single_repeated_row_matches_expanded_constant_feature_rejection():
    for table, targets, weights in [([[1, 2]], [0], [3]), ([[1, 2]] * 3, [0] * 3, None)]:
        with pytest.raises(ValueError, match="varying feature"):
            CausiloClassifier(device="cpu").fit(table, targets, weights)


def test_legacy_unweighted_saved_dataset_without_frequency_field():
    table, targets, _, query = data()
    model = CausiloClassifier(n_estimators=1, device="cpu", use_kv_cache=True).fit(table, targets)
    expected = model.predict_proba(query)
    saved = model.__getstate__()
    del saved["fitted"].dataset.__dict__["frequencies"]
    restored = CausiloClassifier()
    restored.__setstate__(saved)
    np.testing.assert_array_equal(restored.predict_proba(query), expected)


@pytest.mark.parametrize("corruption", ["missing", "zero", "length", "schema"])
def test_invalid_saved_counts_are_rejected_before_checkpoint_loading(corruption, monkeypatch):
    table, targets, weights, _ = data()
    model = CausiloClassifier(n_estimators=1, device="cpu").fit(table, targets, weights)
    saved = model.__getstate__()
    if corruption == "missing":
        saved["fitted"].dataset.frequencies = None
    elif corruption == "zero":
        saved["fitted"].dataset.frequencies = np.zeros(np.count_nonzero(weights), dtype=int)
    elif corruption == "length":
        saved["fitted"].dataset.frequencies = np.ones(2, dtype=int)
    else:
        saved["schema"] = STATE_SCHEMA
    monkeypatch.setattr(checkpoints, "load_pretrained_model", lambda task: pytest.fail("Loaded invalid state"))
    with pytest.raises(ValueError):
        CausiloClassifier().__setstate__(saved)


def test_sklearn_pipeline_routes_optional_fit_counts():
    table, targets, weights, query = data()
    with config_context(enable_metadata_routing=True):
        estimator = CausiloClassifier(n_estimators=1, device="cpu").set_fit_request(sample_weight=True)
        pipeline = Pipeline([("model", estimator)]).fit(table, targets, sample_weight=weights)
    assert np.isfinite(pipeline.predict_proba(query)).all()
    np.testing.assert_array_equal(estimator._engine.state.dataset.frequencies, weights[weights > 0])
