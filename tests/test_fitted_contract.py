import os
import pickle
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.exceptions import NotFittedError

from causilo import CausiloClassifier, CausiloRegressor, checkpoints
from causilo.execution.runner import ModelRunner

pytestmark = pytest.mark.pretrained


@pytest.fixture(autouse=True)
def limit_threads():
    torch.set_num_threads(2)


@pytest.mark.parametrize("estimator_type", [CausiloClassifier, CausiloRegressor])
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("retain", [False, True])
def test_saved_fit_reuses_state(estimator_type, cache, retain, monkeypatch):
    rng = np.random.default_rng(91)
    table = rng.normal(size=(40, 5))
    target = np.arange(40) % 3
    model = estimator_type(n_estimators=1, device="cpu", use_kv_cache=cache, retain_preprocessing=retain)
    model.fit(table, target)
    expected = model.predict(table[:4])
    payload = pickle.dumps(model)
    assert len(payload) < 10_000_000
    monkeypatch.setattr(ModelRunner, "build_cache", lambda *args: pytest.fail("Restore must not rebuild K/V"))
    restored = pickle.loads(payload)
    np.testing.assert_array_equal(expected, restored.predict(table[:4]))
    assert bool(restored._engine.state.dataset.normalized_cache) == retain
    with pytest.raises(ValueError):
        restored.fit(table, target[:-1])
    with pytest.raises(NotFittedError):
        restored.predict(table[:4])


def test_missing_fit():
    estimator = CausiloClassifier(n_estimators=1, device="cpu")
    with pytest.raises(NotFittedError):
        estimator.predict(np.ones((2, 3)))


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("classes", [11, 201])
def test_many_class_fit_prediction_and_restore(cache, classes, monkeypatch):
    table = np.random.default_rng(42).normal(size=(max(40, classes), 3))
    labels = np.asarray([f"class-{index % classes}" for index in range(len(table))])
    model = CausiloClassifier(n_estimators=2, device="cpu", use_kv_cache=cache, random_state=42).fit(
        table, labels
    )
    query = table[:3]
    probabilities = model.predict_proba(query)
    assert probabilities.shape == (len(query), classes)
    assert np.isfinite(probabilities).all() and np.all(probabilities >= 0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_array_equal(model.predict(query), model.classes_[probabilities.argmax(axis=1)])
    assert all(len(dataset.members) == 2 for dataset in model._engine.state.code_datasets)
    monkeypatch.setattr(ModelRunner, "build_cache", lambda *args: pytest.fail("Rebuilt stored K/V"))
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.classes_, model.classes_)
    np.testing.assert_array_equal(restored.predict_proba(query), probabilities)


def test_incomplete_saved_ecoc_cache_is_rejected():
    from dataclasses import replace

    x = np.random.default_rng(12).normal(size=(44, 4))
    model = CausiloClassifier(n_estimators=1, device="cpu", use_kv_cache=True).fit(x, np.arange(44) % 11)
    saved = model.__getstate__()
    saved["fitted"] = replace(saved["fitted"], code_caches=saved["fitted"].code_caches[:-1])
    with pytest.raises(ValueError, match="code row count"):
        CausiloClassifier().__setstate__(saved)


def test_failed_many_class_cache_fit_discards_context(monkeypatch):
    x = np.random.default_rng(12).normal(size=(44, 4))
    model = CausiloClassifier(n_estimators=1, device="cpu", use_kv_cache=True).fit(x, np.arange(44) % 3)
    monkeypatch.setattr(
        ModelRunner, "build_cache", lambda *args: (_ for _ in ()).throw(RuntimeError("cache"))
    )
    with pytest.raises(RuntimeError, match="cache"):
        model.fit(x, np.arange(44) % 11)
    with pytest.raises(NotFittedError):
        model.predict_proba(x[:2])


def test_failed_version_check():
    estimator = CausiloRegressor()
    saved = estimator.__getstate__()
    saved["version"] = "incompatible"
    with pytest.raises(ValueError, match="same Causilo version"):
        estimator.__setstate__(saved)


@pytest.mark.parametrize("estimator_type", [CausiloClassifier, CausiloRegressor])
def test_invalid_device_refit_discards_previous_context(estimator_type):
    model = estimator_type(n_estimators=1, device="cpu")
    table = np.arange(60).reshape(20, 3)
    target = np.arange(20) % 2
    model.fit(table, target)
    model.set_params(device="mps")
    with pytest.raises(ValueError, match="CPU or a single CUDA"):
        model.fit(table, target)
    with pytest.raises(NotFittedError):
        model.predict(table[:2])
    assert model._engine.state is None
    assert not hasattr(model, "n_features_in_")


@pytest.mark.parametrize("estimator_type", [CausiloClassifier, CausiloRegressor])
def test_ensemble_and_query_partition_preserve_predictions(estimator_type, monkeypatch):
    from causilo.execution.direct import BatchPlan, BatchPlanner

    rng = np.random.default_rng(63)
    table = rng.normal(size=(32, 5))
    target = np.arange(32) % 3
    model = estimator_type(n_estimators=4, device="cpu").fit(table, target)
    predict = model.predict_proba if estimator_type is CausiloClassifier else model.predict
    expected = predict(table[:7])
    monkeypatch.setattr(BatchPlanner, "choose", lambda *args: BatchPlan(1, 2))
    np.testing.assert_allclose(predict(table[:7]), expected, atol=2e-4, rtol=2e-4)


def test_prediction_oom_reselects_ensemble_and_preserves_result(monkeypatch):
    rng = np.random.default_rng(71)
    table = rng.normal(size=(24, 4))
    model = CausiloRegressor(n_estimators=4, device="cpu").fit(table, np.arange(24) % 3)
    expected = model.predict(table[:3])
    original = ModelRunner.predict
    attempts = []

    def fail_once(flow, features, targets):
        attempts.append(len(features))
        if len(attempts) == 1:
            raise torch.cuda.OutOfMemoryError("Simulated allocation failure")
        return original(flow, features, targets)

    monkeypatch.setattr(ModelRunner, "predict", fail_once)
    np.testing.assert_allclose(model.predict(table[:3]), expected, atol=2e-4, rtol=2e-4)
    assert attempts[1] < attempts[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA cache portability")
@pytest.mark.parametrize("estimator_type", [CausiloClassifier, CausiloRegressor])
def test_auto_device_cache_restores_on_cpu_without_rebuilding(estimator_type, monkeypatch):
    table = np.random.default_rng(65).normal(size=(30, 5))
    target = np.arange(30) % 3
    model = estimator_type(n_estimators=1, device="auto", use_kv_cache=True).fit(table, target)
    expected = model.predict(table[:4])
    payload = pickle.dumps(model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        ModelRunner, "build_cache", lambda *args: pytest.fail("Restoration rebuilt the cache")
    )
    restored = pickle.loads(payload)
    assert restored._engine.device.type == "cpu"
    for context in restored._engine.state.caches:
        for component in (context.columns[0], context.columns[1], context.prediction):
            assert all(layer.key.dtype == torch.float32 for layer in component.layers)
    actual = restored.predict(table[:4])
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=0.03, rtol=0.03)
    assert model._engine.device.type == "cuda"


@pytest.mark.parametrize("estimator_type", [CausiloClassifier, CausiloRegressor])
def test_partitioned_cache_keeps_feature_group_alignment(estimator_type, monkeypatch):
    from causilo.execution import chunks

    table = np.random.default_rng(52).normal(size=(12, 7))
    target = np.arange(12) % 3
    first = estimator_type(n_estimators=1, device="cpu", use_kv_cache=True).fit(table, target)
    expected = first.predict(table[:3])
    monkeypatch.setattr(chunks, "execution_budget", lambda device: 1)
    second = estimator_type(n_estimators=1, device="cpu", use_kv_cache=True).fit(table, target)
    np.testing.assert_allclose(second.predict(table[:3]), expected, atol=2e-4, rtol=2e-4)
    left = first._engine.state.caches[0]
    right = second._engine.state.caches[0]
    for before, after in ((left.columns[0], right.columns[0]), (left.columns[1], right.columns[1])):
        for original, partitioned in zip(before.layers, after.layers):
            torch.testing.assert_close(original.key, partitioned.key, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(original.value, partitioned.value, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("estimator_type", [CausiloClassifier, CausiloRegressor])
def test_joblib_restoration_in_fresh_process(estimator_type, tmp_path):
    table = pd.DataFrame({"category": ["red", "blue", None] * 8, "amount": np.arange(24, dtype=float)})
    query = pd.DataFrame({"category": ["unseen", None, "blue"], "amount": [3.0, np.nan, 7.0]})
    model = estimator_type(n_estimators=1, device="cpu", use_kv_cache=True).fit(table, np.arange(24) % 3)
    expected = model.predict(query)
    saved, output = tmp_path / "fitted.joblib", tmp_path / "prediction.npy"
    joblib.dump((model, query), saved)
    assert saved.stat().st_size < 10_000_000
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}
    subprocess.run(
        [sys.executable, str(root / "tests/helpers/restore_probe.py"), str(saved), str(output)],
        env=environment,
        check=True,
        capture_output=True,
        timeout=60,
    )
    np.testing.assert_array_equal(np.load(output), expected)


def test_incomplete_saved_cache_is_rejected():
    from dataclasses import replace

    table = np.random.default_rng(7).normal(size=(12, 4))
    model = CausiloRegressor(n_estimators=1, device="cpu", use_kv_cache=True).fit(table, np.arange(12))
    saved = model.__getstate__()
    context = saved["fitted"].caches[0]
    incomplete = replace(context.columns[0], layers=context.columns[0].layers[:-1])
    saved["fitted"].caches = (replace(context, columns=(incomplete, context.columns[1])),)
    with pytest.raises(ValueError, match="cache layout"):
        CausiloRegressor().__setstate__(saved)


def test_cached_query_oom_preserves_all_rows(monkeypatch):
    table = np.random.default_rng(21).normal(size=(15, 4))
    model = CausiloRegressor(n_estimators=1, device="cpu", use_kv_cache=True).fit(table, np.arange(15) % 3)
    expected = model.predict(table[:9])
    original = ModelRunner.predict_cached
    attempts = []

    def fail_once(flow, rows, context):
        attempts.append(rows.shape[1])
        if len(attempts) == 1:
            raise torch.cuda.OutOfMemoryError("Simulated cached query allocation failure")
        return original(flow, rows, context)

    monkeypatch.setattr(ModelRunner, "predict_cached", fail_once)
    np.testing.assert_allclose(model.predict(table[:9]), expected, atol=2e-4, rtol=2e-4)
    assert attempts[1] < attempts[0]
    assert sum(attempts[1:]) == 9


@pytest.mark.parametrize("estimator_type", [CausiloClassifier, CausiloRegressor])
def test_failed_cache_refit_does_not_expose_previous_fit(estimator_type, monkeypatch):
    values = np.random.default_rng(81).normal(size=(18, 4))
    model = estimator_type(device="cpu", n_estimators=1, use_kv_cache=True)
    model.fit(values, np.arange(18) % 3)

    def insufficient_memory(*args):
        raise torch.cuda.OutOfMemoryError("Resident cache cannot fit")

    monkeypatch.setattr(ModelRunner, "build_cache", insufficient_memory)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="Resident cache"):
        model.fit(values, np.arange(18) % 2)
    assert model._engine.state is None
    assert model.use_kv_cache and model.device == "cpu"
    with pytest.raises(NotFittedError):
        model.predict(values[:2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CPU to CUDA cache restoration")
@pytest.mark.parametrize("estimator_type", [CausiloClassifier, CausiloRegressor])
def test_cpu_cache_moves_to_automatic_cuda_without_rebuilding(estimator_type, monkeypatch):
    values = pd.DataFrame({"kind": ["a", None, "b"] * 10, "value": np.linspace(-2, 2, 30)})
    queries = pd.DataFrame({"kind": ["c", "a", None], "value": [np.nan, 0.5, -1.0]})
    with monkeypatch.context() as cpu_environment:
        cpu_environment.setattr(torch.cuda, "is_available", lambda: False)
        model = estimator_type(device="auto", n_estimators=2, use_kv_cache=True)
        model.fit(values, np.arange(30) % 3)
        prediction = "predict_proba" if estimator_type is CausiloClassifier else "predict"
        expected = getattr(model, prediction)(queries)
        payload = pickle.dumps(model)

    def unexpected_rebuild(*args):
        pytest.fail("Restoration must use the saved cache")

    monkeypatch.setattr(ModelRunner, "build_cache", unexpected_rebuild)
    restored = pickle.loads(payload)
    assert restored._engine.device.type == "cuda"
    assert model._engine.device.type == "cpu"
    for before, after in zip(model._engine.state.caches, restored._engine.state.caches):
        for old_block, new_block in zip(
            (*before.columns, before.prediction), (*after.columns, after.prediction)
        ):
            for old, new in zip(old_block.layers, new_block.layers):
                for field in ("key", "value"):
                    tensor = getattr(new, field)
                    assert tensor.device.type == "cuda"
                    torch.testing.assert_close(
                        tensor.cpu(), getattr(old, field).to(tensor.dtype), rtol=0, atol=0
                    )
    actual = getattr(restored, prediction)(queries)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=0.03, rtol=0.03)


@pytest.mark.parametrize("dependency", ["python", "numpy", "pandas", "scipy", "scikit-learn", "torch"])
def test_restore_rejects_dependency_mismatch_before_loading_weights(dependency, monkeypatch):
    model = CausiloClassifier(device="cpu", n_estimators=1).fit(
        np.arange(24).reshape(12, 2), np.arange(12) % 2
    )
    saved = model.__getstate__()
    saved["dependencies"][dependency] = "incompatible"
    monkeypatch.setattr(
        checkpoints, "load_pretrained_model", lambda task: pytest.fail("Loaded incompatible fitted state")
    )
    with pytest.raises(ValueError, match="dependency versions"):
        CausiloClassifier().__setstate__(saved)
