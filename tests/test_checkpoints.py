import hashlib
import io
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from causilo import CausiloClassifier, CausiloRegressor, checkpoints
from causilo.execution.runner import ModelRunner
from causilo.model import Model, ModelConfig


@pytest.fixture(autouse=True)
def offline_runtime(monkeypatch):
    monkeypatch.setattr(checkpoints, "snapshot_download", lambda *a, **k: pytest.fail("Contacted the Hub"))
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(params=[CausiloClassifier, CausiloRegressor])
def constructor(request):
    return request.param


@pytest.fixture
def task(constructor):
    return "classification" if constructor is CausiloClassifier else "regression"


@pytest.fixture
def checkpoint_factory(tmp_path):
    def create(task, name="weights", seed=18):
        folder = tmp_path / name
        folder.mkdir(exist_ok=True)
        config = ModelConfig(
            task=task, width=8, expansion=2, group_size=2, frequencies=2,
            column_latents=3, column_heads=2, column_depths=(1, 1),
            row_heads=2, row_latents=2, row_depths=(1, 1),
            prediction_heads=2, prediction_depth=1, outputs=3 if task == "classification" else 9,
        )
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            model = Model(config)
            for parameter in model.parameters():
                torch.nn.init.normal_(parameter, std=0.1)
        encoded = json.dumps(config.record()).encode()
        (folder / "config.json").write_bytes(encoded)
        save_file(model.state_dict(), folder / "model.safetensors", metadata={
            "architecture": "causilo-v1.0", "format_version": "1", "task": task,
            "config_sha256": hashlib.sha256(encoded).hexdigest(),
        })
        return folder / "model.safetensors"
    return create


@pytest.fixture
def data():
    return np.random.default_rng(42).normal(size=(20, 4)), np.arange(20) % 3


def predictions(estimator, table):
    if isinstance(estimator, CausiloClassifier):
        return estimator.predict_proba(table)
    return estimator.predict(table)


@pytest.mark.parametrize("path", [None, "missing.safetensors", Path("missing")])
def test_construction_clone_and_unfitted_restore_do_not_read_files(constructor, path, monkeypatch):
    monkeypatch.setattr(checkpoints, "load_model", lambda *a, **k: pytest.fail("Loaded weights"))
    estimator = constructor(checkpoint_path=path)
    assert estimator.checkpoint_path is path
    assert clone(estimator).get_params() == estimator.get_params()
    assert clone(Pipeline([("model", estimator)])).get_params()["model__checkpoint_path"] == path
    assert pickle.loads(pickle.dumps(estimator)).get_params() == estimator.get_params()
    if path is None:
        assert repr(estimator) == f"{constructor.__name__}()"


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("as_directory", [False, True])
@pytest.mark.parametrize("serializer", [pickle, joblib], ids=["pickle", "joblib"])
def test_custom_checkpoint_roundtrip(
    constructor, task, checkpoint_factory, data, cache, as_directory, serializer, monkeypatch,
):
    weights = checkpoint_factory(task)
    path = weights.parent if as_directory else str(weights)
    table, target = data
    estimator = constructor(n_estimators=1, device="cpu", use_kv_cache=cache, checkpoint_path=path)
    estimator.fit(table, target)
    expected = predictions(estimator, table[:3])
    logits = estimator.predict_logits(table[:3]) if task == "classification" else None
    monkeypatch.setattr(ModelRunner, "build_cache", lambda *a, **k: pytest.fail("Rebuilt saved cache"))
    payload = io.BytesIO()
    serializer.dump(estimator, payload)
    payload.seek(0)
    restored = serializer.load(payload)
    assert restored.checkpoint_path == path
    np.testing.assert_array_equal(predictions(restored, table[:3]), expected)
    if logits is not None:
        np.testing.assert_array_equal(restored.predict_logits(table[:3]), logits)
    if cache:
        assert restored._engine.state.caches is not None


@pytest.mark.parametrize("cache", [False, True])
def test_logits_respect_custom_checkpoint_capacity(checkpoint_factory, data, cache, monkeypatch):
    weights = checkpoint_factory("classification")  # three native outputs
    table, target = data
    model = CausiloClassifier(
        n_estimators=2, device="cpu", use_kv_cache=cache, checkpoint_path=weights,
    ).fit(table, target)
    assert model.predict_logits(table[:2]).shape == (2, 3)
    model.fit(table, np.arange(len(table)) % 4)
    probabilities = model.predict_proba(table[:2])
    assert probabilities.shape == (2, 4)
    monkeypatch.setattr(model._engine.state.dataset, "query_tables", lambda *a: pytest.fail("Processed query"))
    with pytest.raises(NotImplementedError, match="at most 3 classes.*ECOC"):
        model.predict_logits(table[:2])


def test_default_route_preserves_legacy_saved_state(constructor, task, checkpoint_factory, data, monkeypatch):
    weights = checkpoint_factory(task)
    calls = []

    def official(requested_task):
        calls.append(requested_task)
        return checkpoints.load_checkpoint(weights)

    monkeypatch.setattr(checkpoints, "load_pretrained_model", official)
    monkeypatch.setattr(checkpoints, "checkpoint_digest", lambda *a: pytest.fail("Hashed default weights"))
    table, target = data
    estimator = constructor(n_estimators=1, device="cpu").fit(table, target)
    saved = estimator.__getstate__()
    saved.pop("checkpoint")
    saved["parameters"].pop("checkpoint_path")
    restored = constructor.__new__(constructor)
    restored.__setstate__(saved)
    assert restored.get_params()["checkpoint_path"] is None
    np.testing.assert_array_equal(predictions(restored, table[:3]), predictions(estimator, table[:3]))
    assert calls == [task, task]


@pytest.mark.parametrize("cache", [False, True])
def test_batched_local_checkpoint_identity_and_refit(
    constructor, task, checkpoint_factory, data, cache, monkeypatch,
):
    first = checkpoint_factory(task, "first")
    second = checkpoint_factory(task, "second", 72)
    x, y = data
    tables = {"a": (x, y), "b": (x[:15, :3], y[:15])}
    queries = {"b": x[:2, :3], "a": x[:3]}
    estimator = constructor(n_estimators=2, device="cpu", use_kv_cache=cache, checkpoint_path=first)
    estimator.fit_batched(tables)
    method = "predict_proba_batched" if task == "classification" else "predict_batched"
    expected = getattr(estimator, method)(queries)
    estimator.set_params(checkpoint_path=second)
    with monkeypatch.context() as patch:
        patch.setattr(ModelRunner, "build_cache", lambda *a, **k: pytest.fail("Rebuilt saved cache"))
        restored = pickle.loads(pickle.dumps(estimator))
    assert restored.checkpoint_path == second
    assert restored._engine.checkpoint["path"] == str(first)
    actual = getattr(restored, method)(queries)
    for key in queries:
        np.testing.assert_array_equal(actual[key], expected[key])
    restored.fit_batched(tables)
    assert restored._engine.checkpoint["path"] == str(second)
    payload = pickle.dumps(restored)
    checkpoint_factory(task, "second", 91)
    with pytest.raises(ValueError, match="differs from the weights used to fit"):
        pickle.loads(payload)


def test_legacy_unfitted_state_supplies_new_default(constructor):
    saved = constructor().__getstate__()
    saved.pop("checkpoint")
    saved["parameters"].pop("checkpoint_path")
    restored = constructor.__new__(constructor)
    restored.__setstate__(saved)
    assert clone(restored).get_params() == constructor().get_params()


def test_local_state_without_identity_is_not_loaded_from_hub(constructor, task, checkpoint_factory, data):
    weights = checkpoint_factory(task)
    estimator = constructor(n_estimators=1, device="cpu", checkpoint_path=weights).fit(*data)
    saved = estimator.__getstate__()
    saved.pop("checkpoint")
    restored = constructor.__new__(constructor)
    with pytest.raises(ValueError, match="missing its fitted weight identity"):
        restored.__setstate__(saved)


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("changed_file", ["weights", "config"])
def test_restore_rejects_changed_contents_before_loading(
    constructor, task, checkpoint_factory, data, cache, changed_file, monkeypatch,
):
    weights = checkpoint_factory(task)
    table, target = data
    estimator = constructor(n_estimators=1, device="cpu", use_kv_cache=cache, checkpoint_path=weights)
    estimator.fit(table, target)
    payload = pickle.dumps(estimator)
    if changed_file == "weights":
        checkpoint_factory(task, seed=72)
    else:
        config = weights.with_name("config.json")
        config.write_bytes(config.read_bytes() + b"\n")
    monkeypatch.setattr(checkpoints, "load_checkpoint", lambda *a: pytest.fail("Loaded changed weights"))
    with pytest.raises(ValueError, match="differs from the weights used to fit"):
        pickle.loads(payload)


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("initial, pending", [("a", "b"), ("a", None), (None, "b")])
def test_pending_source_applies_only_on_refit(
    constructor, task, checkpoint_factory, data, cache, initial, pending, monkeypatch,
):
    paths = {None: None, "a": checkpoint_factory(task, "a", 18), "b": checkpoint_factory(task, "b", 72)}
    official = checkpoint_factory(task, "official", 91)
    monkeypatch.setattr(checkpoints, "load_pretrained_model", lambda task: checkpoints.load_checkpoint(official))
    table, target = data
    estimator = constructor(n_estimators=1, device="cpu", use_kv_cache=cache, checkpoint_path=paths[initial])
    estimator.fit(table, target)
    expected = predictions(estimator, table[:3])
    estimator.set_params(checkpoint_path=paths[pending])
    restored = pickle.loads(pickle.dumps(estimator))
    assert restored.checkpoint_path == paths[pending]
    np.testing.assert_array_equal(predictions(restored, table[:3]), expected)
    restored.fit(table, target)
    reference = clone(restored).fit(table, target)
    np.testing.assert_array_equal(predictions(restored, table[:3]), predictions(reference, table[:3]))
    if pending is None:
        assert restored._engine.checkpoint is None
    else:
        assert restored._engine.checkpoint["path"] == str(paths[pending])


def test_relative_fitted_source_survives_working_directory_change(
    constructor, task, checkpoint_factory, data, tmp_path, monkeypatch,
):
    weights = checkpoint_factory(task)
    monkeypatch.chdir(tmp_path)
    table, target = data
    estimator = constructor(n_estimators=1, device="cpu", checkpoint_path=Path("weights")).fit(table, target)
    assert estimator.get_params()["checkpoint_path"] == Path("weights")
    assert estimator._engine.checkpoint["path"] == str(weights)
    payload = pickle.dumps(estimator)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    restored = pickle.loads(payload)
    np.testing.assert_array_equal(predictions(restored, table[:3]), predictions(estimator, table[:3]))


def test_missing_weights(constructor, tmp_path, data):
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        constructor(device="cpu", checkpoint_path=tmp_path / "missing.safetensors").fit(*data)


def test_missing_configuration(constructor, task, checkpoint_factory, data):
    weights = checkpoint_factory(task)
    weights.with_name("config.json").unlink()
    with pytest.raises(FileNotFoundError, match="configuration not found"):
        constructor(device="cpu", checkpoint_path=weights).fit(*data)


def test_wrong_task(constructor, task, checkpoint_factory, data):
    wrong_task = "regression" if task == "classification" else "classification"
    weights = checkpoint_factory(wrong_task)
    with pytest.raises(ValueError, match=f"for {wrong_task}, not {task}"):
        constructor(device="cpu", checkpoint_path=weights).fit(*data)


def test_invalid_path_type(constructor, data):
    with pytest.raises(ValueError, match="checkpoint_path must be"):
        constructor(device="cpu", checkpoint_path=3).fit(*data)


def test_config_validation_is_preserved(constructor, task, checkpoint_factory, data):
    weights = checkpoint_factory(task)
    config = weights.with_name("config.json")
    config.write_bytes(config.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Weights do not match"):
        constructor(device="cpu", checkpoint_path=weights).fit(*data)


def test_failed_source_change_discards_old_fit(constructor, task, checkpoint_factory, data):
    weights = checkpoint_factory(task)
    estimator = constructor(n_estimators=1, device="cpu", checkpoint_path=weights).fit(*data)
    estimator.set_params(checkpoint_path=weights.parent / "missing.safetensors")
    with pytest.raises(FileNotFoundError):
        estimator.fit(*data)
    with pytest.raises(NotFittedError):
        estimator.predict(data[0][:3])
