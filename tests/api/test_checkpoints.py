import io
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from causilo import CausiloClassifier, CausiloRegressor
from causilo.model_selection import LEGACY_FORMAT, RUNTIME15_FORMAT, read_local_checkpoint, select_checkpoint
from causilo.v1_0 import checkpoints
from causilo.v1_0.execution.runner import ModelRunner
from causilo.v1_5 import checkpoints as checkpoints15


@pytest.fixture(autouse=True)
def offline_runtime(monkeypatch, batch_runtime):
    monkeypatch.setattr(checkpoints, "snapshot_download", lambda *a, **k: pytest.fail("Contacted the Hub"))



@pytest.fixture(params=[CausiloClassifier, CausiloRegressor])
def constructor(request):
    return request.param


@pytest.fixture
def task(constructor):
    return "classification" if constructor is CausiloClassifier else "regression"


@pytest.fixture
def data():
    return np.random.default_rng(42).normal(size=(20, 4)), np.arange(20) % 3


def predictions(estimator, table):
    if isinstance(estimator, CausiloClassifier):
        return estimator.predict_proba(table)
    return estimator.predict(table)


def test_construction_and_clone_do_not_load_checkpoints(constructor, monkeypatch):
    path = Path("missing")
    monkeypatch.setattr(checkpoints, "load_model", lambda *a, **k: pytest.fail("Loaded weights"))
    monkeypatch.setattr(checkpoints15, "snapshot_download", lambda *a, **k: pytest.fail("Contacted the Hub"))
    default = constructor()
    light = clone(constructor(mode="light"))
    assert repr(default) == f"{constructor.__name__}()"
    assert (default.model_version, default.mode, default.n_estimators) == ("1.5", None, None)
    assert (light.model_version, light.mode, light.n_estimators) == ("1.5", "light", None)
    assert clone(constructor(1)).n_estimators == 1
    estimator = constructor(checkpoint_path=path)
    assert estimator.checkpoint_path is path
    assert clone(estimator).get_params() == estimator.get_params()
    assert clone(Pipeline([("model", estimator)])).get_params()["model__checkpoint_path"] == path
    assert pickle.loads(pickle.dumps(estimator)).get_params() == estimator.get_params()


@pytest.mark.parametrize("version", ["1.0", "1.5"])
def test_ensemble_modes_match_explicit_counts(constructor, task, version, checkpoint_factory, checkpoint15, data):
    checkpoint = checkpoint_factory(task) if version == "1.0" else checkpoint15(task)
    table, target = data
    query = table[:3]
    options = dict(model_version=version, device="cpu", checkpoint_path=checkpoint)

    light = constructor(**options).set_params(mode="light")
    explicit = constructor(1, **options).fit(table, target)
    np.testing.assert_array_equal(predictions(light.fit(table, target), query), predictions(explicit, query))

    explicit = constructor(8, **options).fit(table, target)
    np.testing.assert_array_equal(predictions(constructor(**options).fit(table, target), query), predictions(explicit, query))

    batched_predict = "predict_proba_batched" if task == "classification" else "predict_batched"
    light.fit_batched({"table": (table, target)})
    explicit = constructor(1, **options).fit_batched({"table": (table, target)})
    actual = getattr(light, batched_predict)({"table": query})["table"]
    expected = getattr(explicit, batched_predict)({"table": query})["table"]
    np.testing.assert_array_equal(actual, expected)

    light.set_params(n_estimators=1)
    with pytest.raises(ValueError, match="either mode or n_estimators"):
        light.fit(table, target)
    with pytest.raises(ValueError, match="either mode or n_estimators"):
        light.fit_batched({"table": (table, target)})
    with pytest.raises(ValueError, match="mode must be"):
        light.set_params(mode="fast", n_estimators=None).fit(table, target)


@pytest.mark.parametrize(("cache", "as_directory", "serializer"), [
    (False, False, pickle), (True, True, joblib),
])
def test_custom_checkpoint_roundtrip(
    constructor, task, checkpoint_factory, data, cache, as_directory, serializer, monkeypatch,
):
    weights = checkpoint_factory(task)
    path = weights.parent if as_directory else str(weights)
    table, target = data
    estimator = constructor(model_version="1.0", n_estimators=1, device="cpu", use_kv_cache=cache, checkpoint_path=path)
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
        model_version="1.0",
        n_estimators=2, device="cpu", use_kv_cache=cache, checkpoint_path=weights,
    ).fit(table, target)
    assert model.predict_logits(table[:2]).shape == (2, 3)
    model.fit(table, np.arange(len(table)) % 4)
    probabilities = model.predict_proba(table[:2])
    assert probabilities.shape == (2, 4)
    monkeypatch.setattr(model._engine.state.dataset, "query_tables", lambda *a: pytest.fail("Processed query"))
    with pytest.raises(NotImplementedError, match="at most 3 classes.*ECOC"):
        model.predict_logits(table[:2])


def test_hosted_fit_restores_the_fitted_hub_source(constructor, task, checkpoint_factory, data, monkeypatch):
    weights = checkpoint_factory(task)
    calls = []

    def official(requested_task):
        calls.append(requested_task)
        return checkpoints.load_checkpoint(weights)

    monkeypatch.setattr(checkpoints, "load_hosted_model", official)
    monkeypatch.setattr(checkpoints, "local_checkpoint_identity", lambda *a, **k: pytest.fail("Hashed default weights"))
    table, target = data
    estimator = constructor(model_version="1.0", n_estimators=1, device="cpu").fit(table, target)
    estimator.set_params(model_version="1.5")
    restored = pickle.loads(pickle.dumps(estimator))
    assert restored.model_version == "1.5"
    np.testing.assert_array_equal(predictions(restored, table[:3]), predictions(estimator, table[:3]))
    assert type(restored._engine) is type(estimator._engine)
    assert calls == [task] * 2


@pytest.mark.parametrize("cache", [False, True])
def test_batched_local_checkpoint_identity_and_refit(
    constructor, task, checkpoint_factory, data, cache, monkeypatch,
):
    first = checkpoint_factory(task, "first")
    second = checkpoint_factory(task, "second", 72)
    x, y = data
    tables = {"a": (x, y), "b": (x[:15, :3], y[:15])}
    queries = {"b": x[:2, :3], "a": x[:3]}
    estimator = constructor(model_version="1.0", n_estimators=2, device="cpu", use_kv_cache=cache, checkpoint_path=first)
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


def test_refit_reuses_loaded_checkpoint_after_files_are_removed(constructor, task, checkpoint_factory, data):
    weights = checkpoint_factory(task)
    x, y = data
    original = constructor(model_version="1.0", n_estimators=1, device="cpu", checkpoint_path=weights)
    original.fit(x, y)
    restored = pickle.loads(pickle.dumps(original))

    weights.with_name("config.json").unlink()
    weights.unlink()
    for estimator in (original, restored):
        expected = predictions(estimator, x[:3])
        model = estimator._engine.model
        estimator.fit_batched({"table": (x, y)})
        predict = estimator.predict_proba_batched if task == "classification" else estimator.predict_batched
        np.testing.assert_array_equal(predict({"table": x[:3]})["table"], expected)
        estimator.fit(x, y)
        assert estimator._engine.model is model
        np.testing.assert_array_equal(predictions(estimator, x[:3]), expected)


@pytest.mark.parametrize("next_device", ["cpu:1", "cpu"])
def test_refit_compares_device_with_restored_engine(constructor, task, checkpoint_factory, data, next_device):
    estimator = constructor(
        model_version="1.0", n_estimators=1, device="cpu", checkpoint_path=checkpoint_factory(task),
    ).fit(*data)
    estimator.set_params(device="cpu:1")
    restored = pickle.loads(pickle.dumps(estimator))
    model = restored._engine.model
    assert restored._engine.device == torch.device("cpu:1")

    restored.set_params(device=next_device).fit(*data)
    assert restored._engine.device == torch.device(next_device)
    if next_device == "cpu:1":
        assert restored._engine.model is model
    else:
        assert restored._engine.model is not model


def test_local_state_without_identity_is_not_loaded_from_hub(constructor, task, checkpoint_factory, data):
    weights = checkpoint_factory(task)
    estimator = constructor(model_version="1.0", n_estimators=1, device="cpu", checkpoint_path=weights).fit(*data)
    saved = estimator.__getstate__()
    saved["checkpoint"].pop("digest")
    restored = constructor.__new__(constructor)
    with pytest.raises(ValueError, match="missing its fitted weight identity"):
        restored.__setstate__(saved)


@pytest.mark.parametrize(("cache", "changed_file"), [(False, "config"), (True, "weights")])
def test_restore_rejects_changed_contents_before_loading(
    constructor, task, checkpoint_factory, data, cache, changed_file, monkeypatch,
):
    weights = checkpoint_factory(task)
    table, target = data
    estimator = constructor(model_version="1.0", n_estimators=1, device="cpu", use_kv_cache=cache, checkpoint_path=weights)
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


@pytest.mark.parametrize(("cache", "initial", "pending"), [
    (False, "a", "b"), (True, "a", None), (True, None, "b"),
])
def test_pending_source_applies_only_on_refit(
    constructor, task, checkpoint_factory, data, cache, initial, pending, monkeypatch,
):
    paths = {None: None, "a": checkpoint_factory(task, "a", 18), "b": checkpoint_factory(task, "b", 72)}
    official = checkpoint_factory(task, "official", 91)
    monkeypatch.setattr(checkpoints, "load_hosted_model", lambda task: checkpoints.load_checkpoint(official))
    table, target = data
    estimator = constructor(model_version="1.0", n_estimators=1, device="cpu", use_kv_cache=cache, checkpoint_path=paths[initial])
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
    estimator = constructor(model_version="1.0", n_estimators=1, device="cpu", checkpoint_path=Path("weights")).fit(table, target)
    assert estimator.get_params()["checkpoint_path"] == Path("weights")
    assert estimator._engine.checkpoint["path"] == str(weights)
    payload = pickle.dumps(estimator)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    restored = pickle.loads(payload)
    np.testing.assert_array_equal(predictions(restored, table[:3]), predictions(estimator, table[:3]))


@pytest.mark.parametrize(("case", "error", "message"), [
    ("weights", FileNotFoundError, "Checkpoint file not found"),
    ("config", FileNotFoundError, "configuration not found"),
    ("task", ValueError, "for regression, not classification"),
    ("type", ValueError, "checkpoint_path must be"),
    ("checksum", ValueError, "Weights do not match"),
])
def test_invalid_checkpoint_is_rejected(case, error, message, checkpoint_factory, tmp_path, data):
    path = tmp_path / "missing.safetensors"
    if case in ("config", "task", "checksum"):
        path = checkpoint_factory("regression" if case == "task" else "classification")
        config = path.with_name("config.json")
        if case == "config":
            config.unlink()
        if case == "checksum":
            config.write_bytes(config.read_bytes() + b"\n")
    if case == "type":
        path = 3
    with pytest.raises(error, match=message):
        CausiloClassifier(model_version="1.0", device="cpu", checkpoint_path=path).fit(*data)


def test_failed_source_change_discards_old_fit(constructor, task, checkpoint_factory, data):
    weights = checkpoint_factory(task)
    estimator = constructor(model_version="1.0", n_estimators=1, device="cpu", checkpoint_path=weights).fit(*data)
    estimator.set_params(checkpoint_path=weights.parent / "missing.safetensors")
    with pytest.raises(FileNotFoundError):
        estimator.fit(*data)
    with pytest.raises(NotFittedError):
        estimator.predict(data[0][:3])


def test_public_checkpoint_download_is_task_scoped(task, monkeypatch, tmp_path):
    folder = "classifier" if task == "classification" else "regressor"

    def download(repository, *, revision, allow_patterns, token):
        assert repository == "nums-ai/causilo-1.5"
        assert revision == "4ca2275bae85014a19422e8c18fcf8f24c5c14e1"
        assert allow_patterns == [f"{folder}/config.json", f"{folder}/model.safetensors"]
        assert token is False
        return tmp_path

    monkeypatch.setattr(checkpoints15, "snapshot_download", download)
    assert checkpoints15.download_checkpoint(select_checkpoint("1.5", task)) == tmp_path / folder


def test_local_checkpoint_selects_its_configured_runtime(tmp_path):
    (tmp_path / "model.safetensors").touch()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"architecture": RUNTIME15_FORMAT}))
    assert select_checkpoint("1.0", "classification", tmp_path).architecture == RUNTIME15_FORMAT


def test_refit_switches_checkpoint_and_drops_old_context(make_batch_model, checkpoint15, monkeypatch):
    sources = []
    checkpoint = read_local_checkpoint(checkpoint15("classification"))

    def load_original(task):
        sources.append(LEGACY_FORMAT)
        return make_batch_model(task)

    def load(task, source):
        sources.append(source.architecture)
        return checkpoints15.load_checkpoint(checkpoint, task)

    monkeypatch.setattr(checkpoints, "load_hosted_model", load_original)
    monkeypatch.setattr(checkpoints15, "load_hosted_model", load)
    table = np.arange(48).reshape(12, 4).astype(float)
    labels = np.tile(["a", "b"], 6)
    model = CausiloClassifier(model_version="1.0", device="cpu", n_estimators=1).fit(table, labels)
    expected = model.predict_proba(table[:2])
    model.set_params(model_version="1.5")
    np.testing.assert_array_equal(model.predict_proba(table[:2]), expected)

    model.fit(table, np.arange(len(table)) % 3)
    assert model.predict_proba(table[:2]).shape == (2, 3)
    np.testing.assert_array_equal(model.classes_, [0, 1, 2])
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(model.predict_proba(table[:2]), restored.predict_proba(table[:2]))
    saved = model.__getstate__()
    saved["checkpoint"] = "earlier-revision"
    with pytest.raises(ValueError, match="different checkpoint"):
        CausiloClassifier().__setstate__(saved)
    with pytest.raises(ValueError):
        model.fit(table, None)
    with pytest.raises(NotFittedError):
        model.predict(table[:2])
    assert not hasattr(model, "classes_")

    model.set_params(model_version="1.0").fit(table, labels)
    np.testing.assert_array_equal(model.predict_proba(table[:2]), expected)
    assert sources[:3] == [LEGACY_FORMAT, RUNTIME15_FORMAT, RUNTIME15_FORMAT]
    assert sources[-1] == LEGACY_FORMAT
