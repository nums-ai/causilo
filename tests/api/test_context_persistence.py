import pickle

import huggingface_hub
import numpy as np
import pytest
import torch

from causilo import CausiloClassifier, CausiloRegressor
from causilo.model_selection import MODEL_SOURCES
from causilo.v1_0.engine import Engine as Engine10
from causilo.v1_5 import checkpoints as checkpoints15
from causilo.v1_5.engine import Engine as Engine15


def test_unsupported_saved_context_schema_requires_refit():
    estimator = CausiloClassifier()
    saved = dict(estimator.__getstate__(), schema="unsupported")
    with pytest.raises(ValueError, match="Refit with the current Causilo installation"):
        estimator.__setstate__(saved)


@pytest.mark.parametrize(("version", "task"), [("1.0", "classification"), ("1.5", "regression")])
def test_python_context_roundtrip_on_requested_device(
    version, task, checkpoint_factory, checkpoint15, batch_runtime, mixed_data, tmp_path,
):
    is_classifier = task == "classification"
    constructor = CausiloClassifier if is_classifier else CausiloRegressor
    other = CausiloRegressor if is_classifier else CausiloClassifier
    checkpoint = checkpoint_factory(task) if version == "1.0" else checkpoint15(task)
    table, query = mixed_data
    targets = np.arange(len(table)) % 3 if is_classifier else np.arange(len(table)) / 3
    model = constructor(
        model_version=version, checkpoint_path=checkpoint, device="cpu", mode="light", use_kv_cache=True,
    ).fit(table, targets)
    prediction_method = "predict_proba" if is_classifier else "predict"
    expected = getattr(model, prediction_method)(query)

    path = tmp_path / "context.pkl"
    model.set_params(device="cuda:999").save_context(path)
    restored = constructor.load_context(path, device="cpu")
    assert restored._engine.device == torch.device("cpu")
    assert (restored.mode, restored.n_estimators) == ("light", None)
    np.testing.assert_array_equal(getattr(restored, prediction_method)(query), expected)
    with pytest.raises(TypeError, match=f"Saved context is for '{task}'"):
        other.load_context(path, device="cpu")


@pytest.mark.parametrize("version", ["1.0", "1.5"])
def test_save_context_protects_fitted_local_checkpoint(version, checkpoint_factory, checkpoint15, mixed_data, tmp_path):
    folder = checkpoint_factory("classification").parent if version == "1.0" else checkpoint15("classification")
    config, weights = folder / "config.json", folder / "model.safetensors"
    table, _ = mixed_data
    estimator = CausiloClassifier(
        model_version=version, checkpoint_path=folder, device="cpu", n_estimators=1,
    ).fit(table, np.arange(len(table)) % 3)
    estimator.set_params(checkpoint_path=tmp_path / "next-checkpoint")
    link = tmp_path / "checkpoint-link"
    link.symlink_to(weights)
    alias = tmp_path / "checkpoint-alias"
    alias.hardlink_to(config)
    originals = {path: path.read_bytes() for path in (config, weights)}

    for target in (config, weights, link, alias):
        with pytest.raises(ValueError, match="Context output must differ"):
            estimator.save_context(target)
    for path, original in originals.items():
        assert path.read_bytes() == original

    context = tmp_path / "context.pkl"
    estimator.save_context(context)
    estimator.save_context(context)
    assert context.is_file()


def test_save_context_protects_cached_checkpoint(checkpoint15, mixed_data, monkeypatch, tmp_path):
    source = MODEL_SOURCES["1.5", "classification"]
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    folder = checkpoint15("classification").rename(snapshot / source.folder)
    monkeypatch.setattr(checkpoints15, "snapshot_download", lambda *args, **kwargs: str(snapshot))
    table, _ = mixed_data
    estimator = CausiloClassifier(device="cpu", n_estimators=1).fit(table, np.arange(len(table)) % 3)

    def cached(repo, name, *, revision):
        assert repo == source.repository and revision == source.revision
        return str(snapshot / name)

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", cached)
    with pytest.raises(ValueError, match="Context output must differ"):
        estimator.save_context(folder / "config.json")


@pytest.mark.parametrize("cache", [False, True])
def test_restore_selects_a_device_for_the_fitted_runtime_before_refit(
    cache, checkpoint_factory, checkpoint15, batch_runtime, monkeypatch, tmp_path,
):
    table = np.random.default_rng(53).normal(size=(24, 4))
    targets = np.arange(len(table)) % 3
    query = table[:5]
    estimator = CausiloClassifier(
        model_version="1.0", checkpoint_path=checkpoint_factory("classification"),
        device="cpu", n_estimators=2, use_kv_cache=cache,
    ).fit(table, targets)
    expected = estimator.predict_proba(query)

    cuda_devices = ["cuda:0", "cuda:1"]
    next_checkpoint = checkpoint15("classification")
    estimator.set_params(
        model_version="1.5", checkpoint_path=next_checkpoint, device=cuda_devices,
    )
    path = tmp_path / "fitted.pkl"
    estimator.save_context(path)
    restored = CausiloClassifier.load_context(path, device="cpu")
    assert isinstance(restored._engine, Engine10)
    assert restored._engine.device == torch.device("cpu")
    assert restored.get_params() == (estimator.get_params() | {"device": "cpu"})
    np.testing.assert_array_equal(restored.predict_proba(query), expected)

    if torch.cuda.device_count() >= 2:
        devices = tuple(torch.device(name) for name in cuda_devices)
    else:
        devices = (torch.device("cpu"), torch.device("cpu:1"))

        def resolve_devices(request):
            assert request == tuple(cuda_devices)
            return devices

        monkeypatch.setattr("causilo.v1_5.engine.resolve_devices", resolve_devices)

    restored.set_params(device=cuda_devices).fit(table, targets)
    assert isinstance(restored._engine, Engine15)
    assert restored._engine.device_request == tuple(cuda_devices)
    assert restored._engine.devices == devices
    assert restored._engine.source.weights == next_checkpoint / "model.safetensors"
    assert set(restored._engine._model_replicas) == set(devices)


@pytest.mark.parametrize(("version", "cache"), [("1.0", False), ("1.5", False), ("1.5", True)])
def test_pickle_restore_applies_device_change_when_the_checkpoint_path_spelling_changes(
    version, cache, checkpoint_factory, checkpoint15, batch_runtime,
):
    table = np.random.default_rng(54).normal(size=(24, 4))
    targets = np.arange(len(table)) % 3
    query = table[:5]
    folder = checkpoint_factory("classification").parent if version == "1.0" else checkpoint15("classification")
    options = dict(model_version=version, checkpoint_path=folder, n_estimators=2, use_kv_cache=cache)
    estimator = CausiloClassifier(device="cpu", **options).fit(table, targets)
    device = "cuda:0" if torch.cuda.is_available() else "cpu:1"
    estimator.set_params(device=device, checkpoint_path=folder / "model.safetensors")

    restored = pickle.loads(pickle.dumps(estimator))
    expected = CausiloClassifier(device=device, **options).fit(table, targets).predict_proba(query)
    assert restored._engine.device == torch.device(device)
    actual = restored.predict_proba(query)
    if cache:
        # Moving a cache and building it on the destination can differ by rounding.
        np.testing.assert_allclose(actual, expected, rtol=0, atol=np.finfo(expected.dtype).eps)
        np.testing.assert_array_equal(actual.argmax(axis=1), expected.argmax(axis=1))
    else:
        np.testing.assert_array_equal(actual, expected)
