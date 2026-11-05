"""Frequency contexts survive the shared scheduler and checkpoint contracts."""

import copy
import hashlib
import json
import pickle

import numpy as np
import pytest
from safetensors.torch import save_file
from sklearn.exceptions import NotFittedError

from causilo import CausiloClassifier, CausiloRegressor, checkpoints
from causilo.execution import batched
from causilo.execution.runner import ModelRunner


@pytest.fixture
def model_loader(make_batch_model, monkeypatch):
    models = {task: make_batch_model(task) for task in ("classification", "regression")}
    monkeypatch.setattr(checkpoints, "load_pretrained_model", lambda task: copy.deepcopy(models[task]))


@pytest.fixture
def sample():
    rng = np.random.default_rng(642)
    table = rng.normal(size=(12, 5))
    table[2, 1] = np.nan
    counts = np.array([1, 4, 2, 0, 3, 1, 5, 2, 1, 3, 2, 1])
    return table, np.arange(len(table)) % 3, counts, table[[1, 2, 5, 9]].copy()


def probabilities(model, query):
    if isinstance(model, CausiloClassifier):
        return model.predict_proba(query)
    return model.predict(query, output_type="raw")


@pytest.mark.parametrize("kind", [CausiloClassifier, CausiloRegressor])
@pytest.mark.parametrize("cache", [False, True])
def test_weighted_members_use_packed_scheduler_and_match_expansion(
    model_loader, sample, kind, cache, monkeypatch,
):
    x, y, counts, query = sample
    indices = np.repeat(np.arange(len(x)), counts)
    options = dict(n_estimators=4, device="cpu", use_kv_cache=cache)
    expanded = kind(**options).fit(x[indices], y[indices])
    expected = probabilities(expanded, query)
    weighted = kind(**options).fit(x, y, sample_count=counts)
    packs = []
    original = batched._execute

    def observe(model, items, device):
        packs.append(len(items))
        assert all(item.fitted is weighted._engine.state.dataset for item in items)
        return original(model, items, device)

    monkeypatch.setattr(batched, "_execute", observe)
    actual = probabilities(weighted, query)
    assert packs and max(packs) > 1, "Weighted members must exercise fused execution"
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("cache", [False, True])
def test_weighted_scheduler_split_reaches_singletons_without_losing_counts(
    model_loader, sample, cache, monkeypatch,
):
    x, y, counts, query = sample
    model = CausiloClassifier(n_estimators=4, device="cpu", use_kv_cache=cache).fit(x, y, counts)
    expected = model.predict_proba(query)
    attempted = []
    singletons = []
    original = batched._singleton

    def fail_pack(model, items, device):
        attempted.append(len(items))
        raise MemoryError("Injected packing failure for frequency contexts")

    def observe_singleton(model, item, device):
        singletons.append(item.key)
        yield from original(model, item, device)

    monkeypatch.setattr(batched, "_execute", fail_pack)
    monkeypatch.setattr(batched, "_singleton", observe_singleton)
    actual = model.predict_proba(query)
    assert attempted and max(attempted) == 4
    assert len(singletons) == 4 and len(set(singletons)) == 4
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("kind", [CausiloClassifier, CausiloRegressor])
@pytest.mark.parametrize("cache", [False, True])
def test_weighted_local_checkpoint_roundtrip_keeps_fitted_identity(
    make_batch_model, sample, tmp_path, kind, cache, monkeypatch,
):
    task = "classification" if kind is CausiloClassifier else "regression"
    model = make_batch_model(task)
    encoded = json.dumps(model.config.record()).encode()
    (tmp_path / "config.json").write_bytes(encoded)
    path = tmp_path / "model.safetensors"
    save_file(model.state_dict(), path, metadata={
        "architecture": "causilo-v1.0", "format_version": "1", "task": task,
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
    })
    monkeypatch.setattr(checkpoints, "snapshot_download", lambda *a, **k: pytest.fail("Contacted the Hub"))
    x, y, counts, query = sample
    estimator = kind(n_estimators=4, device="cpu", use_kv_cache=cache, checkpoint_path=path).fit(x, y, counts)
    expected = probabilities(estimator, query)
    # A future refit source must not replace the source belonging to saved context.
    future = tmp_path / "next-checkpoint.safetensors"
    estimator.set_params(checkpoint_path=future)
    monkeypatch.setattr(ModelRunner, "build_cache", lambda *a, **k: pytest.fail("Rebuilt saved context"))
    restored = pickle.loads(pickle.dumps(estimator))
    assert restored.checkpoint_path == future
    assert restored._engine.checkpoint["path"] == str(path)
    np.testing.assert_array_equal(restored._engine.state.dataset.frequencies, counts[counts > 0])
    np.testing.assert_array_equal(probabilities(restored, query), expected)


@pytest.mark.parametrize("kind", [CausiloClassifier, CausiloRegressor])
def test_failed_weighted_refit_invalidates_single_and_batch_modes(model_loader, sample, kind):
    x, y, counts, query = sample
    estimator = kind(n_estimators=2, device="cpu").fit_batched({"first": (x, y), "second": (x[:, :4], y)})
    with pytest.raises(ValueError, match="sample_count|frequenc|count"):
        estimator.fit(x, y, sample_count=np.zeros_like(counts))
    with pytest.raises(NotFittedError):
        estimator.predict(query)
    with pytest.raises(NotFittedError):
        estimator.predict_batched({"first": query})
    assert estimator._engine.state is None and estimator._engine.batch_states is None
    assert not hasattr(estimator, "n_features_in_batched_")
    estimator.fit(x, y, sample_count=counts)
    assert np.isfinite(probabilities(estimator, query)).all()
    assert estimator._engine.batch_states is None
