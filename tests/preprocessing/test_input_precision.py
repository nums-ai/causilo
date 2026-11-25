import pickle
import random

import numpy as np
import pytest

from causilo import CausiloClassifier, CausiloRegressor
from causilo.v1_0 import checkpoints
from causilo.v1_0.data.ensemble import make_ensemble_members
from causilo.v1_0.execution.runner import ModelRunner


def test_object_and_dataframe_predictions_match(checkpoint_factory, mixed_data, batch_runtime):
    table, query = mixed_data
    labels = np.arange(len(table)) % 3
    options = dict(
        checkpoint_path=checkpoint_factory("classification"), model_version="1.0", n_estimators=1, device="cpu",
    )
    frame = CausiloClassifier(**options).fit(table, labels)
    array = CausiloClassifier(**options).fit(table.to_numpy(dtype=object), labels)
    np.testing.assert_array_equal(
        frame.predict_proba(query), array.predict_proba(query.to_numpy(dtype=object)),
    )


def test_target_scaling_and_prediction_keep_large_offsets(checkpoint_factory, mixed_data, batch_runtime):
    table, query = mixed_data
    target = np.arange(len(table), dtype=np.float64)
    options = dict(
        checkpoint_path=checkpoint_factory("regression"), model_version="1.0", n_estimators=4, device="cpu",
    )
    centered = CausiloRegressor(**options).fit(table, target)
    shifted = CausiloRegressor(**options).fit(table, target + 1e9)
    np.testing.assert_array_equal(
        shifted._engine.state.dataset.targets, centered._engine.state.dataset.targets,
    )
    result = shifted.predict(query)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result - 1e9, centered.predict(query), atol=1e-6, rtol=0)


def test_invalid_cache_counts_fail_before_prediction(checkpoint_factory, mixed_data, batch_runtime, monkeypatch):
    table, query = mixed_data
    model = CausiloClassifier(
        checkpoint_path=checkpoint_factory("classification"), model_version="1.0",
        n_estimators=2, device="cpu", use_kv_cache=True,
    ).fit(table, np.arange(len(table)) % 3)
    state = model._engine.require_state()
    caches = state.caches
    monkeypatch.setattr(ModelRunner, "predict_cached", lambda *a, **kw: pytest.fail("Ran inference"))
    for wrong_count in (caches[:-1], caches + caches[:1]):
        state.caches = wrong_count
        with pytest.raises(ValueError, match="Cache count"):
            model.predict(query)


@pytest.mark.parametrize("seed", [None, True, -1, 1.5])
def test_invalid_version_1_0_seed_fails_before_loading_weights(seed, monkeypatch):
    monkeypatch.setattr(checkpoints, "load_hosted_model", lambda task: pytest.fail("Loaded weights"))
    with pytest.raises(ValueError, match="random_state"):
        CausiloClassifier(model_version="1.0", random_state=seed, device="cpu").fit([[0], [1]], [0, 1])


def test_fixed_seed_refits_preserve_saved_ensemble(checkpoint_factory, batch_runtime):
    table = np.random.default_rng(13).normal(size=(16, 7))
    labels = np.arange(len(table)) % 3
    model = CausiloClassifier(
        checkpoint_path=checkpoint_factory("classification"), model_version="1.0", n_estimators=8, device="cpu",
    )
    assert model.random_state == 42
    expected = make_ensemble_members(7, 3, 8, 42)
    assert [member.normalization for member in expected] == ["none", "rank2gaussian", "robust", "power"] * 2
    assert make_ensemble_members(7, 3, 1, 42)[0].feature_order == tuple(range(7))
    predictions = model.fit(table, labels).predict_proba(table[:3])
    global_state = random.getstate()
    try:
        random.seed(981)
        refit_state = random.getstate()
        model.fit(table, labels)
        assert random.getstate() == refit_state
    finally:
        random.setstate(global_state)
    assert model._engine.require_state().dataset.members == expected
    np.testing.assert_array_equal(model.predict_proba(table[:3]), predictions)

    model.set_params(random_state=np.int64(17))
    assert model._engine.require_state().dataset.members == expected
    model.fit(table, labels)
    assert model._engine.require_state().dataset.members != expected
    restored = pickle.loads(pickle.dumps(model))
    assert restored.random_state == 17
    assert restored._engine.require_state().dataset.members == model._engine.require_state().dataset.members
    np.testing.assert_array_equal(restored.predict_proba(table[:3]), model.predict_proba(table[:3]))
