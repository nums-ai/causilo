import random

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from causilo import CausiloClassifier, CausiloRegressor
from causilo.data.encoding import FeatureEncoder
from causilo.data.ensemble import make_ensemble_members
from causilo.data.normalization import Normalizer


def test_query_uses_fitted_categories_and_column_order():
    training = pd.DataFrame({"kind": ["a", "b", None], "value": [1.0, 2.0, 3.0]})
    schema, _ = FeatureEncoder.fit(training)
    query = pd.DataFrame({"kind": ["new", "a"], "value": [5.0, np.nan]})
    encoded = schema.transform(query)
    assert np.isnan(encoded[0, 0]) and np.isnan(encoded[1, 1])
    with pytest.raises(ValueError, match="name and order"):
        schema.transform(query[["value", "kind"]])


@pytest.mark.parametrize("method", ["none", "power", "rank2gaussian", "robust"])
def test_normalization_keeps_missing_mask_and_fitted_state(method):
    table = np.random.default_rng(12).normal(size=(60, 4))
    table[::5, 1] = np.nan
    transform = Normalizer.fit(table, method)
    first = transform.transform(table)
    transform.transform(table[:3] * 100)
    np.testing.assert_array_equal(first, transform.transform(table))
    np.testing.assert_array_equal(np.isnan(first), np.isnan(table))
    assert np.isfinite(first[~np.isnan(first)]).all()


def test_ensemble_is_independent_of_global_randomness():
    expected = make_ensemble_members(7, 3, 8, 42)
    random.seed(981)
    np.random.seed(83)
    assert make_ensemble_members(7, 3, 8, 42) == expected
    assert [member.normalization for member in expected] == ["none", "rank2gaussian", "robust", "power"] * 2
    assert make_ensemble_members(7, 3, 1, 42)[0].feature_order == tuple(range(7))


@pytest.mark.parametrize("constructor", [CausiloClassifier, CausiloRegressor])
def test_sklearn_clone_does_not_load_weights(constructor):
    model = constructor(n_estimators=2, random_state=17, device="cpu", use_kv_cache=True)
    assert clone(model).get_params() == model.get_params()


@pytest.mark.parametrize("constructor", [CausiloClassifier, CausiloRegressor])
@pytest.mark.parametrize("seed", [None, True, -1, 1.5, "42", np.random.default_rng(42)])
def test_invalid_random_state_fails_before_loading_weights(constructor, seed, monkeypatch):
    from causilo import checkpoints

    monkeypatch.setattr(checkpoints, "load_pretrained_model", lambda task: pytest.fail("Loaded weights"))
    with pytest.raises(ValueError, match="random_state"):
        constructor(random_state=seed, device="cpu").fit([[0], [1]], [0, 1])


@pytest.mark.pretrained
@pytest.mark.parametrize("constructor", [CausiloClassifier, CausiloRegressor])
@pytest.mark.parametrize("cache", [False, True])
def test_public_seed_controls_fitted_ensemble(constructor, cache):
    import pickle

    table = np.random.default_rng(13).normal(size=(16, 7))
    targets = np.arange(16) % 3
    model = constructor(n_estimators=4, device="cpu", use_kv_cache=cache)
    assert model.random_state == 42
    model.fit(table, targets)
    expected = model._engine.require_state().dataset.members
    predictions = model.predict(table[:3])
    global_state = random.getstate()
    model.fit(table, targets)
    assert random.getstate() == global_state
    assert model._engine.require_state().dataset.members == expected
    np.testing.assert_array_equal(model.predict(table[:3]), predictions)
    model.set_params(random_state=np.int64(17))
    assert model._engine.require_state().dataset.members == expected
    model.fit(table, targets)
    assert model._engine.require_state().dataset.members != expected
    restored = pickle.loads(pickle.dumps(model))
    assert restored.random_state == 17
    assert restored._engine.require_state().dataset.members == model._engine.require_state().dataset.members
    np.testing.assert_array_equal(restored.predict(table[:3]), model.predict(table[:3]))
