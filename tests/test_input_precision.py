import numpy as np
import pandas as pd
import pytest
import torch

from causilo import CausiloClassifier, CausiloRegressor
from causilo.data.dataset import PreparedDataset
from causilo.data.encoding import FeatureEncoder
from causilo.execution.runner import ModelRunner


def test_target_standardization_preserves_large_offsets():
    table = np.arange(72, dtype=float).reshape(36, 2)
    target = np.arange(36, dtype=np.float64)
    options = dict(
        task="regression",
        n_estimators=1,
        retain_preprocessing=True,
        random_state=42,
        class_permutation_size=0,
    )
    centered = PreparedDataset.prepare(table, target, **options)
    shifted = PreparedDataset.prepare(table, target + 1e9, **options)
    np.testing.assert_array_equal(centered.targets, shifted.targets)
    assert np.unique(shifted.targets).size == len(target)


def test_object_columns_preserve_measurements_and_categories():
    table = np.array([[1.5, "1", True], [None, "2", False], [3.5, None, True]], dtype=object)
    encoder, _ = FeatureEncoder.fit(table)
    assert encoder.continuous == (0,)
    assert encoder.categories == (1, 2)
    result = encoder.transform(np.array([[2.5, "new", False], [pd.NA, "1", True]], dtype=object))
    assert result[0, -1] == 2.5
    assert np.isnan(result[0, 0]) and np.isnan(result[1, -1])


def test_numeric_categories_keep_explicit_dataframe_type():
    table = pd.DataFrame({"category": pd.Categorical([1, 2, 1]), "value": [1.5, 2.5, 3.5]})
    encoder, _ = FeatureEncoder.fit(table)
    assert encoder.categories == (0,)
    assert encoder.continuous == (1,)


@pytest.fixture
def fitted_inputs():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    table = pd.DataFrame({"category": ["a", "b", None] * 12, "value": np.arange(36, dtype=float)})
    query = pd.DataFrame({"category": ["new", None, "a"], "value": [2.5, np.nan, 7.5]})
    yield table, query
    torch.set_num_threads(previous)


@pytest.mark.pretrained
@pytest.mark.parametrize("constructor", [CausiloClassifier, CausiloRegressor])
def test_object_and_dataframe_predictions_match(constructor, fitted_inputs):
    table, query = fitted_inputs
    target = np.arange(len(table)) % 3
    frame = constructor(n_estimators=8, device="cpu").fit(table, target)
    array = constructor(n_estimators=8, device="cpu").fit(table.to_numpy(dtype=object), target)
    method = "predict_proba" if constructor is CausiloClassifier else "predict"
    np.testing.assert_array_equal(
        getattr(frame, method)(query), getattr(array, method)(query.to_numpy(dtype=object))
    )


@pytest.mark.pretrained
def test_prediction_restoration_preserves_large_offsets(fitted_inputs):
    table, query = fitted_inputs
    target = np.arange(len(table), dtype=np.float64)
    centered = CausiloRegressor(n_estimators=8, device="cpu").fit(table, target)
    shifted = CausiloRegressor(n_estimators=8, device="cpu").fit(table, target + 1e9)
    result = shifted.predict(query)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result - 1e9, centered.predict(query), atol=1e-6, rtol=0)


@pytest.mark.pretrained
@pytest.mark.parametrize("constructor", [CausiloClassifier, CausiloRegressor])
@pytest.mark.parametrize("cache_count", [7, 9])
def test_cache_count_is_checked_before_execution(constructor, cache_count, fitted_inputs, monkeypatch):
    table, query = fitted_inputs
    model = constructor(n_estimators=8, device="cpu", use_kv_cache=True).fit(table, np.arange(len(table)) % 3)
    state = model._engine.require_state()
    state.caches = (state.caches + state.caches[:1])[:cache_count]

    def reject_execution(*args, **kwargs):
        pytest.fail("Incomplete ensemble reached model execution")

    monkeypatch.setattr(ModelRunner, "predict_cached", reject_execution)
    with pytest.raises(ValueError, match="Cache count"):
        model.predict(query)
