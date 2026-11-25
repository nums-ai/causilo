import weakref

import numpy as np
import pytest
import torch

from causilo import CausiloClassifier, CausiloRegressor
from causilo.v1_5.execution import direct, prediction
from causilo.v1_5.execution.direct import EnsembleRunner
from causilo.v1_5.execution.recompute import RecomputeRunner
from causilo.v1_5.execution.runner import ModelRunner
from causilo.v1_5.nn.cache import KVPair
from causilo.v1_5.nn.layers.attention import AttentionLayer


@pytest.mark.parametrize(("task", "weighted"), [("classification", False), ("regression", True)])
def test_host_prediction_layers_are_transferred_once_per_window(checkpoint15, mixed_data, task, weighted, monkeypatch):
    table, query = mixed_data
    classifier = task == "classification"
    model_type = CausiloClassifier if classifier else CausiloRegressor
    targets = np.arange(len(table)) % 3 if classifier else table["a"].to_numpy()
    monkeypatch.setattr(EnsembleRunner, "_ensemble_batch_count", lambda *args: 2)
    fitted = model_type(
        checkpoint_path=checkpoint15(task), device="cpu", n_estimators=4, use_kv_cache=True,
    ).fit(table, targets, sample_count=1 + np.arange(len(table)) % 3 if weighted else None)
    state = fitted._engine.state
    pairs = {id(pair) for group in state.caches for pair in group.prediction.layers.values()}
    moved = []
    move = KVPair.to

    def track(pair, *args, **kwargs):
        if id(pair) in pairs:
            moved.append(id(pair))
        return move(pair, *args, **kwargs)

    def predict():
        return fitted.predict_logits(query) if classifier else fitted.predict(
            query, output_type="quantiles", quantiles=[0.8, 0.1, 0.8],
        )

    monkeypatch.setattr(KVPair, "to", track)
    monkeypatch.setattr(state.runner, "query_batch_size", lambda *args, **kwargs: 5)
    monkeypatch.setattr(
        ModelRunner, "_predict_native_cached",
        lambda runner, rows, cache, weights: RecomputeRunner(runner.model).predict_cached(rows, cache, weights=weights),
    )
    reference = predict()
    assert sorted(moved) == sorted(list(pairs) * 3)
    moved.clear()
    prediction_layers = set(fitted._engine.model.prediction.layers)
    queried = set()
    attention_query = AttentionLayer.query

    def check_weights(layer, rows, context, weights=None):
        if layer in prediction_layers:
            assert weights is state.runner.weights
            queried.add(layer)
        return attention_query(layer, rows, context, weights=weights)

    monkeypatch.setattr(AttentionLayer, "query", check_weights)
    monkeypatch.setattr(state.runner, "cached_query_window", lambda *args, **kwargs: len(query))
    monkeypatch.setattr(state.runner, "_cache_replay_ceiling", lambda: 1)
    monkeypatch.setattr(ModelRunner, "host_prediction_bytes", staticmethod(lambda _: 2**62))
    np.testing.assert_array_equal(predict(), reference)
    assert queried == prediction_layers
    assert sorted(moved) == sorted(pairs)


def test_failed_window_releases_uncommitted_outputs_before_retry(checkpoint15, mixed_data, monkeypatch):
    table, query = mixed_data
    counts = 1 + np.arange(len(table)) % 3
    model = CausiloClassifier(
        checkpoint_path=checkpoint15("classification"), device="cpu", n_estimators=4, use_kv_cache=True,
    ).fit(table, np.arange(len(table)) % 3, sample_count=counts)
    runner = model._engine.state.runner
    monkeypatch.setattr(runner, "query_batch_size", lambda *args, **kwargs: 2)
    expected = model.predict_logits(query)
    predict_window = runner.predict_cached_window
    predict_one = runner.predict_cached
    combine = prediction.classification_logits
    windows = []
    discarded = []
    retried = []

    def track_window(*args, **kwargs):
        outputs = predict_window(*args, **kwargs)
        windows.append(len(outputs))
        if len(windows) == 2:
            discarded.extend(weakref.ref(output) for output in outputs)
        return outputs

    def fail_second_chunk(output, orders):
        if discarded and output is discarded[1]():
            raise MemoryError("Combining the second chunk in the second window")
        return combine(output, orders)

    def track_retry(dataset, chunk, caches, **kwargs):
        assert discarded and all(ref() is None for ref in discarded)
        retried.append(len(chunk))
        return predict_one(dataset, chunk, caches, **kwargs)

    monkeypatch.setattr(runner, "cached_query_window", lambda *args, **kwargs: 4)
    monkeypatch.setattr(runner, "_cache_replay_ceiling", lambda: 2**62)
    monkeypatch.setattr(runner, "predict_cached_window", track_window)
    monkeypatch.setattr(prediction, "classification_logits", fail_second_chunk)
    monkeypatch.setattr(runner, "predict_cached", track_retry)
    np.testing.assert_array_equal(model.predict_logits(query), expected)
    assert windows == [2, 2]
    assert retried == [2, 2, 2, 2]


def test_weighted_cache_window_respects_output_budget(checkpoint15, mixed_data, monkeypatch):
    table, _ = mixed_data
    fitted = CausiloRegressor(
        checkpoint_path=checkpoint15("regression"), device="cpu", n_estimators=4, use_kv_cache=True,
    ).fit(table, table["a"].to_numpy(), sample_count=1 + np.arange(len(table)) % 3)
    state = fitted._engine.state
    runner = state.runner
    monkeypatch.setattr(runner, "device", torch.device("cuda:0"))
    monkeypatch.setattr(runner, "_cache_replay_ceiling", lambda: 1)
    monkeypatch.setattr(direct, "available_bytes", lambda _: 8 << 20)
    window = runner.cached_query_window
    assert window(state.dataset, state.caches, 32, 128) == 128
    levels = np.linspace(0.001, 0.999, 8192)
    assert window(state.dataset, state.caches, 32, 128, levels=levels) <= 32
    assert window(state.dataset, state.caches, 32, 32) == 0

    cache_bytes = runner._model_runner.host_prediction_bytes(state.caches[0])
    monkeypatch.setattr(runner, "_cache_replay_ceiling", lambda: cache_bytes)
    assert window(state.dataset, state.caches, 32, 128) == 0


def test_tiled_prediction_falls_back_before_encoding_a_window(checkpoint15, mixed_data, monkeypatch):
    table, query = mixed_data
    model = CausiloClassifier(
        checkpoint_path=checkpoint15("classification"), device="cpu", n_estimators=1, use_kv_cache=True,
    ).fit(table, np.arange(len(table)) % 3)
    runner = model._engine.state.runner
    monkeypatch.setattr(runner, "query_batch_size", lambda *args, **kwargs: 5)
    expected = model.predict_proba(query)
    encode = ModelRunner._encode_rows
    calls = []

    def track_encode(self, *args, **kwargs):
        calls.append(1)
        return encode(self, *args, **kwargs)

    monkeypatch.setattr(runner, "cached_query_window", lambda *args, **kwargs: len(query))
    monkeypatch.setattr(runner, "_cache_replay_ceiling", lambda: 0)
    monkeypatch.setattr(RecomputeRunner, "native_query_fits", lambda *args: False)
    monkeypatch.setattr(ModelRunner, "_encode_rows", track_encode)
    np.testing.assert_array_equal(model.predict_proba(query), expected)
    assert len(calls) == 3
