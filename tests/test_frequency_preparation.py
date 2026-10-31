"""Prepare frequency metadata once per operation, without retaining device state."""

import copy
import weakref

import numpy as np
import pytest
import torch

from causilo import CausiloClassifier, CausiloRegressor, checkpoints
from causilo.execution import cached
from causilo.execution.cached import cached_predictions
from causilo.execution.direct import BatchPlan, BatchPlanner, direct_predictions
from causilo.execution.recompute import RecomputeRunner
from causilo.execution.runner import ModelRunner
from causilo.model import Model, ModelConfig
from causilo.nn.cache import FrequencyWeights


@pytest.fixture
def data_and_model(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    models = {}
    with torch.random.fork_rng():
        torch.manual_seed(93)
        for task in ("classification", "regression"):
            model = Model(ModelConfig(
                task=task, width=16, expansion=2, group_size=3, frequencies=4,
                column_latents=5, column_heads=2, column_depths=(2, 2),
                row_heads=2, row_latents=2, row_depths=(3, 2),
                prediction_heads=2, prediction_depth=3, outputs=3,
            ))
            for name, parameter in model.named_parameters():
                if "latents" in name or name.endswith("frequencies") or name.endswith("missing"):
                    torch.nn.init.normal_(parameter, std=0.1)
            models[task] = model.eval()
    monkeypatch.setattr(checkpoints, "load_pretrained_model", lambda task: copy.deepcopy(models[task]))
    table = np.random.default_rng(12).normal(size=(10, 5))
    yield table, np.arange(1, 11)
    torch.set_num_threads(previous)


def track_preparation(monkeypatch):
    original = FrequencyWeights.prepare
    calls = []

    def prepare(cls, frequencies, device):
        assert frequencies.device.type == "cpu", "Compute total before moving metadata to the device"
        result = original(frequencies, device)
        calls.append(weakref.ref(result))
        return result

    monkeypatch.setattr(FrequencyWeights, "prepare", classmethod(prepare))
    return calls


@pytest.mark.parametrize("task", ["classification", "regression", "ecoc"])
@pytest.mark.parametrize("mode", ["direct", "cached", "recompute"])
def test_one_preparation_per_fit_or_predict_across_members_chunks_and_codes(
    data_and_model, monkeypatch, task, mode,
):
    table, counts = data_and_model
    labels = np.arange(len(table)) % (5 if task == "ecoc" else 3)
    kind = CausiloRegressor if task == "regression" else CausiloClassifier
    calls = track_preparation(monkeypatch)
    fitted = kind(n_estimators=8, device="cpu", use_kv_cache=mode == "cached").fit(table, labels, counts)
    assert len(calls) == int(mode == "cached")
    assert all(reference() is None for reference in calls), "Fit must not retain prepared metadata"

    predict = fitted.predict if task == "regression" else fitted.predict_proba
    expected = predict(table[:4])
    if mode == "cached":
        monkeypatch.setattr(cached, "execution_budget", lambda device: 1)
        monkeypatch.setattr(cached, "query_memory", lambda config, features, training, queries, device: queries)
    else:
        monkeypatch.setattr(BatchPlanner, "choose", lambda *args: BatchPlan(1, 1, mode=mode))
    method = "predict_cached" if mode == "cached" else "predict"
    runner = RecomputeRunner if mode == "recompute" else ModelRunner
    original = getattr(runner, method)
    executions = []

    def execute(self, *args, **kwargs):
        weights = kwargs["frequencies"]
        assert isinstance(weights, FrequencyWeights)
        assert weights.count == counts.sum()
        executions.append(id(weights))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(runner, method, execute)
    before = len(calls)
    actual = predict(table[:4])
    assert len(calls) == before + 1
    assert len(executions) >= 8
    assert len(set(executions)) == 1
    assert calls[-1]() is None, "Predict must release prepared metadata"
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
    if task == "regression":
        before = len(calls)
        assert np.isfinite(fitted.predict(table[:4], output_type="raw")).all()
        assert len(calls) == before + 1
        assert calls[-1]() is None


@pytest.mark.parametrize("cache", [False, True])
def test_unweighted_execution_does_not_prepare_metadata(data_and_model, monkeypatch, cache):
    table, _ = data_and_model
    calls = track_preparation(monkeypatch)
    for counts in (None, np.ones(len(table), dtype=int)):
        fitted = CausiloClassifier(n_estimators=4, device="cpu", use_kv_cache=cache).fit(
            table, np.arange(len(table)) % 3, counts,
        )
        fitted.predict_proba(table[:3])
    assert calls == []


@pytest.mark.parametrize("cache", [False, True])
def test_standalone_prediction_iterator_prepares_once(data_and_model, monkeypatch, cache):
    table, counts = data_and_model
    fitted = CausiloClassifier(n_estimators=4, device="cpu", use_kv_cache=cache).fit(
        table, np.arange(len(table)) % 3, counts,
    )
    engine = fitted._engine
    state = engine.state
    transformed = state.dataset.query_tables(table[:3])
    calls = track_preparation(monkeypatch)
    with torch.inference_mode():
        if cache:
            results = list(cached_predictions(
                engine.model, state.dataset, transformed, state.cached_members(), engine.device, lambda x: x,
            ))
        else:
            monkeypatch.setattr(BatchPlanner, "choose", lambda *args: BatchPlan(1, 1))
            results = list(direct_predictions(
                engine.model, state.dataset, transformed, engine.device, lambda x: x,
            ))
    assert len(results) == 4
    assert all(torch.isfinite(result).all() for _, result in results)
    assert len(calls) == 1 and calls[0]() is None


@pytest.mark.parametrize("cache", [False, True])
def test_prediction_retry_reuses_prepared_metadata(data_and_model, monkeypatch, cache):
    table, counts = data_and_model
    fitted = CausiloClassifier(n_estimators=4, device="cpu", use_kv_cache=cache).fit(
        table, np.arange(len(table)) % 3, counts,
    )
    expected = fitted.predict_proba(table[:3])
    calls = track_preparation(monkeypatch)
    name = "predict_cached" if cache else "predict"
    original = getattr(ModelRunner, name)
    executions = []

    def fail_once(self, *args, **kwargs):
        executions.append(id(kwargs["frequencies"]))
        if len(executions) == 1:
            raise torch.cuda.OutOfMemoryError("Injected frequency metadata retry")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ModelRunner, name, fail_once)
    actual = fitted.predict_proba(table[:3])
    assert len(calls) == 1 and calls[0]() is None
    assert len(executions) > 1 and len(set(executions)) == 1
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS frequency metadata placement")
def test_accelerator_preparation_uses_cpu_counts(data_and_model, monkeypatch):
    table, counts = data_and_model
    labels = np.arange(len(table)) % 3
    expected = CausiloClassifier(n_estimators=4, device="cpu").fit(table, labels, counts).predict_proba(table[:3])
    calls = track_preparation(monkeypatch)
    fitted = CausiloClassifier(n_estimators=4, device="mps", use_kv_cache=True).fit(table, labels, counts)
    actual = fitted.predict_proba(table[:3])
    assert len(calls) == 2 and all(reference() is None for reference in calls)
    np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
