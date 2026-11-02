from dataclasses import replace

import numpy as np
import pytest
import torch

from causilo.data.dataset import PreparedDataset
from causilo.execution import batched
from causilo.execution.batched import BatchItem, batched_predictions
from causilo.execution.runner import ModelRunner


@pytest.fixture
def setup(make_batch_model):
    model = make_batch_model("regression")
    rng = np.random.default_rng(16)
    items = []
    for index, (training, queries, features) in enumerate(((11, 5, 4), (17, 3, 7), (14, 4, 5))):
        x = rng.normal(size=(training, features))
        x[1, 1] = np.nan
        y = rng.normal(size=training)
        fitted = PreparedDataset.prepare(
            x, y, task="regression", n_estimators=2, retain_preprocessing=False,
            random_state=42, class_permutation_size=model.config.outputs,
        )
        transformed = fitted.query_tables(rng.normal(size=(queries, features)))
        for member_index, member in enumerate(fitted.members_by_normalization()):
            items.append(BatchItem(
                (str(index), member_index), fitted, member, transformed,
                output_type="mean" if index == 1 else "raw",
            ))
    return model, items, torch.device("cpu")


def with_caches(model, items):
    runner = ModelRunner(model)
    device = next(model.parameters()).device
    result = []
    with torch.inference_mode():
        for item in items:
            table = item.fitted.training_table(item.member.normalization)[:, item.member.feature_order]
            target = item.fitted.targets_for(item.member)
            cache = runner.build_cache(
                torch.tensor(table, dtype=torch.float32, device=device).unsqueeze(0),
                torch.tensor(target, dtype=torch.float32, device=device).unsqueeze(0),
            )
            result.append(replace(item, cache=cache))
    return result


def collect(model, items, device):
    pairs = list(batched_predictions(model, items, device))
    assert len(pairs) == len(items)
    assert len({item.key for item, _ in pairs}) == len(items)
    return {item.key: values for item, values in pairs}


@pytest.mark.parametrize("cache", [False, True])
def test_scheduler_fuses_and_preserves_item_identity(setup, cache, monkeypatch):
    model, items, device = setup
    if cache:
        items = with_caches(model, items)
    expected = {}
    with torch.inference_mode():
        for item in items:
            _, expected[item.key] = next(batched._singleton(model, item, device))
    calls = []
    original = batched._execute

    def observed(model, items, device):
        calls.append(len(items))
        return original(model, items, device)

    monkeypatch.setattr(batched, "_execute", observed)
    actual = collect(model, list(reversed(items)), device)
    assert max(calls) > 1
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("cache", [False, True])
def test_oom_splits_without_losing_or_duplicating_outputs(setup, cache, monkeypatch):
    model, items, device = setup
    if cache:
        items = with_caches(model, items)
    expected = collect(model, items, device)
    original = batched._execute
    attempted = []

    def fail_large(model, items, device):
        attempted.append(len(items))
        if len(items) > 2:
            raise torch.cuda.OutOfMemoryError("injected multi-table OOM")
        return original(model, items, device)

    monkeypatch.setattr(batched, "_execute", fail_large)
    actual = collect(model, items, device)
    assert max(attempted) > 2
    assert min(attempted) <= 2
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=3e-5, rtol=3e-5)


def test_tight_budget_uses_singleton_recovery(setup, monkeypatch):
    model, items, device = setup
    expected = collect(model, items, device)
    monkeypatch.setattr(batched, "execution_budget", lambda device: 1)
    monkeypatch.setattr(batched, "_execute", lambda *args: pytest.fail("Fused pack exceeded budget"))
    actual = collect(model, items, device)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=3e-5, rtol=3e-5)


def test_cached_execution_preserves_fitted_contexts(setup):
    model, items, device = setup
    items = with_caches(model, items)
    tensors = [
        tensor for item in items
        for block in (*item.cache.columns, item.cache.prediction)
        for layer in block.layers for tensor in (layer.key, layer.value)
    ]
    snapshots = [tensor.clone() for tensor in tensors]
    collect(model, items, device)
    for tensor, snapshot in zip(tensors, snapshots):
        torch.testing.assert_close(tensor, snapshot, atol=0, rtol=0)


def test_non_oom_failure_is_not_retried(setup, monkeypatch):
    model, items, device = setup
    calls = []

    def fail(*args):
        calls.append(1)
        raise RuntimeError("invalid model input")

    monkeypatch.setattr(batched, "_execute", fail)
    with pytest.raises(RuntimeError, match="invalid model input"):
        collect(model, items, device)
    assert len(calls) == 1
