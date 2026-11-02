from dataclasses import replace

import numpy as np
import pytest
import torch

from causilo.data.dataset import PreparedDataset
from causilo.execution import batched
from causilo.execution.batched import BatchItem, batched_predictions
from causilo.execution.runner import ModelRunner
from causilo.model import Model, ModelConfig


@pytest.fixture
def setup(request):
    task = getattr(request, "param", "regression")
    with torch.random.fork_rng():
        torch.manual_seed(43)
        model = Model(ModelConfig(
            task=task, width=16, expansion=2, group_size=3, frequencies=4,
            column_latents=5, column_heads=2, column_depths=(2, 2),
            row_heads=2, row_latents=2, row_depths=(3, 2),
            prediction_heads=2, prediction_depth=3, outputs=4 if task == "classification" else 9,
        ))
        for name, parameter in model.named_parameters():
            if "latents" in name or name.endswith("frequencies") or name.endswith("missing"):
                torch.nn.init.normal_(parameter, std=0.1)
    rng = np.random.default_rng(16)
    items = []
    for index, (training, queries, features) in enumerate(((11, 5, 4), (17, 3, 7), (14, 4, 5))):
        x = rng.normal(size=(training, features))
        x[1, 1] = np.nan
        y = np.arange(training) % (index + 2) if task == "classification" else rng.normal(size=training)
        fitted = PreparedDataset.prepare(
            x, y, task=task, n_estimators=2, retain_preprocessing=False,
            random_state=42, class_permutation_size=model.config.outputs,
        )
        transformed = fitted.query_tables(rng.normal(size=(queries, features)))
        for member_index, member in enumerate(fitted.members_by_normalization()):
            items.append(BatchItem(
                (str(index), member_index), fitted, member, transformed,
                output_type="mean" if index == 1 else "raw",
            ))
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield model.eval(), items, torch.device("cpu")
    torch.set_num_threads(previous)


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


@pytest.mark.parametrize("setup", ["classification", "regression"], indirect=True)
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("device_name", ["cpu", "mps"])
def test_ragged_scheduler_fuses_and_matches_individual_members(setup, cache, device_name, monkeypatch):
    model, items, device = setup
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    device = torch.device(device_name)
    model.to(device)
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


def test_cache_packing_preserves_original_storage(setup):
    model, items, _ = setup
    items = with_caches(model, items)
    snapshots = [item.cache.prediction.layers[0].key.clone() for item in items]
    training, _, features = batched._dimensions(items, model.config.group_size)
    packed = batched._pack_cache(items, features // model.config.group_size, training)
    assert packed.prediction.layers[0].key.shape[0] == len(items)
    assert packed.columns[0].layers[0].key.shape[1] == features // model.config.group_size
    for index, item in enumerate(items):
        t, _, f = item.shape
        g = (f + model.config.group_size - 1) // model.config.group_size
        torch.testing.assert_close(packed.prediction.layers[0].key[index, :, :t], snapshots[index][0])
        assert not packed.prediction.layers[0].key[index, :, t:].count_nonzero()
        assert not packed.columns[0].layers[0].key[index, g:].count_nonzero()
    packed.prediction.layers[0].key.zero_()
    for item, snapshot in zip(items, snapshots):
        torch.testing.assert_close(item.cache.prediction.layers[0].key, snapshot)


def test_equal_group_shapes_use_unmasked_fast_path(setup, monkeypatch):
    model, items, device = setup
    pair = items[:2]
    calls = []
    original = ModelRunner.predict

    def observed(self, table, targets, **kwargs):
        calls.append(kwargs)
        return original(self, table, targets, **kwargs)

    monkeypatch.setattr(ModelRunner, "predict", observed)
    collect(model, pair, device)
    assert calls == [{}]


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
