"""Frequency-weighted attention must equal explicitly repeated context rows."""

import pytest
import torch

from causilo.execution.chunks import ChunkPlan
from causilo.execution.recompute import RecomputeRunner
from causilo.execution.runner import ModelRunner
from causilo.model import Model, ModelConfig
from causilo.nn.cache import FrequencyWeights, KVPair
from causilo.nn.layers.attention import Attention


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(params=["classification", "regression"])
def model(request):
    with torch.random.fork_rng():
        torch.manual_seed(18)
        result = Model(ModelConfig(
            task=request.param, width=16, expansion=2, group_size=3, frequencies=4,
            column_latents=5, column_heads=2, column_depths=(2, 2),
            row_heads=2, row_latents=2, row_depths=(3, 2),
            prediction_heads=2, prediction_depth=3, outputs=3,
        ))
        for name, parameter in result.named_parameters():
            if "latents" in name or name.endswith("frequencies") or name.endswith("missing"):
                torch.nn.init.normal_(parameter, std=0.1)
    return result.eval()


def inputs(members=1):
    generator = torch.Generator().manual_seed(93)
    table = torch.randn(members, 9, 7, generator=generator)
    table[:, 2, 1] = float("nan")
    targets = torch.arange(6).remainder(3).float().expand(members, -1)
    frequencies = torch.tensor([1, 5, 2, 4, 8, 3])
    expanded = torch.cat((table[:, :6].repeat_interleave(frequencies, dim=1), table[:, 6:]), dim=1)
    expanded_targets = targets.repeat_interleave(frequencies, dim=1)
    return table, targets, frequencies, expanded, expanded_targets


@pytest.mark.parametrize("members", [1, 4, 8])
@pytest.mark.parametrize("cached", [False, True])
def test_weighted_network_matches_expanded_context(model, members, cached):
    table, targets, frequencies, expanded, expanded_targets = inputs(members)
    runner = ModelRunner(model)
    with torch.inference_mode():
        expected = runner.predict(expanded, expanded_targets)
        if cached:
            state = runner.build_cache(table[:, :6], targets, frequencies=frequencies)
            actual = runner.predict_cached(table[:, 6:], state, frequencies=frequencies)
            assert state.prediction.layers[0].key.shape[-2] == len(frequencies)
        else:
            actual = runner.predict(table, targets, frequencies=frequencies)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("tile", [1, 4, 50])
def test_weighted_recompute_matches_expanded_context(model, monkeypatch, tile):
    table, targets, frequencies, expanded, expanded_targets = inputs()
    monkeypatch.setattr(RecomputeRunner, "_limit", lambda self, items, *a, **kw: min(items, tile))
    with torch.inference_mode():
        expected = ModelRunner(model).predict(expanded, expanded_targets)
        actual = RecomputeRunner(model, torch.device("cpu")).predict(
            table, targets, frequencies=frequencies,
        )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_weighted_column_summary_retry_keeps_frequencies(model, monkeypatch):
    table, targets, frequencies, expanded, expanded_targets = inputs()
    layer = model.columns[0].layers[0]
    original = layer.prepare
    attempts = []

    def fail_large_group(value, weights=None):
        attempts.append(value.shape[1])
        assert weights.count == frequencies.sum().item()
        if value.shape[1] > 1:
            raise torch.cuda.OutOfMemoryError("injected weighted column failure")
        return original(value, weights=weights)

    with torch.inference_mode():
        expected = ModelRunner(model).predict(expanded, expanded_targets)
        monkeypatch.setattr(layer, "prepare", fail_large_group)
        actual = RecomputeRunner(model, torch.device("cpu")).predict(
            table, targets, frequencies=frequencies,
        )
    assert attempts[0] > 1 and attempts[-1] == 1
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_large_counts_are_logged_before_half_precision_conversion():
    with torch.random.fork_rng():
        torch.manual_seed(7)
        attention = Attention(8, 2, scale_query=False).half().eval()
        values = torch.randn(1, 2, 2, 4).half()
    frequencies = torch.tensor([100_000, 1])
    weights = FrequencyWeights.prepare(frequencies, torch.device("cpu"))
    query = torch.zeros(1, 2, 3, 4, dtype=torch.float16)
    keys = KVPair(torch.zeros_like(values), values)
    with torch.inference_mode():
        actual = attention.attend(query, keys, weights=weights)
        mean = (values.float() * frequencies[None, None, :, None]).sum(-2) / frequencies.sum()
        expected = attention.output(mean.flatten(-2).half()[:, None].expand(-1, 3, -1))
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


def test_weighted_attention_does_not_mutate_model_or_unweighted_execution(model):
    table, targets, frequencies, _, _ = inputs()
    runner = ModelRunner(model)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.inference_mode():
        expected = runner.predict(table, targets)
        runner.predict(table, targets, frequencies=frequencies)
        state = runner.build_cache(table[:, :6], targets, frequencies=frequencies)
        runner.predict_cached(table[:, 6:], state, frequencies=frequencies)
        actual = runner.predict(table, targets)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


@pytest.mark.parametrize("device_name", ["cuda", "mps"])
def test_accelerator_weighted_direct_cache_and_recompute(model, monkeypatch, device_name):
    available = torch.cuda.is_available() if device_name == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{device_name.upper()} frequency-weighted attention")
    table, targets, frequencies, expanded, expanded_targets = inputs()
    device = torch.device(device_name)
    runner = ModelRunner(model.to(device))
    monkeypatch.setattr(RecomputeRunner, "_limit", lambda self, items, *a, **kw: min(items, 3))
    with torch.inference_mode():
        expected = runner.predict(expanded.to(device), expanded_targets.to(device))
        actual = runner.predict(table.to(device), targets.to(device), frequencies=frequencies.to(device))
        state = runner.build_cache(table[:, :6].to(device), targets.to(device), frequencies=frequencies)
        cached = runner.predict_cached(table[:, 6:].to(device), state, frequencies=frequencies)
        replayed = RecomputeRunner(model, device).predict(table, targets, frequencies=frequencies)
    for result in (actual, cached, replayed.to(device)):
        torch.testing.assert_close(result, expected, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("cached", [False, True])
def test_weighted_member_and_feature_partitioning(model, monkeypatch, cached):
    table, targets, frequencies, expanded, expanded_targets = inputs(4)
    runner = ModelRunner(model)
    original = ChunkPlan.choose

    def small_partitions(cls, *args, **kwargs):
        plan = original(*args, **kwargs)
        plan.limit = min(plan.limit, 2)
        return plan

    with torch.inference_mode():
        expected = runner.predict(expanded, expanded_targets)
        monkeypatch.setattr(ChunkPlan, "choose", classmethod(small_partitions))
        if cached:
            state = runner.build_cache(table[:, :6], targets, frequencies=frequencies)
            actual = runner.predict_cached(table[:, 6:], state, frequencies=frequencies)
        else:
            actual = runner.predict(table, targets, frequencies=frequencies)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_weighted_cache_build_retry_keeps_frequencies(model, monkeypatch):
    table, targets, frequencies, expanded, expanded_targets = inputs()
    runner = ModelRunner(model)
    column = model.columns[0]
    original = column.build_context
    attempts = []

    def fail_large_batch(rows, weights=None):
        attempts.append(rows.shape[0])
        assert weights.count == frequencies.sum().item()
        if rows.shape[0] > 2:
            raise torch.cuda.OutOfMemoryError("injected weighted cache failure")
        return original(rows, weights=weights)

    with torch.inference_mode():
        expected = runner.predict(expanded, expanded_targets)
        monkeypatch.setattr(column, "build_context", fail_large_batch)
        state = runner.build_cache(table[:, :6], targets, frequencies=frequencies)
        actual = runner.predict_cached(table[:, 6:], state, frequencies=frequencies)
    assert attempts[0] > 2 and attempts[-1] <= 2
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
