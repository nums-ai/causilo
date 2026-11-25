import pytest
import torch

from causilo import CausiloRegressor
from causilo.v1_5.execution import recompute
from causilo.v1_5.nn.cache import KVPair
from causilo.v1_5.nn.ecdf import ecdf_ranks, fit_ecdf_context, fit_weighted_ecdf
from causilo.v1_5.nn.frequency import FrequencyWeights
from causilo.v1_5.nn.model import Model
from causilo.v1_5.nn.model_config import ModelConfig


def _replay_runner(monkeypatch):
    runner = recompute.RecomputeRunner.__new__(recompute.RecomputeRunner)
    runner.device = torch.device("cpu")
    runner.summaries = []
    runner._tile_retried = False
    runner._prepare(torch.arange(30).view(1, 10, 3).float(), torch.zeros(1, 8), None)
    room = [500]
    calls = []
    monkeypatch.setattr(runner, "budget", lambda: room[0])

    def encode():
        calls.append(runner._limit(1000, 2, 100))
        return torch.arange(4)

    monkeypatch.setattr(runner, "_encode_context", encode)
    return runner, room, calls


def test_replay_keeps_member_context_across_tile_changes_until_snapshot_exceeds_budget(monkeypatch):
    runner, room, _ = _replay_runner(monkeypatch)
    calls = []

    def encode():
        calls.append(1)
        runner.summaries = [KVPair(torch.zeros(8), torch.zeros(8))]
        return torch.arange(4) + len(calls)

    monkeypatch.setattr(runner, "_encode_context", encode)
    contexts = {}
    first = runner._reuse_context(contexts, 2)
    second = runner._reuse_context(contexts, 5)
    assert not torch.equal(first, second)
    initial_tile = runner._limit(1000, 2, 100)
    room[0] = 700
    assert runner._limit(1000, 2, 100) > initial_tile
    torch.testing.assert_close(runner._reuse_context(contexts, 2), first)
    room[0] = 96  # Four int64 row values and two eight-element float32 projections.
    torch.testing.assert_close(runner._reuse_context(contexts, 5), second)
    assert len(calls) == 2
    room[0] -= 1
    torch.testing.assert_close(runner._reuse_context(contexts, 2), torch.arange(4) + 3)
    assert contexts[2] is None


@pytest.mark.parametrize("operation", ["__init__", "restore"])
def test_optional_replay_allocation_failure_preserves_native_rows(monkeypatch, operation):
    runner, _, calls = _replay_runner(monkeypatch)
    contexts = {}
    failures = []

    def cannot_copy(*args):
        failures.append(operation)
        raise MemoryError("only the optional snapshot copy failed")

    monkeypatch.setattr(recompute.ReplayContext, operation, cannot_copy)
    for _ in range(3):
        torch.testing.assert_close(runner._reuse_context(contexts, 0), torch.arange(4))
    assert len(calls) == 3 and failures == [operation]
    assert contexts == {0: None}


def test_replay_does_not_keep_context_from_a_retried_tile(monkeypatch):
    runner, _, _ = _replay_runner(monkeypatch)
    attempts = []

    def project(section):
        attempts.append(section)
        if len(attempts) == 1:
            raise torch.cuda.OutOfMemoryError("out of memory during the original tile")
        return torch.arange(section.start, section.stop)

    monkeypatch.setattr(runner, "_encode_context", lambda: runner._map(4, 1, project, pending=0, axis=0))
    contexts = {}
    torch.testing.assert_close(runner._reuse_context(contexts, 0), torch.arange(4))
    assert contexts == {0: None}


def test_weighted_replay_ranks_only_new_query_rows(checkpoint15, mixed_data, monkeypatch):
    table, _ = mixed_data
    model = CausiloRegressor(device="cpu", checkpoint_path=checkpoint15("regression")).fit(table, table["a"])
    runner = recompute.RecomputeRunner(model._engine.model)
    monkeypatch.setattr(runner, "budget", lambda: 1 << 28)
    data = torch.arange(56).reshape(1, 14, 4).float().sin()
    data[:, 1::3, 0] = float("nan")
    data[:, :, 2] = float("nan")
    data[:, 4] = data[:, 5]
    targets = torch.arange(9).reshape(1, 9).float()
    weights = FrequencyWeights.prepare(torch.arange(9) % 3 + 1)
    queries = [data[:, 9:12], data[:, 12:]]
    inputs = [torch.cat((data[:, :9], q), dim=1) for q in queries]
    with torch.inference_mode():
        expected = [runner(x, targets, num_classes=0, weights=weights) for x in inputs]

    sorted_rows, ranked_rows = [], []
    fit, rank = recompute.fit_column_context, recompute.ecdf_ranks

    def track_fit(values, context_rows, weights):
        sorted_rows.append(context_rows)
        return fit(values, context_rows, weights)

    def track_rank(values, context):
        ranked_rows.append(values.shape[-2])
        return rank(values, context)

    monkeypatch.setattr(recompute, "fit_column_context", track_fit)
    monkeypatch.setattr(recompute, "ecdf_ranks", track_rank)
    snapshots = {}
    with torch.inference_mode():
        actual = [runner(x, targets, num_classes=0, weights=weights, contexts=snapshots) for x in inputs]
    assert sorted_rows == [9] and ranked_rows == [12, 2]
    for observed, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, reference, atol=0, rtol=0)


@pytest.mark.parametrize("classes", [0, 10])
def test_bounded_target_embedding_skips_query_only_tiles(classes, monkeypatch):
    config = ModelConfig(width=16, column_depth=1, prediction_depth=1, max_classes=classes)
    runner = recompute.RecomputeRunner(Model(config).eval())
    runner.targets = torch.arange(5).reshape(1, 5).float()
    rows = torch.randn(1, 12, config.prediction_width)
    before = rows.clone()
    monkeypatch.setattr(runner, "_limit", lambda *args: 4)
    block = runner.model.prediction
    with torch.inference_mode():
        expected = torch.cat([
            block.add_context_targets(rows[:, start:start + 4], runner.targets[:, start:start + 4])
            for start in range(0, 12, 4)
        ], dim=1)
    labels = []
    with block.target.register_forward_pre_hook(lambda module, args: labels.append(args[0].shape[-1])):
        observed = runner._add_targets(rows)
    assert labels == [4, 1]
    assert torch.equal(expected.view(torch.uint8), observed.view(torch.uint8))
    assert torch.equal(rows.view(torch.uint8), before.view(torch.uint8))


@pytest.mark.parametrize(("context_rows", "weighted", "dtype"), [
    (0, False, torch.float32), (0, True, torch.float32), (1, False, torch.float64),
    (5, False, torch.float64), (5, True, torch.float32),
])
def test_ecdf_midrank_handles_empty_context_and_nonfinite_values(context_rows, weighted, dtype):
    table = torch.tensor([
        [-float("inf"), float("nan"), 1], [2, float("inf"), 1], [-1, -float("inf"), 3],
        [2, float("nan"), 5], [float("inf"), float("nan"), 3],
    ], dtype=dtype).unsqueeze(0)
    query = torch.tensor([
        [-4, 3, 1], [2 + 1e-8, float("nan"), 3], [8, float("inf"), 4], [float("nan"), 0, float("inf")],
    ], dtype=dtype).unsqueeze(0)
    counts = torch.tensor([1, 2, 1, 3, 2])[:context_rows]
    reference = table[:, :context_rows]
    if weighted:
        context = fit_weighted_ecdf(table, counts)
        reference = reference.repeat_interleave(counts, dim=1)
    else:
        context = fit_ecdf_context(table, context_rows)

    keys = reference.unsqueeze(1)
    valid = keys.isfinite()
    denominator = valid.sum(dim=2)
    less = ((keys < query.unsqueeze(2)) & valid).sum(dim=2)
    equal = ((keys == query.unsqueeze(2)) & valid).sum(dim=2)
    expected = (less.to(dtype) + 0.5 * equal) / denominator.clamp_min(1)
    expected = expected.masked_fill((denominator == 0) | ~query.isfinite(), float("nan"))
    actual = ecdf_ranks(query, context)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
