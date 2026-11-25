from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from causilo.v1_5.execution import chunks, memory, recompute
from causilo.v1_5.execution.direct import EnsembleRunner, predict_chunks
from causilo.v1_5.execution.memory import Stage, Workload
from causilo.v1_5.execution.runner import ModelRunner
from causilo.v1_5.nn.cache import BlockCache, KVPair
from causilo.v1_5.nn.model_config import ModelConfig


def test_query_retry_keeps_completed_rows_and_retries_the_failed_offset():
    expected = np.arange(11)
    attempts = []

    def predict(section):
        attempts.append((section.start, section.stop))
        if (section.start, section.stop) == (4, 8):
            raise MemoryError("simulated allocation failure")
        return expected[section]

    parts = predict_chunks(predict, rows=len(expected), batch_size=4, device=torch.device("cpu"))
    np.testing.assert_array_equal(np.concatenate(list(parts)), expected)
    assert attempts == [(0, 4), (4, 8), (4, 6), (6, 8), (8, 10), (10, 11)]


def test_query_retry_uses_the_length_of_the_last_chunk():
    attempts = []

    def predict(section):
        attempts.append((section.start, section.stop))
        if section.start >= 8:
            raise MemoryError("failed allocating the last rows")
        return np.arange(section.start, section.stop)

    with pytest.raises(MemoryError, match="last rows"):
        list(predict_chunks(predict, rows=11, batch_size=8, device=torch.device("cpu")))
    assert attempts == [(0, 8), (8, 11), (8, 9)]


def test_concurrent_host_spills_share_headroom_and_release_on_error(monkeypatch):
    monkeypatch.setattr(memory, "host_available_bytes", lambda: 100)

    def other_lane():
        with memory.reserve_host_spill(20):
            pytest.fail("Both destinations exceed the budget")

    with pytest.raises(ValueError, match="stage failed"):
        with memory.reserve_host_spill(64), ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(MemoryError, match="host memory"):
                pool.submit(other_lane).result()
            raise ValueError("stage failed")
    with memory.reserve_host_spill(80):
        pass


@pytest.mark.parametrize(
    ("num_classes", "budget", "reuse_context", "failure", "expected"),
    [
        (10, "full", False, None, ["direct"]),
        (10, "tile", False, None, ["device_tiles"]),
        (0, "below_tile", False, None, ["replay"]),
        (0, "tile", True, None, ["replay"]),
        (10, "full", False, memory.HostSpillMemoryError, ["direct", "replay"]),
        (0, "tile", False, torch.cuda.OutOfMemoryError, ["device_tiles", "replay"]),
    ],
    ids=["dense", "device-tiles", "replay", "retained-context", "host-full", "cuda-oom"],
)
def test_model_predict_selects_gpu_route_before_spilling(
    monkeypatch, num_classes, budget, reuse_context, failure, expected,
):
    runner = ModelRunner.__new__(ModelRunner)
    runner.device = torch.device("cuda")
    runner.model = SimpleNamespace(
        config=ModelConfig(width=8), parameters=lambda: iter([SimpleNamespace(device=runner.device)]),
    )
    table = SimpleNamespace(shape=(1, 12, 4), to=lambda device: table)
    targets = SimpleNamespace(to=lambda device: targets)
    contexts = {} if reuse_context else None
    weights = object()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    full = memory.estimate_table_workspace_bytes(runner.model, table.shape, runner.device)
    tile = memory.minimum_direct_workspace_bytes(runner.model, table.shape, runner.device)
    assert 0 < tile < full
    gpu_bytes = {"full": full, "tile": tile, "below_tile": tile - 1}[budget]
    monkeypatch.setattr(memory, "host_available_bytes", lambda: full * 2)
    monkeypatch.setattr(recompute.RecomputeRunner, "budget", lambda self: gpu_bytes)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    calls = []

    def predict_direct(x, y, **kwargs):
        calls.append("device_tiles" if chunks._tiled_direct_outputs.get() else "direct")
        assert x is table and y is targets
        assert kwargs == {"num_classes": num_classes, "weights": weights}
        if failure is not None:
            raise failure("simulated allocation failure")
        return table

    def run_replay(self, x, y, **kwargs):
        calls.append("replay")
        assert x is table and y is targets
        options = {"num_classes": num_classes, "weights": weights}
        if contexts is not None:
            options.update(contexts=contexts, first_member=3)
        assert kwargs == options
        if contexts is not None:
            assert kwargs["contexts"] is contexts
        return table

    monkeypatch.setattr(runner, "_predict_direct", predict_direct)
    monkeypatch.setattr(recompute.RecomputeRunner, "__call__", run_replay)
    assert runner.predict(
        table, targets, num_classes=num_classes, weights=weights, contexts=contexts, first_member=3,
    ) is table
    assert calls == expected


@pytest.mark.parametrize("try_gpu", [False, True])
def test_tiled_host_spill_checks_budget_before_allocating(monkeypatch, try_gpu):
    runner = chunks.ChunkRunner(torch.device("cuda"))
    monkeypatch.setattr(memory, "host_available_bytes", lambda: 64)
    calls = []

    def allocate(shape, *, dtype, device, **kwargs):
        calls.append(torch.device(device).type)
        if calls[-1] == "cuda":
            raise torch.cuda.OutOfMemoryError("cannot allocate on device")
        pytest.fail("Attempted a host allocation larger than the budget")

    monkeypatch.setattr(torch, "empty", allocate)
    with ExitStack() as reservations, pytest.raises(MemoryError, match="host memory"):
        runner._allocate_output_buffer((8, 2), torch.float32, try_gpu, reservations)
    assert calls == (["cuda"] if try_gpu else [])


def test_whole_stage_spill_retries_smaller_queries_before_copying(monkeypatch):
    runner = chunks.ChunkRunner(torch.device("cuda"))
    monkeypatch.setattr(memory, "host_available_bytes", lambda: 40)
    monkeypatch.setattr(chunks.ChunkPlan, "choose", lambda *args: SimpleNamespace(limit=1, keep_on_device=False))
    monkeypatch.setattr(chunks, "run_inference", lambda device, operation, part, section, **kw: operation(part, section))
    copied = []
    attempted = []

    def predict(section):
        size = section.stop - section.start
        attempted.append(size)
        rows = SimpleNamespace(shape=(1, size, 1), to=lambda *a, **kw: section)

        def output(part, _section):
            def to_cpu():
                copied.append(size)
                return np.arange(part.start, part.stop)

            return SimpleNamespace(nbytes=8 * size, device=runner.device, cpu=to_cpu)

        return runner.run_indexed(output, rows, Workload(Stage.ROW, size, 1))

    parts = list(predict_chunks(predict, rows=8, batch_size=8, device=torch.device("cpu")))
    np.testing.assert_array_equal(np.concatenate(parts), np.arange(8))
    assert attempted == [8, 4, 4] and copied == [4, 4]


def test_tiled_cache_retry_preserves_row_and_projection_alignment(monkeypatch):
    rows = torch.arange(24).reshape(3, 8, 1).transpose(0, 1).float()
    workload = Workload(Stage.ROW, 3, 1, cache_elements=6)
    budget = workload.bytes_per_item(torch.device("cpu")) * 3 + workload.destination_bytes(8)
    monkeypatch.setattr(chunks, "execution_budget", lambda device: budget)
    attempted = []

    def prefill(part, section):
        attempted.append((section.start, section.stop))
        if len(attempted) == 2:
            raise torch.cuda.OutOfMemoryError("simulated allocation failure after first cache partition")
        return part + 1, BlockCache({0: KVPair(part.unsqueeze(1), (part * 2).unsqueeze(1))})

    runner = chunks.CacheRunner(torch.device("cpu"))
    result, cache = runner.build(prefill, rows, workload)
    assert attempted[:3] == [(0, 3), (3, 6), (0, 1)]
    torch.testing.assert_close(result, rows + 1)
    prediction = runner.query(
        lambda part, keys: part + keys.layers[0].key.squeeze(1) + 2 * keys.layers[0].value.squeeze(1),
        rows, cache, workload,
    )
    torch.testing.assert_close(prediction, 6 * rows)


def test_stage_retry_reraises_original_oom_at_minimum_partition(monkeypatch):
    monkeypatch.setattr(chunks, "execution_budget", lambda device: 0)
    error = torch.cuda.OutOfMemoryError("simulated allocation failure")
    attempted = []

    def forward(part):
        attempted.append(len(part))
        raise error

    with pytest.raises(torch.cuda.OutOfMemoryError) as caught:
        chunks.ChunkRunner(torch.device("cpu")).run(forward, torch.empty(2, 3, 1), Workload(Stage.COLUMN, 3, 1))
    assert caught.value is error
    assert attempted == [1]


def test_oom_reduces_only_the_current_execution_plan(monkeypatch):
    monkeypatch.setattr(chunks, "execution_budget", lambda device: 1024**3)
    runner = chunks.ChunkRunner(torch.device("cpu"))

    def run(rows, capacity):
        expected = torch.arange(rows * 3).reshape(rows, 3, 1)
        attempted = []

        def forward(part):
            attempted.append(len(part))
            if len(part) > capacity:
                raise torch.cuda.OutOfMemoryError("simulated allocation failure")
            return part

        result = runner.run(forward, expected, Workload(Stage.COLUMN, 3, 1))
        torch.testing.assert_close(result, expected)
        return attempted

    assert run(8, 2) == [8, 4, 2, 2, 2, 2]
    assert run(8, 8) == [8]


@pytest.mark.parametrize(("max_classes", "cached"), [(10, True), (10, False), (0, False)])
def test_cuda_query_limits_follow_workspace_and_replay_budget(monkeypatch, max_classes, cached):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (10**9, 10**9))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", lambda device: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    model = SimpleNamespace(config=ModelConfig(max_classes=max_classes, width=16))
    dataset = SimpleNamespace(training=np.empty((4, 4)), ensemble=SimpleNamespace(feature_signs=np.empty((1, 4))))
    lane = EnsembleRunner.__new__(EnsembleRunner)
    lane.model, lane.device = model, torch.device("cuda:1")
    sizes = []
    for limit in (1024**2, 2 * 1024**2):
        lane.memory_budget_bytes = limit
        sizes.append(lane.query_device_limit(dataset, 100_000, cached=cached))
    assert 1 < sizes[0] < sizes[1] <= 100_000
