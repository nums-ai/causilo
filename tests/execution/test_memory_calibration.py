"""Compare the 1.5 workspace estimates with allocations measured on a CUDA device."""

import pytest
import torch

from causilo.v1_5.execution import chunks, memory, recompute
from causilo.v1_5.execution.runner import ModelRunner
from causilo.v1_5.nn.model import Model
from causilo.v1_5.nn.model_config import ModelConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None, reason="Requires NVIDIA CUDA",
)


@pytest.fixture
def runner():
    """Stage workspaces follow the released widths; depth only repeats them."""
    device = torch.device("cuda:0")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = Model(ModelConfig(max_classes=0, column_depth=2, prediction_depth=2))
        for parameter in model.parameters():
            torch.nn.init.normal_(parameter, std=0.05)
    return ModelRunner(model.eval().to(device), device)


def make_table(members, context_rows, query_rows, features):
    generator = torch.Generator().manual_seed(11)
    table = torch.randn(members, context_rows + query_rows, features, generator=generator)
    return table, torch.randn(members, context_rows, generator=generator)


def test_stage_estimates_bound_measured_workspaces(runner, monkeypatch):
    """Every untiled stage must fit its estimate; a large excess marks a stale formula."""
    device = runner.device
    measured = []
    run_indexed = chunks.ChunkRunner.run_indexed

    def measure(self, operation, rows, workload, **options):
        items = rows.shape[:-2].numel()
        torch.cuda.synchronize(device)
        resident = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        output = run_indexed(self, operation, rows, workload, **options)
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) - resident
        measured.append((workload.stage, items * workload.bytes_per_item(device), peak))
        return output

    monkeypatch.setattr(chunks.ChunkRunner, "run_indexed", measure)
    monkeypatch.setattr(memory, "available_bytes", lambda device: 2**50)
    table, targets = make_table(2, 5000, 1000, 24)
    with torch.cuda.device(device), torch.inference_mode():
        runner._predict_direct(table.to(device), targets.to(device), num_classes=0, weights=None)

    assert {stage for stage, _, _ in measured} == set(memory.Stage)
    for stage, estimate, peak in measured:
        assert peak <= estimate <= 5 * peak, (stage, estimate, peak)


def test_tiled_direct_pass_fits_its_minimum_workspace(runner, monkeypatch):
    """At its estimated floor, a direct pass keeps every stage on the device within the budget."""
    device = runner.device
    table, targets = make_table(1, 10_000, 2000, 50)
    shape = tuple(table.shape)
    plans, replays = [], []
    choose = chunks.ChunkPlan.choose.__func__

    def record(cls, items, workload, device):
        plan = choose(cls, items, workload, device)
        plans.append((workload.stage, plan.limit < items, plan.keep_on_device))
        return plan

    monkeypatch.setattr(chunks.ChunkPlan, "choose", classmethod(record))
    monkeypatch.setattr(
        recompute.RecomputeRunner, "__call__", lambda self, *args, **options: replays.append(args),
    )
    with torch.cuda.device(device), torch.inference_mode():
        torch.cuda.empty_cache()
        minimum = memory.minimum_direct_workspace_bytes(runner.model, shape, device)
        limit = torch.cuda.memory_allocated(device) + int(1.02 * minimum / memory.EXECUTION_MEMORY_FRACTION)
        if limit > memory.available_bytes(device):
            pytest.skip("The CUDA device has less free memory than this plan needs")
        with memory.planning_budget(device, limit):
            assert memory.estimate_table_workspace_bytes(runner.model, shape, device) > memory.execution_budget(device)
            torch.cuda.reset_peak_memory_stats(device)
            output = runner.predict(table, targets, num_classes=0)
            torch.cuda.synchronize(device)

    assert not replays and output.shape[:2] == (1, 2000)
    assert any(tiled for _, tiled, _ in plans)
    assert all(keep_on_device for _, _, keep_on_device in plans)
    assert torch.cuda.max_memory_allocated(device) <= limit
