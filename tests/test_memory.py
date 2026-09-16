from dataclasses import replace

import pytest
import torch

from causilo.execution import memory
from causilo.execution.memory import RecomputeMemory, attention_workspace, embedding_workspace, tile_size
from causilo.model import ModelConfig


@pytest.fixture
def config():
    return ModelConfig(
        task="regression", width=16, expansion=2, group_size=3, frequencies=4,
        column_latents=5, column_heads=2, column_depths=(2, 2),
        row_heads=2, row_latents=2, row_depths=(3, 2),
        prediction_heads=2, prediction_depth=3, outputs=9,
    )


def test_tile_size_reserves_pending_buffers_without_fixed_workspace_or_row_caps(monkeypatch):
    gib, mib = 1024**3, 1024**2
    monkeypatch.setattr(memory, "execution_budget", lambda device: 64 * gib)
    device = torch.device("cuda:0")

    limit = tile_size(1000000, mib, device, pending_bytes=4 * gib)

    assert limit * mib + 4 * gib <= 64 * gib
    assert (limit + 1) * mib + 4 * gib > 64 * gib
    assert limit > 16384
    assert tile_size(100, mib, device, kernel_limited=True) == 100  # No four-group cap.


@pytest.mark.parametrize("device, limited", [("cpu", False), ("cuda:0", True)])
def test_kernel_limit_only_bounds_independent_cuda_batches(monkeypatch, device, limited):
    monkeypatch.setattr(memory, "execution_budget", lambda device: 1024**4)
    monkeypatch.setattr(memory, "CUDA_MAX_BATCH_ITEMS", 8)
    device = torch.device(device)

    assert tile_size(100, 1, device, kernel_limited=True) == (8 if limited else 100)
    assert tile_size(100, 1, device, kernel_limited=False) == 100


def test_tile_size_uses_current_free_budget_and_only_reserves_future_buffers(monkeypatch):
    free = 1000
    monkeypatch.setattr(memory, "execution_budget", lambda device: free)
    device = torch.device("cpu")
    assert tile_size(100, 10, device, pending_bytes=200) == 80
    # The buffer is now allocated, so the free budget already reflects it.
    free -= 200
    assert tile_size(100, 10, device) == 80
    # A subsequently allocated context reduces the next layer's tile size.
    free -= 300
    assert tile_size(100, 10, device) == 50


def test_retained_buffers_that_exceed_budget_fail_without_offloading(monkeypatch):
    monkeypatch.setattr(memory, "execution_budget", lambda device: 100)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="Retained recomputation states"):
        tile_size(100, 1, torch.device("cuda:0"), pending_bytes=101)


def test_workspaces_respond_to_architecture_dimensions(config):
    device = torch.device("cpu")
    usual = RecomputeMemory(config, device)
    expanded = RecomputeMemory(replace(config, expansion=16), device)
    more_latents = RecomputeMemory(replace(config, row_latents=4), device)
    deeper = RecomputeMemory(replace(config, row_depths=(5, 2)), device)

    assert expanded.row_workspace(10) > usual.row_workspace(10)
    assert expanded.prediction_workspace(1, final=False) > usual.prediction_workspace(1, final=False)
    assert more_latents.row_bytes(1000) == 2 * usual.row_bytes(1000)
    assert deeper.trace_bytes(1000) == 2 * usual.trace_bytes(1000)
    assert embedding_workspace(replace(config, frequencies=64)) > embedding_workspace(config)


def test_dense_attention_accounts_for_context_length_and_head_count():
    device = torch.device("cpu")
    usual = attention_workspace(100, 1000, 16, 2, 2, device)
    longer = attention_workspace(100, 2000, 16, 2, 2, device)
    more_heads = attention_workspace(100, 1000, 16, 4, 2, device)
    assert longer > usual and more_heads > usual


def test_final_head_outputs_and_summary_depth_are_budgeted(config):
    device = torch.device("cpu")
    small = RecomputeMemory(config, device)
    large = RecomputeMemory(replace(config, outputs=10000), device)
    assert large.prediction_workspace(1, final=True) > small.prediction_workspace(1, final=True)
    assert small.summary_bytes(8, 4) == 2 * small.summary_bytes(8, 2)
