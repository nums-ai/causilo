import pytest
import torch

from causilo.execution import chunks
from causilo.execution.chunks import ChunkRunner
from causilo.execution.memory import Stage, Workload, available_bytes


def test_partition_preserves_shape_and_order(monkeypatch):
    monkeypatch.setattr(chunks, "execution_budget", lambda device: 1)
    value = torch.arange(2 * 3 * 5 * 4).reshape(2, 3, 5, 4).float()
    result = ChunkRunner().run(lambda part: part.sum(-2), value, Workload(Stage.POOL, 5, 4))
    torch.testing.assert_close(result, value.sum(-2), rtol=0, atol=0)


def test_oom_restarts_without_partial_results(monkeypatch):
    monkeypatch.setattr(chunks, "execution_budget", lambda device: 100_000)
    observed = []

    def operation(value):
        observed.append(len(value))
        if len(value) > 2:
            raise torch.cuda.OutOfMemoryError("injected execution failure")
        return value + 3

    value = torch.arange(8 * 5 * 4).reshape(8, 5, 4).float()
    actual = ChunkRunner().run(operation, value, Workload(Stage.POOL, 5, 4))
    torch.testing.assert_close(actual, value + 3)
    assert observed[0] > 2 and observed[-1] <= 2


def test_unrelated_error_is_not_retried():
    def operation(value):
        raise ValueError("semantic failure")

    with pytest.raises(ValueError, match="semantic failure"):
        ChunkRunner().run(operation, torch.ones(2, 3, 4), Workload(Stage.COLUMN, 3, 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel boundary regression")
def test_cuda_attention_handles_more_than_one_grid_of_items():
    value = torch.zeros(65536, 4, 16, device="cuda", dtype=torch.bfloat16)
    sizes = []

    def attention(part):
        sizes.append(len(part))
        heads = part.unsqueeze(1)
        return torch.nn.functional.scaled_dot_product_attention(heads, heads, heads).squeeze(1)

    with torch.inference_mode():
        result = ChunkRunner().run(attention, value, Workload(Stage.POOL, 4, 16))
    torch.testing.assert_close(result, value)
    assert sum(sizes) == len(value)
    assert max(sizes) < len(value)


@pytest.mark.parametrize(
    "free,fraction,expected", [(70, 0.5, 30), (70, 1.0, 80), (10, 1.0, 20), (70, 0.1, 0)]
)
def test_device_budget_respects_allocator_and_physical_limits(monkeypatch, free, fraction, expected):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free, 100))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 20)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 30)
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", lambda device: fraction)
    assert available_bytes(torch.device("cuda:1")) == expected
