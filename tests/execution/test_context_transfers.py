from types import SimpleNamespace

import numpy as np
import pytest
import torch

from causilo import CausiloRegressor
from causilo.v1_5.execution import memory, transfers
from causilo.v1_5.execution.memory import Stage, Workload
from causilo.v1_5.execution.recompute import RecomputeRunner
from causilo.v1_5.nn.cache import BlockCache, KVPair

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def _pair(device, value, *, transposed=False):
    key = torch.full((2, 3, 7, 8), value, device=device, dtype=torch.float16)
    if transposed:
        key = key.transpose(1, 2)
    return KVPair(key, key + 3)


def test_cpu_copy_preserves_views_and_avoids_cuda(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("CPU transfers must not ask CUDA for a stream")

    monkeypatch.setattr(torch.cuda, "Stream", unexpected)
    original = _pair("cpu", 3, transposed=True)
    with transfers.host_cache_copies(torch.device("cpu"), 2**32) as copy:
        result = copy(original)
    assert result.key is original.key and result.value is original.value
    cache = BlockCache({0: original, 1: original})
    assert not transfers.streamable_layers(cache, original.key, None, torch.device("cpu"))


def test_host_memory_respects_parent_limit_and_physical_headroom(tmp_path, monkeypatch):
    membership = tmp_path / "membership"
    membership.write_text("0::/pod/worker")
    root = tmp_path / "sys"
    scope = root / "pod" / "worker"
    scope.mkdir(parents=True)
    (scope / "memory.max").write_text("max")
    (scope.parent / "memory.max").write_text("400")
    (scope.parent / "memory.current").write_text("250")

    def room():
        return memory.host_available_bytes(membership, root, mountinfo=tmp_path / "no-mountinfo")

    physical = SimpleNamespace(available=10000)
    monkeypatch.setattr(memory.psutil, "virtual_memory", lambda: physical)
    assert room() == 150
    physical.available = 80
    assert room() == 80
    (scope.parent / "memory.current").write_text("401")
    assert room() == 0
    membership.write_text("0::/hidden")
    (root / "memory.max").write_text("110")
    (root / "memory.current").write_text("80")
    assert room() == 30
    membership.unlink()
    assert room() == 80


@pytest.mark.parametrize(("version", "member"), [(2, "/work/leaf"), (1, "/pod/work/leaf")])
def test_host_memory_respects_subtree_mounts(tmp_path, monkeypatch, version, member):
    membership = tmp_path / "cgroup"
    membership.write_text(f"5:memory:{member}" if version == 1 else f"0::{member}")
    mount = tmp_path / "memory groups"
    leaf = mount / "work" / "leaf"
    leaf.mkdir(parents=True)
    limit, usage = (
        ("memory.limit_in_bytes", "memory.usage_in_bytes") if version == 1 else ("memory.max", "memory.current")
    )
    for directory, ceiling, used in [(mount, 1000, 100), (leaf.parent, 360, 325), (leaf, 300, 250)]:
        (directory / limit).write_text(str(ceiling))
        (directory / usage).write_text(str(used))
    mountinfo = tmp_path / "mountinfo"
    kind = "cgroup cgroup rw,memory" if version == 1 else "cgroup2 cgroup2 rw"
    encoded_mount = str(mount).replace(" ", r"\040")
    mountinfo.write_text(f"31 29 0:27 /pod {encoded_mount} rw - {kind}")
    monkeypatch.setattr(memory.psutil, "virtual_memory", lambda: SimpleNamespace(available=10000))
    assert memory.host_available_bytes(membership, tmp_path / "default", mountinfo=mountinfo) == 35


@cuda
@torch.inference_mode()
def test_cuda_host_copies_own_storage_and_fallback(monkeypatch):
    device = torch.device("cuda", 0)
    monkeypatch.setattr(transfers, "_MIN_LAYER_BYTES", 1)
    first = _pair(device, 2, transposed=True)
    second = _pair(device, 11, transposed=True)
    caller = torch.cuda.Stream(device=device)
    caller.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(caller):
        with transfers.host_cache_copies(device, 4096) as copy:
            a = copy(first)
        with transfers.host_cache_copies(device, 4096) as copy:
            b = copy(second)
    assert a.key.is_pinned() and b.key.is_pinned() and a.key.data_ptr() != b.key.data_ptr()
    assert a.key.stride() == first.key.stride()
    torch.testing.assert_close(a.key, first.key.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(b.value, second.value.cpu(), atol=0, rtol=0)
    new = torch.empty_like

    def fail_pinning(tensor, **kwargs):
        if kwargs.get("pin_memory"):
            raise torch.cuda.OutOfMemoryError("out of memory while allocating host-pinned pages")
        return new(tensor, **kwargs)

    monkeypatch.setattr(torch, "empty_like", fail_pinning)
    with transfers.host_cache_copies(device, 4096) as copy:
        restored = copy(first)
        other = copy(second)
    assert not restored.key.is_pinned() and not other.key.is_pinned()
    torch.testing.assert_close(restored.value, first.value.cpu(), atol=0, rtol=0)
    monkeypatch.setattr(transfers, "host_available_bytes", lambda: 1)
    with transfers.host_cache_copies(device, 4096) as copy:
        assert not copy(first).key.is_pinned()


@cuda
@torch.inference_mode()
def test_cuda_host_copy_propagates_unrelated_errors(monkeypatch):
    source = _pair("cuda", 7)
    monkeypatch.setattr(transfers, "_MIN_LAYER_BYTES", 1)

    def invalid(*args, **kwargs):
        raise RuntimeError("device-side assertion")

    monkeypatch.setattr(torch, "empty_like", invalid)
    with pytest.raises(RuntimeError, match="device-side assertion"):
        with transfers.host_cache_copies(torch.device("cuda", 0), 4096) as copy:
            copy(source)


@cuda
@torch.inference_mode()
@pytest.mark.parametrize("pinned", [False, True])
def test_cuda_alternating_layers_on_caller_stream_are_exact(monkeypatch, pinned):
    device = torch.device("cuda", 0)
    monkeypatch.setattr(transfers, "_MIN_LAYER_BYTES", 1)
    saved = [_pair("cpu", i, transposed=True) for i in range(5)]
    if pinned:
        saved = [KVPair(p.key.pin_memory(), p.value.pin_memory()) for p in saved]
    cache = BlockCache(dict(enumerate(saved)))
    rows = torch.zeros((2, 4, 8), device=device)
    work = Workload(Stage.PREDICTION, 4, 8, 7, output_elements=4)
    assert transfers.streamable_layers(cache, rows, work, device)
    caller = torch.cuda.Stream(device=device)
    caller.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(caller):
        with transfers.device_cache_layers(cache, device) as layers:
            result = []
            pointers = []
            for layer in layers:
                pointers.append(layer.key.data_ptr())
                result.append((layer.key + layer.value).clone())
    caller.synchronize()
    assert len(set(pointers)) == 2
    for output, source in zip(result, saved, strict=True):
        torch.testing.assert_close(output.cpu(), source.key + source.value, rtol=0, atol=0, check_stride=True)
    monkeypatch.setattr(transfers, "execution_budget", lambda device: 1)
    assert not transfers.streamable_layers(cache, rows, work, device)


@cuda
@torch.inference_mode()
def test_cuda_optional_slot_allocation_falls_back_before_attention(monkeypatch):
    cache = BlockCache({0: _pair("cpu", 2), 1: _pair("cpu", 9)})

    def fail(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("out of memory")

    monkeypatch.setattr(torch, "empty_like", fail)
    with transfers.device_cache_layers(cache, torch.device("cuda", 0)) as layers:
        assert layers is None


@cuda
def test_two_fits_keep_independent_weighted_host_caches(checkpoint15, mixed_data, monkeypatch):
    table, query = mixed_data
    targets = table["a"].to_numpy()
    options = dict(device="cuda", n_estimators=1, checkpoint_path=checkpoint15("regression"), use_kv_cache=True)
    counts = 1 + np.arange(len(table)) % 3
    monkeypatch.setattr(RecomputeRunner, "needed", lambda self, shape: True)
    monkeypatch.setattr(transfers, "_MIN_LAYER_BYTES", 1 << 62)
    expected = CausiloRegressor(**options).fit(table, targets, sample_count=counts).predict(query)
    monkeypatch.setattr(transfers, "_MIN_LAYER_BYTES", 1)
    first = CausiloRegressor(**options).fit(table, targets, sample_count=counts)
    second = CausiloRegressor(**options).fit(table, targets[::-1].copy(), sample_count=counts)
    a, b = [m._engine.require_state().caches[0].prediction.layers[0].key for m in (first, second)]
    assert a.is_pinned() and b.is_pinned() and a.data_ptr() != b.data_ptr()
    observed = first.predict(query)
    assert observed.dtype == expected.dtype and observed.tobytes() == expected.tobytes()
