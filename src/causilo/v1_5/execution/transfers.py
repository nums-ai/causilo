"""Transfer large fitted K/V layers without changing attention batches."""

from contextlib import contextmanager
from threading import Lock

import torch

from ..nn.cache import KVPair
from .memory import execution_budget, host_available_bytes, is_allocation_error

_MIN_LAYER_BYTES = 64 << 20
_host_allocation = Lock()


def _pinning_fits(payload_bytes):
    # Locked-page allocations round up to blocks. Reserve headroom for other users of host RAM.
    registered = torch.cuda.memory.host_memory_stats().get("allocated_bytes.current")
    return registered is not None and registered + 2 * payload_bytes <= host_available_bytes() // 4


def _pinned_pair(pair):
    # Concurrent fits own independent destinations and share only this allocator budget check.
    with _host_allocation:
        if _pinning_fits(pair.key.nbytes + pair.value.nbytes):
            try:
                return KVPair(
                    torch.empty_like(pair.key, device="cpu", pin_memory=True),
                    torch.empty_like(pair.value, device="cpu", pin_memory=True),
                )
            except (MemoryError, RuntimeError) as error:
                if not is_allocation_error(error) and "operation not permitted" not in str(error).lower():
                    raise
    return None


@contextmanager
def host_cache_copies(device, payload_bytes):
    """Give this fitted cache its own host storage and finish every copy before returning it."""
    stream = None
    if device.type == "cuda" and payload_bytes >= _MIN_LAYER_BYTES and _pinning_fits(payload_bytes):
        try:
            stream = torch.cuda.Stream(device=device)
        except torch.cuda.OutOfMemoryError:
            pass
    enabled = stream is not None

    def to_host(pair):
        nonlocal enabled
        if enabled:
            host = _pinned_pair(pair)
            if host is not None:
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    pair.key.record_stream(stream)
                    pair.value.record_stream(stream)
                    host.key.copy_(pair.key, non_blocking=True)
                    host.value.copy_(pair.value, non_blocking=True)
                return host
            enabled = False
        return pair.to("cpu")

    try:
        yield to_host
    finally:
        if stream is not None:
            stream.synchronize()


def streamable_layers(cache, rows, workload, device):
    """Keep native dense/tiling boundaries and the layouts of both projected heads."""
    if device.type != "cuda" or len(cache.layers) < 2:
        return False
    first = cache.layers[0]
    size = first.key.nbytes + first.value.nbytes
    if first.key.device.type != "cpu" or size < _MIN_LAYER_BYTES:
        return False
    members = rows.shape[0]
    full_bytes = len(cache.layers) * size + members * workload.bytes_per_item(device) + workload.destination_bytes(members)
    if full_bytes > execution_budget(device):
        return False
    return all(
        value.device.type == "cpu" and value.shape == reference.shape
        and value.stride() == reference.stride() and value.dtype == reference.dtype
        for pair in cache.layers.values()
        for value, reference in ((pair.key, first.key), (pair.value, first.value))
    )


@contextmanager
def device_cache_layers(cache, device):
    """Stage one ordered layer while the previous layer is running, using two private GPU buffers."""
    first = cache.layers[0]
    compute = torch.cuda.current_stream(device)
    try:
        transfer = torch.cuda.Stream(device=device)
        with torch.cuda.stream(transfer):
            slots = [
                KVPair(torch.empty_like(first.key, device=device), torch.empty_like(first.value, device=device))
                for _ in range(2)
            ]
    except torch.cuda.OutOfMemoryError:
        slots = None
    if slots is None:
        yield None
        return

    def stages():
        done = [None, None]
        for index in range(len(cache.layers)):
            source = cache.layers[index]
            slot = index % 2
            pair = slots[slot]
            with torch.cuda.stream(transfer):
                if done[slot] is not None:
                    transfer.wait_event(done[slot])
                pair.key.copy_(source.key, non_blocking=True)
                pair.value.copy_(source.value, non_blocking=True)
                ready = transfer.record_event()
            compute.wait_event(ready)
            yield pair
            done[slot] = compute.record_event()

    try:
        yield stages()
    finally:
        for pair in slots:
            pair.key.record_stream(compute)
            pair.value.record_stream(compute)
