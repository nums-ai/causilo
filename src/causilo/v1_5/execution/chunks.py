"""Partition independent leading axes while retaining complete attention contexts."""

from __future__ import annotations

import math
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch
from torch import Tensor

from ..nn.cache import BlockCache, KVPair
from .memory import (
    CUDA_MAX_BATCH_ITEMS,
    FP32_BYTES,
    MAX_EXECUTION_ATTEMPTS,
    Workload,
    empty_device_cache,
    execution_budget,
    is_out_of_memory,
    reserve_host_spill,
)
from .precision import run_inference
from .transfers import device_cache_layers, streamable_layers

_tiled_direct_outputs = ContextVar("tiled_direct_outputs", default=False)


@contextmanager
def retain_tiled_direct_outputs():
    """Keep direct stage outputs on the device while a separate tile fits."""
    token = _tiled_direct_outputs.set(True)
    try:
        yield
    finally:
        _tiled_direct_outputs.reset(token)


@dataclass
class ChunkPlan:
    """Execution partition and output placement for one operation."""

    items: int
    limit: int
    device: torch.device
    keep_on_device: bool
    failures: int = 0

    @classmethod
    def choose(cls, items: int, workload: Workload, device: torch.device):
        budget = execution_budget(device)
        item_bytes = max(1, workload.bytes_per_item(device))
        output_bytes = workload.destination_bytes(items)
        keep_on_device = device.type != "cuda" or output_bytes <= budget // 2
        if not keep_on_device and _tiled_direct_outputs.get():
            keep_on_device = output_bytes + item_bytes <= budget
        retained = output_bytes if keep_on_device else items * workload.cache_elements * FP32_BYTES
        kernel_limit = CUDA_MAX_BATCH_ITEMS if device.type == "cuda" else items
        limit = max(1, min(items, kernel_limit, max(0, budget - retained) // item_bytes))
        return cls(items, limit, device, keep_on_device)

    def slices(self):
        for start in range(0, self.items, self.limit):
            yield slice(start, min(start + self.limit, self.items))

    def shrink(self):
        self.failures += 1
        self.limit = max(1, self.limit // 2)
        empty_device_cache(self.device)


def batch_items(value: Tensor, leading: tuple[int, ...], section: slice | None) -> Tensor:
    """Select flattened batch items without materializing a transposed full table."""
    if section is None:
        return value
    if not leading:
        return value.unsqueeze(0)
    if len(leading) == 1:
        return value[section]
    if all(
        value.stride(index) == value.stride(index + 1) * leading[index + 1]
        for index in range(len(leading) - 1)
    ):
        return value.reshape(-1, *value.shape[len(leading):])[section]
    flat = torch.arange(section.start, section.stop, device=value.device)
    coordinates = []
    for width in reversed(leading):
        coordinates.append(flat.remainder(width))
        flat = torch.div(flat, width, rounding_mode="floor")
    return value[tuple(reversed(coordinates))]


def reshape_keys(cache: BlockCache, leading: tuple[int, ...]) -> BlockCache:
    """Restore the batch axes while keeping heads, context and head width intact."""
    return BlockCache({
        index: KVPair(
            pair.key.reshape(*leading, *pair.key.shape[-3:]),
            pair.value.reshape(*leading, *pair.value.shape[-3:]),
        )
        for index, pair in cache.layers.items()
    })


class ChunkRunner:
    """Run a typed stage operation on ordered flat batch partitions."""

    def __init__(self, device: torch.device):
        self.device = device

    def run(self, operation, rows: Tensor, workload: Workload) -> Tensor:
        return self.run_indexed(lambda part, _: operation(part), rows, workload)

    def _allocate_output_buffer(self, shape, dtype, keep_on_device, reservations):
        if keep_on_device:
            try:
                return torch.empty(shape, dtype=dtype, device=self.device)
            except torch.cuda.OutOfMemoryError:
                pass
        if self.device.type == "cuda":
            reservations.enter_context(reserve_host_spill(math.prod(shape) * dtype.itemsize))
            try:
                return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
            except RuntimeError:
                pass
        return torch.empty(shape, dtype=dtype, device="cpu")

    def run_indexed(self, operation, rows: Tensor, workload: Workload, *, on_retry=None) -> Tensor:
        """Keep only one input tile and one assembled destination alive."""
        leading = tuple(rows.shape[:-2])
        items = math.prod(leading)
        plan = ChunkPlan.choose(items, workload, self.device)
        while True:
            destination = result = None
            reservations = None
            try:
                if plan.limit >= items:
                    part = rows.to(self.device, non_blocking=True)
                    result = run_inference(
                        self.device, operation, part, None, stage=workload.stage,
                    )
                    if self.device.type == "cuda" and not plan.keep_on_device and result.device.type != "cpu":
                        with reserve_host_spill(result.nbytes):
                            return result.cpu()
                    return result
                for section in plan.slices():
                    part = batch_items(rows, leading, section).to(self.device, non_blocking=True)
                    result = run_inference(
                        self.device, operation, part, section, stage=workload.stage,
                    )
                    if destination is None:
                        reservations = ExitStack()
                        destination = self._allocate_output_buffer(
                            (items, *result.shape[1:]), result.dtype, plan.keep_on_device, reservations,
                        )
                    destination[section].copy_(result, non_blocking=destination.is_pinned())
                    part = result = None
                if self.device.type == "cuda" and destination.device.type == "cpu" and destination.is_pinned():
                    torch.cuda.current_stream(self.device).synchronize()
                return destination.reshape(*leading, *destination.shape[1:])
            except RuntimeError as error:
                if not is_out_of_memory(error, self.device):
                    raise
                if plan.limit <= 1 or plan.failures >= MAX_EXECUTION_ATTEMPTS - 1:
                    raise
                if destination is not None and self.device.type == "cuda" and destination.is_pinned():
                    torch.cuda.current_stream(self.device).synchronize()
                part = destination = result = None
            finally:
                if reservations is not None:
                    reservations.close()
            if on_retry is not None:
                on_retry()
            plan.shrink()


class CacheRunner:
    """Build and query layer projections with the same batch order as their rows."""

    def __init__(self, device: torch.device):
        self.device = device
        self.chunks = ChunkRunner(device)

    def build(self, operation, rows: Tensor, workload: Workload) -> tuple[Tensor, BlockCache]:
        leading = tuple(rows.shape[:-2])
        items = math.prod(leading)
        projections = None

        def clear_projections():
            nonlocal projections
            projections = None

        def prefill(part, section):
            nonlocal projections
            if section is None or section.start == 0:
                projections = None
            output, prepared = operation(part, section)
            if section is None:
                projections = prepared
                return output
            if projections is None:
                projections = BlockCache({
                    index: KVPair(
                        pair.key.new_empty((items, *pair.key.shape[1:])),
                        pair.value.new_empty((items, *pair.value.shape[1:])),
                    )
                    for index, pair in prepared.layers.items()
                })
            for index, pair in prepared.layers.items():
                retained = projections.layers[index]
                retained.key[section].copy_(pair.key)
                retained.value[section].copy_(pair.value)
            return output

        output = self.chunks.run_indexed(prefill, rows, workload, on_retry=clear_projections)
        return output, reshape_keys(projections, leading)

    def query(self, operation, rows: Tensor, cache: BlockCache, workload: Workload, *, stream_layers=False) -> Tensor:
        leading = tuple(rows.shape[:-2])

        def attend(part, section):
            if stream_layers and section is None and streamable_layers(cache, part, workload, self.device):
                with device_cache_layers(cache, self.device) as layers:
                    if layers is not None:
                        return operation(part, layers)
            selected = BlockCache({
                index: KVPair(
                    batch_items(pair.key, leading, section).to(self.device),
                    batch_items(pair.value, leading, section).to(self.device),
                )
                for index, pair in cache.layers.items()
            })
            return operation(part, selected)

        return self.chunks.run_indexed(attend, rows, workload)
