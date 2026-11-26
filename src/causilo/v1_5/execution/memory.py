"""Shape-based workspace budgets for Causilo 1.5 inference."""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from threading import Lock

import psutil
import torch

from ..nn.model_config import (
    COLUMN_HEADS,
    COLUMN_LATENTS,
    ECDF_FREQUENCIES,
    EXPANSION,
    FOURIER_FREQUENCIES,
    NATIVE_CLASSES,
    PREDICTION_HEADS,
    ROW_HEADS,
    ROW_LATENTS,
)

FP32_BYTES = 4
EXECUTION_MEMORY_FRACTION = 0.8
MAX_EXECUTION_ATTEMPTS = 7
CUDA_MAX_BATCH_ITEMS = 65535


class Stage(Enum):
    """Neural stages with distinct activation shapes and precision."""

    FEATURE_EMBEDDING = "feature_embedding"
    COLUMN = "column"
    ROW = "row"
    POOL = "pool"
    PREDICTION = "prediction"


def attention_workspace(queries: int, context: int, width: int, heads: int, device: torch.device) -> int:
    """Bound projected heads, residuals, feedforward activations and attention scores."""
    vector = queries * width * FP32_BYTES
    projected_context = 5 * context * width * FP32_BYTES
    attention = 7 * vector + projected_context
    feedforward = (3 + 4 * EXPANSION) * vector
    statistics = queries * heads * FP32_BYTES
    streaming = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
    scores = 0 if streaming else 2 * heads * queries * context * FP32_BYTES
    return max(attention + statistics + scores, feedforward)


@dataclass(frozen=True)
class Workload:
    """Temporary stage tensors and retained output for one independent batch item."""

    stage: Stage
    rows: int
    width: int
    context_rows: int = 0
    output_elements: int | None = None
    cache_elements: int = 0

    def destination_bytes(self, items: int) -> int:
        elements = self.rows * self.width if self.output_elements is None else self.output_elements
        return items * (elements + self.cache_elements) * FP32_BYTES

    def bytes_per_item(self, device: torch.device) -> int:
        """Count tensors using the checkpoint's widths and stage token dimensions."""
        rows = self.rows
        vector = rows * self.width * FP32_BYTES
        if self.stage is Stage.FEATURE_EMBEDDING:
            frequencies = FOURIER_FREQUENCIES + ECDF_FREQUENCIES
            waves = rows * 2 * frequencies * FP32_BYTES
            labels = self.context_rows * NATIVE_CLASSES * (torch.int64.itemsize + FP32_BYTES)
            ranks = 2 * self.context_rows * (torch.int64.itemsize + FP32_BYTES)
            return 3 * vector + 3 * waves + labels + ranks
        if self.stage is Stage.COLUMN:
            gather = attention_workspace(COLUMN_LATENTS, self.context_rows, self.width, COLUMN_HEADS, device)
            broadcast = attention_workspace(rows, COLUMN_LATENTS, self.width, COLUMN_HEADS, device)
            return vector + max(gather, broadcast)
        if self.stage is Stage.ROW:
            tokens = rows + ROW_LATENTS
            gather = attention_workspace(ROW_LATENTS, tokens, self.width, ROW_HEADS, device)
            broadcast = attention_workspace(tokens, ROW_LATENTS, self.width, ROW_HEADS, device)
            return 2 * vector + max(gather, broadcast)
        if self.stage is Stage.POOL:
            return 3 * vector
        context = self.context_rows
        attention = attention_workspace(rows, context, self.width, PREDICTION_HEADS, device)
        head = 5 * vector + (self.output_elements or 0) * FP32_BYTES
        return vector + max(attention, head)


def direct_activation_bytes(model, shape: tuple[int, ...]) -> int:
    """Budget the two cell grids that can coexist across a direct stage."""
    members, rows, features = shape
    return 2 * members * rows * (features + ROW_LATENTS) * model.config.width * FP32_BYTES


def estimate_table_workspace_bytes(model, shape: tuple[int, ...], device: torch.device) -> int:
    """Reserve the feature grids and the larger column or prediction workspace."""
    members, rows, features = shape
    config = model.config
    columns = Workload(Stage.COLUMN, rows, config.width, rows)
    prediction = Workload(Stage.PREDICTION, rows, config.prediction_width, rows)
    return direct_activation_bytes(model, shape) + members * max(
        features * columns.bytes_per_item(device), prediction.bytes_per_item(device),
    )


def minimum_direct_workspace_bytes(model, shape: tuple[int, ...], device: torch.device) -> int:
    """Budget the larger of the grid phase and the prediction phase of a tiled direct pass.

    The feature grids are released once rows are pooled, so they never coexist
    with the prediction workspace. The grid phase keeps two minimum stage tiles
    beside both grids; prediction runs one member, its smallest partition.
    """
    members, rows, features = shape
    config = model.config
    persistent = members * rows * (features + 2 * config.prediction_width) * FP32_BYTES
    embedding = Workload(Stage.FEATURE_EMBEDDING, rows, config.width, rows)
    columns = Workload(Stage.COLUMN, rows, config.width, rows)
    prediction = Workload(Stage.PREDICTION, rows, config.prediction_width, rows)
    grid_tile = max(
        embedding.bytes_per_item(device),
        columns.bytes_per_item(device),
        query_row_workspace_bytes(model, features, device),
    )
    return persistent + max(
        direct_activation_bytes(model, shape) + 2 * grid_tile, prediction.bytes_per_item(device),
    )


def query_row_workspace_bytes(model, features: int, device: torch.device) -> int:
    """Size one cached query row; prepared context K/V is already resident."""
    config = model.config
    grid = (features + ROW_LATENTS) * config.width * FP32_BYTES
    column = attention_workspace(1, 0, config.width, COLUMN_HEADS, device)
    row = Workload(Stage.ROW, features, config.width).bytes_per_item(device)
    prediction = attention_workspace(1, 0, config.prediction_width, PREDICTION_HEADS, device)
    return 2 * grid + max(features * column, row, prediction)


_device_budgets = ContextVar("causilo_device_budgets", default=None)


@contextmanager
def planning_budget(device: torch.device, limit: int | None):
    """Apply this device lane's workspace ceiling."""
    if limit is None:
        yield
        return
    budgets = _device_budgets.get() or {}
    ceiling = min(limit, budgets.get(device, limit))
    token = _device_budgets.set({**budgets, device: ceiling})
    try:
        yield
    finally:
        _device_budgets.reset(token)


def _mount_path(value: str) -> Path:
    # Linux escapes whitespace and backslashes in mount roots and mountpoints.
    return Path(re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value))


def _memory_mounts(mountinfo: Path):
    mounts = {"cgroup": [], "cgroup2": []}
    try:
        entries = mountinfo.read_text().splitlines()
    except OSError:
        return mounts
    for entry in entries:
        prefix, _, suffix = entry.partition(" - ")
        fields, filesystem = prefix.split(), suffix.split()
        if len(fields) < 5 or len(filesystem) < 3:
            continue
        kind = filesystem[0]
        if kind == "cgroup2" or (kind == "cgroup" and "memory" in filesystem[2].split(",")):
            mounts[kind].append((_mount_path(fields[3]), _mount_path(fields[4])))
    return mounts


@lru_cache(maxsize=4)
def _cgroup_memory_files(membership: str, root: Path, mountinfo: Path):
    mounts = _memory_mounts(mountinfo)
    files = []
    for entry in membership.splitlines():
        fields = entry.split(":", 2)
        if len(fields) != 3:
            continue
        _, controllers, relative = fields
        if controllers and "memory" not in controllers.split(","):
            continue
        default = root / "memory" if controllers else root
        limit, usage = ("memory.limit_in_bytes", "memory.usage_in_bytes") if controllers else ("memory.max", "memory.current")
        member = Path(relative)
        if not member.is_absolute():
            continue
        for group_root, mount in mounts["cgroup" if controllers else "cgroup2"] or [(Path("/"), default)]:
            if ".." in member.parts:
                directory = mount
            elif member.is_relative_to(group_root):
                directory = mount / member.relative_to(group_root)
            else:
                # The process's cgroup namespace may already have stripped the mount root.
                directory = mount / member.relative_to("/")
                if not directory.is_dir():
                    continue
            while True:
                if directory.is_dir():
                    files.append((directory / limit, directory / usage))
                if directory == mount:
                    break
                directory = directory.parent
    return tuple(dict.fromkeys(files))


def host_available_bytes(
    groups=Path("/proc/self/cgroup"), root=Path("/sys/fs/cgroup"), *, mountinfo=Path("/proc/self/mountinfo"),
) -> int:
    """Read current physical and cgroup headroom, including limits on ancestor groups."""
    room = psutil.virtual_memory().available
    try:
        membership = groups.read_text()
    except OSError:
        return room
    for limit, usage in _cgroup_memory_files(membership, root, mountinfo):
        try:
            remaining = int(limit.read_text()) - int(usage.read_text())
        except (OSError, ValueError):
            continue  # An absent limit or v2's "max" leaves this level unrestricted.
        room = min(room, max(0, remaining))
    return room


_host_spill_lock = Lock()
_pending_host_spills = 0


class HostSpillMemoryError(MemoryError):
    """The assembled output of a tiled stage cannot fit in host memory."""


@contextmanager
def reserve_host_spill(size: int):
    """Budget a host output until its pages are populated, including concurrent GPU lanes."""
    global _pending_host_spills
    if size == 0:
        yield
        return
    with _host_spill_lock:
        if size + _pending_host_spills > host_available_bytes() * EXECUTION_MEMORY_FRACTION:
            raise HostSpillMemoryError("Inference output exceeds the available host memory")
        _pending_host_spills += size
    try:
        yield
    finally:
        with _host_spill_lock:
            _pending_host_spills -= size


def available_bytes(device: torch.device) -> int:
    """Count reusable memory within physical, allocator and device planning limits."""
    if device.type == "cpu":
        return host_available_bytes()
    if device.type == "mps":
        room = torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory()
        return max(0, min(psutil.virtual_memory().available, room))
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reusable = torch.cuda.memory_reserved(device) - allocated
    ceiling = int(total * torch.cuda.get_per_process_memory_fraction(device))
    budgets = _device_budgets.get()
    if budgets is not None and device in budgets:
        ceiling = min(ceiling, budgets[device])
    return max(0, min(free + reusable, ceiling - allocated))


def execution_budget(device: torch.device) -> int:
    return int(available_bytes(device) * EXECUTION_MEMORY_FRACTION)


def is_allocation_error(exc: MemoryError | RuntimeError) -> bool:
    return isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError)) or any(
        reason in str(exc).lower()
        for reason in ("out of memory", "can't allocate memory", "cannot allocate memory")
    )


def is_out_of_memory(error: RuntimeError, device: torch.device) -> bool:
    return (
        isinstance(error, torch.cuda.OutOfMemoryError)
        or "DefaultCPUAllocator: can't allocate memory" in str(error)
        or (device.type == "mps" and str(error).startswith("MPS backend out of memory"))
    )


def empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
