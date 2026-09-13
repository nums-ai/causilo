"""Conservative execution budgets; sizing changes batching, never model settings."""

from dataclasses import dataclass
from enum import Enum

import psutil
import torch

FP32_BYTES = 4
EXECUTION_MEMORY_FRACTION = 0.8
MAX_EXECUTION_ATTEMPTS = 7
# Bound the flattened attention batch to fit the CUDA kernel grid dimension.
CUDA_MAX_BATCH_ITEMS = 65535

# Fixed heuristic workspace allowances, not dimensions read from ModelConfig.
# They preserve the existing sizing policy; actual allocation failures are
# handled by reducing partitions in the execution runners.
COLUMN_CONTEXT_ELEMENTS_PER_WIDTH = 128 * 8
PREDICTION_CONTEXT_BYTES_PER_ROW = 512 * 8


class Stage(Enum):
    """Stages with different activation lifetimes and workspace estimates."""

    COLUMN = "column"
    ROW = "row"
    POOL = "pool"
    PREDICTION = "prediction"


@dataclass(frozen=True)
class Workload:
    """Memory estimate for one independently executable item.

    ``rows`` is the attended sequence length: table rows for column/prediction
    stages, feature groups for row/pool stages. Leading batch axes are counted
    separately as ``items`` by ChunkPlan. Byte counts conservatively assume FP32
    intermediates even when attention uses lower precision.
    """

    stage: Stage
    rows: int
    width: int
    context_rows: int = 0
    output_elements: int | None = None
    cache_elements: int = 0

    def destination_bytes(self, items: int) -> int:
        """Space for the assembled output and any K/V tensors kept across chunks."""
        elements = self.rows * self.width if self.output_elements is None else self.output_elements
        return items * (elements + self.cache_elements) * FP32_BYTES

    def bytes_per_item(self, device: torch.device) -> int:
        """Estimate temporary activations plus attention workspace for one item."""
        activation_copies = {Stage.COLUMN: 12, Stage.ROW: 16, Stage.POOL: 7, Stage.PREDICTION: 15}[self.stage]
        payload = self.rows * self.width * FP32_BYTES * activation_copies
        if self.stage == Stage.COLUMN:
            payload += COLUMN_CONTEXT_ELEMENTS_PER_WIDTH * self.width * FP32_BYTES
        if self.stage == Stage.PREDICTION:
            payload += self.context_rows * PREDICTION_CONTEXT_BYTES_PER_ROW
            native_bfloat = False
            if device.type == "cuda":
                with torch.cuda.device(device):
                    native_bfloat = torch.cuda.is_bf16_supported(including_emulation=False)
            if not native_bfloat:
                # Allow for materialized query-by-context attention matrices.
                payload += 4 * self.rows * self.context_rows * FP32_BYTES
        return payload


def available_bytes(device: torch.device) -> int:
    """Count reusable allocator memory, bounded by both physical and process limits."""
    if device.type == "cpu":
        return psutil.virtual_memory().available
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reusable = torch.cuda.memory_reserved(device) - allocated
    allocator_room = int(total * torch.cuda.get_per_process_memory_fraction(device)) - allocated
    return max(0, min(free + reusable, allocator_room))


def execution_budget(device: torch.device) -> int:
    """Leave headroom for allocations outside the stage workspace estimate."""
    return int(available_bytes(device) * EXECUTION_MEMORY_FRACTION)
