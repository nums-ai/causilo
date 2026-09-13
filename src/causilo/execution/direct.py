"""Predict with training context recomputed for each ensemble/query batch."""

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import torch

from ..data.dataset import PreparedDataset
from ..data.ensemble import EnsembleMember
from ..model import Model, ModelConfig
from .memory import FP32_BYTES, MAX_EXECUTION_ATTEMPTS, Stage, Workload, execution_budget
from .runner import ModelRunner


@dataclass(frozen=True)
class BatchPlan:
    """How many ensemble members and query rows to evaluate in one call."""

    members: int
    queries: int


@dataclass(frozen=True)
class BatchPlanner:
    """Estimate peak live tensors and favor whole-query ensemble batches."""

    config: ModelConfig
    training_rows: int
    features: int

    def estimate(self, members: int, queries: int, device: torch.device) -> int:
        """Estimate the peak stage, including resident feature grids shared by chunks."""
        config = self.config
        rows = self.training_rows + queries
        groups = math.ceil(self.features / config.group_size)
        grid = rows * groups * config.width * FP32_BYTES
        encoding = rows * groups * config.group_size * (config.width + 6 * config.frequencies) * FP32_BYTES
        context = Workload(Stage.COLUMN, rows, config.width, self.training_rows).bytes_per_item(device)
        prediction = Workload(
            Stage.PREDICTION, rows, config.width * config.row_latents, self.training_rows
        ).bytes_per_item(device)
        # Stage partitions share the resident feature grid and one destination grid.
        return members * max(encoding + grid, 3 * grid + context, prediction + grid)

    def choose(self, members: int, queries: int, budget: int, device: torch.device) -> BatchPlan:
        """Reduce concurrent members first, then query rows; retain all training rows."""
        for count in range(members, 0, -1):
            if self.estimate(count, queries, device) <= budget:
                return BatchPlan(count, queries)
        if self.estimate(1, 1, device) > budget:
            raise torch.cuda.OutOfMemoryError(
                "Training context and minimum query exceed the execution budget"
            )
        low, high = 1, queries
        while low < high:
            middle = (low + high + 1) // 2
            if self.estimate(1, middle, device) <= budget:
                low = middle
            else:
                high = middle - 1
        return BatchPlan(1, low)


def _feature_batch(
    fitted: PreparedDataset,
    transformed: dict[str, np.ndarray],
    members: tuple[EnsembleMember, ...],
    query_slice: slice,
) -> np.ndarray:
    """Pack (members, training + query rows, features) with training rows first."""
    tables = []
    for member in members:
        training = fitted.training_table(member.normalization)
        queries = transformed[member.normalization][query_slice]
        table = np.concatenate((training, queries))
        tables.append(table[:, member.feature_order])
        # Only the permuted copies need to stay live until the batch is stacked.
        del table, training, queries
    return np.stack(tables)


def direct_predictions(
    model: Model,
    fitted: PreparedDataset,
    transformed: dict[str, np.ndarray],
    device: torch.device,
    reduce_output: Callable[[torch.Tensor], torch.Tensor],
) -> Iterator[tuple[EnsembleMember, torch.Tensor]]:
    """Yield each member with all its query outputs on CPU, in execution order.

    An OOM restarts the current member batch, discarding its partial query
    outputs. Previously yielded member batches remain complete and valid.
    """
    members = fitted.members_by_normalization()
    query_count = len(next(iter(transformed.values())))
    footprint = BatchPlanner(model.config, len(fitted.features), fitted.features.shape[1])
    budget = execution_budget(device)
    runner = ModelRunner(model)
    start = 0
    while start < len(members):
        completed = False
        for _ in range(MAX_EXECUTION_ATTEMPTS):
            plan = footprint.choose(len(members) - start, query_count, budget, device)
            selected = members[start : start + plan.members]
            pieces = []
            tensor = result = target_tensor = None
            try:
                targets = np.stack([fitted.targets_for(member) for member in selected])
                target_tensor = torch.as_tensor(targets, device=device, dtype=torch.float32)
                for offset in range(0, query_count, plan.queries):
                    tables = _feature_batch(
                        fitted, transformed, selected, slice(offset, offset + plan.queries)
                    )
                    tensor = torch.as_tensor(tables, device=device, dtype=torch.float32)
                    result = runner.predict(tensor, target_tensor)
                    pieces.append(reduce_output(result).cpu())
                    del tensor, result
                completed = True
            except torch.cuda.OutOfMemoryError:
                # Release device references before empty_cache and replanning.
                # Clear this batch's outputs so retried rows cannot be duplicated.
                pieces.clear()
                tensor = result = target_tensor = None
            if completed:
                break
            torch.cuda.empty_cache()
            budget = min(budget // 2, footprint.estimate(plan.members, plan.queries, device) - 1)
        if not completed:
            raise torch.cuda.OutOfMemoryError("Prediction exhausted its memory recovery attempts")
        result = torch.cat(pieces, dim=1)
        for member, prediction in zip(selected, result):
            yield member, prediction
        start += plan.members
