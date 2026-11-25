"""Execute one uncached member, recovering with query chunks or recomputation."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from ..data.dataset import PreparedDataset
from ..data.ensemble import EnsembleMember
from ..model import Model, ModelConfig
from ..nn.cache import FrequencyWeights
from .memory import (
    FP32_BYTES,
    MAX_EXECUTION_ATTEMPTS,
    Stage,
    Workload,
    embedding_workspace,
    empty_device_cache,
    execution_budget,
    is_out_of_memory,
)
from .recompute import RecomputeRunner
from .runner import ModelRunner


@dataclass(frozen=True)
class BatchPlan:
    """Execution route and query chunk size for one member."""
    queries: int
    mode: Literal["direct", "recompute"] = "direct"


@dataclass(frozen=True)
class BatchPlanner:
    """Estimate packed execution memory and plan single-member recovery."""

    config: ModelConfig
    training_rows: int
    features: int

    def estimate(self, members: int, queries: int, device: torch.device) -> int:
        """Estimate the peak stage, including resident feature grids shared by chunks."""
        config = self.config
        rows = self.training_rows + queries
        groups = math.ceil(self.features / config.group_size)
        grid = rows * groups * config.width * FP32_BYTES
        encoding = rows * groups * embedding_workspace(config)
        context = Workload(Stage.COLUMN, rows, config.width, self.training_rows).bytes_per_item(device)
        prediction = Workload(
            Stage.PREDICTION, rows, config.width * config.row_latents, self.training_rows
        ).bytes_per_item(device)
        # Stage partitions share the resident feature grid and one destination grid.
        return members * max(encoding + grid, 3 * grid + context, prediction + grid)

    def choose(self, queries: int, budget: int, device: torch.device) -> BatchPlan:
        """Compare query batching with encoder replay work for one member."""
        if self.estimate(1, queries, device) <= budget:
            return BatchPlan(queries)
        replay = BatchPlan(queries, mode="recompute")
        if self.estimate(1, 1, device) > budget:
            return replay
        low, high = 1, queries
        while low < high:
            middle = (low + high + 1) // 2
            if self.estimate(1, middle, device) <= budget:
                low = middle
            else:
                high = middle - 1
        batches = math.ceil(queries / low)
        training, extra = self._training_work()
        if self._recompute_state_bytes(queries) < budget and (batches - 1) * training > extra:
            return replay
        return BatchPlan(low)

    def _training_work(self) -> tuple[int, int]:
        """Estimate direct training MACs and the extra MACs needed for replay.

        Count dominant attention/linear products, not elapsed time. Query work
        is common to both routes. Kernel efficiency, transfers, normalization
        and elementwise work are omitted; ties favor direct query batching.
        """
        config, count = self.config, self.training_rows
        width, latents = config.width, config.row_latents
        groups = math.ceil(self.features / config.group_size)
        rounds = config.row_depths[0] - 1
        first, second = config.column_depths

        def attention(queries, context, *, feedforward=False):
            work = 2 * (queries + context) * width**2 + 2 * queries * context * width
            return work + (3 * config.expansion * queries * width**2 if feedforward else 0)

        embedding = groups * 2 * config.frequencies * width
        broadcast = (2 + 3 * config.expansion) * width**2 + 2 * config.column_latents * width
        gather = attention(config.column_latents, count, feedforward=True) + 2 * config.column_latents * width**2
        finish = attention(groups, latents)
        update = attention(latents, groups + latents, feedforward=True)
        row = rounds * (attention(groups + latents, latents) + update) + finish
        pool = config.row_depths[1] * update
        encoder = count * (embedding + row + pool) + groups * (first + second) * (gather + count * broadcast)

        # Every layer prepares training K/V; only nonfinal layers update the
        # training rows. This attention term grows quadratically with count.
        prediction_width = width * latents
        prediction = 2 * config.prediction_depth * count * prediction_width**2
        prediction += max(0, config.prediction_depth - 1) * count * (
            (2 + 3 * config.expansion) * prediction_width**2 + 2 * count * prediction_width
        )

        # Recompute embeds training 4 times (3 without a latent trace). Column
        # summaries skip their final broadcast; row features are replayed twice.
        extra_embeddings = 2 + int(rounds > 0)
        extra_broadcasts = (extra_embeddings - 1) * first + max(0, first - 1) + max(0, second - 1)
        extra = count * (
            extra_embeddings * embedding + groups * extra_broadcasts * broadcast
            + (2 * (rounds + 1) - 1) * finish
        )
        return encoder + prediction, extra

    def _recompute_state_bytes(self, queries: int) -> int:
        """Conservatively budget retained states before preferring recomputation.

        Include overlapping row buffers and unreduced outputs in FP32. Stage
        workspaces are sized separately by RecomputeRunner and can still OOM.
        """
        config, count = self.config, self.training_rows
        width = config.width * config.row_latents
        groups = math.ceil(self.features / config.group_size)
        trace = count * (config.row_depths[0] - 1) * width
        rows = 2 * (count + queries) * width
        keys = 2 * count * width
        summaries = 2 * groups * sum(config.column_depths) * config.column_latents * config.width
        return (trace + rows + keys + summaries + queries * config.outputs) * FP32_BYTES


def predict_direct(
    model: Model,
    fitted: PreparedDataset,
    transformed: dict[str, np.ndarray],
    member: EnsembleMember,
    device: torch.device,
    reduce_output: Callable[[torch.Tensor], torch.Tensor],
    *, weights: FrequencyWeights | None,
) -> torch.Tensor:
    """Return one member's CPU outputs, retrying without publishing partial rows."""
    queries = transformed[member.normalization]
    query_count = len(queries)
    footprint = BatchPlanner(model.config, len(fitted.features), fitted.features.shape[1])
    budget = execution_budget(device)
    runner = ModelRunner(model)

    def feature_batch(section):
        training = fitted.training_table(member.normalization)
        table = np.concatenate((training, queries[section]))[:, member.feature_order]
        return np.ascontiguousarray(table[None], dtype=np.float32)

    targets = np.asarray(fitted.targets_for(member)[None], dtype=np.float32)
    for _ in range(MAX_EXECUTION_ATTEMPTS):
        plan = footprint.choose(query_count, budget, device)
        if plan.mode == "recompute":
            return RecomputeRunner(model, device).predict(
                torch.from_numpy(feature_batch(slice(None))), torch.from_numpy(targets), reduce_output, weights=weights
            )[0]
        pieces = []
        tensor = result = target_tensor = None
        try:
            target_tensor = torch.as_tensor(targets, device=device)
            for offset in range(0, query_count, plan.queries):
                tensor = torch.as_tensor(feature_batch(slice(offset, offset + plan.queries)), device=device)
                result = runner.predict(tensor, target_tensor, weights=weights)
                pieces.append(reduce_output(result)[0].cpu())
                tensor = result = None
            return torch.cat(pieces, dim=0)
        except RuntimeError as error:
            if not is_out_of_memory(error, device):
                raise
            # Restart this member after releasing the failed batch and partial
            # outputs. Already completed members belong to the shared scheduler.
            pieces.clear()
            del tensor, result, target_tensor
        empty_device_cache(device)
        budget = min(budget // 2, footprint.estimate(1, plan.queries, device) - 1)
    raise torch.cuda.OutOfMemoryError("Prediction exhausted its memory recovery attempts")
