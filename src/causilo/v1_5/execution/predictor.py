from __future__ import annotations

import torch
from torch import Tensor

from ..nn.cache import BlockCache
from ..nn.frequency import FrequencyWeights
from ..nn.prediction import PredictionBlock
from .chunks import CacheRunner, ChunkRunner, batch_items
from .memory import Stage, Workload


class PredictionRunner:
    """Batch native prediction layers and their reusable caches."""

    def __init__(self, block: PredictionBlock, device: torch.device):
        self.block = block
        self.output_width = block.head.output.out_features
        self.chunks = ChunkRunner(device)
        self.caches = CacheRunner(device)

    def _forward(self, rows: Tensor, targets: Tensor, *, weights: FrequencyWeights | None) -> Tensor:
        conditioned = self.block.add_context_targets(rows, targets)
        queries = self.block(conditioned, context_rows=targets.shape[-1], weights=weights)
        del conditioned
        return self.block.head(queries)

    def __call__(
        self,
        rows: Tensor,
        targets: Tensor,
        *,
        num_classes: int,
        weights: FrequencyWeights | None = None,
    ) -> Tensor:
        leading = tuple(rows.shape[:-2])

        def predict(part, section):
            labels = batch_items(targets, leading, section).to(part.device)
            return self._forward(part, labels, weights=weights)

        query_rows = rows.shape[-2] - targets.shape[-1]
        out = self.chunks.run_indexed(
            predict, rows,
            Workload(
                Stage.PREDICTION, rows.shape[-2], self.block.width, targets.shape[-1],
                output_elements=query_rows * self.output_width,
            ),
        )
        return out[..., :num_classes] if num_classes else out

    def build_cache(
        self,
        rows: Tensor,
        targets: Tensor,
        *,
        weights: FrequencyWeights | None = None,
    ) -> BlockCache:
        """Build prediction K/V from context rows."""
        leading = tuple(rows.shape[:-2])

        def prefill(part, section):
            labels = batch_items(targets, leading, section).to(part.device)
            conditioned = self.block.add_context_targets(part, labels)
            context = self.block.build_context(conditioned, weights=weights)
            return part.new_empty((part.shape[0], 0, self.output_width)), context

        _, cache = self.caches.build(
            prefill, rows,
            Workload(
                Stage.PREDICTION, rows.shape[-2], self.block.width, rows.shape[-2],
                output_elements=0, cache_elements=2 * rows.shape[-2] * self.block.width * len(self.block.layers),
            ),
        )
        return cache

    def predict_cached(
        self,
        rows: Tensor,
        cache: BlockCache,
        *,
        num_classes: int,
        weights: FrequencyWeights | None = None,
    ) -> Tensor:
        """Score query rows using cached context projections."""
        def query(rows, cache):
            return self.block.head(self.block.query(rows, cache, weights=weights))

        context_rows = next(iter(cache.layers.values())).key.shape[-2]
        out = self.caches.query(
            query, rows, cache,
            Workload(
                Stage.PREDICTION, rows.shape[-2], self.block.width, context_rows,
                output_elements=rows.shape[-2] * self.output_width,
            ),
            stream_layers=True,
        )
        return out[..., :num_classes] if num_classes else out
