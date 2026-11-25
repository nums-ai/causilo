from __future__ import annotations

import torch
from torch import Tensor

from ..nn.cache import BlockCache
from ..nn.column import ColumnBlock
from ..nn.ecdf import WeightedECDF, fit_ecdf_context, fit_weighted_ecdf
from ..nn.embeddings import FeatureEmbedding
from ..nn.frequency import FrequencyWeights
from .chunks import CacheRunner, ChunkRunner, batch_items
from .memory import Stage, Workload


class FeatureEmbeddingRunner:
    """Project cells with ranks fitted on the context rows."""

    def __init__(self, block: FeatureEmbedding, device: torch.device):
        self.block = block
        self.chunks = ChunkRunner(device)

    def __call__(
        self,
        features: Tensor,
        targets: Tensor | None,
        *,
        context: Tensor | WeightedECDF | None = None,
        weights: FrequencyWeights | None = None,
    ) -> Tensor:
        values = context.values if isinstance(context, WeightedECDF) else context
        if values is not None and (
            values.ndim != features.ndim
            or values.shape[:-2] != features.shape[:-2]
            or values.shape[-1] != 1
            or values.dtype != features.dtype
        ):
            raise ValueError("Column ECDF cache is incompatible with the current feature layout. Refit the estimator.")
        if values is None and targets is None:
            raise ValueError("Column ECDF cache is missing. Refit the estimator.")
        column_targets = None
        if targets is not None:
            targets = targets.float()
            column_targets = targets.unsqueeze(1).expand(-1, features.shape[1], -1)
        leading = tuple(features.shape[:-2])
        cumulative = context.cumulative if isinstance(context, WeightedECDF) else None

        def embed(part, section):
            def selected(value):
                if value is None:
                    return None
                return batch_items(value, leading, section).to(part.device)

            return self._embed(
                part, selected(column_targets), selected(values), selected(cumulative),
                weights=weights,
            )

        context_rows = values.shape[-2] if values is not None else targets.shape[-1]
        workload = Workload(Stage.FEATURE_EMBEDDING, features.shape[-2], self.block.width, context_rows)
        embeddings = self.chunks.run_indexed(embed, features, workload)
        return embeddings.transpose(1, 2)

    def _embed(
        self,
        features: Tensor,
        targets: Tensor | None,
        context: Tensor | None,
        cumulative: Tensor | None,
        *,
        weights: FrequencyWeights | None,
    ) -> Tensor:
        if context is None:
            reference = fit_column_context(features, targets.shape[-1], weights)
        else:
            reference = WeightedECDF(context, cumulative) if cumulative is not None else context
        return self.block(features, reference, targets)


def column_features(table: Tensor) -> Tensor:
    """Arrange one scalar per cell with columns as batch axes."""
    return table.float().transpose(1, 2).unsqueeze(-1)


def fit_column_context(
    features: Tensor, context_rows: int, weights: FrequencyWeights | None,
) -> Tensor | WeightedECDF:
    if weights is None:
        return fit_ecdf_context(features, context_rows)
    return fit_weighted_ecdf(features, weights.counts)


class ColumnRunner:
    """Run one column attention block and manage its cached projections."""

    def __init__(self, block: ColumnBlock, device: torch.device):
        self.layers = block.layers
        self.width = block.width
        self.chunks = ChunkRunner(device)
        self.caches = CacheRunner(device)

    def __call__(
        self,
        features: Tensor,
        layer_index: int,
        context_rows: int | None,
        *,
        cache: BlockCache | None,
        weights: FrequencyWeights | None,
    ) -> Tensor:
        rows = features.transpose(1, 2).float()
        block = self.layers[layer_index]
        context_length = context_rows if context_rows is not None else block.latents.shape[0]
        cache_elements = 2 * block.latents.numel() if cache is not None and context_rows is not None else 0
        workload = Workload(
            Stage.COLUMN, rows.shape[-2], self.width, context_length, cache_elements=cache_elements,
        )
        if cache is None:
            output = self.chunks.run(
                lambda part: block(part, context_rows, weights=weights), rows, workload,
            )
        elif context_rows is None:
            step_cache = BlockCache({layer_index: cache.layers[layer_index]})

            def query(rows, cache):
                return block.query(rows, cache.layers[layer_index])

            output = self.caches.query(query, rows, step_cache, workload)
        else:
            def build_context(part, _section):
                result, keys = block.build_context(part, context_rows, weights=weights)
                return result, BlockCache({layer_index: keys})

            output, step_cache = self.caches.build(build_context, rows, workload)
            cache.layers.update(step_cache.layers)
        return output.transpose(1, 2)
