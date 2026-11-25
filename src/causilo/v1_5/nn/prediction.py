"""Predict query rows using context K/V at each layer."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cache import BlockCache, KVPair
from .embeddings import TargetEmbedding
from .frequency import FrequencyWeights
from .layers.attention import AttentionLayer
from .layers.normalization import create_norm
from .model_config import EXPANSION, NUM_QUANTILES, PREDICTION_HEADS, ModelConfig


class PredictionBlock(nn.Module):
    """Attend to target-conditioned context through the prediction layers."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.width = config.prediction_width
        self.layers = nn.ModuleList(
            AttentionLayer(self.width, PREDICTION_HEADS, EXPANSION, scale_query=True)
            for _ in range(config.prediction_depth)
        )
        self.target = TargetEmbedding(self.width, config.max_classes)
        self.head = Head(self.width, config.max_classes or NUM_QUANTILES)

    def add_context_targets(self, rows: Tensor, targets: Tensor) -> Tensor:
        """Add targets to the context prefix of a copy of (batch, rows, width)."""
        context_rows = targets.shape[1]
        embedded_targets = self.target(targets)
        output = rows.clone()
        output[:, :context_rows].add_(embedded_targets)
        return output

    def forward(self, rows: Tensor, *, context_rows: int, weights: FrequencyWeights | None = None) -> Tensor:
        """Return query representations after advancing context and query rows."""
        for index in range(len(self.layers) - 1):
            rows = self.layers[index](rows, context_rows=context_rows, weights=weights)
        context = rows[..., :context_rows, :]
        return self.layers[-1](rows[..., context_rows:, :], context, weights=weights)

    def build_context(
        self, rows: Tensor, *, weights: FrequencyWeights | None = None,
    ) -> BlockCache:
        """Prepare the K/V from target-enriched context at every prediction layer."""
        cache = BlockCache()
        for index in range(len(self.layers) - 1):
            rows, cache.layers[index] = self.layers[index].prefill(rows, context_rows=rows.shape[-2], weights=weights)
        cache.layers[len(self.layers) - 1] = self.layers[-1].prepare(rows, contiguous=True)
        return cache

    def query(self, rows: Tensor, cache: BlockCache | Iterable[KVPair], weights: FrequencyWeights | None = None) -> Tensor:
        """Encode query rows with prepared layer contexts."""
        if isinstance(cache, BlockCache):
            layers = cache.layers
            cache = (layers[index] for index in range(len(self.layers)))
        for layer, pair in zip(self.layers, cache, strict=True):
            rows = layer.query(rows, pair, weights=weights)
        return rows


class Head(nn.Module):
    """Map each query representation to logits or native quantiles."""

    def __init__(self, width: int, outputs: int) -> None:
        super().__init__()
        self.normalize = create_norm(width)
        self.hidden = nn.Linear(width, width * 2)
        self.output = nn.Linear(width * 2, outputs)

    def forward(self, rows: Tensor) -> Tensor:
        """Evaluate the head in FP32 when the transformer uses autocast."""
        if not torch.is_autocast_enabled(rows.device.type):
            return self.output(F.gelu(self.hidden(self.normalize(rows))))
        with torch.autocast(device_type=rows.device.type, enabled=False):
            return self.output(F.gelu(self.hidden(self.normalize(rows.float()))))
