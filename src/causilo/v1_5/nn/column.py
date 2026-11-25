"""Gather each column's context into fixed latents and broadcast it to its rows."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .cache import KVPair
from .frequency import FrequencyWeights
from .layers.attention import AttentionLayer
from .layers.normalization import feedforward_residual, large_residual
from .model_config import COLUMN_HEADS, COLUMN_LATENTS, EXPANSION, ModelConfig


class ColumnAttention(AttentionLayer):
    """Prepare large column residuals directly in the feedforward input precision."""

    def _finish(self, rows: Tensor, update: Tensor) -> Tensor:
        if large_residual(rows):
            residual, normalized = feedforward_residual(rows, update, self.feedforward_norm)
            return residual + self.feedforward(normalized)
        return super()._finish(rows, update)


class ColumnLayer(nn.Module):
    """Read context into latent slots, then let every row attend to those slots."""

    def __init__(self, width: int, heads: int, expansion: int, latents: int) -> None:
        super().__init__()
        self.gather = ColumnAttention(width, heads, expansion, scale_query=True)
        self.broadcast = ColumnAttention(width, heads, expansion, scale_query=False)
        self.latents = nn.Parameter(torch.empty(latents, width))

    def _gather_context(
        self, context: Tensor, weights: FrequencyWeights | None, *, separate_kv: bool = False,
    ) -> Tensor:
        latents = self.latents.expand(*context.shape[:-2], *self.latents.shape)
        return self.gather(latents, context, weights=weights, separate_kv=separate_kv)

    def prepare(self, context: Tensor, weights: FrequencyWeights | None = None) -> KVPair:
        """Summarize context rows into broadcast K/V for tiled replay."""
        return self.broadcast.prepare(self._gather_context(context, weights))

    def query(self, rows: Tensor, context: KVPair) -> Tensor:
        """Broadcast prepared column context to the query rows."""
        return self.broadcast.query(rows, context)

    def forward(self, rows: Tensor, context_rows: int, weights: FrequencyWeights | None = None) -> Tensor:
        """Update (*batch, context + query rows, width) without mixing columns."""
        latents = self._gather_context(rows[..., :context_rows, :], weights)
        return self.broadcast(rows, latents)

    def build_context(
        self, rows: Tensor, context_rows: int,
        weights: FrequencyWeights | None = None,
    ) -> tuple[Tensor, KVPair]:
        """Update the context rows and return the broadcast K/V for this layer."""
        latents = self._gather_context(rows[..., :context_rows, :], weights, separate_kv=True)
        return self.broadcast.prefill(rows, latents)


class ColumnBlock(nn.Module):
    """Hold the column layers paired with successive row layers."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.width = config.width
        self.layers = nn.ModuleList(
            ColumnLayer(config.width, COLUMN_HEADS, EXPANSION, COLUMN_LATENTS)
            for _ in range(config.column_depth)
        )
