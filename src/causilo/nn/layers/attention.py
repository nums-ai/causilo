"""Attention primitives with separable query and reusable context projections."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..cache import FrequencyWeights, KVPair
from .feedforward import FeedForward
from .scaling import QueryScale


def unit_rms(value: Tensor) -> Tensor:
    """Normalize each head in FP32 before returning to the incoming precision."""
    precise = value.float()
    denominator = precise.square().mean(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps).sqrt()
    return (precise / denominator).to(value.dtype)


class Attention(nn.Module):
    """Multihead attention with optional head normalization and learned query scaling.

    Token inputs use (*batch, sequence, width). Prepared queries and K/V use
    (*batch, heads, sequence, head_width). The leading axes must align between
    queries and context; sequence lengths may differ.
    """

    def __init__(
        self, width: int, heads: int, normalize_heads: bool = False, scale_query: bool = True
    ) -> None:
        super().__init__()
        self.width = width
        self.heads = heads
        self.normalize_heads = normalize_heads
        self.projection = nn.Linear(width, width * 3)
        self.output = nn.Linear(width, width)
        self.scaling = QueryScale(heads, width // heads) if scale_query else None

    def _heads(self, value: Tensor) -> Tensor:
        """Split width into heads and move heads before the sequence axis."""
        return value.unflatten(-1, (self.heads, self.width // self.heads)).transpose(-3, -2)

    def _query(self, projected: Tensor) -> Tensor:
        query = self._heads(projected)
        return unit_rms(query) if self.normalize_heads else query

    def _keys(self, key: Tensor, value: Tensor) -> KVPair:
        key = self._heads(key)
        return KVPair(unit_rms(key) if self.normalize_heads else key, self._heads(value))

    def prepare_query(self, rows: Tensor) -> Tensor:
        """Use the Q portion of the packed projection without computing K/V."""
        return self._query(
            F.linear(rows, self.projection.weight[: self.width], self.projection.bias[: self.width])
        )

    def prepare_context(self, rows: Tensor) -> KVPair:
        """Use the K/V portion of the packed projection for reusable context."""
        packed = F.linear(rows, self.projection.weight[self.width :], self.projection.bias[self.width :])
        return self._keys(*packed.chunk(2, dim=-1))

    def attend(self, query: Tensor, context: KVPair, weights: FrequencyWeights | None = None) -> Tensor:
        """Attend with projected tensors and return (*batch, queries, width)."""
        if query.shape[-2] == 0:
            return query.new_empty((*query.shape[:-3], 0, self.width))
        leading = query.shape[:-3]
        # SDPA receives a single flattened batch axis; restore all original
        # leading axes after merging heads, without changing query order.
        query = query.reshape(-1, *query.shape[-3:])
        key = context.key.reshape(-1, *context.key.shape[-3:])
        value = context.value.reshape(-1, *context.value.shape[-3:])
        count = key.shape[-2] if weights is None else weights.count
        scaled = self.scaling(query, count) if self.scaling is not None else query
        if weights is None:
            result = F.scaled_dot_product_attention(scaled, key, value)
        else:
            result = F.scaled_dot_product_attention(
                scaled, key, value, attn_mask=weights.log_bias.to(dtype=query.dtype),
            )
        merged = result.transpose(-3, -2).flatten(-2).reshape(*leading, query.shape[-2], self.width)
        return self.output(merged)

    def forward(self, query: Tensor, context: Tensor, weights: FrequencyWeights | None = None) -> Tensor:
        options = {} if weights is None else {"weights": weights}
        return self.attend(self.prepare_query(query), self.prepare_context(context), **options)


class AttentionLayer(nn.Module):
    """Pre-normalized attention and feedforward residual updates, with a cache path."""

    def __init__(
        self, width: int, heads: int, expansion: int, normalize_heads: bool = False, scale_query: bool = True
    ) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(width, eps=1e-5)
        self.feedforward_norm = nn.RMSNorm(width, eps=1e-5)
        self.attention = Attention(width, heads, normalize_heads, scale_query)
        self.feedforward = FeedForward(width, expansion)

    def _finish(self, rows: Tensor, update: Tensor) -> Tensor:
        combined = rows + update
        return combined + self.feedforward(self.feedforward_norm(combined))

    def forward(self, rows: Tensor, context: Tensor, weights: FrequencyWeights | None = None) -> Tensor:
        normalized = self.attention_norm(rows)
        keys = self.attention_norm(context)
        options = {} if weights is None else {"weights": weights}
        return self._finish(rows, self.attention(normalized, keys, **options))

    def prepare(self, context: Tensor) -> KVPair:
        """Normalize and project the context exactly as in direct forward execution."""
        return self.attention.prepare_context(self.attention_norm(context))

    def query(self, rows: Tensor, context: KVPair, weights: FrequencyWeights | None = None) -> Tensor:
        """Apply the full residual layer using already prepared context K/V."""
        projected = self.attention.prepare_query(self.attention_norm(rows))
        options = {} if weights is None else {"weights": weights}
        return self._finish(rows, self.attention.attend(projected, context, **options))
