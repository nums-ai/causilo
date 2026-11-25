"""Packed multihead attention and its residual layer."""

from __future__ import annotations

import math

from torch import Tensor, nn
from torch.nn import functional as F

from ..cache import KVPair
from ..frequency import FrequencyWeights
from .feedforward import FeedForward
from .functional import scaled_attention, unit_rms
from .normalization import create_norm
from .scaling import QueryScale


class Attention(nn.Module):
    """Packed attention with unit-RMS heads and optional learned query scaling.

    Inputs use (*batch, rows, width). Projected queries and context K/V use
    (*batch, heads, rows, head_width) with matching leading axes.
    """

    def __init__(self, width: int, heads: int, *, scale_query: bool) -> None:
        super().__init__()
        self.width = width
        self.heads = heads
        self.projection = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.scaling = QueryScale(heads, width // heads) if scale_query else None

    def _heads(self, rows: Tensor) -> Tensor:
        return rows.view(*rows.shape[:-1], self.heads, self.width // self.heads).transpose(-3, -2)

    def _normalized_heads(self, projection: Tensor) -> Tensor:
        # Fresh Q and K occupy disjoint slices from V, even when packed together.
        return unit_rms(self._heads(projection))

    def prepare_query(self, rows: Tensor) -> Tensor:
        """Project and normalize Q to attend to prepared context."""
        query = F.linear(rows, self.projection.weight[:self.width], self.projection.bias[:self.width])
        return self._normalized_heads(query)

    def prepare_context(self, rows: Tensor, *, contiguous: bool = False) -> KVPair:
        """Project context into reusable K/V without computing queries."""
        packed = F.linear(
            rows, self.projection.weight[self.width:], self.projection.bias[self.width:],
        )
        if contiguous:
            # Match the K/V packing used by the joint projection in prefill.
            key, value = packed.unflatten(-1, (2, self.width)).movedim(-2, 0).contiguous().unbind(0)
        else:
            key, value = packed.split(self.width, dim=-1)
        return KVPair(self._normalized_heads(key), self._heads(value))

    def project(
        self,
        query: Tensor,
        context: Tensor,
        *,
        separate_kv: bool = False,
    ) -> tuple[Tensor, KVPair]:
        """Project Q/K/V together for a direct or prefill attention step."""
        # Column prefill uses separate K and V projections of the same normalized context.
        # A distinct view selects those projections by identity and shares the context storage.
        values = context.view_as(context) if separate_kv else context
        query, key, value = F._in_projection_packed(
            query, context, values,
            self.projection.weight, self.projection.bias,
        )
        return self._normalized_heads(query), KVPair(self._normalized_heads(key), self._heads(value))

    def attend(
        self, query: Tensor, context: KVPair, weights: FrequencyWeights | None = None,
        *, full_query_count: int | None = None,
    ) -> Tensor:
        """Attend with projected tensors and return (*batch, queries, width)."""
        batch_shape = query.shape[:-3]
        attn_mask = None
        if weights is not None:
            context_rows = context.key.shape[-2]
            bias = weights.key_bias(query)
            if bias.shape != (*batch_shape, context_rows):
                raise ValueError(f"Key bias should have shape {(*batch_shape, context_rows)}, but got {bias.shape}")
            attn_mask = bias.reshape(-1, 1, 1, context_rows)
        batch_size = math.prod(batch_shape)
        queries = query.reshape(batch_size, *query.shape[-3:])
        keys = context.key.reshape(batch_size, *context.key.shape[-3:])
        values = context.value.reshape(batch_size, *context.value.shape[-3:])
        if self.scaling is not None:
            count = keys.shape[-2] if weights is None else weights
            queries = self.scaling(queries, count)
        result = scaled_attention(queries, keys, values, attn_mask, full_query_count=full_query_count)
        result = result.reshape(query.shape)
        merged = result.transpose(-3, -2).contiguous().view(*batch_shape, query.shape[-2], self.width)
        return self.output(merged)

    def forward(self, query: Tensor, context: Tensor, *, full_query_count: int | None = None) -> Tensor:
        projected, keys = self.project(query, context)
        return self.attend(projected, keys, full_query_count=full_query_count)


class AttentionLayer(nn.Module):
    """Pre-normalized attention and feedforward residual updates."""

    def __init__(self, width: int, heads: int, expansion: int, *, scale_query: bool) -> None:
        super().__init__()
        self.attention = Attention(width, heads, scale_query=scale_query)
        self.feedforward = FeedForward(width, expansion)
        self.attention_norm = create_norm(width)
        self.feedforward_norm = create_norm(width)

    def _finish(self, rows: Tensor, update: Tensor) -> Tensor:
        combined = rows + update
        return combined + self.feedforward(self.feedforward_norm(combined))

    def prepare(self, context: Tensor, *, contiguous: bool = False) -> KVPair:
        """Prepare K/V for replay without advancing any query rows."""
        return self.attention.prepare_context(self.attention_norm(context), contiguous=contiguous)

    def query(self, rows: Tensor, context: KVPair, weights: FrequencyWeights | None = None) -> Tensor:
        """Apply the residual layer using reusable context K/V."""
        query = self.attention.prepare_query(self.attention_norm(rows))
        update = self.attention.attend(query, context, weights)
        del query
        return self._finish(rows, update)

    def forward(
        self,
        rows: Tensor,
        context: Tensor | None = None,
        *,
        context_rows: int | None = None,
        weights: FrequencyWeights | None = None,
        separate_kv: bool = False,
    ) -> Tensor:
        query, keys = self._project(rows, context, context_rows, separate_kv=separate_kv)
        update = self.attention.attend(query, keys, weights)
        del query, keys
        return self._finish(rows, update)

    def prefill(
        self,
        rows: Tensor,
        context: Tensor | None = None,
        *,
        context_rows: int | None = None,
        weights: FrequencyWeights | None = None,
    ) -> tuple[Tensor, KVPair]:
        """Advance rows and return their layer's context K/V."""
        query, keys = self._project(rows, context, context_rows)
        update = self.attention.attend(query, keys, weights)
        del query
        return self._finish(rows, update), keys

    def _project(
        self,
        rows: Tensor,
        context: Tensor | None,
        context_rows: int | None,
        *,
        separate_kv: bool = False,
    ) -> tuple[Tensor, KVPair]:
        normalized = self.attention_norm(rows)
        if context_rows is not None:
            context = normalized[..., :context_rows, :]
        else:
            context = self.attention_norm(context)
        return self.attention.project(normalized, context, separate_kv=separate_kv)
