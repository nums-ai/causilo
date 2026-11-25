"""Reusable context state for Causilo 1.5 inference."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..nn.cache import BlockCache


def cache_sizes(model, rows: int, features: int, device: torch.device, *, weighted: bool) -> tuple[int, int, int]:
    """Size the column, prediction and ECDF state for one native member."""
    config = model.config
    latents = model.columns.layers[0].latents.shape[0]
    column = 2 * features * latents * config.width * config.column_depth * torch.float32.itemsize
    itemsize = torch.float16.itemsize if device.type == "cuda" else torch.float32.itemsize
    prediction = 2 * rows * config.prediction_width * config.prediction_depth * itemsize
    ranks = rows * features * torch.float32.itemsize
    if weighted:
        ranks += (rows + 1) * features * torch.int64.itemsize
    return column, prediction, ranks


@dataclass
class ContextCache:
    """Column and prediction projections for a group of fitted ensemble members."""

    context_shape: tuple[int, int, int]
    num_classes: int
    columns: BlockCache = field(default_factory=BlockCache)
    prediction: BlockCache = field(default_factory=BlockCache)
    column_ecdf_context: Tensor | None = None
    column_ecdf_cumulative: Tensor | None = None

    def members(self, section: slice) -> ContextCache:
        """Select an ordered member range, for placing a saved cache on other devices."""
        members, rows, width = self.context_shape
        return ContextCache(
            context_shape=(len(range(*section.indices(members))), rows, width),
            num_classes=self.num_classes,
            columns=self.columns[section],
            prediction=self.prediction[section],
            column_ecdf_context=(
                self.column_ecdf_context[section] if self.column_ecdf_context is not None else None
            ),
            column_ecdf_cumulative=(
                self.column_ecdf_cumulative[section] if self.column_ecdf_cumulative is not None else None
            ),
        )

    @property
    def nbytes(self) -> int:
        """Memory occupied by cached tensors."""
        return sum(
            tensor.nbytes
            for cache in (self.columns, self.prediction)
            for pair in cache.layers.values()
            for tensor in (pair.key, pair.value)
        ) + sum(
            tensor.nbytes
            for tensor in (self.column_ecdf_context, self.column_ecdf_cumulative)
            if tensor is not None
        )

    def to(self, device) -> ContextCache:
        """Move the cached projections to the requested device."""
        return ContextCache(
            context_shape=self.context_shape,
            num_classes=self.num_classes,
            columns=self.columns.to(device),
            prediction=self.prediction.to(device),
            column_ecdf_cumulative=(
                self.column_ecdf_cumulative.to(device) if self.column_ecdf_cumulative is not None else None
            ),
            column_ecdf_context=(
                self.column_ecdf_context.to(device=device, dtype=torch.float32)
                if self.column_ecdf_context is not None
                else None
            ),
        )
