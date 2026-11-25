"""Run the neural stages of the Causilo 1.5 model."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import Tensor

from ..nn.cache import BlockCache
from ..nn.ecdf import WeightedECDF
from ..nn.frequency import FrequencyWeights
from ..nn.model import Model
from ..nn.model_config import ROW_LATENTS
from .cache import ContextCache
from .chunks import retain_tiled_direct_outputs
from .column import ColumnRunner, FeatureEmbeddingRunner, column_features, fit_column_context
from .memory import (
    HostSpillMemoryError,
    estimate_table_workspace_bytes,
    minimum_direct_workspace_bytes,
)
from .predictor import PredictionRunner
from .recompute import RecomputeRunner
from .row import RowRunner


class ModelRunner:
    """Run one model's column, row and prediction stages.

    Callers place the model on its device, enter inference mode, and serialize
    predictions on that model. Stage plans belong to each call.
    """

    def __init__(self, model: Model, device: torch.device):
        if any(block.training for block in model.modules()):
            raise ValueError("ModelRunner requires model.eval().")
        self.model = model
        self.device = device
        self.feature_embedding = FeatureEmbeddingRunner(model.feature_embedding, device)
        self.row = RowRunner(model.row, device)
        self.columns = ColumnRunner(model.columns, device)
        self.prediction = PredictionRunner(model.prediction, device)

    def _encode_rows(
        self,
        features: Tensor,
        targets: Tensor | None,
        *,
        context: Tensor | WeightedECDF | None = None,
        cache: BlockCache | None = None,
        weights: FrequencyWeights | None = None,
    ) -> Tensor:
        context_rows = targets.shape[-1] if targets is not None else None
        features = self.feature_embedding(features, targets, context=context, weights=weights)
        latents = None
        last = self.model.config.column_depth - 1
        for index in range(last + 1):
            features = self.columns(
                features, index, context_rows, cache=cache, weights=weights,
            )
            updated = self.row.refine(features, latents, index)
            if index != last:
                latents, features = updated.split((ROW_LATENTS, features.shape[-2]), dim=-2)
                # Keep the latent prefix without retaining the full grid through the next row layer.
                latents = latents.contiguous()
                del updated
        del features, latents
        return self.row.pool(updated)

    def predict(
        self,
        table: Tensor,
        targets: Tensor,
        *,
        num_classes: int,
        weights: FrequencyWeights | None = None,
        contexts=None,
        first_member: int = 0,
    ) -> Tensor:
        """Use stage tiling and fall back to bounded feature replay when necessary."""
        replay = RecomputeRunner(self.model) if self.device.type == "cuda" else None
        options = {} if contexts is None else {"contexts": contexts, "first_member": first_member}
        keep_outputs = False
        if replay is not None:
            shape = tuple(table.shape)
            budget = replay.budget()
            if estimate_table_workspace_bytes(self.model, shape, self.device) > budget:
                # Keep the activation grids on the device, or reuse an existing replay context.
                keep_outputs = (
                    contexts is None and minimum_direct_workspace_bytes(self.model, shape, self.device) <= budget
                )
                if not keep_outputs:
                    return replay(table, targets, num_classes=num_classes, weights=weights, **options)
        try:
            with retain_tiled_direct_outputs() if keep_outputs else nullcontext():
                return self._predict_direct(
                    table.to(self.device), targets.to(self.device), num_classes=num_classes, weights=weights,
                )
        except (torch.cuda.OutOfMemoryError, HostSpillMemoryError):
            if replay is None:
                raise
        # Release the failed forward traceback before running the tiled replay.
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
        return replay(table, targets, num_classes=num_classes, weights=weights, **options)

    def _predict_direct(
        self, table: Tensor, targets: Tensor, *, num_classes: int, weights: FrequencyWeights | None,
    ) -> Tensor:
        context_rows = targets.shape[1]
        assert context_rows <= table.shape[1], "Number of training samples exceeds total samples"
        rows = self._encode_rows(column_features(table), targets, weights=weights)
        return self.prediction(rows, targets, num_classes=num_classes, weights=weights)

    def build_cache(
        self,
        table: Tensor,
        targets: Tensor,
        *,
        num_classes: int,
        weights: FrequencyWeights | None = None,
    ) -> ContextCache:
        """Encode training rows and return their reusable projections."""
        bounded = self.device.type == "cuda" and table.shape[0] == 1
        replay = RecomputeRunner(self.model) if bounded else None
        if replay is not None and replay.needed(tuple(table.shape)):
            return replay.build_cache(table, targets, num_classes=num_classes, weights=weights)
        try:
            return self._build_cache_direct(
                table.to(self.device), targets.to(self.device),
                num_classes=num_classes, weights=weights,
            )
        except torch.cuda.OutOfMemoryError:
            if replay is None:
                raise
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
        return replay.build_cache(table, targets, num_classes=num_classes, weights=weights)

    def _build_cache_direct(
        self, table: Tensor, targets: Tensor, *, num_classes: int, weights: FrequencyWeights | None,
    ) -> ContextCache:
        features = column_features(table)
        context = fit_column_context(features, targets.shape[1], weights)
        weighted_ecdf = isinstance(context, WeightedECDF)
        cache = ContextCache(
            context_shape=table.shape,
            num_classes=num_classes,
            column_ecdf_context=context.values if weighted_ecdf else context,
            column_ecdf_cumulative=context.cumulative if weighted_ecdf else None,
        )
        rows = self._encode_rows(features, targets, context=context, cache=cache.columns, weights=weights)
        cache.prediction = self.prediction.build_cache(rows, targets, weights=weights)
        return cache

    def predict_cached(
        self,
        table: Tensor,
        cache: ContextCache,
        *,
        weights: FrequencyWeights | None = None,
    ) -> Tensor:
        """Evaluate query rows with a prepared cache, in native output order."""
        rows = self._encode_cached_rows(table, cache, weights)
        return self._predict_native_cached(rows, cache, weights)

    def _encode_cached_rows(
        self, table: Tensor, cache: ContextCache, weights: FrequencyWeights | None = None,
    ) -> Tensor:
        """Encode one query chunk against the fitted feature and column context."""
        if not cache.context_shape[1]:
            raise ValueError("No training rows in the supplied cache")
        if table.shape[-1] != cache.context_shape[-1]:
            raise ValueError("Cached feature count differs from the query table")
        context = (
            WeightedECDF(cache.column_ecdf_context, cache.column_ecdf_cumulative)
            if cache.column_ecdf_cumulative is not None else cache.column_ecdf_context
        )
        return self._encode_rows(column_features(table), None, context=context, cache=cache.columns, weights=weights)

    @staticmethod
    def host_prediction_bytes(cache: ContextCache) -> int:
        return sum(
            tensor.nbytes for pair in cache.prediction.layers.values()
            for tensor in (pair.key, pair.value) if tensor.device.type == "cpu"
        )

    def predict_host_cache(self, tables, cache: ContextCache, query_sizes, replay_ceiling: int, *, weights, finish):
        """Use one ensemble group's CPU prediction K/V across independent query chunks."""
        replay = RecomputeRunner(self.model)
        host_bytes = self.host_prediction_bytes(cache)
        members = cache.context_shape[0]
        layer_bytes = max(pair.key.nbytes + pair.value.nbytes for pair in cache.prediction.layers.values())
        # Reject obviously tiled shapes before running even the first feature/row encoder.
        if host_bytes <= replay_ceiling or not replay.native_query_fits(members, max(query_sizes), layer_bytes):
            # Near the dense/replay boundary the ordinary runner decides after encoding the rows.
            outputs = []
            for table in tables:
                output = self.predict_cached(table, cache, weights=weights)
                outputs.append(finish(output).cpu())
                del output, table
            return outputs

        row_bytes = sum(query_sizes) * members * self.model.config.prediction_width * torch.float32.itemsize
        # Keep both sides of a row update plus the current layer with room for tiled activations.
        keep_on_device = 2 * row_bytes + layer_bytes <= replay.budget() // 2
        rows = []
        for table in tables:
            hidden = self._encode_cached_rows(table, cache, weights=weights)
            if not replay.native_query_fits(members, hidden.shape[1], layer_bytes, hidden.nbytes):
                raise MemoryError("The original cache replay may partition this query chunk")
            # Start every chunk's feature and row stages with the native device allocation.
            rows.append(hidden.cpu())
            del table, hidden
        if keep_on_device:
            rows = [hidden.to(self.device) for hidden in rows]
        return replay.predict_cached_chunks(rows, cache, weights=weights, keep_on_device=keep_on_device, finish=finish)

    def _predict_native_cached(
        self, rows: Tensor, cache: ContextCache, weights: FrequencyWeights | None,
    ) -> Tensor:
        replay = RecomputeRunner(self.model) if self.device.type == "cuda" else None
        if replay is not None and self.host_prediction_bytes(cache) > replay.budget():
            return replay.predict_cached(rows, cache, weights=weights)
        try:
            return self.prediction.predict_cached(
                rows, cache.prediction, num_classes=cache.num_classes, weights=weights,
            )
        except torch.cuda.OutOfMemoryError:
            if replay is None:
                raise
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
        return replay.predict_cached(rows, cache, weights=weights)
