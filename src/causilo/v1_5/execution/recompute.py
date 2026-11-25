"""Bound inference memory with column summaries and replayable row latents."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
from torch import Tensor

from ..nn.cache import BlockCache, KVPair
from ..nn.ecdf import WeightedECDF, ecdf_ranks
from ..nn.frequency import FrequencyWeights
from ..nn.model_config import ROW_LATENTS
from .cache import ContextCache, cache_sizes
from .column import column_features, fit_column_context
from .memory import (
    CUDA_MAX_BATCH_ITEMS,
    EXECUTION_MEMORY_FRACTION,
    Stage,
    available_bytes,
    estimate_table_workspace_bytes,
    is_allocation_error,
)
from .precision import run_inference
from .transfers import host_cache_copies

logger = logging.getLogger(__name__)


class ReplayContext:
    """Host copy of the bounded encoder output for one native member."""

    def __init__(self, rows, summaries, ecdf):
        self.rows = rows.cpu()
        self.summaries = tuple(layer.to("cpu") for layer in summaries)
        self.ecdf = ecdf
        self.device_bytes = self.rows.nbytes + sum(layer.key.nbytes + layer.value.nbytes for layer in self.summaries)

    def restore(self, device):
        return self.rows.to(device), [layer.to(device) for layer in self.summaries]


class RecomputeRunner:
    """Retain compact context state; materialize feature grids only inside bounded tiles.

    Column passes span all context rows for a feature tile. Row passes replay
    earlier broadcasts for a row tile. Prediction retains one layer's context K/V,
    then updates rows from that immutable snapshot. State belongs to one call.
    """

    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.summaries: list[KVPair] = []
        self.broadcasts: list[Tensor] = []
        self._tile_retried = False

    def budget(self) -> int:
        return int(available_bytes(self.device) * EXECUTION_MEMORY_FRACTION)

    def needed(self, shape: tuple[int, ...]) -> bool:
        """Reserve full-grid projection/FFN workspace before uploading model inputs."""
        return estimate_table_workspace_bytes(self.model, shape, self.device) > self.budget()

    def _run(self, stage: Stage, operation: Callable, *args):
        return run_inference(self.device, operation, *args, stage=stage)

    def _limit(self, total: int, per_item: int, pending: int) -> int:
        room = self.budget() - pending
        if room <= 0:
            raise torch.cuda.OutOfMemoryError("Retained inference state exceeds the recomputation budget")
        limit = max(1, min(total, CUDA_MAX_BATCH_ITEMS, room // max(1, per_item)))
        return self._align(limit) if limit < total else limit

    def _align(self, limit: int) -> int:
        """Favor aligned CUDA batches without padding or changing attended context."""
        if limit >= 128:
            return limit // 128 * 128
        return limit

    def _map(
        self, total: int, per_item: int, operation: Callable, *, pending: int, axis: int = 1,
    ):
        """Commit successful tiles once; release failed temporaries before shrinking."""
        limit = self._limit(total, per_item, pending)
        destinations = None
        start = failures = 0

        def attempt(section):
            result = operation(section)
            outputs = result if isinstance(result, tuple) else (result,)
            buffers = destinations
            if buffers is None:
                buffers = []
                for value in outputs:
                    shape = list(value.shape)
                    shape[axis] = total
                    buffers.append(value.new_empty(shape))
                buffers = tuple(buffers)
            return outputs, buffers

        while start < total:
            section = slice(start, min(total, start + limit))
            try:
                outputs, buffers = attempt(section)
            except torch.cuda.OutOfMemoryError:
                self._tile_retried = True
                failures += 1
            else:
                destinations = buffers
                for output, destination in zip(outputs, destinations, strict=True):
                    destination.narrow(axis, start, section.stop - start).copy_(output)
                outputs = buffers = output = destination = None
                start = section.stop
                continue
            if section.stop - start == 1 or failures >= 7:
                raise torch.cuda.OutOfMemoryError("Minimum recomputation tile or retry limit reached")
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
            limit = self._align((section.stop - start) // 2)
            logger.info("Retrying inference tile with %d items", limit)
        return destinations[0] if len(destinations) == 1 else destinations

    def _embed(self, table: Tensor, ranks: Tensor, targets: Tensor | None = None) -> Tensor:
        column = self.model.feature_embedding
        values = column_features(table.to(self.device))
        if targets is not None:
            targets = targets.to(self.device).float()
        ranks = ranks.to(self.device).transpose(1, 2).unsqueeze(-1)

        def embed():
            embeddings = column.project_features(values, ranks)
            if targets is not None:
                labels = targets.unsqueeze(1).expand(-1, values.shape[1], -1)
                embeddings = column.add_context_targets(embeddings, labels)
            return embeddings

        return self._run(Stage.FEATURE_EMBEDDING, embed).transpose(1, 2)

    def _column(self, features: Tensor, index: int, section: slice) -> Tensor:
        summary = self.summaries[index]
        selected = summary[:, section]
        block = self.model.columns.layers[index]
        return self._run(
            Stage.COLUMN,
            block.query,
            features.transpose(1, 2).float(),
            selected,
        ).transpose(1, 2)

    def _replay(self, rows: slice, features: slice, depth: int) -> Tensor:
        state = self._embed(self.table[:, rows, features], self.rank_features[:, rows, features], self.targets[:, rows])
        for index in range(depth):
            state = self._column(state, index, features)
            state = self._run(
                Stage.ROW,
                self.model.row.layers[index].replay_features,
                state,
                self.broadcasts[index][:, rows],
            )
        return state

    def _prepare_column(self, index: int) -> KVPair:
        block = self.model.columns.layers[index]
        width = self.model.config.width
        # Include input/output activations, FFN and a conservative gather-score bound.
        per_feature = self.context_rows * 4 * (
            16 * width + 2 * block.gather.attention.heads * block.latents.shape[0]
        )
        pending = 2 * self.features * block.latents.shape[0] * width * 4

        def prepare(section):
            values = self._replay(slice(0, self.context_rows), section, index).transpose(1, 2).float()

            cache = self._run(Stage.COLUMN, block.prepare, values, self.weights)
            return cache.key, cache.value

        key, value = self._map(self.features, per_feature, prepare, pending=pending)
        return KVPair(key, value)

    def _row_workspace(self) -> int:
        return 4 * (self.features + ROW_LATENTS) * self.model.config.width * 24

    def _pool(self, latents: Tensor) -> Tensor:
        row = self.model.row
        return self._run(Stage.ROW, row.pool_latents, latents)

    def _encode_context(self) -> Tensor:
        row = self.model.row
        carried = None
        all_features = slice(0, self.features)
        for index, block in enumerate(row.layers):
            logger.info("Recompute column/row cycle %d/%d", index + 1, len(row.layers))
            self.summaries.append(self._prepare_column(index))
            final = index + 1 == len(row.layers)

            def advance(section, index=index, block=block, carried=carried, final=final):
                values = self._column(self._replay(section, all_features, index), index, all_features)
                latents = (
                    row.latents.to(values).expand(1, values.shape[1], row.num_latents, row.width)
                    if carried is None
                    else carried[:, section]
                )
                if final:
                    latents, _ = self._run(Stage.ROW, block, latents, values)
                    return self._pool(latents)
                return self._run(Stage.ROW, block.forward_with_replay, latents, values)

            pending = self.context_rows * row.num_latents * row.width * 4 * (1 if final else 2)
            result = self._map(self.context_rows, self._row_workspace(), advance, pending=pending)
            del advance
            if final:
                context = result
            else:
                carried, broadcast = result
                self.broadcasts.append(broadcast)
            result = None
        carried = broadcast = None
        self.broadcasts.clear()
        return context

    def _reuse_context(self, contexts, member):
        saved = contexts.get(member) if contexts is not None else None
        if saved is not None:
            try:
                if saved.device_bytes <= self.budget():
                    self._prepare_ranks(saved.ecdf)
                    rows, self.summaries = saved.restore(self.device)
                    return rows
            except (MemoryError, RuntimeError) as error:
                if not is_allocation_error(error):
                    raise
            contexts[member] = None
            del saved

        ecdf = self._prepare_ranks()
        if contexts is None or member in contexts:
            del ecdf
            return self._encode_context()

        self._tile_retried = False
        rows = self._encode_context()
        # Keep only a completed context. A failed optional copy leaves the native result usable.
        contexts[member] = None
        if not self._tile_retried:
            try:
                contexts[member] = ReplayContext(rows, self.summaries, ecdf)
            except (MemoryError, RuntimeError) as error:
                if not is_allocation_error(error):
                    raise
        return rows

    def _encode(self, contexts=None, member=0) -> Tensor:
        context = self._reuse_context(contexts, member)
        row = self.model.row
        all_features = slice(0, self.features)
        query_table = self.table[:, self.context_rows:]
        # A replay hit holds only query ranks; a fresh encoding also holds the context prefix.
        query_ranks = self.rank_features[:, -query_table.shape[1]:]

        def query_rows(section):
            values = self._embed(query_table[:, section], query_ranks[:, section])
            latents = row.latents.to(values).expand(1, values.shape[1], row.num_latents, row.width)
            for index, block in enumerate(row.layers):
                values = self._column(values, index, all_features)
                latents, values = self._run(Stage.ROW, block, latents, values)
            return self._pool(latents)

        queries = self.table.shape[1] - self.context_rows
        query = self._map(
            queries,
            self._row_workspace(),
            query_rows,
            pending=queries * self.model.config.prediction_width * 4,
        )
        self.summaries.clear()
        return torch.cat((context, query), dim=1)

    def _add_targets(self, rows: Tensor) -> Tensor:
        predictor = self.model.prediction
        width = rows.shape[-1]

        def add_labels(section):
            values = rows[:, section]
            if section.start >= self.targets.shape[1]:
                return values
            labels = self.targets[:, section].to(self.device).float()
            return self._run(Stage.PREDICTION, predictor.add_context_targets, values, labels)

        return self._map(rows.shape[1], width * 4 * 8, add_labels, pending=rows.numel() * rows.element_size())

    def _prediction_keys(self, block, rows: Tensor) -> KVPair:
        width = rows.shape[-1]

        def prepare(section):
            cache = self._run(Stage.PREDICTION, block.prepare, rows[:, section])
            return cache.key, cache.value

        key, value = self._map(
            self.context_rows, width * 4 * 10, prepare, axis=2, pending=2 * self.context_rows * width * 4,
        )
        return KVPair(key, value)

    @staticmethod
    def _prediction_workspace(members, queries, width, output_width):
        return members * width * 4 * 32, members * queries * output_width * 4

    def native_query_fits(self, members: int, queries: int, largest_layer_bytes: int, pooled_bytes: int = 0) -> bool:
        """Check the original full query tile before and after encoding its actual pooled rows."""
        width = self.model.config.prediction_width
        output_width = max(width, self.model.prediction.head.output.out_features)
        per_row, pending = self._prediction_workspace(members, queries, width, output_width)
        # Leave space for a staged K/V and a subsequent layer's live rows, including allocation rounding.
        reserve = 2 * (largest_layer_bytes + pooled_bytes)
        return queries <= CUDA_MAX_BATCH_ITEMS and per_row * queries + pending + reserve <= self.budget()

    def _advance_prediction(self, block, rows: Tensor, cache: KVPair, *, start=0, final=False, whole=False) -> Tensor:
        predictor = self.model.prediction
        members, size, width = rows.shape

        def update(section):
            values = rows[:, start + section.start : start + section.stop].to(self.device)

            def apply():
                result = block.query(values, cache, weights=self.weights)
                return predictor.head(result) if final else result

            return self._run(Stage.PREDICTION, apply)

        total = size - start
        out_width = predictor.head.output.out_features if final else width
        per_row, pending = self._prediction_workspace(members, total, width, out_width)
        if whole:
            if self._limit(total, per_row, pending) < total:
                raise MemoryError("Reusing K/V would change the native query tile")
            return update(slice(0, total)).contiguous()
        return self._map(total, per_row, update, pending=pending)

    def _predict(self, rows: Tensor) -> Tensor:
        blocks = self.model.prediction.layers
        rows = self._add_targets(rows)
        for index, block in enumerate(blocks):
            logger.info("Recompute prediction layer %d/%d", index + 1, len(blocks))
            final = index + 1 == len(blocks)
            cache = self._prediction_keys(block, rows)
            rows = self._advance_prediction(block, rows, cache, start=self.context_rows if final else 0, final=final)
            del cache
        return rows

    def _prepare(self, table: Tensor, targets: Tensor, weights: FrequencyWeights | None) -> None:
        self.table = table.detach().cpu()
        self.targets = targets.detach().cpu()
        self.weights = weights
        self.context_rows, self.features = self.targets.shape[1], self.table.shape[2]

    def _prepare_ranks(self, context: Tensor | WeightedECDF | None = None) -> Tensor | WeightedECDF:
        if context is None:
            values = column_features(self.table)
            context = fit_column_context(values, self.context_rows, self.weights)
        else:
            values = column_features(self.table[:, self.context_rows:])
        ranks = ecdf_ranks(values, context)
        self.rank_features = ranks.squeeze(-1).transpose(1, 2)
        return context

    def _clear_state(self) -> None:
        self.summaries.clear()
        self.broadcasts.clear()
        self.table = self.targets = self.rank_features = self.weights = None

    def build_cache(
        self, table: Tensor, targets: Tensor, *, num_classes: int, weights: FrequencyWeights | None = None,
    ) -> ContextCache:
        """Build one native member's context in tiles and spill each K/V layer to host."""
        if table.shape[0] != 1 or not 0 <= num_classes <= self.model.config.max_classes:
            raise ValueError("Bounded cache prefill requires one native ensemble member")
        if table.shape[1] != targets.shape[1] or not targets.shape[1]:
            raise ValueError("Bounded cache prefill requires matching nonempty context rows and targets")
        rows, features = table.shape[1:]
        column_bytes, prediction_bytes, context_bytes = cache_sizes(
            self.model, rows, features, self.device, weighted=weights is not None,
        )
        if column_bytes + prediction_bytes + context_bytes > available_bytes(torch.device("cpu")) * EXECUTION_MEMORY_FRACTION:
            raise MemoryError("Bounded native context cache exceeds the available host memory")
        try:
            self._prepare(table, targets, weights)
            context = self._prepare_ranks()
            rows = self._encode_context()
            with host_cache_copies(self.device, column_bytes + prediction_bytes) as to_host:
                column_cache = BlockCache({index: to_host(entry) for index, entry in enumerate(self.summaries)})
                self.summaries.clear()
                self.table = self.rank_features = None
                rows = self._add_targets(rows)
                self.targets = None
                prediction_cache = BlockCache()
                blocks = self.model.prediction.layers
                for index, block in enumerate(blocks):
                    cache = self._prediction_keys(block, rows)
                    prediction_cache.layers[index] = to_host(cache)
                    if index + 1 != len(blocks):
                        rows = self._advance_prediction(block, rows, cache)
                    del cache
            weighted = isinstance(context, WeightedECDF)
            return ContextCache(
                context_shape=tuple(table.shape),
                num_classes=num_classes,
                columns=column_cache,
                prediction=prediction_cache,
                column_ecdf_context=context.values if weighted else context,
                column_ecdf_cumulative=context.cumulative if weighted else None,
            )
        finally:
            self._clear_state()

    def predict_cached_chunks(
        self, chunks: list[Tensor], cache: ContextCache, *, weights, keep_on_device: bool, finish: Callable,
    ) -> list[Tensor]:
        """Replay one host K/V layer across a window of independent query chunks."""
        self.weights = weights
        try:
            blocks = self.model.prediction.layers
            for index, block in enumerate(blocks):
                layer = cache.prediction.layers[index].to(self.device)
                final = index + 1 == len(blocks)
                for chunk, rows in enumerate(chunks):
                    result = self._advance_prediction(block, rows, layer, final=final, whole=True)
                    if final:
                        if cache.num_classes:
                            result = result[..., :cache.num_classes]
                        chunks[chunk] = finish(result).cpu()
                    else:
                        chunks[chunk] = result if keep_on_device else result.cpu()
                    del result
                del layer
            return chunks
        finally:
            self._clear_state()

    def predict_cached(
        self, rows: Tensor, cache: ContextCache, *, weights: FrequencyWeights | None = None,
    ) -> Tensor:
        """Replay cached queries with only the active prediction layer on the device."""
        self.weights = weights
        try:
            blocks = self.model.prediction.layers
            for index, block in enumerate(blocks):
                layer = cache.prediction.layers[index].to(self.device)
                rows = self._advance_prediction(block, rows, layer, final=index + 1 == len(blocks))
                del layer
            return rows[..., :cache.num_classes] if cache.num_classes else rows
        finally:
            self._clear_state()

    def __call__(
        self, table: Tensor, targets: Tensor, *, num_classes: int, weights: FrequencyWeights | None = None,
        contexts=None, first_member: int = 0,
    ) -> Tensor:
        if targets.shape[1] < 1 or table.shape[1] <= targets.shape[1]:
            raise ValueError("Recomputation requires context and query rows")
        if num_classes > self.model.config.max_classes:
            raise ValueError("Bounded replay supports native class targets only")
        predictions = []
        try:
            for index in range(table.shape[0]):
                self._prepare(table[index : index + 1], targets[index : index + 1], weights)
                predictions.append(self._predict(self._encode(contexts, first_member + index)).cpu())
            outputs = predictions[0] if len(predictions) == 1 else torch.cat(predictions)
            return outputs[..., :num_classes] if num_classes else outputs
        finally:
            self._clear_state()
