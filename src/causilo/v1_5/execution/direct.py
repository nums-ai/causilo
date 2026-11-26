"""Single-device execution for the Causilo 1.5 ensemble."""

from __future__ import annotations

import gc
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import torch

from ..nn.layers.scaling import reuse_length_gains
from ..nn.model_config import COLUMN_LATENTS
from .memory import available_bytes, is_allocation_error, planning_budget
from .quantiles import regression_stats

if TYPE_CHECKING:
    from ..data.dataset import FeatureTable, PreparedDataset
    from ..nn.model import Model
    from .cache import ContextCache

_ENSEMBLE_BATCH_SIZE = 8
_FLOAT32_BYTES = np.dtype(np.float32).itemsize
_Chunk = TypeVar("_Chunk")


def concatenate_parts(parts: list[np.ndarray], *, axis: int) -> np.ndarray:
    """Join independent results, preserving a complete result's storage."""
    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=axis)


@dataclass(frozen=True, slots=True)
class ReducedMean:
    """One final regression mean per query row, already combined across members."""

    values: np.ndarray


def _host_output(output, finish=None) -> np.ndarray | ReducedMean:
    reduced = finish(output) if finish is not None else None
    if reduced is not None:
        means = reduced.cpu().numpy()
        if np.isfinite(means).all():
            return ReducedMean(means)
        # Keep the standard target validation and error behavior for nonfinite forecasts.
    return output.float().cpu().numpy()


def predict_chunks(
    predict_chunk: Callable[[slice], _Chunk], *, rows: int, batch_size: int, device: torch.device,
) -> Iterator[_Chunk]:
    """Retry allocation failures at the same offset with a smaller batch."""
    start = 0
    while start < rows:
        section = slice(start, min(rows, start + batch_size))
        try:
            result = predict_chunk(section)
        except (MemoryError, RuntimeError) as exc:
            if section.stop - start <= 1 or not is_allocation_error(exc):
                raise
        else:
            yield result
            del result
            start = section.stop
            continue
        gc.collect()
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        batch_size = (section.stop - start) // 2


class PredictionLock:
    """Share synchronization across runners, recreating an unlocked lock on load."""

    def __init__(self):
        self.lock = Lock()

    def __reduce__(self):
        return type(self), ()


class EnsembleRunner:
    """Run ordered, bounded groups of ensemble members on the selected device."""

    def __init__(
        self, model: Model, device: torch.device, frequencies=None, *, memory_budget_bytes=None,
        prediction_lock=None,
    ):
        from ..nn.frequency import FrequencyWeights
        from .runner import ModelRunner

        self.model = model
        self.device = device
        self.memory_budget_bytes = memory_budget_bytes

        self._model_runner = ModelRunner(model, device)
        self._prediction_lock = PredictionLock() if prediction_lock is None else prediction_lock
        self._length_gains = {}
        self._stream = None
        self.weights = (
            FrequencyWeights.prepare(torch.as_tensor(frequencies, device=device)) if frequencies is not None else None
        )

    def __getstate__(self):
        """Keep fitted data and shared references, excluding stream-local execution caches."""
        return self.__dict__ | {"_stream": None, "_length_gains": {}}

    @contextmanager
    def _inference(self):
        cuda = self.device.type == "cuda"
        device_context = torch.cuda.device(self.device) if cuda else nullcontext()
        with (
            self._prediction_lock.lock, device_context, torch.inference_mode(),
            planning_budget(self.device, self.memory_budget_bytes),
        ):
            if cuda:
                stream = torch.cuda.current_stream(self.device)
                if self._stream is None or stream.cuda_stream != self._stream.cuda_stream:
                    self._length_gains.clear()
                    self._stream = stream
            with reuse_length_gains(self._length_gains if cuda else None):
                yield

    def query_batch_size(self, dataset: PreparedDataset, rows: int, *, cached: bool, output_width=0) -> int:
        host_limit = self.host_query_limit(dataset, rows, cached=cached, output_width=output_width)
        return self.query_device_limit(dataset, host_limit, cached=cached, output_width=output_width)

    def query_chunks(self, prepare, predict, *, rows: int, batch_size: int) -> Iterator[np.ndarray]:
        return predict_chunks(lambda section: predict(prepare(section)), rows=rows, batch_size=batch_size, device=self.device)

    @staticmethod
    def host_query_limit(dataset: PreparedDataset, rows: int, *, cached: bool, output_width=0) -> int:
        """Estimate host memory for the full context and signed query views."""
        train_rows, width = dataset.training.shape
        members = len(dataset.ensemble.feature_signs)
        feature_bytes = (2 * members + dataset.view_count) * width * _FLOAT32_BYTES
        # Uncached batches also hold the sorted context used for cell ranks.
        fixed_bytes = 0 if cached else train_rows * (feature_bytes + members * width * _FLOAT32_BYTES)
        # Account for member results, their aligned order statistics, and the float64 reduction.
        bytes_per_row = feature_bytes + (3 * members + 2) * output_width * _FLOAT32_BYTES
        available = available_bytes(torch.device("cpu"))
        return max(1, min(rows, (available - fixed_bytes) // bytes_per_row))

    def query_device_limit(
        self, dataset: PreparedDataset, rows: int, *, cached: bool, members: int | None = None, output_width=0,
    ) -> int:
        """Bound query tiles that otherwise keep a full context or query grid on the device."""
        if self.device.type != "cuda" or (self.memory_budget_bytes is None and not output_width):
            return rows
        from .memory import query_row_workspace_bytes

        width = dataset.training.shape[1]
        budget = self._working_memory()
        if members is None:
            members = len(dataset.ensemble.feature_signs)
        if output_width:
            # Budget for the head and the interpolation intermediates, including both tails.
            per_row = 8 * min(members, _ENSEMBLE_BATCH_SIZE) * output_width * _FLOAT32_BYTES
            rows = max(1, min(rows, budget // per_row))
            if self.memory_budget_bytes is None:
                return rows
        # The CUDA regressor already plans memory for each stage of its replay.
        # Size its resident input and compact query buffers here, without
        # repeating the full context just to shrink the replay's outer batch.
        if not cached and not self.model.config.max_classes:
            bytes_per_row = _FLOAT32_BYTES * max(width, self.model.config.prediction_width * 2)
            return max(1, min(rows, budget // bytes_per_row))
        train_rows = 0 if cached else len(dataset.training)
        if cached:
            resident_members = min(members, _ENSEMBLE_BATCH_SIZE)
        else:
            resident_members = min(members, self._members_per_batch(train_rows + rows, width, budget))
        input_rows = budget // (resident_members * width * _FLOAT32_BYTES) - train_rows
        # A native direct pass can already tile independent feature/row axes.
        # Do not replay its context merely to fit the untiled grid estimate.
        if not cached:
            return max(1, min(rows, input_rows))
        per_row = resident_members * query_row_workspace_bytes(self.model, width, self.device)
        return max(1, min(rows, input_rows, budget // per_row))

    def _working_memory(self) -> int:
        from .memory import EXECUTION_MEMORY_FRACTION

        with planning_budget(self.device, self.memory_budget_bytes):
            return int(available_bytes(self.device) * EXECUTION_MEMORY_FRACTION)

    def _members_per_batch(self, rows: int, width: int, budget: int) -> int:
        from .memory import estimate_table_workspace_bytes

        per_member = estimate_table_workspace_bytes(self.model, (1, rows, width), self.device)
        return min(_ENSEMBLE_BATCH_SIZE, max(1, budget // per_member))

    def member_capacity(self, rows: int, width: int) -> int:
        if self.device.type == "cuda":
            return self._members_per_batch(rows, width, self._working_memory())
        return _ENSEMBLE_BATCH_SIZE

    def _ensemble_batch_count(self, shape: tuple[int, ...]) -> int:
        members, rows, width = shape
        batch_size = self.member_capacity(rows, width)
        return (members + batch_size - 1) // batch_size

    def replay_context_bytes(self, dataset: PreparedDataset) -> int:
        """Reserve host room for each member's pooled rows and column projections."""
        config = self.model.config
        rows, features = dataset.training.shape
        columns = 2 * features * config.column_depth * COLUMN_LATENTS * config.width
        bytes_per_member = (rows * (config.prediction_width + features) + columns) * _FLOAT32_BYTES
        if self.weights is not None:
            bytes_per_member += (rows + 1) * features * np.dtype(np.int64).itemsize
        return bytes_per_member * len(dataset.ensemble.feature_signs)

    def can_retain_replay(self, dataset: PreparedDataset) -> bool:
        return self.device.type == "cuda" and (
            self.replay_context_bytes(dataset) <= available_bytes(torch.device("cpu")) // 4
        )

    def build_cache(self, dataset: PreparedDataset, *, train: FeatureTable | None = None) -> tuple[ContextCache, ...]:
        """Prefill ensemble members or ECOC rows under one cache placement budget."""
        if train is None:
            train = dataset.training_table()
        rows, width = train.shape
        code_rows = dataset.code_rows
        items = code_rows or len(dataset.ensemble.feature_signs)
        caches = []
        with self._inference():
            groups = code_rows or self._ensemble_batch_count((items, rows, width))
            batch_size = (items + groups - 1) // groups
            # Leave room for later groups and query activations.
            device_budget = available_bytes(self.device) // 4 if self.device.type == "cuda" else 0
            start = 0
            while start < items:
                stop = min(start + batch_size, items)
                try:
                    cache = self._prefill(dataset, train, slice(start, stop))
                    if self.device.type == "cuda":
                        size = cache.nbytes
                        if size > device_budget:
                            cache = cache.to("cpu")
                        else:
                            device_budget -= size
                except (MemoryError, RuntimeError) as exc:
                    if stop - start <= 1 or not is_allocation_error(exc):
                        raise
                else:
                    caches.append(cache)
                    start = stop
                    continue
                cache = None
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                batch_size = (stop - start) // 2
        return tuple(caches)

    def _prefill(self, dataset: PreparedDataset, train: FeatureTable, section: slice) -> ContextCache:
        code_row = section.start if dataset.code_rows else None
        members = section if code_row is None else slice(None)
        # A bounded prefill reads host context in tiles when the dense path cannot fit.
        x = torch.as_tensor(dataset.signed_views(train, members))
        y = torch.as_tensor(dataset.targets_for(members, code_row=code_row), dtype=torch.float32)
        return self._model_runner.build_cache(
            x, y, num_classes=dataset.num_classes, weights=self.weights,
        )

    def place_caches(self, caches: tuple[ContextCache, ...]) -> tuple[ContextCache, ...]:
        """Keep restored groups on the execution device when they fit comfortably."""
        if self.device.type != "cuda":
            return tuple(cache.to(self.device) for cache in caches)
        with planning_budget(self.device, self.memory_budget_bytes):
            budget = available_bytes(self.device) // 4
            placed = []
            for index, cache in enumerate(caches):
                size = cache.nbytes
                if size > budget:
                    placed.append(cache)
                    continue
                try:
                    moved = cache.to(self.device)
                except (MemoryError, RuntimeError) as error:
                    if not is_allocation_error(error):
                        raise
                    # Saved groups already own usable CPU projections.
                    placed.extend(caches[index:])
                    break
                placed.append(moved)
                budget -= size
        return tuple(placed)

    def validate_caches(self, dataset: PreparedDataset, caches: tuple[ContextCache, ...]) -> None:
        from .cache import ContextCache

        classes = dataset.num_classes
        code_rows = dataset.code_rows
        if code_rows and (not isinstance(caches, tuple) or len(caches) != code_rows):
            raise ValueError("Saved 1.5 cache must contain one context per ECOC code row")
        items = code_rows or len(dataset.ensemble.feature_signs)
        rows, width = dataset.training.shape
        if not isinstance(caches, tuple) or not caches or any(
            not isinstance(cache, ContextCache)
            or cache.context_shape[0] < 1 or cache.context_shape[1:] != (rows, width)
            or cache.num_classes != classes
            for cache in caches
        ) or sum(cache.context_shape[0] for cache in caches) != items:
            raise ValueError("Saved 1.5 cache layout does not match the fitted table and ensemble")

    def predict_cached(
        self,
        dataset: PreparedDataset,
        query: FeatureTable,
        caches: tuple[ContextCache, ...],
        *,
        output_type=None,
        levels=None,
        finish=None,
    ) -> np.ndarray | ReducedMean:
        predictions = []
        start = 0
        with self._inference():
            for cache in caches:
                stop = start + cache.context_shape[0]
                x = torch.as_tensor(dataset.signed_views(query, slice(start, stop)), device=self.device)
                out = self._model_runner.predict_cached(x, cache, weights=self.weights)
                if output_type is not None:
                    out = regression_stats(out, output_type, levels)
                result = _host_output(out, finish if len(caches) == 1 else None)
                del x, out
                if isinstance(result, ReducedMean):
                    return result
                predictions.append(result)
                start = stop
        return concatenate_parts(predictions, axis=0)

    def _cache_replay_ceiling(self) -> int:
        from .memory import EXECUTION_MEMORY_FRACTION

        # The original replay budget can only decrease from this absolute process limit.
        limit = self.memory_budget_bytes
        if limit is None:
            total = torch.cuda.get_device_properties(self.device).total_memory
            limit = total * torch.cuda.get_per_process_memory_fraction(self.device)
        return int(limit * EXECUTION_MEMORY_FRACTION)

    def cached_query_window(
        self, dataset, caches, batch_size: int, rows: int, *, levels=None, concurrent_lanes: int = 1,
    ) -> int:
        """Bound the preprocessed queries, pooled rows and outputs retained on the host."""
        if self.device.type != "cuda" or rows <= batch_size:
            return 0
        budget = self._cache_replay_ceiling()
        if not any(self._model_runner.host_prediction_bytes(cache) > budget for cache in caches):
            return 0
        from .memory import EXECUTION_MEMORY_FRACTION

        group = max(cache.context_shape[0] for cache in caches)
        width = dataset.training.shape[1]
        members = len(dataset.ensemble.feature_signs)
        # Include the raw head and all requested member quantiles, including complementary levels.
        outputs = max(self.model.prediction.head.output.out_features, len(levels) if levels is not None else 0)
        # One signed group plus stored chunks and the source, aligned and final member outputs.
        bytes_per_row = _FLOAT32_BYTES * (
            (group + dataset.view_count) * width + group * self.model.config.prediction_width + 3 * members * outputs
        )
        host_room = int(available_bytes(torch.device("cpu")) * EXECUTION_MEMORY_FRACTION) // concurrent_lanes
        chunks = host_room // (bytes_per_row * batch_size)
        return min(rows, chunks * batch_size)

    def predict_cached_window(self, dataset, queries, caches, *, output_type=None, levels=None):
        """Run a window of independent query chunks against each fixed ensemble group."""
        predictions = [[] for _ in queries]
        query_sizes = tuple(map(len, queries))

        def finish(out):
            return regression_stats(out, output_type, levels) if output_type is not None else out

        start = 0
        with self._inference():
            replay_ceiling = self._cache_replay_ceiling()
            for cache in caches:
                stop = start + cache.context_shape[0]
                members = slice(start, stop)
                outputs = self._cached_group_window(dataset, queries, cache, members, query_sizes, replay_ceiling, finish)
                for parts, output in zip(predictions, outputs, strict=True):
                    parts.append(output.float().numpy())
                del outputs, output
                start = stop
        return [concatenate_parts(parts, axis=0) for parts in predictions]

    def _cached_group_window(self, dataset, queries, cache, members, query_sizes, replay_ceiling, finish):
        tables = (torch.as_tensor(dataset.signed_views(query, members), device=self.device) for query in queries)
        return self._model_runner.predict_host_cache(
            tables, cache, query_sizes, replay_ceiling, weights=self.weights, finish=finish,
        )

    def code_query_window(self, dataset, caches, batch_size: int, rows: int) -> int:
        """Budget the decoder's accumulators and a single transient code-row cache."""
        if self.device.type != "cuda" or rows <= batch_size:
            return 0
        if caches is not None and not any(self._model_runner.host_prediction_bytes(c) for c in caches):
            return 0
        from .cache import cache_sizes
        from .memory import EXECUTION_MEMORY_FRACTION

        room = int(available_bytes(torch.device("cpu")) * EXECUTION_MEMORY_FRACTION)
        context_rows, features = dataset.training.shape
        if caches is None:
            columns, prediction, ranks = cache_sizes(
                self.model, context_rows, features, self.device, weighted=self.weights is not None,
            )
            # Building the feature projections dominates temporary cache storage on short, wide tables.
            if columns >= prediction:
                return 0
            room -= columns + prediction + ranks
        # Include class scores, per-chunk appearance counts and the final joined result.
        bytes_per_row = _FLOAT32_BYTES * (3 * features + self.model.config.prediction_width)
        bytes_per_row += np.dtype(np.float64).itemsize * (3 * dataset.codec.class_count + 8 * dataset.num_classes)
        chunks = max(0, room) // (batch_size * bytes_per_row)
        return min(rows, chunks * batch_size)

    def predict_code_window(self, dataset, queries, caches, train):
        """Use one code's context for a window, then advance each decoder in codebook order."""
        from ...v1_0.ecoc import ECOCAccumulator
        from .outputs import classification_probabilities

        accumulators = [ECOCAccumulator(dataset.codec) for _ in queries]
        sizes = tuple(map(len, queries))
        orders = dataset.ensemble.class_permutations
        with self._inference():
            ceiling = self._cache_replay_ceiling()
            for row in range(dataset.code_rows):
                cache = self._prefill(dataset, train, slice(row, row + 1)) if caches is None else caches[row]
                # A CPU group that fits alongside query activations can be staged once.
                if self._model_runner.host_prediction_bytes(cache) and cache.nbytes <= self._working_memory() // 2:
                    cache = cache.to(self.device)
                outputs = self._cached_group_window(dataset, queries, cache, slice(None), sizes, ceiling, lambda out: out)
                for decoder, output in zip(accumulators, outputs, strict=True):
                    decoder.add(classification_probabilities(output.float().numpy(), orders))
                del cache, outputs, output
        return [decoder.finish() for decoder in accumulators]

    def predict(
        self, dataset: PreparedDataset, table: FeatureTable, targets: np.ndarray, *, output_type=None, levels=None,
        replay_contexts=None, finish=None,
    ) -> np.ndarray | ReducedMean:
        def predict_members(section):
            x = torch.from_numpy(dataset.signed_views(table, section)).float()
            y = torch.from_numpy(targets[section]).float()
            # Select the memory plan before allocating the complete input on CUDA.
            replay = {} if replay_contexts is None else {"contexts": replay_contexts, "first_member": section.start}
            output = self._model_runner.predict(x, y, num_classes=dataset.num_classes, weights=self.weights, **replay)
            if output_type is not None:
                output = regression_stats(output, output_type, levels)
            return _host_output(output, finish if section.stop - section.start == len(targets) else None)

        with self._inference():
            count = self._ensemble_batch_count((len(targets), *table.shape))
            predictions = []
            start = 0
            base_size, extra = divmod(len(targets), count)
            for index in range(count):
                size = base_size + (index < extra)
                for result in predict_chunks(
                    lambda members: predict_members(slice(start + members.start, start + members.stop)),
                    rows=size, batch_size=size, device=self.device,
                ):
                    if isinstance(result, ReducedMean):
                        return result
                    predictions.append(result)
                start += size
        return concatenate_parts(predictions, axis=0)

    def predict_table(
        self, dataset: PreparedDataset, table: FeatureTable, *, output_type=None, levels=None,
        replay_contexts=None, finish=None,
    ) -> np.ndarray | ReducedMean:
        return self.predict(
            dataset, table, dataset.targets_for(slice(None)), output_type=output_type, levels=levels,
            **({"finish": finish} if finish is not None else {}),
            **({"replay_contexts": replay_contexts} if replay_contexts is not None else {}),
        )

    def predict_codes(
        self, dataset: PreparedDataset, table: np.ndarray, caches: tuple[ContextCache, ...] | None,
    ) -> Iterator[np.ndarray]:
        """Yield the native logits for each ECOC row in codebook order."""
        if caches is not None:
            for cache in caches:
                yield self.predict_cached(dataset, table, (cache,))
            return
        members = slice(None)
        for code_row in range(dataset.code_rows):
            yield self.predict(dataset, table, dataset.targets_for(members, code_row=code_row))
