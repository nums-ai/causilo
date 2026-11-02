"""Fuse independent table members while preserving their identities and contexts."""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import groupby

import numpy as np
import torch

from ..data.dataset import PreparedDataset
from ..data.ensemble import EnsembleMember
from ..model import Model
from ..nn.cache import BlockCache, KVPair
from ..outputs import reduce_regression_output
from .cached import predict_cached, query_memory
from .direct import BatchPlanner, predict_direct
from .memory import FP32_BYTES, empty_device_cache, execution_budget, is_out_of_memory
from .runner import ModelCache, ModelRunner


@dataclass(frozen=True)
class BatchItem:
    """One table/member prediction, with an optional unpadded fitted cache."""

    key: tuple[str | int, int]
    fitted: PreparedDataset
    member: EnsembleMember
    transformed: dict[str, np.ndarray]
    cache: ModelCache | None = None
    output_type: str = "raw"

    @property
    def shape(self) -> tuple[int, int, int]:
        """Training rows, query rows, and retained features."""
        return (
            len(self.fitted.features),
            len(self.transformed[self.member.normalization]),
            self.fitted.features.shape[1],
        )


def _dimensions(items, group_size):
    training = max(item.shape[0] for item in items)
    queries = max(item.shape[1] for item in items)
    groups = max(math.ceil(item.shape[2] / group_size) for item in items)
    return training, queries, groups * group_size


def _estimate(model, items, device):
    """Include temporary packed caches; original caches already consume device memory."""
    c = model.config
    training, queries, features = _dimensions(items, c.group_size)
    count = len(items)
    if items[0].cache is None:
        workspace = BatchPlanner(c, training, features).estimate(count, queries, device)
        packed = count * ((training + queries) * features + training) * FP32_BYTES
    else:
        workspace = count * query_memory(c, features, training, queries, device)
        groups = features // c.group_size
        # FP32 is conservative even when retained caches use mixed precision.
        cache = 2 * (
            groups * sum(c.column_depths) * c.column_latents * c.width
            + c.prediction_depth * training * c.width * c.row_latents
        ) * FP32_BYTES
        packed = count * (cache + queries * features * FP32_BYTES)
    # The head and regression sorting can materialize full output tensors.
    outputs = count * queries * c.outputs * FP32_BYTES * 3
    return workspace + packed + outputs


def _fits_budget(model, items, device):
    """Check host packing before allocation, separately from CUDA workspace."""
    training, queries, features = _dimensions(items, model.config.group_size)
    cached = items[0].cache is not None
    rows, targets = (queries, 0) if cached else (training + queries, training)
    inputs = len(items) * (rows * features + targets)
    outputs = len(items) * queries * model.config.outputs
    # Allow a per-member feature permutation copy alongside the padded packs.
    scratch = max((q if cached else max(t, q)) * f for t, q, f in (item.shape for item in items))
    host = (inputs + outputs + scratch) * FP32_BYTES
    if host > execution_budget(torch.device("cpu")):
        return False
    required = _estimate(model, items, device)
    if device.type != "cuda":
        # CPU/MPS workspaces and host packs draw from the same physical memory.
        required += host
    return required <= execution_budget(device)


def _pack_cache(items, groups, training):
    """Copy unpadded member caches into a temporary batch without modifying them."""
    def block(states, *, column):
        layers = []
        for index in range(len(states[0].layers)):
            reference = states[0].layers[index]
            shape = list(reference.key.shape)
            shape[0] = len(states)
            shape[1 if column else -2] = groups if column else training
            key = reference.key.new_zeros(shape)
            value = reference.value.new_zeros(shape)
            for batch_index, state in enumerate(states):
                source = state.layers[index]
                if column:
                    key[batch_index, : source.key.shape[1]] = source.key[0]
                    value[batch_index, : source.value.shape[1]] = source.value[0]
                else:
                    key[batch_index, :, : source.key.shape[-2]] = source.key[0]
                    value[batch_index, :, : source.value.shape[-2]] = source.value[0]
            layers.append(KVPair(key, value))
        return BlockCache(tuple(layers))

    return ModelCache(
        tuple(block([item.cache.columns[index] for item in items], column=True) for index in range(2)),
        block([item.cache.prediction for item in items], column=False),
    )


def _reduce(model, item, result):
    if model.config.task == "classification":
        return result
    return reduce_regression_output(result, item.output_type)


def _execute(model, items, device):
    """Finish all outputs before publishing any, so a retry cannot duplicate items."""
    training, queries, features = _dimensions(items, model.config.group_size)
    cached = items[0].cache is not None
    rows = queries if cached else training + queries
    tables = np.zeros((len(items), rows, features), dtype=np.float32)
    targets = None if cached else np.zeros((len(items), training), dtype=np.float32)
    for index, item in enumerate(items):
        t, q, f = item.shape
        order = item.member.feature_order
        query = item.transformed[item.member.normalization]
        start = 0 if cached else training
        tables[index, start : start + q, :f] = query[:, order]
        if not cached:
            tables[index, :t, :f] = item.fitted.training_table(item.member.normalization)[:, order]
            targets[index, :t] = item.fitted.targets_for(item.member)
    lengths = {}
    if any(item.shape[0] != training for item in items):
        lengths["training_lengths"] = torch.tensor([item.shape[0] for item in items], device=device)
    if any(math.ceil(item.shape[2] / model.config.group_size) != features // model.config.group_size
           for item in items):
        lengths["feature_counts"] = torch.tensor([item.shape[2] for item in items], device=device)
    runner = ModelRunner(model)
    table = torch.as_tensor(tables, device=device)
    if cached:
        context = _pack_cache(items, features // model.config.group_size, training)
        result = runner.predict_cached(table, context, **lengths)
    else:
        result = runner.predict(table, torch.as_tensor(targets, device=device), **lengths)
    # CPU logits are views into the full batch; give each member independent
    # storage so unfinished tables cannot keep completed tables' outputs alive.
    return [
        _reduce(model, item, result[index, : item.shape[1]]).to(
            "cpu", copy=model.config.task == "classification"
        )
        for index, item in enumerate(items)
    ]


def _singleton(model, item, device):
    """Reuse the established query chunking and recomputation recovery paths."""
    def reduce_output(value):
        return _reduce(model, item, value)
    if item.cache is None:
        result = predict_direct(model, item.fitted, item.transformed, item.member, device, reduce_output)
    else:
        result = predict_cached(
            model, item.fitted, item.transformed, item.member, item.cache, device, reduce_output
        )
    yield item, result


def _run_group(model, items, device) -> Iterator[tuple[BatchItem, torch.Tensor]]:
    if len(items) == 1:
        yield from _singleton(model, items[0], device)
        return
    outputs = None
    if _fits_budget(model, items, device):
        try:
            outputs = _execute(model, items, device)
        except (MemoryError, RuntimeError) as error:
            if not isinstance(error, MemoryError) and not is_out_of_memory(error, device):
                raise
        if outputs is not None:
            # Drop the scheduler's reference before yielding, allowing each
            # completed table to release its outputs during reconstruction.
            outputs.reverse()
            for item in items:
                yield item, outputs.pop()
            return
        empty_device_cache(device)
    # Split after dropping the failed call's tensors. Each child retains all
    # rows and members; a singleton eventually uses the existing recovery path.
    middle = len(items) // 2
    yield from _run_group(model, items[:middle], device)
    yield from _run_group(model, items[middle:], device)


def batched_predictions(model: Model, items: list[BatchItem], device: torch.device):
    """Yield independently identified member outputs, fusing compatible workloads.

    Preserve input order and split only for cache compatibility or memory limits.
    Callers supply each table's members consecutively in fitted ensemble order.
    Query tables must already be validated and contain at least one row.
    """
    with torch.inference_mode():
        for _, group in groupby(items, key=lambda item: item.cache is not None):
            yield from _run_group(model, list(group), device)
