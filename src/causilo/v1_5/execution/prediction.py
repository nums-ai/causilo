"""Table scheduling and output reconstruction for the Causilo 1.5 runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..data.dataset import join_context
from .device_mean import trimmed_mean_reducer
from .direct import EnsembleRunner, ReducedMean, concatenate_parts
from .fit import ContextState, SampledFitState
from .memory import is_allocation_error
from .outputs import (
    classification_logits,
    classification_probabilities,
    member_quantile_levels,
    regression_predictions,
    trimmed_member_quantiles,
    trimmed_regression_predictions,
)
from .sampled import predict_sampled_context

if TYPE_CHECKING:
    from ..engine import Engine


def prepare_query(state: ContextState, table):
    """Validate one batch query and retain its layout for chunking."""
    return table, state.dataset.encoder.query_layout(table)


def predict_context(
    engine: Engine, state: ContextState, table, *, output_type, levels,
    layout: tuple[int, np.ndarray | None] | None = None,
) -> np.ndarray:
    """Run one fitted table in query chunks and combine its ensemble outputs."""
    if isinstance(state, SampledFitState):
        if layout is None:
            layout = state.dataset.encoder.query_layout(table)
        return predict_sampled_context(engine, state, table, output_type=output_type, levels=levels, layout=layout)
    dataset, runner = state.dataset, state.runner
    caches, codec = state.caches, dataset.codec
    classification = engine.task == "classification"
    if codec is not None and output_type == "logits":
        limit = engine.model.config.max_classes
        raise NotImplementedError(
            f"predict_logits supports native classification with at most {limit} classes "
            "for the fitted checkpoint; ECOC logits are not supported"
        )
    if layout is None:
        layout = dataset.encoder.query_layout(table)
    rows, _ = layout
    output_width = 0
    target_limits = dataset.target_limits
    trim = dataset.ensemble.trim_count
    if classification:
        model_output, member_levels = None, None
    elif target_limits is not None:
        output_levels, member_levels, alignment = trimmed_member_quantiles(output_type, levels)
        model_output = "raw" if output_levels is None else "quantiles"
        native_width = engine.model.prediction.head.output.out_features
        output_width = max(native_width, len(member_levels)) if member_levels is not None else native_width
    else:
        model_output = output_type
        member_levels = member_quantile_levels(output_type, levels, dataset.ensemble.target_signs)
    batch_size = runner.query_batch_size(dataset, rows, cached=caches is not None, output_width=output_width)
    train = dataset.training_table() if caches is None else None
    finish = None
    if target_limits is not None and output_type == "mean" and isinstance(runner, EnsembleRunner):
        finish = trimmed_mean_reducer(runner, dataset, caches, alignment)
    finish_options = {} if finish is None else {"finish": finish}

    class_orders = dataset.ensemble.class_permutations
    # Bounded replay can keep a member's context for the duration of this prediction.
    replay_options = {}
    if caches is None and codec is None and batch_size < rows:
        if runner.can_retain_replay(dataset):
            replay_options["replay_contexts"] = {}

    def predict_chunk(inputs):
        if codec is not None:
            return codec.decode_rows(
                classification_probabilities(outputs, class_orders)
                for outputs in runner.predict_codes(dataset, inputs, caches)
            )
        if caches is None:
            outputs = runner.predict_table(
                dataset, inputs, output_type=model_output, levels=member_levels, **replay_options, **finish_options,
            )
        else:
            outputs = runner.predict_cached(
                dataset, inputs, caches, output_type=model_output, levels=member_levels, **finish_options,
            )
        if isinstance(outputs, ReducedMean):
            return outputs.values
        return combine_members(outputs)

    def combine_members(outputs):
        if classification:
            combine = classification_logits if output_type == "logits" else classification_probabilities
            return combine(outputs, class_orders)
        if target_limits is not None:
            return trimmed_regression_predictions(
                outputs, dataset.target_preprocessor, dataset.ensemble.target_signs, output_type=output_type,
                levels=output_levels, alignment=alignment, target_limits=target_limits, trim=trim,
            )
        return regression_predictions(
            outputs, dataset.target_preprocessor, dataset.ensemble.target_signs,
            output_type=output_type, levels=levels,
        )

    def predict_window(start, stop):
        queries = [
            dataset.query_table(table, slice(i, min(i + batch_size, stop)), layout=layout)
            for i in range(start, stop, batch_size)
        ]
        if codec is not None:
            return runner.predict_code_window(dataset, queries, caches, train)
        outputs = runner.predict_cached_window(dataset, queries, caches, output_type=model_output, levels=member_levels)
        return [combine_members(out) for out in outputs]

    window = 0
    if rows > batch_size:
        if codec is not None:
            window = runner.code_query_window(dataset, caches, batch_size, rows)
        elif caches is not None:
            window = runner.cached_query_window(dataset, caches, batch_size, rows, levels=member_levels)
    chunks = []
    start = 0
    while window > batch_size and rows - start > batch_size:
        stop = min(rows, start + window)
        try:
            completed = predict_window(start, stop)
        except (RuntimeError, MemoryError) as error:
            if not is_allocation_error(error):
                raise
            break
        chunks.extend(completed)
        start = stop
    if start < rows:
        # A single chunk and any uncommitted rows use the ordinary allocation retries.
        def prepare_chunk(section):
            section = slice(start + section.start, start + section.stop)
            query = dataset.query_table(table, section, layout=layout)
            return join_context(train, query) if caches is None else query

        chunks.extend(runner.query_chunks(prepare_chunk, predict_chunk, rows=rows - start, batch_size=batch_size))
    return concatenate_parts(chunks, axis=0)


def predict_contexts(engine: Engine, queries, output_options):
    """Predict each requested table using its own fitted context."""
    results = {}
    for key, (table, layout) in queries.items():
        state = engine.batch_states[key]
        output_type, levels = output_options[key]
        try:
            results[key] = predict_context(engine, state, table, output_type=output_type, levels=levels, layout=layout)
        except (ValueError, TypeError) as error:
            raise type(error)(f"Table {key!r}: {error}") from error
    return results
