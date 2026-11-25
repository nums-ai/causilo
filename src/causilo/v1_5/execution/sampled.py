"""Predict across members fitted to independent context rows."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from ..data.dataset import TableViews
from .direct import concatenate_parts, predict_chunks
from .memory import available_bytes
from .outputs import (
    _average_members,
    classification_logits,
    classification_probabilities,
    finish_trimmed_quantiles,
    member_quantile_levels,
    regression_predictions,
    trimmed_member_quantiles,
)


def predict_sampled_context(engine, state, table, *, output_type, levels, layout):
    members = state.members
    classification = engine.task == "classification"
    trimmed = state.trim_count > 0
    native_width = 1
    if classification:
        native_width = state.dataset.num_classes
        class_orders = np.concatenate([member.dataset.ensemble.class_permutations for member in members])
        combine_classes = classification_logits if output_type == "logits" else classification_probabilities
    elif trimmed:
        output_levels, common_levels, alignment = trimmed_member_quantiles(output_type, levels)
        native_width = engine.model.prediction.head.output.out_features
        if common_levels is not None:
            native_width = max(native_width, len(common_levels))
    elif output_type == "raw":
        native_width = engine.model.prediction.head.output.out_features
    elif levels is not None:
        native_width = 2 * len(levels)

    rows, _ = layout
    host = torch.device("cpu")
    room = available_bytes(host)
    # Bound the ensemble-wide arrays in addition to each runner's own query budget.
    output_limit = max(1, room // (32 * len(members) * native_width))
    batch_size = min(output_limit, *(
        member.runner.query_batch_size(
            member.dataset, rows, cached=member.caches is not None,
            output_width=native_width if trimmed else 0,
        ) for member in members
    ))
    lanes = defaultdict(list)
    for index, member in enumerate(members):
        lanes[member.runner.device].append(index)
    retain_replay = rows > batch_size and all(
        member.caches is None and member.runner.device.type == "cuda" for member in members
    ) and sum(member.runner.replay_context_bytes(member.dataset) for member in members) <= room // 4
    replay = [{} for _ in members] if retain_replay else [None] * len(members)

    def predict_member(index, section):
        member = members[index]
        dataset, runner = member.dataset, member.runner
        signs = dataset.ensemble.target_signs
        if classification:
            model_output, member_levels = None, None
        elif trimmed:
            model_output = "raw" if output_levels is None else "quantiles"
            member_levels = common_levels
        else:
            model_output = output_type
            member_levels = member_quantile_levels(output_type, levels, signs)

        train = None

        def predict_rows(offset):
            nonlocal train
            part = slice(section.start + offset.start, section.start + offset.stop)
            try:
                query = dataset.query_table(table, part, layout=layout)
            except (MemoryError, RuntimeError):
                # Release the retained context before retrying query preparation.
                train = None
                raise
            if member.caches is not None:
                return runner.predict_cached(
                    dataset, query, member.caches, output_type=model_output, levels=member_levels,
                )
            if train is None:
                train = dataset.training_table()
            joint = TableViews((query,), (train,))
            return runner.predict_table(
                dataset, joint, output_type=model_output, levels=member_levels, replay_contexts=replay[index],
            )

        size = section.stop - section.start
        parts = list(predict_chunks(predict_rows, rows=size, batch_size=size, device=runner.device))
        # The context is needed only until the member's query retries finish.
        del train
        outputs = concatenate_parts(parts, axis=1)
        del parts
        if classification:
            return outputs[0]
        if trimmed:
            sign = signs[0]
            selected = outputs[0][..., alignment[1 if sign < 0 else 0]]
            return dataset.target_preprocessor.inverse_transform(selected.astype(np.float64) * sign)
        return regression_predictions(
            outputs, dataset.target_preprocessor, signs, output_type=output_type, levels=levels,
        )

    def predict_section(section, executor=None):
        def predict_lane(indices):
            return [predict_member(index, section) for index in indices]

        groups = map(predict_lane, lanes.values()) if executor is None else executor.map(predict_lane, lanes.values())
        outputs = np.stack([output for group in groups for output in group])
        if classification:
            return combine_classes(outputs, class_orders)
        combined = _average_members(outputs, state.trim_count, dtype=np.float64)
        if trimmed:
            return finish_trimmed_quantiles(combined, output_type, state.target_limits)
        return combined

    if len(lanes) == 1:
        chunks = list(predict_chunks(predict_section, rows=rows, batch_size=batch_size, device=host))
    else:
        with ThreadPoolExecutor(max_workers=len(lanes)) as executor:
            chunks = list(predict_chunks(
                lambda section: predict_section(section, executor), rows=rows, batch_size=batch_size, device=host,
            ))
    return concatenate_parts(chunks, axis=0)
