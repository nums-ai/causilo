"""Shared member scheduling and output reconstruction for one or more tables."""

import torch

from .ecoc import ECOCAccumulator
from .execution.batched import BatchItem, batched_predictions
from .outputs import classification_probabilities, regression_predictions, select_regression_output


def predict_contexts(model, states, queries, output_options, device):
    """Execute validated queries and restore table and fitted ensemble order.

    A single-table prediction uses the same path with one context. Fitted state
    and caches are read-only; only temporary execution packs combine members.
    """
    if not queries:
        return {}
    results = {}
    decoders = {key: ECOCAccumulator(state.codec) for key, state in states.items() if state.codec is not None}
    rounds = max(len(state.code_datasets) if state.codec is not None else 1 for state in states.values())
    with torch.inference_mode():
        for code_index in range(rounds):
            items, members_by_id = [], {}
            for key, state in states.items():
                if state.codec is None:
                    if code_index:
                        continue
                    dataset, caches = state.dataset, state.caches
                else:
                    if code_index >= len(state.code_datasets):
                        continue
                    dataset = state.code_datasets[code_index]
                    caches = None if state.code_caches is None else state.code_caches[code_index]
                members = dataset.members_by_normalization()
                members_by_id[key] = members
                member_caches = (
                    ((member, None) for member in members)
                    if caches is None else state.cached_members(caches, dataset)
                )
                for index, (member, cache) in enumerate(member_caches):
                    items.append(BatchItem(
                        key=(key, index), fitted=dataset, member=member, transformed=queries[key],
                        cache=cache,
                        output_type=output_options[key][0],
                    ))
            pending = {key: {} for key in members_by_id}
            for item, value in batched_predictions(model, items, device):
                key, index = item.key
                pending[key][index] = value
                del value
                members = members_by_id[key]
                if len(pending[key]) != len(members):
                    continue
                # Execution may reorder members. Aggregate in fitted order and
                # release this table's outputs before requesting more work.
                values = [pending[key][index] for index in range(len(members))]
                del pending[key]
                state = states[key]
                if model.config.task == "classification":
                    probabilities = classification_probabilities(zip(members, values))
                    if state.codec is None:
                        results[key] = probabilities
                    else:
                        decoders[key].add(probabilities)
                    del probabilities
                else:
                    output_type, levels = output_options[key]
                    result = regression_predictions(values, state.dataset.target_encoder)
                    results[key] = select_regression_output(result, output_type, levels)
                    del result
                del values
    results.update((key, decoder.finish()) for key, decoder in decoders.items())
    return {key: results[key] for key in queries}
