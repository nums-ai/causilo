"""Shared member scheduling and output reconstruction for one or more tables."""

import torch

from .ecoc import ECOCAccumulator
from .execution.batched import BatchItem, batched_predictions
from .nn.cache import FrequencyWeights
from .outputs import classification_probabilities, regression_predictions, select_regression_output


def predict_contexts(model, states, queries, output_options, device):
    """Execute validated queries and combine each table's ordered member outputs.

    A single-table prediction uses the same path with one context. Fitted state
    and caches are read-only; only temporary execution packs combine members.
    """
    if not queries:
        return {}
    # All ensemble and ECOC members of a fitted context share these counts.
    # Keep prepared device metadata local to this prediction operation.
    weights = {
        key: FrequencyWeights.prepare(torch.from_numpy(state.dataset.frequencies), device)
        for key, state in states.items() if state.dataset.frequencies is not None
    }
    results = {}
    decoders = {key: ECOCAccumulator(state.codec) for key, state in states.items() if state.codec is not None}
    rounds = max(len(state.code_datasets) if state.codec is not None else 1 for state in states.values())
    with torch.inference_mode():
        for code_index in range(rounds):
            items = []
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
                member_caches = (
                    ((member, None) for member in members)
                    if caches is None else state.cached_members(caches, dataset)
                )
                for index, (member, cache) in enumerate(member_caches):
                    items.append(BatchItem(
                        key=(key, index), fitted=dataset, member=member, transformed=queries[key],
                        cache=cache, frequencies=weights.get(key),
                        output_type=output_options[key][0],
                    ))
            predictions = []
            for item, value in batched_predictions(model, items, device):
                predictions.append((item.member, value))
                del value
                if len(predictions) != len(item.fitted.members):
                    continue
                # The scheduler preserves table/member order, including across
                # memory splits. Finish this table before requesting more work.
                key = item.key[0]
                state = states[key]
                if model.config.task == "classification":
                    probabilities = classification_probabilities(predictions)
                    if state.codec is None:
                        results[key] = probabilities
                    else:
                        decoders[key].add(probabilities)
                    del probabilities
                else:
                    output_type, levels = output_options[key]
                    result = regression_predictions((value for _, value in predictions), state.dataset.target_encoder)
                    results[key] = select_regression_output(result, output_type, levels)
                    del result
                predictions.clear()
    results.update((key, decoder.finish()) for key, decoder in decoders.items())
    return {key: results[key] for key in queries}
