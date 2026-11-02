"""ID-based fitted contexts and per-table output reconstruction."""

from collections.abc import Mapping

import numpy as np
import torch
from sklearn.exceptions import NotFittedError

from .ecoc import ECOCAccumulator
from .execution.batched import BatchItem, batched_predictions
from .outputs import (
    classification_probabilities,
    regression_predictions,
    select_regression_output,
    validate_regression_output,
)


def validate_mapping(tables, *, allow_empty=False):
    if not isinstance(tables, Mapping) or (not tables and not allow_empty):
        raise ValueError("Expected a nonempty mapping of table IDs to data")
    if any(not isinstance(key, str) or not key for key in tables):
        raise ValueError("Table IDs must be nonempty strings")


def fit_batched(estimator, tables, task):
    from .estimators import engine_for_fit, invalidate_fit, publish_fitted_attributes

    invalidate_fit(estimator)
    validate_mapping(tables)
    for key, value in tables.items():
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(f"Table {key!r}: expected (X, y)")
    engine = engine_for_fit(estimator, task)
    states = {}
    for key, (X, y) in tables.items():
        try:
            states[key] = engine.prepare_state(
                X, y, n_estimators=estimator.n_estimators,
                use_kv_cache=estimator.use_kv_cache,
                retain_preprocessing=estimator.retain_preprocessing,
                random_state=estimator.random_state,
            )
        except (ValueError, TypeError) as error:
            raise type(error)(f"Table {key!r}: {error}") from error
    # Publish once, after every context has been prepared successfully.
    engine.batch_states = states
    publish_fitted_attributes(estimator)
    return estimator


def _prepare_queries(engine, tables):
    """Validate every query before dispatching any model work."""
    validate_mapping(tables, allow_empty=True)
    queries, outputs = {}, {}
    for key, value in tables.items():
        if key not in engine.batch_states:
            raise ValueError(f"Table {key!r}: unknown fitted table ID")
        try:
            output_type, levels = "mean", None
            if isinstance(value, Mapping):
                allowed = {"X", "output_type", "quantiles"} if engine.task == "regression" else {"X"}
                unknown = set(value) - allowed
                if unknown or "X" not in value:
                    raise ValueError("Expected X and only supported prediction options")
                X = value["X"]
                output_type = value.get("output_type", "mean")
                quantiles = value.get("quantiles")
                levels = validate_regression_output(output_type, quantiles)
            else:
                X = value
            shape = np.asarray(X).shape
            width = engine.batch_states[key].dataset.encoder.width
            if len(shape) != 2 or shape[1] != width:
                raise ValueError(f"Expected a two-dimensional table with {width} feature columns")
            if shape[0] == 0:
                raise ValueError("Query table must contain at least one row")
            queries[key] = engine.batch_states[key].dataset.query_tables(X)
            outputs[key] = output_type, levels
        except (ValueError, TypeError) as error:
            raise type(error)(f"Table {key!r}: {error}") from error
    return queries, outputs


def predict_batched(estimator, tables):
    engine = getattr(estimator, "_engine", None)
    if engine is None or (engine.state is None and engine.batch_states is None):
        raise NotFittedError("Call fit_batched successfully before batched prediction")
    if engine.batch_states is None:
        raise ValueError("This estimator was fitted with fit; use predict or refit with fit_batched")
    queries, output_options = _prepare_queries(engine, tables)
    if not queries:
        return {}
    states = {key: engine.batch_states[key] for key in queries}
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
                for index, member in enumerate(members):
                    items.append(BatchItem(
                        key=(key, index), fitted=dataset, member=member, transformed=queries[key],
                        cache=None if caches is None else caches[index],
                        output_type=output_options[key][0],
                    ))
            pending = {key: {} for key in members_by_id}
            for item, value in batched_predictions(engine.model, items, engine.device):
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
                if engine.task == "classification":
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
    return {key: results[key] for key in tables}
