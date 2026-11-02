"""ID-based fitted contexts and per-table output reconstruction."""

from collections.abc import Mapping

import numpy as np
from sklearn.exceptions import NotFittedError

from .outputs import validate_regression_output
from .prediction import predict_contexts


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
    states = {key: engine.batch_states[key] for key in queries}
    return predict_contexts(engine.model, states, queries, output_options, engine.device)
