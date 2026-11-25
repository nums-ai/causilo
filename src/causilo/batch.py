"""ID-based fitted contexts and per-table output reconstruction."""

from collections.abc import Mapping

from sklearn.exceptions import NotFittedError

from .model_selection import runtime_for
from .validation import validate_regression_output


def validate_mapping(tables, *, allow_empty=False):
    if not isinstance(tables, Mapping) or (not tables and not allow_empty):
        raise ValueError("Expected a nonempty mapping of table IDs to data")
    if any(not isinstance(key, str) or not key for key in tables):
        raise ValueError("Table IDs must be nonempty strings")


def fit_batched(estimator, tables, task):
    from .estimators import engine_for_fit, fit_options, invalidate_fit, publish_fitted_attributes

    invalidate_fit(estimator)
    validate_mapping(tables)
    for key, value in tables.items():
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(f"Table {key!r}: expected (X, y)")
    engine = engine_for_fit(estimator, task)
    options = fit_options(estimator, engine)
    states = {}
    for key, (X, y) in tables.items():
        try:
            states[key] = engine.prepare_state(X, y, **options)
        except (ValueError, TypeError) as error:
            raise type(error)(f"Table {key!r}: {error}") from error
    # Publish once, after every context has been prepared successfully.
    engine.batch_states = states
    publish_fitted_attributes(estimator)
    return estimator


def prepare_queries(engine, tables, prepare_query):
    """Validate every query before dispatching any model work."""
    validate_mapping(tables, allow_empty=True)
    allowed = {"X", "output_type", "quantiles"} if engine.task == "regression" else {"X"}
    queries, outputs = {}, {}
    for key, value in tables.items():
        if key not in engine.batch_states:
            raise ValueError(f"Table {key!r}: unknown fitted table ID")
        try:
            output_type, levels = "mean", None
            if isinstance(value, Mapping):
                if set(value) - allowed or "X" not in value:
                    raise ValueError("Expected X and only supported prediction options")
                table = value["X"]
                output_type = value.get("output_type", "mean")
                levels = validate_regression_output(output_type, value.get("quantiles"))
            else:
                table = value
            queries[key] = prepare_query(engine.batch_states[key], table)
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
    return runtime_for(engine.source.architecture).predict_batched(engine, tables)
