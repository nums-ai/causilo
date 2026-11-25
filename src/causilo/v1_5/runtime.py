"""Connect the public estimator interface to the Causilo 1.5 runtime."""

from .batch import predict_batched as predict_batched
from .engine import Engine
from .serialization import export_fitted as export_fitted
from .serialization import restore_fitted as restore_fitted


def make_engine(task, device, source, *, memory_budgets):
    return Engine(task, device, source, memory_budgets=memory_budgets)


def publish_fitted_attributes(estimator):
    """Expose schema metadata only after the complete fit has succeeded."""
    engine = estimator._engine
    if engine.batch_states is not None:
        attributes = {key: state.dataset.fitted_attributes() for key, state in engine.batch_states.items()}
        for name in ("n_features_in_", "feature_names_in_", "classes_"):
            if name != "classes_" or engine.task == "classification":
                setattr(
                    estimator, f"{name}batched_",
                    {key: fitted[name] for key, fitted in attributes.items() if name in fitted},
                )
        return
    estimator.__dict__.update(engine.require_state().dataset.fitted_attributes())
