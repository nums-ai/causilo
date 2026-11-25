"""Connect the public estimator interface to the original Causilo 1.0 runtime."""

import numpy as np

from .batch import predict_batched as predict_batched
from .engine import Engine
from .serialization import export_fitted as export_fitted
from .serialization import restore_fitted as restore_fitted


def publish_fitted_attributes(estimator):
    """Expose schema metadata only after the complete fit has succeeded."""
    engine = estimator._engine
    if engine.batch_states is not None:
        datasets = {key: state.dataset for key, state in engine.batch_states.items()}
        estimator.n_features_in_batched_ = {key: data.encoder.width for key, data in datasets.items()}
        estimator.feature_names_in_batched_ = {
            key: np.asarray(data.encoder.names, dtype=object)
            for key, data in datasets.items()
            if data.encoder.names is not None and all(isinstance(name, str) for name in data.encoder.names)
        }
        if engine.task == "classification":
            estimator.classes_batched_ = {key: data.target_encoder.classes_ for key, data in datasets.items()}
        return
    state = engine.require_state().dataset
    estimator.n_features_in_ = state.encoder.width
    if state.encoder.names is not None and all(isinstance(name, str) for name in state.encoder.names):
        estimator.feature_names_in_ = np.asarray(state.encoder.names, dtype=object)
    if engine.task == "classification":
        estimator.classes_ = state.target_encoder.classes_


def make_engine(task, device, source, *, memory_budgets):
    if isinstance(device, tuple):
        raise ValueError("An ordered CUDA device list requires Causilo 1.5")
    if memory_budgets:
        raise ValueError("Memory budgets require selected Causilo 1.5 CUDA devices such as 'cuda:0'")
    return Engine(task, device, source)
