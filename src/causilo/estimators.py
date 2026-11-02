"""scikit-learn entry points; estimator construction never loads model weights."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import NotFittedError

from .engine import Engine
from .outputs import select_regression_output, validate_regression_output
from .serialization import export_estimator, import_estimator


def invalidate_fit(estimator):
    """Discard both fitted modes before validating or preparing a replacement."""
    for name in (
        "classes_", "n_features_in_", "feature_names_in_", "classes_batched_",
        "n_features_in_batched_", "feature_names_in_batched_",
    ):
        estimator.__dict__.pop(name, None)
    engine = getattr(estimator, "_engine", None)
    if engine is not None:
        engine.state = None
        engine.batch_states = None


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


def fit_adapter(estimator, X, y, task):
    """Invalidate old fitted attributes before replacing the engine's context."""
    invalidate_fit(estimator)
    engine = engine_for_fit(estimator, task)
    engine.fit(
        X,
        y,
        n_estimators=estimator.n_estimators,
        use_kv_cache=estimator.use_kv_cache,
        retain_preprocessing=estimator.retain_preprocessing,
        random_state=estimator.random_state,
    )
    publish_fitted_attributes(estimator)
    return estimator


def engine_for_fit(estimator, task):
    engine = getattr(estimator, "_engine", None)
    if (
        engine is None
        or engine.device_request != estimator.device
        or engine.checkpoint_path != estimator.checkpoint_path
    ):
        engine = Engine(task, estimator.device, estimator.checkpoint_path)
        estimator._engine = engine
    return engine


def fitted_engine(estimator):
    engine = getattr(estimator, "_engine", None)
    if engine is None:
        raise NotFittedError("Call fit successfully before prediction")
    engine.require_state()
    return engine


def supported_input_tags(tags):
    tags.input_tags.allow_nan = True
    tags.input_tags.categorical = True
    return tags


class CausiloClassifier(ClassifierMixin, BaseEstimator):
    """Classify tables using a fixed pretrained model and training-row context.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of preprocessing/permutation ensemble members, sharing weights.
        For many-class classification, every output-code row uses this ensemble.
    random_state : int, default=42
        Nonnegative seed for feature and class permutations; None is unsupported.
    device : str, default="auto"
        CPU, MPS, or one CUDA device. Auto selects CUDA when available, otherwise CPU.
    use_kv_cache : bool, default=False
        Precompute attention context during fit, retaining it on the device.
    retain_preprocessing : bool, default=True
        Keep transformed training tables instead of recomputing them for prediction.
    checkpoint_path : str or path-like, default=None
        Manually downloaded classifier ``model.safetensors`` (or its directory), with
        ``config.json`` beside it. None downloads the pinned checkpoint from the Hub.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Original class labels in predict_proba column order.
    n_features_in_ : int
        Input width before constant features are removed.
    feature_names_in_ : numpy.ndarray
        Fitted column names, available when all DataFrame names are strings.
    """

    __getstate__ = export_estimator
    __setstate__ = import_estimator

    def __sklearn_tags__(self):
        return supported_input_tags(super().__sklearn_tags__())

    def __init__(
        self,
        n_estimators: int = 8,
        *,
        random_state: int = 42,
        device: str = "auto",
        use_kv_cache: bool = False,
        retain_preprocessing: bool = True,
        checkpoint_path=None,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.device = device
        self.use_kv_cache = use_kv_cache
        self.retain_preprocessing = retain_preprocessing
        self.checkpoint_path = checkpoint_path

    def fit(self, X, y):
        """Prepare a context from X (rows, features) and y (rows,); return self.

        No gradient training occurs. Feature NaNs are supported, target NaNs
        are rejected. The native head uses error-correcting output
        codes when there are more than ten target classes.
        Parameter changes take effect on refit; a failed refit clears old state.
        """
        return fit_adapter(self, X, y, "classification")

    def predict_proba(self, X):
        """Return (rows, classes) probabilities using the fitted feature schema."""
        return fitted_engine(self).predict(X)

    def predict(self, X):
        """Return the original label with highest probability for each input row."""
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]

    def fit_batched(self, tables):
        """Fit independent contexts from {table_id: (X, y)}; return self.

        String IDs connect contexts to queries. A refit replaces all contexts;
        failure invalidates both single-table and batched fitted state.
        """
        from .batch import fit_batched

        return fit_batched(self, tables, "classification")

    def predict_proba_batched(self, tables):
        """Return {table_id: probabilities} for any subset of fitted IDs.

        Values are query tables. Columns follow classes_batched_[table_id].
        """
        from .batch import predict_batched

        return predict_batched(self, tables)

    def predict_batched(self, tables):
        """Return {table_id: labels}, preserving the input ID order."""
        probabilities = self.predict_proba_batched(tables)
        return {
            key: self.classes_batched_[key][values.argmax(axis=1)]
            for key, values in probabilities.items()
        }


class CausiloRegressor(RegressorMixin, BaseEstimator):
    """Predict one continuous target using a fixed pretrained context model.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of preprocessing/permutation ensemble members, sharing weights.
    random_state : int, default=42
        Nonnegative seed for feature permutations; None is unsupported.
    device : str, default="auto"
        CPU, MPS, or one CUDA device. Auto selects CUDA when available, otherwise CPU.
    use_kv_cache : bool, default=False
        Precompute and retain attention context during fit for repeated prediction.
    retain_preprocessing : bool, default=True
        Retain transformed training tables; encoded training data is always kept.
    checkpoint_path : str or path-like, default=None
        Manually downloaded regressor ``model.safetensors`` (or its directory), with
        ``config.json`` beside it. None downloads the pinned checkpoint from the Hub.

    Fitted attributes are n_features_in_ and, for DataFrames with string column
    names, feature_names_in_. Change settings before fit or refit to apply them.
    """

    __getstate__ = export_estimator
    __setstate__ = import_estimator

    def __sklearn_tags__(self):
        return supported_input_tags(super().__sklearn_tags__())

    def __init__(
        self,
        n_estimators: int = 8,
        *,
        random_state: int = 42,
        device: str = "auto",
        use_kv_cache: bool = False,
        retain_preprocessing: bool = True,
        checkpoint_path=None,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.device = device
        self.use_kv_cache = use_kv_cache
        self.retain_preprocessing = retain_preprocessing
        self.checkpoint_path = checkpoint_path

    def fit(self, X, y):
        """Prepare X (rows, features) and finite y (rows,) without weight updates.

        Return self. Feature NaNs are supported; a failed refit clears old state.
        """
        return fit_adapter(self, X, y, "regression")

    def predict(self, X, *, output_type: str = "mean", quantiles=None):
        """Predict in target units using the selected output mode.

        mean (default) and median return (rows,); raw returns (rows, 999).
        quantiles returns (rows, requested levels) for probabilities in (0, 1).
        """
        levels = validate_regression_output(output_type, quantiles)
        engine = fitted_engine(self)
        if output_type == "mean":
            return engine.predict(X)
        raw = engine.predict_raw(X)
        return select_regression_output(raw, output_type, levels)

    def fit_batched(self, tables):
        """Fit independent contexts from {table_id: (X, y)}; return self."""
        from .batch import fit_batched

        return fit_batched(self, tables, "regression")

    def predict_batched(self, tables):
        """Predict any subset of fitted IDs, with independent output settings.

        Each value is X (mean output), or a mapping with X, output_type and
        optional quantiles. Output types and quantile rules match predict().
        Returns {table_id: array} in input ID order.
        """
        from .batch import predict_batched

        return predict_batched(self, tables)
