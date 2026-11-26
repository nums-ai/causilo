"""scikit-learn entry points; estimator construction never loads model weights."""

from collections.abc import Mapping
from numbers import Integral

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import NotFittedError

from .model_selection import (
    DEFAULT_MODEL_VERSION,
    RUNTIME15_FORMAT,
    runtime_for,
    select_checkpoint,
)
from .serialization import export_estimator, import_estimator, read_context, write_context
from .validation import validate_regression_output


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
    runtime_for(estimator._engine.source.architecture).publish_fitted_attributes(estimator)


def device_key(request):
    """Snapshot device lists without changing the estimator parameter as cloned by sklearn."""
    if isinstance(request, (list, tuple)):
        if not request or any(not isinstance(item, str) for item in request):
            raise ValueError("Select one or more CUDA device names")
        return tuple(request)
    return request


def memory_budget_key(request) -> dict[str, int]:
    """Validate and snapshot the estimator's byte budgets."""
    if request is None:
        return {}
    if not isinstance(request, Mapping) or any(
        not isinstance(name, str)
        or isinstance(limit, bool)
        or not isinstance(limit, Integral)
        or limit <= 0
        for name, limit in request.items()
    ):
        raise ValueError("device_memory_budget_bytes requires device names and positive integer byte counts")
    return {name: int(limit) for name, limit in request.items()}


def fit_options(estimator, engine):
    """Forward the fitted-context settings supported by the selected runtime."""
    mode = estimator.mode
    n_estimators = estimator.n_estimators
    if mode is not None:
        if not isinstance(mode, str) or mode not in ("default", "light"):
            raise ValueError("mode must be 'default', 'light', or None")
        if n_estimators is not None:
            raise ValueError("Specify either mode or n_estimators")
    if n_estimators is None:
        n_estimators = 1 if mode == "light" else 8

    options = {
        "n_estimators": n_estimators,
        "use_kv_cache": estimator.use_kv_cache,
        "retain_preprocessing": estimator.retain_preprocessing,
        "random_state": estimator.random_state,
    }
    if engine.source.architecture == RUNTIME15_FORMAT:
        options["max_context"] = estimator.max_context
    return options


def fit_adapter(estimator, X, y, task, sample_count=None):
    """Invalidate old fitted attributes before replacing the engine's context."""
    invalidate_fit(estimator)
    engine = engine_for_fit(estimator, task)
    engine.fit(
        X,
        y,
        sample_count=sample_count,
        **fit_options(estimator, engine),
    )
    publish_fitted_attributes(estimator)
    return estimator


def engine_for_fit(estimator, task):
    checkpoint_request = (estimator.model_version, estimator.checkpoint_path)
    budgets = memory_budget_key(estimator.device_memory_budget_bytes)
    device = device_key(estimator.device)
    engine = getattr(estimator, "_engine", None)
    if (
        engine is not None
        and engine.model is not None
        and engine.task == task
        and getattr(estimator, "_checkpoint_request", None) == checkpoint_request
        and engine.device_request == device
        and getattr(engine, "memory_budgets", {}) == budgets
    ):
        return engine

    version, path = checkpoint_request
    source = select_checkpoint(version, task, path)
    runtime = runtime_for(source.architecture)
    engine = runtime.make_engine(task, device, source, memory_budgets=budgets)
    estimator._engine = engine
    estimator._checkpoint_request = checkpoint_request
    return engine


def fitted_engine(estimator):
    engine = getattr(estimator, "_engine", None)
    if engine is None:
        raise NotFittedError("Call fit successfully before prediction")
    return engine


def supported_input_tags(tags):
    tags.input_tags.allow_nan = True
    tags.input_tags.categorical = True
    return tags


class CausiloClassifier(ClassifierMixin, BaseEstimator):
    """Classify rows using fixed model weights and a context of labeled rows.

    Parameters
    ----------
    mode : {"default", "light"} or None, default=None
        Use the default ensemble or Light mode. Available for both model versions.
    n_estimators : int or None, default=None
        Override the number of ensemble members. Specify either this or mode.
        The default uses eight members and Light uses one.
        For 1.5 contexts with more than ten classes, each ECOC code row uses
        one member. In 1.0 each code row uses the configured ensemble.
    random_state : int or None, default=42
        Nonnegative seed for feature and class permutations and ECOC codebooks.
        Causilo 1.5 accepts None for unseeded fits.
    device : str or sequence of str, default="auto"
        CPU, MPS, or CUDA placement. Causilo 1.5 also accepts an ordered list of CUDA
        devices and assigns distinct ensemble members to them. Auto selects CUDA
        when available, otherwise CPU.
    device_memory_budget_bytes : mapping of str to int, default=None
        Optional workspace planning budgets for selected Causilo 1.5 CUDA devices.
        PyTorch's own allocator limits also apply.
    use_kv_cache : bool, default=False
        Cache attention context during fit for repeated prediction.
    retain_preprocessing : bool, default=True
        For 1.0, also keep normalized copies of the encoded training table.
        For 1.5, keep the fitted float32 feature views if True or transform
        the encoded training table on demand if False.
    max_context : int or None, default=None
        Causilo 1.5 uses all eligible context rows by default. Set a positive
        integer to cap the number of rows per member and sample larger contexts.
    model_version : str, default="1.5"
        Checkpoint release to download when checkpoint_path is omitted.
        Version 1.5 uses public checkpoints that can be downloaded without signing in.
    checkpoint_path : str or path-like, default=None
        Manually downloaded classifier ``model.safetensors`` (or its directory), with
        ``config.json`` beside it. None downloads the pinned checkpoint from the Hub.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Original class labels in predict_proba and predict_logits column order.
    n_features_in_ : int
        Input width before constant features are removed.
    feature_names_in_ : numpy.ndarray
        Fitted column names, available when all DataFrame names are strings.
    """

    __getstate__ = export_estimator
    __setstate__ = import_estimator
    save_context = write_context

    @classmethod
    def load_context(cls, path, *, device="auto", device_memory_budget_bytes=None):
        """Restore a saved classifier context using the requested device and budgets."""
        return read_context(
            cls, path, task="classification", device=device, device_memory_budget_bytes=device_memory_budget_bytes,
        )

    def __sklearn_tags__(self):
        return supported_input_tags(super().__sklearn_tags__())

    def __init__(
        self,
        n_estimators: int | None = None,
        *,
        mode: str | None = None,
        random_state: int = 42,
        device: str | list[str] | tuple[str, ...] = "auto",
        device_memory_budget_bytes=None,
        use_kv_cache: bool = False,
        retain_preprocessing: bool = True,
        checkpoint_path=None,
        model_version: str = DEFAULT_MODEL_VERSION,
        max_context: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.mode = mode
        self.random_state = random_state
        self.device = device
        self.device_memory_budget_bytes = device_memory_budget_bytes
        self.use_kv_cache = use_kv_cache
        self.retain_preprocessing = retain_preprocessing
        self.checkpoint_path = checkpoint_path
        self.model_version = model_version
        self.max_context = max_context

    def fit(self, X, y, sample_count=None):
        """Prepare a context from X (rows, features) and y (rows,); return self.

        Feature NaNs are supported, target NaNs
        are rejected. Fits above the native class limit use error-correcting
        output codes (ECOC).
        Parameter changes take effect on refit; a failed refit clears old state.

        sample_count optionally gives nonnegative integer repetition counts,
        one per row. Zero-count rows are ignored. Counts reproduce repeated
        observations without expanding the table; fractional importance weights
        are unsupported. Omitting counts preserves the ordinary fit behavior.
        """
        return fit_adapter(self, X, y, "classification", sample_count)

    def predict_proba(self, X):
        """Return (rows, classes) probabilities using the fitted feature schema."""
        return fitted_engine(self).predict(X)

    def predict_logits(self, X):
        """Return class-aligned ensemble logits before softmax.

        The NumPy array has shape (n_samples, n_classes), with columns in
        classes_ order and two columns for binary classification. Applying
        softmax along the class axis gives predict_proba(X).

        Requires a successful single-table fit within the checkpoint's native
        class limit. ECOC classification raises NotImplementedError before inference.
        """
        return fitted_engine(self).predict(X, output_type="logits")

    def predict(self, X):
        """Return the original label with highest probability for each input row."""
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]

    def differentiable_predictor(self, *, output_type: str = "probability"):
        """Return a fixed-context tensor predictor for input differentiation.

        Supports a model fitted with one estimator and numeric features;
        output_type is 'probability' or 'logits'. Existing fit/predict behavior
        is unchanged. See docs/differentiable.md for the supported contract.
        """
        from .differentiable import DifferentiablePredictor

        return DifferentiablePredictor(self, output_type=output_type)

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
    """Predict one continuous target using fixed model weights and a context of labeled rows.

    Parameters
    ----------
    mode : {"default", "light"} or None, default=None
        Use the default ensemble or Light mode. Available for both model versions.
    n_estimators : int or None, default=None
        Override the number of ensemble members. Specify either this or mode.
        The default uses eight members and Light uses one.
    random_state : int or None, default=42
        Nonnegative seed for feature permutations.
        Causilo 1.5 accepts None to use an unseeded ensemble.
    device : str or sequence of str, default="auto"
        CPU, MPS, or CUDA placement. Causilo 1.5 also accepts an ordered list of CUDA
        devices and assigns distinct ensemble members to them. Auto selects CUDA
        when available, otherwise CPU.
    device_memory_budget_bytes : mapping of str to int, default=None
        Optional workspace planning budgets for selected Causilo 1.5 CUDA devices.
        PyTorch's own allocator limits also apply.
    use_kv_cache : bool, default=False
        Precompute and retain attention context during fit for repeated prediction.
    retain_preprocessing : bool, default=True
        For 1.0, also keep normalized copies of the encoded training table.
        For 1.5, keep the fitted float32 feature views if True or transform
        the encoded training table on demand if False.
    max_context : int or None, default=None
        Causilo 1.5 uses all eligible context rows by default. Set a positive
        integer to cap the number of rows per member and sample larger contexts.
    model_version : str, default="1.5"
        Checkpoint release to download when checkpoint_path is omitted.
        Version 1.5 uses public checkpoints that can be downloaded without signing in.
    checkpoint_path : str or path-like, default=None
        Manually downloaded regressor ``model.safetensors`` (or its directory), with
        ``config.json`` beside it. None downloads the pinned checkpoint from the Hub.

    Fitted attributes are n_features_in_ and, for DataFrames with string column
    names, feature_names_in_. Change settings before fit or refit to apply them.
    """

    __getstate__ = export_estimator
    __setstate__ = import_estimator
    save_context = write_context

    @classmethod
    def load_context(cls, path, *, device="auto", device_memory_budget_bytes=None):
        """Restore a saved regressor context using the requested device and budgets."""
        return read_context(
            cls, path, task="regression", device=device, device_memory_budget_bytes=device_memory_budget_bytes,
        )

    def __sklearn_tags__(self):
        return supported_input_tags(super().__sklearn_tags__())

    def __init__(
        self,
        n_estimators: int | None = None,
        *,
        mode: str | None = None,
        random_state: int = 42,
        device: str | list[str] | tuple[str, ...] = "auto",
        device_memory_budget_bytes=None,
        use_kv_cache: bool = False,
        retain_preprocessing: bool = True,
        checkpoint_path=None,
        model_version: str = DEFAULT_MODEL_VERSION,
        max_context: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.mode = mode
        self.random_state = random_state
        self.device = device
        self.device_memory_budget_bytes = device_memory_budget_bytes
        self.use_kv_cache = use_kv_cache
        self.retain_preprocessing = retain_preprocessing
        self.checkpoint_path = checkpoint_path
        self.model_version = model_version
        self.max_context = max_context

    def fit(self, X, y, sample_count=None):
        """Prepare preprocessing and an inference context from X and finite y.

        Return self. Feature NaNs are supported; a failed refit clears old state.
        sample_count optionally gives nonnegative integer repetition counts;
        zero-count rows are ignored and fractional weights are unsupported.
        """
        return fit_adapter(self, X, y, "regression", sample_count)

    def predict(self, X, *, output_type: str = "mean", quantiles=None):
        """Predict in target units using the selected output mode.

        mean (default) and median return (rows,); raw returns (rows, 999).
        quantiles returns (rows, requested levels) for probabilities in (0, 1).
        """
        levels = validate_regression_output(output_type, quantiles)
        return fitted_engine(self).predict(X, output_type=output_type, levels=levels)

    def differentiable_predictor(self, *, output_type: str = "mean"):
        """Return a fixed-context tensor predictor for input differentiation.

        Supports a model fitted with one estimator and numeric features;
        output_type is 'mean' or 'median', in original target units. Existing
        fit/predict behavior is unchanged. See docs/differentiable.md.
        """
        from .differentiable import DifferentiablePredictor

        return DifferentiablePredictor(self, output_type=output_type)

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
