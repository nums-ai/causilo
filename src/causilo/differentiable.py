"""Differentiable query prediction with fixed, SDK-owned fitted context."""

import numpy as np
import torch
from torch import Tensor, nn
from torch.func import functional_call

from .execution.runner import ModelCache, ModelRunner
from .model import Model
from .nn.cache import FrequencyWeights
from .outputs import reduce_regression_output
from .serialization import transfer_state


def _tensor_stamp(value: Tensor):
    version = None if torch.is_inference(value) else value._version
    return id(value), version, value.device, value.dtype, tuple(value.shape), value.data_ptr()


class _CachedPrediction(nn.Module):
    """Let functional_call use detached parameters without changing their flags."""

    def __init__(self, model: Model) -> None:
        super().__init__()
        self.model = model

    def forward(self, query: Tensor, cache: ModelCache, weights: FrequencyWeights | None) -> Tensor:
        return ModelRunner(self.model).predict_cached(query, cache, weights=weights)


class DifferentiablePredictor:
    """An ephemeral callable created by an estimator's differentiable_predictor().

    Supports a single fitted estimator, numeric finite queries and the fitted
    ``none`` normalizer. It never refits or changes the estimator's options.
    Returned predictions are CPU tensors in the public predictor's output units.
    See docs/differentiable.md for input, lifetime, and supported-output contracts.
    """

    def __init__(self, estimator, *, output_type: str | None = None) -> None:
        from .estimators import fitted_engine

        engine = fitted_engine(estimator)
        state = engine.require_state()
        dataset = state.dataset
        if len(dataset.members) != 1:
            raise NotImplementedError("Differentiable prediction requires a model fitted with n_estimators=1")
        if state.codec is not None:
            raise NotImplementedError("ECOC differentiable prediction is not supported")
        if dataset.encoder.categories:
            raise NotImplementedError("Categorical features do not have continuous input derivatives")
        member = dataset.members[0]
        normalizer = dataset.normalizers[member.normalization]
        if (
            member.normalization != "none"
            or normalizer.normalizer is not None
            or normalizer.quantile_scale is not None
        ):
            raise NotImplementedError("Differentiable prediction supports the fitted none normalizer only")
        self.task = engine.task
        choices = ("probability", "logits") if self.task == "classification" else ("mean", "median")
        output_type = choices[0] if output_type is None else output_type
        if not isinstance(output_type, str) or output_type not in choices:
            raise ValueError(f"output_type must be one of {choices}")
        if (
            output_type == "logits"
            and len(dataset.target_encoder.classes_) > min(10, engine.model.config.outputs)
        ):
            raise NotImplementedError("Differentiable logits require the public predict_logits class limit")
        self.output_type = output_type
        self.device = engine.device
        self.n_features_in_ = dataset.encoder.width
        self.feature_names_in_ = dataset.encoder.names
        self.retained_features_ = tuple(
            np.asarray(dataset.encoder.continuous)[dataset.encoder.retained].tolist()
        )
        self._estimator = estimator
        self._engine = engine
        self._state = state
        self._model = engine.model
        self._dataset = dataset
        self._member = member
        self._fitted_caches = state.caches
        self._cache: ModelCache | None = None
        self._frequency_weights: FrequencyWeights | None = None
        self._forward = _CachedPrediction(self._model)
        tensors = dict(self._forward.named_parameters()) | dict(self._forward.named_buffers())
        self._stamps = {name: _tensor_stamp(value) for name, value in tensors.items()}
        with torch.inference_mode(False), torch.no_grad():
            # Parameter views share storage, not autograd history. Clone the
            # exceptional inference tensors so backward can save their values.
            self._functional_state = {
                name: value.detach().clone() if torch.is_inference(value) else value.detach()
                for name, value in tensors.items()
            }
            self._normalization = tuple(
                torch.tensor(getattr(normalizer, name), device="cpu", dtype=torch.float64)
                for name in ("center", "spread", "lower", "upper")
            )
        self.check_fitted()

    def check_fitted(self) -> None:
        """Reject refits/replaced context or weights; never repair by refitting."""
        engine = getattr(self._estimator, "_engine", None)
        if (
            engine is not self._engine
            or engine.state is not self._state
            or engine.model is not self._model
            or self._state.dataset is not self._dataset
            or self._state.caches is not self._fitted_caches
        ):
            raise RuntimeError("Fitted context or weights changed; create a new differentiable predictor")
        if any(module.training for module in self._model.modules()):
            raise ValueError("Differentiable prediction requires the fitted model in evaluation mode")
        tensors = dict(self._forward.named_parameters()) | dict(self._forward.named_buffers())
        if {name: _tensor_stamp(value) for name, value in tensors.items()} != self._stamps:
            raise RuntimeError("Model parameters or buffers changed; create a new differentiable predictor")

    def prepare(self) -> str:
        """Prepare SDK-owned context; return a status string, never cache tensors.

        Optional: calling the predictor also prepares its context lazily.
        Repeated calls reuse this object's cache without modifying the fitted cache.
        """
        self.check_fitted()
        if self._cache is not None:
            return "reused_private_cache"
        with torch.inference_mode(False), torch.no_grad():
            self._frequency_weights = self._engine._frequency_weights(self._dataset)
            if self._fitted_caches is not None:
                _, cache = self._state.cached_members()[0]
                self._cache = transfer_state(cache, self.device, clone=True)
                action = "cloned_fitted_cache"
            else:
                features = self._dataset.training_table("none")[:, self._member.feature_order]
                targets = self._dataset.targets_for(self._member)
                self._cache = ModelRunner(self._model).build_cache(
                    self._engine._tensor(features), self._engine._tensor(targets),
                    weights=self._frequency_weights,
                )
                action = "built_private_cache_from_fitted_context"
        self.check_fitted()
        return action

    def __call__(
        self, X: Tensor, *, feature_names=None, return_preprocessed: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Predict from a float32/float64 tensor in original fitted column order.

        Set X.requires_grad to differentiate with respect to original input units.
        return_preprocessed=True returns (predictions, query_features), where the
        float32 query features precede member permutation; their columns map to
        retained_features_. This optional intermediate contains query data only.
        """
        self.check_fitted()
        if not isinstance(X, torch.Tensor) or X.dtype not in (torch.float32, torch.float64):
            raise TypeError("X must be a float32 or float64 torch.Tensor in fitted column order")
        if X.layout != torch.strided:
            raise TypeError("X must be a dense tensor in fitted column order")
        if torch.is_inference(X):
            raise ValueError("Create query tensors outside torch.inference_mode for input differentiation")
        if X.ndim != 2 or not len(X) or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have at least one row and {self.n_features_in_} feature columns")
        if feature_names is not None and tuple(feature_names) != self.feature_names_in_:
            raise ValueError("Query feature names must match fitted names and order")
        if not torch.isfinite(X).all():
            raise ValueError("Differentiable queries must contain only finite numeric values")
        with torch.inference_mode(False), torch.enable_grad():
            if self._cache is None:
                self.prepare()
            prepared = self._preprocess(X)
            raw = functional_call(
                self._forward, self._functional_state,
                (prepared[:, self._member.feature_order][None], self._cache, self._frequency_weights),
                strict=True,
            )[0]
            predictions = self._restore_output(raw)
        self.check_fitted()
        return (predictions, prepared) if return_preprocessed else predictions

    def _preprocess(self, X: Tensor) -> Tensor:
        """Match the fitted none normalizer while retaining the original-unit graph."""
        # Separate device and dtype conversions in both directions: a combined
        # .to(device, dtype) can request MPS double during backward.
        numeric = X.cpu().double()[:, self.retained_features_]
        center, spread, lower, upper = self._normalization
        scaled = (numeric - center) / spread
        values = torch.where(scaled < lower, lower - torch.asinh(lower - scaled), scaled)
        values = torch.where(values > upper, upper + torch.asinh(values - upper), values)
        prepared = values.float().to(self.device)
        if not torch.isfinite(prepared).all():
            raise ValueError("Fitted preprocessing produced nonfinite query features")
        return prepared

    def _restore_output(self, raw: Tensor) -> Tensor:
        """Restore class order or target units using the public prediction contract."""
        if self.task == "classification":
            predictions = raw[:, self._member.class_order].float().cpu()
            return predictions.softmax(dim=-1) if self.output_type == "probability" else predictions
        point = reduce_regression_output(raw, self.output_type)
        if self.output_type == "median":
            point = point[:, raw.shape[-1] // 2]
        scaler = self._dataset.target_encoder
        return point.cpu().double() * float(scaler.scale_[0]) + float(scaler.mean_[0])

    def __reduce_ex__(self, protocol):
        raise TypeError("Differentiable predictors are ephemeral; recreate one from the fitted estimator")
