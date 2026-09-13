"""Connect fixed pretrained weights to fitted data, caches, and ensemble outputs."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch
from sklearn.exceptions import NotFittedError

from . import checkpoints
from .data.dataset import PreparedDataset
from .data.ensemble import EnsembleMember
from .execution.cached import cached_predictions
from .execution.direct import direct_predictions
from .execution.runner import ModelCache, ModelRunner


def resolve_device(request: str) -> torch.device:
    """Resolve auto placement once; explicit unavailable devices fail instead of falling back."""
    if request == "auto":
        request = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(request)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Causilo supports CPU or a single CUDA device")
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(f"Requested CUDA device {index} is unavailable")
        device = torch.device("cuda", index)
    return device


@dataclass
class FitState:
    """Successful fit only; caches follow ``dataset.members_by_normalization()``.

    Keep the stored tuple layout stable for fitted-state restoration. Consumers
    obtain member/cache pairs through ``cached_members`` rather than rebuilding
    the positional association themselves.
    """

    dataset: PreparedDataset
    caches: tuple[ModelCache, ...] | None

    def cached_members(self) -> tuple[tuple[EnsembleMember, ModelCache], ...]:
        """Pair every cache with its input permutation, rejecting incomplete state."""
        if self.caches is None:
            raise ValueError("This fit does not contain K/V caches")
        members = self.dataset.members_by_normalization()
        if len(members) != len(self.caches):
            raise ValueError("Cache count does not match the fitted ensemble")
        return tuple(zip(members, self.caches))


class Engine:
    """Own one task/device model and replace its training context on each fit."""

    def __init__(self, task: str, device: str) -> None:
        self.task = task
        self.device_request = device
        self.device = resolve_device(device)
        self.model = None
        self.state: FitState | None = None

    def fit(
        self,
        table,
        targets,
        *,
        n_estimators: int,
        use_kv_cache: bool,
        retain_preprocessing: bool,
        random_state: int,
    ) -> None:
        """Prepare context without optimizing weights; a failed fit stays unfitted."""
        self.state = None
        if isinstance(n_estimators, bool) or not isinstance(n_estimators, Integral) or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
        if isinstance(random_state, bool) or not isinstance(random_state, Integral) or random_state < 0:
            raise ValueError("random_state must be a nonnegative integer")
        for name, value in (("use_kv_cache", use_kv_cache), ("retain_preprocessing", retain_preprocessing)):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be a Boolean")
        if self.model is None:
            self.model = checkpoints.load_pretrained_model(self.task).to(self.device)
        prepared = PreparedDataset.prepare(
            table,
            targets,
            task=self.task,
            n_estimators=int(n_estimators),
            retain_preprocessing=retain_preprocessing,
            max_classes=self.model.config.outputs,
            random_state=int(random_state),
        )
        caches = None
        if use_kv_cache:
            runner = ModelRunner(self.model)
            collected = []
            with torch.inference_mode():
                for member in prepared.members_by_normalization():
                    features = prepared.training_table(member.normalization)[:, member.feature_order]
                    y = prepared.targets_for(member)
                    collected.append(runner.build_cache(self._tensor(features), self._tensor(y)))
            caches = tuple(collected)
        # Publish only a complete context; partially built caches stay local.
        self.state = FitState(dataset=prepared, caches=caches)

    def _tensor(self, value: np.ndarray) -> torch.Tensor:
        """Place one member on the device and prepend its singleton batch axis."""
        return torch.as_tensor(
            np.ascontiguousarray(value), device=self.device, dtype=torch.float32
        ).unsqueeze(0)

    def require_state(self) -> FitState:
        """Expose fitted state only after all preparation has succeeded."""
        if self.state is None:
            raise NotFittedError("Call fit successfully before prediction")
        return self.state

    def predict(self, table) -> np.ndarray:
        """Return probabilities (queries, classes) or regression points (queries,)."""
        state = self.require_state()
        fitted = state.dataset
        numeric = fitted.encoder.transform(table)
        outputs = []

        def reduce_output(result):
            # Preserve the established sorted reduction order for regression.
            # Sorting is algebraically unnecessary, but changing summation order
            # can change floating-point results and saved-state reproducibility.
            return result if self.task == "classification" else result.sort(dim=-1).values.mean(dim=-1)

        with torch.inference_mode():
            transformed = {
                name: transform.transform(numeric) for name, transform in fitted.normalizers.items()
            }
            if state.caches is None:
                predictions = direct_predictions(self.model, fitted, transformed, self.device, reduce_output)
            else:
                predictions = cached_predictions(
                    self.model, fitted, transformed, state.cached_members(), self.device, reduce_output
                )
            for member, result in predictions:
                if self.task == "classification":
                    # class_order[original_id] gives the member's output column.
                    outputs.append(result[:, member.class_order].numpy())
                else:
                    point = result.numpy().astype(np.float64)
                    outputs.append(fitted.target_encoder.inverse_transform(point[:, None])[:, 0])
        combined = np.mean(outputs, axis=0)
        if self.task == "classification":
            # Ensemble logits before softmax; averaging probabilities would
            # implement a different prediction policy.
            probabilities = np.exp(combined - combined.max(axis=-1, keepdims=True))
            return probabilities / probabilities.sum(axis=-1, keepdims=True)
        return combined
