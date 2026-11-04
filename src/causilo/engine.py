"""Connect fixed pretrained weights to fitted data, caches, and ensemble outputs."""

import os
from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np
import torch
from sklearn.exceptions import NotFittedError

from . import checkpoints
from .data.dataset import PreparedDataset
from .data.ensemble import EnsembleMember
from .ecoc import ECOCCodec
from .execution.runner import ModelCache, ModelRunner
from .nn.cache import FrequencyWeights
from .prediction import predict_contexts


def resolve_device(request: str) -> torch.device:
    """Resolve auto placement once; explicit unavailable devices fail instead of falling back."""
    if request == "auto":
        request = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(request)
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("Causilo supports CPU, MPS, or a single CUDA device")
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(f"Requested CUDA device {index} is unavailable")
        device = torch.device("cuda", index)
    if device.type == "mps":
        if device.index not in (None, 0) or not torch.backends.mps.is_available():
            raise ValueError("Requested MPS device is unavailable")
        device = torch.device("mps")
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
    codec: ECOCCodec | None = None
    code_caches: tuple[tuple[ModelCache, ...], ...] | None = None
    code_datasets: tuple[PreparedDataset, ...] | None = None

    def cached_members(
        self, caches: tuple[ModelCache, ...] | None = None, dataset: PreparedDataset | None = None
    ) -> tuple[tuple[EnsembleMember, ModelCache], ...]:
        """Pair every cache with its input permutation, rejecting incomplete state."""
        caches = self.caches if caches is None else caches
        if caches is None:
            raise ValueError("This fit does not contain K/V caches")
        dataset = self.dataset if dataset is None else dataset
        members = dataset.members_by_normalization()
        if len(members) != len(caches):
            raise ValueError("Cache count does not match the fitted ensemble")
        return tuple(zip(members, caches))


class Engine:
    """Own one task/device model and replace its training context on each fit."""

    def __init__(self, task: str, device: str, checkpoint_path=None) -> None:
        if checkpoint_path is not None and not isinstance(checkpoint_path, (str, os.PathLike)):
            raise ValueError("checkpoint_path must be None or a path")
        self.task = task
        self.device_request = device
        self.checkpoint_path = checkpoint_path
        self.checkpoint = None
        self.device = resolve_device(device)
        self.model = None
        self.state: FitState | None = None
        self.batch_states: dict[str, FitState] | None = None

    def fit(
        self,
        table,
        targets,
        *,
        n_estimators: int,
        use_kv_cache: bool,
        retain_preprocessing: bool,
        random_state: int,
        sample_count=None,
    ) -> None:
        """Prepare context without optimizing weights; a failed fit stays unfitted."""
        self.state = None
        self.batch_states = None
        self.state = self.prepare_state(
            table, targets, n_estimators=n_estimators, use_kv_cache=use_kv_cache,
            retain_preprocessing=retain_preprocessing, random_state=random_state, sample_count=sample_count,
        )

    def prepare_state(
        self,
        table,
        targets,
        *,
        n_estimators: int,
        use_kv_cache: bool,
        retain_preprocessing: bool,
        random_state: int,
        sample_count=None,
    ) -> FitState:
        """Build a complete context without publishing or replacing fitted state."""
        if isinstance(n_estimators, bool) or not isinstance(n_estimators, Integral) or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
        if isinstance(random_state, bool) or not isinstance(random_state, Integral) or random_state < 0:
            raise ValueError("random_state must be a nonnegative integer")
        for name, value in (("use_kv_cache", use_kv_cache), ("retain_preprocessing", retain_preprocessing)):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be a Boolean")
        if self.model is None:
            model, checkpoint = checkpoints.load_model(self.task, self.checkpoint_path)
            self.model = model.to(self.device)
            self.checkpoint = checkpoint
        prepared = PreparedDataset.prepare(
            table,
            targets,
            task=self.task,
            n_estimators=int(n_estimators),
            retain_preprocessing=retain_preprocessing,
            random_state=int(random_state),
            class_permutation_size=self.model.config.outputs,
            sample_count=sample_count,
        )
        codec = None
        code_datasets = None
        if (
            self.task == "classification"
            and len(prepared.target_encoder.classes_) > self.model.config.outputs
        ):
            codec = ECOCCodec(
                len(prepared.target_encoder.classes_),
                self.model.config.outputs,
                int(random_state),
            )
            encoded = codec.encode(prepared.targets)
            # Share preprocessing and retain all members for every target context.
            code_datasets = tuple(replace(prepared, targets=row) for row in encoded)
        caches = None
        code_caches = None
        if use_kv_cache:
            frequencies = self._frequency_weights(prepared)
            options = {} if frequencies is None else {"frequencies": frequencies}
            runner = ModelRunner(self.model)
            with torch.inference_mode():
                if codec is None:
                    caches = self._build_caches(runner, prepared, **options)
                else:
                    code_caches = tuple(self._build_caches(runner, dataset, **options) for dataset in code_datasets)
        return FitState(
            dataset=prepared,
            caches=caches,
            codec=codec,
            code_caches=code_caches,
            code_datasets=code_datasets,
        )

    def _frequency_weights(self, fitted: PreparedDataset) -> FrequencyWeights | None:
        """Prepare execution-only metadata once for all fitted cache contexts."""
        if fitted.frequencies is None:
            return None
        return FrequencyWeights.prepare(torch.from_numpy(fitted.frequencies), self.device)

    def _build_caches(
        self, runner: ModelRunner, fitted: PreparedDataset, frequencies: FrequencyWeights | None = None
    ) -> tuple[ModelCache, ...]:
        collected = []
        if frequencies is None:
            frequencies = self._frequency_weights(fitted)
        options = {} if frequencies is None else {"frequencies": frequencies}
        for member in fitted.members_by_normalization():
            features = fitted.training_table(member.normalization)[:, member.feature_order]
            collected.append(
                runner.build_cache(self._tensor(features), self._tensor(fitted.targets_for(member)), **options)
            )
        return tuple(collected)

    def _tensor(self, value: np.ndarray) -> torch.Tensor:
        """Place one member on the device and prepend its singleton batch axis."""
        return torch.as_tensor(
            np.ascontiguousarray(value), device=self.device, dtype=torch.float32
        ).unsqueeze(0)

    def require_state(self) -> FitState:
        """Expose fitted state only after all preparation has succeeded."""
        if self.batch_states is not None:
            raise ValueError("This estimator was fitted with fit_batched; use a batched prediction method")
        if self.state is None:
            raise NotFittedError("Call fit successfully before prediction")
        return self.state

    def predict(self, table, *, output_type="mean", levels=None) -> np.ndarray:
        """Run a single fitted context through the shared prediction pipeline."""
        state = self.require_state()
        transformed = state.dataset.query_tables(table)
        if not len(next(iter(transformed.values()))):
            raise ValueError("Query table must contain at least one row")
        return predict_contexts(
            self.model, {0: state}, {0: transformed}, {0: (output_type, levels)}, self.device
        )[0]
