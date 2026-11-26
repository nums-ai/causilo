"""Connect fixed pretrained weights to fitted data, caches, and ensemble outputs."""

import weakref
from functools import cached_property
from numbers import Integral
from threading import Lock

import numpy as np
import torch
from sklearn.exceptions import NotFittedError

from ..estimators import device_key, memory_budget_key
from ..model_selection import CheckpointSource
from . import checkpoints
from .execution import fit, prediction
from .execution.fit import ContextState
from .execution.precision import supports_bf16
from .nn.checkpoint import build_model
from .nn.layers.feedforward import FeedForward

# Loaded weights shared by live estimators with the same checkpoint, task and device.
# Entries disappear when no estimator holds the model.
_shared_models = weakref.WeakValueDictionary()
_shared_models_lock = Lock()


def clear_shared_models() -> None:
    """Stop sharing loaded weights with new estimators; existing estimators keep theirs."""
    with _shared_models_lock:
        _shared_models.clear()


def resolve_device(request: str) -> torch.device:
    """Resolve auto placement once; explicit unavailable devices fail instead of falling back."""
    if request == "auto":
        request = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(request)
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("Causilo supports CPU, MPS, and CUDA devices")
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


def resolve_devices(request) -> tuple[torch.device, ...]:
    if not isinstance(request, tuple):
        return (resolve_device(request),)
    if any(
        item != "cuda" and not (item.startswith("cuda:") and item[5:].isascii() and item[5:].isdecimal())
        for item in request
    ):
        raise ValueError("Select one or more CUDA devices")
    devices = tuple(resolve_device(item) for item in request)
    if len(set(devices)) != len(devices):
        raise ValueError("CUDA devices in an ensemble must be unique")
    return devices


class Engine:
    """Own one task's weights and replace its training context on each fit."""

    def __init__(
        self, task: str, device, source: CheckpointSource, *, memory_budgets=None,
    ) -> None:
        self.task = task
        self.device_request = device_key(device)
        self.source = source
        self.checkpoint = None
        self.devices = resolve_devices(self.device_request)
        self.device = self.devices[0]
        self.memory_budgets = memory_budget_key(memory_budgets)
        allowed = {str(device) for device in self.devices if device.type == "cuda"}
        if self.memory_budgets.keys() - allowed:
            raise ValueError("Memory budgets require selected Causilo 1.5 CUDA devices such as 'cuda:0'")
        self.model = None
        self._model_replicas = {}
        self.state: ContextState | None = None
        self.batch_states: dict[str, ContextState] | None = None

    @cached_property
    def bf16_output_projection(self) -> bool:
        """Choose one projection precision for all replicas of this engine."""
        return all(supports_bf16(device) for device in self.devices)

    def _check_budget(self, model, device):
        limit = self.memory_budgets.get(str(device))
        if limit is None:
            return
        weight_bytes = sum(value.nbytes for value in model.state_dict().values())
        projection_bytes = torch.bfloat16.itemsize * sum(
            parameter.numel()
            for layer in model.modules() if isinstance(layer, FeedForward)
            for parameter in layer.output.parameters()
        )
        if weight_bytes >= limit or (
            weight_bytes + projection_bytes >= limit and self.bf16_output_projection
        ):
            raise ValueError(f"The memory budget for {device} must exceed the model weight size")

    def _place_model(self, model, device):
        self._check_budget(model, device)
        model.to(device)
        if self.bf16_output_projection:
            for layer in model.modules():
                if isinstance(layer, FeedForward):
                    layer.prepare_bf16_output()
        return model

    def _shares(self, model) -> bool:
        """Reuse a loaded model only if its output projection matches this engine's precision."""
        layer = next(layer for layer in model.modules() if isinstance(layer, FeedForward))
        return (layer._output_weight_bf16 is not None) == self.bf16_output_projection

    def initialize_model(self, n_members=1, *, expected_digest=None):
        """Place the active replicas, retaining them across fits.

        Replicas already loaded by another live estimator are shared. The rest come
        from one checkpoint read: CUDA receives it directly, other devices via the CPU.
        """
        devices = self.devices[:n_members]
        missing = [device for device in devices if device not in self._model_replicas]
        if missing:
            if self.checkpoint is not None:
                expected_digest = self.checkpoint["digest"]
            identity = checkpoints.checkpoint_identity(self.source, expected_digest=expected_digest)
            source = checkpoints.source_key(self.source, identity)
            keys = {device: (self.task, source, str(device)) for device in missing}
            with _shared_models_lock:
                shared = {device: _shared_models.get(key) for device, key in keys.items()}
            placed = {}
            for device, model in shared.items():
                if model is not None:
                    self._check_budget(model, device)
                    if self._shares(model):
                        placed[device] = model
            unloaded = [device for device in missing if device not in placed]
            if unloaded:
                first = unloaded[0]

                def check_budgets(model):
                    for device in unloaded:
                        self._check_budget(model, device)

                budgeted = any(str(device) in self.memory_budgets for device in unloaded)
                template, identity = checkpoints.load_model(
                    self.task, self.source, expected_digest=expected_digest, identity=identity,
                    device=first if first.type == "cuda" else None,
                    validate=check_budgets if budgeted else None,
                )
                loaded = {}
                for device in unloaded[1:]:
                    # Each replica gets its own modules and copies the template's weights.
                    replica = build_model(template.config, template.state_dict())
                    loaded[device] = self._place_model(replica, device)
                loaded[first] = self._place_model(template, first)
                with _shared_models_lock:
                    # Another thread may have loaded the same weights meanwhile; keep one copy.
                    for device, model in loaded.items():
                        current = _shared_models.setdefault(keys[device], model)
                        placed[device] = current if current is model or self._shares(current) else model
            self._model_replicas.update(placed)
            self.model = self._model_replicas[self.device]
            self.checkpoint = identity
        return {device: self._model_replicas[device] for device in devices}

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
        max_context=None,
    ) -> None:
        """Prepare context without optimizing weights; a failed fit stays unfitted."""
        self.state = None
        self.batch_states = None
        self.state = self.prepare_state(
            table, targets, n_estimators=n_estimators, use_kv_cache=use_kv_cache,
            retain_preprocessing=retain_preprocessing, random_state=random_state, sample_count=sample_count,
            max_context=max_context,
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
        max_context=None,
    ) -> ContextState:
        """Build a complete context without publishing or replacing fitted state."""
        if isinstance(n_estimators, bool) or not isinstance(n_estimators, Integral) or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
        if random_state is not None:
            if isinstance(random_state, bool) or not isinstance(random_state, Integral) or random_state < 0:
                raise ValueError("random_state must be a nonnegative integer")
            random_state = int(random_state)
        for name, value in (("use_kv_cache", use_kv_cache), ("retain_preprocessing", retain_preprocessing)):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be a Boolean")
        return fit.prepare_state(
            self, table, targets, n_estimators=n_estimators, use_kv_cache=use_kv_cache,
            retain_preprocessing=retain_preprocessing, random_state=random_state, sample_count=sample_count,
            max_context=max_context,
        )

    def require_state(self) -> ContextState:
        """Expose fitted state only after all preparation has succeeded."""
        if self.batch_states is not None:
            raise ValueError("This estimator was fitted with fit_batched; use a batched prediction method")
        if self.state is None:
            raise NotFittedError("Call fit successfully before prediction")
        return self.state

    def predict(self, table, *, output_type="mean", levels=None) -> np.ndarray:
        """Run a single fitted context through the selected prediction pipeline."""
        return prediction.predict_context(
            self, self.require_state(), table, output_type=output_type, levels=levels,
        )
