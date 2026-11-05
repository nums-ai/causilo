"""Store fitted context without weights and restore only into a matching runtime."""

import math
import sys
from dataclasses import fields, is_dataclass, replace
from importlib.metadata import version

import torch

from . import checkpoints
from .data.frequency import validate_frequencies
from .ecoc import ECOCCodec
from .engine import Engine, FitState
from .execution.memory import Stage
from .execution.precision import stage_dtype
from .model import ModelConfig

STATE_SCHEMA = "causilo-fitted-1"
FREQUENCY_STATE_SCHEMA = "causilo-fitted-2"
BATCH_STATE_SCHEMA = "causilo-batched-fitted-1"


def compact_state(state):
    if state is not None and state.codec is not None:
        # Reconstruct target-only ECOC views after loading, avoiding joblib copies.
        return replace(state, code_datasets=None)
    return state


def runtime_versions() -> dict[str, str]:
    """Record versions that determine estimator, array, and tensor reconstruction."""
    dependencies = ("numpy", "pandas", "scipy", "scikit-learn", "torch")
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        **{name: version(name) for name in dependencies},
    }


def validate_cache_layout(state: FitState, config: ModelConfig) -> None:
    """Check stored layer counts and K/V dimensions before moving caches to a device."""
    cache_groups = [] if state.caches is None else [(state.dataset, state.caches)]
    if state.codec is not None:
        rows = len(state.codec.codebook)
        if state.caches is not None or state.code_datasets is None or len(state.code_datasets) != rows:
            raise ValueError("Saved ECOC state has the wrong context count")
        if state.code_caches is not None:
            if len(state.code_caches) != rows:
                raise ValueError("Saved ECOC cache layout has the wrong code row count")
            cache_groups.extend(zip(state.code_datasets, state.code_caches))
    elif state.code_caches is not None or state.code_datasets is not None:
        raise ValueError("Saved ECOC contexts require a codebook")
    # Column caches have a feature-group axis; prediction caches attend to the
    # entire training-row sequence after feature groups have been pooled.
    for dataset, caches in cache_groups:
        groups = math.ceil(dataset.features.shape[1] / config.group_size)
        column = (1, groups, config.column_heads, config.column_latents, config.width // config.column_heads)
        prediction = (
            1,
            config.prediction_heads,
            len(dataset.features),
            config.width * config.row_latents // config.prediction_heads,
        )
        if len(caches) != len(dataset.members):
            raise ValueError("Saved cache layout has the wrong ensemble count")
        for context in caches:
            for saved, depth, dimensions in (
                (context.columns[0], config.column_depths[0], column),
                (context.columns[1], config.column_depths[1], column),
                (context.prediction, config.prediction_depth, prediction),
            ):
                if len(saved.layers) != depth or any(
                    tuple(layer.key.shape) != dimensions or tuple(layer.value.shape) != dimensions
                    for layer in saved.layers
                ):
                    raise ValueError("Saved cache layout does not match the pretrained architecture")


def transfer_state(value, device, *, clone: bool = False, dtype=None):
    """Recursively move tensors in fitted dataclasses, leaving non-tensor state intact.

    Export clones tensors onto CPU so the saved state does not alias live cache
    storage. Restoration may also convert floating tensors to the destination
    stage's precision; integer tensors retain their dtype.
    """
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
        return tensor.clone() if clone else tensor
    if isinstance(value, ECOCCodec):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                field.name: transfer_state(getattr(value, field.name), device, clone=clone, dtype=dtype)
                for field in fields(value)
            }
        )
    if isinstance(value, tuple):
        return tuple(transfer_state(item, device, clone=clone, dtype=dtype) for item in value)
    if isinstance(value, dict):
        return {key: transfer_state(item, device, clone=clone, dtype=dtype) for key, item in value.items()}
    return value


def export_estimator(estimator) -> dict:
    """Return fitted context and its weight identity without copying model weights."""
    from . import __version__

    engine = getattr(estimator, "_engine", None)
    state = engine.state if engine is not None else None
    batched = engine is not None and engine.batch_states is not None
    state = (
        {key: compact_state(value) for key, value in engine.batch_states.items()}
        if batched else compact_state(state)
    )
    schema = BATCH_STATE_SCHEMA if batched else STATE_SCHEMA
    if not batched and state is not None and state.dataset.frequencies is not None:
        # Older readers must not silently treat compressed rows as single observations.
        schema = FREQUENCY_STATE_SCHEMA
    return {
        "version": __version__,
        "schema": schema,
        "dependencies": runtime_versions(),
        "revision": checkpoints.RELEASE_COMMIT,
        "checkpoint": engine.checkpoint.copy() if state is not None and engine.checkpoint else None,
        "parameters": estimator.get_params(deep=False),
        "task": engine.task if engine is not None else None,
        "model_config": engine.model.config.record()
        if engine is not None and engine.model is not None
        else None,
        "fitted": transfer_state(state, "cpu", clone=True),
    }


def import_estimator(estimator, saved: dict) -> None:
    """Restore parameters and fitted context, loading weights but never rebuilding K/V."""
    from . import __version__

    if saved["version"] != __version__ or saved["schema"] not in {STATE_SCHEMA, FREQUENCY_STATE_SCHEMA, BATCH_STATE_SCHEMA}:
        raise ValueError("Restore requires the same Causilo version and fitted-state schema")
    if saved.get("dependencies") != runtime_versions():
        raise ValueError("Restore requires matching Python and dependency versions")
    if saved["revision"] != checkpoints.RELEASE_COMMIT:
        raise ValueError("Saved state belongs to a different pretrained checkpoint")
    state = saved["fitted"]
    if saved["schema"] == BATCH_STATE_SCHEMA:
        if not isinstance(state, dict) or not state or any(
            not isinstance(key, str) or not key or not isinstance(value, FitState)
            for key, value in state.items()
        ):
            raise ValueError("Saved batch must contain string IDs and fitted contexts")
        contexts = state.values()
    else:
        contexts = (state,)
    for context in contexts:
        frequencies = None if context is None else context.dataset.frequencies
        if saved["schema"] == FREQUENCY_STATE_SCHEMA:
            if frequencies is None:
                raise ValueError("Saved frequency-weighted state requires repetition counts")
            frequencies = validate_frequencies(frequencies, len(context.dataset.features))
            if (frequencies == 0).any():
                raise ValueError("Saved frequency-weighted state cannot contain zero-count rows")
        elif frequencies is not None:
            raise ValueError("Saved repetition counts require the frequency-weighted state schema")
    from .estimators import invalidate_fit, publish_fitted_attributes

    invalidate_fit(estimator)
    estimator.__dict__.pop("_engine", None)
    estimator.__dict__.update({"checkpoint_path": None, **saved["parameters"]})
    if saved["fitted"] is None:
        return
    if "checkpoint" not in saved and estimator.checkpoint_path is not None:
        raise ValueError("Saved local checkpoint is missing its fitted weight identity")
    # Auto resolves again on the current host; an explicit unavailable device
    # is rejected by Engine before any context is installed.
    # Fitted context belongs to the source used during fit, even if set_params
    # selected a different source for the next fit. Older payloads used the Hub.
    checkpoint = saved.get("checkpoint")
    path = checkpoint["path"] if checkpoint is not None else None
    engine = Engine(saved["task"], estimator.device, path)
    model, engine.checkpoint = checkpoints.load_model(
        engine.task, path, expected_digest=checkpoint["digest"] if checkpoint is not None else None,
    )
    engine.model = model.to(engine.device)
    if engine.model.config.record() != saved["model_config"]:
        raise ValueError("Saved cache does not match the model shape")
    if saved["schema"] == BATCH_STATE_SCHEMA:
        engine.batch_states = {key: restore_state(value, engine) for key, value in state.items()}
    else:
        engine.state = restore_state(saved["fitted"], engine)
    estimator._engine = engine
    publish_fitted_attributes(estimator)


def restore_state(state: FitState, engine: Engine) -> FitState:
    """Validate and move a fitted context without rebuilding preprocessing or K/V."""
    if state.codec is not None and state.code_datasets is None:
        # Reuse the saved codebook and transforms; neither fitting nor K/V rebuilds
        # are needed to restore each row's targets and shared preprocessing.
        encoded = state.codec.encode(state.dataset.targets)
        state = replace(
            state, code_datasets=tuple(replace(state.dataset, targets=row) for row in encoded)
        )
    validate_cache_layout(state, engine.model.config)
    if state.caches is not None or state.code_caches is not None:
        # Apply the same stage precision policy used by cache construction.
        column_dtype = stage_dtype(engine.task, Stage.COLUMN, engine.device)
        prediction_dtype = stage_dtype(engine.task, Stage.PREDICTION, engine.device)

        def move(caches):
            if caches is None:
                return None
            return tuple(
                replace(
                    context,
                    columns=tuple(
                        transfer_state(column, engine.device, dtype=column_dtype)
                        for column in context.columns
                    ),
                    prediction=transfer_state(context.prediction, engine.device, dtype=prediction_dtype),
                )
                for context in caches
            )

        state = replace(
            state,
            caches=move(state.caches),
            code_caches=None
            if state.code_caches is None
            else tuple(move(caches) for caches in state.code_caches),
        )
    return state
