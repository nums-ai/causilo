"""Store fitted context without weights and restore only into a matching runtime."""

import math
from dataclasses import fields, is_dataclass, replace

import torch

from ..serialization import fitted_contexts
from . import checkpoints
from .data.frequency import validate_frequencies
from .ecoc import ECOCCodec
from .engine import Engine, FitState
from .execution.memory import Stage
from .execution.precision import stage_dtype
from .model import ModelConfig


def compact_state(state):
    if state.codec is not None:
        # Reconstruct target-only ECOC views after loading, avoiding joblib copies.
        return replace(state, code_datasets=None)
    return state


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
                    raise ValueError("Saved cache layout does not match the checkpoint architecture")


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


def export_fitted(engine: Engine) -> dict:
    """Return fitted context and its weight identity without copying model weights."""
    state = (
        {key: compact_state(value) for key, value in engine.batch_states.items()}
        if engine.batch_states is not None else compact_state(engine.state)
    )
    return {
        "checkpoint": engine.checkpoint.copy() if engine.checkpoint is not None else engine.source.revision,
        "model_config": engine.model.config.record(),
        "fitted": transfer_state(state, "cpu", clone=True),
    }


def restore_fitted(engine: Engine, saved: dict, digest: str | None) -> None:
    """Restore the fitted context and place its saved K/V."""
    state = saved["fitted"]
    for context in fitted_contexts(state, FitState):
        frequencies = context.dataset.frequencies
        if frequencies is not None:
            frequencies = validate_frequencies(frequencies, len(context.dataset.features))
            if (frequencies == 0).any():
                raise ValueError("Saved frequency-weighted state cannot contain zero-count rows")

    model, engine.checkpoint = checkpoints.load_model(
        engine.task, engine.source, expected_digest=digest,
    )
    engine.model = model.to(engine.device)
    if engine.model.config.record() != saved["model_config"]:
        raise ValueError("Saved cache does not match the model shape")
    if isinstance(state, dict):
        engine.batch_states = {key: restore_state(value, engine) for key, value in state.items()}
    else:
        engine.state = restore_state(state, engine)


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
