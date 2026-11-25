"""Save and restore fitted contexts without serializing model weights."""

from ..serialization import fitted_contexts
from .data.dataset import PreparedDataset
from .data.frequency import validate_frequencies
from .engine import Engine
from .execution.serialization import SavedContext, SavedSampledContext, export_context, restore_context


def export_fitted(engine: Engine) -> dict:
    """Record the prepared contexts and the checkpoint that produced them."""
    contexts = (
        export_context(engine.state) if engine.batch_states is None
        else {key: export_context(state) for key, state in engine.batch_states.items()}
    )
    return {
        "fitted": contexts,
        "checkpoint": engine.checkpoint.copy() if engine.checkpoint is not None else engine.source.revision,
        "model_config": engine.model.config.to_dict(),
        "cache_device_type": engine.device.type,
        "cache_bf16_output_projection": engine.bf16_output_projection,
    }


def restore_fitted(engine: Engine, saved: dict, digest: str | None) -> None:
    """Restore contexts using the current checkpoint and its cache precision."""
    fitted = saved["fitted"]
    contexts = fitted_contexts(fitted, (SavedContext, SavedSampledContext))
    members = 1
    for context in contexts:
        sampled = isinstance(context, SavedSampledContext)
        datasets = context.datasets if sampled else (context.dataset,)
        if sampled and len(datasets) < 2:
            raise ValueError("Saved sampled context must contain multiple members")
        for dataset in datasets:
            if not isinstance(dataset, PreparedDataset):
                raise ValueError("Saved context does not match the checkpoint architecture")
            if dataset.frequencies is not None:
                counts = validate_frequencies(dataset.frequencies, len(dataset.training))
                if (counts == 0).any():
                    raise ValueError("Saved frequency-weighted context cannot contain zero-count rows")
        count = len(datasets) if sampled else len(datasets[0].ensemble.feature_signs)
        members = max(members, count)
    engine.initialize_model(members, expected_digest=digest)
    if engine.model.config.to_dict() != saved["model_config"]:
        raise ValueError("Saved cache does not match the model shape")

    reuse_cache = (
        engine.device.type == saved["cache_device_type"]
        and engine.bf16_output_projection == saved["cache_bf16_output_projection"]
    )

    def restore(context):
        return restore_context(context, engine, reuse_cache=reuse_cache)

    if isinstance(fitted, dict):
        engine.batch_states = {key: restore(context) for key, context in fitted.items()}
    else:
        engine.state = restore(fitted)
