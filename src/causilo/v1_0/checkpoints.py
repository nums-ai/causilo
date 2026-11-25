"""Load official or local safetensors weights and verify their configuration."""

from pathlib import Path

from huggingface_hub import snapshot_download

from ..checkpoint_io import assign_tensors, read_tensors
from ..model_selection import (
    LEGACY_FORMAT,
    LEGACY_REPOSITORY,
    LEGACY_REVISION,
    CheckpointSource,
    LocalCheckpoint,
    ModelSource,
    local_checkpoint_identity,
    read_local_checkpoint,
)
from .model import Model, ModelConfig

REPOSITORY = LEGACY_REPOSITORY
RELEASE_COMMIT = LEGACY_REVISION


def load_checkpoint(checkpoint: LocalCheckpoint | Path) -> Model:
    """Validate the format and config hash, then materialize an evaluation model."""
    if not isinstance(checkpoint, LocalCheckpoint):
        checkpoint = read_local_checkpoint(checkpoint)
    record = checkpoint.config
    expected = {"architecture": LEGACY_FORMAT, "format_version": 1, "config_version": 1}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported model configuration")
    config = ModelConfig(**record["model"])
    return assign_tensors(Model, config, read_tensors(checkpoint, task=config.task))


def load_hosted_model(task: str) -> Model:
    """Fetch the task's files at the pinned revision, reusing the Hub cache when available."""
    folder = "classifier" if task == "classification" else "regressor"
    root = snapshot_download(
        REPOSITORY,
        revision=RELEASE_COMMIT,
        allow_patterns=[f"{folder}/config.json", f"{folder}/model.safetensors"],
    )
    model = load_checkpoint(Path(root) / folder / "model.safetensors")
    if model.config.task != task:
        raise ValueError("The official checkpoint has the wrong prediction task")
    return model


def load_model(
    task: str, source: CheckpointSource, *, expected_digest=None,
) -> tuple[Model, dict[str, str] | None]:
    """Return the model and its local source, or None for the pinned Hub source."""
    if isinstance(source, ModelSource):
        return load_hosted_model(task), None
    identity = local_checkpoint_identity(source, expected_digest=expected_digest)
    model = load_checkpoint(source)
    if model.config.task != task:
        raise ValueError(f"The checkpoint at {source.weights} is for {model.config.task}, not {task}")
    return model, identity
