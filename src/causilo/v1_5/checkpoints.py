"""Load official or local safetensors weights and verify their configuration."""

from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from ..checkpoint_io import read_tensors
from ..model_selection import (
    CheckpointSource,
    LocalCheckpoint,
    ModelSource,
    local_checkpoint_identity,
    read_local_checkpoint,
)
from .nn.checkpoint import build_model, validate_checkpoint


def load_checkpoint(checkpoint: LocalCheckpoint, task: str) -> torch.nn.Module:
    """Validate a resolved checkpoint before constructing the selected model."""
    config = validate_checkpoint(checkpoint.config)
    checkpoint_task = checkpoint.config["task"]
    if checkpoint_task != task:
        raise ValueError(f"The checkpoint at {checkpoint.weights} is for {checkpoint_task}, not {task}")
    return build_model(config, read_tensors(checkpoint, task=task))


def download_checkpoint(source):
    """Download only the config and weights pinned by the selected model."""
    prefix = f"{source.folder}/" if source.folder else ""
    root = snapshot_download(
        source.repository,
        revision=source.revision,
        allow_patterns=[f"{prefix}config.json", f"{prefix}model.safetensors"],
        token=False,
    )
    return Path(root) / source.folder


def load_hosted_model(task: str, source: ModelSource) -> torch.nn.Module:
    """Fetch the selected task checkpoint at its pinned revision."""
    checkpoint = read_local_checkpoint(download_checkpoint(source))
    if checkpoint.architecture != source.architecture:
        raise ValueError("The official checkpoint has the wrong architecture")
    return load_checkpoint(checkpoint, task)


def load_model(
    task: str, source: CheckpointSource, *, expected_digest=None
) -> tuple[torch.nn.Module, dict[str, str] | None]:
    """Load a selected model and record local weight identity for fitted-state persistence."""
    if isinstance(source, ModelSource):
        return load_hosted_model(task, source), None
    identity = local_checkpoint_identity(source, expected_digest=expected_digest)
    return load_checkpoint(source, task), identity
