"""Load official or local safetensors weights and verify their configuration."""

import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from ..checkpoint_io import read_tensors, read_tensors_to_device
from ..model_selection import (
    CheckpointSource,
    LocalCheckpoint,
    ModelSource,
    local_checkpoint_identity,
    read_local_checkpoint,
)
from .nn.checkpoint import build_model, validate_checkpoint
from .nn.model import Model


def load_checkpoint(checkpoint: LocalCheckpoint, task: str, device=None, validate=None) -> torch.nn.Module:
    """Validate a resolved checkpoint before constructing the selected model.

    A CUDA device receives the weights directly, using at most one upload chunk of
    device memory beyond them; otherwise they stay on the CPU.
    ``validate`` receives a weightless model of the checkpoint's shape before any
    weights are read, so callers can reject it without allocating device memory.
    """
    config = validate_checkpoint(checkpoint.config)
    checkpoint_task = checkpoint.config["task"]
    if checkpoint_task != task:
        raise ValueError(f"The checkpoint at {checkpoint.weights} is for {checkpoint_task}, not {task}")
    # One weightless model serves validation, the target dtypes, and the loaded model itself.
    with torch.device("meta"):
        shape = Model(config)
    if validate is not None:
        validate(shape)
    if device is not None and torch.device(device).type == "cuda":
        # Convert on the device, one tensor at a time, to the dtypes the model keeps.
        dtypes = {name: value.dtype for name, value in shape.state_dict().items()}
        tensors = read_tensors_to_device(checkpoint, task=task, device=device, dtypes=dtypes)
    else:
        tensors = read_tensors(checkpoint, task=task)
    return build_model(config, tensors, model=shape)


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


def load_hosted_model(task: str, source: ModelSource, device=None, validate=None) -> torch.nn.Module:
    """Fetch the selected task checkpoint at its pinned revision."""
    checkpoint = read_local_checkpoint(download_checkpoint(source))
    if checkpoint.architecture != source.architecture:
        raise ValueError("The official checkpoint has the wrong architecture")
    return load_checkpoint(checkpoint, task, device, validate)


def weights_stamp(source: CheckpointSource) -> tuple[int, ...] | None:
    """Return file attributes that change when local weights are rewritten or replaced."""
    if isinstance(source, ModelSource):
        return None
    status = os.stat(source.weights)
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns


def checkpoint_identity(source: CheckpointSource, *, expected_digest=None) -> dict[str, str] | None:
    """Fingerprint local weights; pinned hosted revisions need no fingerprint."""
    if isinstance(source, ModelSource):
        return None
    return local_checkpoint_identity(source, expected_digest=expected_digest)


def source_key(source: CheckpointSource, identity: dict[str, str] | None) -> tuple[str, ...]:
    """Name the exact weights so loaded models can be shared between estimators."""
    if isinstance(source, ModelSource):
        return ("hosted", source.repository, source.revision, source.folder)
    return ("local", identity["path"], identity["digest"])


def load_model(
    task: str, source: CheckpointSource, *, expected_digest=None, identity=None, device=None, validate=None,
) -> tuple[torch.nn.Module, dict[str, str] | None]:
    """Load a selected model and record local weight identity for fitted-state persistence.

    Pass a precomputed identity to skip fingerprinting the local weights again.
    """
    if isinstance(source, ModelSource):
        return load_hosted_model(task, source, device, validate), None
    if identity is None:
        identity = local_checkpoint_identity(source, expected_digest=expected_digest)
    return load_checkpoint(source, task, device, validate), identity
