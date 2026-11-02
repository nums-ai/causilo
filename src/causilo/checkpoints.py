"""Load official or local safetensors weights and verify their configuration."""

import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from .model import Model, ModelConfig

REPOSITORY = "nums-ai/causilo"
RELEASE_COMMIT = "94f2bd91db0737d4da59f347910662905ecb5a09"


def load_checkpoint(weights: Path) -> Model:
    """Validate the format and config hash, then materialize an evaluation model.

    The configuration is read from ``config.json`` beside the weights file.
    """
    encoded = weights.with_name("config.json").read_bytes()
    record = json.loads(encoded)
    expected = {"architecture": "causilo-v1.0", "format_version": 1, "config_version": 1}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported model configuration")
    config = ModelConfig(**record["model"])
    with safe_open(weights, framework="pt") as artifact:
        metadata = artifact.metadata()
        required = {
            "architecture": "causilo-v1.0",
            "format_version": "1",
            "task": config.task,
            "config_sha256": hashlib.sha256(encoded).hexdigest(),
        }
        if metadata != required:
            raise ValueError("Weights do not match their task and configuration")
        tensors = {key: artifact.get_tensor(key) for key in artifact.keys()}
    # Construct shapes without allocating a second full copy of the weights.
    with torch.device("meta"):
        model = Model(config)
    model.load_state_dict(tensors, strict=True, assign=True)
    return model.eval()


def load_pretrained_model(task: str) -> Model:
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


def checkpoint_digest(weights: Path) -> str:
    """Identify a local configuration/weights pair without retaining its file bytes."""
    digest = hashlib.sha256(weights.with_name("config.json").read_bytes())
    with weights.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(task: str, checkpoint_path=None, *, expected_digest=None) -> tuple[Model, dict | None]:
    """Return the model and its local source, or None for the pinned Hub source."""
    if checkpoint_path is None:
        return load_pretrained_model(task), None
    weights = Path(checkpoint_path).expanduser().absolute()
    if weights.is_dir():
        weights = weights / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {weights}")
    config = weights.with_name("config.json")
    if not config.is_file():
        raise FileNotFoundError(f"Checkpoint configuration not found beside the weights: {config}")
    digest = checkpoint_digest(weights)
    if expected_digest is not None and digest != expected_digest:
        raise ValueError("Local checkpoint differs from the weights used to fit the saved estimator")
    model = load_checkpoint(weights)
    if model.config.task != task:
        raise ValueError(f"The checkpoint at {weights} is for {model.config.task}, not {task}")
    return model, {"path": str(weights), "digest": digest}
