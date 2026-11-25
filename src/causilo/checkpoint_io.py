"""Read verified inference weights and assign them to an empty model."""

from hashlib import sha256

import torch
from safetensors import safe_open

from .model_selection import LocalCheckpoint


def read_tensors(checkpoint: LocalCheckpoint, *, task: str) -> dict[str, torch.Tensor]:
    required = {
        "architecture": checkpoint.architecture,
        "format_version": "1",
        "task": task,
        "config_sha256": sha256(checkpoint.config_bytes).hexdigest(),
    }
    with safe_open(checkpoint.weights, framework="pt", device="cpu") as artifact:
        if artifact.metadata() != required:
            raise ValueError("Weights do not match their task and configuration")
        return {name: artifact.get_tensor(name) for name in artifact.keys()}


def assign_tensors(model_type, config, tensors, *, match_dtype=False):
    """Allocate only the module structure before assigning its weights."""
    with torch.device("meta"):
        model = model_type(config)
    if match_dtype:
        expected = model.state_dict()
        tensors = {
            name: tensor.to(dtype=expected[name].dtype)
            if name in expected and tensor.dtype != expected[name].dtype else tensor
            for name, tensor in tensors.items()
        }
    model.load_state_dict(tensors, strict=True, assign=True)
    return model.eval()
