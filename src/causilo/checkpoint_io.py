"""Read verified inference weights and assign them to an empty model."""

import json
import math
import mmap
import warnings
from contextlib import contextmanager
from hashlib import sha256

import torch
from safetensors import safe_open

from .model_selection import LocalCheckpoint

SAFETENSORS_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
    "BOOL": torch.bool,
}


@contextmanager
def open_verified(checkpoint: LocalCheckpoint, task: str):
    """Open the weights after checking that they belong to this task and configuration."""
    required = {
        "architecture": checkpoint.architecture,
        "format_version": "1",
        "task": task,
        "config_sha256": sha256(checkpoint.config_bytes).hexdigest(),
    }
    with safe_open(checkpoint.weights, framework="pt", device="cpu") as artifact:
        if artifact.metadata() != required:
            raise ValueError("Weights do not match their task and configuration")
        yield artifact


def read_tensors(checkpoint: LocalCheckpoint, *, task: str) -> dict[str, torch.Tensor]:
    with open_verified(checkpoint, task) as artifact:
        return {name: artifact.get_tensor(name) for name in artifact.keys()}


def weight_layout(path, names):
    """Return the tensor data offset, length and byte ranges, or None if one buffer cannot hold views."""
    with open(path, "rb") as stream:
        size = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(size))
    header.pop("__metadata__", None)
    if header.keys() != names:
        return None
    entries, length = {}, 0
    for name, entry in header.items():
        dtype = SAFETENSORS_DTYPES.get(entry["dtype"])
        begin, end = entry["data_offsets"]
        if dtype is None or begin % dtype.itemsize:
            return None
        if end - begin != math.prod(entry["shape"]) * dtype.itemsize:
            return None
        entries[name] = (dtype, tuple(entry["shape"]), begin, end)
        length = max(length, end)
    return 8 + size, length, entries


def read_tensors_to_device(checkpoint: LocalCheckpoint, *, task: str, device) -> dict[str, torch.Tensor]:
    """Copy all weights to the device at once, then split them into separate tensors.

    One host-to-device copy avoids a transfer per tensor, which is slow when other
    processes share the GPU. Each tensor is then copied on the device into its own
    allocation: views at arbitrary offsets of one buffer change kernel selection
    and results. Loading briefly holds the weights twice on the device.
    Other layouts use per-tensor copies.
    """
    with open_verified(checkpoint, task) as artifact:
        names = set(artifact.keys())
    layout = weight_layout(checkpoint.weights, names)
    if layout is None:
        return {name: tensor.to(device) for name, tensor in read_tensors(checkpoint, task=task).items()}
    start, length, entries = layout
    if not length:
        return {name: torch.empty(entry[1], dtype=entry[0], device=device) for name, entry in entries.items()}
    with (
        checkpoint.weights.open("rb") as stream,
        mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data,
    ):
        with warnings.catch_warnings():
            # The mapping is read-only and is only copied from.
            warnings.simplefilter("ignore", UserWarning)
            host = torch.frombuffer(data, dtype=torch.uint8, count=length, offset=start)
        # On the CPU this stays the mapping itself; the copies below must finish before it closes.
        flat = host.to(device)
        tensors = {
            name: flat[begin:end].view(dtype).view(shape).clone()
            for name, (dtype, shape, begin, end) in entries.items()
        }
        del host, flat
    return tensors

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
