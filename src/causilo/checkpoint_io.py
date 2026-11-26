"""Read verified inference weights and assign them to an empty model."""

import json
import math
import mmap
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
# Bytes per host-to-device transfer when loading weights directly onto a GPU.
UPLOAD_CHUNK_BYTES = 64 * 1024 * 1024


def required_metadata(checkpoint: LocalCheckpoint, task: str) -> dict[str, str]:
    return {
        "architecture": checkpoint.architecture,
        "format_version": "1",
        "task": task,
        "config_sha256": sha256(checkpoint.config_bytes).hexdigest(),
    }


@contextmanager
def open_verified(checkpoint: LocalCheckpoint, task: str):
    """Open the weights after checking that they belong to this task and configuration."""
    with safe_open(checkpoint.weights, framework="pt", device="cpu") as artifact:
        if artifact.metadata() != required_metadata(checkpoint, task):
            raise ValueError("Weights do not match their task and configuration")
        yield artifact


def read_tensors(checkpoint: LocalCheckpoint, *, task: str) -> dict[str, torch.Tensor]:
    with open_verified(checkpoint, task) as artifact:
        return {name: artifact.get_tensor(name) for name in artifact.keys()}


def weight_layout(stream, required):
    """Read the header from an open file and return the data offset, length and tensor byte ranges.

    Returns None for metadata or layouts the direct path does not handle; safetensors
    then validates the file and reports the error.
    """
    try:
        size = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(size))
        if header.pop("__metadata__", None) != required:
            return None
        entries, length = {}, 0
        for name, entry in header.items():
            dtype = SAFETENSORS_DTYPES.get(entry["dtype"])
            shape = tuple(entry["shape"])
            begin, end = entry["data_offsets"]
            if dtype is None or begin < 0 or begin % dtype.itemsize:
                return None
            if end - begin != math.prod(shape) * dtype.itemsize:
                return None
            entries[name] = (dtype, shape, begin, end)
            length = max(length, end)
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    return 8 + size, length, entries


def upload_groups(entries):
    """Group tensors in file order into byte ranges of about UPLOAD_CHUNK_BYTES each."""
    group, low = [], 0
    for name, (dtype, shape, begin, end) in sorted(entries.items(), key=lambda item: item[1][2]):
        if group and end - low > UPLOAD_CHUNK_BYTES:
            yield low, group
            group = []
        if not group:
            # Every tensor begins at a multiple of its item size, and item sizes divide 8.
            low = begin - begin % 8
        group.append((name, dtype, shape, begin, end))
    if group:
        yield low, group


def read_tensors_to_device(
    checkpoint: LocalCheckpoint, *, task: str, device, dtypes=None,
) -> dict[str, torch.Tensor]:
    """Copy the weights to the device in a few large transfers and split them into separate tensors.

    A transfer per tensor is slow when other processes share the GPU. Each transfer
    covers about UPLOAD_CHUNK_BYTES of the file, and each tensor is then copied on the
    device into its own allocation, converted to ``dtypes[name]`` when given: views at
    arbitrary offsets of one buffer change kernel selection and results. Loading uses
    at most one chunk of device memory beyond the weights. The header, metadata and
    data come from one open file. Other layouts use per-tensor copies.
    """
    dtypes = dtypes or {}
    with checkpoint.weights.open("rb") as stream:
        layout = weight_layout(stream, required_metadata(checkpoint, task))
        if layout is not None and layout[1]:
            start, length, entries = layout
            # A private mapping is writable, so torch does not warn; nothing writes to it.
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY) as data:
                host = torch.frombuffer(data, dtype=torch.uint8, count=length, offset=start)
                tensors = {}
                for low, group in upload_groups(entries):
                    # On the CPU this stays the mapping itself; the copies below finish before it closes.
                    chunk = host[low:max(end for *_, end in group)].to(device)
                    for name, dtype, shape, begin, end in group:
                        view = chunk[begin - low:end - low].view(dtype).view(shape)
                        target = dtypes.get(name, dtype)
                        tensors[name] = view.clone() if target == dtype else view.to(target)
                    del chunk
                del host
            return tensors
    if layout is not None:
        return {
            name: torch.empty(shape, dtype=dtypes.get(name, dtype), device=device)
            for name, (dtype, shape, _, _) in layout[2].items()
        }
    return {
        name: tensor.to(device=device, dtype=dtypes.get(name, tensor.dtype))
        for name, tensor in read_tensors(checkpoint, task=task).items()
    }


def assign_tensors(model_type, config, tensors, *, match_dtype=False, model=None):
    """Allocate only the module structure before assigning its weights.

    ``model`` may be a weightless (meta) model of this configuration that was already built.
    """
    if model is None:
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
