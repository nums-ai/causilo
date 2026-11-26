from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from ..nn.layers.feedforward import FeedForward
from .memory import Stage


def supports_bf16(device: torch.device) -> bool:
    """Check native BF16 support on the selected CUDA device."""
    if device.type != "cuda":
        return False
    with torch.cuda.device(device):
        return torch.cuda.is_bf16_supported(including_emulation=False)


def place_model(
    model: torch.nn.Module, device: torch.device, *, bf16_output_projection: bool,
) -> torch.nn.Module:
    """Place fixed weights and prepare the selected output-projection precision."""
    model.to(device)
    if bf16_output_projection:
        for layer in model.modules():
            if isinstance(layer, FeedForward):
                layer.prepare_bf16_output()
    return model


def stage_dtype(stage: Stage, device: torch.device) -> torch.dtype:
    """Choose the autocast dtype for a neural stage."""
    if device.type == "cuda" and stage is not Stage.FEATURE_EMBEDDING:
        return torch.float16
    return torch.float32


def run_inference(
    device: torch.device,
    forward_fn: Callable[..., Tensor],
    *args,
    stage: Stage,
) -> Tensor:
    """Apply the stage precision without modifying checkpoint parameters."""
    # Keep regular tensors when the caller has not entered inference mode.
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=stage_dtype(stage, device) == torch.float16,
        ),
    ):
        return forward_fn(*args)
