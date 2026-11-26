"""Load a hosted network for integrations that manage pretrained weights."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .model_selection import DEFAULT_MODEL_VERSION, select_checkpoint

if TYPE_CHECKING:
    from torch import nn


def load_pretrained_model(
    task: str, model_version: str = DEFAULT_MODEL_VERSION, device: str = "cpu",
) -> nn.Module:
    """Load the pinned task network on one device, using the runtime's precision policy.

    ``task`` is ``"classification"`` or ``"regression"``. This loads weights only;
    preprocessing and labeled contexts are prepared by the estimators. Each call
    returns a network; callers may share it across independent fitted contexts.
    """
    if task not in ("classification", "regression"):
        raise ValueError("task must be 'classification' or 'regression'")
    source = select_checkpoint(model_version, task)
    if not isinstance(device, str):
        raise ValueError("device must name one CPU, MPS, or CUDA device, or 'auto'")
    if model_version == "1.0":
        from .v1_0.checkpoints import load_model
        from .v1_0.engine import resolve_device

        target = resolve_device(device)
        model, _ = load_model(task, source)
        return model.to(target)

    from .v1_5.checkpoints import load_model
    from .v1_5.engine import resolve_device
    from .v1_5.execution.precision import place_model, supports_bf16

    target = resolve_device(device)
    model, _ = load_model(task, source)
    return place_model(model, target, bf16_output_projection=supports_bf16(target))
