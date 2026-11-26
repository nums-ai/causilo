"""Read Causilo 1.5 sizes from the checkpoint configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any

from ...checkpoint_io import assign_tensors
from ...model_selection import RUNTIME15_FORMAT
from .model import Model
from .model_config import ModelConfig


def validate_checkpoint(record: Mapping[str, Any]) -> ModelConfig:
    """Read the native model dimensions and verify the prediction task."""
    if (
        record.get("architecture") != RUNTIME15_FORMAT
        or record.get("format_version") != 1
        or record.get("config_version") != 1
    ):
        raise ValueError("Unsupported Causilo 1.5 configuration")
    config = record.get("model")
    if not isinstance(config, Mapping):
        raise ValueError("Checkpoint must contain a model configuration mapping.")
    if config.keys() != {field.name for field in fields(ModelConfig)}:
        raise ValueError("Checkpoint must contain the complete Causilo 1.5 model dimensions, without unknown fields.")
    model_config = ModelConfig(**config)
    task = "classification" if model_config.max_classes else "regression"
    if record.get("task") != task:
        raise ValueError("Checkpoint task does not match its model configuration.")
    return model_config


def build_model(config: ModelConfig, tensors, *, model=None):
    """Build the shared 1.5 model from the weights and configuration, optionally into a meta model."""
    return assign_tensors(Model, config, tensors, match_dtype=True, model=model)
