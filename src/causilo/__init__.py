"""Tabular prediction with fixed model weights and an inference context."""

__version__ = "1.5.1"

from .checkpoints import load_pretrained_model
from .estimators import CausiloClassifier, CausiloRegressor

__all__ = ["CausiloClassifier", "CausiloRegressor", "load_pretrained_model"]
