"""Tabular prediction with fixed model weights and an inference context."""

__version__ = "1.5.1"

from .estimators import CausiloClassifier, CausiloRegressor

__all__ = ["CausiloClassifier", "CausiloRegressor"]
