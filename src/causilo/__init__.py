"""Tabular prediction with fixed pretrained context models."""

__version__ = "1.0.2"

from .estimators import CausiloClassifier, CausiloRegressor

__all__ = ["CausiloClassifier", "CausiloRegressor"]
