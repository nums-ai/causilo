"""Context-fitted target encoding and mapping predictions back to target units."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.validation import column_or_1d


class TargetPreprocessor:
    """Encode labels or standardize regression targets before the model's float32 cast."""

    def __init__(self, classification: bool):
        self.classification = classification

    def fit_transform(self, y, *, sample_count=None) -> np.ndarray:
        if self.classification:
            self.encoder_ = LabelEncoder()
            return self.encoder_.fit_transform(y)
        values = column_or_1d(y, dtype=np.float64, warn=True).reshape(-1, 1)
        self.encoder_ = StandardScaler()
        standardized = self.encoder_.fit_transform(values, sample_weight=sample_count)
        return standardized.ravel().astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return self.encoder_.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)
