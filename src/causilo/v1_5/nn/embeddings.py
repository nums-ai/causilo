"""Map individual cells and context targets into the model width."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .ecdf import WeightedECDF, ecdf_ranks
from .model_config import ECDF_FREQUENCIES, FOURIER_FREQUENCIES, ModelConfig


def _fourier_features(values: Tensor, frequencies: Tensor) -> Tensor:
    phases = 2 * math.pi * values.unsqueeze(-1) * frequencies
    return torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)


class FeatureEmbedding(nn.Module):
    """Project raw values and context-fitted ranks, with a learned missing state."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        width = config.width
        self.width = width
        self.frequencies = nn.Parameter(torch.empty(1, FOURIER_FREQUENCIES))
        self.rank_frequencies = nn.Parameter(torch.empty(1, ECDF_FREQUENCIES))
        self.projection = nn.Linear(2 * (FOURIER_FREQUENCIES + ECDF_FREQUENCIES) + 1, width)
        self.missing = nn.Parameter(torch.empty(1, width))
        self.target = TargetEmbedding(width, config.max_classes)

    def forward(
        self, features: Tensor, context: Tensor | WeightedECDF, targets: Tensor | None,
    ) -> Tensor:
        """Embed each cell and condition its context prefix on the fitted targets."""
        embeddings = self.project_features(features, ecdf_ranks(features, context))
        return self.add_context_targets(embeddings, targets) if targets is not None else embeddings

    def project_features(self, features: Tensor, ranks: Tensor) -> Tensor:
        missing = torch.isnan(features)
        values = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        waves = _fourier_features(values, self.frequencies).masked_fill(missing.unsqueeze(-1), 0.0)
        rank_waves = _fourier_features(torch.nan_to_num(ranks), self.rank_frequencies)
        rank_waves = rank_waves.masked_fill(~torch.isfinite(ranks).unsqueeze(-1), 0.0)
        encoded = torch.cat((waves, values.unsqueeze(-1), rank_waves), dim=-1)
        embeddings = self.projection(encoded).sum(dim=-2)
        return embeddings + missing.to(embeddings.dtype) @ self.missing.to(embeddings.dtype)

    def add_context_targets(self, embeddings: Tensor, targets: Tensor) -> Tensor:
        """Add target embeddings to the context prefix in place."""
        context_rows = targets.shape[-1]
        embeddings[..., :context_rows, :].add_(self.target(targets))
        return embeddings


class TargetEmbedding(nn.Module):
    """Project one-hot class IDs or scalar regression targets."""

    def __init__(self, width: int, classes: int) -> None:
        super().__init__()
        self.classes = classes
        self.projection = nn.Linear(classes or 1, width)

    def forward(self, targets: Tensor) -> Tensor:
        if self.classes:
            values = F.one_hot(targets.long(), self.classes).float()
        else:
            values = targets.unsqueeze(-1)
        return self.projection(values)
