"""Mix feature groups within each row through a fixed number of latent vectors.

All blocks treat leading axes independently. Feature tensors are
(*batch, feature_groups, width); in ModelRunner, *batch includes table rows.
"""

import torch
from torch import Tensor, nn

from .layers.attention import Attention
from .layers.feedforward import FeedForward


class LatentUpdate(nn.Module):
    """Let latent queries read both existing latents and feature-group tokens."""

    def __init__(self, width: int, heads: int, expansion: int) -> None:
        super().__init__()
        self.context_norm = nn.RMSNorm(width, eps=1e-5)
        self.read = Attention(width, heads)
        self.output_norm = nn.RMSNorm(width, eps=1e-5)
        self.feedforward = FeedForward(width, expansion)

    def forward(self, latents: Tensor, features: Tensor) -> Tensor:
        """Return updated (*batch, row_latents, width) latent tokens."""
        context = self.context_norm(torch.cat((latents, features), dim=-2))
        latents = latents + self.read(context[..., : latents.shape[-2], :], context)
        return latents + self.feedforward(self.output_norm(latents))


class FeatureUpdate(nn.Module):
    """Apply a residual attention update from latent context to arbitrary row tokens."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.context_norm = nn.RMSNorm(width, eps=1e-5)
        self.read = Attention(width, heads)

    def forward(self, rows: Tensor, latents: Tensor) -> Tensor:
        return rows + self.read(self.context_norm(rows), self.context_norm(latents))


class RowLayer(nn.Module):
    """Update feature/latent tokens, then refine latents from their combined context."""

    def __init__(self, width: int, heads: int, expansion: int) -> None:
        super().__init__()
        self.features = FeatureUpdate(width, heads)
        self.update_latents = LatentUpdate(width, heads, expansion)

    def forward(self, latents: Tensor, features: Tensor) -> tuple[Tensor, Tensor]:
        count = latents.shape[-2]
        combined = self.features(torch.cat((latents, features), dim=-2), latents)
        latents, features = combined[..., :count, :], combined[..., count:, :]
        return self.update_latents(latents, features), features


class RowBlock(nn.Module):
    """Mix feature groups while preserving their count for the next column stage."""

    def __init__(self, width: int, heads: int, expansion: int, num_latents: int, depth: int) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.empty(num_latents, width))
        self.rounds = nn.ModuleList(RowLayer(width, heads, expansion) for _ in range(depth - 1))
        self.finish = FeatureUpdate(width, heads)

    def forward(self, features: Tensor) -> Tensor:
        """Return features with the same (*batch, feature_groups, width) layout."""
        latents = self.latents.to(features).expand(*features.shape[:-2], *self.latents.shape)
        for refinement in self.rounds:
            latents, features = refinement(latents, features)
        return self.finish(features, latents)


class RowPool(nn.Module):
    """Summarize a variable number of feature groups into a fixed-width row vector."""

    def __init__(self, width: int, heads: int, expansion: int, num_latents: int, depth: int) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.empty(num_latents, width))
        self.rounds = nn.ModuleList(LatentUpdate(width, heads, expansion) for _ in range(depth))
        self.output_norm = nn.RMSNorm(width, eps=1e-5)

    def forward(self, features: Tensor) -> Tensor:
        """Map (*batch, feature_groups, width) to (*batch, row_latents * width)."""
        latents = self.latents.to(features).expand(*features.shape[:-2], *self.latents.shape)
        for update in self.rounds:
            latents = update(latents, features)
        return self.output_norm(latents).flatten(-2)
