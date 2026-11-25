"""Mix features within each row through a fixed set of latent tokens."""

import torch
from torch import Tensor, nn

from .layers.attention import Attention
from .layers.feedforward import FeedForward
from .layers.normalization import create_norm, feedforward_residual, large_residual
from .model_config import EXPANSION, ROW_HEADS, ROW_LATENTS, ModelConfig


class RowLayer(nn.Module):
    """Gather features into latents, then broadcast to features between stages."""

    def __init__(self, width: int, heads: int, *, final: bool) -> None:
        super().__init__()
        self.final = final
        self.gather = Attention(width, heads, scale_query=True)
        self.broadcast = Attention(width, heads, scale_query=True)
        self.feedforward = FeedForward(width, EXPANSION)
        self.gather_norm = create_norm(width)
        self.broadcast_norm = create_norm(width)
        self.feedforward_norm = create_norm(width)

    @staticmethod
    def _gather(latents: Tensor, features: Tensor, attention: Attention, norm: nn.Module) -> Tensor:
        query = norm(latents)
        context = torch.cat((query, norm(features)), dim=-2)
        return latents + attention(query, context)

    def replay_features(self, features: Tensor, broadcast_latents: Tensor) -> Tensor:
        """Update a feature slice from the latent state of the full row."""
        query = self.broadcast_norm(features)
        context = self.broadcast_norm(broadcast_latents)
        return features + self.broadcast(query, context)

    def _mix_tokens(
        self, latents: Tensor, features: Tensor, *, keep_features: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        broadcast_latents = self._gather(latents, features, self.gather, self.gather_norm)
        # Replay will reconstruct feature updates from the gathered latents on demand.
        combined = torch.cat((broadcast_latents, features), dim=-2) if keep_features else broadcast_latents
        full_query_count = None if keep_features else ROW_LATENTS + features.shape[-2]
        query = self.broadcast_norm(combined)
        # A distinct, packed K/V view keeps the original projection layout even on latent-only replay.
        context = query[..., :ROW_LATENTS, :].contiguous()
        if large_residual(combined, extra_elements=0 if keep_features else features.numel()):
            update = self.broadcast(query, context, full_query_count=full_query_count)
            del query, context
            combined, normalized = feedforward_residual(combined, update, self.feedforward_norm, prefix=ROW_LATENTS)
            del update
            latents = combined[..., :ROW_LATENTS, :] + self.feedforward(normalized)
            return latents, combined, broadcast_latents
        combined = combined + self.broadcast(query, context, full_query_count=full_query_count)
        del query, context
        latents = combined[..., :ROW_LATENTS, :]
        latents = latents + self.feedforward(self.feedforward_norm(latents))
        return latents, combined, broadcast_latents

    def forward_with_replay(self, latents: Tensor, features: Tensor) -> tuple[Tensor, Tensor]:
        """Return updated latents and the gathered state used to replay feature updates."""
        updated, _, broadcast_latents = self._mix_tokens(latents, features, keep_features=False)
        return updated, broadcast_latents

    def forward(self, latents: Tensor, features: Tensor, *, packed: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        if not self.final:
            updated, combined, _ = self._mix_tokens(latents, features)
            features = combined[..., ROW_LATENTS:, :]
            if not packed:
                return updated, features
            if torch.is_grad_enabled() or updated.dtype != combined.dtype:
                return torch.cat((updated, features), dim=-2)
            # The broadcast residual already owns the feature suffix of the next layer's input.
            combined[..., :ROW_LATENTS, :].copy_(updated)
            return combined

        # Only latents reach the prediction block. The final layer uses both
        # learned projections to gather feature context into those latents.
        latents = self._gather(latents, features, self.broadcast, self.broadcast_norm)
        latents = self._gather(latents, features, self.gather, self.gather_norm)
        updated = latents + self.feedforward(self.feedforward_norm(latents))
        return updated if packed else (updated, features)


class RowBlock(nn.Module):
    """Carry a row's latents across the paired column and row layers."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.width = config.width
        self.num_latents = ROW_LATENTS
        self.output_width = config.prediction_width
        depth = config.column_depth
        self.layers = nn.ModuleList(
            RowLayer(config.width, ROW_HEADS, final=index == depth - 1) for index in range(depth)
        )
        self.latents = nn.Parameter(torch.empty(self.num_latents, config.width))
        self.output_norm = create_norm(config.width)

    def pool_latents(self, latents: Tensor) -> Tensor:
        """Concatenate normalized latents into a fixed-width row vector."""
        return self.output_norm(latents).flatten(-2)

    def refine(self, features: Tensor, latents: Tensor | None, *, layer_index: int) -> Tensor:
        if latents is None:
            latents = self.latents.to(features).expand(*features.shape[:-2], self.num_latents, self.width)
        layer = self.layers[layer_index]
        return layer(latents, features, packed=True)
