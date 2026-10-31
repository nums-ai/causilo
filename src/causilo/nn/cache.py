"""Plain tensor containers that can be moved and serialized without rebuilding attention."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FrequencyWeights:
    """Row multiplicities prepared once per execution, shared by every attention layer.

    The additive bias broadcasts across members, feature groups, heads and
    queries. Compute its logarithm in FP32 before any lower-precision cast so
    large counts cannot overflow FP16. This metadata is not part of K/V caches.
    """

    log_bias: Tensor
    count: int

    @classmethod
    def resolve(cls, frequencies, device: torch.device) -> "FrequencyWeights | None":
        """Reuse prepared metadata, while accepting raw counts at runner entry points."""
        if frequencies is None or isinstance(frequencies, cls):
            return frequencies
        return cls.prepare(frequencies, device)

    @classmethod
    def prepare(cls, frequencies: Tensor, device: torch.device) -> "FrequencyWeights":
        return cls(
            frequencies.to(device=device, dtype=torch.float32).log()[None, None, :],
            int(frequencies.sum().item()),
        )


@dataclass(frozen=True)
class KVPair:
    """Projected keys/values shaped (*batch, heads, context_length, head_width)."""

    key: Tensor
    value: Tensor


@dataclass(frozen=True)
class BlockCache:
    """Layer K/V pairs in forward execution order; query rows never enter this state."""

    layers: tuple[KVPair, ...]


@dataclass(frozen=True)
class CacheResult:
    """Updated training activations plus the context collected while producing them."""

    rows: Tensor
    cache: BlockCache
