"""Key/value projections for column and prediction layers."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor


@dataclass(frozen=True)
class KVPair:
    """Projected keys and values shaped (*batch, heads, context_rows, head_width)."""

    key: Tensor
    value: Tensor

    def __getitem__(self, indices) -> KVPair:
        """Slice key/value along batch dimensions."""
        return KVPair(key=self.key[indices], value=self.value[indices])

    def to(self, device) -> KVPair:
        """Move this pair to the requested device."""
        return KVPair(
            key=self.key.to(device),
            value=self.value.to(device),
        )


@dataclass
class BlockCache:
    """Map each layer index to its prepared K/V."""

    layers: dict[int, KVPair] = field(default_factory=dict)

    def __getitem__(self, indices) -> BlockCache:
        """Slice all layers along the leading batch axes."""
        return BlockCache({index: pair[indices] for index, pair in self.layers.items()})

    def to(self, device) -> BlockCache:
        """Move all layers to the requested device."""
        return BlockCache({index: pair.to(device) for index, pair in self.layers.items()})
