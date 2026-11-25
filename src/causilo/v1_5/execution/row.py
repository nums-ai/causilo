from __future__ import annotations

import torch
from torch import Tensor

from ..nn.row import RowBlock
from .chunks import ChunkRunner, batch_items
from .memory import Stage, Workload


class RowRunner:
    """Own the buffers and precision policy for row interaction."""

    def __init__(self, block: RowBlock, device: torch.device):
        self.block = block
        self.chunks = ChunkRunner(device)

    def pool(self, latents: Tensor) -> Tensor:
        return self.chunks.run(
            self.block.pool_latents, latents,
            Workload(Stage.POOL, latents.shape[-2], self.block.width, output_elements=self.block.output_width),
        )

    def refine(self, features: Tensor, latents: Tensor | None, layer_index: int) -> Tensor:
        tokens = self.block.num_latents
        if not self.block.layers[layer_index].final:
            tokens += features.shape[-2]
        leading = tuple(features.shape[:-2])

        def refine(part, section):
            context = None if latents is None else batch_items(latents, leading, section).to(part.device)
            return self.block.refine(part, context, layer_index=layer_index)

        return self.chunks.run_indexed(
            refine, features,
            Workload(Stage.ROW, features.shape[-2], self.block.width, output_elements=tokens * self.block.width),
        )
