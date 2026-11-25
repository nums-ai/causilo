"""Compressed row multiplicities used by context attention and empirical ranks."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, eq=False)
class FrequencyWeights:
    """Counts owned by one fitted runner; identity distinguishes fitted tables."""

    counts: Tensor
    log_counts: Tensor
    total: Tensor

    @classmethod
    def prepare(cls, counts: Tensor):
        # log(n) is a pre-softmax correction. The length-dependent query scale
        # uses the expanded length as if all repetitions were materialized.
        return cls(counts, counts.to(dtype=torch.float32).log(), counts.sum())

    def key_bias(self, queries: Tensor) -> Tensor:
        """Broadcast log counts over the leading axes of projected queries."""
        return self.log_counts.to(device=queries.device).expand(*queries.shape[:-3], -1)
