import math

import pytest
import torch
from torch.nn import functional as F

from causilo.nn.embeddings import FeatureEmbedding


@pytest.mark.parametrize("group_size", [1, 3, 5])
@pytest.mark.parametrize("features", [1, 7, 15])
def test_sum_before_projection_preserves_missing_padding_and_bias(group_size, features):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(18)
        embedding = FeatureEmbedding(width=16, group_size=group_size, frequencies=4)
        torch.nn.init.normal_(embedding.frequencies, std=0.1)
        torch.nn.init.normal_(embedding.missing, std=0.1)
        table = torch.randn(2, 5, features)
        table[:, 0] = float("nan")
        table[0, 1, 0] = float("inf")
        table[1, 2, 0] = float("-inf")

    # Preserve the released computation as a numerical reference, especially
    # its bias contribution for missing features and numeric-zero padding.
    grouped = F.pad(table, (0, (-features) % group_size)).unflatten(-1, (-1, group_size))
    missing = grouped.isnan()
    phase = torch.nan_to_num(grouped, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(-1)
    phase = phase * (2 * math.pi) * embedding.frequencies
    waves = torch.cat((phase.sin(), phase.cos()), dim=-1).masked_fill(missing.unsqueeze(-1), 0)
    with torch.inference_mode():
        expected = embedding.projection(waves).sum(-2) / math.sqrt(group_size)
        expected = expected + (missing.float() @ embedding.missing) / math.sqrt(group_size)
        actual = embedding(table)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
