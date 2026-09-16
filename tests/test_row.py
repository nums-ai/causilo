import pytest
import torch

from causilo.nn.row import RowBlock


@pytest.fixture
def row_block():
    def make(depth):
        with torch.random.fork_rng():
            torch.manual_seed(83)
            block = RowBlock(width=16, heads=2, expansion=2, num_latents=2, depth=depth).double().eval()
            torch.nn.init.normal_(block.latents, std=0.1)
        return block

    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield make
    torch.set_num_threads(previous)


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("batch", [(), (5,), (2, 5)])
def test_row_latents_reconstruct_full_and_reordered_feature_subsets(row_block, depth, batch):
    block = row_block(depth)
    features = torch.randn(*batch, 7, 16, generator=torch.Generator().manual_seed(87), dtype=torch.float64)
    original = features.clone()
    groups = torch.tensor([4, 1, 6])
    with torch.inference_mode():
        expected = block(features)
        trace = block.capture_latents(features)
        actual = block.replay_features(features, trace)
        subset = block.replay_features(features.index_select(-2, groups), trace)
        if depth == 1:
            torch.testing.assert_close(block.replay_features(features, None), expected, atol=1e-10, rtol=1e-10)

    assert trace.shape == (*batch, depth - 1, 2, 16)
    assert trace.dtype == features.dtype and trace.device == features.device
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(subset, expected.index_select(-2, groups), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(features, original, atol=0, rtol=0)


def test_row_replay_rejects_missing_or_mismatched_latent_states(row_block):
    block = row_block(3)
    features = torch.zeros(2, 7, 16, dtype=torch.float64)
    with torch.inference_mode():
        trace = block.capture_latents(features)
        with pytest.raises(ValueError, match="requires the captured"):
            block.replay_features(features, None)
        with pytest.raises(ValueError, match="trace shape"):
            block.replay_features(features, trace[:, :1])
        with pytest.raises(ValueError, match="trace shape"):
            block.replay_features(features, trace[:1])
