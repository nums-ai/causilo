"""Ragged tensor execution must preserve each independent table's prediction."""

from dataclasses import replace

import pytest
import torch

from causilo.execution.chunks import ChunkPlan
from causilo.execution.runner import ModelCache, ModelRunner
from causilo.nn.cache import BlockCache, KVPair
from causilo.nn.layers.attention import Attention


@pytest.fixture
def small_model(request, make_batch_model):
    return make_batch_model(request.param)


def tables():
    rng = torch.Generator().manual_seed(53)
    specs = [(3, 2, 2), (7, 4, 5), (5, 3, 9)]
    result = []
    for training, query, features in specs:
        table = torch.randn(1, training + query, features, generator=rng)
        table[0, 0, 0] = torch.nan
        targets = torch.arange(training).remainder(3).float()[None]
        result.append((table, targets))
    return result


def pack_tables(items):
    training = max(y.shape[1] for _, y in items)
    queries = max(x.shape[1] - y.shape[1] for x, y in items)
    width = max(x.shape[2] for x, _ in items)
    x = torch.zeros(len(items), training + queries, width)
    y = torch.zeros(len(items), training)
    counts, features = [], []
    for i, (table, targets) in enumerate(items):
        t, q, f = targets.shape[1], table.shape[1] - targets.shape[1], table.shape[2]
        x[i, :t, :f] = table[0, :t]
        x[i, training:training + q, :f] = table[0, t:]
        y[i, :t] = targets
        counts.append(t)
        features.append(f)
    return x, y, torch.tensor(counts), torch.tensor(features)


def pack_caches(caches):
    def block(states, axis):
        packed = []
        for layers in zip(*(state.layers for state in states)):
            pairs = []
            size = max(layer.key.shape[axis] for layer in layers)
            for layer in layers:
                shape = list(layer.key.shape)
                shape[axis] = size
                key, value = layer.key.new_zeros(shape), layer.value.new_zeros(shape)
                section = [slice(None)] * key.ndim
                section[axis] = slice(0, layer.key.shape[axis])
                key[tuple(section)], value[tuple(section)] = layer.key, layer.value
                pairs.append(KVPair(key, value))
            packed.append(KVPair(torch.cat([p.key for p in pairs]), torch.cat([p.value for p in pairs])))
        return BlockCache(tuple(packed))

    return ModelCache(
        tuple(block([cache.columns[i] for cache in caches], 1) for i in range(2)),
        block([cache.prediction for cache in caches], 2),
    )


@pytest.mark.parametrize("small_model", ["classification", "regression"], indirect=True)
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_ragged_predictions_match_individual_tables(small_model, cached, chunked, device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    small_model.to(device)
    runner = ModelRunner(small_model)
    items = tables()
    packed = pack_tables(items)
    items = [(x.to(device), y.to(device)) for x, y in items]
    with torch.inference_mode():
        expected = [runner.predict(x, y) for x, y in items]
        caches = [runner.build_cache(x[:, :y.shape[1]], y) for x, y in items] if cached else None
        x, y, training_lengths, feature_counts = (value.to(device) for value in packed)
        if chunked:
            choose = ChunkPlan.choose
            monkeypatch.setattr(ChunkPlan, "choose", lambda *args: replace(choose(*args), limit=2))
        if cached:
            actual = runner.predict_cached(
                x[:, y.shape[1]:], pack_caches(caches),
                training_lengths=training_lengths, feature_counts=feature_counts,
            )
        else:
            actual = runner.predict(x, y, training_lengths=training_lengths, feature_counts=feature_counts)
    for i, wanted in enumerate(expected):
        tolerance = 2e-3 if device == "cuda" else 3e-5
        torch.testing.assert_close(actual[i, :wanted.shape[1]], wanted[0], atol=tolerance, rtol=tolerance)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("small_model", ["classification", "regression"], indirect=True)
def test_padding_and_other_tables_do_not_change_real_outputs(small_model):
    runner = ModelRunner(small_model)
    items = tables()
    x, y, training_lengths, feature_counts = pack_tables(items)
    with torch.inference_mode():
        expected = runner.predict(x, y, training_lengths=training_lengths, feature_counts=feature_counts)
        # Neither padded training rows, other queries, nor wholly padded feature
        # groups may leak into the first table's real query outputs.
        x[0, training_lengths[0]:y.shape[1], :] = 50
        y[0, training_lengths[0]:] = 2
        x[0, y.shape[1] + 2:, :] = -31
        x[0, :, 3:] = 100  # The partial first group retains its original zero padding.
        x[1:] *= 17
        actual = runner.predict(x, y, training_lengths=training_lengths, feature_counts=feature_counts)
    torch.testing.assert_close(actual[0, :2], expected[0, :2], atol=3e-5, rtol=3e-5)


def test_attention_uses_valid_length_for_mask_and_query_scaling():
    with torch.random.fork_rng(), torch.inference_mode():
        torch.manual_seed(12)
        module = Attention(16, 2).eval()
        q, kv = torch.randn(2, 4, 16), torch.randn(2, 9, 16)
        lengths = torch.tensor([3, 7])
        actual = module(q, kv, key_lengths=lengths)
        expected = torch.cat([module(q[i:i + 1], kv[i:i + 1, :count]) for i, count in enumerate(lengths)])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
