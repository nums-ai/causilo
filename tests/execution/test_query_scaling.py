import numpy as np
import pytest
import torch

from causilo import CausiloClassifier
from causilo.v1_5.nn.frequency import FrequencyWeights
from causilo.v1_5.nn.layers.scaling import QueryScale, reuse_length_gains


def test_query_scale_can_be_captured_as_one_graph():
    layer = QueryScale(2, 4).eval()
    query = torch.randn(2, 2, 3, 4)
    with torch.inference_mode():
        compiled = torch.compile(layer, fullgraph=True, backend="eager")
        torch.testing.assert_close(compiled(query, 9), layer(query, 9), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_query_length_cache_respects_precision_and_does_not_reuse_batched_lengths():
    layer = QueryScale(2, 4).cuda().eval()
    query = torch.randn(2, 2, 3, 4, device="cuda")
    lengths = torch.tensor([8, 11], device="cuda")
    cache = {}
    with torch.inference_mode():
        for autocast in (False, True):
            with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
                for count in (8, lengths, 8, 13):
                    expected = layer(query, count)
                    with reuse_length_gains(cache):
                        actual = layer(query, count)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(cache) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_weighted_length_gains_use_fitted_identity_without_a_device_scalar_read():
    from torch.utils._python_dispatch import TorchDispatchMode

    class NoScalarRead(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert func != torch.ops.aten._local_scalar_dense.default
            return func(*args, **(kwargs or {}))

    layer = QueryScale(2, 4).cuda().eval()
    query = torch.randn(2, 2, 3, 4, device="cuda")
    counts = torch.tensor([3, 5, 1], device="cuda")
    first, second = FrequencyWeights.prepare(counts), FrequencyWeights.prepare(counts + 2)
    calls = []
    cache = {}
    with torch.inference_mode():
        for dtype in (torch.float32, torch.float16):
            with torch.autocast("cuda", enabled=dtype == torch.float16):
                for weights in (first, second):
                    expected = layer._length_gain(query, weights.total)
                    with layer.length.register_forward_pre_hook(lambda *args: calls.append(1)), reuse_length_gains(cache):
                        with NoScalarRead():
                            miss = layer._length_gain(query, weights)
                            hit = layer._length_gain(query, weights)
                    assert hit is miss
                    torch.testing.assert_close(hit, expected, atol=0, rtol=0)
    assert len(calls) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_weighted_fitted_length_gains_follow_the_active_stream(checkpoint15, mixed_data):
    table, query = mixed_data
    model = CausiloClassifier(
        device="cuda:0", checkpoint_path=checkpoint15("classification"), n_estimators=1, use_kv_cache=True,
    ).fit(table, np.arange(len(table)) % 3, sample_count=1 + np.arange(len(table)) % 3)
    runner = model._engine.state.runner
    expected = model.predict_proba(query)
    first, second = torch.cuda.current_stream(), torch.cuda.Stream()
    for stream in (second, second, first):
        stream.wait_stream(runner._stream)
        with torch.cuda.stream(stream):
            np.testing.assert_array_equal(model.predict_proba(query), expected)
            assert runner._stream.cuda_stream == stream.cuda_stream


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip, reason="Requires NVIDIA CUDA")
@pytest.mark.parametrize("packed", [False, True])
def test_query_gain_matches_fp16_rounding(packed):
    pytest.importorskip("triton")
    from causilo.v1_5.kernels.triton.query_gain import apply_query_gain
    from causilo.v1_5.nn.layers.scaling import CONTENT_GAIN_LIMIT, bounded_multiplier

    content = torch.arange(65536, device="cuda", dtype=torch.int32).short().view(torch.float16).reshape(2, 4, -1, 16)
    if packed:
        query = torch.randn_like(content.transpose(1, 2).contiguous()).transpose(1, 2)
    else:
        query = torch.randn_like(content)
    length = torch.rand(2 if packed else 1, 4, 1, 16, device="cuda", dtype=torch.float16)
    expected = query * (length * bounded_multiplier(content, CONTENT_GAIN_LIMIT))
    actual = apply_query_gain(query, length, content, CONTENT_GAIN_LIMIT)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True, check_stride=True)
