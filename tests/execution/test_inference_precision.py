import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from causilo import CausiloRegressor
from causilo.model_selection import RUNTIME15_FORMAT, read_local_checkpoint
from causilo.v1_5.checkpoints import load_checkpoint
from causilo.v1_5.execution import precision, quantiles
from causilo.v1_5.nn.checkpoint import validate_checkpoint
from causilo.v1_5.nn.layers import normalization
from causilo.v1_5.nn.layers.attention import Attention
from causilo.v1_5.nn.layers.feedforward import FeedForward
from causilo.v1_5.nn.layers.functional import unit_rms
from causilo.v1_5.nn.model_config import ModelConfig


def test_model_placement_keeps_checkpoint_parameters():
    model = FeedForward(width=8, expansion=2).eval()
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    placed = precision.place_model(model, torch.device("cpu"), bf16_output_projection=True)
    assert placed is model
    torch.testing.assert_close(model.state_dict(), expected, rtol=0, atol=0)
    assert torch.equal(model._output_weight_bf16, model.output.weight.bfloat16())
    assert torch.equal(model._output_bias_bf16, model.output.bias.bfloat16())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA model placement")
def test_model_placement_can_keep_fp32_output_on_bf16_hardware():
    device = torch.device("cuda:0")
    if not precision.supports_bf16(device):
        pytest.skip("Requires native CUDA BF16")
    model = FeedForward(width=8, expansion=2).eval()
    precision.place_model(model, device, bf16_output_projection=False)
    assert model.output.weight.device == device
    assert model._output_weight_bf16 is None and model._output_bias_bf16 is None


@torch.inference_mode()
def test_cpu_prefix_normalization_preserves_the_full_residual():
    rows = torch.randn(2, 7, 12)
    update = torch.randn_like(rows)
    norm = normalization.create_norm(12)
    residual, hidden = normalization.feedforward_residual(rows, update, norm, prefix=4)
    expected = rows + update
    torch.testing.assert_close(residual, expected, rtol=0, atol=0)
    torch.testing.assert_close(hidden, norm(expected[:, :4]), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA residual normalization")
@pytest.mark.parametrize(("dtype", "prefix"), [(torch.float16, 0), (torch.float32, 4)])
def test_cuda_residual_and_normalization_match_torch(dtype, prefix):
    if not normalization.HAS_TRITON or torch.version.hip is not None:
        pytest.skip("Requires NVIDIA Triton")
    width = ModelConfig().width
    norm = normalization.create_norm(width).cuda()
    rows = torch.randn(1, 7, width, device="cuda", dtype=dtype).expand(3, -1, -1)
    update = torch.randn(rows.shape, device="cuda", dtype=torch.float16)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        residual, hidden = normalization.feedforward_residual(rows, update, norm, prefix=prefix)
        expected = rows + update
        normalized = norm(expected[:, :prefix] if prefix else expected).half()
    torch.testing.assert_close(residual, expected, rtol=0, atol=0)
    torch.testing.assert_close(hidden, normalized, rtol=0, atol=0)


def test_loading_half_precision_checkpoint_preserves_values(checkpoint15):
    path = checkpoint15("regression", weight_dtype=torch.float16)
    checkpoint = read_local_checkpoint(path)
    expected = {name: weights.float() for name, weights in load_file(path / "model.safetensors").items()}
    model = load_checkpoint(checkpoint, "regression")

    assert model.config.to_dict() == checkpoint.config["model"]
    assert not model.training
    torch.testing.assert_close(model.state_dict(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(("changes", "message"), [
    ({"architecture": "other"}, "Unsupported Causilo 1.5 configuration"),
    ({"format_version": 2}, "Unsupported Causilo 1.5 configuration"),
    ({"config_version": 2}, "Unsupported Causilo 1.5 configuration"),
    ({"task": "regression"}, "Checkpoint task does not match"),
    ({"model": {"width": 16}}, "complete Causilo 1.5 model dimensions"),
])
def test_loader_validates_native_config(changes, message):
    record = {
        "architecture": RUNTIME15_FORMAT,
        "format_version": 1,
        "config_version": 1,
        "task": "classification",
        "model": ModelConfig(width=16, column_depth=2, prediction_depth=2).to_dict(),
    }
    with pytest.raises(ValueError, match=message):
        validate_checkpoint(record | changes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA SwiGLU precision")
def test_swiglu_product_matches_bf16_rounding():
    pytest.importorskip("triton")
    if torch.version.hip is not None or not precision.supports_bf16(torch.device("cuda")):
        pytest.skip("Requires native NVIDIA BF16")
    from causilo.v1_5.kernels.triton.swiglu import gated_product

    bits = torch.arange(-32768, 32768, device="cuda", dtype=torch.int32).to(torch.int16)
    gate = bits.view(torch.float16)
    value = gate.roll(1009)
    # An extra cell exercises the masked tail of the final tile.
    gate = torch.cat((gate, gate[:1]))
    value = torch.cat((value, value[:1]))
    expected = (torch.nn.functional.silu(gate.float()) * value.float()).bfloat16()
    actual = gated_product(value, gate)
    valid = ~expected.isnan()
    assert torch.equal(actual.isnan(), expected.isnan())
    assert torch.equal(actual.view(torch.int16)[valid], expected.view(torch.int16)[valid])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA SwiGLU precision")
def test_cuda_swiglu_projects_a_wide_gate_product_in_bf16():
    device = torch.device("cuda")
    if not precision.supports_bf16(device):
        pytest.skip("Requires native CUDA BF16")

    layer = FeedForward(width=8, expansion=2).to(device)
    with torch.inference_mode():
        layer.value.weight.zero_()
        layer.gate.weight.zero_()
        layer.value.bias.fill_(512)
        layer.gate.bias.fill_(512)
        layer.output.weight.fill_(1e-5)
        layer.output.bias.zero_()
        rows = torch.zeros((2, 8), device=device)
        with torch.autocast("cuda", dtype=torch.float16):
            fp32_output = layer(rows)
        layer.prepare_bf16_output()
        with torch.autocast("cuda", dtype=torch.float16):
            bf16_output = layer(rows)

    expected = torch.full_like(rows, 42.0)
    assert not torch.equal(fp32_output, expected)
    torch.testing.assert_close(bf16_output, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="Requires NVIDIA CUDA")
def test_compiled_residual_uses_the_native_normalization(monkeypatch):
    pytest.importorskip("triton")
    from causilo.v1_5.kernels.triton import residual_normalization

    class FusedKernelSelected(Exception):
        pass

    def mark_fused_kernel(*args):
        raise FusedKernelSelected

    monkeypatch.setattr(residual_normalization, "prepare_feedforward", mark_fused_kernel)
    width = ModelConfig().width
    norm = normalization.create_norm(width).cuda()
    rows = torch.randn(7, 9, width, device="cuda")
    update = torch.randn_like(rows, dtype=torch.float16)

    def prepare(rows, update):
        return normalization.feedforward_residual(rows, update, norm, prefix=4)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        with pytest.raises(FusedKernelSelected):
            prepare(rows, update)
        residual, normalized = torch.compile(prepare, fullgraph=True, backend="eager")(rows, update)
        expected = rows + update
        torch.testing.assert_close(residual, expected, rtol=0, atol=0)
        torch.testing.assert_close(normalized, norm(expected[:, :4]), rtol=0, atol=0, check_dtype=True)


def _torch_head_rms(heads):
    data = heads.float()
    rms = data.square().mean(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps).sqrt()
    return (data / rms).to(heads.dtype)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="Requires NVIDIA CUDA")
@pytest.mark.parametrize("width", [8, 32, 64, 128])
def test_cuda_packed_head_normalization_matches_torch_reduction(width):
    pytest.importorskip("triton")
    generator = torch.Generator(device="cuda").manual_seed(17)
    packed = torch.randn((5, 257, 7, width), device="cuda", dtype=torch.float16, generator=generator)
    packed[0, 0, 0].zero_()
    packed[0, 0, 1].mul_(1e-4)
    heads = packed.transpose(-3, -2)
    expected = _torch_head_rms(heads)
    with torch.inference_mode():
        actual = unit_rms(heads)
    assert actual.data_ptr() == heads.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, check_stride=True)


@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")),
])
@torch.inference_mode()
def test_head_normalization_fallback_matches_torch_for_strided_inputs(device):
    heads = torch.randn(2, 17, 3, 16, dtype=torch.float16, device=device).transpose(-3, -2)[..., ::2]
    original = heads.clone()
    expected = _torch_head_rms(heads)
    torch.testing.assert_close(unit_rms(heads), expected, rtol=0, atol=0)
    torch.testing.assert_close(heads, original, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="Requires NVIDIA CUDA")
def test_cuda_projected_cache_keeps_only_two_disjoint_head_buffers():
    pytest.importorskip("triton")
    layer = Attention(width=32, heads=4, scale_query=False).cuda().half()
    rows = torch.randn(2, 37, 32, dtype=torch.float16, device="cuda")
    context = rows[:, :33]
    original = rows.clone()
    with torch.inference_mode():
        query, key, value = torch.nn.functional._in_projection_packed(
            rows, context, context, layer.projection.weight, layer.projection.bias,
        )
        actual_query, joint = layer.project(rows, context)
        torch.testing.assert_close(actual_query, _torch_head_rms(layer._heads(query)), rtol=0, atol=0)
        expected_key = _torch_head_rms(layer._heads(key))
        for cache in (joint, layer.prepare_context(context, contiguous=True)):
            torch.testing.assert_close(cache.key, expected_key, rtol=0, atol=0, check_stride=True)
            torch.testing.assert_close(cache.value, layer._heads(value), rtol=0, atol=0, check_stride=True)
            storage = cache.key.untyped_storage()
            assert storage.data_ptr() == cache.value.untyped_storage().data_ptr()
            value_bytes = cache.value.numel() * cache.value.element_size()
            assert storage.nbytes() == 2 * value_bytes
            assert cache.value.data_ptr() == cache.key.data_ptr() + value_bytes
    assert torch.equal(rows, original)


def test_regression_preserves_variation_around_large_target_offsets(checkpoint15, mixed_data):
    table, query = mixed_data
    targets = np.arange(len(table), dtype=np.float64)
    offset = 1e9
    estimator = CausiloRegressor(
        device="cpu", checkpoint_path=checkpoint15("regression"),
        use_kv_cache=True, n_estimators=8, random_state=42,
    )
    options = dict(output_type="quantiles", quantiles=[0.15, 0.5, 0.85])
    centered = estimator.fit(table, targets).predict(query, **options)
    shifted = estimator.fit(table, targets + offset).predict(query, **options)
    assert shifted.dtype == np.float64
    assert np.ptp(shifted) > 0
    np.testing.assert_allclose(shifted - offset, centered, atol=np.spacing(offset), rtol=0)


def test_native_quantile_tails_follow_adjacent_points_and_preserve_request_order():
    from causilo.v1_0.quantiles import interpolate_quantiles as interpolate_v1_0

    values = torch.tensor([[-1000.0, 0.0, 3.0, 4.0, 4.0], [-1.0, -1.0, 0.0, 2.0, 1000.0]], dtype=torch.float64)
    levels = [0.99, 0.65, 0.01, 0.65]
    actual = quantiles.interpolate_quantiles(values, levels)
    expected = interpolate_v1_0(values.numpy(), np.asarray(levels))
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6, rtol=1e-6)
    assert actual[0, 0] == values[0, -1]
    assert actual[1, 2] == values[1, 0]
    torch.testing.assert_close(
        quantiles.interpolate_quantiles(values, [0.65, 0.65]), actual[:, [1, 3]], rtol=0, atol=0,
    )
