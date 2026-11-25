from types import SimpleNamespace

import numpy as np
import pytest
import torch

from causilo import CausiloRegressor
from causilo.v1_5.data.target import TargetPreprocessor
from causilo.v1_5.execution import prediction
from causilo.v1_5.execution.device_mean import trimmed_mean_reducer
from causilo.v1_5.execution.direct import ReducedMean, _host_output
from causilo.v1_5.execution.outputs import trimmed_member_quantiles, trimmed_regression_predictions
from causilo.v1_5.kernels import HAS_TRITON


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_device_trimmed_mean_restores_member_signs_and_preserves_nonfinite_validation():
    device = torch.device("cuda", torch.cuda.current_device())
    if not HAS_TRITON or torch.version.hip is not None or torch.cuda.get_device_capability(device)[0] != 9:
        pytest.skip("Requires NVIDIA Hopper and Triton")
    levels, _, alignment = trimmed_member_quantiles("mean", None)
    signs = np.array([-1, 1, 1, -1, -1, 1, -1, 1], dtype=np.float32)
    target = TargetPreprocessor(classification=False)
    target.fit_transform([-1.0, 1.0])
    dataset = SimpleNamespace(
        ensemble=SimpleNamespace(trim_count=2, target_signs=signs),
        target_preprocessor=target,
        target_limits=(-1.0, 1.0),
    )
    runner = SimpleNamespace(device=device, _working_memory=lambda: 1 << 20)
    reduce = trimmed_mean_reducer(runner, dataset, None, alignment)
    assert reduce is not None

    base = np.linspace(-1, 1, len(levels), dtype=np.float32)
    members = np.tile(np.stack([base, 2 * base + 1]), (8, 1, 1))
    members[1:3, 0, 2:5] += 3
    members[5:8, 1, 7:12] -= 4

    def native_input(values):
        aligned = [member if sign > 0 else -member[:, ::-1] for member, sign in zip(values, signs, strict=True)]
        return torch.from_numpy(np.stack(aligned)).to(device)

    def host(outputs):
        return trimmed_regression_predictions(
            outputs, target, signs, output_type="mean", levels=levels,
            alignment=alignment, target_limits=dataset.target_limits, trim=2,
        )

    raw = native_input(members)
    means = _host_output(raw, reduce)
    assert isinstance(means, ReducedMean)
    np.testing.assert_allclose(means.values, host(raw.cpu().numpy()), rtol=1e-14, atol=1e-14)

    members[:5, 0, 18] = np.inf
    raw = native_input(members)
    fallback = _host_output(raw, reduce)
    np.testing.assert_array_equal(fallback, raw.cpu().numpy())
    with pytest.raises(ValueError, match="infinity"):
        host(fallback)

    target.fit_transform([1e12 - 1, 1e12 + 1])
    assert trimmed_mean_reducer(runner, dataset, None, alignment) is None


@pytest.mark.parametrize("cached", [False, True])
def test_chunk_unwraps_a_completed_mean_or_combines_native_outputs(checkpoint15, mixed_data, monkeypatch, cached):
    table, query = mixed_data
    model = CausiloRegressor(
        checkpoint_path=checkpoint15("regression"), n_estimators=8, use_kv_cache=cached, device="cpu",
    ).fit(table, np.arange(len(table), dtype=float))
    expected = model.predict(query)

    calls = []

    def reduce(outputs):
        calls.append(outputs.shape)
        return torch.full((outputs.shape[1],), 123.0, dtype=torch.float64) if len(calls) == 1 else None

    monkeypatch.setattr(prediction, "trimmed_mean_reducer", lambda *args: reduce)
    np.testing.assert_array_equal(model.predict(query), np.full(len(query), 123.0))
    np.testing.assert_array_equal(model.predict(query), expected)
    assert calls == [(8, len(query), 1001)] * 2
