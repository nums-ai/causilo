import json
from functools import cache as memoize

import numpy as np
import pytest
import torch

import causilo
from causilo import CausiloClassifier, CausiloRegressor, load_pretrained_model
from causilo.model_selection import read_local_checkpoint, select_checkpoint
from causilo.v1_0 import checkpoints as checkpoints10
from causilo.v1_5 import checkpoints as checkpoints15
from causilo.v1_5.engine import Engine
from causilo.v1_5.execution.precision import supports_bf16
from causilo.v1_5.nn.layers.feedforward import FeedForward


@pytest.fixture
def hosted_checkpoints(monkeypatch, checkpoint_factory, checkpoint15, batch_runtime):
    paths = {
        (version, task): (
            checkpoint_factory(task, name=f"legacy-{task}") if version == "1.0" else checkpoint15(task)
        )
        for version in ("1.0", "1.5")
        for task in ("classification", "regression")
    }
    monkeypatch.setattr(
        checkpoints10, "load_hosted_model",
        lambda task: checkpoints10.load_checkpoint(paths["1.0", task]),
    )

    def load15(task, source):
        assert source == select_checkpoint("1.5", task)
        return checkpoints15.load_checkpoint(read_local_checkpoint(paths["1.5", task]), task)

    monkeypatch.setattr(checkpoints15, "load_hosted_model", load15)
    return paths


@pytest.mark.parametrize("version", ["1.0", "1.5"])
@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is unavailable",
    )),
])
def test_public_loader_preserves_weights_and_placement(hosted_checkpoints, version, task, device):
    model = load_pretrained_model(task, model_version=version, device=device)
    assert not any(module.training for module in model.modules())
    assert all(parameter.device == torch.device(device) for parameter in model.parameters())
    checkpoints = checkpoints10 if version == "1.0" else checkpoints15
    expected, _ = checkpoints.load_model(task, select_checkpoint(version, task))
    for name, tensor in expected.state_dict().items():
        assert torch.equal(model.state_dict()[name].cpu(), tensor)
    if version == "1.5":
        bf16 = supports_bf16(torch.device(device))
        for layer in model.modules():
            if isinstance(layer, FeedForward):
                if bf16:
                    assert torch.equal(layer._output_weight_bf16, layer.output.weight.bfloat16())
                    assert torch.equal(layer._output_bias_bf16, layer.output.bias.bfloat16())
                else:
                    assert layer._output_weight_bf16 is None and layer._output_bias_bf16 is None


@pytest.mark.parametrize("task", ["binary", "regressor", None])
def test_public_loader_validates_task_before_loading(task, monkeypatch):
    monkeypatch.setattr(checkpoints15, "load_model", lambda *a, **k: pytest.fail("Loaded weights"))
    with pytest.raises(ValueError, match="task must be"):
        load_pretrained_model(task)


@pytest.mark.parametrize("version", ["1.0", "1.5"])
@pytest.mark.parametrize("device", [["cpu"], None, "meta"])
def test_public_loader_validates_single_device_before_loading(version, device, monkeypatch):
    for checkpoints in (checkpoints10, checkpoints15):
        monkeypatch.setattr(checkpoints, "load_model", lambda *a, **k: pytest.fail("Loaded weights"))
    with pytest.raises(ValueError, match="device|supports"):
        load_pretrained_model("regression", model_version=version, device=device)


def test_public_loader_validates_model_version():
    with pytest.raises(ValueError, match="model_version"):
        load_pretrained_model("classification", model_version="2.0")


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("cache", [False, True])
def test_hosted_fits_can_share_the_public_loader(hosted_checkpoints, monkeypatch, task, cache):
    constructor = CausiloClassifier if task == "classification" else CausiloRegressor
    table = np.random.default_rng(18).normal(size=(24, 4))
    targets = np.arange(len(table)) % 3 if task == "classification" else table[:, 0] ** 2
    query = table[:3]
    options = dict(n_estimators=1, device="cpu", use_kv_cache=cache)
    predict = "predict_proba" if task == "classification" else "predict"
    expected = getattr(constructor(**options).fit(table, targets), predict)(query)
    load = causilo.load_pretrained_model

    @memoize
    def shared_loader(task, model_version="1.5", device="cpu"):
        json.dumps((task, model_version, device))  # Frameworks must be able to record the inputs.
        return load(task, model_version=model_version, device=device)

    monkeypatch.setattr(causilo, "load_pretrained_model", shared_loader)
    fits = [constructor(**options).fit(table, targets) for _ in range(9)]
    info = shared_loader.cache_info()
    assert (info.misses, info.hits) == (1, 8)
    assert all(fit._engine.model is fits[0]._engine.model for fit in fits)
    assert len({id(fit._engine.state) for fit in fits}) == 9
    for fit in fits:
        np.testing.assert_array_equal(getattr(fit, predict)(query), expected)


def test_custom_hosted_source_keeps_its_own_loading_path(checkpoint15, monkeypatch):
    from dataclasses import replace

    source = replace(select_checkpoint("1.5", "regression"), revision="custom-revision")
    model = checkpoints15.load_checkpoint(read_local_checkpoint(checkpoint15("regression")), "regression")
    calls = []

    def load(task, requested, **kwargs):
        calls.append((task, requested))
        return model, None

    monkeypatch.setattr(checkpoints15, "load_model", load)
    monkeypatch.setattr(causilo, "load_pretrained_model", lambda *a, **k: pytest.fail("Loaded default weights"))
    engine = Engine("regression", "cpu", source)
    engine.initialize_model()
    assert engine.model is model
    assert calls == [("regression", source)]
