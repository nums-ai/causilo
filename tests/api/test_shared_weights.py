import copy
import gc

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from causilo import CausiloClassifier, CausiloRegressor, checkpoint_io
from causilo.checkpoint_io import read_tensors, read_tensors_to_device
from causilo.model_selection import read_local_checkpoint
from causilo.v1_5 import checkpoints

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def record_loads(monkeypatch):
    loads = []
    load = checkpoints.load_model

    def record(*args, **kwargs):
        loads.append(kwargs.get("device"))
        return load(*args, **kwargs)

    monkeypatch.setattr(checkpoints, "load_model", record)
    return loads


def labels(table):
    return np.arange(len(table)) % 3


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("device", DEVICES)
def test_single_buffer_upload_matches_per_tensor_weights(checkpoint15, weight_dtype, device, monkeypatch):
    checkpoint = read_local_checkpoint(checkpoint15("classification", weight_dtype=weight_dtype))
    expected = read_tensors(checkpoint, task="classification")
    uploaded = read_tensors_to_device(checkpoint, task="classification", device=device)
    assert uploaded.keys() == expected.keys()
    for name, tensor in expected.items():
        assert uploaded[name].device == torch.device(device)
        assert uploaded[name].dtype == tensor.dtype
        assert torch.equal(uploaded[name].cpu(), tensor)
    # Separate aligned allocations, as with per-tensor loading, keep kernel results unchanged.
    storages = {tensor.untyped_storage().data_ptr() for tensor in uploaded.values() if tensor.numel()}
    assert len(storages) == sum(tensor.numel() > 0 for tensor in uploaded.values())
    assert all(tensor.data_ptr() % 64 == 0 for tensor in uploaded.values() if tensor.numel())

    monkeypatch.setattr(checkpoint_io, "weight_layout", lambda *args: None)
    fallback = read_tensors_to_device(checkpoint, task="classification", device=device)
    for name, tensor in expected.items():
        assert torch.equal(fallback[name].cpu(), tensor)
    with pytest.raises(ValueError, match="do not match their task"):
        read_tensors_to_device(checkpoint, task="regression", device=device)


@pytest.mark.parametrize("device", DEVICES)
def test_direct_device_load_matches_host_load(checkpoint15, device):
    checkpoint = read_local_checkpoint(checkpoint15("classification"))
    host = checkpoints.load_checkpoint(checkpoint, "classification")
    direct = checkpoints.load_checkpoint(checkpoint, "classification", device)
    for (name, expected), (other, observed) in zip(host.state_dict().items(), direct.state_dict().items()):
        assert name == other
        assert observed.device == torch.device(device)
        assert torch.equal(observed.cpu(), expected)


def test_live_estimators_share_loaded_weights(checkpoint15, mixed_data, monkeypatch):
    loads = record_loads(monkeypatch)
    X, query = mixed_data
    path = checkpoint15("classification")
    first = CausiloClassifier(checkpoint_path=path, device="cpu", n_estimators=2).fit(X, labels(X))
    second = CausiloClassifier(checkpoint_path=path, device="cpu", n_estimators=2).fit(X, labels(X))
    copied = copy.deepcopy(first)
    assert len(loads) == 1
    assert second._engine.model is first._engine.model
    assert copied._engine.model is first._engine.model
    expected = first.predict_proba(query)
    np.testing.assert_array_equal(second.predict_proba(query), expected)
    np.testing.assert_array_equal(copied.predict_proba(query), expected)

    regressor = CausiloRegressor(checkpoint_path=checkpoint15("regression"), device="cpu", n_estimators=1)
    regressor.fit(X, X["a"])
    assert len(loads) == 2
    assert regressor._engine.model is not first._engine.model

    del first, second, copied, regressor
    gc.collect()
    CausiloClassifier(checkpoint_path=path, device="cpu", n_estimators=2).fit(X, labels(X))
    assert len(loads) == 3


def test_changed_local_weights_are_loaded_again(checkpoint15, mixed_data, monkeypatch):
    loads = record_loads(monkeypatch)
    X, query = mixed_data
    folder = checkpoint15("classification")
    first = CausiloClassifier(checkpoint_path=folder, device="cpu", n_estimators=1).fit(X, labels(X))

    weights = folder / "model.safetensors"
    with safe_open(weights, framework="pt") as artifact:
        metadata = artifact.metadata()
    tensors = load_file(weights)
    tensors = {name: tensor * 1.5 for name, tensor in tensors.items()}
    save_file(tensors, weights, metadata=metadata)

    second = CausiloClassifier(checkpoint_path=folder, device="cpu", n_estimators=1).fit(X, labels(X))
    assert len(loads) == 2
    assert second._engine.model is not first._engine.model
    assert second._engine.checkpoint["digest"] != first._engine.checkpoint["digest"]
    for name, tensor in second._engine.model.state_dict().items():
        assert torch.equal(tensor, tensors[name])
