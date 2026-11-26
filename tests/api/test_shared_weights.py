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
from causilo.v1_5 import engine as engine15

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


@pytest.mark.parametrize("chunk_bytes", [None, 256])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("device", DEVICES)
def test_chunked_upload_matches_per_tensor_weights(checkpoint15, weight_dtype, device, chunk_bytes, monkeypatch):
    if chunk_bytes is not None:
        # Small chunks split the file into many transfers, some holding one tensor larger than a chunk.
        monkeypatch.setattr(checkpoint_io, "UPLOAD_CHUNK_BYTES", chunk_bytes)
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
def test_upload_converts_to_requested_dtypes(checkpoint15, device, monkeypatch):
    monkeypatch.setattr(checkpoint_io, "UPLOAD_CHUNK_BYTES", 256)
    checkpoint = read_local_checkpoint(checkpoint15("classification", weight_dtype=torch.float16))
    expected = read_tensors(checkpoint, task="classification")
    dtypes = {name: torch.float32 for name in expected}
    for fallback in (False, True):
        if fallback:
            monkeypatch.setattr(checkpoint_io, "weight_layout", lambda *args: None)
        uploaded = read_tensors_to_device(checkpoint, task="classification", device=device, dtypes=dtypes)
        for name, tensor in expected.items():
            assert uploaded[name].dtype == torch.float32
            assert torch.equal(uploaded[name].cpu(), tensor.float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16])
def test_upload_uses_at_most_one_chunk_beyond_the_weights(checkpoint15, weight_dtype, monkeypatch):
    chunk_bytes = 4096
    monkeypatch.setattr(checkpoint_io, "UPLOAD_CHUNK_BYTES", chunk_bytes)
    checkpoint = read_local_checkpoint(checkpoint15("classification", weight_dtype=weight_dtype))
    expected = read_tensors(checkpoint, task="classification")
    dtypes = {name: torch.float32 for name in expected}
    largest = max(tensor.nbytes for tensor in expected.values())
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    uploaded = read_tensors_to_device(checkpoint, task="classification", device="cuda", dtypes=dtypes)
    torch.cuda.synchronize()
    weights = torch.cuda.memory_allocated() - base
    peak = torch.cuda.max_memory_allocated() - base
    # One chunk spans at most chunk_bytes, or one tensor plus its alignment padding.
    chunk_allocation = -(-(max(chunk_bytes, largest) + 8) // 512) * 512
    assert peak <= weights + chunk_allocation
    assert len(uploaded) == len(expected)


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


def test_checkpoint_replaced_during_load_is_rejected(checkpoint15, mixed_data, monkeypatch):
    X, _ = mixed_data
    folder = checkpoint15("classification")
    weights = folder / "model.safetensors"
    with safe_open(weights, framework="pt") as artifact:
        metadata = artifact.metadata()
    tensors = {name: tensor * 1.5 for name, tensor in load_file(weights).items()}
    load = checkpoints.load_model

    def replace_then_load(*args, **kwargs):
        # The weights were fingerprinted before this point; an atomic replace swaps the file.
        replacement = folder / "replacement.safetensors"
        save_file(tensors, replacement, metadata=metadata)
        replacement.replace(weights)
        return load(*args, **kwargs)

    monkeypatch.setattr(checkpoints, "load_model", replace_then_load)
    with pytest.raises(ValueError, match="changed while it was loading"):
        CausiloClassifier(checkpoint_path=folder, device="cpu", n_estimators=1).fit(X, labels(X))
    assert not engine15._shared_models

    monkeypatch.setattr(checkpoints, "load_model", load)
    model = CausiloClassifier(checkpoint_path=folder, device="cpu", n_estimators=1).fit(X, labels(X))
    for name, tensor in model._engine.model.state_dict().items():
        assert torch.equal(tensor, tensors[name])


def test_projection_precision_mismatch_loads_separate_weights(checkpoint15, mixed_data, monkeypatch):
    loads = record_loads(monkeypatch)
    X, _ = mixed_data
    path = checkpoint15("classification")

    def fit(bf16):
        monkeypatch.setattr(engine15, "supports_bf16", lambda device: bf16)
        return CausiloClassifier(checkpoint_path=path, device="cpu", n_estimators=1).fit(X, labels(X))

    bf16 = fit(True)
    full = fit(False)
    assert len(loads) == 2
    assert full._engine.model is not bf16._engine.model
    # The first loaded model stays the shared one for engines with its precision.
    assert fit(True)._engine.model is bf16._engine.model
    assert len(loads) == 2


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_replicas_on_two_gpus_match_host_weights(checkpoint15, mixed_data, monkeypatch):
    loads = record_loads(monkeypatch)
    X, query = mixed_data
    path = checkpoint15("classification")
    expected = checkpoints.load_checkpoint(read_local_checkpoint(path), "classification").state_dict()
    devices = ["cuda:0", "cuda:1"]
    first = CausiloClassifier(checkpoint_path=path, device=devices, n_estimators=4).fit(X, labels(X))
    replicas = first._engine._model_replicas
    assert sorted(str(device) for device in replicas) == devices
    for device, replica in replicas.items():
        for name, tensor in replica.state_dict().items():
            assert tensor.device == device
            assert torch.equal(tensor.cpu(), expected[name])
    second = CausiloClassifier(checkpoint_path=path, device=devices, n_estimators=4).fit(X, labels(X))
    assert len(loads) == 1
    assert all(second._engine._model_replicas[device] is model for device, model in replicas.items())
    np.testing.assert_array_equal(second.predict_proba(query), first.predict_proba(query))
