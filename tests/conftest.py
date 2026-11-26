import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from causilo.model_selection import RUNTIME15_FORMAT
from causilo.v1_0.model import Model, ModelConfig


@pytest.fixture(autouse=True)
def isolated_shared_models():
    """Load weights afresh in each test, since tests replace checkpoints and loaders."""
    from causilo.v1_5.engine import clear_shared_models

    clear_shared_models()
    yield
    clear_shared_models()


@pytest.fixture
def batch_runtime():
    """Keep small batch tests inexpensive without changing other test modules."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def make_batch_model(batch_runtime):
    """Build deterministic multi-layer models; API tests can request native output sizes."""
    def make(task, outputs=None):
        with torch.random.fork_rng():
            torch.manual_seed(34)
            model = Model(ModelConfig(
                task=task, width=16, expansion=2, group_size=3, frequencies=4,
                column_latents=5, column_heads=2, column_depths=(2, 2),
                row_heads=2, row_latents=2, row_depths=(3, 2),
                prediction_heads=2, prediction_depth=3,
                outputs=outputs if outputs is not None else (10 if task == "classification" else 9),
            ))
            for name, parameter in model.named_parameters():
                if "latents" in name or name.endswith(("frequencies", "missing")):
                    torch.nn.init.normal_(parameter, std=0.1)
        return model.eval()
    return make


def write_checkpoint(folder, record, tensors, *, task):
    folder.mkdir(exist_ok=True)
    config = json.dumps(record).encode()
    (folder / "config.json").write_bytes(config)
    weights = folder / "model.safetensors"
    save_file(tensors, weights, metadata={
        "architecture": record["architecture"],
        "format_version": str(record["format_version"]),
        "task": task,
        "config_sha256": hashlib.sha256(config).hexdigest(),
    })
    return weights


@pytest.fixture
def checkpoint_factory(tmp_path):
    def create(task, name="weights", seed=18):
        folder = tmp_path / name
        config = ModelConfig(
            task=task, width=8, expansion=2, group_size=2, frequencies=2,
            column_latents=3, column_heads=2, column_depths=(1, 1),
            row_heads=2, row_latents=2, row_depths=(1, 1),
            prediction_heads=2, prediction_depth=1, outputs=3 if task == "classification" else 9,
        )
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            model = Model(config)
            for parameter in model.parameters():
                torch.nn.init.normal_(parameter, std=0.1)
        return write_checkpoint(folder, config.record(), model.state_dict(), task=task)
    return create


@pytest.fixture
def checkpoint15(tmp_path):
    """Build a small local 1.5 checkpoint for runtime tests."""
    from causilo.v1_5.nn.model import Model as Model15
    from causilo.v1_5.nn.model_config import ModelConfig as ModelConfig15

    def make(task, *, weight_dtype=torch.float32, depth=2):
        path = tmp_path / task
        config = ModelConfig15(
            max_classes=10 if task == "classification" else 0,
            width=16, column_depth=depth, prediction_depth=depth,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            model = Model15(config)
            for parameter in model.parameters():
                torch.nn.init.normal_(parameter, std=0.15)
        record = {
            "architecture": RUNTIME15_FORMAT,
            "format_version": 1, "config_version": 1, "task": task,
            "model": config.to_dict(),
        }
        tensors = {name: value.to(weight_dtype) for name, value in model.state_dict().items()}
        write_checkpoint(path, record, tensors, task=task)
        return path
    return make


@pytest.fixture
def mixed_data():
    data = pd.DataFrame(
        np.random.default_rng(83).normal(size=(47, 4)), columns=["a", "b", "c", "d"],
    )
    data.loc[::4, "d"] = np.nan
    data["kind"] = pd.Categorical(np.resize(["red", "blue", "green", None], len(data)))
    return data.iloc[:35], data.iloc[35:]
