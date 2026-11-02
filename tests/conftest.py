import pytest
import torch

from causilo.model import Model, ModelConfig


@pytest.fixture
def batch_runtime():
    """Keep small batch tests inexpensive without changing other test modules."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
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
