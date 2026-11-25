"""Register the neural stages used by Causilo 1.5."""

from torch import nn

from .column import ColumnBlock
from .embeddings import FeatureEmbedding
from .model_config import ModelConfig
from .prediction import PredictionBlock
from .row import RowBlock


class Model(nn.Module):
    """Register paired column and row layers, followed by query prediction."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_embedding = FeatureEmbedding(config)
        self.columns = ColumnBlock(config)
        self.row = RowBlock(config)
        self.prediction = PredictionBlock(config)
