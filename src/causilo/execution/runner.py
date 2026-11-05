"""Wire model blocks together while keeping batching and precision outside the network.

Axis names: B = ensemble batch, R = table rows, T = training rows,
Q = query rows, G = feature groups, D = model width, L = row latents.
Direct execution puts the T training rows before the Q query rows.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from ..model import Model
from ..nn.cache import BlockCache, FrequencyWeights
from .chunks import CacheRunner, ChunkRunner
from .memory import Stage, Workload
from .precision import stage_autocast


@dataclass(frozen=True)
class ModelCache:
    """Reusable context for one member, stored in network execution order.

    Each column layer holds K/V shaped (1, G, heads, column_latents, head_width).
    Each prediction layer holds (1, heads, T, prediction_head_width). No query
    rows contribute to either cache.
    """

    columns: tuple[BlockCache, BlockCache]
    prediction: BlockCache


def inject_targets(rows: Tensor, targets: Tensor) -> Tensor:
    """Add embedded labels to the training prefix without modifying query rows.

    ``rows`` is (B, R, G, D) or (B, R, L*D); ``targets`` is already embedded as
    (B, T, D) or (B, T, L*D). Feature-level labels broadcast over G.
    """
    count = targets.shape[1]
    if rows.ndim == 4:
        targets = targets.unsqueeze(-2)
    return torch.cat((rows[:, :count] + targets, rows[:, count:]), dim=1)


class ModelRunner:
    """Run the same column/row/prediction stages with direct or cached context."""

    def __init__(self, model: Model) -> None:
        self.model = model
        self.chunks = ChunkRunner()
        self.cache_runner = CacheRunner()

    def _column(
        self, component, features: Tensor, count: int, training_lengths: Tensor | None = None,
        weights: FrequencyWeights | None = None,
    ) -> Tensor:
        """Attend over R for each feature group; preserve (B, R, G, D)."""
        if self.model.config.task == "regression":
            features = features.float()
        arranged = features.transpose(1, 2)
        # (B, G, R, D): B and G can be partitioned independently, R cannot.
        workload = Workload(Stage.COLUMN, arranged.shape[-2], arranged.shape[-1], count)
        options = {} if weights is None else {"weights": weights}
        with stage_autocast(self.model.config.task, Stage.COLUMN, features.device):
            if training_lengths is None:
                result = self.chunks.run(
                    lambda part: component(part, context_rows=count, **options), arranged, workload
                )
            else:
                lengths = training_lengths[:, None].expand(arranged.shape[:2]).reshape(-1)
                result = self.chunks.run_indexed(
                    lambda part, section: component(
                        part, context_rows=count, key_lengths=lengths[section], **options
                    ),
                    arranged, workload,
                )
        return result.transpose(1, 2)

    def _row(
        self, component, features: Tensor, stage: Stage, group_lengths: Tensor | None = None
    ) -> Tensor:
        """Mix feature groups per row, returning (B, R, G, D) or pooled (B, R, L*D)."""
        output = (
            self.model.config.row_latents * self.model.config.width
            if stage == Stage.POOL
            else features.shape[-2] * features.shape[-1]
        )
        workload = Workload(stage, features.shape[-2], features.shape[-1], output_elements=output)
        with stage_autocast(self.model.config.task, stage, features.device):
            if group_lengths is None:
                return self.chunks.run(component, features, workload)
            lengths = group_lengths[:, None].expand(features.shape[:2]).reshape(-1)
            return self.chunks.run_indexed(
                lambda part, section: component(part, group_lengths=lengths[section]), features, workload
            )

    def _embed(self, table: Tensor, targets: Tensor | None = None) -> Tensor:
        """Map (B, R, features) to (B, R, G, D), injecting known targets only."""
        with stage_autocast(self.model.config.task, Stage.COLUMN, table.device):
            features = self.model.feature_embedding(table)
            if targets is not None:
                features = inject_targets(features, self.model.feature_target(targets))
            return features

    def predict(
        self, table: Tensor, targets: Tensor, *, weights: FrequencyWeights | None = None,
        training_lengths: Tensor | None = None,
        feature_counts: Tensor | None = None,
    ) -> Tensor:
        """Return raw (B, Q, outputs) scores from table (B, T+Q, features).

        Targets have shape (B, T). Only the training prefix supplies attention
        keys/values; queries cannot influence training context or one another.
        Optional lengths exclude padding in independently sized tables. Queries
        start at T even when one item's training prefix is shorter. Feature
        padding must be numeric zero, preserving partial final groups.
        """
        model, count = self.model, targets.shape[1]
        options = {} if weights is None else {"weights": weights}
        groups = self._group_lengths(feature_counts)
        features = self._embed(table, targets)
        features = self._column(model.columns[0], features, count, training_lengths, **options)
        features = self._row(model.row, features, Stage.ROW, groups)
        features = self._column(model.columns[1], features, count, training_lengths, **options)
        rows = self._row(model.pool, features, Stage.POOL, groups)
        del features
        with stage_autocast(model.config.task, Stage.PREDICTION, rows.device):
            rows = inject_targets(rows, model.row_target(targets))
            workload = Workload(
                Stage.PREDICTION,
                rows.shape[-2],
                rows.shape[-1],
                count,
                output_elements=(rows.shape[-2] - count) * rows.shape[-1],
            )
            if training_lengths is None:
                predicted = self.chunks.run(
                    lambda part: model.prediction(part, context_rows=count, **options), rows, workload
                )
            else:
                predicted = self.chunks.run_indexed(
                    lambda part, section: model.prediction(
                        part, context_rows=count, key_lengths=training_lengths[section], **options
                    ), rows, workload,
                )
        return model.head(predicted)

    def _group_lengths(self, feature_counts: Tensor | None) -> Tensor | None:
        if feature_counts is None:
            return None
        size = self.model.config.group_size
        return (feature_counts + size - 1) // size

    def build_cache(
        self, table: Tensor, targets: Tensor, *, weights: FrequencyWeights | None = None
    ) -> ModelCache:
        """Process training rows once and retain each layer's reusable K/V tensors.

        Inputs are (B, T, features) and (B, T); the engine builds one member at
        a time (B=1). Intermediate activations are released after their last use.
        """
        model = self.model
        options = {} if weights is None else {"weights": weights}
        features = self._embed(table, targets)
        with stage_autocast(model.config.task, Stage.COLUMN, table.device):
            first = self.cache_runner.build_column(model.columns[0], features.transpose(1, 2), **options)
        first_state = first.cache
        features = self._row(model.row, first.rows.transpose(1, 2), Stage.ROW)
        del first
        with stage_autocast(model.config.task, Stage.COLUMN, table.device):
            context_input = features.float() if model.config.task == "regression" else features
            second = self.cache_runner.build_column(model.columns[1], context_input.transpose(1, 2), **options)
        second_state = second.cache
        rows = self._row(model.pool, second.rows.transpose(1, 2), Stage.POOL)
        del second, features, context_input
        with stage_autocast(model.config.task, Stage.PREDICTION, table.device):
            rows = inject_targets(rows, model.row_target(targets))
            return ModelCache((first_state, second_state), model.prediction.build_context(rows, **options))

    def predict_cached(
        self, table: Tensor, state: ModelCache, *, weights: FrequencyWeights | None = None,
        training_lengths: Tensor | None = None,
        feature_counts: Tensor | None = None,
    ) -> Tensor:
        """Return (B, Q, outputs) using aligned caches and optional per-item lengths.

        Padded column cache groups are excluded from row attention. Padded
        prediction keys are excluded using training_lengths at every layer.
        """
        model = self.model
        options = {} if weights is None else {"weights": weights}
        groups = self._group_lengths(feature_counts)
        features = self._embed(table)
        with stage_autocast(model.config.task, Stage.COLUMN, table.device):
            features = self.cache_runner.query(
                model.columns[0], features.transpose(1, 2), state.columns[0], Stage.COLUMN
            ).transpose(1, 2)
        features = self._row(model.row, features, Stage.ROW, groups)
        with stage_autocast(model.config.task, Stage.COLUMN, table.device):
            context_input = features.float() if model.config.task == "regression" else features
            features = self.cache_runner.query(
                model.columns[1], context_input.transpose(1, 2), state.columns[1], Stage.COLUMN
            ).transpose(1, 2)
        rows = self._row(model.pool, features, Stage.POOL, groups)
        del features, context_input
        with stage_autocast(model.config.task, Stage.PREDICTION, table.device):
            prediction = self.cache_runner.query(
                model.prediction, rows, state.prediction, Stage.PREDICTION, key_lengths=training_lengths, **options
            )
        return model.head(prediction)
