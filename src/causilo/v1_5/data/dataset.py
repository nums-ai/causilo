"""Fitted table transforms and ensemble views for Causilo 1.5."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import column_or_1d

from ...v1_0.ecoc import ECOCCodec
from ..nn.model_config import NATIVE_CLASSES
from .frequency import validate_frequencies
from .schema import table_shape

if TYPE_CHECKING:
    from .columns import CanonicalColumns
    from .encoding import FeatureEncoder
    from .ensemble import EnsemblePlan
    from .feature_views import FeatureViews
    from .target import TargetPreprocessor


@dataclass(frozen=True)
class TableViews:
    """Keep shared unsigned views and attach context without copying its rows."""

    values: tuple[np.ndarray, ...]
    context: tuple[np.ndarray, ...] = ()

    @property
    def shape(self) -> tuple[int, int]:
        rows, width = self.values[0].shape
        return rows + (len(self.context[0]) if self.context else 0), width

    def __len__(self) -> int:
        return self.shape[0]

    def select_queries(self, section: slice) -> TableViews:
        return TableViews(tuple(view[section] for view in self.values), self.context)


FeatureTable = np.ndarray | TableViews


def _view_table(views: tuple[np.ndarray, ...]) -> FeatureTable:
    return views[0] if len(views) == 1 else TableViews(views)


def join_context(train: FeatureTable, query: FeatureTable) -> FeatureTable:
    if isinstance(query, TableViews):
        return TableViews(query.values, train.values)
    return np.concatenate((train, query), axis=0)


def select_query_rows(table: FeatureTable, section: slice) -> FeatureTable:
    return table.select_queries(section) if isinstance(table, TableViews) else table[section]


def take_rows(table, indices):
    return table.iloc[indices] if isinstance(table, pd.DataFrame) else np.asarray(table)[indices]


def prepare_inputs(table, targets, sample_count, classification):
    """Validate target rows and remove zero counts before context selection."""
    if targets is None:
        kind = "classifier" if classification else "regressor"
        raise ValueError(f"This {kind} requires y to be passed, but the target y is None.")
    rows, _ = table_shape(table)
    labels = column_or_1d(targets, warn=True)
    if len(labels) != rows:
        raise ValueError("Targets must contain one value per training row")
    frequencies = None
    if sample_count is not None:
        frequencies = validate_frequencies(sample_count, rows)
        keep = frequencies > 0
        if not keep.all():
            table = take_rows(table, keep)
            labels, frequencies = labels[keep], frequencies[keep]
        if np.all(frequencies == 1):
            frequencies = None
    if pd.isna(labels).any():
        raise ValueError("Targets cannot contain missing values")
    if classification:
        check_classification_targets(labels)
    return table, labels, frequencies


@dataclass(frozen=True)
class PreparedDataset:
    """Share unsigned context preprocessing and apply member signs on demand."""

    encoder: FeatureEncoder
    columns: CanonicalColumns
    normalizer: FeatureViews
    target_preprocessor: TargetPreprocessor
    training: FeatureTable
    targets: np.ndarray
    ensemble: EnsemblePlan
    retain_preprocessing: bool
    frequencies: np.ndarray | None
    codec: ECOCCodec | None = None
    target_limits: tuple[float, float] | None = None

    @classmethod
    def prepare(
        cls, table, targets, *, task, n_estimators, random_state, retain_preprocessing=True, sample_count=None,
    ) -> PreparedDataset:
        table, labels, frequencies = prepare_inputs(table, targets, sample_count, task == "classification")
        return cls.from_inputs(
            table, labels, frequencies, task=task, n_estimators=n_estimators, random_state=random_state,
            retain_preprocessing=retain_preprocessing,
        )

    @classmethod
    def from_inputs(
        cls, table, labels, frequencies, *, task, n_estimators, random_state, retain_preprocessing,
        member_index=None, feature_layout=None,
    ) -> PreparedDataset:
        from .columns import CanonicalColumns
        from .encoding import FeatureEncoder
        from .ensemble import feature_view_order, make_ensemble_plan
        from .feature_views import FeatureViews
        from .target import TargetPreprocessor

        classification = task == "classification"
        target_preprocessor = TargetPreprocessor(classification=classification)
        encoded_targets = target_preprocessor.fit_transform(labels, sample_count=frequencies)
        if not np.isfinite(encoded_targets).all():
            raise ValueError("Encoded targets must be finite")
        layout, columns = feature_layout if feature_layout is not None else (None, None)
        encoder, encoded = FeatureEncoder.fit(table, sample_count=frequencies, layout=layout)
        if columns is None:
            effective_rows = len(encoded) if frequencies is None else int(frequencies.sum())
            columns = CanonicalColumns.fit(encoded, effective_rows=effective_rows)
        training = columns.transform(encoded)
        del encoded
        classes = len(target_preprocessor.encoder_.classes_) if classification else None
        codec = None
        if classes is not None and classes > NATIVE_CLASSES:
            seed = random_state
            if seed is None or seed >= 2**32:
                seed = int(np.random.SeedSequence(seed).generate_state(1)[0])
            codec = ECOCCodec(classes, NATIVE_CLASSES, seed)
            classes = NATIVE_CLASSES
            n_estimators = 1
        ensemble = make_ensemble_plan(
            n_features=columns.n_groups,
            n_classes=classes,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        selected_view = None
        if member_index is not None:
            # Distinct contexts can reuse a sign/permutation when the table has few distinct features.
            slot = member_index % len(ensemble.feature_signs)
            section = slice(slot, slot + 1)
            views = feature_view_order(n_estimators, n_estimators)
            selected_view = views[member_index]
            ensemble = replace(
                ensemble, feature_signs=ensemble.feature_signs[section],
                class_permutations=None if not classification else ensemble.class_permutations[section],
                target_signs=None if classification else ensemble.target_signs[section],
                feature_views=(0,), trim_count=0,
            )
        ensemble = replace(ensemble, feature_signs=ensemble.feature_signs[:, columns.groups])
        normalizer, views = FeatureViews._fit_transform(
            training, frequencies, view_count=max(ensemble.feature_views) + 1,
            retain=retain_preprocessing, selected_view=selected_view,
        )
        if views is not None:
            training = _view_table(views)

        target_limits = None
        if not classification and ensemble.trim_count:
            numeric_targets = labels.astype(np.float64, copy=False)
            target_limits = (float(numeric_targets.min()), float(numeric_targets.max()))
        return cls(
            encoder, columns, normalizer, target_preprocessor, training,
            encoded_targets, ensemble, retain_preprocessing, frequencies, codec, target_limits,
        )

    @property
    def num_classes(self) -> int:
        """Return the class width passed to the native head."""
        orders = self.ensemble.class_permutations
        return orders.shape[1] if orders is not None else 0

    @property
    def code_rows(self) -> int:
        return 0 if self.codec is None else len(self.codec.codebook)

    def fitted_attributes(self):
        attributes = {"n_features_in_": self.encoder.width}
        names = self.encoder.feature_names
        if names is not None:
            attributes["feature_names_in_"] = names
        if self.target_preprocessor.classification:
            attributes["classes_"] = self.target_preprocessor.encoder_.classes_
        return attributes

    @property
    def view_count(self) -> int:
        return self.normalizer.view_count

    def _normalize(self, values: np.ndarray) -> FeatureTable:
        return _view_table(self.normalizer.transform(values))

    def training_table(self) -> FeatureTable:
        return self.training if self.retain_preprocessing else self._normalize(self.training)

    def query_table(self, table, section: slice, *, layout: tuple[int, np.ndarray | None] | None = None) -> FeatureTable:
        query = table.iloc[section] if hasattr(table, "iloc") else table[section]
        encoded = self.encoder.transform(query, layout=layout)
        return self._normalize(self.columns.transform(encoded))

    def targets_for(self, members: slice, *, code_row: int | None = None) -> np.ndarray:
        orders = self.ensemble.class_permutations
        if code_row is not None:
            # Each ECOC code runs one member and maps labels directly to its native symbols.
            orders = self.codec.codebook[code_row, None, :]
        if orders is not None:
            return orders[members, self.targets]
        return self.targets[None] * self.ensemble.target_signs[members, None]

    def signed_views(self, table: FeatureTable, members: slice) -> np.ndarray:
        signs = self.ensemble.feature_signs[members]
        if isinstance(table, TableViews):
            views = self.ensemble.feature_views[members]
            if not table.context and len(signs) == 1 and (signs[0] == 1).all():
                return table.values[views[0]][None]
            context_rows = len(table.context[0]) if table.context else 0
            signed = np.empty((len(signs), *table.shape), dtype=np.float32)
            for destination, index, sign in zip(signed, views, signs, strict=True):
                if context_rows:
                    np.multiply(table.context[index], sign, out=destination[:context_rows])
                np.multiply(table.values[index], sign, out=destination[context_rows:])
            return signed
        if len(signs) == 1 and table.dtype == signs.dtype and (signs[0] == 1).all():
            return table[None]
        return table[None] * signs[:, None]
