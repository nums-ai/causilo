"""Prepare Causilo 1.5 contexts and their execution runners."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..data.context_sampling import sample_context_rows, validate_context_limit
from ..data.dataset import PreparedDataset, prepare_inputs, take_rows
from ..data.encoding import fit_sampling_columns
from ..data.ensemble import regression_trim
from ..nn.model_config import NATIVE_CLASSES
from .direct import EnsembleRunner
from .parallel import ParallelEnsembleRunner, member_sections

if TYPE_CHECKING:
    from ..engine import Engine
    from .cache import ContextCache


@dataclass(frozen=True)
class FitState:
    """Prepared context, execution runner, and reusable caches."""

    dataset: PreparedDataset
    runner: EnsembleRunner | ParallelEnsembleRunner
    # Native caches follow ensemble order. ECOC stores one entry per code row.
    caches: tuple[ContextCache, ...] | None


def make_runner(engine: Engine, dataset: PreparedDataset) -> EnsembleRunner | ParallelEnsembleRunner:
    """Allocate ensemble members over the devices selected for this fit."""
    members = len(dataset.ensemble.feature_signs)
    lanes = {
        device: EnsembleRunner(
            model, device, dataset.frequencies, memory_budget_bytes=engine.memory_budgets.get(str(device)),
        )
        for device, model in engine.initialize_model(members).items()
    }
    if len(lanes) == 1:
        return lanes[engine.device]
    capacities = tuple(lane.member_capacity(*dataset.training.shape) for lane in lanes.values())
    sections = member_sections(members, tuple(lanes), capacities)
    return ParallelEnsembleRunner(tuple((lanes[device], section) for device, section in sections))


@dataclass(frozen=True)
class SampledFitState:
    """One prepared context and cache per independently sampled member."""

    members: tuple[FitState, ...]
    trim_count: int
    target_limits: tuple[float, float] | None

    @property
    def dataset(self) -> PreparedDataset:
        # All members have the same input schema and fitted class order.
        return self.members[0].dataset


ContextState = FitState | SampledFitState


def make_sampled_runners(engine: Engine, datasets):
    """Place member runners in order, sharing one checkpoint replica on each device."""
    replicas = engine.initialize_model(len(datasets))
    rows = max(len(dataset.training) for dataset in datasets)
    width = max(dataset.training.shape[1] for dataset in datasets)
    locks = {device: Lock() for device in replicas}
    lanes = {
        device: EnsembleRunner(
            model, device, memory_budget_bytes=engine.memory_budgets.get(str(device)),
            prediction_lock=locks[device],
        )
        for device, model in replicas.items()
    }
    capacities = tuple(runner.member_capacity(rows, width) for runner in lanes.values())
    for device, section in member_sections(len(datasets), tuple(lanes), capacities):
        runner = lanes[device]
        for dataset in datasets[section]:
            if dataset.frequencies is None:
                yield runner
            else:
                yield EnsembleRunner(
                    runner.model, device, dataset.frequencies,
                    memory_budget_bytes=runner.memory_budget_bytes, prediction_lock=locks[device],
                )


def prepare_state(
    engine: Engine, table, targets, *, n_estimators, use_kv_cache, retain_preprocessing,
    random_state, sample_count, max_context,
) -> ContextState:
    """Fit preprocessing before loading the selected checkpoint and building caches."""
    limit = validate_context_limit(max_context)
    classification = engine.task == "classification"
    table, labels, frequencies = prepare_inputs(table, targets, sample_count, classification)
    options = {
        "task": engine.task,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "retain_preprocessing": retain_preprocessing,
    }
    if limit is None or len(labels) <= limit:
        dataset = PreparedDataset.from_inputs(table, labels, frequencies, **options)
        runner = make_runner(engine, dataset)
        return FitState(dataset, runner, runner.build_cache(dataset) if use_kv_cache else None)

    fit_seed = random_state
    if fit_seed is None:
        fit_seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
    encoded_labels = None
    member_count = n_estimators
    if classification:
        classes, encoded_labels = np.unique(labels, return_inverse=True)
        if limit < len(classes):
            raise ValueError("max_context must be at least the number of observed classes.")
        if len(classes) > NATIVE_CLASSES:
            member_count = 1
    else:
        labels = labels.astype(np.float64, copy=False)
        if not np.isfinite(labels).all():
            raise ValueError("Targets must be finite")
    if not isinstance(table, pd.DataFrame):
        table = np.asarray(table)
    feature_layout = fit_sampling_columns(table, frequencies)
    trim_count = 0 if classification else regression_trim(member_count)
    lower, upper = np.inf, -np.inf
    datasets = []
    for index in range(member_count):
        rows = sample_context_rows(len(labels), limit, seed=fit_seed, member_index=index, labels=encoded_labels)
        sampled_labels = labels[rows]
        if trim_count:
            lower = min(lower, float(sampled_labels.min()))
            upper = max(upper, float(sampled_labels.max()))
        counts = None if frequencies is None else frequencies[rows]
        if counts is not None and (counts == 1).all():
            counts = None
        datasets.append(PreparedDataset.from_inputs(
            take_rows(table, rows), sampled_labels, counts,
            **(options | {"random_state": fit_seed}),
            member_index=index if member_count > 1 else None, feature_layout=feature_layout,
        ))

    if member_count == 1:
        dataset = datasets[0]
        runner = make_runner(engine, dataset)
        return FitState(dataset, runner, runner.build_cache(dataset) if use_kv_cache else None)
    members = tuple(
        FitState(dataset, runner, runner.build_cache(dataset) if use_kv_cache else None)
        for dataset, runner in zip(datasets, make_sampled_runners(engine, datasets), strict=True)
    )
    return SampledFitState(members, trim_count, (lower, upper) if trim_count else None)
