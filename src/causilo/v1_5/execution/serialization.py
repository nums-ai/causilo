"""Represent and restore saved Causilo 1.5 contexts and K/V caches."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING

import torch

from ..data.dataset import PreparedDataset
from .fit import ContextState, FitState, SampledFitState, make_runner, make_sampled_runners

if TYPE_CHECKING:
    from ..engine import Engine
    from .cache import ContextCache


@dataclass(frozen=True)
class SavedContext:
    """Fitted data and optional CPU caches for one table."""

    dataset: PreparedDataset
    caches: tuple[ContextCache, ...] | None


@dataclass(frozen=True)
class SavedSampledContext:
    """The prepared contexts in their original member order."""

    members: tuple[SavedContext, ...]
    trim_count: int
    target_limits: tuple[float, float] | None

    @property
    def datasets(self) -> tuple[PreparedDataset, ...]:
        return tuple(member.dataset for member in self.members)


def _copy_cache_to_cpu(value):
    """Copy cache tensors to CPU without sharing storage with the live cache."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{field.name: _copy_cache_to_cpu(getattr(value, field.name)) for field in fields(value)}
        )
    if isinstance(value, tuple):
        return tuple(_copy_cache_to_cpu(item) for item in value)
    if isinstance(value, dict):
        return {key: _copy_cache_to_cpu(item) for key, item in value.items()}
    return value


def export_context(state: ContextState) -> SavedContext | SavedSampledContext:
    """Save prepared data and CPU caches for the next execution runner."""
    if isinstance(state, SampledFitState):
        return SavedSampledContext(
            tuple(export_context(member) for member in state.members), state.trim_count, state.target_limits,
        )
    return SavedContext(dataset=state.dataset, caches=_copy_cache_to_cpu(state.caches))


def restore_context(context: SavedContext | SavedSampledContext, engine: Engine, *, reuse_cache: bool) -> ContextState:
    """Restore a 1.5 runner and its device-specific cache."""
    def restore_member(saved, runner):
        dataset, caches = saved.dataset, saved.caches
        if dataset.num_classes > engine.model.config.max_classes:
            raise ValueError("Refit the saved many-class context with the current Causilo installation")
        if caches is not None:
            runner.validate_caches(dataset, caches)
            caches = runner.place_caches(caches) if reuse_cache else runner.build_cache(dataset)
        return FitState(dataset, runner, caches)

    if isinstance(context, SavedSampledContext):
        members = tuple(
            restore_member(saved, runner)
            for saved, runner in zip(context.members, make_sampled_runners(engine, context.datasets), strict=True)
        )
        return SampledFitState(members, context.trim_count, context.target_limits)
    return restore_member(context, make_runner(engine, context.dataset))
