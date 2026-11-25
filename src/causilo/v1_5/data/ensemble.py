"""Seeded, column-balanced X-sign views with balanced target transforms."""

from __future__ import annotations

import random
from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class EnsemblePlan:
    """Member-aligned signs and optional class-index permutations."""

    feature_signs: np.ndarray
    target_signs: np.ndarray | None
    class_permutations: np.ndarray | None
    feature_views: tuple[int, ...]
    trim_count: int = 0


def feature_view_order(n_estimators: int, count: int) -> tuple[int, ...]:
    """Balance the views, using both non-Base slots for Robust at E4 and E5."""
    other, extra_base = divmod(count, 2)
    base = other + extra_base
    robust = (other + 1) // 2
    if n_estimators in (4, 5):
        robust = other
    return (0,) * base + (1,) * robust + (2,) * (other - robust)


def regression_trim(count: int) -> int:
    """Drop one prediction from each end for every four fitted members."""
    return count // 4


def _negative_members(groups: list[list[int]], capacity: int, count: int, rng: random.Random) -> set[int]:
    """Split each equal-view group within the capacity of the remaining bits."""
    quota = count // 2
    if max(map(len, groups)) <= capacity:
        return set(rng.sample(range(1, count), quota))
    selected: set[int] = set()
    room = []
    candidates = []
    for index, group in enumerate(groups):
        eligible = [member for member in group if member != 0]
        lower = max(0, len(group) - capacity)
        selected.update(rng.sample(eligible, lower))
        room.append(min(len(eligible), capacity) - lower)
        candidates.extend((member, index) for member in eligible if member not in selected)
    rng.shuffle(candidates)
    for member, index in candidates:
        if len(selected) == quota:
            break
        if room[index]:
            selected.add(member)
            room[index] -= 1
    return selected


def make_ensemble_plan(
    *, n_features: int, n_classes: int | None, n_estimators: int, random_state: int | None
) -> EnsemblePlan:
    """Return unique X/target views with identity first and balanced columns.

    Each feature is negative in exactly floor(actual member count / 2) views.
    Target signs or seeded class-label shifts follow a balanced cycle. Refine
    equal-view groups one feature at a time, leaving enough suffix combinations
    to distinguish every member without rejection. Changing the requested count
    can regenerate X masks; fixed counts and seeds reproduce the same plan.
    """
    if isinstance(n_estimators, bool) or not isinstance(n_estimators, Integral) or n_estimators < 1:
        raise ValueError("n_estimators must be a positive integer.")
    if n_features < 1:
        raise ValueError("At least one retained feature is required.")
    if n_classes is not None and n_classes < 1:
        raise ValueError("n_classes must be positive or None for regression.")
    shift_count = 2 if n_classes is None else n_classes
    count = min(int(n_estimators), (1 << int(n_features)) * shift_count)
    views = feature_view_order(n_estimators, count)
    rng = random.Random(random_state)
    target_order = [0, *rng.sample(range(1, shift_count), shift_count - 1)]
    shifts = [target_order[member % shift_count] for member in range(count)]
    groups = [
        [member for member in range(target, count, shift_count)] for target in range(min(count, shift_count))
    ]
    features = np.ones((count, n_features), dtype=np.float32)
    for column in range(n_features):
        negative = _negative_members(groups, 1 << (n_features - column - 1), count, rng)
        features[list(negative), column] = -1
        if len(groups) < count:
            groups = [
                subgroup
                for group in groups
                for subgroup in (
                    [member for member in group if member not in negative],
                    [member for member in group if member in negative],
                )
                if subgroup
            ]
    if n_classes is None:
        trim_count = regression_trim(count)
        targets = np.array([1 - 2 * target for target in shifts], dtype=np.float32)
        return EnsemblePlan(features, targets, None, views, trim_count)
    classes = np.arange(n_classes) - np.asarray(shifts)[:, None]
    classes %= n_classes
    return EnsemblePlan(features, None, classes, views)
