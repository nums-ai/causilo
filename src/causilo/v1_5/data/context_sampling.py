"""Deterministic row budgets for individual ensemble members."""

from __future__ import annotations

from numbers import Integral
from operator import index

import numpy as np


def validate_context_limit(limit: int | None) -> int | None:
    """Accept an optional positive integer row budget."""
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 1:
        raise ValueError("max_context must be a positive integer or None.")
    return int(limit)


def sample_context_rows(
    n_rows: int,
    max_context: int | None,
    *,
    seed: int | None,
    member_index: int,
    labels: np.ndarray | None = None,
) -> np.ndarray | None:
    """Draw original row positions for one member, preserving their input order.

    A fixed seed and member index determine an independent uniform draw without
    replacement. Integer-encoded labels let the sampler add one row per absent
    class, replacing the fewest sampled rows needed. The budget must fit all
    observed classes. Pass only eligible rows and their aligned labels.

    Return None when the member can keep the entire context.
    """
    limit = validate_context_limit(max_context)
    if limit is None or limit >= n_rows:
        return None

    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != (n_rows,) or labels.dtype.kind not in "iu":
            raise ValueError("labels must contain one encoded integer per eligible context row.")
        classes = np.unique(labels)
        if limit < len(classes):
            raise ValueError("max_context must be at least the number of observed classes.")

    entropy = None if seed is None else abs(index(seed))
    rng = np.random.default_rng(np.random.SeedSequence(entropy, spawn_key=(member_index,)))
    rows = rng.choice(n_rows, size=limit, replace=False).astype(np.intp, copy=False)
    if labels is not None:
        present, keep = np.unique(labels[rows], return_index=True)
        missing = np.setdiff1d(classes, present, assume_unique=True)
        if missing.size:
            # Keep one randomly ordered sample per present class and replace only
            # duplicate-class slots. A shuffled first match samples each missing class uniformly.
            replaceable = np.ones(limit, dtype=bool)
            replaceable[keep] = False
            slots = rng.choice(np.flatnonzero(replaceable), size=len(missing), replace=False)
            candidates = np.flatnonzero(np.isin(labels, missing))
            rng.shuffle(candidates)
            _, picks = np.unique(labels[candidates], return_index=True)
            rows[slots] = candidates[picks]
    rows.sort()
    return rows
