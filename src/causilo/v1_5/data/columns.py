"""Order retained context columns by content and group exact duplicates."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cmp_to_key, lru_cache
from itertools import groupby

import numpy as np

_PREFIX_ROWS = 8
_SUFFIX_BLOCK_ROWS = 1024


def _column_keys(values: np.ndarray) -> np.ndarray:
    """Canonicalize a temporary column block for byte comparisons."""
    keys = np.asarray(values, dtype="<f8", order="F")
    keys[np.isnan(keys)] = np.nan
    keys[keys == 0] = 0.0
    return keys


def _varies(column: np.ndarray) -> bool:
    first = column[0]
    return (~np.isnan(column)).any() if np.isnan(first) else (column != first).any()


def _ordered_groups(table: np.ndarray, indices: list[int]):
    pending = [(0, indices)]
    while pending:
        start, indices = pending.pop()
        if len(indices) == 1 or start == len(table):
            yield indices
            continue

        stop = min(len(table), start + (_SUFFIX_BLOCK_ROWS if start else _PREFIX_ROWS))
        keys = _column_keys(table[start:stop, indices])
        buckets = defaultdict(list)
        for index, column in zip(indices, keys.T, strict=True):
            buckets[column.tobytes()].append(index)
        # Continue only tied groups. Read suffix values in blocks so a large
        # tie does not require another full context grid.
        pending.extend((stop, buckets[key]) for key in sorted(buckets, reverse=True))


@dataclass(frozen=True)
class CanonicalColumns:
    order: np.ndarray
    groups: np.ndarray
    ties: tuple[tuple[int, int], ...]

    @classmethod
    def fit(cls, table: np.ndarray, *, effective_rows: int) -> CanonicalColumns:
        """Select varying columns and order them by the first differing value's bytes.

        A single effective observation retains all columns. Column order and
        duplicate groups are invariant to explicit row repetition.
        """
        if effective_rows == 1:
            retained = list(range(table.shape[1]))
        else:
            retained = [index for index, column in enumerate(table.T) if _varies(column)]
        return cls._from_groups(_ordered_groups(table, retained) if retained else ())

    @classmethod
    def fit_streamed(cls, columns, read_column) -> CanonicalColumns:
        """Find the exact full-context order without materializing the encoded table."""
        buckets = defaultdict(list)
        for index, column in columns:
            if _varies(column):
                prefix = _column_keys(column[:_PREFIX_ROWS].copy()).tobytes()
                buckets[prefix].append(index)

        @lru_cache(maxsize=2)
        def suffix(index):
            # Read full suffixes only for matching prefixes and bound the cache
            # in columns, even if every feature shares the same prefix.
            return _column_keys(read_column(index)[_PREFIX_ROWS:].copy()).tobytes()

        def compare(left, right):
            a, b = suffix(left), suffix(right)
            return (a > b) - (a < b)

        def groups():
            for prefix in sorted(buckets):
                indices = buckets[prefix]
                if len(indices) == 1:
                    yield indices
                else:
                    ordered = sorted(sorted(indices), key=cmp_to_key(compare))
                    for _, matching in groupby(ordered, key=suffix):
                        yield list(matching)

        return cls._from_groups(groups())

    @classmethod
    def _from_groups(cls, ordered_groups) -> CanonicalColumns:
        order, groups, ties = [], [], []
        for group, indices in enumerate(ordered_groups):
            start = len(order)
            order.extend(indices)
            groups.extend([group] * len(indices))
            if len(indices) > 1:
                ties.append((start, len(order)))

        if not order:
            raise ValueError("At least one retained feature is required.")
        return cls(np.asarray(order, dtype=np.intp), np.asarray(groups, dtype=np.intp), tuple(ties))

    @property
    def n_groups(self) -> int:
        return int(self.groups[-1]) + 1

    def transform(self, table: np.ndarray) -> np.ndarray:
        """Apply the fitted order and resolve context ties within each query row."""
        ordered = table[:, self.order]
        for start, stop in self.ties:
            block = ordered[:, start:stop]
            block[np.isnan(block)] = np.nan
            block[block == 0] = 0.0
            block.sort(axis=1)
        return ordered
