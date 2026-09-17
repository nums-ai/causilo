"""Balanced active/rest codes and streaming class-probability reconstruction."""

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np


def _row_budget(classes: int, symbols: int) -> int:
    """Apply the fixed redundancy-four budget without dropping class coverage."""
    digits, representable = 0, 1
    while representable < classes:
        digits += 1
        representable *= symbols
    coverage = (classes + symbols - 2) // (symbols - 1)
    minimum = max(digits, coverage)
    return min(4 * minimum, max(minimum, 4 * digits))


def _balanced_codes(classes: int, symbols: int, rows: int, rng: np.random.RandomState) -> np.ndarray:
    """Schedule the least-used classes first, with seeded random tie breaking."""
    active_width = symbols - 1
    codes = np.full((rows, classes), active_width, dtype=np.int64)
    uses = np.zeros(classes, dtype=np.int64)
    for output in codes:
        # Usage is the primary key; randomness only orders equal-usage classes.
        ordered = np.lexsort((rng.random_sample(classes), uses))
        assigned = ordered[:active_width]
        output[assigned] = rng.permutation(active_width)
        uses[assigned] += 1
    return codes


def _separation(codes: np.ndarray, rest: int) -> tuple[int, float]:
    """Score pairwise Hamming distance using active-set co-occurrence counts.

    Active symbols in one row are distinct. Two class codes therefore differ
    exactly when at least one class is active, regardless of symbol permutation.
    """
    active = (codes != rest).astype(np.int64)
    appearances = active.sum(axis=0)
    both_active = active.T @ active
    distance = appearances[:, None] + appearances[None, :] - both_active
    pairs = distance[np.triu_indices(codes.shape[1], k=1)]
    return int(pairs.min()), float(pairs.mean())


@dataclass(frozen=True)
class ECOCCodec:
    """Immutable, seeded output codes with the fixed Causilo many-class recipe."""

    class_count: int
    symbol_count: int
    random_state: int
    codebook: np.ndarray = field(init=False, repr=False)
    rest_symbol: int = field(init=False)

    def __post_init__(self) -> None:
        parameters = (self.class_count, self.symbol_count, self.random_state)
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            for value in parameters
        ):
            raise ValueError("Class count, symbol count and random_state must be integers")
        if self.symbol_count < 2:
            raise ValueError("symbol_count must be at least two")
        if self.class_count <= self.symbol_count:
            raise ValueError("ECOC requires more classes than output symbols")
        if not 0 <= self.random_state < 2**32:
            raise ValueError("random_state must fit in an unsigned 32-bit integer")

        classes, symbols = int(self.class_count), int(self.symbol_count)
        object.__setattr__(self, "rest_symbol", symbols - 1)
        seeds = np.random.RandomState(int(self.random_state))
        attempts = 50 if classes <= 200 else 1
        chosen, score = None, None
        for _ in range(attempts):
            rng = np.random.RandomState(seeds.randint(0, 2**31 - 1))
            codes = _balanced_codes(classes, symbols, _row_budget(classes, symbols), rng)
            # Above 200 classes there is no competing candidate to score.
            quality = _separation(codes, symbols - 1) if attempts > 1 else (0, 0.0)
            if score is None or quality > score:
                chosen, score = codes, quality
        chosen.setflags(write=False)
        object.__setattr__(self, "codebook", chosen)

    def __setstate__(self, state: dict) -> None:
        """Unpickling restores the fitted schedule rather than making a new one."""
        for name, value in state.items():
            object.__setattr__(self, name, value)
        self.codebook.setflags(write=False)

    def encode(self, labels: np.ndarray) -> np.ndarray:
        indices = np.asarray(labels)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("ECOC labels must be a one-dimensional integer array")
        if np.any((indices < 0) | (indices >= self.class_count)):
            raise ValueError("ECOC label is outside the class range")
        return np.ascontiguousarray(np.take(self.codebook, indices, axis=1))

    def decode(self, row_probabilities: np.ndarray) -> np.ndarray:
        """Decode a dense (code rows, query rows, output symbols) array."""
        rows = np.asarray(row_probabilities)
        if rows.ndim != 3 or rows.shape[0] != len(self.codebook) or rows.shape[2] != self.symbol_count:
            raise ValueError("ECOC probabilities must have shape (code rows, query rows, output symbols)")
        return self.decode_rows(rows)

    def decode_rows(self, row_probabilities: Iterable[np.ndarray]) -> np.ndarray:
        """Accumulate only active log probabilities, consuming one code row at a time.

        Auxiliary storage is O(query rows * classes), not O(code rows * query
        rows * classes). Rest probabilities do not contribute to class scores.
        """
        scores = None
        appearances = np.zeros(self.class_count, dtype=np.int64)
        consumed = 0
        for index, values in enumerate(row_probabilities):
            if index >= len(self.codebook):
                raise ValueError("ECOC probabilities have too many code rows")
            probabilities = np.asarray(values, dtype=np.float64)
            if probabilities.ndim != 2 or probabilities.shape[1] != self.symbol_count:
                raise ValueError("Each ECOC probability row must have shape (query rows, output symbols)")
            if scores is None:
                scores = np.zeros((len(probabilities), self.class_count), dtype=np.float64)
            elif len(probabilities) != len(scores):
                raise ValueError("ECOC probability rows must have the same query count")
            if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
                raise ValueError("ECOC probabilities must be finite and nonnegative")
            # Scaling before summing also accepts unnormalized finite weights
            # whose direct sum would overflow. Normal probabilities are unchanged.
            largest = probabilities.max(axis=1, keepdims=True)
            if np.any(largest <= 0):
                raise ValueError("ECOC probability rows must have positive sums")
            scaled = probabilities / largest
            normalized = scaled / scaled.sum(axis=1, keepdims=True)
            classes = np.flatnonzero(self.codebook[index] != self.rest_symbol)
            symbols = self.codebook[index, classes]
            scores[:, classes] += np.log(np.maximum(normalized[:, symbols], 1e-12))
            appearances[classes] += 1
            consumed += 1
        if consumed != len(self.codebook):
            raise ValueError("ECOC probabilities have too few code rows")
        scores /= appearances
        scores -= scores.max(axis=1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= scores.sum(axis=1, keepdims=True)
        return scores
