# Inference details

## Regression outputs

For `CausiloRegressor`, choose the prediction output with `output_type`:

| `output_type` | Output | Shape |
| --- | --- | --- |
| `"mean"` (default) | Mean prediction | `(n_samples,)` |
| `"median"` | Ensemble's 0.5 quantile | `(n_samples,)` |
| `"quantiles"` | Quantiles at the requested probabilities | `(n_samples, n_quantiles)` |
| `"raw"` | All 999 native quantiles | `(n_samples, 999)` |

```python
mean = regressor.predict(X_test)
median = regressor.predict(X_test, output_type="median")
quantiles = regressor.predict(
    X_test, output_type="quantiles", quantiles=[0.05, 0.5, 0.95]
)
```

Quantile prediction requires an explicit list of probabilities and preserves
request order and duplicates. A single probability still returns a two-dimensional array. Probabilities must
be finite and strictly between 0 and 1. Interpolation is linear between native
levels 0.001 through 0.999; outside them, exponential tails are estimated from
the two outermost native quantiles on each side. More extreme probabilities can produce large values.
Tail extrapolation does not affect the default mean prediction.

For advanced use, `predict(X, output_type="raw")` returns all 999 native quantiles
at probabilities `np.arange(1, 1000) / 1000`. These are sorted per ensemble member,
restored to target units and averaged across members; they are not unprocessed
model-head tensors. The `quantiles` argument is only accepted with
`output_type="quantiles"`.

## Precision and reproducibility

CUDA uses FP16 mixed precision, with regression column stages and both output heads in FP32. CPU and MPS execution use FP32. Regression target scaling and output restoration use float64 on the CPU.

The same inputs and seed reproduce the fitted feature and class permutations without changing global RNG state. Floating-point results can vary with execution device and batch shape; bitwise determinism is not enforced.

## Large inputs without a fitted K/V cache

Causilo first reduces concurrent ensemble members while keeping all query rows.
If one member still exceeds the memory budget, it chooses query batching or
column-summary recomputation based on estimated computation cost. All training
rows and ensemble members are preserved.

Recomputation processes feature groups and rows in tiles, keeping compact
summaries and row states on the execution device instead of a full hidden
feature grid. Prediction uses the full training context and retains temporary
K/V for one layer at a time.

Tile sizes adapt to available memory, and OOM retries shrink failed tiles.
Retained states and a layer's full training K/V must still fit on the device;
intermediate CPU offloading is not supported.

## Restoring a fitted estimator

With `device="auto"`, restoration selects an available device again. An explicitly configured device must be available. Saved state includes fitted preprocessing and optional attention caches, while pretrained weights are loaded from the pinned checkpoint. Use matching Causilo and dependency versions, including the Python major/minor version.

## Aggregated observations and repetition counts

Both estimators accept optional `sample_weight` in `fit`. Here a weight is an
**integer repetition count**: a row with count 30 represents 30 identical
observations with the same features and target. Counts are not loss weights or
arbitrary importance scores. Omitting them keeps the usual `fit(X, y)` behavior.

For binomial data, represent each group by a success row and a failure row:

```python
import numpy as np
from causilo import CausiloClassifier

# X_group contains one feature row per group; successes and trials are counts.
# This indexing works for a NumPy feature table.
rows = np.repeat(np.arange(len(X_group)), 2)
X = X_group[rows]
y = np.tile([1, 0], len(X_group))
counts = np.column_stack([successes, trials - successes]).ravel()

classifier = CausiloClassifier().fit(X, y, sample_weight=counts)
p_success = classifier.predict_proba(X_test)[:, list(classifier.classes_).index(1)]
```

For a DataFrame, use `X_group.iloc[rows]` to preserve column names and types.
This creates at most two rows per group, regardless of its trial count. Different
observations can only be combined when their features **and target** are identical;
the estimator does not automatically group rows. Success proportions alone do
not recover the individual observations when features vary within a group.

Counts must be finite nonnegative integers, one per input row, with a positive
sum no larger than `2**53`. Integer-valued floats are accepted; fractional values
are rejected. Zero-count rows are removed before fitting feature mappings,
classes, or preprocessing. All-one counts use the ordinary unweighted path.

Repetition counts are applied to preprocessing, context attention, and its
sample-count scaling. Regression also uses them when scaling targets. Many-class
classification, retained K/V caches, and memory-bounded recomputation use the
same counts. Predictions agree with explicit row repetition within floating-point
precision; execution shape and device can change rounding. Savings depend on
how many repeated observations the compact rows represent.

For rank normalization, missing-value means are computed from distinct observed
values and their counts in both paths. This prevents tiny summation differences
from changing tied ranks. Newly fitted models, including unweighted fits, can
therefore produce different predictions in affected missing-value edge cases.
Previously saved models retain their fitted filling values and ranks on restore.

Counts are retained in saved fitted estimators. Weighted fits use a new saved-state
schema so versions without frequency support reject them instead of silently
ignoring the counts. Existing unweighted fitted states retain their schema and
remain readable under the existing matching-version requirements.
