# Inference details

The classifier and regressor use Causilo 1.5 by default. Specify
`mode="default"` or omit both `mode` and `n_estimators` to use the
default ensemble. In 1.5, classification uses four members (two Base and
two Robust) and regression uses eight. With `model_version="1.0"`,
both tasks use eight by default. Choose `mode="light"` for one member
with either model version.

To set an explicit ensemble size, pass `n_estimators`. Passing both `mode`
and `n_estimators` with non-None values raises a `ValueError` at fit time.
To reproduce the classification default from Causilo 1.5.0, use
`CausiloClassifier(n_estimators=8)`.

See [checkpoints](checkpoints.md) for the weights, revisions, and access
instructions for each task and model version.

`fit(X, y)` prepares preprocessing and the inference context from the labeled
rows in `X` and `y`. When enabled, it also prepares the context cache.
`predict(X)` applies the fitted preprocessing and uses the fixed checkpoint
to make predictions for query rows.

## Context size

Both modes of Causilo 1.5 use all eligible context rows by default
(`max_context=None`). Set a positive integer to cap the number of rows per
estimator. Sampling can reduce memory use and runtime on large tables.

```python
classifier = CausiloClassifier(max_context=200_000)
```

Tables at or below the cap use all eligible rows. For larger tables, each estimator
draws its own rows without replacement during `fit`. Feature types, retained
columns, and their order come from all eligible rows. Category mappings,
frequency values, and normalization are fitted on each estimator’s selected
rows. A fixed `random_state` reproduces the draws. Classification
keeps every observed class, so the cap must be at least the number of classes.
Many-class ECOC code rows share one sampled context. All predictions reuse
the rows selected during fit. When `sample_count` is supplied, only positive-count
rows are eligible and each selected row keeps its original count. The cap
has no effect on Causilo 1.0.

## Input columns

With 1.5, fit and predict with the same unique DataFrame column labels. Query
columns can be reordered because prediction aligns them by name. With 1.0,
use the same names in the same order. Array queries use the fitted column
positions. In object arrays, a column with numeric values and optional missing
cells is treated as numeric. Strings and booleans can still be categorical
features in the same array.

With 1.0, encode datetime and duration columns before fitting. With 1.5, encode
datetime, duration, period, complex, and other unsupported column types as
numeric or categorical features before fitting or predicting.

In 1.5, the labeled context determines how categorical features are encoded.
An unseen numeric category between two observed numeric levels is encoded by
linearly interpolating their codes. Any frequency feature uses the same weights.
Missing values, unseen text categories, and numeric categories outside the observed
range keep their usual encoding. Keep feature types consistent between fit and prediction.

## Feature preprocessing

Causilo 1.5 classification and regression use the same feature views.

| `n_estimators` | Feature views |
| --- | --- |
| 1 (Light) | Base |
| 2 | Base, Robust |
| 3 | Base × 2, Robust |
| 4 | Base × 2, Robust × 2 |
| 5 | Base × 3, Robust × 2 |
| 8 | Base × 4, Robust × 2, Rank-Gaussian × 2 |

Base and Robust start with standardization fitted on observed context cells.
Robust then applies median and interquartile scaling with a soft limit.
Rank-Gaussian uses the Causilo 1.0 Rank2Gaussian transform. It temporarily
fills missing context cells with their feature mean, fits mid-ranks on the
encoded values, then standardizes the Gaussian scores. All three views use
context-fitted `arcsinh` tail compression and preserve missing values.

When the number of distinct member permutations limits a fit, the shared
recipe scales the view counts to the fitted ensemble size.
Regression trims one value from each end for every four fitted members.

## Classification outputs

For a fitted `CausiloClassifier`, choose the prediction method for the output
you need:

| Method | Output | Shape |
| --- | --- | --- |
| `predict(X)` | Predicted class labels | `(n_samples,)` |
| `predict_proba(X)` | Class probabilities | `(n_samples, n_classes)` |
| `predict_logits(X)` | Combined logits before softmax | `(n_samples, n_classes)` |

```python
labels = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)
logits = classifier.predict_logits(X_test)
```

All three methods return NumPy arrays. Probability and logit columns follow
`classes_`, with two columns for binary classification. Applying
`scipy.special.softmax(logits, axis=1)` matches `predict_proba(X_test)`.

The 1.5 ensemble aligns each member's logits to the fitted class order first.
For each row and class, it averages the logits of all fitted members.

`predict_logits` supports single-table fits within the checkpoint's native
class limit (10 for the hosted checkpoints). Beyond that limit, use
`predict_proba` or `predict` for the fitted many-class classifier.

For 1.5, contexts with up to ten classes use the configured ensemble. With
more than ten, the classifier uses error-correcting output codes (ECOC).
Each code row runs one estimator for any ensemble size.
Code rows run on the first selected device. In 1.0, each code row uses the
configured ensemble.

## Regression outputs

For `CausiloRegressor`, choose the prediction output with `output_type`:

| `output_type` | Output | Shape |
| --- | --- | --- |
| `"mean"` (default) | Mean prediction | `(n_samples,)` |
| `"median"` | Ensemble's 0.5 quantile | `(n_samples,)` |
| `"quantiles"` | Quantiles at the requested probabilities | `(n_samples, n_quantiles)` |
| `"raw"` | All native quantiles | `(n_samples, n_native_quantiles)` |

```python
mean = regressor.predict(X_test)
median = regressor.predict(X_test, output_type="median")
quantiles = regressor.predict(
    X_test, output_type="quantiles", quantiles=[0.05, 0.5, 0.95]
)
```

Quantile prediction preserves the order and duplicates in the requested list.
A single probability still returns a two-dimensional array. Probabilities
must be finite and strictly between 0 and 1. Both versions interpolate
linearly between native quantiles. Beyond either end, the two nearest native
quantiles determine the slope in log-probability space. Extreme probabilities
can produce large values.

`predict(X, output_type="raw")` returns the combined quantiles at the
checkpoint's native levels. The hosted models have 999 levels. Use
`output_type="quantiles"` to choose the probabilities.

1.5 centers and scales targets in float64 before conversion for the model
and returns float64 predictions in the original target units.

The 1.5 ensemble restores each member's target orientation before averaging.
At each quantile, the ensemble drops one prediction from each end for every
four fitted members. E4 averages the middle two predictions, E5 the middle
three, and E8 the middle four. When trimming is applied,
the mean prediction integrates this quantile distribution with endpoints that
include the fitted target range. Ensembles with fewer than four members
average all available predictions.

## Independent table batches

Use `fit_batched` with nonempty string IDs to share model weights across
independent tables. Row counts, feature schemas, and class sets can differ.

```python
regressor.fit_batched({
    "sales": (X_train_sales, y_train_sales),
    "price": (X_train_price, y_train_price),
})
predictions = regressor.predict_batched({
    "sales": X_test_sales,  # mean output
    "price": {"X": X_test_price, "output_type": "quantiles", "quantiles": [0.1, 0.5, 0.9]},
})
```

Regression output options follow [Regression outputs](#regression-outputs)
and can differ by table; passing `X` alone returns mean predictions.

For classification, use `predict_batched({id: X})` for labels or
`predict_proba_batched({id: X})` for probabilities. Probability columns follow
`classes_batched_[id]`.

Query any subset of fitted IDs; results preserve query ID order. Each query
must match its fitted schema and contain at least one row. An empty mapping
returns `{}`.

1.0 can group compatible members from different tables into one model call.
1.5 processes tables in order and batches ensemble members within each table.
The same batch can therefore have different throughput and peak memory use
across the two versions.

Constructor options apply to all tables. Both `fit` and `fit_batched` replace
the previous fit; a failed refit leaves the estimator unfitted. Use the
prediction methods matching the fitted mode.

See the runnable [batch example](../examples/batched_prediction.py).

## Precision and reproducibility

CUDA inference uses mixed precision. CPU and MPS use FP32. Use the same inputs,
`n_estimators`, and an integer `random_state` to reproduce the ensemble
selection and ECOC codebook. 1.0 requires an integer seed. In 1.5,
`random_state=None` leaves both unseeded. The number of 1.5 ensemble members
is limited by the available distinct views of the features and targets.
Floating-point results can vary with execution device and batch shape.

## Memory use

Causilo can reduce execution batches or split query rows when memory is tight.
Each query still uses the fitted context and ensemble. Large contexts can
exceed device memory.

On Linux, 1.5 sizes host allocations using both available RAM and the memory
limit of the process's cgroup. If a shared query batch exceeds host memory,
1.5 retries it with fewer rows. On multiple GPUs, a device allocation failure
retries the affected GPU's work.

1.0 uses one execution device. For 1.5, pass an ordered CUDA device list to
distribute ensemble members. One estimator uses the first device. Set
`device_memory_budget_bytes` to guide batch sizing on each GPU. Allow
additional memory for model weights and tensors already on the device.

```python
model = CausiloRegressor(
    n_estimators=8,
    device=["cuda:0", "cuda:1"],
    device_memory_budget_bytes={"cuda:0": 64 * 1024**3, "cuda:1": 48 * 1024**3},
)
```

Use `use_kv_cache=True` to reuse the context across repeated predictions.
Cache memory grows with the context, ensemble size, and number of fitted
tables. 1.0 stores the cache on the execution device. On CUDA, 1.5 can also
use host memory for caches when device memory is limited. Fitting a large
cache can pin part of the available host RAM to overlap transfers with GPU
work. Each GPU can stage two layers at a time and share a layer's keys and
values across query chunks for its ensemble members. This also applies when
the fitted context uses row frequencies supplied through `sample_count`.

With `use_kv_cache=False`, the memory-bounded path can reuse each member's
context encoding and sorted feature values within a prediction call. A change
in tile size can reuse the encoding if it fits the current device budget.
For classification with more than ten classes, 1.5 can process several query
chunks with one codebook row's context at a time. Temporary state is released
when the call returns.

In 1.0, `retain_preprocessing=False` keeps the encoded training table and rebuilds
normalized copies as needed. In 1.5, the default stores one float32 table
for each fitted view. Setting it
to `False` stores the encoded float64 table and applies the fitted transforms
when the context is needed.

## Restoring a fitted estimator

Save the fitted context with `save_context()` and select the execution device
when you load it. By default, the loader uses `device="auto"` on the current host.
You can also pass `device_memory_budget_bytes` for selected 1.5 CUDA devices.

```python
classifier.save_context("context.pkl")
restored = CausiloClassifier.load_context("context.pkl", device="cpu")
labels = restored.predict(X_test)
```

The saved context includes batch IDs, preprocessing, and optional attention
caches. Restore uses the checkpoint that produced the fitted context. Changes
to `model_version` or `checkpoint_path` apply at the next `fit()`. 1.5 may
rebuild the cache when restoring on a different device.

Load fitted-state files from trusted sources with matching Causilo and
dependency versions. Refit from the original training data to replace an earlier
saved fit. Python pickle and joblib round trips of the estimator use the device
and memory settings in the saved estimator parameters.

Frameworks can pickle Causilo 1.5 fitted attributes directly. Restoration
recreates unlocked locks, preserving those shared by sampled members. CUDA
streams and temporary length caches are rebuilt during prediction. Estimator
pickle and `save_context()` use the checkpoint-aware restoration described above.

## Local checkpoints

Pass `checkpoint_path` with the weights file or directory described in
[Download and load](checkpoints.md#download-and-load). The local configuration
selects the runtime.

A saved fitted estimator records the absolute local path and a digest of the
configuration and weights. Restoration requires the same files at that path and
rejects changed contents before reusing fitted context or attention caches.
The absolute path is used when restoring from another working directory.

Changing `checkpoint_path` with `set_params` takes effect on the next fit. Until
then, prediction and saved fitted state use the previously loaded weights. Keep
local checkpoint files unchanged while using a fitted estimator; to load updated
weights at the same path, construct a new estimator. Relocating the files for
restoration is not supported by this interface.

## Aggregated observations and repetition counts

Both estimators accept optional `sample_count` in `fit`. Each entry is an
**integer repetition count**: a row with count 30 represents 30 identical
observations with the same features and target. Counts are not loss weights or
arbitrary importance scores. Omitting them keeps the usual `fit(X, y)` behavior.
`fit_batched` does not accept counts.

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

classifier = CausiloClassifier(model_version="1.0").fit(X, y, sample_count=counts)
p_success = classifier.predict_proba(X_test)[:, list(classifier.classes_).index(1)]
```

For a DataFrame, use `X_group.iloc[rows]` to preserve column names and types.
This creates at most two rows per group, regardless of its trial count. Different
observations can only be combined when their features **and target** are identical;
the estimator does not automatically group rows. Success proportions alone do
not recover the individual observations when features vary within a group.

Counts must be finite nonnegative integers, one per input row, with a positive
sum no larger than `2**53`. Integer-valued floats are accepted. Zero-count rows
are skipped when fitting. Both versions store one row per supplied observation
with a positive count. Results can differ from explicit repetition because of
floating-point rounding. Saved estimators retain their row counts.
