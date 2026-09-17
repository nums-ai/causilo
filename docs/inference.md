# Inference details

## Classification above ten classes

The classification checkpoint has ten output symbols. When a fitted target contains more than ten
classes, Causilo deterministically constructs a redundant error-correcting output code from
`random_state`. Each code row is evaluated as an ordinary classification problem within the native
head width, and mean log likelihood across active code symbols reconstructs probabilities in the
original `classes_` order. Classification with at most ten classes retains the direct path.

Many-class prediction runs the native model once per code row, so it is slower than direct
classification. With `use_kv_cache=True`, fit prepares a separate target-conditioned cache for every
code row, trading additional device memory and fitted-state size for faster repeated prediction.

Every code row retains the requested `n_estimators` ensemble views. The fixed recipe uses nine
active classes plus a rest symbol, redundancy four, and up to fifty codebook candidates (one
candidate above two hundred classes). Each candidate balances active coverage; minimum and then
mean Hamming distance select the winner. The row budget never falls below `ceil(classes / 9)`:
11--72 classes use eight rows, 100 classes use twelve, and 1,000 classes use 112. Redundancy is
limited by the row budget, so it does not guarantee four active appearances for every class.

The codebook and per-row context are saved with the fitted estimator; restoring does not regenerate
codes or rebuild K/V. Causilo implements the active/rest recipe also used by
[EXAONE-Tabular](https://github.com/LGAI-Research/EXAONE-Tabular/blob/cf55bd2d74aeb9c0b5d5d4f509d05831251a827e/src/exaonetabular/ecoc.py).
Code scheduling uses usage-priority ordering, candidate scoring uses active-set co-occurrences,
and decoding streams each row into a class-score accumulator rather than materializing a
rows-by-samples-by-classes tensor. No EXAONE package or model weights are required.

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

CUDA uses FP16 mixed precision, with regression column stages and both output heads in FP32. CPU execution uses FP32. Regression target scaling and output restoration use float64.

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
