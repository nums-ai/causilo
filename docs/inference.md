# Inference details

## Independent table batches

Both estimators support independent fitted contexts keyed by nonempty string IDs.
Tables may have different training/query row counts, feature schemas, and class
sets. Each query must match its own fitted table's schema.

```python
from causilo import CausiloRegressor

regressor = CausiloRegressor(device="cuda", n_estimators=8)
regressor.fit_batched({
    "sales": (X_train_sales, y_train_sales),
    "price": (X_train_price, y_train_price),
})
predictions = regressor.predict_batched({
    "sales": X_test_sales,  # mean output
    "price": {
        "X": X_test_price,
        "output_type": "quantiles",
        "quantiles": [0.1, 0.5, 0.9],
    },
})
```

Results are dictionaries in query ID order. IDs, not positions, match queries
to contexts. Query any subset repeatedly; an empty query mapping returns `{}`.
Individual query tables must contain at least one row. Unknown IDs, schema
mismatches, and invalid per-table options raise errors naming the table. All
queries are validated before prediction starts; prediction does not refit or
replace any context. There are no common output options or `overrides` argument.

For classification, use `predict_batched({id: X})` for original labels and
`predict_proba_batched({id: X})` for probabilities. Each probability matrix's
columns follow `classes_batched_[id]`; different tables may have different
classes, including many-class ECOC contexts. Batch fits expose
`n_features_in_batched_` and `feature_names_in_batched_` (the latter contains only
IDs fitted with string-named DataFrame columns), without changing the types of
the single-table sklearn attributes.

`fit_batched` returns `self`. Each `fit` or `fit_batched` replaces the entire
previous fit, including when a refit fails. Single-table prediction requires
`fit`; batched prediction requires `fit_batched`. Construction and sklearn
cloning remain lazy and unfitted. Settings such as `checkpoint_path`, `device`, `n_estimators`,
`random_state`, `retain_preprocessing`, and `use_kv_cache` apply to all tables and
take effect on refit. Preprocessing, target scaling, category mappings, and
seeded ensemble permutations remain independent per table.

Weights are shared. The execution planner groups similarly sized table/member
items into bounded microbatches. It preserves partial final feature groups with
numeric-zero padding, excludes extra groups and padded context rows from
attention, and uses the valid context length for query scaling. Query padding
is removed before per-table aggregation. Same-size inputs avoid unnecessary
masks. OOM recovery reduces the execution batch and then uses the existing
single-member query partitioning/recomputation paths, retaining all training
rows and ensemble members. Different regression output modes can share one
execution batch; outputs are restored to each table's target units.

As soon as all ensemble members for a table are available, prediction aggregates
them in fitted member order and releases their intermediate outputs. Completed
tables therefore do not accumulate member outputs across the whole request.
Temporary execution batches, incomplete tables, transformed queries, and final
results still consume memory. Median outputs own their storage rather than
retaining the full native-quantile matrix.

With `use_kv_cache=True`, fit builds unpadded per-table/member caches. Prediction
temporarily packs compatible caches for batched execution and accounts for
their additional memory. Pickle/joblib save all contexts under a separate
batched-state schema, omit weights, and enforce the same runtime/revision
checks as single-table serialization. Restoring caches never rebuilds them.

Batch composition can change floating-point rounding. Numerical equivalence
within device-appropriate tolerances is the contract, not bitwise equality
between independent and batched execution. CUDA throughput and memory usage
must be measured on the intended workload and hardware.

Batch validation uses the pinned checkpoint
`94f2bd91db0737d4da59f347910662905ecb5a09`. On Apple M5 Pro with Python 3.12.15
and PyTorch 2.13.0, the offline suite passed 276 tests (15 device-related skips)
and the full pretrained suite passed 72 tests (10 device-related skips).
CPU and explicit MPS were covered; CUDA remains unverified.
See [the batch example](../examples/batched_prediction.py) and
[the benchmark](../examples/benchmark_batched.py) for runnable usage and a
small synthetic sequential-versus-batched comparison.

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

With `device="auto"`, restoration selects an available device again. An explicitly configured device must be available. Saved state includes fitted preprocessing and optional attention caches, while weights are loaded from the checkpoint used during fit. Use matching Causilo and dependency versions, including the Python major/minor version.

## Local checkpoints

By default, estimators download and cache the pinned official checkpoint. To use
local weights without contacting Hugging Face, pass `checkpoint_path`:

```python
from causilo import CausiloClassifier, CausiloRegressor

classifier = CausiloClassifier(checkpoint_path="causilo/classifier/model.safetensors")
regressor = CausiloRegressor(checkpoint_path="causilo/regressor")
```

The argument accepts a string or `pathlib.Path` pointing to a weights file or a
directory containing `model.safetensors`. Its matching `config.json` must be in
the same directory. For offline use, copy both files from the task's folder in
the [model repository](https://huggingface.co/nums-ai/causilo) to the target machine.
Compatible custom weights are also supported: their configuration, tensor names
and shapes, and safetensors metadata must follow the Causilo checkpoint format.
Construction and sklearn cloning do not read files; loading happens during fit.

A saved fitted estimator records the absolute local path and a digest of the
configuration and weights. Restoration requires the same files at that path and
rejects changed contents before reusing fitted context or attention caches.
Changing the working directory does not change the stored fitted source. The
digest checks identity, not whether the weights are an official release.

Changing `checkpoint_path` with `set_params` takes effect on the next fit. Until
then, prediction and saved fitted state use the previously loaded weights. Keep
local checkpoint files unchanged while using a fitted estimator; to load updated
weights at the same path, construct a new estimator. Relocating the files for
restoration is not supported by this interface.
