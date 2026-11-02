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

Constructor options apply to all tables. Both `fit` and `fit_batched` replace
the previous fit; a failed refit leaves the estimator unfitted. Use the
prediction methods matching the fitted mode.

See the runnable [batch example](../examples/batched_prediction.py).

## Precision and reproducibility

CUDA uses FP16 mixed precision, with regression column stages and both output heads in FP32. CPU and MPS execution use FP32. Regression target scaling and output restoration use float64 on the CPU.

The same inputs and seed reproduce the fitted feature and class permutations without changing global RNG state. Floating-point results can vary with execution device and batch shape; bitwise determinism is not enforced.

## Memory use

Causilo automatically reduces execution batches or splits query rows when
memory is tight, preserving all training rows and ensemble members. Without a
fitted K/V cache, it can also recompute intermediate values to reduce memory use.
Large contexts can still exceed device memory.

`use_kv_cache=True` speeds up repeated prediction by retaining a cache for each
table and ensemble member, at the cost of additional device memory.

## Restoring a fitted estimator

With `device="auto"`, restoration selects an available device again. An explicitly configured device must be available. Saved state includes fitted preprocessing and optional attention caches, while weights are loaded from the checkpoint used during fit. Use matching Causilo and dependency versions, including the Python major/minor version.

Pickle/joblib also preserve every ID and fitted context from `fit_batched`,
without storing model weights. Saved K/V caches are restored without rebuilding
them, for both single-table and batched fits.

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
