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

CUDA uses FP16 mixed precision, with regression column stages and both output heads in FP32. CPU execution uses FP32. Regression target scaling and output restoration use float64.

The same inputs and seed reproduce the fitted feature and class permutations without changing global RNG state. Floating-point results can vary with execution device and batch shape; bitwise determinism is not enforced.

## Restoring a fitted estimator

With `device="auto"`, restoration selects an available device again. An explicitly configured device must be available. Saved state includes fitted preprocessing and optional attention caches, while pretrained weights are loaded from the pinned checkpoint. Use matching Causilo and dependency versions, including the Python major/minor version.
