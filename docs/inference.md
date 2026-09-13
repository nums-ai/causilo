# Execution details

## Precision and reproducibility

CUDA uses FP16 mixed precision, with regression column stages and both output heads in FP32. CPU execution uses FP32. Regression target scaling and output restoration use float64.

The same inputs and seed reproduce the fitted feature and class permutations without changing global RNG state. Floating-point results can vary with execution device and batch shape; bitwise determinism is not enforced.

## Restoring a fitted estimator

With `device="auto"`, restoration selects an available device again. An explicitly configured device must be available. Saved state includes fitted preprocessing and optional attention caches, while pretrained weights are loaded from the pinned checkpoint. Use matching Causilo and dependency versions, including the Python major/minor version.
