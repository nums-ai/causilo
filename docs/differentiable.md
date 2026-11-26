# Differentiable query prediction

`differentiable_predictor()` creates a separate callable for input differentiation
of an already fitted model. Ordinary `fit`, `predict`, `predict_proba`, and
`predict_logits` keep their existing behavior. This interface is for local
PyTorch use; it does not train the model or refit its context.
The SDK returns predictions with an input autograd graph. Callers select an
output or objective and use `torch.autograd.grad` to calculate derivatives.

## Supported scope

- A successful single-table fit with **one estimator**, numeric features, and
  the fitted `none` normalization. Here `none` still includes the SDK's fitted
  standardization and arcsinh tail compression; neither is bypassed.
- Finite, dense float32 or float64 query tensors in original fitted column order.
  Constant training columns stay in the query schema and have zero derivative.
- Classification probabilities or native logits, with columns in `classes_`
  order. Logits use the same class-count restriction as `predict_logits`.
- Regression mean or median in original target units.
- Fitted `sample_count`, zero-count row removal, weighted preprocessing, either
  fitted K/V-cache setting, and either `retain_preprocessing` setting.

Categorical queries, missing query values, ensembles, ECOC, batched fits,
regression quantiles/raw outputs, and gradients with respect to training rows,
targets, counts, or model parameters are outside this initial interface. An
unsupported fitted model raises an error; it is never silently refitted or
converted to a different prediction function. Higher-order derivatives and
`torch.func` transforms of this callable are not part of this initial contract.

## Differentiate an existing fit

The estimator below must already have been fitted with the supported options.
The default `n_estimators=8` is not supported by this initial interface.
Changing constructor options with `set_params` does not change an existing fit;
the callable uses the fitted context until an explicit refit replaces it.

```python
import torch

forward = regressor.differentiable_predictor(output_type="mean")
x = torch.tensor(X_query, dtype=torch.float64, requires_grad=True)
prediction = forward(x)
gradient, = torch.autograd.grad(prediction.sum(), x)
```

For classification:

```python
forward = classifier.differentiable_predictor(output_type="probability")
probabilities = forward(x)
gradient, = torch.autograd.grad(probabilities[:, 0].sum(), x)
```

Use `output_type="logits"` for classification logits and `"median"` for a
regression median. Regression returns `(rows,)`; classification returns
`(rows, classes)`. Predictions are CPU tensors, float32 for classification and
float64 for regression. Input gradients are in original feature units and return
to the input tensor's device/dtype. Query rows do not attend to each other, so
differentiating a sum obtains the corresponding per-row derivatives.

Create query tensors outside `torch.inference_mode()`. The callable enables
autograd internally, including when called inside an outer no-grad/inference
scope; perform downstream loss operations and differentiation with grad enabled.
It preserves upstream computation on non-leaf inputs. Model parameters are
detached for this forward operation: calling `backward()` does not populate or
clear the fitted model's parameter gradients or change its `requires_grad` flags.

Tensor inputs have no column labels. When fitted names exist, pass
`feature_names=...` to validate their order, or arrange tensors in the documented
`forward.feature_names_in_` order. `n_features_in_` is the original input width.

## Optional query intermediates and context reuse

Extensions that also report gradients in normalized coordinates can request:

```python
prediction, prepared = forward(x, return_preprocessed=True)
original, normalized = torch.autograd.grad(prediction.sum(), (x, prepared))
```

For classification, select an output column before taking the sum. `prepared`
contains only the current query's float32 normalized features, on the model
device and before feature permutation. Its columns correspond to the immutable
`retained_features_` index tuple in the original schema. This does not return
training tables, targets, frequency weights, fitted statistics, or K/V caches
as API payloads. The autograd graph nevertheless retains context-derived tensors
for backward, which same-process code may inspect; this does not hide context
from an untrusted caller.

`forward.prepare()` is optional and returns only a status string:
`built_private_cache_from_fitted_context`, `cloned_fitted_cache`, or
`reused_private_cache`. The callable invokes it lazily if needed. Even when
ordinary prediction was fitted without a K/V cache, this mode builds its own
cache; it may require substantially more memory than the ordinary uncached path.
Reuse the callable across queries when the fitted model is unchanged; creating
one per request repeats context preparation. Graphs are built for the current
query, and a new forward pass is needed after backward releases its graph.
Autograd also retains intermediate activations for backward. Split large query
tensors into smaller batches when needed; ordinary inference memory estimates
do not bound the full autograd graph. Release the callable and any remaining
autograd graphs when finished.

Keep the fitted context and parameters read-only while using a callable. Refits,
replaced state, training mode, and tracked tensor changes are rejected by
`check_fitted()` and each prediction. Arbitrary NumPy edits or mutations through
external storage aliases are not comprehensively detected. Concurrent mutation,
reentrant use of one fitted model, and serialization of this ephemeral callable
are unsupported. Recreate it from a fitted estimator after restoration/refitting.
The fitted estimator itself remains serializable after differentiation; the
callable is not attached to its saved state.

## Precision, validation, and information handling

Preprocessing uses CPU float64 before conversion to the model's float32 inputs.
Device transfers and dtype conversions are separate so MPS backward never needs
float64 on the GPU. CPU preprocessing and output transfers add overhead; this is
not a performance claim. Derivatives treat casts as continuous and do not model
float quantization. Sorting for a median is locally differentiable away from
ties; derivatives at ties follow PyTorch's selected ordering.

Derivatives describe the current query, not a guaranteed change over a finite
step. Curvature can make a fixed-step difference disagree with the local slope;
very small steps amplify floating-point rounding. Batch shape and execution
device can also affect results, as with ordinary prediction. Validate the actual
predictions at changed inputs when using gradients to propose input changes.

The automated tests compare against ordinary predictions and central finite
differences, cover weighted/cache/preprocessing-retention combinations, and check
fitted-state and parameter-gradient preservation. They use small deterministic
random models without checkpoint downloads. Device-specific tests skip unavailable
hardware. Opt-in `pretrained` tests use the official pinned checkpoint with
synthetic data for CPU and MPS prediction/gradient checks, including a CPU
comparison of integrated local slopes against finite prediction changes. CUDA,
large-context memory/performance, and information-reconstruction attacks require separate
validation before making claims about those settings.

This is an interface boundary, not a security boundary against code in the same
Python process. Open model checkpoints do not make a user's fitted data public.
Predictions, gradients, and transformed queries can reveal information about a
fitted context. Do not treat these outputs as anonymized or expose this callable
to untrusted hosted clients without a separate service policy and security design.

Until a release includes this method, install and record the exact SDK source
revision. The existing package version alone does not identify this development
API. No checkpoint, fitted dataset, or cache is bundled with the implementation.
