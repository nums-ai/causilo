"""Behavioral contracts against public predictions; no pretrained downloads."""

import copy
import io
import pickle
from dataclasses import fields, is_dataclass

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from causilo import CausiloClassifier, CausiloRegressor, checkpoints
from causilo.execution.runner import ModelRunner


@pytest.fixture
def small_checkpoint(monkeypatch, make_batch_model):
    monkeypatch.setattr(checkpoints, "load_pretrained_model", lambda task: make_batch_model(task))


def data():
    rng = np.random.default_rng(124)
    context = rng.normal(size=(14, 5))
    context[:, -1] = 9
    query = rng.normal(size=(3, 5))
    query[:, -1] = 9
    # Exercise both compressed tails as well as the central preprocessing path.
    query[1, 0], query[2, 0] = -20, 20
    counts = np.array([0, 40, 2, 1, 5, 1, 2, 4, 1, 3, 1, 2, 1, 3])
    targets = {"classification": np.array(["z", "a", "m"] * 5)[:14],
               "regression": rng.normal(size=14) * 3 + 7}
    return context, query, counts, targets


def cached_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from cached_tensors(getattr(value, field.name))
    elif isinstance(value, tuple):
        for item in value:
            yield from cached_tensors(item)


def public_prediction(model, query, output):
    if output == "probability":
        return model.predict_proba(query)
    if output == "logits":
        return model.predict_logits(query)
    return model.predict(query, output_type=output)


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize("task,output", [
    ("classification", "probability"), ("classification", "logits"),
    ("regression", "mean"), ("regression", "median"),
])
@pytest.mark.parametrize("cache,retain", [(False, True), (False, False), (True, True), (True, False)])
@pytest.mark.parametrize("weighted", [False, True])
def test_prediction_gradient_and_fitted_state(
    small_checkpoint, device, task, output, cache, retain, weighted,
):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    context, query, counts, targets = data()
    kind = CausiloClassifier if task == "classification" else CausiloRegressor
    model = kind(n_estimators=1, device=device, use_kv_cache=cache, retain_preprocessing=retain).fit(
        context, targets[task], sample_count=counts if weighted else None,
    )
    expected = public_prediction(model, query, output)
    state = model._engine.state
    dataset_before = pickle.dumps(state.dataset)
    caches = state.caches
    cache_before = [(t, t.clone()) for t in cached_tensors(caches)]
    parameters = [(p, p.detach().clone(), p.requires_grad) for p in model._engine.model.parameters()]
    # Existing parameter gradients must remain intact even if the caller uses backward().
    parameters[0][0].grad = torch.full_like(parameters[0][0], .25)
    old_grad = parameters[0][0].grad.clone()
    forward = model.differentiable_predictor(output_type=output)
    assert forward.prepare() == (
        "cloned_fitted_cache" if cache else "built_private_cache_from_fitted_context"
    )
    assert forward.prepare() == "reused_private_cache"
    x = torch.tensor(query, dtype=torch.float64, requires_grad=True)
    result, prepared = forward(x, return_preprocessed=True)
    assert result.device.type == "cpu"
    assert prepared.shape == (3, 4)
    assert forward.retained_features_ == (0, 1, 2, 3)
    np.testing.assert_allclose(result.detach().numpy(), expected, atol=3e-6, rtol=3e-6)
    selected = result[:, 0] if result.ndim == 2 else result
    gradient, normalized_gradient = torch.autograd.grad(selected.sum(), (x, prepared), retain_graph=True)
    assert torch.isfinite(normalized_gradient).all()
    assert torch.equal(gradient[:, -1], torch.zeros(3, dtype=torch.float64))
    step = .003
    finite_difference = np.empty_like(query)
    for column in range(query.shape[1]):
        shift = np.zeros_like(query)
        shift[:, column] = step
        plus = public_prediction(model, query + shift, output)
        minus = public_prediction(model, query - shift, output)
        delta = (plus - minus) / (2 * step)
        finite_difference[:, column] = delta[:, 0] if delta.ndim == 2 else delta
    np.testing.assert_allclose(gradient.numpy(), finite_difference, atol=3e-4, rtol=.03)
    selected.sum().backward()
    torch.testing.assert_close(x.grad, gradient)
    split = torch.cat([forward(part) for part in x.split(1)])
    torch.testing.assert_close(split, result, atol=3e-6, rtol=3e-6)
    assert model._engine.state is state and state.caches is caches
    assert pickle.dumps(state.dataset) == dataset_before
    for tensor, before in cache_before:
        assert torch.equal(tensor, before)
    for index, (parameter, before, flag) in enumerate(parameters):
        assert torch.equal(parameter, before) and parameter.requires_grad == flag
        if index:
            assert parameter.grad is None
        else:
            assert torch.equal(parameter.grad, old_grad)
    np.testing.assert_array_equal(public_prediction(model, query, output), expected)


def fitted_classifier():
    context, query, _, targets = data()
    return CausiloClassifier(n_estimators=1, device="cpu").fit(context, targets["classification"]), query


def test_names_composed_inputs_and_outer_inference_mode(small_checkpoint):
    context, query, _, targets = data()
    names = ("a", "b", "c", "d", "constant")
    model = CausiloClassifier(n_estimators=1, device="cpu").fit(
        pd.DataFrame(context, columns=names), targets["classification"],
    )
    with torch.inference_mode():
        forward = model.differentiable_predictor()
    assert forward.feature_names_in_ == names
    source = torch.tensor(query, dtype=torch.float64, requires_grad=True)
    query_tensor = source * 2
    with torch.inference_mode():
        predictions = forward(query_tensor, feature_names=names)
    assert predictions.requires_grad
    derivative = torch.autograd.grad(predictions[:, 0].sum(), source)[0]
    assert torch.isfinite(derivative).all()
    expected = model.predict_proba(pd.DataFrame(query * 2, columns=names))
    np.testing.assert_allclose(predictions.detach().numpy(), expected, atol=2e-6, rtol=2e-6)
    with pytest.raises(ValueError, match="names"):
        forward(source, feature_names=tuple(reversed(names)))


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_float32_queries_on_model_device(small_checkpoint, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    context, query, counts, targets = data()
    model = CausiloRegressor(n_estimators=1, device=device, use_kv_cache=True).fit(
        context, targets["regression"], sample_count=counts,
    )
    forward = model.differentiable_predictor()
    x = torch.tensor(query, dtype=torch.float32, device=device, requires_grad=True)
    result = forward(x)
    gradient = torch.autograd.grad(result.sum(), x)[0]
    assert gradient.device == x.device and gradient.dtype == x.dtype
    assert torch.isfinite(gradient).all()
    original = x.detach().cpu().double().requires_grad_()
    expected = forward(original)
    derivative = torch.autograd.grad(expected.sum(), original)[0]
    torch.testing.assert_close(gradient.cpu().double(), derivative, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(
        result.detach().numpy(), model.predict(x.detach().cpu().numpy()), atol=3e-6, rtol=3e-6,
    )


def test_failed_forward_restores_parameter_identity_and_grad_flags(small_checkpoint, monkeypatch):
    model, query = fitted_classifier()
    parameters = list(model._engine.model.parameters())
    parameters[0].requires_grad_(False)
    flags = [parameter.requires_grad for parameter in parameters]
    forward = model.differentiable_predictor()
    forward.prepare()
    with monkeypatch.context() as local:
        def fail(*args, **kwargs):
            raise RuntimeError("injected forward failure")

        local.setattr(ModelRunner, "predict_cached", fail)
        with pytest.raises(RuntimeError, match="injected"):
            forward(torch.tensor(query, requires_grad=True))
    assert all(a is b for a, b in zip(parameters, model._engine.model.parameters()))
    assert [parameter.requires_grad for parameter in parameters] == flags
    expected = model.predict_proba(query)
    np.testing.assert_allclose(forward(torch.tensor(query)).detach().numpy(), expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("bad", [
    np.zeros((2, 5)), torch.ones(2, 5, dtype=torch.int64), torch.ones(5),
    torch.ones(0, 5), torch.ones(2, 4), torch.full((2, 5), float("nan")),
    torch.full((2, 5), float("inf")), torch.ones(2, 5).to_sparse(),
])
def test_invalid_queries_fail_before_cache_build(small_checkpoint, bad, monkeypatch):
    model, _ = fitted_classifier()
    forward = model.differentiable_predictor()
    monkeypatch.setattr(forward, "prepare", lambda: pytest.fail("Prepared invalid query"))
    with pytest.raises((TypeError, ValueError)):
        forward(bad)


def test_inference_query_and_serialization_rejected(small_checkpoint):
    model, query = fitted_classifier()
    forward = model.differentiable_predictor()
    with torch.inference_mode():
        x = torch.tensor(query)
    with pytest.raises(ValueError, match="outside"):
        forward(x)
    with pytest.raises(TypeError, match="ephemeral"):
        pickle.dumps(forward)


def test_missing_training_values_and_fit_under_inference_mode(small_checkpoint):
    context, query, counts, targets = data()
    context[1, 0] = np.nan
    with torch.inference_mode():
        model = CausiloRegressor(n_estimators=1, device="cpu", use_kv_cache=True).fit(
            context, targets["regression"], sample_count=counts,
        )
        forward = model.differentiable_predictor()
    x = torch.tensor(query, dtype=torch.float64, requires_grad=True)
    prediction = forward(x)
    gradient = torch.autograd.grad(prediction.sum(), x)[0]
    assert torch.isfinite(gradient).all()
    np.testing.assert_allclose(prediction.detach().numpy(), model.predict(query), atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("classes", [1, 2, 10])
def test_class_output_shape_and_query_independence(small_checkpoint, classes):
    context, query, counts, _ = data()
    labels = np.arange(len(context)) % classes
    model = CausiloClassifier(n_estimators=1, device="cpu").fit(context, labels, sample_count=counts)
    x = torch.tensor(query, dtype=torch.float64, requires_grad=True)
    probabilities = model.differentiable_predictor()(x)
    assert probabilities.shape == (len(query), classes)
    np.testing.assert_allclose(
        probabilities.detach().numpy(), model.predict_proba(query), atol=3e-6, rtol=3e-6,
    )
    gradients = torch.autograd.grad(probabilities[:, 0].sum(), x, retain_graph=True)[0]
    for row in range(len(query)):
        individual = torch.autograd.grad(probabilities[row, 0], x, retain_graph=True)[0]
        torch.testing.assert_close(individual[row], gradients[row])
        other_rows = [index for index in range(len(query)) if index != row]
        assert torch.count_nonzero(individual[other_rows]) == 0
    if classes == 1:
        torch.testing.assert_close(probabilities, torch.ones_like(probabilities))
        assert torch.count_nonzero(gradients) == 0


def test_unfitted_unsupported_and_refit_contracts(small_checkpoint):
    with pytest.raises(NotFittedError):
        CausiloClassifier().differentiable_predictor()
    context, query, _, targets = data()
    for kind, task in [(CausiloClassifier, "classification"), (CausiloRegressor, "regression")]:
        model = kind(n_estimators=2, device="cpu").fit(context, targets[task])
        state = model._engine.state
        with pytest.raises(NotImplementedError, match="n_estimators=1"):
            model.differentiable_predictor()
        assert model._engine.state is state
    model, query = fitted_classifier()
    forward = model.differentiable_predictor()
    with pytest.raises(ValueError, match="output_type"):
        model.differentiable_predictor(output_type="labels")
    model.fit(context, targets["classification"])
    with pytest.raises(RuntimeError, match="changed"):
        forward(torch.tensor(query))
    model.fit_batched({"table": (context, targets["classification"])})
    with pytest.raises(ValueError, match="fit_batched"):
        model.differentiable_predictor()


@pytest.mark.parametrize("kind,task", [
    (CausiloClassifier, "classification"), (CausiloRegressor, "regression"),
])
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("serializer", [pickle, joblib], ids=["pickle", "joblib"])
def test_differentiation_preserves_estimator_serialization(small_checkpoint, kind, task, cache, serializer):
    context, query, counts, targets = data()
    model = kind(n_estimators=1, device="cpu", use_kv_cache=cache, retain_preprocessing=False).fit(
        context, targets[task], sample_count=counts,
    )
    parameters, attributes = model.get_params(), set(vars(model))
    output = "probability" if task == "classification" else "mean"
    expected = public_prediction(model, query, output)
    x = torch.tensor(query, dtype=torch.float64, requires_grad=True)
    prediction = model.differentiable_predictor()(x)
    selected = prediction[:, 0] if task == "classification" else prediction
    gradient = torch.autograd.grad(selected.sum(), x)[0]
    assert model.get_params() == parameters and set(vars(model)) == attributes
    buffer = io.BytesIO()
    serializer.dump(model, buffer)
    buffer.seek(0)
    restored = serializer.load(buffer)
    np.testing.assert_array_equal(public_prediction(restored, query, output), expected)
    restored_prediction = restored.differentiable_predictor()(x)
    restored_selected = restored_prediction[:, 0] if task == "classification" else restored_prediction
    restored_gradient = torch.autograd.grad(restored_selected.sum(), x)[0]
    torch.testing.assert_close(restored_prediction, prediction)
    torch.testing.assert_close(restored_gradient, gradient)
    assert restored.get_params() == parameters


@pytest.mark.parametrize("kind,task", [
    (CausiloClassifier, "classification"), (CausiloRegressor, "regression"),
])
def test_fitted_options_are_preserved_until_refit(small_checkpoint, kind, task):
    context, query, _, targets = data()
    model = kind(n_estimators=1, device="cpu").fit(context, targets[task])
    x = torch.tensor(query, dtype=torch.float64, requires_grad=True)
    forward = model.differentiable_predictor()
    expected = forward(x)
    model.set_params(n_estimators=8, use_kv_cache=True, random_state=8)
    torch.testing.assert_close(forward(x), expected)
    torch.testing.assert_close(model.differentiable_predictor()(x), expected)
    unfitted = clone(model)
    assert unfitted.get_params() == model.get_params()
    with pytest.raises(NotFittedError):
        unfitted.differentiable_predictor()
    model.fit(context, targets[task])
    with pytest.raises(RuntimeError, match="changed"):
        forward(x)
    with pytest.raises(NotImplementedError, match="n_estimators=1"):
        model.differentiable_predictor()


@pytest.mark.parametrize("categorical", [False, True])
def test_categorical_and_ecoc_are_explicitly_unsupported(small_checkpoint, categorical):
    context, _, _, targets = data()
    table = pd.DataFrame(context)
    if categorical:
        table[0] = ["a", "b"] * 7
    y = targets["classification"] if categorical else np.arange(14)
    model = CausiloClassifier(n_estimators=1, device="cpu").fit(table, y)
    with pytest.raises(NotImplementedError, match="Categorical" if categorical else "ECOC"):
        model.differentiable_predictor()


@pytest.mark.parametrize("mutation", ["replace", "inplace", "train", "failed_refit"])
def test_stale_predictor_is_rejected(small_checkpoint, mutation):
    model, query = fitted_classifier()
    forward = model.differentiable_predictor()
    forward.prepare()
    if mutation == "replace":
        model._engine.model = copy.deepcopy(model._engine.model)
    elif mutation == "inplace":
        with torch.no_grad():
            next(model._engine.model.parameters()).add_(1)
    elif mutation == "train":
        model._engine.model.train()
    else:
        with pytest.raises(ValueError):
            model.fit(query, [0])
    with pytest.raises((RuntimeError, ValueError)):
        forward(torch.tensor(query))


@pytest.mark.pretrained
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("task,output", [
    ("classification", "probability"), ("classification", "logits"),
    ("regression", "mean"), ("regression", "median"),
])
def test_pretrained_prediction_and_gradient(batch_runtime, task, output, cache, device):
    """Opt-in smoke test with the official pinned checkpoint and synthetic data."""
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    context, query, counts, targets = data()
    kind = CausiloClassifier if task == "classification" else CausiloRegressor
    model = kind(n_estimators=1, device=device, use_kv_cache=cache).fit(
        context, targets[task], sample_count=counts,
    )
    x = torch.tensor(query[:1], dtype=torch.float64, requires_grad=True)
    prediction = model.differentiable_predictor(output_type=output)(x)
    expected = public_prediction(model, query[:1], output)
    np.testing.assert_allclose(prediction.detach().numpy(), expected, atol=3e-5, rtol=3e-5)
    chosen = prediction[:, 0] if prediction.ndim == 2 else prediction
    gradient = torch.autograd.grad(chosen.sum(), x)[0].numpy()
    step = .003
    differences = np.empty_like(query[:1])
    for column in range(x.shape[1]):
        shift = np.zeros_like(query[:1])
        shift[:, column] = step
        plus = public_prediction(model, query[:1] + shift, output)
        minus = public_prediction(model, query[:1] - shift, output)
        difference = (plus - minus) / (2 * step)
        differences[:, column] = difference[:, 0] if difference.ndim == 2 else difference
    np.testing.assert_allclose(gradient, differences, atol=3e-4, rtol=.03)
    assert all(parameter.grad is None for parameter in model._engine.model.parameters())


@pytest.mark.pretrained
def test_pretrained_gradients_integrate_to_prediction_changes(batch_runtime):
    """Check curved probability paths without assuming a constant local slope."""
    rng = np.random.default_rng(124)
    context, query = rng.normal(size=(24, 3)), rng.normal(size=(3, 3))
    model = CausiloClassifier(n_estimators=1, device="cpu", use_kv_cache=True).fit(
        context, np.arange(24) % 3, sample_count=np.array([0, 20, 2, 1, 5, 1] * 4),
    )
    forward = model.differentiable_predictor()
    nodes, weights = np.polynomial.legendre.leggauss(5)
    step = .003
    for column in range(query.shape[1]):
        shift = np.zeros_like(query)
        shift[:, column] = step
        expected = model.predict_proba(query + shift) - model.predict_proba(query - shift)
        integrated = np.zeros_like(expected, dtype=np.float64)
        for node, weight in zip(nodes, weights):
            x = torch.tensor(query + node * shift, dtype=torch.float64, requires_grad=True)
            probabilities = forward(x)
            for output in range(probabilities.shape[1]):
                gradient = torch.autograd.grad(
                    probabilities[:, output].sum(), x,
                    retain_graph=output < probabilities.shape[1] - 1,
                )[0]
                integrated[:, output] += step * weight * gradient[:, column].numpy()
        # Compare output units: tiny finite-difference steps amplify FP32
        # rounding, while larger steps average slopes over a curved interval.
        np.testing.assert_allclose(integrated, expected, atol=3e-5, rtol=3e-5)
