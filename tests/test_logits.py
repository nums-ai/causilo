import pickle
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import softmax
from sklearn.exceptions import NotFittedError

from causilo import CausiloClassifier, checkpoints
from causilo import prediction as prediction_pipeline
from causilo.execution.runner import ModelRunner
from causilo.outputs import classification_logits, classification_probabilities


@pytest.fixture
def small_checkpoint(monkeypatch, make_batch_model):
    monkeypatch.setattr(checkpoints, "load_pretrained_model", lambda task: make_batch_model(task))


def tables():
    rng = np.random.default_rng(62)
    table = pd.DataFrame({
        "kind": ["red", "blue", None] * 10,
        "amount": rng.normal(size=30),
        "other": rng.normal(size=30),
        "constant": 1,
    })
    table.loc[2, "amount"] = np.nan
    query = pd.DataFrame({
        "kind": ["new", None, "red"],
        "amount": [0.25, np.nan, -0.5],
        "other": [1.0, 0.0, -1.0],
        "constant": 1,
    })
    return table, query


@pytest.mark.parametrize("n_estimators", [1, 2])
def test_known_logits_restore_class_order_and_average(n_estimators, monkeypatch):
    pretrained = SimpleNamespace(config=SimpleNamespace(outputs=4, task="classification"))
    pretrained.to = lambda device: pretrained
    monkeypatch.setattr(checkpoints, "load_pretrained_model", lambda task: pretrained)
    model = CausiloClassifier(n_estimators=n_estimators, device="cpu").fit(
        np.arange(12).reshape(6, 2), ["zebra", "ant", "moose"] * 2,
    )
    dataset = model._engine.state.dataset
    orders = [(2, 0, 1), (1, 2, 0)]
    dataset.members = tuple(replace(member, class_order=order) for member, order in zip(dataset.members, orders))
    raw = [
        torch.tensor([[2., 4., 8., 100.], [10., 20., 30., 200.]]),
        torch.tensor([[6., 10., 14., 300.], [40., 50., 60., 400.]]),
    ]
    # Absolute values catch offsets that a softmax-only comparison would miss.
    expected = [[8., 2., 4.], [30., 10., 20.]] if n_estimators == 1 else [[9., 8., 5.], [40., 35., 30.]]
    pairs = list(zip(dataset.members, raw))
    np.testing.assert_array_equal(classification_logits(pairs), expected)
    np.testing.assert_allclose(classification_probabilities(pairs), softmax(expected, axis=1), rtol=1e-6)

    def predictions(model, items, device):
        for item, values in zip(items, raw):
            yield item, values

    monkeypatch.setattr(prediction_pipeline, "batched_predictions", predictions)
    query = [[0.25, 0.5], [0.75, 1.0]]
    np.testing.assert_array_equal(model.classes_, ["ant", "moose", "zebra"])
    np.testing.assert_array_equal(model.predict_logits(query), expected)
    np.testing.assert_allclose(model.predict_proba(query), softmax(expected, axis=1), rtol=1e-6)


@pytest.mark.parametrize("classes", [1, 2, 3, 10])
@pytest.mark.parametrize("n_estimators", [1, 8])
@pytest.mark.parametrize("cache,retain", [(False, True), (False, False), (True, True), (True, False)])
def test_logits_match_probabilities_and_class_labels(small_checkpoint, classes, n_estimators, cache, retain):
    table, query = tables()
    labels = np.asarray([f"label-{classes - 1 - i % classes:02d}" for i in range(len(table))])
    model = CausiloClassifier(
        n_estimators=n_estimators, device="cpu", use_kv_cache=cache, retain_preprocessing=retain,
    ).fit(table, labels)
    logits = model.predict_logits(query)
    assert isinstance(logits, np.ndarray)
    assert logits.shape == (len(query), classes)
    assert np.isfinite(logits).all()
    np.testing.assert_array_equal(model.classes_, np.unique(labels))
    np.testing.assert_allclose(softmax(logits, axis=1), model.predict_proba(query), atol=1e-7, rtol=1e-6)
    np.testing.assert_array_equal(model.predict(query), model.classes_[logits.argmax(axis=1)])


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_logits_saved_model_roundtrip(small_checkpoint, cache, weighted, monkeypatch):
    table, query = tables()
    counts = np.arange(len(table)) % 4 if weighted else None
    model = CausiloClassifier(n_estimators=4, device="cpu", use_kv_cache=cache).fit(
        table, np.arange(len(table)) % 3, sample_count=counts,
    )
    expected = model.predict_logits(query)
    payload = pickle.dumps(model)
    monkeypatch.setattr(ModelRunner, "build_cache", lambda *a: pytest.fail("Rebuilt K/V cache"))
    restored = pickle.loads(payload)
    np.testing.assert_array_equal(restored.classes_, model.classes_)
    np.testing.assert_array_equal(restored.predict_logits(query), expected)


@pytest.mark.parametrize("classes", [2, 10])
@pytest.mark.parametrize("cache", [False, True])
def test_logits_with_counts_match_expanded_rows(small_checkpoint, classes, cache):
    table, query = tables()
    labels = np.asarray([f"label-{i % classes:02d}" for i in range(len(table))])
    counts = np.arange(len(table)) % 4 + 1
    counts[0] = 0
    labels[0] = "ignored"
    options = dict(n_estimators=8, device="cpu", use_kv_cache=cache, retain_preprocessing=False)
    model = CausiloClassifier(**options).fit(table, labels, sample_count=counts)
    rows = np.repeat(np.arange(len(table)), counts)
    expanded = CausiloClassifier(**options).fit(table.iloc[rows], labels[rows])
    logits = model.predict_logits(query)
    assert logits.shape == (len(query), classes)
    np.testing.assert_array_equal(model.classes_, np.unique(labels[counts > 0]))
    np.testing.assert_allclose(logits, expanded.predict_logits(query), atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(softmax(logits, axis=1), model.predict_proba(query), atol=1e-7, rtol=1e-6)


def test_logits_require_successful_fit(small_checkpoint):
    table, query = tables()
    model = CausiloClassifier(n_estimators=1, device="cpu")
    with pytest.raises(NotFittedError):
        model.predict_logits(query)
    model.fit(table, np.arange(len(table)) % 2)
    with pytest.raises(ValueError):
        model.fit(table, [0])
    with pytest.raises(NotFittedError):
        model.predict_logits(query)


def test_logits_reject_batched_fit(small_checkpoint):
    table, query = tables()
    model = CausiloClassifier(n_estimators=1, device="cpu").fit_batched({
        "table": (table, np.arange(len(table)) % 2),
    })
    with pytest.raises(ValueError, match="fit_batched"):
        model.predict_logits(query)


@pytest.mark.parametrize("query,error", [
    (np.empty((0, 2)), "at least one row"),
    ([[0., 1., 2.]], "feature columns"),
    ([0., 1.], "feature columns"),
    ([[0., np.inf]], "Infinite"),
    (pd.DataFrame({"b": [0.], "a": [1.]}), "name and order"),
])
def test_logits_validate_queries_before_inference(small_checkpoint, query, error, monkeypatch):
    model = CausiloClassifier(n_estimators=1, device="cpu").fit(
        pd.DataFrame({"a": [0., 1., 2.], "b": [2., 1., 0.]}), [0, 1, 0],
    )
    monkeypatch.setattr(prediction_pipeline, "batched_predictions", lambda *a: pytest.fail("Ran inference"))
    for predict in (model.predict_logits, model.predict_proba):
        with pytest.raises(ValueError, match=error):
            predict(query)


@pytest.mark.parametrize("outputs,classes", [(10, 11), (3, 4), (11, 11)])
@pytest.mark.parametrize("cache", [False, True])
def test_logits_reject_unsupported_classes_before_query_processing(
    make_batch_model, outputs, classes, cache, monkeypatch,
):
    monkeypatch.setattr(checkpoints, "load_pretrained_model", lambda task: make_batch_model(task, outputs))
    table = np.random.default_rng(11).normal(size=(2 * classes, 3))
    model = CausiloClassifier(n_estimators=1, device="cpu", use_kv_cache=cache).fit(
        table, np.arange(len(table)) % classes,
    )
    probabilities = model.predict_proba(table[:2])
    assert probabilities.shape == (2, classes)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-7)
    monkeypatch.setattr(model._engine.state.dataset, "query_tables", lambda *a: pytest.fail("Processed query"))
    monkeypatch.setattr(prediction_pipeline, "batched_predictions", lambda *a: pytest.fail("Ran inference"))
    with pytest.raises(NotImplementedError, match=f"at most {min(10, outputs)} classes.*ECOC"):
        model.predict_logits(table[:2])


@pytest.mark.parametrize("cache", [False, True])
def test_logits_preserve_results_across_memory_splits(small_checkpoint, cache, monkeypatch):
    from causilo.execution import batched, cached
    from causilo.execution.direct import BatchPlan, BatchPlanner

    table, _ = tables()
    model = CausiloClassifier(n_estimators=4, device="cpu", use_kv_cache=cache).fit(
        table, np.arange(len(table)) % 3,
    )
    expected = model.predict_logits(table.iloc[:7])
    monkeypatch.setattr(batched, "execution_budget", lambda device: 1)
    if cache:
        monkeypatch.setattr(cached, "execution_budget", lambda device: 64)
        monkeypatch.setattr(cached, "query_memory", lambda config, features, rows, queries, device: 32 * queries)
    else:
        monkeypatch.setattr(BatchPlanner, "choose", lambda *a: BatchPlan(2))
    name = "predict_cached" if cache else "predict"
    original = getattr(ModelRunner, name)
    chunks = []

    def observed(runner, features, context, **kwargs):
        chunks.append(features.shape[1] if cache else features.shape[1] - context.shape[1])
        return original(runner, features, context, **kwargs)

    monkeypatch.setattr(ModelRunner, name, observed)
    np.testing.assert_allclose(model.predict_logits(table.iloc[:7]), expected, atol=2e-5, rtol=2e-5)
    assert max(chunks) <= 2 and sum(chunks) == 4 * 7


@pytest.mark.parametrize("cache", [False, True])
def test_logits_retry_out_of_memory_without_losing_members(small_checkpoint, cache, monkeypatch):
    table, query = tables()
    model = CausiloClassifier(n_estimators=4, device="cpu", use_kv_cache=cache).fit(
        table, np.arange(len(table)) % 3,
    )
    expected = model.predict_logits(query)
    name = "predict_cached" if cache else "predict"
    original = getattr(ModelRunner, name)
    attempts = []

    def fail_once(runner, features, context, **kwargs):
        attempts.append(len(features))
        if len(attempts) == 1:
            raise torch.cuda.OutOfMemoryError("Simulated allocation failure")
        return original(runner, features, context, **kwargs)

    monkeypatch.setattr(ModelRunner, name, fail_once)
    np.testing.assert_allclose(model.predict_logits(query), expected, atol=2e-5, rtol=2e-5)
    assert attempts[0] == model.n_estimators
    assert all(size < attempts[0] for size in attempts[1:])
    assert sum(attempts[1:]) == model.n_estimators


@pytest.mark.pretrained
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("n_estimators", [1, 8])
@pytest.mark.parametrize("classes", [2, 3, 10])
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_pretrained_logits_and_restore(batch_runtime, device, n_estimators, classes, cache, weighted, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    table, query = tables()
    labels = np.asarray([f"label-{classes - 1 - i % classes:02d}" for i in range(len(table))])
    counts = np.arange(len(table)) % 4 + 1 if weighted else None
    if weighted:
        counts[0] = 0
        labels[0] = "ignored"
    model = CausiloClassifier(n_estimators=n_estimators, device=device, use_kv_cache=cache).fit(
        table, labels, sample_count=counts,
    )
    probabilities = model.predict_proba(query)
    logits = model.predict_logits(query)
    assert np.isfinite(logits).all()
    assert logits.shape == (len(query), classes)
    np.testing.assert_array_equal(model.classes_, np.unique(labels[counts > 0] if weighted else labels))
    np.testing.assert_allclose(softmax(logits, axis=1), probabilities, atol=1e-7, rtol=1e-6)
    np.testing.assert_array_equal(model.predict(query), model.classes_[logits.argmax(axis=1)])
    payload = pickle.dumps(model)
    monkeypatch.setattr(ModelRunner, "build_cache", lambda *a: pytest.fail("Rebuilt K/V cache"))
    restored = pickle.loads(payload)
    np.testing.assert_array_equal(restored.predict_logits(query), logits)
