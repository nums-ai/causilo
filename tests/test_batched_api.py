import io
import pickle
import weakref
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from causilo import CausiloClassifier, CausiloRegressor, checkpoints
from causilo import batch as batch_api
from causilo.execution.runner import ModelRunner
from causilo.model import Model, ModelConfig


@pytest.fixture
def small_checkpoints(monkeypatch):
    calls = []

    def load(task):
        calls.append(task)
        with torch.random.fork_rng():
            torch.manual_seed(34)
            model = Model(ModelConfig(
                task=task, width=8, expansion=2, group_size=3, frequencies=3,
                column_latents=3, column_heads=2, column_depths=(2, 1),
                row_heads=2, row_latents=2, row_depths=(2, 2),
                prediction_heads=2, prediction_depth=2,
                outputs=10 if task == "classification" else 999,
            ))
            for name, parameter in model.named_parameters():
                if "latents" in name or name.endswith(("frequencies", "missing")):
                    torch.nn.init.normal_(parameter, std=0.1)
        return model.eval()

    monkeypatch.setattr(checkpoints, "load_pretrained_model", load)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield calls
    torch.set_num_threads(previous)


def data():
    rng = np.random.default_rng(62)
    a = pd.DataFrame(rng.normal(size=(17, 4)), columns=list("abcd"))
    a["kind"] = ["first", "second", None] * 5 + ["first", "second"]
    a["constant"] = 1
    a.loc[2, "a"] = np.nan
    b = rng.normal(size=(23, 8))
    b[3, 4] = np.nan
    return a, b


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("retain", [False, True])
def test_classifier_ragged_contexts_match_independent_fits(small_checkpoints, cache, retain):
    a, b = data()
    ya = np.asarray(["yes" if i % 2 else "no" for i in range(len(a))])
    yb = np.arange(len(b)) % 13  # independent many-class ECOC context
    params = dict(n_estimators=2, device="cpu", use_kv_cache=cache, retain_preprocessing=retain)
    expected = {
        "a": CausiloClassifier(**params).fit(a, ya).predict_proba(a.iloc[:3]),
        "b": CausiloClassifier(**params).fit(b, yb).predict_proba(b[:5]),
    }
    model = CausiloClassifier(**params)
    assert model.fit_batched({"a": (a, ya), "b": (b, yb)}) is model
    actual = model.predict_proba_batched({"b": b[:5], "a": a.iloc[:3]})
    assert list(actual) == ["b", "a"]
    assert not hasattr(model, "classes_")
    assert not hasattr(model, "n_features_in_")
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(actual[key].sum(axis=1), 1)
    predicted = model.predict_batched({"a": a.iloc[:3]})
    np.testing.assert_array_equal(predicted["a"], model.classes_batched_["a"][actual["a"].argmax(axis=1)])
    assert model.predict_batched({}) == {}
    assert len(small_checkpoints) == 3  # one model load per estimator, not per table


@pytest.mark.parametrize("cache", [False, True])
def test_regressor_mixed_outputs_scales_and_batch_composition(small_checkpoints, cache):
    a, b = data()
    ya = 1e9 + np.arange(len(a), dtype=float) / 13
    yb = -300 + np.arange(len(b), dtype=float) * 17
    params = dict(n_estimators=4, device="cpu", use_kv_cache=cache)
    single_a = CausiloRegressor(**params).fit(a, ya)
    single_b = CausiloRegressor(**params).fit(b, yb)
    model = CausiloRegressor(**params).fit_batched({"a": (a, ya), "b": (b, yb)})
    levels = [0.9, 0.1, 0.1, 0.0005]
    actual = model.predict_batched({
        "a": a.iloc[:3],
        "b": {"X": b[:5], "output_type": "quantiles", "quantiles": levels},
    })
    np.testing.assert_allclose(actual["a"], single_a.predict(a.iloc[:3]), atol=2e-4, rtol=1e-10)
    np.testing.assert_allclose(actual["b"], single_b.predict(b[:5], output_type="quantiles", quantiles=levels),
                               atol=2e-4, rtol=2e-5)
    for output in ("median", "raw"):
        alone = model.predict_batched({"b": {"X": b[:2], "output_type": output}})["b"]
        np.testing.assert_allclose(alone, single_b.predict(b[:2], output_type=output), atol=2e-4, rtol=2e-5)


@pytest.mark.parametrize("output_type", ["mean", "median", "quantiles", "raw"])
def test_completed_tables_release_members_and_preserve_sum_order(small_checkpoints, monkeypatch, output_type):
    x = np.arange(12).reshape(6, 2)
    y = np.tile([-1.0, 1.0], 3)  # identity target scaling
    model = CausiloRegressor(device="cpu", n_estimators=4).fit_batched({
        key: (x, y) for key in ("a", "b", "c")
    })
    references = {key: [] for key in ("a", "b", "c")}
    # This order interleaves tables and deliberately changes floating-point
    # cancellation unless aggregation restores fitted member order.
    order = [("a", 0), ("b", 2), ("a", 2), ("b", 0), ("a", 1), ("a", 3),
             ("b", 1), ("b", 3), ("c", 3), ("c", 2), ("c", 1), ("c", 0)]

    def predictions(model, items, device):
        items = {item.key: item for item in items}
        for key in order:
            item = items[key]
            shape = (item.shape[1],) if output_type == "mean" else (item.shape[1], 999)
            value = torch.full(shape, [1e20, 1, -1e20, 3][key[1]], dtype=torch.float32)
            references[key[0]].append(weakref.ref(value))
            yield item, value
            del value
            if key == ("a", 3):
                assert all(reference() is None for reference in references["a"])
                assert all(reference() is not None for reference in references["b"])
            if key == ("b", 3):
                assert all(reference() is None for reference in references["b"])

    monkeypatch.setattr(batch_api, "batched_predictions", predictions)
    options = {"output_type": output_type}
    if output_type == "quantiles":
        options["quantiles"] = [0.9, 0.1]
    results = model.predict_batched({key: {"X": x[:rows], **options} for key, rows in (("c", 1), ("b", 3), ("a", 2))})
    assert list(results) == ["c", "b", "a"]
    for result in results.values():
        np.testing.assert_array_equal(result, np.full(result.shape, 0.75))
        assert result.base is None  # point outputs must not retain 999 quantiles
    assert all(reference() is None for refs in references.values() for reference in refs)


@pytest.mark.parametrize("constructor", [CausiloClassifier, CausiloRegressor])
def test_mode_changes_failed_refits_and_clone(small_checkpoints, constructor):
    a, _ = data()
    y = np.arange(len(a)) % 3
    model = constructor(device="cpu", n_estimators=1).fit(a, y)
    model.fit_batched({"a": (a, y)})
    with pytest.raises(ValueError, match="fit_batched"):
        model.predict(a.iloc[:1])
    cloned = clone(model)
    assert len(small_checkpoints) == 1
    with pytest.raises(NotFittedError):
        cloned.predict_batched({"a": a.iloc[:1]})
    with pytest.raises(ValueError, match="bad"):
        model.fit_batched({"a": (a, y), "bad": (a, y[:-1])})
    with pytest.raises(NotFittedError):
        model.predict_batched({"a": a.iloc[:1]})
    assert not hasattr(model, "classes_batched_")
    model.fit_batched({"a": (a, y)}).fit(a, y)
    with pytest.raises(ValueError, match="fit_batched"):
        model.predict_batched({"a": a.iloc[:1]})
    assert not hasattr(model, "n_features_in_batched_")
    model.fit_batched({"a": (a, y)})
    model.set_params(device="xpu")
    with pytest.raises(ValueError):
        model.fit_batched({"a": (a, y)})
    with pytest.raises(NotFittedError):
        model.predict_batched({"a": a.iloc[:1]})


@pytest.mark.parametrize("constructor", [CausiloClassifier, CausiloRegressor])
def test_failed_cache_fit_never_publishes_partial_batch(small_checkpoints, monkeypatch, constructor):
    _, x = data()
    y = np.arange(len(x)) % 3
    model = constructor(device="cpu", n_estimators=2, use_kv_cache=True).fit(x, y)
    build_cache = ModelRunner.build_cache
    calls = 0

    def fail_second_table(runner, table, targets):
        nonlocal calls
        calls += 1
        assert model._engine.state is None
        assert model._engine.batch_states is None
        if calls == 3:
            raise RuntimeError("injected cache failure")
        return build_cache(runner, table, targets)

    monkeypatch.setattr(ModelRunner, "build_cache", fail_second_table)
    with pytest.raises(RuntimeError, match="injected cache failure"):
        model.fit_batched({"a": (x, y), "b": (x, y)})
    assert calls == 3
    for method, query in ((model.predict, x[:1]), (model.predict_batched, {"a": x[:1]})):
        with pytest.raises(NotFittedError):
            method(query)
    assert not hasattr(model, "n_features_in_")
    assert not hasattr(model, "n_features_in_batched_")


@pytest.mark.parametrize("cache", [False, True])
def test_independent_ecoc_decoders_with_different_round_counts(small_checkpoints, cache):
    _, x = data()
    x = np.tile(x, (4, 1))
    params = dict(device="cpu", n_estimators=2, use_kv_cache=cache)
    tables = {"a": (x, np.arange(len(x)) % 11), "b": (x, np.arange(len(x)) % 73)}
    queries = {"b": x[:3], "a": x[:2]}
    expected = {
        key: CausiloClassifier(**params).fit(*tables[key]).predict_proba(query)
        for key, query in queries.items()
    }
    model = CausiloClassifier(**params).fit_batched(tables)
    assert len(model._engine.batch_states["a"].code_datasets) != len(model._engine.batch_states["b"].code_datasets)
    actual = model.predict_proba_batched(queries)
    for key in queries:
        np.testing.assert_allclose(actual[key], expected[key], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("tables", [{}, [], {1: ([], [])}, {"": ([], [])}, {"a": ([],)}])
def test_invalid_fit_input_does_not_load_weights(small_checkpoints, tables):
    with pytest.raises(ValueError):
        CausiloRegressor().fit_batched(tables)
    assert small_checkpoints == []


@pytest.mark.parametrize("options", [
    {"output_type": "mean"}, {"X": [[1]], "output_type": "invalid"},
    {"X": [[1]], "output_type": "quantiles"},
    {"X": [[1]], "quantiles": [0.5]},
    {"X": [[1]], "output_type": "quantiles", "quantiles": [0]},
    {"X": [[1]], "device": "cpu"},
])
def test_bad_options_validate_before_execution(small_checkpoints, monkeypatch, options):
    a, _ = data()
    model = CausiloRegressor(device="cpu", n_estimators=1).fit_batched({"a": (a, np.arange(len(a)))})
    monkeypatch.setattr(ModelRunner, "predict", lambda *a, **kw: pytest.fail("Ran invalid query"))
    with pytest.raises(ValueError, match="Table 'a'"):
        model.predict_batched({"a": options})
    with pytest.raises(ValueError, match="unknown"):
        model.predict_batched({"unknown": a.iloc[:1]})
    with pytest.raises(ValueError, match="Table 'a'"):
        model.predict_batched({"a": a.iloc[:1, ::-1]})
    assert model._engine.batch_states is not None


def test_narrow_dataframe_error_names_numpy_fitted_table(small_checkpoints):
    _, b = data()
    model = CausiloRegressor(device="cpu", n_estimators=1).fit_batched({"wide": (b, np.arange(len(b)))})
    with pytest.raises(ValueError, match="Table 'wide'"):
        model.predict_batched({"wide": pd.DataFrame(b[:2, :1])})


def test_restore_validates_every_table_cache(small_checkpoints):
    a, b = data()
    model = CausiloRegressor(device="cpu", n_estimators=1, use_kv_cache=True).fit_batched({
        "a": (a, np.arange(len(a))), "b": (b, np.arange(len(b))),
    })
    saved = model.__getstate__()
    state = saved["fitted"]["b"]
    saved["fitted"]["b"] = replace(state, caches=())
    with pytest.raises(ValueError, match="ensemble count"):
        CausiloRegressor().__setstate__(saved)


@pytest.mark.parametrize("constructor", [CausiloClassifier, CausiloRegressor])
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("serializer", [pickle, joblib])
def test_saved_batch_shares_weights_and_restores_without_fit(small_checkpoints, monkeypatch, constructor, cache, serializer):
    a, b = data()
    model = constructor(device="cpu", n_estimators=2, use_kv_cache=cache).fit_batched({
        "a": (a, np.arange(len(a)) % 3), "b": (b, np.arange(len(b)) % 11),
    })
    query = {"a": a.iloc[:2], "b": b[:3]}
    method = "predict_proba_batched" if constructor is CausiloClassifier else "predict_batched"
    expected = getattr(model, method)(query)
    payload = io.BytesIO()
    serializer.dump(model, payload)
    monkeypatch.setattr(ModelRunner, "build_cache", lambda *a, **kw: pytest.fail("Rebuilt K/V"))
    monkeypatch.setattr(constructor, "fit_batched", lambda *a, **kw: pytest.fail("Refitted"))
    payload.seek(0)
    restored = serializer.load(payload)
    actual = getattr(restored, method)(query)
    for key in query:
        np.testing.assert_array_equal(expected[key], actual[key])
    assert len(small_checkpoints) == 2
    assert restored._engine.state is None


@pytest.mark.pretrained
@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize("constructor", [CausiloClassifier, CausiloRegressor])
@pytest.mark.parametrize("cache", [False, True])
def test_real_checkpoint_batched_parity(device, constructor, cache):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        a, b = data()
        tables = {"a": (a, np.arange(len(a)) % 3), "b": (b, np.arange(len(b)) % 11)}
        queries = {"a": a.iloc[:2], "b": b[:3]}
        params = dict(device=device, n_estimators=2, use_kv_cache=cache)
        expected = {}
        for key, (X, y) in tables.items():
            single = constructor(**params).fit(X, y)
            expected[key] = (single.predict_proba if constructor is CausiloClassifier else single.predict)(queries[key])
            del single
        model = constructor(**params).fit_batched(tables)
        method = model.predict_proba_batched if constructor is CausiloClassifier else model.predict_batched
        actual = method(queries)
        for key in queries:
            np.testing.assert_allclose(actual[key], expected[key], atol=5e-4, rtol=5e-4)
    finally:
        torch.set_num_threads(previous)
