import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.exceptions import NotFittedError

from causilo import CausiloRegressor, checkpoints
from causilo import engine as engine_module
from causilo.execution.runner import ModelRunner
from causilo.quantiles import interpolate_quantiles, validate_quantiles


@pytest.fixture
def known_head_outputs(monkeypatch):
    pretrained = SimpleNamespace(config=SimpleNamespace(outputs=3))
    pretrained.to = lambda device: pretrained
    monkeypatch.setattr(checkpoints, "load_pretrained_model", lambda task: pretrained)

    def predictions(model, fitted, transformed, device, reduce_output):
        # Crossed member outputs distinguish sorting each member from sorting the average.
        raw = [[[9.0, -1.0, 1.0], [4.0, 0.0, 2.0]], [[0.0, 4.0, 2.0], [6.0, 0.0, -6.0]]]
        for member, values in zip(fitted.members_by_normalization(), raw):
            yield member, reduce_output(torch.tensor(values))

    monkeypatch.setattr(engine_module, "direct_predictions", predictions)
    return CausiloRegressor(n_estimators=2, device="cpu").fit([[0.0], [1.0]], [10.0, 14.0])


def test_output_modes_and_requested_order(known_head_outputs):
    model = known_head_outputs
    query = [[0.25], [0.75]]
    raw = model.predict(query, output_type='raw')
    np.testing.assert_array_equal(raw, [[11, 15, 25], [6, 14, 22]])
    np.testing.assert_array_equal(model.predict(query), [17, 14])
    np.testing.assert_array_equal(model.predict(query, output_type='mean'), [17, 14])
    np.testing.assert_array_equal(
        model.predict(query, output_type='quantiles', quantiles=[.625, .25, .5, .25]),
        [[20, 11, 15, 11], [18, 6, 14, 6]],
    )
    assert model.predict(query, output_type='quantiles', quantiles=[.5]).shape == (2, 1)
    median = model.predict(query, output_type='median')
    assert median.shape == (2,)
    np.testing.assert_array_equal(median, [15, 14])
    np.testing.assert_array_equal(
        median, model.predict(query, output_type='quantiles', quantiles=[.5])[:, 0])


@pytest.mark.parametrize('options', [
    {'output_type': 'invalid'}, {'output_type': True},
    *[{'output_type': mode, 'quantiles': [.5]} for mode in ('mean', 'median', 'raw')],
    *[{'output_type': 'quantiles', 'quantiles': q} for q in (None, [], .5, [[.5]], [0], [1], [np.nan], ['.5'])],
])
def test_invalid_requests_fail_before_inference(known_head_outputs, options, monkeypatch):
    model = known_head_outputs
    monkeypatch.setattr(model._engine, 'predict_raw', lambda X: pytest.fail('Ran inference'))
    monkeypatch.setattr(model._engine, 'predict', lambda X: pytest.fail('Ran inference'))
    with pytest.raises(ValueError):
        model.predict([[0.0]], **options)


def test_outputs_require_successful_fit(known_head_outputs):
    fitted = known_head_outputs
    with pytest.raises(ValueError):
        fitted.fit([[0.0], [1.0]], [10.0])
    for model in [CausiloRegressor(), fitted]:
        for mode in ('mean', 'median', 'raw', 'quantiles'):
            levels = [.5] if mode == 'quantiles' else None
            with pytest.raises(NotFittedError):
                model.predict([[0.0]], output_type=mode, quantiles=levels)


def test_native_knots_and_exponential_reference():
    native = np.arange(1, 1000) / 1000
    requests = validate_quantiles([1e-8, 1e-4, *native, .9999, 1 - 1e-8])
    for inverse, side in [
        (lambda p: 7 + 2 * np.log(p), slice(0, 2)),
        (lambda p: 7 - 2 * np.log1p(-p), slice(-2, None)),
    ]:
        q = inverse(native)[None, :]
        result = interpolate_quantiles(q, requests)
        np.testing.assert_array_equal(result[:, 2:-2], q)
        np.testing.assert_allclose(result[0, side], inverse(requests[side]), rtol=1e-12, atol=1e-12)
        assert np.all(np.diff(result) >= 0)


def test_ties_extreme_probabilities_and_target_units():
    requests = validate_quantiles([np.nextafter(0., 1.), .01, .5, .99, np.nextafter(1., 0.)])
    q = np.array([[1, 1, 2, 3, 3], [7, 7, 7, 7, 7]], dtype=float)
    result = interpolate_quantiles(q, requests)
    np.testing.assert_array_equal(result, [[1, 1, 2, 3, 3], [7]*5])
    q = np.array([[-2, -1, 0, 3, 5]], dtype=float)
    np.testing.assert_allclose(interpolate_quantiles(q*4+1e9, requests),
                               interpolate_quantiles(q, requests)*4+1e9, rtol=0, atol=2e-7)
    with pytest.raises(ValueError, match='nonfinite'):
        interpolate_quantiles(np.array([[-1e308, 1e308]]), requests)


@pytest.mark.pretrained
@pytest.mark.parametrize('cache,retain', [(False, False), (True, True)])
def test_pretrained_outputs_and_restore(cache, retain, monkeypatch):
    torch.set_num_threads(2)
    rng = np.random.default_rng(51)
    table = pd.DataFrame({'kind': ['red', 'blue', None]*8, 'amount': rng.normal(size=24)})
    query = pd.DataFrame({'kind': ['new', None, 'red'], 'amount': [.25, np.nan, -.5]})
    model = CausiloRegressor(n_estimators=4, device='cpu', use_kv_cache=cache,
                            retain_preprocessing=retain).fit(table, 1e9 + 20*rng.normal(size=24))
    point = model.predict(query)
    raw = model.predict(query, output_type='raw')
    levels = np.r_[.0001, np.arange(1, 1000)/1000, .9999]
    selected = model.predict(query, output_type='quantiles', quantiles=levels)
    assert raw.shape == (3, 999) and selected.shape == (3, 1001)
    assert raw.dtype == selected.dtype == np.float64
    np.testing.assert_array_equal(selected[:, 1:-1], raw)
    median = model.predict(query, output_type='median')
    assert median.shape == (3,)
    np.testing.assert_array_equal(median, selected[:, 500])
    np.testing.assert_array_equal(model.predict(query), point)
    np.testing.assert_allclose(raw.mean(axis=1), point, atol=1e-5, rtol=0)
    with pytest.raises(ValueError, match='name and order'):
        model.predict(query[['amount', 'kind']], output_type='raw')
    saved = pickle.dumps(model)
    monkeypatch.setattr(ModelRunner, 'build_cache', lambda *args: pytest.fail('Rebuilt K/V cache'))
    restored = pickle.loads(saved)
    np.testing.assert_array_equal(restored.predict(query, output_type='quantiles', quantiles=levels), selected)
    np.testing.assert_array_equal(restored.predict(query, output_type='median'), median)


@pytest.mark.pretrained
@pytest.mark.parametrize('cache', [False, True])
def test_pretrained_output_partitioning(cache, monkeypatch):
    from causilo.execution import cached
    from causilo.execution.direct import BatchPlan, BatchPlanner

    torch.set_num_threads(2)
    rng = np.random.default_rng(71)
    table = rng.normal(size=(24, 5))
    model = CausiloRegressor(n_estimators=4, device='cpu', use_kv_cache=cache).fit(
        table, 10 + 2*rng.normal(size=24))
    options = dict(output_type='quantiles', quantiles=[1e-6, .05, .5, .95, 1-1e-6])
    expected = model.predict(table[:7], **options)
    if cache:
        monkeypatch.setattr(cached, 'execution_budget', lambda device: 64)
        monkeypatch.setattr(cached, 'query_memory', lambda shape, features, training_rows, queries, device: 32*queries)
    else:
        monkeypatch.setattr(BatchPlanner, 'choose', lambda *args: BatchPlan(1, 2))
    np.testing.assert_allclose(model.predict(table[:7], **options), expected, atol=2e-3, rtol=2e-4)
