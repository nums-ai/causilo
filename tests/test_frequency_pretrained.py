"""Expanded-row oracle checks against the pinned official checkpoints."""

import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from causilo import CausiloClassifier, CausiloRegressor
from causilo.execution.direct import BatchPlan, BatchPlanner

pytestmark = pytest.mark.pretrained


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("kind", [CausiloClassifier, CausiloRegressor])
@pytest.mark.parametrize("cache", [False, True])
def test_official_frequency_fit_matches_expansion(kind, cache, monkeypatch):
    rng = np.random.default_rng(824)
    table = pd.DataFrame(rng.normal(size=(14, 5)), columns=list("abcde"))
    table.loc[3, "a"] = np.nan
    table["category"] = ["a", "b", None, "b", "c", "a", "a"] * 2
    weights = np.array([1, 3, 2, 4, 5, 1, 2, 4, 0, 3, 1, 2, 1, 3])
    labels = np.arange(14) % 3 if kind is CausiloClassifier else rng.normal(size=14) * 5 + 10
    indices = np.repeat(np.arange(14), weights)
    query = table.iloc[[2, 3, 6]].copy()
    options = dict(n_estimators=8, device="cpu", use_kv_cache=cache)
    expanded = kind(**options).fit(table.iloc[indices], labels[indices])
    weighted = kind(**options).fit(table, labels, sample_weight=weights)

    def predict(model):
        return model.predict_proba(query) if kind is CausiloClassifier else model.predict(query, output_type="raw")

    expected = predict(expanded)
    np.testing.assert_allclose(predict(weighted), expected, atol=2e-4, rtol=2e-4)
    restored = pickle.loads(pickle.dumps(weighted))
    np.testing.assert_array_equal(predict(restored), predict(weighted))
    if not cache:
        monkeypatch.setattr(BatchPlanner, "choose", lambda *args: BatchPlan(1, 2, mode="recompute"))
        np.testing.assert_allclose(predict(weighted), expected, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("tie", [0.3, np.nextafter(0.3, 1.0)])
@pytest.mark.parametrize("cache", [False, True])
def test_official_missing_mean_rank_ties_match_expansion(tie, cache, monkeypatch):
    table = np.array([np.nan, .5, 0., tie, tie, np.nan])[:, None]
    weights = np.array([8, 6, 4, 1, 5, 1])
    labels = np.array([0, 1, 0, 1, 1, 0])
    query = np.array([[0.], [.2], [tie], [.5], [np.nan]])
    options = dict(n_estimators=4, device="cpu", use_kv_cache=cache)
    expanded = CausiloClassifier(**options).fit(
        np.repeat(table, weights, axis=0), np.repeat(labels, weights)
    )
    weighted = CausiloClassifier(**options).fit(table, labels, sample_weight=weights)
    expected = expanded.predict_proba(query)
    np.testing.assert_allclose(weighted.predict_proba(query), expected, atol=2e-5, rtol=2e-5)
    if not cache:
        monkeypatch.setattr(BatchPlanner, "choose", lambda *args: BatchPlan(1, 2, mode="recompute"))
        np.testing.assert_allclose(weighted.predict_proba(query), expected, atol=2e-5, rtol=2e-5)
