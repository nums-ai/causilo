from contextlib import nullcontext
from dataclasses import replace

import numpy as np
import pytest
import torch

from causilo import CausiloClassifier
from causilo.v1_0.ecoc import ECOCCodec
from causilo.v1_5.data.dataset import PreparedDataset
from causilo.v1_5.execution import direct
from causilo.v1_5.execution.cache import ContextCache


def test_many_class_predictions_use_one_member_per_code(checkpoint15, mixed_data):
    table, query = mixed_data
    labels = np.array([f"class {index % 11}" for index in range(len(table))])
    options = dict(
        device="cpu", checkpoint_path=checkpoint15("classification"), use_kv_cache=True,
    )
    light = CausiloClassifier(n_estimators=1, **options).fit(table, labels)
    standard = CausiloClassifier(n_estimators=8, **options).fit(table, labels)

    state = standard._engine.state
    codebook = ECOCCodec(11, 10, 42).codebook
    np.testing.assert_array_equal(state.dataset.codec.codebook, codebook)
    assert state.dataset.ensemble.feature_signs.shape[0] == 1
    assert state.dataset.num_classes == 10
    assert len(state.caches) == len(codebook)
    assert all(context.context_shape[0] == 1 for context in state.caches)

    np.testing.assert_array_equal(standard.classes_, np.unique(labels))
    probabilities = standard.predict_proba(query)
    np.testing.assert_array_equal(probabilities, light.predict_proba(query))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_array_equal(standard.predict(query), standard.classes_[probabilities.argmax(axis=1)])
    with pytest.raises(NotImplementedError, match="ECOC logits"):
        standard.predict_logits(query)


@pytest.mark.parametrize("seed", [None, 2**64 + 42])
def test_codebook_accepts_unset_and_large_seeds(mixed_data, seed):
    table, _ = mixed_data
    labels = np.arange(len(table)) % 11
    options = dict(task="classification", n_estimators=8, random_state=seed)
    dataset = PreparedDataset.prepare(table, labels, **options)
    np.testing.assert_array_equal(dataset.targets, labels)
    for row, expected in enumerate(dataset.codec.encode(labels)):
        np.testing.assert_array_equal(dataset.targets_for(slice(None), code_row=row), expected[None, :])
    if seed is not None:
        replay = PreparedDataset.prepare(table, labels, **options)
        np.testing.assert_array_equal(dataset.codec.codebook, replay.codec.codebook)


def test_ecoc_prefill_shares_the_cache_budget(checkpoint15, mixed_data, monkeypatch):
    table, _ = mixed_data
    model = CausiloClassifier(device="cpu", checkpoint_path=checkpoint15("classification"))
    model.fit(table, np.arange(len(table)) % 11)
    state = model._engine.state
    runner = state.runner
    runner.device = torch.device("cuda:0")
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: None)
    monkeypatch.setattr(direct, "available_bytes", lambda _: 32)
    prepared, spilled = [], []

    def prefill(dataset, train, section):
        assert dataset is state.dataset
        assert (section.start, section.stop) == (len(prepared), len(prepared) + 1)
        cache = ContextCache((1, *train.shape), dataset.num_classes, column_ecdf_context=torch.zeros(1))
        prepared.append(cache)
        return cache

    move = ContextCache.to

    def to_host(cache, device):
        assert device == "cpu"
        spilled.append(cache)
        return move(cache, device)

    monkeypatch.setattr(runner, "_prefill", prefill)
    monkeypatch.setattr(ContextCache, "to", to_host)
    caches = runner.build_cache(state.dataset)
    assert len(caches) == state.dataset.code_rows
    assert caches[0] is prepared[0] and caches[1] is prepared[1]
    assert all(left is right for left, right in zip(spilled, prepared[2:], strict=True))


def test_mixed_class_batch_uses_per_table_member_counts(checkpoint15, mixed_data, tmp_path):
    table, query = mixed_data
    labels = np.arange(len(table))
    options = dict(device="cpu", checkpoint_path=checkpoint15("classification"), use_kv_cache=True)
    fitted = CausiloClassifier(n_estimators=8, **options).fit_batched({
        "native": (table, labels % 10),
        "ecoc": (table, labels % 11),
    })
    queries = {"ecoc": query, "native": query}
    probabilities = fitted.predict_proba_batched(queries)
    for key, classes, members in (("native", 10, 8), ("ecoc", 11, 1)):
        single = CausiloClassifier(n_estimators=members, **options).fit(table, labels % classes)
        np.testing.assert_array_equal(probabilities[key], single.predict_proba(query))

    destination = tmp_path / "context.pkl"
    fitted.save_context(destination)
    restored = CausiloClassifier.load_context(destination, device="cpu")
    for key, values in restored.predict_proba_batched(queries).items():
        np.testing.assert_array_equal(values, probabilities[key])

    saved = fitted.__getstate__()
    context = saved["fitted"]["ecoc"]
    saved["fitted"]["ecoc"] = replace(context, caches=context.caches[:-1])
    with pytest.raises(ValueError, match="one context per ECOC code row"):
        CausiloClassifier().__setstate__(saved)


@pytest.mark.parametrize("cached", [False, True])
def test_ecoc_query_window_keeps_code_targets_separate(checkpoint15, mixed_data, monkeypatch, cached):
    table, query = mixed_data
    model = CausiloClassifier(
        device="cpu", checkpoint_path=checkpoint15("classification"), n_estimators=8, use_kv_cache=cached,
    ).fit(table, np.arange(len(table)) % 11, sample_count=1 + np.arange(len(table)) % 3)
    state = model._engine.state
    runner = state.runner
    monkeypatch.setattr(runner, "query_batch_size", lambda *args, **kwargs: 3)
    expected = model.predict_proba(query)
    prefills = []
    original = runner._prefill

    def track(dataset, train, section):
        prefills.append(section.start)
        return original(dataset, train, section)

    monkeypatch.setattr(runner, "_prefill", track)
    monkeypatch.setattr(runner, "_cache_replay_ceiling", lambda: 2**60)
    monkeypatch.setattr(runner, "_working_memory", lambda: 0)
    monkeypatch.setattr(runner, "code_query_window", lambda *args: len(query))
    actual = model.predict_proba(query)
    assert prefills == ([] if cached else list(range(state.dataset.code_rows)))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)


def test_ecoc_partial_window_retries_only_uncommitted_queries(checkpoint15, mixed_data, monkeypatch):
    table, query = mixed_data
    model = CausiloClassifier(
        device="cpu", checkpoint_path=checkpoint15("classification"), use_kv_cache=True,
    ).fit(table, np.arange(len(table)) % 11)
    runner = model._engine.state.runner
    monkeypatch.setattr(runner, "query_batch_size", lambda *args, **kwargs: 2)
    expected = model.predict_proba(query)
    monkeypatch.setattr(runner, "code_query_window", lambda *args: 4)
    monkeypatch.setattr(runner, "_cache_replay_ceiling", lambda: 2**60)
    monkeypatch.setattr(runner, "_working_memory", lambda: 0)
    code_rows = model._engine.state.dataset.code_rows
    group_window = runner._cached_group_window
    per_chunk = runner.predict_codes
    windows, retried = [], []

    def fail_mid_codebook(*args):
        windows.append(1)
        if len(windows) == code_rows + 2:
            raise MemoryError("second window exhausted its budget after one code")
        return group_window(*args)

    def track_retry(dataset, values, caches):
        retried.append(len(values))
        return per_chunk(dataset, values, caches)

    monkeypatch.setattr(runner, "_cached_group_window", fail_mid_codebook)
    monkeypatch.setattr(runner, "predict_codes", track_retry)
    np.testing.assert_array_equal(model.predict_proba(query), expected)
    assert len(windows) == code_rows + 2
    assert sum(retried) == len(query) - 4


def test_ecoc_reuse_respects_host_budget(checkpoint15, mixed_data, monkeypatch):
    table, _ = mixed_data
    model = CausiloClassifier(device="cpu", checkpoint_path=checkpoint15("classification"), use_kv_cache=False)
    model.fit(table, np.arange(len(table)) % 11)
    state = model._engine.state
    runner = state.runner
    monkeypatch.setattr(runner, "device", torch.device("cuda:0"))
    narrow = replace(state.dataset, training=np.empty((10_000, 4), dtype=np.float32))
    wide = replace(state.dataset, training=np.empty((32, 128), dtype=np.float32))
    monkeypatch.setattr(direct, "available_bytes", lambda _: 1)
    assert runner.code_query_window(narrow, None, 2, 10) == 0
    monkeypatch.setattr(direct, "available_bytes", lambda _: 2**40)
    assert runner.code_query_window(wide, None, 2, 10) == 0
    assert runner.code_query_window(narrow, None, 2, 10) == 10
    assert runner.code_query_window(narrow, None, 10, 10) == 0
    # Fitted projections have already been built, regardless of the table shape.
    monkeypatch.setattr(runner._model_runner, "host_prediction_bytes", lambda _: 1)
    assert runner.code_query_window(wide, (object(),), 2, 10) == 10
