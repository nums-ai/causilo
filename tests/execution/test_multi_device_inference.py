import pickle

import numpy as np
import pytest
import torch
from sklearn.base import clone

from causilo import CausiloClassifier, CausiloRegressor
from causilo.estimators import device_key
from causilo.v1_5 import checkpoints
from causilo.v1_5.data.dataset import PreparedDataset
from causilo.v1_5.engine import resolve_devices
from causilo.v1_5.execution import direct, prediction
from causilo.v1_5.execution.cache import ContextCache
from causilo.v1_5.execution.direct import EnsembleRunner
from causilo.v1_5.execution.parallel import ParallelEnsembleRunner, member_sections


def test_validates_and_orders_requested_cuda_devices(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    devices = resolve_devices(device_key(["cuda:1", "cuda:0", "cuda:2"]))
    assert tuple(str(device) for device in devices) == ("cuda:1", "cuda:0", "cuda:2")
    assert list(member_sections(8, devices, (8, 4, 2))) == [
        (devices[0], slice(0, 5)), (devices[1], slice(5, 7)), (devices[2], slice(7, 8)),
    ]
    assert list(member_sections(1, devices, (8, 4, 2))) == [(devices[0], slice(0, 1))]
    with pytest.raises(ValueError, match="unique"):
        resolve_devices(("cuda", "cuda:0"))
    with pytest.raises(ValueError, match="requires Causilo 1.5"):
        CausiloClassifier(model_version="1.0", device=["cuda:0", "cuda:1"]).fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError, match="CUDA"):
        resolve_devices(("cpu", "cuda:1"))
    with pytest.raises(ValueError, match="CUDA device names"):
        device_key([])


def test_memory_budget_is_cloned_and_validated_for_selected_devices(checkpoint15, mixed_data, monkeypatch):
    X, _ = mixed_data
    model = CausiloClassifier(
        checkpoint_path=checkpoint15("classification"),
        device="cpu", device_memory_budget_bytes={"cuda:0": 1024**3},
    )
    assert clone(model).device_memory_budget_bytes == {"cuda:0": 1024**3}
    with pytest.raises(ValueError, match="selected Causilo"):
        model.fit(X, np.arange(len(X)) % 3)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    model.set_params(device=["cuda:0", "cuda:1"], device_memory_budget_bytes={"cuda:1": True})
    with pytest.raises(ValueError, match="positive integer"):
        model.fit(X, np.arange(len(X)) % 3)
    model.set_params(device_memory_budget_bytes={"cuda:2": 1024**3})
    with pytest.raises(ValueError, match="selected Causilo"):
        model.fit(X, np.arange(len(X)) % 3)
    model.set_params(device_memory_budget_bytes={"cuda:0": 1})
    with pytest.raises(ValueError, match="exceed the model weight size"):
        model.fit(X, np.arange(len(X)) % 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_refit_reuses_restored_model_only_for_same_budget(checkpoint15, mixed_data):
    X, query = mixed_data
    y = np.arange(len(X)) % 3
    initial_budget = {"cuda:0": 1 << 30}
    restored_budget = {"cuda:0": 2 << 30}
    estimator = CausiloClassifier(
        checkpoint_path=checkpoint15("classification"), n_estimators=1, use_kv_cache=True,
        device="cuda:0", device_memory_budget_bytes=initial_budget,
    ).fit(X, y)
    expected = estimator.predict_proba(query)
    estimator.set_params(device_memory_budget_bytes=restored_budget)
    restored = pickle.loads(pickle.dumps(estimator))
    model = restored._engine.model
    assert restored._engine.memory_budgets == restored_budget
    np.testing.assert_array_equal(restored.predict_proba(query), expected)

    restored.fit(X, y)
    assert restored._engine.model is model
    restored.set_params(device_memory_budget_bytes=initial_budget).fit(X, y)
    assert restored._engine.memory_budgets == initial_budget
    assert restored._engine.model is not model


@pytest.fixture
def simulated_devices(monkeypatch):
    # CPU replicas exercise placement and assembly without initializing CUDA.
    devices = (torch.device("cpu"), torch.device("cpu:1"))
    monkeypatch.setattr(
        "causilo.v1_5.engine.resolve_devices",
        lambda request: devices if isinstance(request, tuple) else devices[:1],
    )
    return devices


def test_refit_and_restore_read_one_checkpoint_for_all_active_replicas(
    checkpoint15, mixed_data, monkeypatch, simulated_devices,
):
    load = checkpoints.load_model
    reads = []

    def record(*args, expected_digest):
        reads.append(expected_digest)
        return load(*args, expected_digest=expected_digest)

    monkeypatch.setattr(checkpoints, "load_model", record)
    table, _ = mixed_data
    model = CausiloClassifier(
        checkpoint_path=checkpoint15("classification"), device=["cuda:0", "cuda:1"],
        n_estimators=1, use_kv_cache=False,
    )
    targets = np.arange(len(table)) % 3
    model.fit(table, targets)
    assert reads == [None]
    assert len(model._engine._model_replicas) == 1
    digest = model._engine.checkpoint["digest"]
    model.set_params(n_estimators=8).fit(table, targets)
    assert reads == [None, digest]
    model.fit(table, targets)
    assert reads == [None, digest]
    restored = pickle.loads(pickle.dumps(model))
    assert reads == [None, digest, digest]
    assert len(restored._engine._model_replicas) == len(simulated_devices)


def test_uncached_member_groups_materialize_only_their_own_inputs(checkpoint15, mixed_data, monkeypatch):
    context, query = mixed_data
    model = CausiloClassifier(
        checkpoint_path=checkpoint15("classification"), device="cpu", n_estimators=8, use_kv_cache=False,
    ).fit(context, np.arange(len(context)) % 3)
    expected = model.predict_proba(query)
    signed_views = PreparedDataset.signed_views
    group_sizes = []

    def observe(dataset, table, section):
        view = signed_views(dataset, table, section)
        group_sizes.append(len(view))
        return view

    monkeypatch.setattr(EnsembleRunner, "_ensemble_batch_count", lambda self, shape: shape[0])
    monkeypatch.setattr(PreparedDataset, "signed_views", observe)
    np.testing.assert_allclose(model.predict_proba(query), expected, rtol=2e-6, atol=2e-6)
    assert group_sizes == [1] * 8


def test_restore_keeps_remaining_cache_groups_on_cpu_after_transfer_oom(
    checkpoint15, mixed_data, monkeypatch, simulated_devices,
):
    monkeypatch.setattr(EnsembleRunner, "_ensemble_batch_count", lambda self, shape: shape[0])
    X, query = mixed_data
    model = CausiloClassifier(
        checkpoint_path=checkpoint15("classification"), device="cpu", n_estimators=8, use_kv_cache=True,
    ).fit(X, np.arange(len(X)) % 3)
    expected = model.predict_proba(query)
    saved = model.__getstate__()
    saved["parameters"]["device"] = ["cuda:0", "cuda:1"]
    calls = {0: 0, 1: 0}
    place, move = EnsembleRunner.place_caches, ContextCache.to

    def place_on_simulated_gpu(self, caches):
        lane = EnsembleRunner.__new__(EnsembleRunner)
        lane.device = torch.device("cuda", simulated_devices.index(self.device))
        lane.memory_budget_bytes = None
        placed = place(lane, caches)
        assert all(a is b for a, b in zip(placed[1:], caches[1:], strict=True))
        return placed

    def move_group(self, device):
        lane = torch.device(device).index
        calls[lane] += 1
        if calls[lane] == 2:
            raise torch.cuda.OutOfMemoryError("optional cache transfer")
        return move(self, "cpu")

    monkeypatch.setattr(EnsembleRunner, "place_caches", place_on_simulated_gpu)
    monkeypatch.setattr(ContextCache, "to", move_group)
    monkeypatch.setattr(direct, "available_bytes", lambda device: 1 << 40)
    restored = CausiloClassifier()
    restored.__setstate__(saved)
    assert calls == {0: 2, 1: 2}
    np.testing.assert_allclose(restored.predict_proba(query), expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(("task", "classes", "n_estimators", "use_kv_cache"), [
    ("classification", 10, 1, False),
    ("classification", 10, 8, False),
    ("classification", 10, 8, True),
    ("classification", 12, 8, True),
    ("regression", 0, 8, True),
])
def test_device_group_keeps_fitted_member_and_output_order(
    checkpoint15, mixed_data, task, classes, n_estimators, use_kv_cache, monkeypatch, simulated_devices,
):
    monkeypatch.setattr(
        EnsembleRunner, "member_capacity", lambda self, *_: 4 if self.device == simulated_devices[0] else 2,
    )
    X, query = mixed_data
    if task == "classification":
        constructor, y = CausiloClassifier, np.arange(len(X)) % classes
    else:
        constructor, y = CausiloRegressor, X["a"].to_numpy() ** 2 + 2 * X["b"].to_numpy()
    options = dict(
        checkpoint_path=checkpoint15(task),
        n_estimators=n_estimators, use_kv_cache=use_kv_cache,
    )
    reference = constructor(**options, device="cpu").fit(X, y)
    candidate = constructor(**options, device=["cuda:0", "cuda:1"]).fit(X, y)
    runner = candidate._engine.state.runner
    parallel = n_estimators > 1 and classes <= 10
    assert isinstance(runner, ParallelEnsembleRunner) == parallel

    def collect(estimator):
        if task == "classification":
            return estimator.predict_proba(query)
        return estimator.predict(query, output_type="quantiles", quantiles=[0.02, 0.87, 0.5, 0.02])

    expected, observed = collect(reference), collect(candidate)
    np.testing.assert_allclose(observed, expected, atol=2e-6, rtol=2e-6)
    if task == "classification":
        np.testing.assert_array_equal(candidate.classes_, reference.classes_)
        np.testing.assert_array_equal(candidate.predict(query), reference.predict(query))
    np.testing.assert_array_equal(collect(pickle.loads(pickle.dumps(candidate))), observed)

    if use_kv_cache and parallel:
        # A cache written by a single device must keep the member order after redistribution.
        from_serial = reference.__getstate__()
        from_serial["parameters"]["device"] = ["cuda:0", "cuda:1"]
        monkeypatch.setattr(type(runner), "build_cache", lambda *_: pytest.fail("Expected saved K/V reuse"))
        restored = constructor()
        restored.__setstate__(from_serial)
        np.testing.assert_allclose(collect(restored), expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("use_kv_cache", [False, True])
def test_device_group_retries_only_failed_lane(
    checkpoint15, mixed_data, use_kv_cache, monkeypatch, simulated_devices,
):
    first, second = simulated_devices
    X, query = mixed_data
    model = CausiloClassifier(
        checkpoint_path=checkpoint15("classification"),
        device=["cuda:0", "cuda:1"], use_kv_cache=use_kv_cache, n_estimators=8,
    ).fit(X, np.arange(len(X)) % 3)
    expected = model.predict_proba(query)
    calls = {first: [], second: []}
    max_rows = len(query) // 2
    name = "predict_cached" if use_kv_cache else "predict"
    prediction = getattr(EnsembleRunner, name)

    def predict(self, *args, **kwargs):
        rows = len(args[1]) if use_kv_cache else len(args[1]) - args[2].shape[1]
        calls[self.device].append(rows)
        if self.device == second and rows > max_rows:
            raise torch.cuda.OutOfMemoryError("simulated failure on the second lane")
        return prediction(self, *args, **kwargs)

    monkeypatch.setattr(EnsembleRunner, name, predict)
    np.testing.assert_allclose(model.predict_proba(query), expected, atol=2e-6, rtol=2e-6)
    assert calls == {first: [len(query)], second: [len(query), max_rows, max_rows]}

    calls[first].clear()
    calls[second].clear()
    max_rows = 0
    with pytest.raises(torch.cuda.OutOfMemoryError, match="second lane"):
        model.predict_proba(query)
    assert calls[first] == [len(query)] and calls[second][-1] == 1


@pytest.mark.parametrize("fail_during", ["transform", "join"])
def test_shared_host_preparation_retries_before_dispatch(
    checkpoint15, mixed_data, monkeypatch, simulated_devices, fail_during,
):
    X, query = mixed_data
    model = CausiloClassifier(
        checkpoint_path=checkpoint15("classification"), device=["cuda:0", "cuda:1"],
        n_estimators=8, use_kv_cache=fail_during == "transform",
    ).fit(X, np.arange(len(X)) % 3)
    expected = model.predict_proba(query)
    dataset = model._engine.state.dataset
    query_table, join_context = PreparedDataset.query_table, prediction.join_context
    attempts = []

    def transform(self, table, section, **kwargs):
        size = section.stop - section.start
        attempts.append(size)
        if fail_during == "transform" and size == len(query):
            raise MemoryError("shared query transform")
        return query_table(self, table, section, **kwargs)

    def join(train, chunk):
        if fail_during == "join" and train is dataset.training and len(chunk) == len(query):
            raise MemoryError("shared context and query table")
        return join_context(train, chunk)

    monkeypatch.setattr(PreparedDataset, "query_table", transform)
    monkeypatch.setattr(prediction, "join_context", join)
    np.testing.assert_allclose(model.predict_proba(query), expected, atol=2e-6, rtol=2e-6)
    assert attempts == [len(query), len(query) // 2, len(query) // 2]


def test_parallel_context_replay_is_scoped_to_device_and_prediction(
    checkpoint15, mixed_data, monkeypatch, simulated_devices,
):
    X, query = mixed_data
    model = CausiloRegressor(
        checkpoint_path=checkpoint15("regression"), device=["cuda:0", "cuda:1"], use_kv_cache=False, n_estimators=8,
    ).fit(X, np.arange(len(X), dtype=float))
    expected = model.predict(query)
    monkeypatch.setattr(ParallelEnsembleRunner, "query_batch_size", lambda self, dataset, rows, **kw: 4)
    monkeypatch.setattr(EnsembleRunner, "can_retain_replay", lambda self, dataset: True)
    monkeypatch.setattr(EnsembleRunner, "query_device_limit", lambda self, dataset, rows, **kw: min(2, rows))
    original = EnsembleRunner.predict
    contexts = {device: [] for device in simulated_devices}

    def observe(self, *args, replay_contexts=None, **kwargs):
        assert replay_contexts is not None
        assert replay_contexts.setdefault("device", self.device) == self.device
        contexts[self.device].append(replay_contexts)
        return original(self, *args, replay_contexts=replay_contexts, **kwargs)

    monkeypatch.setattr(EnsembleRunner, "predict", observe)
    for _ in range(2):
        np.testing.assert_allclose(model.predict(query), expected, rtol=2e-6, atol=2e-6)
    for seen in contexts.values():
        middle = len(seen) // 2
        assert middle > 1
        assert all(item is seen[0] for item in seen[:middle])
        assert all(item is seen[middle] for item in seen[middle:])
        assert seen[0] is not seen[middle]
    first, second = simulated_devices
    assert contexts[first][0] is not contexts[second][0]


def test_parallel_cached_window_only_retries_failed_lane(checkpoint15, mixed_data, monkeypatch, simulated_devices):
    first, second = simulated_devices
    X, query = mixed_data
    counts = 1 + np.arange(len(X)) % 3
    model = CausiloClassifier(
        checkpoint_path=checkpoint15("classification"), device=["cuda:0", "cuda:1"], use_kv_cache=True, n_estimators=8,
    ).fit(X, np.arange(len(X)) % 3, sample_count=counts)
    expected = model.predict_proba(query)
    monkeypatch.setattr(ParallelEnsembleRunner, "query_batch_size", lambda self, dataset, rows, **kw: 4)
    monkeypatch.setattr(
        EnsembleRunner, "query_device_limit",
        lambda self, dataset, rows, **kw: min(2 if self.device == second else 4, rows),
    )

    def allowed(self, dataset, caches, batch_size, rows, *, concurrent_lanes=1, **kw):
        assert concurrent_lanes == 2
        return rows

    monkeypatch.setattr(EnsembleRunner, "cached_query_window", allowed)
    original = EnsembleRunner.predict_cached
    windows = {device: [] for device in simulated_devices}
    fallback = {device: [] for device in simulated_devices}

    def attempt(self, view, chunks, caches, **kw):
        assert self.weights is not None
        windows[self.device].append(list(map(len, chunks)))
        if self.device == second:
            raise torch.cuda.OutOfMemoryError("simulated lane window allocation failure")
        return [original(self, view, chunk, caches, **kw) for chunk in chunks]

    def fallback_chunk(self, view, chunk, caches, **kw):
        fallback[self.device].append(len(chunk))
        return original(self, view, chunk, caches, **kw)

    monkeypatch.setattr(EnsembleRunner, "predict_cached_window", attempt)
    monkeypatch.setattr(EnsembleRunner, "predict_cached", fallback_chunk)
    np.testing.assert_allclose(model.predict_proba(query), expected, rtol=2e-6, atol=2e-6)
    assert len(windows[first]) == len(windows[second]) == 1
    assert max(windows[first][0]) == 4 and max(windows[second][0]) == 2
    assert fallback == {first: [], second: windows[second][0]}
