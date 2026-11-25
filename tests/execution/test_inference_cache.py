import pickle
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import joblib
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from causilo import CausiloClassifier, CausiloRegressor
from causilo.model_selection import read_local_checkpoint
from causilo.v1_5.checkpoints import load_checkpoint
from causilo.v1_5.execution import precision
from causilo.v1_5.execution.direct import EnsembleRunner
from causilo.v1_5.execution.memory import Stage
from causilo.v1_5.execution.predictor import PredictionRunner
from causilo.v1_5.execution.recompute import RecomputeRunner
from causilo.v1_5.execution.runner import ModelRunner
from causilo.v1_5.nn.cache import KVPair
from causilo.v1_5.nn.column import ColumnLayer
from causilo.v1_5.nn.model_config import ROW_HEADS, ROW_LATENTS, ModelConfig
from causilo.v1_5.nn.prediction import PredictionBlock
from causilo.v1_5.nn.row import RowLayer

cpu_and_cuda = [
    "cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")),
]


@pytest.mark.parametrize("device", cpu_and_cuda)
def test_packed_and_bounded_row_updates_match_full_broadcast(device):
    width = 256 if device == "cuda" else 32
    layer = RowLayer(width, ROW_HEADS, final=False).to(device).eval()
    latents = torch.randn(1, 19, ROW_LATENTS, width, device=device)
    values = torch.randn(1, 19, 48, width, device=device)[..., ::2, :]
    original = (latents.clone(), values.clone())
    projected = []

    def inspect_broadcast(module, args):
        projected.append(args[0].shape[-2])

    with layer.broadcast.register_forward_pre_hook(inspect_broadcast), torch.inference_mode(), torch.autocast(
        device, dtype=torch.float16, enabled=device == "cuda",
    ):
        expected, updated_features = layer(latents, values)
        packed = layer(latents, values, packed=True)
        expected_trace = layer._gather(latents, values, layer.gather, layer.gather_norm)
        updated, trace = layer.forward_with_replay(latents, values)
    combined = ROW_LATENTS + values.shape[-2]
    assert projected == [combined, combined, ROW_LATENTS]
    torch.testing.assert_close(packed, torch.cat((expected, updated_features), dim=-2), atol=0, rtol=0)
    torch.testing.assert_close(updated, expected, atol=0, rtol=0)
    torch.testing.assert_close(trace, expected_trace, atol=0, rtol=0)
    torch.testing.assert_close((latents, values), original, atol=0, rtol=0)


@pytest.mark.parametrize("device", cpu_and_cuda)
def test_column_prefill_normalizes_context_once_and_preserves_kv(device):
    layer = ColumnLayer(width=16, heads=2, expansion=2, latents=4).to(device).eval()
    with torch.inference_mode():
        layer.latents.normal_()
        rows = torch.randn(2, 20, 16, device=device)[:, ::2]
        context = rows[:, :7]
        gather = layer.gather
        attention = gather.attention
        norm = gather.attention_norm
        latents = layer.latents.expand(2, -1, -1)
        projections = zip((norm(latents), norm(context), norm(context)), attention.projection.weight.chunk(3),
                          attention.projection.bias.chunk(3), strict=True)
        q, k, v = (F.linear(x, w, b) for x, w, b in projections)
        pair = KVPair(attention._normalized_heads(k), attention._heads(v))
        ref_latents = gather._finish(latents, attention.attend(attention._normalized_heads(q), pair))
        expected, expected_cache = layer.broadcast.prefill(rows, ref_latents)

        norm_shapes = []
        with norm.register_forward_hook(lambda module, inputs, result: norm_shapes.append(inputs[0].shape)):
            actual, cache = layer.build_context(rows, 7)
    assert norm_shapes == [latents.shape, context.shape]
    for got, want in ((actual, expected), (cache.key, expected_cache.key), (cache.value, expected_cache.value)):
        torch.testing.assert_close(got, want, atol=0, rtol=0, check_stride=True)


@pytest.mark.parametrize("device", cpu_and_cuda)
def test_final_prediction_prefill_builds_kv_without_evaluating_queries(device, monkeypatch):
    device = torch.device("cuda:0" if device == "cuda" else device)
    config = ModelConfig(max_classes=0, width=8, column_depth=1, prediction_depth=2)
    block = PredictionBlock(config).to(device).eval()
    rows = torch.randn(2, 7, config.prediction_width, device=device)
    targets = torch.randn(2, 7, device=device)
    final = block.layers[-1]
    with torch.inference_mode():
        def prefill():
            conditioned = block.add_context_targets(rows, targets)
            context = block.layers[0](conditioned, context_rows=targets.shape[-1])
            return final.prefill(context[:, :0], context)[1]

        reference = precision.run_inference(device, prefill, stage=Stage.PREDICTION)

        def unexpected_query(*args, **kwargs):
            pytest.fail("The last cache layer should only prepare context K/V")

        monkeypatch.setattr(final, "prefill", unexpected_query)
        monkeypatch.setattr(block.head, "forward", unexpected_query)
        cached = PredictionRunner(block, device).build_cache(rows, targets).layers[1]
    for actual, expected in ((cached.key, reference.key), (cached.value, reference.value)):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0, check_stride=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_restore_rebuilds_cuda_cache_when_bf16_support_changes(checkpoint15, mixed_data, monkeypatch):
    if not precision.supports_bf16(torch.device("cuda")):
        pytest.skip("Requires native CUDA BF16")

    X, query = mixed_data
    targets = X["a"].to_numpy()
    options = dict(
        device="cuda", checkpoint_path=checkpoint15("regression"),
        n_estimators=1, use_kv_cache=True, retain_preprocessing=False,
    )
    with monkeypatch.context() as placement:
        placement.setattr("causilo.v1_5.engine.supports_bf16", lambda device: False)
        model = CausiloRegressor(**options).fit(X, targets)
        saved = model.__getstate__()

    restored = CausiloRegressor()
    monkeypatch.setattr(EnsembleRunner, "place_caches", lambda *_: pytest.fail("Restore must rebuild K/V"))
    restored.__setstate__(saved)
    fresh = CausiloRegressor(**options).fit(X, targets)
    np.testing.assert_array_equal(restored.predict(query), fresh.predict(query))


@pytest.mark.parametrize(("task", "classes", "n_estimators"), [
    ("classification", 3, 8), ("classification", 12, 1), ("regression", 0, 8),
])
def test_weighted_cache_matches_repeated_rows_and_restores_without_prefill(
    checkpoint15, mixed_data, task, classes, n_estimators, monkeypatch,
):
    table, query = mixed_data
    targets = np.arange(len(table)) % classes if classes else 2 + table["a"].to_numpy() * 3
    constructor = CausiloClassifier if task == "classification" else CausiloRegressor
    options = dict(
        device="cpu", n_estimators=n_estimators, checkpoint_path=checkpoint15(task),
    )
    counts = np.resize([0, 4, 3, 2, 1, 4, 4], len(table))
    rows = np.repeat(np.arange(len(table)), counts)
    reference = constructor(**options).fit(table.iloc[rows], targets[rows])
    cached = constructor(**options, use_kv_cache=True, retain_preprocessing=False).fit(
        table, targets, sample_count=counts,
    )

    def predict(estimator):
        if classes:
            return estimator.predict_proba(query)
        return estimator.predict(query, output_type="quantiles", quantiles=[0.06, 0.8, 0.06])

    observed = predict(cached)
    np.testing.assert_allclose(predict(reference), observed, atol=1e-6, rtol=2e-6)
    if classes:
        np.testing.assert_array_equal(cached.classes_, reference.classes_)

    monkeypatch.setattr(EnsembleRunner, "build_cache", lambda *_: pytest.fail("Expected saved K/V reuse"))
    restored = pickle.loads(pickle.dumps(cached))
    np.testing.assert_array_equal(predict(restored), observed)


def test_bounded_regression_prefill_reuses_host_layers_across_queries_and_restore(
    checkpoint15, mixed_data, monkeypatch,
):
    table, query = mixed_data
    targets = table["a"].to_numpy()
    options = dict(
        device="cpu", n_estimators=8, checkpoint_path=checkpoint15("regression"), use_kv_cache=True,
    )
    queries = (query.iloc[:5], query.iloc[5:])
    counts = 1 + np.arange(len(table)) % 3

    def values(fitted):
        return [fitted.predict(rows, output_type="raw") for rows in queries]

    expected = values(CausiloRegressor(**options).fit(table, targets, sample_count=counts))
    prefilled = []
    initial_encode = RecomputeRunner._encode_context

    def encode(replay):
        prefilled.append(replay.context_rows)
        return initial_encode(replay)

    def build(runner, table, targets, *, num_classes, weights):
        return RecomputeRunner(runner.model).build_cache(
            table, targets, num_classes=num_classes, weights=weights,
        )

    def predict(runner, rows, cache, weights):
        return RecomputeRunner(runner.model).predict_cached(rows, cache, weights=weights)

    monkeypatch.setattr(EnsembleRunner, "_ensemble_batch_count", lambda _, shape: shape[0])
    monkeypatch.setattr(RecomputeRunner, "_encode_context", encode)
    monkeypatch.setattr(RecomputeRunner, "_limit", lambda _, size, *args: min(size, 7))
    monkeypatch.setattr(ModelRunner, "build_cache", build)
    monkeypatch.setattr(ModelRunner, "_predict_native_cached", predict)
    fitted = CausiloRegressor(**options).fit(table, targets, sample_count=counts)
    observed = values(fitted)
    restored = values(pickle.loads(pickle.dumps(fitted)))

    assert prefilled == [len(table)] * 8
    for left, right, after_restore in zip(expected, observed, restored, strict=True):
        np.testing.assert_allclose(right, left, atol=3e-6, rtol=3e-6)
        np.testing.assert_array_equal(after_restore, right)


def test_zero_weight_rows_do_not_select_many_class_inference(checkpoint15, mixed_data):
    table, query = mixed_data
    targets = np.arange(len(table)) % 11
    keep = targets != 0
    options = dict(
        device="cpu", n_estimators=8, use_kv_cache=True,
        checkpoint_path=checkpoint15("classification"),
    )
    fitted = CausiloClassifier(**options).fit(table, targets, sample_count=keep.astype(np.int64))
    reference = CausiloClassifier(**options).fit(table.iloc[keep], targets[keep])

    np.testing.assert_array_equal(fitted.classes_, np.arange(1, 11))
    np.testing.assert_array_equal(fitted.predict_logits(query), reference.predict_logits(query))


def test_batched_cache_joblib_preserves_independent_tables(checkpoint15, mixed_data, tmp_path):
    X, query = mixed_data
    y = np.arange(len(X)) % 3
    options = dict(device="cpu", n_estimators=1, checkpoint_path=checkpoint15("classification"))
    batched = CausiloClassifier(**options, use_kv_cache=True).fit_batched({
        "first": (X, y), "narrow": (X.iloc[2:, :2], y[2:]),
    })
    states = batched._engine.batch_states
    assert states["first"].runner.model is states["narrow"].runner.model
    assert states["first"].caches is not states["narrow"].caches
    path = tmp_path / "batch.joblib"
    joblib.dump(batched, path)
    restored = joblib.load(path)
    queries = {"narrow": query.iloc[:, :2], "first": query.iloc[:, ::-1]}
    results = restored.predict_proba_batched(queries)
    assert list(results) == list(queries)
    for key, table, targets in (("first", X, y), ("narrow", X.iloc[2:, :2], y[2:])):
        expected = CausiloClassifier(**options).fit(table, targets).predict_proba(queries[key])
        np.testing.assert_allclose(results[key], expected, atol=1e-6, rtol=2e-6)


def test_prefill_retries_the_failed_member_group_and_stops_at_one(checkpoint15, mixed_data, monkeypatch):
    X, query = mixed_data
    y = np.arange(len(X)) % 3
    options = dict(device="cpu", n_estimators=8, checkpoint_path=checkpoint15("classification"))
    model = CausiloClassifier(**options, use_kv_cache=True)
    real_prefill = EnsembleRunner._prefill
    attempts = []
    capacity = 2

    def prefill(self, dataset, train, section):
        attempts.append((section.start, section.stop))
        if section.start >= 4 and section.stop - section.start > capacity:
            raise torch.cuda.OutOfMemoryError("simulated prefill allocation failure")
        return real_prefill(self, dataset, train, section)

    monkeypatch.setattr(EnsembleRunner, "_ensemble_batch_count", lambda *_: 2)
    monkeypatch.setattr(EnsembleRunner, "_prefill", prefill)
    model.fit(X, y)
    expected = CausiloClassifier(**options).fit(X, y).predict_proba(query)
    np.testing.assert_allclose(model.predict_proba(query), expected, atol=1e-6, rtol=2e-6)
    assert attempts == [(0, 4), (4, 8), (4, 6), (6, 8)]

    attempts.clear()
    capacity = 0
    with pytest.raises(torch.cuda.OutOfMemoryError, match="prefill allocation failure"):
        model.fit(X, y)
    assert attempts == [(0, 4), (4, 8), (4, 6), (4, 5)]


def test_cached_predictions_share_an_estimator_safely(checkpoint15, mixed_data, monkeypatch):
    X, query = mixed_data
    model = CausiloClassifier(
        checkpoint_path=checkpoint15("classification"), device="cpu", n_estimators=1, use_kv_cache=True,
    ).fit(X, np.arange(len(X)) % 3)
    expected = model.predict_proba(query)
    runner = model._engine.state.runner
    model_runner = runner._model_runner
    predict_cached = model_runner.predict_cached
    busy, release, other_started = Event(), Event(), Event()

    def pause_first(*args, **kwargs):
        if not busy.is_set():
            busy.set()
            assert release.wait(5)
        return predict_cached(*args, **kwargs)

    def predict_second():
        other_started.set()
        return model.predict_proba(query)

    monkeypatch.setattr(model_runner, "predict_cached", pause_first)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(model.predict_proba, query)
        try:
            assert busy.wait(5)
            second = pool.submit(predict_second)
            assert other_started.wait(5)
            assert not second.done()
        finally:
            release.set()
        np.testing.assert_array_equal(first.result(), expected)
        np.testing.assert_array_equal(second.result(), expected)


def test_runner_rejects_models_with_training_submodules(checkpoint15):
    model = load_checkpoint(read_local_checkpoint(checkpoint15("classification")), "classification")
    model.prediction.head.train()
    with pytest.raises(ValueError, match="requires model.eval"):
        ModelRunner(model, torch.device("cpu"))


def test_batch_checks_all_query_schemas_before_prediction(checkpoint15, mixed_data, monkeypatch):
    X, query = mixed_data
    model = CausiloClassifier(
        device="cpu", n_estimators=1, checkpoint_path=checkpoint15("classification"),
    ).fit_batched({key: (X, np.arange(len(X)) % 3) for key in ("first", "second")})

    runner = model._engine.batch_states["first"].runner
    monkeypatch.setattr(runner, "predict", lambda *args, **kwargs: pytest.fail("Ran before validating all tables"))
    with pytest.raises(ValueError, match="Table 'second': Prediction columns must match"):
        model.predict_proba_batched({"first": query, "second": query.rename(columns={"kind": "unknown"})})


def test_runner_releases_embedding_grid_before_row_attention(checkpoint15, monkeypatch):
    model = load_checkpoint(read_local_checkpoint(checkpoint15("classification")), "classification")
    runner = ModelRunner(model, torch.device("cpu"))
    features = torch.arange(40).reshape(1, 10, 4).float()
    targets = torch.arange(6).remainder(3).reshape(1, 6).float()
    embed, refine = runner.feature_embedding, runner.row.refine
    grids = []

    def track_grid(*args, **kwargs):
        result = embed(*args, **kwargs)
        grids.append(weakref.ref(result))
        return result

    def check_row(part, latents, index):
        if index == 0:
            assert grids[-1]() is None
        return refine(part, latents, index)

    monkeypatch.setattr(runner, "feature_embedding", track_grid)
    monkeypatch.setattr(runner.row, "refine", check_row)
    with torch.inference_mode():
        runner.predict(features, targets, num_classes=3)
        cache = runner.build_cache(features[:, :6], targets, num_classes=3)
        runner.predict_cached(features[:, 6:], cache)
    assert len(grids) == 3
