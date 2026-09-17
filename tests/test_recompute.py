import weakref

import numpy as np
import pytest
import torch

from causilo.data.dataset import PreparedDataset
from causilo.execution import direct, memory, recompute
from causilo.execution.direct import BatchPlanner, direct_predictions
from causilo.execution.recompute import RecomputeRunner, RetryPlan
from causilo.execution.runner import ModelRunner
from causilo.model import Model, ModelConfig


def bound_tiles(monkeypatch, size):
    original = RecomputeRunner._limit

    def limited(runner, *args, **kwargs):
        return min(size, original(runner, *args, **kwargs))

    monkeypatch.setattr(RecomputeRunner, "_limit", limited)


@pytest.fixture
def small_model(request):
    task = getattr(request, "param", "regression")
    with torch.random.fork_rng():
        torch.manual_seed(18)
        model = Model(ModelConfig(
            task=task, width=16, expansion=2, group_size=3, frequencies=4,
            column_latents=5, column_heads=2, column_depths=(2, 2),
            row_heads=2, row_latents=2, row_depths=(3, 2),
            prediction_heads=2, prediction_depth=3, outputs=3 if task == "classification" else 9,
        ))
        for name, parameter in model.named_parameters():
            if "latents" in name or name.endswith("frequencies") or name.endswith("missing"):
                torch.nn.init.normal_(parameter, std=0.1)
        table = torch.randn(1, 17, 7)
        table[0, 2, 1] = float("nan")
        targets = torch.arange(11).remainder(3).float().unsqueeze(0)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield model.eval(), table, targets
    torch.set_num_threads(previous)


@pytest.mark.parametrize("small_model", ["classification", "regression"], indirect=True)
@pytest.mark.parametrize("tile", [1, 4, 50])
def test_recomputed_logits_match_direct_with_padding_and_missing_values(small_model, monkeypatch, tile):
    model, table, targets = small_model
    monkeypatch.setattr(RecomputeRunner, "_limit", lambda self, items, *a, **kw: min(items, tile))
    with torch.inference_mode():
        expected = ModelRunner(model).predict(table, targets)
        actual = RecomputeRunner(model, torch.device("cpu")).predict(table, targets)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_recomputed_queries_never_change_training_context(small_model, monkeypatch):
    model, table, targets = small_model
    bound_tiles(monkeypatch, 4)
    runner = RecomputeRunner(model, torch.device("cpu"))
    changed = table.clone()
    changed[:, -3:] *= 100
    with torch.inference_mode():
        expected = runner.predict(table, targets)
        actual = runner.predict(changed, targets)
    torch.testing.assert_close(actual[:, :3], expected[:, :3], atol=0, rtol=0)


@pytest.mark.parametrize("features", [112, 995])
def test_recomputed_wide_feature_groups_keep_their_order(small_model, monkeypatch, features):
    model, _, targets = small_model
    generator = torch.Generator().manual_seed(53)
    table = torch.randn(1, 17, features, generator=generator)
    table[:, 3, ::7] = float("nan")
    monkeypatch.setattr(RecomputeRunner, "_limit", lambda self, items, *a, **kw: min(items, 7))
    with torch.inference_mode():
        expected = ModelRunner(model).predict(table, targets)
        actual = RecomputeRunner(model, torch.device("cpu")).predict(table, targets)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_recomputed_projection_oom_preserves_full_training_keys(small_model, monkeypatch):
    model, table, targets = small_model
    runner = RecomputeRunner(model, torch.device("cpu"))
    with torch.inference_mode():
        expected = runner.predict(table, targets)
    layer = model.prediction.layers[0]
    original = layer.prepare
    lengths = []

    def fail_large_projection(value):
        lengths.append(value.shape[1])
        if value.shape[1] > 3:
            raise torch.cuda.OutOfMemoryError("injected K/V projection failure")
        return original(value)

    monkeypatch.setattr(layer, "prepare", fail_large_projection)
    with torch.inference_mode():
        actual = runner.predict(table, targets)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert lengths[0] == targets.shape[1]
    assert sum(length for length in lengths if length <= 3) == targets.shape[1]


@pytest.mark.pretrained
@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_public_predictions_keep_all_ensemble_members_when_recomputed(task, device, monkeypatch):
    from causilo import CausiloClassifier, CausiloRegressor

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA prediction parity")
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        table = np.random.default_rng(32).normal(size=(24, 7))
        table[3, 2] = np.nan
        cls = CausiloClassifier if task == "classification" else CausiloRegressor
        fitted = cls(n_estimators=8, device=device).fit(table, np.arange(24) % 3)
        method = fitted.predict_proba if task == "classification" else fitted.predict
        expected = method(table[:5])
        monkeypatch.setattr(direct, "execution_budget", lambda device: 1)
        bound_tiles(monkeypatch, 7)
        actual = method(table[:5])
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("small_model", ["classification", "regression"], indirect=True)
def test_cuda_recomputation_keeps_intermediate_states_on_device(small_model, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA intermediate placement")
    model, table, targets = small_model
    device = torch.device("cuda:0")
    runner = RecomputeRunner(model.to(device), device)
    monkeypatch.setattr(memory, "CUDA_MAX_BATCH_ITEMS", 4)
    original_map = runner._map
    original_replay = runner._replay_row
    original_cpu = torch.Tensor.cpu
    assembled = []
    replayed = []
    returned = []
    output_shape = (1, table.shape[1] - targets.shape[1], model.config.outputs)

    def check_map(source, operation, **kwargs):
        result = original_map(source, operation, **kwargs)
        assert result.device == device
        if kwargs.get("inplace") and result.shape == source.shape and result.dtype == source.dtype:
            assert result.is_set_to(source)
        assembled.append(result.shape)
        return result

    def check_replay(features, trace):
        assert features.device == trace.device == device
        assert features.shape[1] <= 4
        replayed.append(True)
        return original_replay(features, trace)

    def return_predictions_only(tensor, *args, **kwargs):
        assert tensor.device == device and tensor.shape == output_shape
        returned.append(True)
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(runner, "_map", check_map)
    monkeypatch.setattr(runner, "_replay_row", check_replay)
    monkeypatch.setattr(torch.Tensor, "cpu", return_predictions_only)
    with torch.inference_mode():
        result = runner.predict(table, targets)

    assert assembled and replayed and returned == [True]
    assert result.device.type == "cpu" and result.shape == output_shape
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("allocation_fits", [True, False])
def test_row_buffer_allocation_oom_retries_on_same_device(device, allocation_fits, small_model, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA destination allocation")
    model, _, _ = small_model
    device = torch.device(device)
    runner = RecomputeRunner(model.to(device), device)
    source = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)
    original_new_empty = torch.Tensor.new_empty
    temporaries = []
    attempts = []

    def transform(part, section):
        assert all(reference() is None for reference in temporaries)
        assert part.device.type == device.type
        result = part + 1
        temporaries.append(weakref.ref(result))
        attempts.append((section.start, part.shape[1]))
        return result

    def fail_allocation(tensor, *args, **kwargs):
        assert tensor.device.type == device.type
        if not allocation_fits or len(attempts) == 1:
            raise torch.cuda.OutOfMemoryError("injected row buffer allocation failure")
        return original_new_empty(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "new_empty", fail_allocation)
    with torch.inference_mode():
        if allocation_fits:
            result = runner._map(source, transform, axis=1, limit=4)
        else:
            with pytest.raises(torch.cuda.OutOfMemoryError, match="Minimum recomputation tile"):
                runner._map(source, transform, axis=1, limit=4)

    if allocation_fits:
        assert attempts == [(0, 4), (0, 2), (2, 2), (4, 2), (6, 2)]
        assert result.device.type == device.type
        torch.testing.assert_close(result, (source + 1).to(device))
    else:
        assert attempts == [(0, 4), (0, 2), (0, 1)]
    assert all(reference() is None for reference in temporaries)
    torch.testing.assert_close(source, torch.arange(8, dtype=torch.float32).reshape(1, 8, 1))


def test_column_summary_oom_retries_group_without_partial_state(small_model, monkeypatch):
    model, table, targets = small_model
    runner = RecomputeRunner(model, torch.device("cpu"))
    with torch.inference_mode():
        expected = runner.predict(table, targets)
    original = model.columns[0].layers[0].prepare
    widths = []

    def fail_groups(value):
        widths.append(value.shape[1])
        if value.shape[1] > 1:
            raise torch.cuda.OutOfMemoryError("injected column workspace failure")
        return original(value)

    monkeypatch.setattr(model.columns[0].layers[0], "prepare", fail_groups)
    with torch.inference_mode():
        actual = runner.predict(table, targets)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert widths[0] > 1 and widths[-1] == 1


def test_row_trace_oom_retries_after_completed_rows(small_model, monkeypatch):
    model, table, targets = small_model
    runner = RecomputeRunner(model, torch.device("cpu"))
    bound_tiles(monkeypatch, 4)
    with torch.inference_mode():
        expected = runner.predict(table, targets)
    original = runner._row_trace
    calls = []

    def fail_second_tile(value):
        calls.append(value.shape[1])
        if len(calls) == 2:
            raise torch.cuda.OutOfMemoryError("injected row checkpoint failure")
        return original(value)

    monkeypatch.setattr(runner, "_row_trace", fail_second_tile)
    with torch.inference_mode():
        actual = runner.predict(table, targets)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert calls[:3] == [4, 4, 2]


def test_replayed_row_oom_never_reapplies_completed_updates(small_model, monkeypatch):
    model, table, targets = small_model
    runner = RecomputeRunner(model, torch.device("cpu"))
    bound_tiles(monkeypatch, 4)
    with torch.inference_mode():
        expected = runner.predict(table, targets)
    original = runner._replay_row
    calls = []

    def fail_second_tile(value, trace):
        calls.append(value.shape[1])
        if len(calls) == 2:
            raise torch.cuda.OutOfMemoryError("injected row replay failure")
        return original(value, trace)

    monkeypatch.setattr(runner, "_replay_row", fail_second_tile)
    with torch.inference_mode():
        actual = runner.predict(table, targets)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert calls[:3] == [4, 4, 2]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_prediction_oom_never_reapplies_completed_row_updates(small_model, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA row update recovery")
    device = torch.device("cuda:0" if device == "cuda" else device)
    model, table, targets = small_model
    runner = RecomputeRunner(model.to(device), device)
    bound_tiles(monkeypatch, 4)
    with torch.inference_mode():
        expected = runner.predict(table, targets)
    layer = model.prediction.layers[0]
    original = layer.query
    calls = []

    def fail_second_tile(value, context):
        calls.append(value.shape[1])
        assert context.key.shape[-2] == targets.shape[1]
        if len(calls) == 2:
            raise torch.cuda.OutOfMemoryError("injected prediction failure")
        return original(value, context)

    monkeypatch.setattr(layer, "query", fail_second_tile)
    with torch.inference_mode():
        actual = runner.predict(table, targets)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert calls[:3] == [4, 4, 2]


def test_context_retry_discards_partial_projection(small_model, monkeypatch):
    model, table, targets = small_model
    runner = RecomputeRunner(model, torch.device("cpu"))
    bound_tiles(monkeypatch, 4)
    with torch.inference_mode():
        expected = runner.predict(table, targets)
    layer = model.prediction.layers[0]
    original = layer.prepare
    inputs = []

    def fail_second_projection(value):
        inputs.append(value.clone())
        if len(inputs) == 2:
            raise torch.cuda.OutOfMemoryError("failure after partially filling training K/V")
        return original(value)

    monkeypatch.setattr(layer, "prepare", fail_second_projection)
    with torch.inference_mode():
        actual = runner.predict(table, targets)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    # Restart from the first training row with the smaller projection tile.
    assert [part.shape[1] for part in inputs[:3]] == [4, 4, 2]
    torch.testing.assert_close(inputs[2], inputs[0][:, :2], atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_retry_releases_failed_temporaries_before_clearing_cache(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA temporary lifetime")
    device = torch.device(device)
    retry = RetryPlan(4, device)
    temporaries = []
    sizes = []
    cleared = []

    def operation(size):
        assert all(reference() is None for reference in temporaries)
        temporary = torch.empty(8, device=device)
        temporaries.append(weakref.ref(temporary))
        sizes.append(size)
        if size > 1:
            raise torch.cuda.OutOfMemoryError("injected allocation failure")
        return size

    def empty_cache():
        assert all(reference() is None for reference in temporaries)
        cleared.append(True)

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    assert retry.run(operation) == 1
    assert sizes == [4, 2, 1]
    assert all(reference() is None for reference in temporaries)
    assert len(cleared) == (2 if device.type == "cuda" else 0)


def test_retry_counts_failures_across_completed_tiles(monkeypatch):
    monkeypatch.setattr(recompute, "MAX_EXECUTION_ATTEMPTS", 2)
    retry = RetryPlan(8, torch.device("cpu"))
    sizes = []

    def first_tile(size):
        sizes.append(size)
        if size > 4:
            raise torch.cuda.OutOfMemoryError("first tile failure")
        return size

    def second_tile(size):
        sizes.append(size)
        raise torch.cuda.OutOfMemoryError("second tile failure")

    assert retry.run(first_tile) == 4
    with pytest.raises(torch.cuda.OutOfMemoryError, match="retry limit"):
        retry.run(second_tile)
    assert sizes == [8, 4, 4]


def test_retry_stops_at_minimum_tile_and_preserves_error_message():
    retry = RetryPlan(1, torch.device("cpu"), error_message="Full context does not fit")
    calls = []

    def operation(size):
        calls.append(size)
        raise torch.cuda.OutOfMemoryError("injected allocation failure")

    with pytest.raises(torch.cuda.OutOfMemoryError, match="Full context does not fit"):
        retry.run(operation)
    assert calls == [1]


def test_retry_propagates_non_oom_errors_without_retrying():
    retry = RetryPlan(8, torch.device("cpu"))
    calls = []

    def operation(size):
        calls.append(size)
        raise ValueError("invalid model input")

    with pytest.raises(ValueError, match="invalid model input"):
        retry.run(operation)
    assert calls == [8] and retry.limit == 8 and retry.failures == 0


@pytest.mark.parametrize("fit_members", [0, 1, 3, 8])
def test_planner_selects_route_and_member_count_at_budget_boundary(small_model, fit_members):
    model, table, targets = small_model
    planner = BatchPlanner(model.config, targets.shape[1], table.shape[-1])
    device = torch.device("cpu")
    queries = table.shape[1] - targets.shape[1]
    one_member = planner.estimate(1, queries, device)
    budget = fit_members * one_member if fit_members else one_member - 1

    plan = planner.choose(8, queries, budget, device)

    assert plan.mode == ("direct" if fit_members else "recompute")
    assert plan.members == max(1, fit_members)
    assert plan.queries == queries


@pytest.mark.parametrize(
    "training, features, capacity, mode",
    [
        (1000, 995, 500, "direct"),
        (1000, 995, 250, "recompute"),
        (100000, 995, 500, "recompute"),
        (1000, 7, 500, "recompute"),
    ],
)
def test_planner_compares_repeated_training_with_replay(small_model, training, features, capacity, mode):
    model, _, _ = small_model
    device = torch.device("cpu")
    planner = BatchPlanner(model.config, training, features)
    queries = 1000
    budget = planner.estimate(1, capacity, device)
    assert planner._recompute_state_bytes(queries) < budget

    plan = planner.choose(8, queries, budget, device)

    assert plan.mode == mode and plan.members == 1
    assert plan.queries == (capacity if mode == "direct" else queries)
    if mode == "direct":
        assert planner.estimate(1, plan.queries, device) <= budget
        assert planner.estimate(1, plan.queries + 1, device) > budget


def test_planner_keeps_query_batches_when_recompute_states_do_not_fit(small_model):
    model, _, _ = small_model
    planner = BatchPlanner(model.config, 11, 7)
    device = torch.device("cpu")
    queries, capacity = 1000000, 5
    budget = planner.estimate(1, capacity, device)
    assert planner._recompute_state_bytes(queries) > budget

    plan = planner.choose(8, queries, budget, device)

    assert plan.mode == "direct" and plan.members == 1 and plan.queries == capacity


def test_planner_recomputes_when_minimum_direct_query_does_not_fit(small_model):
    model, _, _ = small_model
    planner = BatchPlanner(model.config, 1000, 995)
    device = torch.device("cpu")
    queries = 1000
    budget = planner.estimate(1, 1, device) - 1

    plan = planner.choose(8, queries, budget, device)

    assert plan.mode == "recompute" and plan.members == 1 and plan.queries == queries


def test_planner_prefers_query_batching_on_equal_estimated_work(small_model, monkeypatch):
    model, _, _ = small_model
    planner = BatchPlanner(model.config, 1000, 995)
    device = torch.device("cpu")
    budget = planner.estimate(1, 500, device)
    monkeypatch.setattr(BatchPlanner, "_training_work", lambda self: (100, 100))

    plan = planner.choose(8, 1000, budget, device)

    assert plan.mode == "direct" and plan.queries == 500


def test_direct_oom_replans_remaining_members_into_recompute(small_model, monkeypatch):
    model, table, targets = small_model
    device = torch.device("cpu")
    count = targets.shape[1]
    fitted = PreparedDataset.prepare(
        table[0, :count].numpy(), targets[0].numpy(), task=model.config.task,
        n_estimators=3, retain_preprocessing=True, random_state=42,
        class_permutation_size=model.config.outputs,
    )
    query = fitted.encoder.transform(table[0, count:].numpy())
    transformed = {name: normalizer.transform(query) for name, normalizer in fitted.normalizers.items()}
    planner = BatchPlanner(model.config, count, fitted.features.shape[1])
    budget = planner.estimate(1, len(query), device)
    monkeypatch.setattr(direct, "execution_budget", lambda _: budget)
    with torch.inference_mode():
        expected = list(direct_predictions(model, fitted, transformed, device, lambda value: value))
    original_predict = ModelRunner.predict
    original_choose = BatchPlanner.choose
    direct_calls = []
    modes = []

    def fail_second_member(runner, *args):
        direct_calls.append(True)
        if len(direct_calls) == 2:
            raise torch.cuda.OutOfMemoryError("failure after an already completed member")
        return original_predict(runner, *args)

    def record_plan(planner, *args):
        plan = original_choose(planner, *args)
        modes.append(plan.mode)
        return plan

    monkeypatch.setattr(ModelRunner, "predict", fail_second_member)
    monkeypatch.setattr(BatchPlanner, "choose", record_plan)
    with torch.inference_mode():
        actual = list(direct_predictions(model, fitted, transformed, device, lambda value: value))

    assert modes == ["direct", "direct", "recompute"]
    assert len(direct_calls) == 2
    assert [member for member, _ in actual] == [member for member, _ in expected]
    for (_, result), (_, reference) in zip(actual, expected):
        torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-5)
