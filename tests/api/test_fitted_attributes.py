import pickle
from threading import Lock

import numpy as np
import pytest

from causilo import CausiloClassifier, CausiloRegressor


def restore_attributes(estimator):
    """Exercise frameworks that bypass the estimator's checkpoint-aware pickle hooks."""
    restored = type(estimator).__new__(type(estimator))
    restored.__dict__.update(pickle.loads(pickle.dumps(estimator.__dict__)))
    return restored


def outputs(estimator, query):
    if isinstance(estimator, CausiloClassifier):
        return estimator.predict_proba(query)
    return estimator.predict(query, output_type="quantiles", quantiles=[0.1, 0.5, 0.9])


@pytest.mark.parametrize("version", ["1.0", "1.5"])
@pytest.mark.parametrize("case", ["binary", "multiclass", "ecoc", "regression"])
@pytest.mark.parametrize("members", [1, 8])
@pytest.mark.parametrize("cache", [False, True])
def test_fitted_attributes_pickle_before_and_after_predict(
    version, case, members, cache, checkpoint_factory, checkpoint15, mixed_data, batch_runtime,
):
    task = "regression" if case == "regression" else "classification"
    constructor = CausiloRegressor if task == "regression" else CausiloClassifier
    checkpoint = checkpoint_factory(task) if version == "1.0" else checkpoint15(task)
    train, query = mixed_data
    targets = (
        np.linspace(-4, 7, len(train)) ** 3 if task == "regression"
        else np.arange(len(train)) % {"binary": 2, "multiclass": 3, "ecoc": 13}[case]
    )
    estimator = constructor(
        model_version=version, n_estimators=members, device="cpu", use_kv_cache=cache,
        checkpoint_path=checkpoint,
    ).fit(train, targets)
    before = restore_attributes(estimator)
    expected = outputs(estimator, query)
    np.testing.assert_array_equal(outputs(before, query), expected)
    after = restore_attributes(estimator)
    np.testing.assert_array_equal(outputs(after, query), expected)
    if version == "1.5":
        engine = after._engine
        assert engine.model is engine._model_replicas[engine.device]
        assert engine.state.runner.model is engine.model
        assert engine.state.runner._model_runner.model is engine.model


@pytest.mark.parametrize("version", ["1.0", "1.5"])
@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("cache", [False, True])
def test_batched_fitted_attributes_pickle(
    version, task, cache, checkpoint_factory, checkpoint15, mixed_data, batch_runtime,
):
    train, query = mixed_data
    constructor = CausiloClassifier if task == "classification" else CausiloRegressor
    checkpoint = checkpoint_factory(task) if version == "1.0" else checkpoint15(task)
    target = np.arange(len(train)) % 3 if task == "classification" else np.arange(len(train)) ** 2
    estimator = constructor(
        model_version=version, n_estimators=2, device="cpu", use_kv_cache=cache,
        checkpoint_path=checkpoint,
    ).fit_batched({"first": (train, target), "second": (train.iloc[:23], target[:23])})
    queries = {"second": query.iloc[:2], "first": query}
    method = "predict_proba_batched" if task == "classification" else "predict_batched"
    expected = getattr(estimator, method)(queries)
    restored = restore_attributes(estimator)
    actual = getattr(restored, method)(queries)
    assert actual.keys() == expected.keys()
    for key, values in actual.items():
        np.testing.assert_array_equal(values, expected[key])


@pytest.mark.parametrize("cache", [False, True])
def test_sampled_pickle_preserves_shared_models_and_fresh_shared_locks(
    cache, checkpoint15, mixed_data, batch_runtime,
):
    train, query = mixed_data
    estimator = CausiloRegressor(
        n_estimators=4, max_context=17, device="cpu", use_kv_cache=cache,
        checkpoint_path=checkpoint15("regression"),
    ).fit(train, np.arange(len(train)) ** 2, sample_count=2 + np.arange(len(train)) % 3)
    expected = estimator.predict(query)
    runners = [member.runner for member in estimator._engine.state.members]
    shared_lock = runners[0]._prediction_lock
    assert all(runner._prediction_lock is shared_lock for runner in runners)
    assert len({id(runner) for runner in runners}) == len(runners)
    stream = Lock()  # An unpicklable sentinel exercises the CUDA handle boundary on CPU.
    gains = {"transient": Lock()}
    for runner in runners:
        runner._stream = stream
        runner._length_gains = gains
    with shared_lock.lock:
        restored = restore_attributes(estimator)
        restored_runners = [member.runner for member in restored._engine.state.members]
        restored_lock = restored_runners[0]._prediction_lock
        assert restored_lock is not shared_lock
        assert not restored_lock.lock.locked()
        assert all(runner._prediction_lock is restored_lock for runner in restored_runners)
        assert shared_lock.lock.locked()
    for original, loaded in zip(runners, restored_runners, strict=True):
        assert original._stream is stream and original._length_gains is gains
        assert loaded._stream is None and loaded._length_gains == {}
        assert loaded.model is restored._engine.model
        assert loaded._model_runner.model is restored._engine.model
    np.testing.assert_array_equal(restored.predict(query), expected)
