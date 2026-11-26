from weakref import ref

import numpy as np
import pytest
import torch

from causilo import CausiloRegressor
from causilo.v1_5.data.dataset import PreparedDataset
from causilo.v1_5.execution.direct import EnsembleRunner
from causilo.v1_5.execution.runner import ModelRunner


@pytest.mark.parametrize("query_allocation_failure", [False, True])
def test_sampled_retry_reuses_context_until_query_preparation_needs_memory(
    checkpoint15, mixed_data, monkeypatch, query_allocation_failure,
):
    train, query = mixed_data
    model = CausiloRegressor(
        n_estimators=2, max_context=17, use_kv_cache=False, retain_preprocessing=False,
        checkpoint_path=checkpoint15("regression"), device="cpu",
    ).fit(train, np.arange(len(train), dtype=float) ** 2)
    members = model._engine.require_state().members
    lock = members[0].runner._prediction_lock.lock
    retry_size = 1 if query_allocation_failure else 2
    with monkeypatch.context() as patch:
        patch.setattr(EnsembleRunner, "query_batch_size", lambda *args, **kwargs: retry_size)
        expected = model.predict(query)

    transforms = dict.fromkeys((id(member.dataset) for member in members), 0)
    normalize = PreparedDataset._normalize
    prepare_query = PreparedDataset.query_table
    predict = ModelRunner.predict
    last_context = {}
    attempts = []

    def count_training_transforms(dataset, values):
        normalized = normalize(dataset, values)
        if values is dataset.training:
            key = id(dataset)
            transforms[key] += 1
            last_context[key] = ref(normalized)
        return normalized

    def check_query_space(dataset, table, section, *, layout):
        context = last_context.get(id(dataset))
        if section.stop - section.start == 2 and context is not None and context() is not None:
            raise MemoryError("injected query preparation allocation failure")
        return prepare_query(dataset, table, section, layout=layout)

    def fail_large_queries(runner, table, targets, **kwargs):
        assert lock.locked()
        rows = table.shape[1] - targets.shape[1]
        attempts.append(rows)
        if rows > 2:
            raise torch.cuda.OutOfMemoryError("injected query allocation failure")
        return predict(runner, table, targets, **kwargs)

    monkeypatch.setattr(PreparedDataset, "_normalize", count_training_transforms)
    monkeypatch.setattr(EnsembleRunner, "query_batch_size", lambda *args, **kwargs: 4)
    monkeypatch.setattr(ModelRunner, "predict", fail_large_queries)
    if query_allocation_failure:
        monkeypatch.setattr(PreparedDataset, "query_table", check_query_space)

    np.testing.assert_array_equal(model.predict(query), expected)
    outer_chunks = len(query) // 4
    steps = [4, 1, 1, 1, 1] if query_allocation_failure else [4, 2, 2]
    transforms_per_chunk = 2 if query_allocation_failure else 1
    assert attempts == steps * outer_chunks * len(members)
    assert list(transforms.values()) == [outer_chunks * transforms_per_chunk] * len(members)
    assert all(context() is None for context in last_context.values())
    assert not lock.locked()
