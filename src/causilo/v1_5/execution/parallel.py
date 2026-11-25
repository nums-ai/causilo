"""Run independent 1.5 ensemble members across CUDA devices."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import torch

from ..data.dataset import TableViews, join_context, select_query_rows
from .direct import concatenate_parts, predict_chunks
from .memory import is_allocation_error


def member_sections(members, devices, capacities):
    """Balance member batches using each device's estimated in-flight capacity."""
    jobs = min(members, len(devices))
    counts = [1] * jobs
    order = range(jobs)
    for _ in range(members - jobs):
        index = min(order, key=lambda i: (counts[i] + 1) / capacities[i])
        counts[index] += 1
    stop = 0
    for device, count in zip(devices, counts):
        yield device, slice(stop, stop + count)
        stop += count


def _slice_dataset(dataset, section):
    plan = dataset.ensemble
    return replace(dataset, ensemble=replace(
        plan,
        feature_signs=plan.feature_signs[section],
        class_permutations=None if plan.class_permutations is None else plan.class_permutations[section],
        target_signs=None if plan.target_signs is None else plan.target_signs[section],
        feature_views=plan.feature_views[section],
    ))


class ParallelEnsembleRunner:
    """Use the same single-device runtime and recover original member order."""

    def __init__(self, lanes):
        self._lanes = lanes

    def _map_lanes(self, work):
        # Stage managers belong to the single-device runners. Worker calls
        # enter the same runtime's device and inference contexts as serial calls.
        with ThreadPoolExecutor(max_workers=len(self._lanes)) as executor:
            return list(executor.map(lambda pair: work(*pair), self._lanes))

    def query_batch_size(self, dataset, rows, *, cached, output_width=0):
        host_limit = self._lanes[0][0].host_query_limit(dataset, rows, cached=cached, output_width=output_width)
        if not output_width:
            return host_limit
        return min(
            runner.query_device_limit(
                dataset, host_limit, cached=cached, members=section.stop - section.start, output_width=output_width,
            )
            for runner, section in self._lanes
        )

    @staticmethod
    def query_chunks(prepare, predict, *, rows, batch_size):
        # Only shared host preparation retries here. Each lane retries its own prediction.
        for table in predict_chunks(prepare, rows=rows, batch_size=batch_size, device=torch.device("cpu")):
            yield predict(table)
            del table

    def validate_caches(self, dataset, caches):
        self._lanes[0][0].validate_caches(dataset, caches)

    def can_retain_replay(self, dataset):
        # All lanes use the same host, so budget for every member before any lane retains context.
        return self._lanes[0][0].can_retain_replay(dataset)

    def predict_table(self, dataset, table, *, output_type=None, levels=None, replay_contexts=None):
        """Pack signed inputs per device, and retry the affected lane and query offset."""
        context_rows = len(dataset.training)
        query_count = len(table) - context_rows
        if replay_contexts is not None:
            for _, section in self._lanes:
                replay_contexts.setdefault(section.start, {})

        def predict_lane(runner, section):
            view = _slice_dataset(dataset, section)
            y = view.targets_for(slice(None))
            limit = runner.query_device_limit(view, query_count, cached=False)

            contexts = None
            if replay_contexts is not None:
                contexts = replay_contexts[section.start]
            elif limit < query_count and self.can_retain_replay(dataset):
                contexts = {}

            def predict_rows(rows):
                if rows.stop - rows.start == query_count:
                    joint = table
                elif isinstance(table, TableViews):
                    joint = table.select_queries(rows)
                else:
                    joint = join_context(table[:context_rows], table[context_rows + rows.start:context_rows + rows.stop])
                return runner.predict(
                    view, joint, y, output_type=output_type, levels=levels, replay_contexts=contexts,
                )

            arrays = predict_chunks(
                predict_rows, rows=query_count, batch_size=limit, device=runner.device,
            )
            return concatenate_parts(list(arrays), axis=1)

        return np.concatenate(self._map_lanes(predict_lane), axis=0)

    def build_cache(self, dataset):
        train = dataset.training_table()
        groups = self._map_lanes(
            lambda runner, section: runner.build_cache(_slice_dataset(dataset, section), train=train),
        )
        return tuple(cache for group in groups for cache in group)

    @staticmethod
    def _lane_caches(caches, section):
        start = 0
        selected = []
        for cache in caches:
            stop = start + cache.context_shape[0]
            if start < section.stop and stop > section.start:
                if start < section.start or stop > section.stop:
                    cache = cache.members(slice(max(0, section.start - start), min(stop, section.stop) - start))
                selected.append(cache)
            if stop >= section.stop:
                break
            start = stop
        return tuple(selected)

    def place_caches(self, caches):
        groups = self._map_lanes(lambda runner, section: runner.place_caches(self._lane_caches(caches, section)))
        return tuple(cache for group in groups for cache in group)

    def cached_query_window(self, dataset, caches, batch_size, rows, *, levels=None):
        # Count the eventual outputs of the entire ensemble even for asymmetric lane capacities.
        limits = (
            runner.cached_query_window(
                dataset, self._lane_caches(caches, section), batch_size, rows,
                levels=levels, concurrent_lanes=len(self._lanes),
            )
            for runner, section in self._lanes
        )
        return min((limit for limit in limits if limit > batch_size), default=0)

    def predict_cached(self, dataset, query, caches, *, output_type=None, levels=None):
        return self.predict_cached_window(dataset, [query], caches, output_type=output_type, levels=levels)[0]

    def predict_cached_window(self, dataset, queries, caches, *, output_type=None, levels=None):
        def predict_lane(runner, section):
            view = _slice_dataset(dataset, section)
            lane_caches = self._lane_caches(caches, section)
            max_members = max(cache.context_shape[0] for cache in lane_caches)
            options = {"output_type": output_type, "levels": levels}
            parts = []
            ends = [0]
            for query in queries:
                limit = runner.query_device_limit(view, len(query), cached=True, members=max_members)
                parts.extend(select_query_rows(query, slice(start, start + limit)) for start in range(0, len(query), limit))
                ends.append(len(parts))

            max_rows = max(map(len, parts))
            rows = sum(map(len, parts))
            window = runner.cached_query_window(
                dataset, lane_caches, max_rows, rows, levels=levels, concurrent_lanes=len(self._lanes),
            ) if len(parts) > 1 else 0
            count = len(parts) if window >= rows else max(1, window // max_rows)
            outputs = []
            for start in range(0, len(parts), count):
                group = parts[start:start + count]
                if len(group) > 1:
                    try:
                        completed = runner.predict_cached_window(view, group, lane_caches, **options)
                    except (RuntimeError, MemoryError) as error:
                        if not is_allocation_error(error):
                            raise
                    else:
                        outputs.extend(completed)
                        continue
                # Retry only this lane's uncommitted rows, using its normal allocation fallback.
                for part in group:
                    results = predict_chunks(
                        lambda rows: runner.predict_cached(view, select_query_rows(part, rows), lane_caches, **options),
                        rows=len(part), batch_size=len(part), device=runner.device,
                    )
                    outputs.append(concatenate_parts(list(results), axis=1))
            return [concatenate_parts(outputs[a:b], axis=1) for a, b in zip(ends[:-1], ends[1:], strict=True)]

        lanes = self._map_lanes(predict_lane)
        return [concatenate_parts(list(parts), axis=0) for parts in zip(*lanes, strict=True)]
