"""Compare warm fit+predict for four small tables; this is not a throughput guarantee.

Run with --device cpu, cuda, or mps. Checkpoints must be available or downloadable.
The median of three trials excludes weight loading and includes preprocessing.
"""

import argparse
import json
import statistics
import time

import numpy as np
import torch

from causilo import CausiloClassifier, CausiloRegressor
from causilo.checkpoints import RELEASE_COMMIT

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
args = parser.parse_args()
torch.set_num_threads(2)
rng = np.random.default_rng(49)
tables = {}
queries = {}
for i in range(4):
    t, q, f = 32 + i * 3, 8 + i, 4 + i
    x = rng.normal(size=(t, f))
    tables[str(i)] = (x, np.arange(t) % 3)
    queries[str(i)] = rng.normal(size=(q, f))

device = args.device
for cls in [CausiloClassifier, CausiloRegressor]:
    single = cls(device=device, n_estimators=2)
    batch = cls(device=device, n_estimators=2)

    def synchronize():
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

    def serial():
        result = {}
        for key, (X, y) in tables.items():
            single.fit(X, y)
            fn = single.predict_proba if cls is CausiloClassifier else single.predict
            result[key] = fn(queries[key])
        return result

    def batched():
        batch.fit_batched(tables)
        fn = batch.predict_proba_batched if cls is CausiloClassifier else batch.predict_batched
        return fn(queries)

    reference, actual = serial(), batched()  # weights and kernels warm
    for key in actual:
        np.testing.assert_allclose(actual[key], reference[key], atol=5e-4, rtol=5e-4)
    measurements = {"serial": [], "batched": []}
    peaks = {"serial": [], "batched": []}
    for _ in range(3):
        for name, fn in [("serial", serial), ("batched", batched)]:
            synchronize()
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            fn()
            synchronize()
            measurements[name].append(time.perf_counter() - started)
            if device == "cuda":
                peaks[name].append(torch.cuda.max_memory_allocated())
    medians = {key: statistics.median(values) for key, values in measurements.items()}
    print(json.dumps({
        "device": device, "task": cls.__name__, "tables": 4, "n_estimators": 2,
        "use_kv_cache": False, "warm_fit_predict_seconds": medians,
        "speedup": medians["serial"] / medians["batched"],
        "cuda_process_peak_allocated_bytes": {
            key: max(values) if values else None for key, values in peaks.items()
        },
        "revision": RELEASE_COMMIT, "torch": torch.__version__,
    }), flush=True)
    single = batch = None
    if device == "mps":
        torch.mps.empty_cache()
