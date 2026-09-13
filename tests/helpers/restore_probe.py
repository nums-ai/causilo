"""Exercise restoration in a fresh validation process without rebuilding caches."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch

from causilo.execution.runner import ModelRunner


def reject_rebuild(*args):
    raise AssertionError("Restoration unexpectedly rebuilt fitted context")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("saved", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    ModelRunner.build = reject_rebuild
    torch.set_num_threads(2)
    estimator, query = joblib.load(arguments.saved)
    np.save(arguments.output, estimator.predict(query))
