"""ID-based fitted contexts and per-table output reconstruction."""

import numpy as np
import pandas as pd

from ..batch import prepare_queries
from .prediction import predict_contexts


def prepare_query(state, table):
    shape = np.shape(table)
    width = state.dataset.encoder.width
    if len(shape) != 2 or (shape[1] != width and not isinstance(table, pd.DataFrame)):
        raise ValueError(f"Expected a two-dimensional table with {width} feature columns")
    if shape[0] == 0:
        raise ValueError("Query table must contain at least one row")
    return state.dataset.query_tables(table)


def predict_batched(engine, tables):
    queries, output_options = prepare_queries(engine, tables, prepare_query)
    states = {key: engine.batch_states[key] for key in queries}
    return predict_contexts(engine.model, states, queries, output_options, engine.device)
