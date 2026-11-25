"""Predict a set of independently fitted Causilo 1.5 tables."""

from ..batch import prepare_queries
from .execution.prediction import predict_contexts, prepare_query


def predict_batched(engine, tables):
    queries, output_options = prepare_queries(engine, tables, prepare_query)
    return predict_contexts(engine, queries, output_options)
