"""Shared output contracts and ensemble reconstruction for both inference APIs."""

import numpy as np

from .quantiles import interpolate_quantiles


def select_regression_output(result, output_type, levels):
    if output_type == "median":
        # A point output must not retain the full native-quantile matrix.
        return result[:, result.shape[1] // 2].copy()
    if output_type == "quantiles":
        return interpolate_quantiles(result, levels)
    return result


def reduce_regression_output(result, output_type):
    """Sort on the execution device, preserving the point prediction's sum order."""
    ordered = result.sort(dim=-1).values
    return ordered.mean(dim=-1) if output_type == "mean" else ordered


def classification_logits(predictions):
    """Restore class order and average (member, CPU logits) pairs in fitted order."""
    logits = [result[:, member.class_order].numpy() for member, result in predictions]
    return np.mean(logits, axis=0)


def classification_probabilities(predictions):
    """Apply softmax to the class-aligned, ensemble-averaged logits."""
    combined = classification_logits(predictions)
    probabilities = np.exp(combined - combined.max(axis=-1, keepdims=True))
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def regression_predictions(predictions, target_encoder):
    """Restore each CPU member output to target units before averaging."""
    restored = []
    for result in predictions:
        values = result.numpy().astype(np.float64)
        restored.append(target_encoder.inverse_transform(values.reshape(-1, 1)).reshape(values.shape))
    return np.mean(restored, axis=0)
