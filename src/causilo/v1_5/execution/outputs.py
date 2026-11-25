"""Restore Causilo 1.5 member outputs to their fitted class order and target units."""

import numpy as np

MEAN_LEVELS = np.r_[0.0001, np.arange(1, 1000) / 1000, 0.9999]
MEDIAN_LEVEL = np.array([0.5])
_MEAN_MASS = np.diff(np.r_[0.0, MEAN_LEVELS, 1.0])
_MEAN_MASS /= _MEAN_MASS.sum()
_MEAN_WEIGHTS = (_MEAN_MASS[:-1] + _MEAN_MASS[1:]) * 0.5


def _average_members(aligned, trim, *, dtype=None):
    if trim:
        aligned.sort(axis=0)
        aligned = aligned[trim:-trim]
    return np.mean(aligned, axis=0, dtype=dtype)


def classification_logits(outputs, class_orders):
    """Restore class order and average logits per row and class."""
    aligned = np.empty(outputs.shape, dtype=outputs.dtype)
    for member, output, order in zip(aligned, outputs, class_orders, strict=True):
        member[:] = output[..., order]
    return aligned.mean(axis=0)


def classification_probabilities(outputs, class_orders):
    """Restore class order, average logits and normalize probabilities."""
    probabilities = classification_logits(outputs, class_orders)
    probabilities -= probabilities.max(axis=-1, keepdims=True)
    np.exp(probabilities, out=probabilities)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities


def member_quantile_levels(output_type, levels, signs):
    if output_type != "quantiles":
        return None
    requested = levels.tolist()
    if np.any(signs < 0):
        # A negative-target member contributes the negated complementary quantile.
        return requested + [1 - level for level in requested]
    return requested


def regression_predictions(outputs, target_preprocessor, signs, *, output_type, levels):
    """Restore target signs, average members and restore the original target units."""
    if np.any(signs < 0):
        signs = signs.reshape((-1,) + (1,) * (outputs.ndim - 1))
        if output_type == "quantiles":
            count = len(levels)
            outputs = np.where(signs > 0, outputs[..., :count], -outputs[..., count:])
        elif output_type == "raw":
            outputs = np.where(signs > 0, outputs, -outputs[..., ::-1])
        else:
            outputs = outputs * signs
    # The inverse is affine. Restore once in float64 to preserve small differences
    # around a large target mean and avoid a float64 copy of every member's output.
    return target_preprocessor.inverse_transform(_average_members(outputs, 0, dtype=np.float64))


def trimmed_member_quantiles(output_type, levels):
    """Choose output probabilities and request both target orientations once."""
    if output_type == "mean":
        levels = MEAN_LEVELS
    elif output_type == "median":
        levels = MEDIAN_LEVEL
    forward, reverse = slice(None), slice(None, None, -1)
    if levels is None:
        return levels, None, (forward, reverse)
    positive = levels.astype(np.float32)
    negative = (1 - levels).astype(np.float32)
    if np.array_equal(positive, negative[::-1]):
        return levels, positive.tolist(), (forward, reverse)
    unique, indices = np.unique(np.concatenate((positive, negative)), return_inverse=True)
    return levels, unique.tolist(), (indices[:len(levels)], indices[len(levels):])


def trimmed_regression_predictions(
    outputs, target_preprocessor, signs, *, output_type, levels, alignment, target_limits, trim,
):
    """Average the retained member quantiles at each row and probability."""
    count = outputs.shape[-1] if levels is None else len(levels)
    aligned = np.empty((len(signs), *outputs.shape[1:-1], count), dtype=outputs.dtype)
    forward, reverse = alignment
    for member, output, sign in zip(aligned, outputs, signs, strict=True):
        if sign < 0:
            np.negative(output[..., reverse], out=member)
        else:
            member[:] = output[..., forward]
    combined = _average_members(aligned, trim, dtype=np.float64)
    del aligned
    quantiles = target_preprocessor.inverse_transform(combined)
    return finish_trimmed_quantiles(quantiles, output_type, target_limits)


def finish_trimmed_quantiles(quantiles, output_type, target_limits):
    """Return integrated means, medians, or the combined quantiles."""
    if output_type == "median":
        return quantiles[:, 0].copy()
    if output_type == "mean":
        lower = np.minimum(quantiles[:, 0], target_limits[0])
        upper = np.maximum(quantiles[:, -1], target_limits[1])
        # Integrate consecutive trapezoids using the sum of the two weights
        # incident to each interior value, without materializing their edges.
        integrated = np.einsum("ij,j->i", quantiles, _MEAN_WEIGHTS)
        return integrated + lower * (_MEAN_MASS[0] * 0.5) + upper * (_MEAN_MASS[-1] * 0.5)
    return quantiles
