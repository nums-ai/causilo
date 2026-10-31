"""Statistics for integer row multiplicities, without materializing repeated rows."""

import math

import numpy as np
import scipy
import sklearn
from scipy import optimize, special, stats


def validate_frequencies(sample_weight, rows):
    """Return owned integer counts; reject importance weights and unsafe totals."""
    weights = np.asarray(sample_weight)
    if weights.shape != (rows,) or weights.dtype.kind not in "iuf":
        raise ValueError("sample_weight must contain one integer frequency per training row")
    if not np.isfinite(weights).all() or (weights < 0).any() or (weights != np.floor(weights)).any():
        raise ValueError("sample_weight frequencies must be finite nonnegative integers")
    # Float64 preprocessing must be able to represent the total exactly. Check
    # before casting so unsigned integers and large floats cannot wrap around.
    if not 0 < sum(map(int, weights)) <= 2**53:
        raise ValueError("Total sample_weight frequency must be between 1 and 2**53")
    return weights.astype(np.int64, copy=True)


def weighted_mean_var(values, weights):
    mean = np.average(values, axis=0, weights=weights)
    variance = np.average((values - mean) ** 2, axis=0, weights=weights)
    return mean, variance


def frequency_quantile(column, weights, probabilities):
    """Match NumPy's linear quantile on repeated values, including even medians."""
    order = np.argsort(column, kind="stable")
    column, weights = column[order], weights[order]
    cumulative = np.cumsum(weights)
    positions = (int(cumulative[-1]) - 1) * np.asarray(probabilities)
    lower = np.floor(positions).astype(np.int64)
    upper = np.ceil(positions).astype(np.int64)
    lo = column[np.searchsorted(cumulative, lower, side="right")]
    hi = column[np.searchsorted(cumulative, upper, side="right")]
    return lo + (hi - lo) * (positions - lower)


def weighted_outlier_bounds(values, weights):
    lower, upper = np.zeros(values.shape[1]), np.zeros(values.shape[1])
    for index, column in enumerate(values.T):
        observed = ~np.isnan(column)
        column, frequency = column[observed], weights[observed]
        total = int(frequency.sum())
        if not total:
            continue
        correction = int(total > 1)
        mean = np.average(column, weights=frequency)
        variance = np.dot(frequency, (column - mean) ** 2) / (total - correction)
        preliminary = max(math.sqrt(variance), 1e-6)
        central = np.abs(column - mean) <= 4 * preliminary
        selected, frequency = column[central], frequency[central]
        total = int(frequency.sum())
        center = np.average(selected, weights=frequency) if total else 0.0
        variance = np.dot(frequency, (selected - center) ** 2) / (total - correction) if total > correction else 1.0
        spread = max(math.sqrt(variance), 1e-6)
        lower[index], upper[index] = center - 4 * spread, center + 4 * spread
    return lower, upper


def _weighted_log_var(log_values, frequency):
    total = frequency.sum()
    log_mean = special.logsumexp(log_values, b=frequency) - np.log(total)
    log_diff, _ = special.logsumexp(
        np.stack((log_values, np.full_like(log_values, log_mean))),
        b=np.stack((np.ones_like(log_values), -np.ones_like(log_values))),
        axis=0, return_sign=True,
    )
    return special.logsumexp(2 * log_diff, b=frequency) - np.log(total)


def weighted_yeojohnson_lambda(column, frequency):
    """Maximize the repeated-sample likelihood using the installed optimizer policy.

    sklearn 1.6 uses Brent on the transformed variance; newer versions delegate
    to SciPy, which switched to overflow-safe bounded search in SciPy 1.12.
    """
    total = int(frequency.sum())
    mean, variance = weighted_mean_var(column, frequency)
    eps = np.finfo(np.float64).eps
    if variance <= total * eps * variance + (total * mean * eps) ** 2:
        return 1.0
    jacobian = np.dot(frequency, np.sign(column) * np.log1p(np.abs(column)))

    legacy = tuple(map(int, sklearn.__version__.split(".")[:2])) < (1, 7)
    bounded = not legacy and tuple(map(int, scipy.__version__.split(".")[:2])) >= (1, 12)

    def objective(parameter):
        if legacy:
            transformed = stats.yeojohnson(column, parameter)
            _, var = weighted_mean_var(transformed, frequency)
            logvar = np.log(var) if var >= np.finfo(np.float64).tiny else -np.inf
        elif np.all(column >= 0):
            if abs(parameter) < eps:
                _, var = weighted_mean_var(np.log1p(column), frequency)
                logvar = np.log(var)
            else:
                logvar = _weighted_log_var(parameter * np.log1p(column), frequency) - 2 * np.log(abs(parameter))
        elif np.all(column < 0):
            if abs(parameter - 2) < eps:
                _, var = weighted_mean_var(np.log1p(-column), frequency)
                logvar = np.log(var)
            else:
                logvar = _weighted_log_var((2 - parameter) * np.log1p(-column), frequency) - 2 * np.log(abs(2 - parameter))
        else:
            transformed = stats.yeojohnson(column, parameter)
            _, var = weighted_mean_var(transformed, frequency)
            logvar = np.log(var) if var >= np.finfo(np.float64).tiny else -np.inf
        likelihood = -total / 2 * logvar + (parameter - 1) * jacobian
        return -likelihood if np.isfinite(likelihood) else np.inf

    if not bounded:
        return optimize.brent(objective, brack=(-2, 2))
    max_log = np.log1p(20 * np.max(np.abs(column)))
    info = np.finfo(np.float64)
    lower = (np.log(info.tiny) - np.log(info.eps)) / (2 * max_log)
    upper = (np.log(info.max) + np.log(info.eps)) / (2 * max_log)
    if np.all(column < 0):
        lower, upper = 2 - upper, 2 - lower
    elif np.any(column < 0):
        lower, upper = max(2 - upper, lower), min(2 - lower, upper)
    return optimize.fminbound(objective, lower, upper, xtol=1.48e-8)
