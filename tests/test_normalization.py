import pickle

import numpy as np
import pytest
from sklearn.preprocessing import PowerTransformer

from causilo import CausiloClassifier, CausiloRegressor
from causilo.data.normalization import Normalizer, compress_tails, outlier_bounds


def legacy_power_normalizer(table):
    """Reproduce the original power path with sklearn's internal scaler."""
    result = Normalizer.fit(table, "none")
    filled = np.where(np.isnan(table), result.fill, table)
    scaled = (filled - result.center) / result.spread
    result.normalizer = PowerTransformer().fit(scaled)
    result.lower, result.upper = outlier_bounds(result.normalizer.transform(scaled))
    return result


def test_finite_power_outputs_match_original_path_exactly():
    rng = np.random.default_rng(271)
    training = rng.lognormal(size=(200, 6))
    training[::7, 2] = np.nan
    training[:, 4] = 3.0
    query = rng.lognormal(size=(45, 6))
    query[::3, 1] = np.nan
    original = legacy_power_normalizer(training)
    guarded = Normalizer.fit(training, "power")
    np.testing.assert_array_equal(original.normalizer.lambdas_, guarded.normalizer.power.lambdas_)
    for values in (training, query):
        np.testing.assert_array_equal(original.transform(values), guarded.transform(values))


def test_power_overflow_stays_missing_without_changing_other_cells_or_state():
    training = np.zeros((100, 2))
    training[0] = [-1, 1]
    query = np.array([[1e10, 0], [0, -1e10], [-1, 1], [np.nan, 0]])
    input_copy = query.copy()
    original = legacy_power_normalizer(training)
    guarded = Normalizer.fit(training, "power")
    before = pickle.dumps(guarded)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="infinity|too large"):
            original.transform(query)
    output = guarded.transform(query)
    generated = np.array([[True, False], [False, True], [False, False], [False, False]])
    np.testing.assert_array_equal(np.isnan(output), np.isnan(query) | generated)
    assert not np.isinf(output).any()
    safe_query = np.where(generated, guarded.fill, query)
    np.testing.assert_array_equal(output[~generated], original.transform(safe_query)[~generated])
    np.testing.assert_array_equal(guarded.transform(training), original.transform(training))
    np.testing.assert_array_equal(output, np.concatenate([guarded.transform(row[None]) for row in query]))
    np.testing.assert_array_equal(query, input_copy)
    assert pickle.dumps(guarded) == before
    np.testing.assert_array_equal(pickle.loads(before).transform(query), output)


def test_scaling_overflow_also_stays_missing():
    training = np.array([[-1.0], [0.0], [1.0]])
    guarded = Normalizer.fit(training, "power")
    # A fitted scaler with a very narrow training distribution can overflow
    # even when the distribution transform itself produces a finite value.
    guarded.normalizer.scale.scale_[:] = 1e-310
    output = guarded.normalizer.transform(np.array([[1.0], [0.0]]))
    assert np.isnan(output[0, 0])
    assert not np.isinf(output).any()


def test_outlier_bounds_ignore_missing_observations_per_column():
    values = np.array([[1.0, np.nan, 7.0, 1.0], [2.0, np.nan, np.nan, 2.0],
                       [np.nan, np.nan, np.nan, 3.0], [3.0, np.nan, np.nan, 4.0]])
    lower, upper = outlier_bounds(values)
    assert lower[1] == upper[1] == 0
    for index in (0, 2, 3):
        observed = values[~np.isnan(values[:, index]), index, None]
        expected = outlier_bounds(observed)
        assert lower[index] == expected[0][0]
        assert upper[index] == expected[1][0]
    np.testing.assert_array_equal(np.isnan(compress_tails(values, lower, upper)), np.isnan(values))


@pytest.mark.pretrained
@pytest.mark.parametrize("constructor", [CausiloClassifier, CausiloRegressor])
def test_power_overflow_produces_finite_ensemble_predictions(constructor):
    training = np.zeros((100, 2))
    training[0] = [-1, 1]
    query = np.array([[1e10, 0], [0, -1e10], [-1, 1], [np.nan, 0]])
    targets = np.arange(len(training)) % 2
    model = constructor(n_estimators=8, device="cpu").fit(training, targets)
    predictions = model.predict(query)
    assert np.isfinite(predictions).all()
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(model)).predict(query), predictions)
