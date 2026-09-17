import pickle

import numpy as np
import pytest

from causilo.ecoc import ECOCCodec


@pytest.mark.parametrize("classes,rows", [(11, 8), (20, 8), (72, 8), (73, 9), (200, 23), (201, 23)])
def test_codebook_coverage_and_reproducibility(classes, rows):
    codec = ECOCCodec(classes, 10, 42)
    np.testing.assert_array_equal(codec.codebook, ECOCCodec(classes, 10, 42).codebook)
    assert codec.codebook.shape == (rows, classes)
    active = codec.codebook != codec.rest_symbol
    assert np.all(active.sum(axis=1) == 9)
    assert np.all(active.sum(axis=0) >= 1)
    assert np.ptp(active.sum(axis=0)) <= 1
    for row in codec.codebook:
        np.testing.assert_array_equal(np.sort(row[row != codec.rest_symbol]), np.arange(9))
    assert np.unique(codec.codebook.T, axis=0).shape[0] == classes
    restored = pickle.loads(pickle.dumps(codec))
    np.testing.assert_array_equal(restored.codebook, codec.codebook)
    assert not codec.codebook.flags.writeable and not restored.codebook.flags.writeable


@pytest.mark.parametrize("classes", [11, 201])
def test_encode_decode_recovers_labels(classes):
    codec = ECOCCodec(classes, 10, 42)
    labels = np.arange(classes)[::-1]
    encoded = codec.encode(labels)
    np.testing.assert_array_equal(encoded, codec.codebook[:, labels])
    probabilities = np.full((len(codec.codebook), classes, 10), 0.01 / 9)
    np.put_along_axis(probabilities, encoded[:, :, None], 0.99, axis=2)
    decoded = codec.decode(probabilities)
    np.testing.assert_array_equal(decoded.argmax(axis=1), labels)
    assert np.isfinite(decoded).all() and np.all(decoded >= 0)
    np.testing.assert_allclose(decoded.sum(axis=1), 1.0)
    np.testing.assert_array_equal(codec.decode_rows(iter(probabilities)), decoded)


def test_decode_matches_active_probability_mean():
    codec = ECOCCodec(20, 10, 42)
    weights = np.random.default_rng(42).uniform(0.01, 1, (len(codec.codebook), 3, 10))
    probabilities = weights / weights.sum(axis=2, keepdims=True)
    expected = np.empty((3, 20))
    for label in range(20):
        active = np.flatnonzero(codec.codebook[:, label] != codec.rest_symbol)
        selected = probabilities[active, :, codec.codebook[active, label]]
        expected[:, label] = selected.prod(axis=0) ** (1 / len(active))
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(codec.decode(weights), expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    "arguments",
    [(10, 10, 42), (11, 1, 42), (11, 10, -1), (11, 10, 2**32), (True, 10, 42), (11, 10, 1.5)],
)
def test_invalid_configuration(arguments):
    with pytest.raises(ValueError):
        ECOCCodec(*arguments)


@pytest.mark.parametrize("labels", [np.array([0.5]), np.array([-1]), np.array([11]), np.ones((2, 2))])
def test_invalid_labels(labels):
    with pytest.raises(ValueError):
        ECOCCodec(11, 10, 42).encode(labels)


@pytest.mark.parametrize(
    "probabilities",
    [
        np.ones((1, 2, 10)),
        np.ones((8, 2, 9)),
        np.full((8, 2, 10), np.nan),
        np.full((8, 2, 10), np.inf),
        np.full((8, 2, 10), -0.1),
        np.zeros((8, 2, 10)),
    ],
)
def test_invalid_probabilities(probabilities):
    with pytest.raises(ValueError):
        ECOCCodec(11, 10, 42).decode(probabilities)


@pytest.mark.parametrize(
    "rows",
    [
        [np.full((2, 10), 0.1)] * 7,
        [np.full((2, 10), 0.1)] * 9,
        [np.full((2, 10), 0.1), np.full((3, 10), 0.1)],
    ],
)
def test_inconsistent_streamed_rows(rows):
    with pytest.raises(ValueError):
        ECOCCodec(11, 10, 42).decode_rows(iter(rows))


def test_empty_queries():
    codec = ECOCCodec(11, 10, 42)
    assert codec.encode(np.array([], dtype=np.int64)).shape == (8, 0)
    assert codec.decode(np.empty((8, 0, 10))).shape == (0, 11)
