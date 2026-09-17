import pickle

import numpy as np
import pytest

from causilo.ecoc import ECOCCodec, _separation


@pytest.mark.parametrize(
    "classes,rows",
    [
        (11, 8),
        (20, 8),
        (72, 8),
        (73, 9),
        (100, 12),
        (101, 12),
        (108, 12),
        (109, 13),
        (200, 23),
        (201, 23),
        (706, 79),
        (1000, 112),
    ],
)
def test_codebook_is_deterministic_complete_and_separating(classes, rows):
    first = ECOCCodec(classes, 10, 42)
    second = ECOCCodec(classes, 10, 42)
    np.testing.assert_array_equal(first.codebook, second.codebook)
    assert first.codebook.shape == (rows, classes)
    assert np.all((first.codebook >= 0) & (first.codebook < 10))
    assert np.all(np.any(first.codebook != first.rest_symbol, axis=0))
    active = first.codebook != first.rest_symbol
    assert np.all(active.sum(axis=1) == 9)
    assert np.ptp(active.sum(axis=0)) <= 1
    for row in first.codebook:
        np.testing.assert_array_equal(np.sort(row[row != first.rest_symbol]), np.arange(9))
    assert len({tuple(column) for column in first.codebook.T}) == classes
    assert not first.codebook.flags.writeable


def test_encode_and_log_likelihood_decode_recover_classes():
    codec = ECOCCodec(20, 10, 17)
    labels = np.arange(20)
    encoded = codec.encode(labels)
    rows = np.full((len(codec.codebook), len(labels), 10), 0.01 / 9)
    for row in range(len(codec.codebook)):
        rows[row, np.arange(len(labels)), encoded[row]] = 0.99
    decoded = codec.decode(rows)
    np.testing.assert_array_equal(decoded.argmax(axis=1), labels)
    np.testing.assert_allclose(decoded.sum(axis=1), 1.0)
    assert np.isfinite(decoded).all()


def test_codec_pickle_preserves_read_only_codebook():
    restored = pickle.loads(pickle.dumps(ECOCCodec(11, 10, 4)))
    assert not restored.codebook.flags.writeable
    np.testing.assert_array_equal(restored.codebook, ECOCCodec(11, 10, 4).codebook)


@pytest.mark.parametrize(
    "arguments",
    [
        (10, 10, 42),
        (11, 1, 42),
        (11, 10, -1),
        (11, 10, 2**32),
        (True, 10, 42),
        (11, 10, 1.5),
    ],
)
def test_invalid_codec_configuration(arguments):
    with pytest.raises(ValueError):
        ECOCCodec(*arguments)


def test_decode_rejects_invalid_probabilities():
    codec = ECOCCodec(11, 10, 42)
    with pytest.raises(ValueError, match="shape"):
        codec.decode(np.ones((1, 2, 10)))
    invalid = np.ones((len(codec.codebook), 2, 10))
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        codec.decode(invalid)


@pytest.mark.parametrize("value", [np.array(1), np.ones((8, 10)), np.ones((8, 2, 9))])
def test_decode_rejects_wrong_rank_and_width(value):
    with pytest.raises(ValueError, match="shape"):
        ECOCCodec(11, 10, 42).decode(value)


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1])
def test_decode_rejects_nonfinite_and_negative_values(value):
    codec = ECOCCodec(11, 10, 42)
    probabilities = np.full((len(codec.codebook), 2, 10), 0.1)
    probabilities[0, 0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        codec.decode(probabilities)


def test_decode_rejects_zero_vectors():
    codec = ECOCCodec(11, 10, 42)
    with pytest.raises(ValueError, match="positive sums"):
        codec.decode(np.zeros((len(codec.codebook), 2, 10)))


@pytest.mark.parametrize("labels", [np.array([0.5]), np.array([-1]), np.array([11]), np.ones((2, 2))])
def test_encode_rejects_invalid_labels(labels):
    with pytest.raises(ValueError):
        ECOCCodec(11, 10, 42).encode(labels)


def test_encode_empty_queries_and_decode_empty_queries():
    codec = ECOCCodec(11, 10, 42)
    assert codec.encode(np.array([], dtype=np.int64)).shape == (8, 0)
    assert codec.decode(np.empty((8, 0, 10))).shape == (0, 11)


def test_global_random_state_is_unchanged():
    saved = np.random.get_state()
    ECOCCodec(20, 10, 42)
    after = np.random.get_state()
    assert saved[0] == after[0] and saved[2:] == after[2:]
    np.testing.assert_array_equal(saved[1], after[1])


@pytest.mark.parametrize("classes,symbols", [(7, 2), (17, 3), (23, 10), (100, 10)])
def test_cooccurrence_score_matches_direct_hamming_distance(classes, symbols):
    codec = ECOCCodec(classes, symbols, 11)
    distances = [
        sum(a != b for a, b in zip(codec.codebook[:, left], codec.codebook[:, right]))
        for left in range(classes)
        for right in range(left + 1, classes)
    ]
    assert _separation(codec.codebook, codec.rest_symbol) == (min(distances), np.mean(distances))


@pytest.mark.parametrize("classes,symbols", [(7, 2), (17, 3), (23, 10), (201, 10)])
def test_streaming_decode_matches_geometric_mean_definition(classes, symbols):
    codec = ECOCCodec(classes, symbols, 91)
    weights = np.random.default_rng(17).uniform(0.01, 1.0, (len(codec.codebook), 5, symbols))
    normalized = weights / weights.sum(axis=2, keepdims=True)
    expected = np.empty((5, classes))
    for label in range(classes):
        indices = np.flatnonzero(codec.codebook[:, label] != codec.rest_symbol)
        selected = normalized[indices, :, codec.codebook[indices, label]]
        expected[:, label] = selected.prod(axis=0) ** (1.0 / len(indices))
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(codec.decode(weights), expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(codec.decode_rows(row for row in weights), codec.decode(weights))


@pytest.mark.parametrize("delta", [-1, 1])
def test_streaming_decode_rejects_wrong_code_row_count(delta):
    codec = ECOCCodec(11, 10, 42)
    with pytest.raises(ValueError, match="code rows"):
        codec.decode_rows(np.full((len(codec.codebook) + delta, 2, 10), 0.1))


def test_streaming_decode_rejects_inconsistent_query_counts():
    codec = ECOCCodec(11, 10, 42)
    with pytest.raises(ValueError, match="query count"):
        codec.decode_rows([np.full((2, 10), 0.1), np.full((3, 10), 0.1)])


def test_streaming_decode_rejects_wrong_output_width():
    with pytest.raises(ValueError, match="shape"):
        ECOCCodec(11, 10, 42).decode_rows([np.ones((2, 9))])


def test_decode_accepts_large_finite_unnormalized_weights():
    codec = ECOCCodec(11, 10, 42)
    weights = np.full((len(codec.codebook), 2, 10), np.finfo(np.float64).max)
    np.testing.assert_allclose(codec.decode(weights), np.full((2, 11), 1 / 11))


def test_decode_clips_zero_active_probabilities():
    codec = ECOCCodec(11, 10, 42)
    probabilities = np.zeros((len(codec.codebook), 2, 10))
    probabilities[:, :, codec.rest_symbol] = 1
    np.testing.assert_allclose(codec.decode(probabilities), np.full((2, 11), 1 / 11))
