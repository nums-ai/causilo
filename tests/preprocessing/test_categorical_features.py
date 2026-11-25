import numpy as np
import pandas as pd
import pytest

from causilo.v1_5.data.categorical import CategoricalEncoder
from causilo.v1_5.data.columns import CanonicalColumns
from causilo.v1_5.data.encoding import FeatureEncoder


def test_weighted_dictionary_categories_match_raw_fit():
    names = [f"c{i:02d}" for i in range(55)]
    labels = [*names, *names, None, names[0]]
    table = pd.DataFrame({
        "id": pd.Categorical(labels, categories=["unused", *reversed(names)], ordered=True),
        "bucket": pd.Categorical([None, 3, 7, 3] * 28, categories=[11, 7, 3]),
        "empty": pd.Categorical([None] * len(labels), categories=["unused"]),
    })
    original = table.copy(deep=True)
    counts = np.arange(len(table), dtype=np.int64) % 5 + 1
    plain = table.to_numpy(dtype=object, na_value=np.nan)
    actual, encoded = CategoricalEncoder.fit(table, counts)
    expected, reference = CategoricalEncoder.fit(plain, counts)
    pd.testing.assert_frame_equal(table, original)
    np.testing.assert_array_equal(encoded, reference)
    for left, right in zip(actual.ordinal.categories_, expected.ordinal.categories_):
        pd.testing.assert_index_equal(pd.Index(left, dtype=object), pd.Index(right, dtype=object))
        assert [type(x) for x in left] == [type(x) for x in right]
    assert actual.log_counts.keys() == expected.log_counts.keys() == {0}
    np.testing.assert_array_equal(actual.log_counts[0], expected.log_counts[0])
    query = np.array([["unknown", 3, None], [None, 7, None], [names[0], 7, None]], dtype=object)
    result = actual.transform(query)
    np.testing.assert_array_equal(result, expected.transform(query))
    np.testing.assert_array_equal(result[:2, -1], [0, np.nan])


def test_numeric_categories_interpolate_unseen_interior_values_only():
    context = pd.DataFrame({
        "level": pd.Categorical([9.0, 0.0, 2.0, None, 9.0, 2.0, 0.0]),
        "text": pd.Categorical(["9", "0", "2", None, "9", "2", "0"]),
        "flag": pd.Categorical([True, False, True, None, False, False, True]),
    })
    query = pd.DataFrame({
        "level": pd.Categorical([1.0, 6.0, -1.0, 11.0, 2.0, None, np.inf]),
        "text": pd.Categorical(["1", "6", "-1", "11", "2", None, "inf"]),
        "flag": pd.Categorical([False, False, True, None, True, True, True]),
    })
    encoder, fitted = CategoricalEncoder.fit(context, None)
    np.testing.assert_array_equal(encoder.transform(context), fitted)
    transformed = encoder.transform(query)
    np.testing.assert_allclose(transformed[:, 0], [0.5, 1 + 4 / 7, np.nan, np.nan, 1, np.nan, np.nan])
    np.testing.assert_array_equal(transformed[:, 1], [np.nan, np.nan, np.nan, np.nan, 1, np.nan, np.nan])
    np.testing.assert_array_equal(transformed[:, 2], [0, 0, 1, np.nan, 1, 1, 1])


def test_numeric_category_interpolation_uses_weighted_log_frequencies():
    levels = np.arange(0, 120, 2)
    context = pd.DataFrame({"level": pd.Categorical(levels)})
    weights = np.arange(1, len(levels) + 1)
    encoder, _ = CategoricalEncoder.fit(context, weights)
    query = pd.DataFrame({"level": pd.Categorical([81, -1, 120, None, 80, 82])})
    log_low, log_high = np.log1p([41, 42])
    np.testing.assert_array_equal(encoder.transform(query), [
        [40.5, (log_low + log_high) / 2], [np.nan, 0], [np.nan, 0],
        [np.nan, np.nan], [40, log_low], [41, log_high],
    ])


def test_numeric_category_interpolation_handles_wide_ranges_and_keeps_unsafe_integers_unknown():
    largest = np.finfo(np.float64).max
    wide = np.array([[-largest], [largest]], dtype=object)
    large_ids = np.array([[0], [2**53 + 1]], dtype=object)
    wide_encoder, _ = CategoricalEncoder.fit(wide, None)
    id_encoder, _ = CategoricalEncoder.fit(large_ids, None)
    with np.errstate(over="raise", invalid="raise"):
        np.testing.assert_array_equal(wide_encoder.transform([[0.0]]), [[0.5]])
    np.testing.assert_array_equal(id_encoder.transform([[10], [2**53 + 1]]), [[np.nan], [1]])


def test_object_category_codes_distinguish_unknown_and_missing():
    labels = ["repeat"] * 2 + [f"v{i:02d}" for i in range(51)] + [None]
    encoder, _ = FeatureEncoder.fit(np.array(labels, dtype=object)[:, None])
    rows = np.array(["repeat", "unseen", pd.NA, None, np.nan], dtype=object)[:, None]
    result = encoder.transform(rows)
    np.testing.assert_array_equal(result[:, 0], [0, np.nan, np.nan, np.nan, np.nan])
    np.testing.assert_array_equal(result[:, 1], [np.log1p(2), 0, np.nan, np.nan, np.nan])
    assert rows[2, 0] is pd.NA and rows[3, 0] is None


def test_object_categories_respect_custom_missing_values():
    class MissingLabel(str):
        def __ne__(self, other):
            return True

    class AmbiguousLabel(str):
        def __ne__(self, other):
            raise ValueError("undefined comparison")

    context = np.array([["popular"], ["rare"]], dtype=object)
    encoder, _ = FeatureEncoder.fit(context)
    query = np.array([["popular"], [MissingLabel("popular")], [AmbiguousLabel("popular")]], dtype=object)
    np.testing.assert_array_equal(encoder.transform(query)[:, 0], [0.0, np.nan, np.nan])


def test_numeric_categorical_codes_keep_null_and_unknown_values():
    context = pd.DataFrame({"kind": pd.Categorical([25, 9, None, 25], categories=[25, 9, 100])})
    encoder, values = FeatureEncoder.fit(context)
    np.testing.assert_array_equal(values[:, 0], [1.0, 0.0, np.nan, 1.0])
    query = pd.DataFrame({"kind": pd.Categorical([25, None, 100, 9], categories=[9, 100, 25])})
    np.testing.assert_array_equal(encoder.transform(query)[:, 0], [1.0, np.nan, np.nan, 0.0])
    np.testing.assert_array_equal(query["kind"].cat.codes, [2, -1, 1, 0])


def test_typed_categorical_vocabulary_preserves_custom_missing_values():
    class MissingLabel(str):
        def __ne__(self, other):
            return True

    encoder, _ = FeatureEncoder.fit(pd.DataFrame({"kind": pd.Categorical(["b", "a", "b"])}))
    # Constructing the category index as object keeps the custom scalar on every pandas string backend.
    vocabulary = pd.Index([MissingLabel("a"), "b"], dtype=object)
    query = pd.DataFrame({"kind": pd.Categorical.from_codes([0, 1, -1], categories=vocabulary)})
    np.testing.assert_array_equal(encoder.transform(query)[:, 0], [np.nan, 1.0, np.nan])
    np.testing.assert_array_equal(query["kind"].cat.codes, [0, 1, -1])


@pytest.mark.parametrize(("cardinality", "repetitions", "width"), [(50, 2, 2), (51, 0, 2), (51, 2, 3)])
def test_count_feature_needs_51_observed_categories_and_varying_counts(cardinality, repetitions, width):
    labels = [f"c{index:02d}" for index in range(cardinality)] + ["c00"] * repetitions
    table = pd.DataFrame({"category": labels, "measurement": np.arange(len(labels))})
    query = pd.DataFrame({"category": ["c00", "unseen", pd.NA], "measurement": [999, 2, 0]})
    encoder, context = FeatureEncoder.fit(table)
    columns = CanonicalColumns.fit(context, effective_rows=len(context))
    assert len(columns.order) == width
    encoded = encoder.transform(query)
    np.testing.assert_array_equal(encoded[:, 0], [0.0, np.nan, np.nan])
    np.testing.assert_array_equal(encoded[:, -1], query["measurement"])
    if width == 3:
        np.testing.assert_array_equal(encoded[:, 1], [np.log1p(3), 0.0, np.nan])
