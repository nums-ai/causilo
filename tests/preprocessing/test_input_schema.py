import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from causilo.v1_0.data.dataset import PreparedDataset
from causilo.v1_0.data.encoding import FeatureEncoder
from causilo.v1_5.data import dataset as dataset_v1_5
from causilo.v1_5.data.encoding import FeatureEncoder as FeatureEncoder15


def test_label_encoding_preserves_each_versions_signed_zero_classes():
    table = np.arange(8).reshape(4, 2)
    labels = np.array([0.0, -0.0, 1.0, 1.0])
    options = dict(task="classification", n_estimators=1, random_state=42)
    version_1_0 = PreparedDataset.prepare(
        table, labels, retain_preprocessing=True, class_permutation_size=10, **options,
    )
    version_1_5 = dataset_v1_5.PreparedDataset.prepare(table, labels, **options)
    reference_1_0 = LabelEncoder().fit(labels)
    reference_1_5 = LabelEncoder()
    ids = reference_1_5.fit_transform(labels)
    np.testing.assert_array_equal(version_1_0.targets, reference_1_0.transform(labels))
    np.testing.assert_array_equal(version_1_5.targets, ids)
    np.testing.assert_array_equal(
        np.signbit(version_1_0.target_encoder.classes_), np.signbit(reference_1_0.classes_),
    )
    np.testing.assert_array_equal(
        np.signbit(version_1_5.fitted_attributes()["classes_"]), np.signbit(reference_1_5.classes_),
    )


@pytest.mark.parametrize("encoder_class", [FeatureEncoder, FeatureEncoder15])
def test_named_queries_follow_the_fitted_schema(encoder_class):
    table = pd.DataFrame({"kind": ["a", "b", None], 20: [1., 3., 2.]})
    encoder, _ = encoder_class.fit(table)
    query = encoder.transform
    query_table = pd.DataFrame({"kind": ["unseen", "a"], 20: [5., np.nan]})
    expected = [[np.nan, 5.], [0., np.nan]]
    np.testing.assert_array_equal(query(query_table), expected)
    np.testing.assert_array_equal(query(query_table.to_numpy()), expected)
    if encoder_class is FeatureEncoder:
        with pytest.raises(ValueError, match="name and order"):
            query(query_table.iloc[:, ::-1])
    else:
        np.testing.assert_array_equal(query(query_table.iloc[:, ::-1]), expected)
    with pytest.raises(ValueError, match="Prediction columns must match"):
        query(query_table.rename(columns={20: "other"}))
    with pytest.raises(ValueError, match="Expected 2 feature columns"):
        query(np.zeros((1, 3)))


def test_tuple_column_names_match_after_reordering():
    names = pd.MultiIndex.from_tuples([("inputs", "amount"), ("inputs", "size")])
    table = pd.DataFrame([[1., 0.], [2., 3.], [3., 1.]], columns=names)
    encoder, expected = FeatureEncoder15.fit(table)
    np.testing.assert_array_equal(encoder.transform(table.iloc[:, ::-1]), expected)


@pytest.mark.parametrize(("values", "error"), [
    (pd.date_range("2026-01-01", periods=3, tz="UTC"), "Date and duration"),
    (pd.to_timedelta([1, 2, 3], unit="s"), "Date and duration"),
    (pd.period_range("2026-01-01", periods=3, freq="D"), "Date and duration"),
    (np.array([0j, 1j, 2j]), "Complex feature values"),
    (pd.IntervalIndex.from_breaks([0, 1, 2, 3]), "Unsupported feature dtype"),
])
def test_unsupported_feature_dtypes_fail_in_fit_and_query(values, error):
    train = pd.DataFrame({"amount": [1., 2., 3.], "size": [0., 3., 1.]})
    invalid = train.assign(size=values)
    with pytest.raises(TypeError, match=error):
        FeatureEncoder15.fit(invalid)
    encoder, _ = FeatureEncoder15.fit(train)
    with pytest.raises(TypeError, match=error):
        encoder.query_layout(invalid)


def test_context_requires_nonempty_features_and_one_finite_target_per_row():
    options = dict(task="regression", n_estimators=1, random_state=42)
    table = np.arange(24).reshape(12, 2)
    with pytest.raises(ValueError, match="1d array"):
        dataset_v1_5.PreparedDataset.prepare(table, np.arange(12).reshape(6, 2), **options)
    with pytest.raises(ValueError, match="at least one row"):
        dataset_v1_5.PreparedDataset.prepare([], [], **options)
    with pytest.raises(ValueError, match="Targets cannot contain missing values"):
        dataset_v1_5.PreparedDataset.prepare(table, np.full(12, np.nan), **options)


@pytest.mark.parametrize("fit_encoder", [FeatureEncoder, FeatureEncoder15])
def test_object_columns_preserve_measurements_and_categories(fit_encoder):
    table = np.array([[1.5, "1", True], [None, "2", False], [3.5, None, True]], dtype=object)
    encoder, _ = fit_encoder.fit(table)
    result = encoder.transform(np.array([[2.5, "new", False], [pd.NA, "1", True]], dtype=object))
    np.testing.assert_array_equal(result, [[np.nan, 0, 2.5], [0, 1, np.nan]])


def test_numeric_only_object_table_keeps_unseen_measurements():
    table = np.array([[1.5, 3], [None, 4], [3.5, 5]], dtype=object)
    encoder, _ = FeatureEncoder15.fit(table)
    query = np.array([[2.5, 2], [pd.NA, 6]], dtype=object)
    np.testing.assert_array_equal(encoder.transform(query), [[2.5, 2], [np.nan, 6]])


def test_numeric_categories_keep_explicit_dataframe_type():
    table = pd.DataFrame({"category": pd.Categorical([1, 2, 1]), "value": [1.5, 2.5, 3.5]})
    encoder, _ = FeatureEncoder.fit(table)
    assert encoder.categories == (0,)
    assert encoder.continuous == (1,)
