"""Fit a regressor and inspect point predictions."""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from causilo import CausiloRegressor

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = CausiloRegressor().fit(X_train, y_train)
print("R-squared:", model.score(X_test, y_test))
print("Predictions:", model.predict(X_test.iloc[:3]))
