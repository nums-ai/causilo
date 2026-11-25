"""Fit a classifier and inspect class probabilities."""

from _model_options import model_options
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from causilo import CausiloClassifier

options = model_options(__doc__)

X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
model = CausiloClassifier(**options).fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))
print("Probabilities:", model.predict_proba(X_test.iloc[:3]))
