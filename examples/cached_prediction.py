"""Reuse fitted attention context for successive prediction calls."""

from _model_options import model_options
from sklearn.datasets import load_iris

from causilo import CausiloClassifier

options = model_options(__doc__)

X, y = load_iris(return_X_y=True, as_frame=True)
model = CausiloClassifier(use_kv_cache=True, **options).fit(X.iloc[::2], y.iloc[::2])
for start in (1, 11, 21):
    print(model.predict_proba(X.iloc[start:start + 5]))
