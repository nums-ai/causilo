"""Save fitted context and restore it without another fit."""

from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
from sklearn.datasets import load_diabetes

from causilo import CausiloRegressor

X, y = load_diabetes(return_X_y=True, as_frame=True)
model = CausiloRegressor(use_kv_cache=True).fit(X.iloc[:100], y.iloc[:100])
query = X.iloc[100:105]
expected = model.predict(query)
with TemporaryDirectory() as folder:
    path = Path(folder) / "fitted.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(restored.predict(query), expected)
    print(restored.predict(query))
