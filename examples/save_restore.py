"""Save fitted context and restore it without another fit."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from _model_options import model_options
from sklearn.datasets import load_diabetes

from causilo import CausiloRegressor

options = model_options(__doc__)

X, y = load_diabetes(return_X_y=True, as_frame=True)
model = CausiloRegressor(use_kv_cache=True, **options).fit(X.iloc[:100], y.iloc[:100])
query = X.iloc[100:105]
expected = model.predict(query)
with TemporaryDirectory() as folder:
    path = Path(folder) / "context.pkl"
    model.save_context(path)
    restored = CausiloRegressor.load_context(path, device=model.device)
    np.testing.assert_array_equal(restored.predict(query), expected)
    print(restored.predict(query))
