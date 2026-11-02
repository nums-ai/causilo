"""Fit differently sized tables, then request independent regression outputs."""

from sklearn.datasets import make_regression

from causilo import CausiloRegressor

sales, sales_target = make_regression(n_samples=80, n_features=5, random_state=1)
price, price_target = make_regression(n_samples=100, n_features=8, random_state=2)

model = CausiloRegressor(n_estimators=2)
model.fit_batched({
    "sales": (sales[:60], sales_target[:60]),
    "price": (price[:70], price_target[:70]),
})
predictions = model.predict_batched({
    "sales": sales[60:],
    "price": {
        "X": price[70:],
        "output_type": "quantiles",
        "quantiles": [0.1, 0.5, 0.9],
    },
})
for table_id, values in predictions.items():
    print(table_id, values.shape, values[:3])

# The context stays fitted; query a subset without depending on input order.
print("Repeated sales query:", model.predict_batched({"sales": sales[60:62]})["sales"])
