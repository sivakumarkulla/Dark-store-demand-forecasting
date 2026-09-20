"""
Train a 6-hour-ahead demand forecaster per category and SAVE the models.

Target  : total units sold in the next 6 hours (t+1 .. t+6)
Models  : XGBoost point forecast + XGBoost 90th-percentile forecast (used for alerts)
Baseline: "same 6-hour window one week ago"
Split   : time-based (train before SPLIT, test after). Never random.

Run from project root:  python src/train.py
"""
import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from features import build

SPLIT = "2026-04-15"   # train before, test after

df = pd.read_csv("data/hourly_sales.csv", parse_dates=["time"])
os.makedirs("models", exist_ok=True)


def fit(X, y, **kw):
    return xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=0, **kw).fit(X, y)


results = []
for cat, d in df.groupby("category"):
    X, y, base = build(d)
    ok = X.notna().all(axis=1) & y.notna() & base.notna()
    X, y, base = X[ok], y[ok], base[ok]
    train = X["time"] < SPLIT
    feats = [c for c in X.columns if c != "time"]

    point = fit(X.loc[train, feats], y[train])
    p90 = fit(X.loc[train, feats], y[train], objective="reg:quantileerror", quantile_alpha=0.9)
    p90.save_model(f"models/p90_{cat}.json")          # used by alerts.py
    point.save_model(f"models/point_{cat}.json")

    Xt, yt, bt = X.loc[~train, feats], y[~train], base[~train]
    pred, hi = point.predict(Xt), p90.predict(Xt)
    surge = yt >= yt.quantile(0.90)                    # top 10% demand windows in test

    results.append({
        "category": cat,
        "MAE model": mean_absolute_error(yt, pred),
        "MAE baseline": mean_absolute_error(yt, bt),
        "surge MAE model": mean_absolute_error(yt[surge], pred[surge]),
        "surge MAE baseline": mean_absolute_error(yt[surge], bt[surge]),
        "p90 coverage": (yt.to_numpy() <= hi).mean(),
        "p90 covers surges": (yt[surge].to_numpy() <= hi[surge.to_numpy()]).mean(),
    })

    imp = pd.Series(point.feature_importances_, index=feats).sort_values(ascending=False)
    print(f"\n[{cat}] top features:", ", ".join(f"{k} ({v:.2f})" for k, v in imp.head(6).items()))

print("\n=== Results on held-out test period (units per 6h window) ===")
print(pd.DataFrame(results).set_index("category").round(2).T)
print("\nSaved models to models/")
