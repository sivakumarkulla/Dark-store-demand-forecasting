"""
Live alert engine.

Every run:
  1. Fetch recent + forecast weather (Open-Meteo)
  2. Build the same features the model was trained on
  3. Predict p90 demand for the next 6 hours per category
  4. Compare to current stock -> alert (print / Discord) and log

WHAT IS LIVE vs SIMULATED:
  live      : the clock and the weather forecast
  simulated : recent sales history (we generate it from the real recent weather,
              because there is no real sales feed). Stock levels are set on the dashboard.
  In a real deployment those two would be database queries.

Run from project root:
    python src/live_alerts.py            # run once
    python src/live_alerts.py --loop     # run every hour (Ctrl+C to stop)
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from alerts import LOG_PATH, SAFETY_PCT, get_stock, send_discord
from features import H, build
from simdata import add_flags, simulate_units
from weather import get_recent_and_forecast


def run_once(wx=None, t=None):
    """wx / t can be passed in for offline testing; normally both come from live data."""
    if wx is None:
        wx = get_recent_and_forecast()
    if t is None:
        # decision time = the last fully completed hour (IST)
        t = pd.Timestamp.now(tz="Asia/Kolkata").floor("h").tz_localize(None) - pd.Timedelta(hours=1)

    wx = wx.dropna()
    wx = wx[wx["time"] <= t + pd.Timedelta(hours=H)]           # history + next 6 forecast hours
    if (wx["time"] <= t).sum() < 170 or wx["time"].max() < t + pd.Timedelta(hours=H):
        raise RuntimeError("Not enough weather history/forecast around the decision time.")

    df = add_flags(wx)                                          # no IPL matches outside Mar-May
    sales = simulate_units(df, np.random.default_rng(123))      # SIMULATED recent sales
    sales.loc[sales["time"] > t, "units"] = np.nan              # the future is unknown

    print(f"\n=== Live check at {t + pd.Timedelta(hours=1)} "
          f"(features up to {t}, forecasting next {H}h) ===\n")
    stock_levels = get_stock()
    log_rows = []
    for cat, d in sales.groupby("category"):
        X, _, _ = build(d)
        row = X[X["time"] == t]
        feats = [c for c in X.columns if c != "time"]
        if row.empty or row[feats].isna().any(axis=None):
            print(f"[{cat}] incomplete features, skipping.")
            continue

        model = xgb.XGBRegressor()
        model.load_model(f"models/p90_{cat}.json")
        p90 = float(model.predict(row[feats])[0])
        needed = p90 * (1 + SAFETY_PCT)
        stock = stock_levels[cat]
        short = needed - stock
        rain = float(row["rain_next6"].iloc[0])

        if short > 0:
            msg = (f"ALERT  {cat}: forecast (p90) {p90:.0f} units in next {H}h, stock {stock}. "
                   f"Short by {short:.0f}. Restock now.")
            send_discord(msg)
        else:
            msg = f"OK     {cat}: forecast (p90) {p90:.0f}, stock {stock}. Enough cover."
        print(msg)
        print(f"       forecast rain next {H}h = {rain:.1f} mm")

        log_rows.append({"time": t, "category": cat, "p90_forecast": round(p90, 1),
                         "stock": stock, "shortfall": round(max(short, 0), 1),
                         "alert": int(short > 0), "actual_demand": np.nan})

    if log_rows:
        pd.DataFrame(log_rows).to_csv(LOG_PATH, mode="a", index=False,
                                      header=not os.path.exists(LOG_PATH))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true", help="repeat every hour")
    args = ap.parse_args()

    if not args.loop:
        run_once()
    else:
        while True:
            try:
                run_once()
            except Exception as e:                              # keep the loop alive
                print(f"Run failed: {e}")
            nxt = pd.Timestamp.now().floor("h") + pd.Timedelta(hours=1, minutes=1)
            time.sleep(max(60, (nxt - pd.Timestamp.now()).total_seconds()))
