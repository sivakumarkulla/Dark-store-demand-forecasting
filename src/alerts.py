"""
Alert engine (replay mode).

Picks a moment in the test period, predicts the 90th-percentile demand for the
next 6 hours per category, compares it to current stock, and alerts if short.

Run from project root:
    python src/alerts.py                          # auto-picks a rainy evening
    python src/alerts.py --time "2026-07-15 18:00"

Optional Discord alerts: set the DISCORD_WEBHOOK_URL environment variable.
"""
import argparse
import json
import os
import pandas as pd
import requests
import xgboost as xgb
from features import build

STOCK = {"tea": 200, "cold_drinks": 300}   # DEFAULT stock (units); the dashboard can override it
STOCK_FILE = "data/stock.json"            # custom stock saved from the dashboard
SAFETY_PCT = 0.10                         # extra buffer on top of the p90 forecast
TEST_START = "2026-04-15"                 # same split date as train.py
LOG_PATH = "data/alerts_log.csv"


def get_stock():
    """Current stock per category: defaults, overridden by anything saved in data/stock.json."""
    stock = dict(STOCK)
    try:
        with open(STOCK_FILE) as f:
            saved = json.load(f)
        stock.update({k: int(v) for k, v in saved.items() if k in STOCK})
    except (FileNotFoundError, ValueError, TypeError):
        pass
    return stock


def set_stock(updates):
    """Save custom stock levels, e.g. set_stock({"tea": 150}). Only known categories are kept."""
    current = get_stock()
    current.update({k: int(v) for k, v in updates.items() if k in STOCK})
    os.makedirs(os.path.dirname(STOCK_FILE), exist_ok=True)
    tmp = STOCK_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(current, f)
    os.replace(tmp, STOCK_FILE)


def reset_stock():
    """Go back to the default stock levels."""
    if os.path.exists(STOCK_FILE):
        os.remove(STOCK_FILE)


def send_discord(text):
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return
    try:
        requests.post(url, json={"content": text}, timeout=10)
    except Exception as e:
        print(f"(Discord send failed: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", help='e.g. "2026-07-15 18:00" (must be in the test period)')
    args = ap.parse_args()

    df = pd.read_csv("data/hourly_sales.csv", parse_dates=["time"])
    built = {cat: build(d) for cat, d in df.groupby("category")}

    # choose "now"
    if args.time:
        now = pd.Timestamp(args.time)
    else:
        X, _, _ = built["tea"]
        cand = X[(X["time"] >= TEST_START) & (X["hour"] == 18)].dropna()
        now = cand.loc[cand["rain_next6"].idxmax(), "time"]   # rainiest 6pm in test period

    print(f"\n=== Replay at {now} (forecast window: next {6} hours) ===\n")
    stock_levels = get_stock()
    log_rows = []
    for cat, (X, y, _) in built.items():
        row = X[X["time"] == now]
        if row.empty or row.isna().any(axis=None):
            print(f"[{cat}] no complete features for {now}; pick another time.")
            continue
        feats = [c for c in X.columns if c != "time"]
        model = xgb.XGBRegressor()
        model.load_model(f"models/p90_{cat}.json")

        p90 = float(model.predict(row[feats])[0])
        needed = p90 * (1 + SAFETY_PCT)
        stock = stock_levels[cat]
        short = needed - stock
        actual = y[row.index[0]]
        rain = float(row["rain_next6"].iloc[0])
        match = int(row["match_next6"].iloc[0])

        if short > 0:
            msg = (f"ALERT  {cat}: forecast (p90) {p90:.0f} units in next 6h, stock {stock}. "
                   f"Short by {short:.0f}. Restock now.")
            send_discord(msg)
        else:
            msg = f"OK     {cat}: forecast (p90) {p90:.0f}, stock {stock}. Enough cover."
        print(msg)
        print(f"       context: rain next 6h = {rain:.1f} mm, match hours = {match} | "
              f"actual demand (replay only) = {actual:.0f}")

        log_rows.append({"time": now, "category": cat, "p90_forecast": round(p90, 1),
                         "stock": stock, "shortfall": round(max(short, 0), 1),
                         "alert": int(short > 0), "actual_demand": actual})

    if log_rows:
        pd.DataFrame(log_rows).to_csv(LOG_PATH, mode="a", index=False,
                                      header=not os.path.exists(LOG_PATH))
        print(f"\nLogged to {LOG_PATH}")


if __name__ == "__main__":
    main()