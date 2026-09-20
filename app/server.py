"""
Control-room dashboard server (FastAPI).

Run from the project root:
    python app/server.py
then open http://127.0.0.1:8000

Reads files created by the earlier steps:
    data/hourly_sales.csv, models/*.json, data/alerts_log.csv, data/backtest_results.csv
"""
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)                                   # the scripts use paths relative to the project root
sys.path.insert(0, str(ROOT / "src"))

from alerts import (LOG_PATH, SAFETY_PCT, STOCK, TEST_START,          # noqa: E402
                    get_stock, reset_stock, set_stock)
from features import H, build                                # noqa: E402
from weather import get_recent_and_forecast                  # noqa: E402

app = FastAPI(title="Dark store control room")
INDEX = Path(__file__).with_name("index.html")


def pretty(cat):
    return cat.replace("_", " ").capitalize()


def read_log():
    if not os.path.exists(LOG_PATH):
        return None
    log = pd.read_csv(LOG_PATH, parse_dates=["time"])
    return log if len(log) else None


@app.get("/")
def index():
    return FileResponse(INDEX)


@app.get("/api/status")
def status():
    """Latest logged check per category, compared against current stock."""
    log = read_log()
    if log is None:
        return {"ready": False,
                "message": "No checks logged yet. Run python src/live_alerts.py, then press Run check."}
    latest = log.sort_values("time").groupby("category").tail(1).sort_values("category")
    stock_levels = get_stock()
    items = []
    for _, r in latest.iterrows():
        cat = r["category"]
        stock = stock_levels.get(cat)
        if stock is None:
            continue
        p90 = float(r["p90_forecast"])
        need = p90 * (1 + SAFETY_PCT)
        alert = need > stock
        items.append({
            "category": cat, "label": pretty(cat),
            "p90": round(p90), "need": round(need), "stock": stock,
            "alert": bool(alert),
            "custom": stock != STOCK[cat],
            "short": max(1, round(need - stock)) if alert else 0,
            "spare": 0 if alert else round(stock - need),
        })
    t = latest["time"].max()
    return {"ready": True, "features_until": t.isoformat(),
            "forecast_from": (t + pd.Timedelta(hours=1)).isoformat(), "items": items}


class StockUpdate(BaseModel):
    category: str
    units: int


@app.post("/api/stock")
def update_stock(u: StockUpdate):
    """Save a custom stock level. The forecast does not change, so the dashboard updates instantly."""
    if u.category not in STOCK:
        raise HTTPException(404, "Unknown category")
    if not 0 <= u.units <= 100000:
        raise HTTPException(400, "Stock must be between 0 and 100,000 units")
    set_stock({u.category: u.units})
    return {"ok": True}


@app.post("/api/stock/reset")
def reset_all_stock():
    reset_stock()
    return {"ok": True}


@app.get("/api/alerts")
def alerts(limit: int = 8):
    """Recent checks, newest first. One entry per alerted category, or 'All clear'."""
    log = read_log()
    if log is None:
        return {"events": []}
    events = []
    for t, g in log.sort_values("time", ascending=False).groupby("time", sort=False):
        hit = g[g["alert"] == 1]
        if hit.empty:
            events.append({"time": t.isoformat(), "kind": "clear", "text": "All clear"})
        else:
            for _, r in hit.iterrows():
                events.append({"time": t.isoformat(), "kind": "alert",
                               "text": f"{pretty(r['category'])} short by {round(r['shortfall'])}"})
        if len(events) >= limit:
            break
    return {"events": events[:limit]}


@lru_cache(maxsize=8)
def forecast_frame(cat):
    """Actual vs forecast for every test-period hour (forecast is made at that hour)."""
    df = pd.read_csv("data/hourly_sales.csv", parse_dates=["time"])
    X, y, _ = build(df[df["category"] == cat])
    feats = [c for c in X.columns if c != "time"]
    ok = X.notna().all(axis=1) & y.notna() & (X["time"] >= TEST_START)
    point, p90 = xgb.XGBRegressor(), xgb.XGBRegressor()
    point.load_model(f"models/point_{cat}.json")
    p90.load_model(f"models/p90_{cat}.json")
    return pd.DataFrame({
        "time": X.loc[ok, "time"], "actual": y[ok],
        "forecast": point.predict(X.loc[ok, feats]), "p90": p90.predict(X.loc[ok, feats]),
    }).reset_index(drop=True)


@app.get("/api/forecast")
def forecast(category: str = "tea", hours: int = 24):
    if category not in STOCK:
        raise HTTPException(404, "Unknown category")
    try:
        fr = forecast_frame(category).tail(max(2, min(hours, 168)))
    except Exception as e:
        return JSONResponse({"error": f"Forecast unavailable: {e}"}, status_code=500)
    return {"category": category, "label": pretty(category),
            "times": [t.isoformat() for t in fr["time"]],
            "actual": fr["actual"].astype(float).round(1).tolist(),
            "forecast": fr["forecast"].astype(float).round(1).tolist(),
            "p90": fr["p90"].astype(float).round(1).tolist()}


_weather_cache = {"key": None, "ts": 0.0, "hours": []}


@app.get("/api/weather")
def weather():
    """Rain and temperature for the next 6 hours (live from Open-Meteo, cached 10 minutes)."""
    start = pd.Timestamp.now(tz="Asia/Kolkata").floor("h").tz_localize(None)
    if _weather_cache["key"] != start or time.time() - _weather_cache["ts"] > 600:
        try:
            wx = get_recent_and_forecast(past_days=1, forecast_days=2)
        except Exception as e:
            return {"hours": [], "error": str(e)[:120]}
        win = wx[(wx["time"] >= start) & (wx["time"] < start + pd.Timedelta(hours=H))]
        _weather_cache.update(key=start, ts=time.time(), hours=[
            {"label": f"{t.hour:02d}h", "rain": round(float(r), 1), "temp": round(float(c))}
            for t, r, c in zip(win["time"], win["rain_mm"], win["temp_c"])])
    return {"hours": _weather_cache["hours"]}


@app.get("/api/backtest")
def backtest():
    path = "data/backtest_results.csv"
    if not os.path.exists(path):
        return {"ready": False}
    red = pd.read_csv(path)["lost_units_reduction_%"]
    return {"ready": True, "median": round(float(red.median())),
            "wins": int((red > 0).sum()), "n": int(len(red))}


@app.post("/api/check")
def check():
    """Run the live check now (fetches weather, predicts, logs, sends Discord if configured)."""
    try:
        from live_alerts import run_once
        run_once()
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)[:200]}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    print("Open http://127.0.0.1:8000  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=8000)