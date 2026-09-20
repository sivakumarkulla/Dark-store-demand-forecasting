"""
Synthetic dark-store sales generator (1 store, 2 categories).

- Weather: REAL hourly rain + temperature from Open-Meteo (free, no API key).
  Falls back to simulated weather if the API is unreachable.
- Sales: SIMULATED, with known effects injected so you can check the model learns them.
- IPL match nights: SIMULATED schedule (replace with a real CSV later if you want).

Run from project root:  python src/generate_data.py
Output: data/hourly_sales.csv
"""
import os
import numpy as np
import pandas as pd
import requests
from simdata import LAT, LON, add_flags, simulate_units

rng = np.random.default_rng(42)

START, END = "2024-01-01", "2026-08-31"


def get_weather():
    try:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": LAT, "longitude": LON,
                "start_date": START, "end_date": END,
                "hourly": "precipitation,temperature_2m",
                "timezone": "Asia/Kolkata",
            },
            timeout=60,
        )
        r.raise_for_status()
        h = r.json()["hourly"]
        print("Using REAL Open-Meteo weather")
        return pd.DataFrame({
            "time": pd.to_datetime(h["time"]),
            "rain_mm": h["precipitation"],
            "temp_c": h["temperature_2m"],
        })
    except Exception as e:
        print(f"Open-Meteo failed ({e}). Using SIMULATED weather instead.")
        t = pd.date_range(START, END + " 23:00", freq="h")
        rainy = rng.random(len(t)) < 0.08
        return pd.DataFrame({
            "time": t,
            "rain_mm": np.where(rainy, rng.exponential(3, len(t)), 0.0),
            "temp_c": 28 + 4 * np.sin(2 * np.pi * (t.hour - 9) / 24) + rng.normal(0, 1, len(t)),
        })


df = get_weather().dropna().reset_index(drop=True)

# Simulated IPL match nights (Mar 25 - May 31 each year, ~35 days per season)
match_days = set()
for yr in (2024, 2025, 2026):
    days = pd.date_range(f"{yr}-03-25", f"{yr}-05-31").normalize()
    match_days.update(rng.choice(days, size=35, replace=False))

df = add_flags(df, match_days)
data = simulate_units(df, rng)

os.makedirs("data", exist_ok=True)
data.to_csv("data/hourly_sales.csv", index=False)

print(f"Saved {len(data):,} rows to data/hourly_sales.csv")
print(data.groupby(["category", "is_rain"])["units"].mean().round(1).unstack())
