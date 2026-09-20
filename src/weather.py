"""Live weather from Open-Meteo (free, no API key)."""
import pandas as pd
import requests
from simdata import LAT, LON


def get_recent_and_forecast(past_days=8, forecast_days=2):
    """Hourly rain + temperature: the last `past_days` days plus the next `forecast_days` days.

    Uses the same source (Open-Meteo) and units as the training data, so the model
    sees the same kind of numbers at prediction time as it did in training.
    Times are Asia/Kolkata local time, timezone-naive.
    """
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": LAT, "longitude": LON,
            "hourly": "precipitation,temperature_2m",
            "past_days": past_days, "forecast_days": forecast_days,
            "timezone": "Asia/Kolkata",
        },
        timeout=30,
    )
    r.raise_for_status()
    h = r.json()["hourly"]
    return pd.DataFrame({
        "time": pd.to_datetime(h["time"]),
        "rain_mm": h["precipitation"],
        "temp_c": h["temperature_2m"],
    })
