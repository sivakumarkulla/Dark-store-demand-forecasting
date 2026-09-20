"""Shared simulation code: calendar flags and synthetic sales. Used by generate_data.py and live_alerts.py."""
import numpy as np
import pandas as pd

LAT, LON = 19.07, 72.88                        # Mumbai
CATEGORIES = {"tea": 18, "cold_drinks": 25}    # average units per hour (base level)

# Daily shape: small morning bump, big 6-11 PM peak
HOUR_SHAPE = np.array([0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.7, 0.8, 0.6, 0.6, 0.7,
                       0.8, 0.8, 0.7, 0.8, 0.9, 1.1, 1.6, 1.9, 2.0, 1.9, 1.4, 0.7])


def add_flags(df, match_days=()):
    """df needs columns: time, rain_mm, temp_c. Adds calendar and event flags."""
    df = df.copy()
    df["hour"] = df["time"].dt.hour
    df["dow"] = df["time"].dt.dayofweek            # Mon=0 ... Sun=6
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_friday_night"] = ((df["dow"] == 4) & (df["hour"] >= 20)).astype(int)
    df["is_rain"] = (df["rain_mm"] > 0.5).astype(int)
    df["is_match"] = (
        df["time"].dt.normalize().isin(list(match_days)) & df["hour"].between(19, 23)
    ).astype(int)
    return df


def simulate_units(df, rng):
    """Add simulated hourly sales for every category. Returns a long-format DataFrame."""
    rows = []
    for cat, base in CATEGORIES.items():
        m = base * HOUR_SHAPE[df["hour"]]
        m = m * (1 + 0.15 * df["is_weekend"]) * (1 + 0.25 * df["is_friday_night"])
        if cat == "tea":
            m = m * (1 + 0.5 * df["is_rain"]) * (1 + 0.10 * df["is_match"])
        else:  # cold_drinks: rises with heat and matches; dips slightly in rain
            m = m * (1 + 0.04 * (df["temp_c"] - 28)) * (1 + 0.6 * df["is_match"]) * (1 - 0.15 * df["is_rain"])
        out = df.copy()
        out["category"] = cat
        out["units"] = rng.poisson(np.clip(m, 0.1, None))
        rows.append(out)
    return pd.concat(rows).sort_values(["time", "category"]).reset_index(drop=True)
