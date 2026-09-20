"""Shared feature building, used by train.py and alerts.py."""
import pandas as pd

H = 6  # forecast horizon in hours


def next_h_sum(x):
    """Sum of x over t+1 .. t+H, aligned to time t."""
    s = pd.Series(x)
    return s[::-1].rolling(H).sum()[::-1].shift(-1).to_numpy()


def build(d):
    """Build features X, target y (next-6h units) and baseline for ONE category."""
    d = d.sort_values("time").reset_index(drop=True)
    u = d["units"]
    X = pd.DataFrame({"time": d["time"]})
    X["hour"], X["dow"] = d["hour"], d["dow"]
    # past demand (known at time t)
    X["lag_0"], X["lag_1"] = u, u.shift(1)
    X["lag_24"], X["lag_168"] = u.shift(24), u.shift(168)
    X["roll_24"], X["roll_168"] = u.rolling(24).mean(), u.rolling(168).mean()
    # next-6h context knowable in advance (weather forecast, schedule)
    X["rain_next6"] = next_h_sum(d["rain_mm"])
    X["temp_next6"] = next_h_sum(d["temp_c"]) / H
    X["match_next6"] = next_h_sum(d["is_match"])
    X["fri_night_next6"] = next_h_sum(d["is_friday_night"])
    X["weekend_next6"] = next_h_sum(d["is_weekend"])
    y = pd.Series(next_h_sum(u))
    baseline = y.shift(168)  # same window last week
    return X, y, baseline
