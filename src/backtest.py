"""
Backtest: model-driven emergency replenishment vs a fixed reorder point (ROP).

We replay the TEST period hour by hour and simulate a dark store's stock.

Simulation rules (all ASSUMPTIONS - change them and see what happens):
  - Every day at 06:00 the store is refilled to CAPACITY (scheduled delivery).
  - Demand in each hour = the (simulated) units actually demanded. If stock runs
    out, the missing units are LOST SALES and that hour counts as a stockout hour.
  - An emergency order can be placed at the end of any hour (one at a time).
    It refills the store to capacity and arrives LEAD hours later.
  - Policy A (baseline): order when stock <= a fixed reorder point.
  - Policy B (model):    order when p90 forecast * (1 + safety) > current stock.

FAIR COMPARISON: a high enough fixed ROP can always avoid stockouts by ordering
constantly. So for each scenario we compare the model against the fixed-ROP
policy that places about the SAME NUMBER of emergency orders.

Run from project root:  python src/backtest.py   (needs models/ from train.py)
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from features import build

TEST_START = "2026-04-15"          # same split as train.py
SAFETY_PCT = 0.10                  # same buffer as alerts.py
REFILL_HOUR = 6
LEADS = [2, 3, 4]                  # hours from order to arrival
CAP_DAYS = [0.6, 0.75, 0.9]        # store capacity as a fraction of average daily demand
ROP_FRACS = np.arange(0.02, 0.99, 0.01)   # fixed ROP grid (fraction of capacity); fine steps for close order matching

df = pd.read_csv("data/hourly_sales.csv", parse_dates=["time"])


def load(cat, d):
    """Return test-period hourly demand, hours, p90 forecasts, and avg daily demand (from train)."""
    d = d.sort_values("time").reset_index(drop=True)
    X, y, _ = build(d)
    feats = [c for c in X.columns if c != "time"]
    ok = X.notna().all(axis=1) & y.notna() & (X["time"] >= TEST_START)
    idx = np.where(ok)[0]                      # contiguous block of test hours
    model = xgb.XGBRegressor()
    model.load_model(f"models/p90_{cat}.json")
    p90 = model.predict(X.loc[idx, feats])
    daily = d.loc[d["time"] < TEST_START, "units"].mean() * 24
    return {"demand": d.loc[idx, "units"].to_numpy(float),
            "hour": d.loc[idx, "hour"].to_numpy(),
            "p90": p90, "daily": daily}


def simulate(demand, hour, capacity, lead, trigger):
    stock, arrive_at, qty = capacity, None, 0.0
    lost, stockout_hours, orders = 0.0, 0, 0
    for i in range(len(demand)):
        if hour[i] == REFILL_HOUR:
            stock = capacity                               # scheduled refill
        if arrive_at == i:
            stock = min(capacity, stock + qty)             # emergency order arrives
            arrive_at = None
        sold = min(stock, demand[i])
        if demand[i] > sold:
            lost += demand[i] - sold
            stockout_hours += 1
        stock -= sold
        # decision at the end of hour i
        if arrive_at is None and stock < capacity and trigger(i, stock):
            qty = capacity - stock
            arrive_at = i + 1 + lead
            orders += 1
    return lost, stockout_hours, orders


data = {cat: load(cat, d) for cat, d in df.groupby("category")}
total_demand = sum(v["demand"].sum() for v in data.values())
hours_total = sum(len(v["demand"]) for v in data.values())


def run_policy(cap_days, lead, make_trigger):
    lost = so = orders = 0
    for cat, v in data.items():
        cap = cap_days * v["daily"]
        l, s, o = simulate(v["demand"], v["hour"], cap, lead, make_trigger(v, cap))
        lost, so, orders = lost + l, so + s, orders + o
    return lost, so, orders


rows = []
for cap_days in CAP_DAYS:
    for lead in LEADS:
        m_lost, m_so, m_orders = run_policy(
            cap_days, lead,
            lambda v, cap: (lambda i, s: v["p90"][i] * (1 + SAFETY_PCT) > s))

        curve = []
        for f in ROP_FRACS:
            l, s, o = run_policy(cap_days, lead, lambda v, cap, f=f: (lambda i, st: st <= f * cap))
            curve.append((f, l, s, o))
        f, b_lost, b_so, b_orders = min(curve, key=lambda c: abs(c[3] - m_orders))   # matched orders

        rows.append({
            "capacity_days": cap_days, "lead_h": lead,
            "model_orders": m_orders, "base_orders": b_orders, "base_ROP_%cap": round(f * 100),
            "model_lost_%": 100 * m_lost / total_demand, "base_lost_%": 100 * b_lost / total_demand,
            "model_stockout_h": m_so, "base_stockout_h": b_so,
            "lost_units_reduction_%": 100 * (1 - m_lost / b_lost) if b_lost else np.nan,
        })

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(f"\nTest period: {hours_total // len(data):,} hours per category, {len(data)} categories\n")
print(res.round(2).to_string(index=False))
res.to_csv("data/backtest_results.csv", index=False)

red = res["lost_units_reduction_%"]
print(f"\nLost-sales reduction vs fixed ROP with matched emergency orders: "
      f"median {red.median():.0f}%, range {red.min():.0f}% to {red.max():.0f}% "
      f"across {len(res)} scenarios")
print("Saved data/backtest_results.csv")
