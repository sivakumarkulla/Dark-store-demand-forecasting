# Dark store demand forecasting and restock alerts

A forecasting tool for quick-commerce dark stores. It predicts demand for the next 6 hours per product category, using sales history and the live weather forecast, and raises an alert when predicted demand is higher than the stock on the shelf, early enough to order an emergency restock.

## The problem

Demand at a dark store is spiky: rain lifts tea and snacks, cricket nights lift cold drinks, and Friday evenings run hot. Restocking from the main warehouse takes hours, so it has to be triggered before the spike, not when the shelf is already empty. A fixed reorder point ("restock below 50 units") cannot see rain coming.

## How it works

```
Sales history + weather forecast + calendar/match flags
        |
Features: lags (1h, 24h, 168h), rolling means, rain / match / weekend in the next 6h
        |
XGBoost: point forecast + 90th-percentile forecast (next 6 hours of demand)
        |
Alert rule: p90 forecast x 1.10 safety buffer > current stock  ->  alert (dashboard + Discord)
        |
Backtest: replay the test period and compare with a fixed reorder point
```

The 90th-percentile forecast is used for alerts because running out costs more than overstocking.

## What is real and what is simulated

| Part | Status |
|---|---|
| Historical weather (rain, temperature) | Real, from Open-Meteo (Mumbai) |
| Live weather forecast | Real, from Open-Meteo |
| Sales data | **Simulated.** No public dark-store dataset exists. Effects (rain, match nights, Friday evenings, heat) are injected on purpose so the model can be checked against known truth |
| IPL match schedule | Simulated |
| Recent sales in live mode and stock levels | Simulated. In a real deployment these would be database queries |

All results below come from simulated sales data and should be read as a demonstration that the pipeline works, not as real-world performance.

## Results

Time-based split: train on data before 2026-04-15, test after. Baseline: demand in the same 6-hour window one week earlier.

**Forecast error (mean absolute error, units per 6-hour window)**

| | Cold drinks | Tea |
|---|---|---|
| All hours, model vs baseline | 10.2 vs 22.8 | 8.8 vs 17.0 |
| Top-10% surge windows, model vs baseline | 16.7 vs 59.2 | 14.6 vs 35.9 |
| p90 forecast coverage (target 90%) | 91% | 90% |
| p90 coverage on surge windows | 70% | 73% |

**Backtest.** The test period is replayed hour by hour with simulated stock. The store is refilled to capacity at 6 AM, and an emergency order arrives 2-4 hours after it is placed. The model's alerts are compared with the fixed reorder point that places about the same number of emergency orders, so the comparison is not won simply by ordering more.

| Capacity (days of demand) | Lead time | Lost sales, model | Lost sales, fixed reorder point | Reduction |
|---|---|---|---|---|
| 0.60 | 2h | 1.29% | 1.09% | -18% |
| 0.60 | 3h | 4.29% | 4.08% | -5% |
| 0.60 | 4h | 7.63% | 9.88% | 23% |
| 0.75 | 2h | 0.06% | 0.23% | 73% |
| 0.75 | 3h | 0.96% | 1.67% | 43% |
| 0.75 | 4h | 2.93% | 3.50% | 16% |
| 0.90 | 2h | 0.00% | 0.07% | 100% |
| 0.90 | 3h | 0.03% | 0.45% | 94% |
| 0.90 | 4h | 0.83% | 3.16% | 74% |

Median reduction in lost sales: **43%**. The model was better in 7 of 9 scenarios and worse in the two with the smallest store capacity.

## Limitations

- **Sales are simulated**, so the model is learning effects that were planted by me. The point is to show the full pipeline and to verify the model recovers known effects, not to claim real-world accuracy.
- **The forecast is treated as perfect in training and backtesting.** The model is given the actual rain for the next 6 hours. A real weather forecast is noisy, so real gains would be smaller.
- **Surge coverage is 70-73%, below 90%.** The p90 forecast misses about a quarter to a third of the biggest spikes. A higher quantile or a larger safety buffer would trade fewer misses for more overstock.
- **The model loses in small-store scenarios.** When capacity is only 0.6 days of demand, the alert fires so often it is no more selective than a fixed reorder point. I have not tested why.
- **Backtest assumptions** (capacity, lead time, 6 AM refill, one order in flight) are mine, not from a real operation.
- One store and two categories only.

## Quickstart

Requires Python 3.10+.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python src/generate_data.py       # downloads real weather, simulates sales -> data/hourly_sales.csv
python src/train.py               # trains and saves models, prints the results table
python src/backtest.py            # model alerts vs fixed reorder point -> data/backtest_results.csv
python src/alerts.py              # replay demo: one alert check on a rainy evening
python src/live_alerts.py         # live check with the current weather forecast
python app/server.py              # dashboard at http://127.0.0.1:8000
```

Optional Discord alerts: set the `DISCORD_WEBHOOK_URL` environment variable to a Discord webhook URL. Never commit the URL to the repository.

## Dashboard

A single-page control room served by FastAPI: alert banner, stock-versus-forecast gauges, a 6-hour weather strip, forecast-versus-actual chart, alert history, and backtest summary. Stock levels can be edited by clicking the stock number on a tile (saved to `data/stock.json`). The "Run check" button fetches the live forecast, predicts, logs, and sends Discord alerts.

## Project structure

```
app/
  server.py          FastAPI backend and JSON endpoints
  index.html         Dashboard front end
src/
  simdata.py         Calendar flags and simulated sales
  generate_data.py   Real weather + simulated sales -> CSV
  features.py        Lag, rolling and next-6-hours features
  train.py           XGBoost point and p90 models, evaluation
  alerts.py          Alert rule, stock handling, Discord, replay demo
  weather.py         Live weather from Open-Meteo
  live_alerts.py     Live alert check (run once or hourly)
  backtest.py        Simulation vs fixed reorder point
```

## Tech stack

Python, pandas, XGBoost, scikit-learn, FastAPI, Open-Meteo API.

## Possible next steps

- Real sales and stock feed from a database (PostgreSQL)
- Train and backtest with noisy weather forecasts instead of actual weather
- More stores and categories, with a cross-store risk view
- Tune the alert quantile and safety buffer against a cost of lost sales vs overstock
