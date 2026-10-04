# GB Electricity Demand Forecasting

Forecasts Great Britain's national electricity demand **1 hour, 24 hours and 7 days ahead**,
and compares Prophet, XGBoost and a Temporal Fusion Transformer against a simple
"same hour last week" baseline, using only information that would have been available
at forecast time.

> Status: **Phase 2 (features + baseline) complete.** Sections marked _TBD_ are filled in later phases.

## 1. Problem

The grid operator has to balance supply and demand every second. Generators are
scheduled hours to days ahead, so demand forecasts at several horizons drive cost and
reliability: an hour ahead for real-time balancing, a day ahead for market scheduling,
a week ahead for planning plant availability.

Success means beating the naive baseline at every horizon **in an honest backtest**:
walk-forward evaluation, no shuffling, and weather *forecasts* (not observed weather)
as inputs.

## 2. Data

| Source | What | Resolution | Range used |
|---|---|---|---|
| [NESO Historic Demand Data](https://www.neso.energy/data-portal/historic-demand-data) | `ND` (National Demand, MW) | 30 min | 2021-01-01 → ~3 weeks ago |
| [Open-Meteo Historical Weather](https://open-meteo.com/en/docs/historical-weather-api) | Reanalysis ("actual") weather | hourly | 2021-01-01 → ~1 week ago |
| [Open-Meteo Previous Runs](https://open-meteo.com/en/docs/previous-runs-api) | Real forecasts at 1–7 day lead times (ECMWF IFS 0.25°) | hourly | 2024-03-13 → yesterday |
| [Open-Meteo Historical Forecast](https://open-meteo.com/en/docs/historical-forecast-api) | First hours of each model run, stitched together (ECMWF IFS 0.25°) | hourly | 2024-03-13 → yesterday |

**What ND is.** National Demand is the demand met by transmission-connected generation.
It *excludes* rooftop and small-scale ("embedded") solar and wind. So on a sunny day,
ND falls even though people use just as much electricity. That makes solar radiation a
genuine demand driver, alongside temperature.

**Weather.** Five cities stand in for their regions and are combined using population
weights: London 0.52, Birmingham 0.16, Manchester 0.16, Glasgow 0.10, Cardiff 0.06.
Variables: temperature, apparent temperature, humidity, wind speed, cloud cover,
precipitation, solar (shortwave) radiation.

**Why three weather sources?** Training on observed weather is fine. But *testing* a
7-day-ahead forecast with the weather that actually happened is cheating, because
nobody knows next week's weather exactly. The Previous Runs API gives what the forecast
said 1, 2, … 7 days before each hour, so the backtest sees realistic weather errors.
The Historical Forecast API sounds like the right tool but isn't: it stitches together
the first few hours of every model run, so it's effectively a 0–6 hour forecast.

| National temperature error vs reanalysis | MAE |
|---|---|
| Historical Forecast API (stitched) | 0.23 °C |
| Forecast made 1 day ahead | 0.39 °C |
| Forecast made 7 days ahead | 1.79 °C |

All forecast sources use **one model (ECMWF IFS 0.25°)**. Open-Meteo's default
"best match" uses the Met Office 2 km model for UK cities. Over London that runs ~0.5 °C
warmer than the reanalysis we train on, an offset that would look like forecast error
but isn't.

### Cleaning decisions

| Issue | What we do | Why |
|---|---|---|
| Settlement periods, not timestamps | Period 1 starts at local midnight; convert local midnight → UTC and add (period − 1) × 30 min | One rule that works for normal days and clock-change days |
| Clock changes (46 / 50 periods) | Store everything in UTC | UTC has no missing or repeated hours; local time is used later only for calendar features |
| Changing date formats across years | Try each known format per file | NESO switched from `01-JAN-2021` to `01-Jan-23` to `2025-01-01` |
| Half-hourly → hourly | Mean of the two half-hours | ND is power (MW), so average it rather than add it up |
| Missing values | Interpolate gaps ≤ 2 h, leave longer gaps missing | Don't invent data the model is then scored on |
| Outliers | Remove only physically impossible values or isolated spikes (>10× the typical half-hour change away from neighbours) | Real extremes (cold snaps, Christmas) are what we need to forecast |
| Radiation/precipitation timestamps | Shift back 1 h | Open-Meteo labels them at the end of the hour; demand hours are labelled at the start |

What the cleaning actually found is in `data/processed/cleaning_report.json` and in the
[exploration notebook](notebooks/01_exploration.ipynb).

## 3. Approach

### What "1h / 24h / 7d ahead" means exactly

A forecast is made at an **origin** time *t*. Horizon *h* is the hour that **ends**
*h* hours later. At 09:00 the newest demand we know is the 08:00–09:00 hour, so:

| Horizon | Forecasting (from 09:00 Mon) | Newest demand it may use | Weather forecast it may use |
|---|---|---|---|
| 1h | 09:00–10:00 Mon | 08:00–09:00 Mon | issued ≥ 1 day before (lead day 1) |
| 24h | 08:00–09:00 Tue | 08:00–09:00 Mon | issued ≥ 2 days before (lead day 2) |
| 168h | 08:00–09:00 next Mon | 08:00–09:00 Mon, i.e. a week before the target | issued 7 days before (lead day 7) |

Every hour in the test period is forecast once **per horizon**, so each horizon is
scored on ~21,000 target hours covering every hour of the day.

### Features

| Group | Features | Leakage rule |
|---|---|---|
| Calendar (UK local time) | hour, weekday, month, weekend, England & Wales bank holiday, Scottish bank holiday, Christmas period, smooth day-of-year | Always known in advance |
| Demand lags | `lag_k` for k ∈ {1, 2, 3, 24, 48, 168, 336} h; 24h and 168h rolling means | Only lags **k ≥ h**; rolling windows end at the newest known hour. The 168h model gets just `lag_168`, `lag_336` and rolling means ending a week back |
| Weather | temperature, wind speed, solar radiation; 24h mean temperature; heating degrees max(0, 15.5 − T); cooling degrees max(0, T − 20) | **Training:** actual weather. **Testing:** the weather forecast that existed at the origin |

**Which weather forecast is legal?** A weather model run becomes available about 8 hours
after it starts. The lead day for horizon *h* is ⌈(h + 8) / 24⌉: day 1 for 1h, day 2 for
24h, and day 8 for 168h. Day 8 doesn't exist, so we cap at day 7. That last case uses a
forecast published up to ~8h after the origin, which is a small optimism we accept and
document. Where a forecast value is missing we fall back to *older* forecasts only.

Wind appears twice in its effect on National Demand. It makes buildings lose heat
faster, and it powers small wind farms that reduce the demand the transmission network
sees.

### Rolling-origin backtest

```
train ─────────────────────────────┤ test Q2-2024
train ──────────────────────────────────┤ test Q3-2024
train ───────────────────────────────────────┤ test Q4-2024     … 10 folds to Sep 2026
```

- Folds are calendar quarters from **April 2024** (when forecast weather begins) to the
  latest data. Each fold retrains on **all data before it** (expanding window, from
  January 2021).
- Folds are split by **forecast origin**. A 7-day forecast made on 30 June belongs to the
  fold that starts in April, even though its target is in July.
- Data is never shuffled. Tests check that changing "future" demand or observed weather
  never changes a feature.
- Metrics per horizon: MAE (MW), RMSE (MW), MAPE (%); plus pinball loss for quantiles.

### Baseline: same hour last week

ŷ(Tue 18:00) = demand at the previous Tue 18:00. It gets the daily and weekly shape right
for free and is legal at every horizon up to 7 days. Its accuracy is the same at every
horizon, because it ignores everything newer than a week. So any gain at 1h and 24h has
to come from the models using fresher information.

## 4. Architecture

_TBD (Mermaid diagram)._

## 5. Results

Backtest: April 2024 – September 2026, 10 quarterly folds, forecast weather.
Models are added in Phase 3.

| Model | Horizon | MAE (MW) | RMSE (MW) | MAPE |
|---|---|---|---|---|
| Same hour last week | 1h | 2,057 | 2,781 | 8.17% |
| Same hour last week | 24h | 2,053 | 2,775 | 8.15% |
| Same hour last week | 168h | 2,053 | 2,777 | 8.15% |

The baseline's worst days are the cases models need to handle: cold snaps
(7–8 Jan 2025), bank holidays, Easter, clock-change weekends, and the end of the
June 2026 heatwave, when it copied a 31 °C week into a 19 °C one.

## 6. Key findings

_TBD._

## 7. Limitations

- Honest (forecast-weather) evaluation can only start in March 2024, when Open-Meteo's
  archive of 1–7 day forecasts begins.
- Five cities are a coarse proxy for national weather.
- NESO publishes outturn about 3 weeks late, so the "latest actuals" are never today's.
- Weather-model runs become available a few hours after their start time. For the very
  end of the 7-day window no older forecast exists, so those hours are slightly optimistic.

## 8. How to run

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 automatically.

```bash
uv sync
uv run python -m demand_forecast.data.download_demand   # ~30 s
uv run python -m demand_forecast.data.download_weather  # ~10 min (free-tier rate limits)
uv run python -m demand_forecast.data.clean
uv run python notebooks/build_exploration.py            # builds + runs the notebook
uv run python -m demand_forecast.evaluation.backtest --model baseline
uv run pytest
```

Downloads are cached in `data/raw/`. Re-running only refreshes the current year.

## Roadmap

1. ✅ Setup and data
2. ✅ Features and baseline
3. Models (Prophet, XGBoost, TFT)
4. Evaluation and write-up
5. Serving (FastAPI, Streamlit, Docker) and deployment

## Data licences

NESO data is used under the NESO Open Data Licence. Weather data is from
[Open-Meteo](https://open-meteo.com/) (CC BY 4.0), used under its free non-commercial terms.
