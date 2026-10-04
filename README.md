# GB Electricity Demand Forecasting

Forecasts Great Britain's national electricity demand **1 hour, 24 hours and 7 days ahead**,
and compares Prophet, XGBoost and a Temporal Fusion Transformer against a simple
"same hour last week" baseline, using only information that would have been available
at forecast time.

> Status: **Phase 1 (data) complete.** Sections marked _TBD_ are filled in later phases.

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

_TBD (Phases 2–3)._ Planned: calendar + UK bank holiday + horizon-aware lag features,
a "same hour last week" baseline, rolling-origin backtests from April 2024 onward, and
MAE / RMSE / MAPE per horizon (plus pinball loss for TFT's prediction intervals).

## 4. Architecture

_TBD (Mermaid diagram)._

## 5. Results

_TBD (Phase 4)._

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
uv run pytest
```

Downloads are cached in `data/raw/`. Re-running only refreshes the current year.

## Roadmap

1. ✅ Setup and data
2. Features and baseline
3. Models (Prophet, XGBoost, TFT)
4. Evaluation and write-up
5. Serving (FastAPI, Streamlit, Docker) and deployment

## Data licences

NESO data is used under the NESO Open Data Licence. Weather data is from
[Open-Meteo](https://open-meteo.com/) (CC BY 4.0), used under its free non-commercial terms.
