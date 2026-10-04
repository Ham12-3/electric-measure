# Project notes for Claude

GB national electricity demand (NESO "ND") forecasting at 1h / 24h / 168h horizons.
Compares Prophet, XGBoost and TFT against a "same hour last week" baseline using
rolling-origin backtests. Owner is new to time series - explain forecasting decisions
briefly, in plain English, in comments and the README.

## Working agreement
- Work in phases (see README "Roadmap"). Stop at the end of each phase, summarise,
  show key results, and wait for go-ahead.

## Hard rules
- **Never shuffle** time series data. Splits are always by time.
- **No leakage.** A feature for a prediction made at origin `t` may only use data known
  at `t`. Lagged demand must respect the horizon (a 168h forecast cannot use yesterday).
- Honest evaluation uses **forecast weather** (`weather_forecast.parquet`, by lead day),
  not actual weather. Actual weather is used for training and for the "cheating" comparison.
  Pick the lead day conservatively: a model run is published a few hours after its
  init time, so use `lead_day = ceil((h + publication_delay) / 24)`, capped at 7.
- Every model is compared to the **same hour last week** baseline.
- **Rolling-origin (walk-forward)** backtesting only, never a single split.
- Set seeds (`config.yaml: seed`) everywhere randomness exists.
- Keep modules separate: download / clean / features / models / evaluation / serving.

## Data facts (checked 2026-10-04)
- NESO CKAN: `https://api.neso.energy/api/3/action/package_show?id=historic-demand-data`.
  One CSV per year; current year is `demanddataupdate_YYYY.csv` (link changes - discover
  via API). Downloads 302-redirect to signed storage URLs. ~3 week publication lag.
- NESO date formats differ by year: `01-JAN-2021`, `01-Jan-23`, `2025-01-01`.
  Current-year file has `FORECAST_ACTUAL_INDICATOR`; keep `A` only.
- Settlement period 1 starts at 00:00 UK local time; 46 periods on spring clock change,
  50 in autumn. We store everything in **UTC**; calendar features should use UK local time.
- Hourly ND = mean of the two half-hours (MW is power, not energy).
- ND excludes embedded (rooftop) solar and wind, so sunny hours *reduce* ND.
- Open-Meteo: archive (reanalysis), previous_runs (`<var>_previous_dayN`, full coverage
  for `ecmwf_ifs025` from 2024-03-13), historical_forecast (stitched first hours of each
  run - only near-zero lead time, NOT honest for 24h/7d).
- All forecast sources use `ecmwf_ifs025`. Don't use `best_match`: for UK cities it picks
  UKMO 2 km, which has a ~+0.5 degC bias vs the ERA5 archive (train/test skew).
- Known forecast gap: 2026-04-17..22, humidity/apparent temp/cloud/precip missing at
  some lead days (temp/wind/radiation are complete). Fall back to the next-older lead day.
- ECMWF open data is 3/6-hourly beyond ~day 5, so day 6-7 hourly radiation is much noisier.
- Open-Meteo radiation/precipitation are "preceding hour" values; clean.py shifts them
  to start-of-hour labels to match demand.
- Free tier: 600 calls/min, 5k/hour, 10k/day; weight = max(1, vars/10) * max(1, days/14).

## Layout
- `configs/config.yaml` - all tunables (cities + weights, dates, horizons, seed).
- `src/demand_forecast/data/` - `download_demand.py`, `download_weather.py`, `clean.py`.
- `data/raw/` cached downloads, `data/processed/` parquet outputs (both gitignored).
- `notebooks/` exploration; built by `notebooks/build_exploration.py`.

## Commands
- `uv sync` - install (Python 3.12, pinned in `.python-version`).
- `uv run python -m demand_forecast.data.download_demand`
- `uv run python -m demand_forecast.data.download_weather` (~10 min, rate-limit paced)
- `uv run python -m demand_forecast.data.clean`
- `uv run pytest`
- Windows note: if `import demand_forecast` fails after adding files, run
  `uv sync --reinstall-package demand-forecast`.
