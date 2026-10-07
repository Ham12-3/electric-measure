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

## Forecasting conventions (Phase 2)
- Hours are labelled by their START (UTC). Origin `t`, horizon `h` -> target `T = t + h - 1`;
  newest known demand is hour `T - h`. So demand lag `k` is legal iff `k >= h`.
- Weather lead day `N(h) = ceil((h + 8) / 24)`, capped at 7 (h=1->1, 24->2, 168->7).
  Fallback for missing values: older leads only. 168h is mildly optimistic (documented).
- Weather features in test mode are built ONLY from the horizon's forecast series
  (including the 24h rolling temp). Training uses actual weather.
- Folds: quarterly, split by ORIGIN, expanding window from 2021-01-01, first origin
  2024-04-01. One fresh model per (fold, horizon); seeds reset per fit.
- Model interface: `fit(train_df, h)` / `predict(test_df, h) -> DataFrame[y_pred, (q10, q90)]`
  (see `models/base.py`). Register new models in `evaluation/backtest.py:model_registry`.
- Baseline uses `lag_168` (UTC), so it's misaligned by 1h in the week after a clock change.

## Models (Phase 3)
- `xgboost` (direct, one model per horizon), `prophet` (horizon_independent; 4 conditional
  daily seasonalities {GMT,BST}x{workday,day off}), `tft` (Darts TFTModel, quantiles
  0.1/0.5/0.9, horizon_independent, refit every 2 folds). Hyperparameters are fixed in
  config.yaml - do NOT tune on backtest folds.
- TFT honest weather relies on a causal decoder (`full_attention=False`); `tests/test_tft.py`
  guards it. Covariates past the target are ffill-padded (harmless because causal).

## Performance notes (4-core i5 laptop, no GPU)
- NEVER run two CPU-heavy jobs at once: XGBoost/torch OpenMP threads thrash when
  oversubscribed (an XGB fit went from ~20s to 1,277s next to Prophet).
- Backtest times: baseline <1 min, XGBoost ~7 min, Prophet ~15 min, TFT ~80 min.
- Tool background jobs are killed at ~30 min; launch long runs detached
  (PowerShell `Start-Process ... -RedirectStandardOutput artifacts/logs/<x>.log`).
- TFT speed is dominated by batch overhead: batch 256 is ~2x faster per sample than 128;
  hidden 16 vs 32 makes little difference.

## Serving (Phase 5)
- Production = XGBoost at horizons [1,6,24,72,168]; hour h uses smallest H >= h (legal).
  Train: `python -m demand_forecast.serving.production`; daily job: `serving.pipeline`.
- Live demand = NESO processed + Elexon INDO (`data/download_recent.py`, matches ND to
  0.10%; max 28 days per request). Live weather = Open-Meteo forecast API, ecmwf_ifs025.
- Deps: core + extras `research` (prophet, darts[torch]) and `serve` (fastapi, uvicorn,
  streamlit). Use `uv sync --all-extras` locally; Docker installs `--extra serve` only.
  Don't run `uv sync` while a torch job is running.
- Final results: reports/results.md (XGBoost best everywhere; TFT intervals under-cover).

## Layout
- `configs/config.yaml` - all tunables (cities + weights, dates, horizons, seed).
- `src/demand_forecast/data/` - `download_demand.py`, `download_weather.py`, `clean.py`,
  `dataset.py` (loads processed files).
- `src/demand_forecast/features/` - `calendar.py`, `lags.py`, `weather.py`, `build.py`.
- `src/demand_forecast/models/` - `base.py` (interface), `baseline.py`.
- `src/demand_forecast/evaluation/` - `metrics.py`, `backtest.py` (CLI).
- `data/raw/` cached downloads, `data/processed/` parquet outputs, `artifacts/` backtest
  predictions (all gitignored). `reports/metrics/*.csv` and `reports/figures/` are committed.
- `notebooks/` exploration; built by `notebooks/build_exploration.py`.

## Commands
- `uv sync` - install (Python 3.12, pinned in `.python-version`).
- `uv run python -m demand_forecast.data.download_demand`
- `uv run python -m demand_forecast.data.download_weather` (~10 min, rate-limit paced)
- `uv run python -m demand_forecast.data.clean`
- `uv run python -m demand_forecast.evaluation.backtest --model baseline [--weather actual]`
- `uv run pytest`
- Windows: set `PYTHONIOENCODING=utf-8` when printing tables with non-ASCII characters.
- Windows note: if `import demand_forecast` fails after adding files, run
  `uv sync --reinstall-package demand-forecast`.
