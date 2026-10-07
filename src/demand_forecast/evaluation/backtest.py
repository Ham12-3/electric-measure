"""Rolling-origin (walk-forward) backtesting.

How it works, in plain English:
1. Split the test period into consecutive quarters ("folds").
2. For each fold, train a fresh model on everything BEFORE the fold starts.
3. Use it to forecast every hour in the fold, at each horizon, as if we were living
   through that quarter: each forecast only sees demand and weather forecasts that
   existed at its origin.
4. Move forward a quarter and repeat. Scores are pooled over all folds.

This is like re-running the model in production every quarter. A single train/test split
would rest on one arbitrary date, and wouldn't show whether the model holds up as the
system changes.

Folds are defined by forecast ORIGIN, not target. A 7-day-ahead forecast made on the
last day of a fold targets the next fold, and it must come from the model trained for
the fold its origin is in. (Otherwise we'd use a model trained on data after the origin.)

Usage:
    uv run python -m demand_forecast.evaluation.backtest --model baseline [--weather actual]
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from demand_forecast.config import PROJECT_ROOT, load_config
from demand_forecast.data.dataset import Dataset, load_dataset
from demand_forecast.evaluation.metrics import score_table
from demand_forecast.features.build import build_features
from demand_forecast.features.weather import forecast_weather_for_horizon
from demand_forecast.models.base import Forecaster

log = logging.getLogger(__name__)

WEATHER_MODES = ("forecast", "actual")  # "actual" = cheating, for comparison only


@dataclass(frozen=True)
class Fold:
    number: int
    cutoff: pd.Timestamp  # training targets < cutoff <= test origins
    end: pd.Timestamp     # test origins < end


def make_folds(test_start: str | pd.Timestamp, data_end: pd.Timestamp, months: int) -> list[Fold]:
    start = pd.Timestamp(test_start, tz="UTC") if not isinstance(test_start, pd.Timestamp) else test_start
    # Month-start ranges begin at the NEXT month start, so add `start` explicitly.
    edges = [start] + [d for d in pd.date_range(start, data_end, freq=f"{months}MS", tz="UTC") if d > start]
    edges.append(data_end + pd.Timedelta(hours=1))
    return [Fold(i, a, b) for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]))]


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


class FeatureCache:
    """Builds each (horizon, weather source) feature table once and reuses it across folds."""

    def __init__(self, ds: Dataset, cfg: dict):
        self.ds, self.cfg, self._cache = ds, cfg, {}

    def get(self, h: int, weather: str) -> pd.DataFrame:
        key = (h, weather)
        if key not in self._cache:
            if weather == "forecast":
                f = self.cfg["features"]
                w = forecast_weather_for_horizon(self.ds.weather_forecast, h,
                                                 f["publication_delay_hours"], f["weather_vars"])
            else:
                w = self.ds.weather_actual
            self._cache[key] = build_features(self.ds.demand, w, h, self.cfg)
        return self._cache[key]


def split_fold(cache: FeatureCache, fold: Fold, h: int, weather_mode: str,
               train_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training rows use actual weather; test rows use `weather_mode` weather."""
    train = cache.get(h, "actual")
    train = train[(train.index >= train_start) & (train.index < fold.cutoff)].dropna(subset=["y"])
    test = cache.get(h, weather_mode)
    test = test[(test["origin"] >= fold.cutoff) & (test["origin"] < fold.end)].dropna(subset=["y"])
    return train, test


def run_backtest(model_factory: Callable[[], Forecaster], ds: Dataset, cfg: dict,
                 weather_mode: str = "forecast", horizons: list[int] | None = None) -> pd.DataFrame:
    horizons = horizons or cfg["horizons_hours"]
    bt = cfg["backtest"]
    folds = make_folds(bt["test_start"], ds.demand.dropna().index[-1], bt["fold_months"])
    train_start = pd.Timestamp(bt["train_start"], tz="UTC")
    cache = FeatureCache(ds, cfg)
    shared = getattr(model_factory, "horizon_independent", False)
    refit_every = getattr(model_factory, "refit_every_folds", 1)
    results, importances = [], []
    previous_fold_models: dict[int | None, Forecaster] = {}
    for fold in folds:
        fold_models: dict[int | None, Forecaster] = {}
        for h in horizons:
            train, test = split_fold(cache, fold, h, weather_mode, train_start)
            key = None if shared else h
            t0 = time.perf_counter()
            if key in fold_models:
                model, note = fold_models[key], "shared"
            elif fold.number % refit_every != 0 and key in previous_fold_models:
                model, note = previous_fold_models[key], "reused from previous fold"
            else:
                set_seeds(cfg["seed"])
                model = model_factory()
                model.fit(train, h)
                note = f"fit on {len(train)} rows"
                if hasattr(model, "importance"):
                    importances.append(model.importance.rename("gain").to_frame()
                                       .assign(fold=fold.number, horizon=h))
            fold_models[key] = model
            pred = model.predict(test, h, context=cache.get(h, weather_mode))
            log.info("%-8s h=%-3d fold %d %s..%s test=%d %-28s %.1fs", model.name, h, fold.number,
                     fold.cutoff.date(), fold.end.date(), len(test), note, time.perf_counter() - t0)
            results.append(pred.reindex(test.index).assign(
                y=test["y"], origin=test["origin"], fold=fold.number, horizon=h,
                model=model.name, weather=weather_mode))
        previous_fold_models = fold_models
    out = pd.concat(results).reset_index()
    out.attrs["importance"] = pd.concat(importances).reset_index(names="feature") if importances else None
    return out


def model_registry() -> dict[str, Callable[[], Forecaster]]:
    from demand_forecast.models.baseline import SameHourLastWeek
    from demand_forecast.models.prophet_ import ProphetForecaster
    from demand_forecast.models.tft import TFTForecaster
    from demand_forecast.models.xgb import XGBForecaster
    return {"baseline": SameHourLastWeek, "xgboost": XGBForecaster, "prophet": ProphetForecaster,
            "tft": TFTForecaster}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(model_registry()))
    parser.add_argument("--weather", choices=WEATHER_MODES, default="forecast")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config()
    preds = run_backtest(model_registry()[args.model], load_dataset(), cfg, args.weather)

    out_dir = PROJECT_ROOT / "artifacts" / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    importance = preds.attrs.pop("importance", None)
    preds.to_parquet(out_dir / f"{args.model}__{args.weather}.parquet")
    if importance is not None:
        importance.to_parquet(out_dir / f"{args.model}__{args.weather}__importance.parquet")

    table = score_table(preds, ["model", "weather", "horizon"])
    metrics_dir = PROJECT_ROOT / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(metrics_dir / f"{args.model}__{args.weather}.csv")
    with pd.option_context("display.float_format", "{:,.2f}".format):
        print(table)


if __name__ == "__main__":
    main()
