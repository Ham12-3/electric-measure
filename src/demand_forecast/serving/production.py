"""Production XGBoost models and the 168-hour forecast they produce.

XGBoost won the backtest at every horizon, so it serves the forecast. It's a "direct"
model: one model per horizon. The API needs every hour 1..168, so we train a handful of
horizons H (config: serving.model_horizons) and forecast hour h with the model for the
smallest H >= h.

Why that's still leakage-free: a model trained for horizon H only uses demand lags >= H.
Since H >= h, every lag it reads is from before the forecast origin. For example, hour 30
uses the 72h model: slightly less fresh information than ideal, but never peeking.

Usage (daily retrain):
    uv run python -m demand_forecast.serving.production
"""

from __future__ import annotations

import bisect
import datetime as dt
import json
import logging
from pathlib import Path

import pandas as pd
import xgboost as xgb

from demand_forecast.config import PROJECT_ROOT, load_config
from demand_forecast.features.build import build_features, feature_columns
from demand_forecast.models.xgb import XGBForecaster

log = logging.getLogger(__name__)

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
FORECAST_HOURS = 168


def model_horizon_for(h: int, horizons: list[int]) -> int:
    """Smallest trained horizon >= h (always legal: it only uses older demand)."""
    hs = sorted(horizons)
    i = bisect.bisect_left(hs, h)
    if i == len(hs):
        raise ValueError(f"no model covers horizon {h}h (max {hs[-1]}h)")
    return hs[i]


def train_production(demand: pd.Series, weather_actual: pd.DataFrame, cfg: dict,
                     out_dir: Path = MODEL_DIR, xgb_params: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = cfg["serving"]["model_horizons"]
    train_start = pd.Timestamp(cfg["backtest"]["train_start"], tz="UTC")
    meta = {"model": "xgboost", "trained_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "trained_until": demand.dropna().index[-1].isoformat(), "horizons": {}}
    for H in horizons:
        frame = build_features(demand, weather_actual, H, cfg)
        train = frame[frame.index >= train_start].dropna(subset=["y"])
        model = XGBForecaster(xgb_params)
        model.fit(train, H)
        model.model.save_model(out_dir / f"xgb_h{H}.json")
        meta["horizons"][str(H)] = {"columns": model.columns, "n_trees": model.best_n, "rows": len(train)}
        log.info("trained h=%dh on %d rows (%d trees)", H, len(train), model.best_n)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


class ProductionModel:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.meta = json.loads((model_dir / "meta.json").read_text())
        self.horizons = sorted(int(h) for h in self.meta["horizons"])
        self.models = {}
        for H in self.horizons:
            booster = xgb.XGBRegressor()
            booster.load_model(model_dir / f"xgb_h{H}.json")
            self.models[H] = booster

    def forecast(self, demand: pd.Series, weather: pd.DataFrame, cfg: dict,
                 hours: int = FORECAST_HOURS) -> pd.DataFrame:
        """Forecast the `hours` hours after the last known demand hour.

        `demand`: hourly history (needs >= 3 weeks for the lags/rolling means).
        `weather`: hourly weather covering 24h before the origin through the last target
        (in live use, Open-Meteo's forecast; it's what would be known right now).
        """
        demand = demand.dropna()
        origin = demand.index[-1] + pd.Timedelta("1h")
        targets = pd.date_range(origin, periods=hours, freq="1h", tz="UTC", name="target")
        history = demand[demand.index >= origin - pd.Timedelta(days=28)]
        y = history.reindex(history.index.union(targets)).asfreq("1h")
        rows = []
        for H in self.horizons:
            hs = [h for h in range(1, hours + 1) if model_horizon_for(h, self.horizons) == H]
            if not hs:
                continue
            frame = build_features(y, weather, H, cfg)
            cols = self.meta["horizons"][str(H)]["columns"]
            sel = frame.loc[targets[[h - 1 for h in hs]], cols]
            rows.append(pd.DataFrame({"horizon_h": hs, "model_horizon_h": H,
                                      "forecast_mw": self.models[H].predict(sel)}, index=sel.index))
        out = pd.concat(rows).sort_index()
        out.attrs["origin"] = origin
        return out


def main() -> None:
    from demand_forecast.data.dataset import load_dataset

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ds = load_dataset()
    meta = train_production(ds.demand, ds.weather_actual, load_config())
    log.info("saved %d models to %s (data until %s)", len(meta["horizons"]), MODEL_DIR, meta["trained_until"])


if __name__ == "__main__":
    main()
