"""Assemble the feature table for one horizon.

One row per target hour T. Columns: `y` (the demand to predict), `origin` (when the
forecast is made), then calendar, lag and weather features. Which weather to use is the
caller's choice: actual weather for training, the horizon's forecast weather for honest
testing.
"""

from __future__ import annotations

import pandas as pd

from demand_forecast.features.calendar import calendar_features
from demand_forecast.features.lags import lag_features
from demand_forecast.features.weather import weather_features


def build_features(demand: pd.Series, weather: pd.DataFrame, h: int, cfg: dict) -> pd.DataFrame:
    fcfg = cfg["features"]
    index = demand.index
    w = weather_features(weather.reindex(index), fcfg["weather_vars"],
                         fcfg["hdd_base_c"], fcfg["cdd_base_c"])
    frame = pd.concat([
        pd.DataFrame({"y": demand, "origin": index - pd.Timedelta(hours=h - 1)}, index=index),
        calendar_features(index),
        lag_features(demand, h, fcfg["demand_lags_hours"], fcfg["rolling_windows_hours"]),
        w,
    ], axis=1)
    frame.index.name = "target"
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ("y", "origin")]
