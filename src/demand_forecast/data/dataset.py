"""Load the cleaned files from data/processed/ into one object."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from demand_forecast.config import processed_dir


@dataclass
class Dataset:
    demand: pd.Series             # hourly ND (MW), UTC index; label = start of hour
    weather_actual: pd.DataFrame  # hourly reanalysis, UTC index
    weather_forecast: pd.DataFrame  # MultiIndex (time, lead_day), forecast for that time


def load_dataset() -> Dataset:
    p = processed_dir()
    demand = pd.read_parquet(p / "demand_hourly.parquet")["nd_mw"]
    return Dataset(
        demand=demand.asfreq("1h"),
        weather_actual=pd.read_parquet(p / "weather_actual.parquet").asfreq("1h"),
        weather_forecast=pd.read_parquet(p / "weather_forecast.parquet"),
    )
