import numpy as np
import pandas as pd
import pytest

from demand_forecast.config import load_config
from demand_forecast.data.dataset import Dataset


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def synthetic_ds() -> Dataset:
    """Five weeks of fake hourly data. Forecast values encode their lead day (lead * 100)."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=24 * 35, freq="1h", tz="UTC", name="time")
    hour = idx.hour.to_numpy()
    demand = pd.Series(25000 + 5000 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 300, len(idx)),
                       index=idx)
    vars_ = ["temperature_2m", "wind_speed_10m", "shortwave_radiation"]
    actual = pd.DataFrame({v: rng.normal(10, 3, len(idx)) for v in vars_}, index=idx)
    fc = pd.concat([
        pd.DataFrame({v: np.full(len(idx), lead * 100.0) for v in vars_}, index=idx).assign(lead_day=lead)
        for lead in range(1, 8)
    ]).reset_index().set_index(["time", "lead_day"]).sort_index()
    return Dataset(demand=demand, weather_actual=actual, weather_forecast=fc)
