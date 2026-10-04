"""Feature tests. The leakage tests are the important ones: they change data that a
forecaster couldn't have known yet, and assert that no feature changes."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from demand_forecast.evaluation.backtest import FeatureCache
from demand_forecast.features.build import build_features, feature_columns
from demand_forecast.features.calendar import calendar_features
from demand_forecast.features.lags import lag_features
from demand_forecast.features.weather import (
    forecast_weather_for_horizon,
    lead_day_for_horizon,
)

HORIZONS = [1, 24, 168]


@pytest.mark.parametrize("h", HORIZONS)
def test_lag_features_only_use_demand_known_at_origin(synthetic_ds, cfg, h):
    demand = synthetic_ds.demand
    target = demand.index[24 * 20]
    latest_known = target - pd.Timedelta(hours=h)
    lags, windows = cfg["features"]["demand_lags_hours"], cfg["features"]["rolling_windows_hours"]

    before = lag_features(demand, h, lags, windows).loc[target]
    tampered = demand.copy()
    tampered[tampered.index > latest_known] += 1e6  # change everything not yet known
    after = lag_features(tampered, h, lags, windows).loc[target]

    pd.testing.assert_series_equal(before, after)


@pytest.mark.parametrize("h", HORIZONS)
def test_no_lag_shorter_than_horizon(synthetic_ds, cfg, h):
    cols = lag_features(synthetic_ds.demand, h, cfg["features"]["demand_lags_hours"],
                        cfg["features"]["rolling_windows_hours"]).columns
    lags = [int(c.split("_")[1]) for c in cols if c.startswith("lag_")]
    assert min(lags) == h
    assert "lag_168" in cols  # the baseline needs it at every horizon


def test_seven_day_horizon_cannot_see_yesterday(synthetic_ds, cfg):
    cols = lag_features(synthetic_ds.demand, 168, cfg["features"]["demand_lags_hours"],
                        cfg["features"]["rolling_windows_hours"]).columns
    assert "lag_24" not in cols and "lag_1" not in cols


@pytest.mark.parametrize("h,expected", [(1, 1), (16, 1), (17, 2), (24, 2), (48, 3), (160, 7), (168, 7)])
def test_lead_day_for_horizon(h, expected):
    # Forecast for hour T (ending T+1) at lead N was out at T+1 - 24N + 8h; it must be <= origin.
    assert lead_day_for_horizon(h, publication_delay_hours=8) == expected


@pytest.mark.parametrize("h", HORIZONS)
def test_forecast_weather_uses_the_legal_lead_day(synthetic_ds, h):
    w = forecast_weather_for_horizon(synthetic_ds.weather_forecast, h, 8, ["temperature_2m"])
    assert (w["temperature_2m"] == 100.0 * lead_day_for_horizon(h, 8)).all()


def test_forecast_weather_falls_back_only_to_older_runs(synthetic_ds):
    fc = synthetic_ds.weather_forecast.copy()
    t0 = fc.index.get_level_values("time")[100]
    fc.loc[(t0, 2), "temperature_2m"] = np.nan  # day-2 forecast missing for one hour
    w = forecast_weather_for_horizon(fc, 24, 8, ["temperature_2m"])  # h=24 -> lead 2
    assert w.loc[t0, "temperature_2m"] == 300.0  # day 3 (older), never day 1 (newer)


@pytest.mark.parametrize("h", HORIZONS)
def test_forecast_mode_features_ignore_actual_weather(synthetic_ds, cfg, h):
    """In honest mode, observed weather must have no influence on test features at all."""
    tampered = dataclasses.replace(synthetic_ds, weather_actual=synthetic_ds.weather_actual + 50)
    a = FeatureCache(synthetic_ds, cfg).get(h, "forecast")
    b = FeatureCache(tampered, cfg).get(h, "forecast")
    pd.testing.assert_frame_equal(a, b)
    # ...whereas the "actual" (cheating) features do change, so the test can fail.
    assert not FeatureCache(synthetic_ds, cfg).get(h, "actual").equals(FeatureCache(tampered, cfg).get(h, "actual"))


def test_build_features_origin_and_columns(synthetic_ds, cfg):
    frame = build_features(synthetic_ds.demand, synthetic_ds.weather_actual, 24, cfg)
    assert (frame.index - frame["origin"] == pd.Timedelta(hours=23)).all()
    cols = feature_columns(frame)
    assert "y" not in cols and "origin" not in cols
    assert {"hour", "is_holiday_ew", "lag_24", "temperature_2m", "hdd", "temp_mean_24h"} <= set(cols)


def test_calendar_uses_local_time_and_flags_holidays():
    idx = pd.DatetimeIndex(["2025-07-01 06:00", "2025-12-25 12:00", "2025-12-29 12:00",
                            "2025-08-04 12:00", "2025-08-25 12:00"], tz="UTC")
    cal = calendar_features(idx)
    assert cal["hour"].iloc[0] == 7  # 06:00 UTC = 07:00 BST
    assert cal["is_holiday_ew"].iloc[1] == 1 and cal["is_christmas_period"].iloc[1] == 1
    assert cal["is_holiday_ew"].iloc[2] == 0 and cal["is_christmas_period"].iloc[2] == 1
    # First Monday of August is a Scottish bank holiday; last Monday is England & Wales.
    assert (cal["is_holiday_sct"].iloc[3], cal["is_holiday_ew"].iloc[3]) == (1, 0)
    assert (cal["is_holiday_sct"].iloc[4], cal["is_holiday_ew"].iloc[4]) == (0, 1)
