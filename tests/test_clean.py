import numpy as np
import pandas as pd
import pytest

from demand_forecast.data.clean import (
    align_preceding_hour,
    expected_periods,
    flag_outliers,
    forecast_to_long,
    national_average,
    parse_settlement_dates,
    settlement_to_utc,
    to_hourly,
)


@pytest.mark.parametrize("raw", ["01-JAN-2021", "01-Jan-23", "2023-01-01"])
def test_parse_all_neso_date_formats(raw):
    parsed = parse_settlement_dates(pd.Series([raw]))
    assert parsed.iloc[0].month == 1 and parsed.iloc[0].day == 1


def test_expected_periods_on_clock_change_days():
    dates = pd.Series(pd.to_datetime(["2025-03-30", "2025-06-01", "2025-10-26"]))
    assert expected_periods(dates).tolist() == [46, 48, 50]


def _day(date: str, n: int) -> tuple[pd.Series, pd.Series]:
    return pd.Series(pd.to_datetime([date] * n)), pd.Series(range(1, n + 1))


def test_spring_clock_change_has_no_gaps_in_utc():
    # 46 periods: 00:00 GMT ... clocks jump at 01:00 GMT -> still contiguous in UTC.
    utc = settlement_to_utc(*_day("2025-03-30", 46))
    assert utc[0] == pd.Timestamp("2025-03-30 00:00", tz="UTC")
    assert ((utc[1:] - utc[:-1]) == pd.Timedelta("30min")).all()
    # Last period ends at local midnight BST (= 23:00 UTC).
    assert utc[-1] == pd.Timestamp("2025-03-30 22:30", tz="UTC")


def test_autumn_clock_change_has_no_duplicates_in_utc():
    # Local midnight is still BST (23:00 UTC the previous day); 50 periods follow.
    utc = settlement_to_utc(*_day("2025-10-26", 50))
    assert utc[0] == pd.Timestamp("2025-10-25 23:00", tz="UTC")
    assert utc.is_unique
    assert utc[-1] == pd.Timestamp("2025-10-26 23:30", tz="UTC")


def test_summer_period_one_is_2300_utc_previous_day():
    utc = settlement_to_utc(pd.Series(pd.to_datetime(["2025-07-01"])), pd.Series([1]))
    assert utc[0] == pd.Timestamp("2025-06-30 23:00", tz="UTC")


def test_hourly_is_mean_and_needs_both_halfhours():
    idx = pd.date_range("2025-01-01", periods=4, freq="30min", tz="UTC")
    half = pd.DataFrame({"nd_mw": [100.0, 200.0, 300.0, np.nan],
                         "is_interpolated": [False] * 4}, index=idx)
    hourly = to_hourly(half)
    assert hourly["nd_mw"].iloc[0] == 150.0
    assert np.isnan(hourly["nd_mw"].iloc[1])


def test_flag_outliers_catches_spike_but_not_smooth_peak():
    t = np.arange(48 * 7)
    nd = pd.Series(25000 + 8000 * np.sin(2 * np.pi * t / 48))
    nd.iloc[100] = nd.iloc[100] + 15000  # single bad reading
    nd.iloc[200] = 5000                  # physically impossible
    flags = flag_outliers(nd, lo=10000, hi=65000)
    assert flags.iloc[100] and flags.iloc[200]
    assert flags.sum() == 2


def test_align_preceding_hour_shifts_only_accumulated_vars():
    idx = pd.date_range("2025-06-01 12:00", periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame({"shortwave_radiation": [1.0, 2.0, 3.0],
                       "shortwave_radiation_previous_day2": [1.0, 2.0, 3.0],
                       "temperature_2m": [10.0, 11.0, 12.0]}, index=idx)
    out = align_preceding_hour(df, ["shortwave_radiation"])
    assert out["shortwave_radiation"].iloc[0] == 2.0  # 12:00 now means 12:00-13:00
    assert out["shortwave_radiation_previous_day2"].iloc[0] == 2.0
    assert out["temperature_2m"].iloc[0] == 10.0


def test_national_average_renormalises_missing_city():
    idx = pd.date_range("2025-01-01", periods=2, freq="1h", tz="UTC")
    frames = {"a": pd.DataFrame({"t": [10.0, 10.0]}, index=idx),
              "b": pd.DataFrame({"t": [20.0, np.nan]}, index=idx)}
    nat = national_average(frames, {"a": 0.75, "b": 0.25})
    assert nat["t"].tolist() == [12.5, 10.0]


def test_forecast_to_long_keys_by_lead_day():
    idx = pd.date_range("2025-01-01", periods=2, freq="1h", tz="UTC", name="time")
    wide = pd.DataFrame({"t_previous_day1": [1.0, 2.0], "t_previous_day2": [3.0, 4.0]}, index=idx)
    long = forecast_to_long(wide, ["t"], [1, 2])
    assert long.loc[(idx[0], 2), "t"] == 3.0
    assert len(long) == 4
