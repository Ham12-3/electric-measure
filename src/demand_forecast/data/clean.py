"""Turn raw NESO + Open-Meteo downloads into clean hourly parquet files.

Outputs (data/processed/):
    demand_halfhourly.parquet     ND at native resolution, UTC index, with quality flags
    demand_hourly.parquet         target series: hourly mean ND (MW), UTC index
    weather_actual.parquet        national population-weighted reanalysis weather, hourly
    weather_forecast.parquet      national forecast weather, one row per (time, lead_day)
    weather_hist_forecast.parquet national "stitched" historical-forecast weather, hourly
    cleaning_report.json          what was fixed and how much

Key decisions (plain English):
* Everything is stored in UTC. UK local time jumps an hour twice a year; UTC never does,
  so every hour exists exactly once and there are no gaps or duplicates to special-case.
* NESO counts settlement periods from local midnight: period 1 = 00:00-00:30 UK time.
  On the spring clock change there are 46 periods, in autumn 50. Converting
  "local midnight -> UTC, then add (period-1) x 30 min" handles all three day types.
* Hourly demand is the MEAN of the two half-hours, because ND is power (MW), not energy.
* Short gaps (<= 2h) are linearly interpolated; longer gaps stay missing. Inventing hours
  of data would make the backtest score itself on made-up values.
* Outliers: we only remove values that are physically impossible or isolated spikes that
  disagree wildly with both neighbours. Genuine extremes (cold snaps, Christmas, the 2022
  price shock) are real demand and must stay in.

Usage:
    uv run python -m demand_forecast.data.clean
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from demand_forecast.config import load_config, processed_dir, raw_dir

log = logging.getLogger(__name__)

UK_TZ = "Europe/London"
# NESO has changed its date format over the years: "01-JAN-2021", "01-Jan-23", "2025-01-01".
DATE_FORMATS = ("%d-%b-%Y", "%d-%b-%y", "%Y-%m-%d")


# --------------------------------------------------------------------------- demand

def parse_settlement_dates(values: pd.Series) -> pd.Series:
    """Parse NESO settlement dates, trying each known format until one fits every row."""
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(values, format=fmt, errors="coerce")
        if parsed.notna().all():
            return parsed
    bad = values[pd.to_datetime(values, format=DATE_FORMATS[-1], errors="coerce").isna()]
    raise ValueError(f"unrecognised settlement date format, e.g. {bad.iloc[0]!r}")


def expected_periods(dates: pd.Series) -> pd.Series:
    """Number of half-hours in each UK local day: 48 normally, 46 / 50 on clock changes."""
    midnight = pd.DatetimeIndex(dates).tz_localize(UK_TZ)
    next_midnight = pd.DatetimeIndex(dates + pd.Timedelta(days=1)).tz_localize(UK_TZ)
    hours = (next_midnight.tz_convert("UTC") - midnight.tz_convert("UTC")) / pd.Timedelta("30min")
    return pd.Series(hours.astype(int), index=dates.index)


def settlement_to_utc(dates: pd.Series, periods: pd.Series) -> pd.DatetimeIndex:
    """Start time (UTC) of each settlement period."""
    local_midnight = pd.DatetimeIndex(dates).tz_localize(UK_TZ).tz_convert("UTC")
    return local_midnight + pd.to_timedelta((periods.to_numpy() - 1) * 30, unit="min")


def load_raw_demand(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in sorted(paths):
        df = pd.read_csv(path)
        # The live current-year file mixes actuals ("A") with forecasts ("F"); keep actuals.
        if "FORECAST_ACTUAL_INDICATOR" in df.columns:
            df = df[df["FORECAST_ACTUAL_INDICATOR"] == "A"]
        df["SETTLEMENT_DATE"] = parse_settlement_dates(df["SETTLEMENT_DATE"])
        frames.append(df[["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND",
                          "EMBEDDED_SOLAR_GENERATION", "EMBEDDED_WIND_GENERATION"]])
    return pd.concat(frames, ignore_index=True)


def flag_outliers(nd: pd.Series, lo: float, hi: float, k: float = 10.0) -> pd.Series:
    """True where a value is impossible or an isolated spike.

    A spike is a point far from the median of itself and its 2 neighbours on each side.
    "Far" means more than k times the typical half-hour-to-half-hour change in demand.
    Ramps and peaks are smooth, so they barely move this residual; single bad readings
    stand out sharply.
    """
    impossible = (nd < lo) | (nd > hi)
    resid = nd - nd.rolling(5, center=True, min_periods=3).median()
    typical_step = nd.diff().abs().median()
    spike = resid.abs() > k * typical_step
    return impossible | spike


def clean_demand(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    dcfg = cfg["demand"]
    report: dict = {"raw_rows": len(raw)}

    # 1. Check every day has the right number of periods for its type (46 / 48 / 50).
    counts = raw.groupby("SETTLEMENT_DATE")["SETTLEMENT_PERIOD"].agg(["count", "max"])
    expected = expected_periods(pd.Series(counts.index, index=counts.index))
    bad_days = counts[(counts["count"] != expected) | (counts["max"] != expected)]
    report["clock_change_days"] = int((expected != 48).sum())
    report["days_with_wrong_period_count"] = [d.date().isoformat() for d in bad_days.index]

    # 2. Settlement date + period -> UTC timestamp.
    df = pd.DataFrame({
        "nd_mw": raw["ND"].astype(float).to_numpy(),
        "embedded_solar_mw": raw["EMBEDDED_SOLAR_GENERATION"].astype(float).to_numpy(),
        "embedded_wind_mw": raw["EMBEDDED_WIND_GENERATION"].astype(float).to_numpy(),
    }, index=settlement_to_utc(raw["SETTLEMENT_DATE"], raw["SETTLEMENT_PERIOD"]))
    df.index.name = "time"

    # 3. Duplicates (e.g. overlapping files): keep the latest published value.
    report["duplicate_rows"] = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # 4. Put on a complete 30-min grid so missing periods become explicit NaNs.
    full = pd.date_range(df.index[0], df.index[-1], freq="30min", tz="UTC", name="time")
    df = df.reindex(full)
    report["missing_halfhours"] = int(df["nd_mw"].isna().sum())

    # 5. Outliers -> NaN (then treated like any other gap).
    outlier = flag_outliers(df["nd_mw"], dcfg["plausible_min_mw"], dcfg["plausible_max_mw"])
    report["outliers_removed"] = int(outlier.sum())
    report["outlier_times"] = [t.isoformat() for t in df.index[outlier]][:50]
    df["is_outlier"] = outlier
    df.loc[outlier, "nd_mw"] = np.nan

    # 6. Fill only short gaps. limit_area="inside" never extrapolates past the ends.
    was_nan = df["nd_mw"].isna()
    df["nd_mw"] = df["nd_mw"].interpolate(
        method="time", limit=dcfg["max_interp_halfhours"], limit_area="inside")
    df["is_interpolated"] = was_nan & df["nd_mw"].notna()
    report["interpolated_halfhours"] = int(df["is_interpolated"].sum())
    report["still_missing_halfhours"] = int(df["nd_mw"].isna().sum())
    return df, report


def to_hourly(halfhourly: pd.DataFrame) -> pd.DataFrame:
    """Hourly mean ND. An hour is only valid if both of its half-hours are present."""
    grouped = halfhourly["nd_mw"].resample("1h")
    hourly = pd.DataFrame({
        "nd_mw": grouped.mean(),
        "n_halfhours": grouped.count(),
        "any_interpolated": halfhourly["is_interpolated"].resample("1h").max().astype(bool),
    })
    hourly.loc[hourly["n_halfhours"] < 2, "nd_mw"] = np.nan
    return hourly


# --------------------------------------------------------------------------- weather

def load_weather_json(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC")
    return df.set_index("time")


def load_city_frames(source: str, cfg: dict) -> dict[str, pd.DataFrame]:
    frames = {}
    folder = raw_dir("weather", source)
    for city in cfg["weather"]["cities"]:
        files = sorted(folder.glob(f"{city}_*.json"))
        if not files:
            raise FileNotFoundError(f"no {source} weather for {city}; run download_weather")
        df = pd.concat([load_weather_json(p) for p in files]).sort_index()
        frames[city] = df[~df.index.duplicated(keep="last")]
    return frames


def align_preceding_hour(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Relabel 'preceding hour' variables to the start of the hour they describe.

    Open-Meteo stamps hourly radiation/precipitation at the END of the hour
    (13:00 = average over 12:00-13:00). Our demand hour 12:00 also means 12:00-13:00,
    so these columns move back one hour. Instantaneous variables are left as they are.
    """
    df = df.copy()
    cols = [c for c in df.columns if any(c == v or c.startswith(v + "_") for v in variables)]
    df[cols] = df[cols].shift(-1)
    return df


def national_average(frames: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    """Population-weighted mean across cities, re-normalising weights if a city is missing."""
    cols = list(next(iter(frames.values())).columns)
    index = sorted(set().union(*(f.index for f in frames.values())))
    out = {}
    for col in cols:
        values = pd.DataFrame({c: f[col] for c, f in frames.items()}).reindex(index)
        w = pd.Series(weights)[values.columns]
        mask = values.notna()
        out[col] = (values.fillna(0) * w).sum(axis=1) / (mask * w).sum(axis=1).replace(0, np.nan)
    result = pd.DataFrame(out)
    result.index.name = "time"
    return result


def clean_weather_source(source: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    wcfg = cfg["weather"]
    frames = load_city_frames(source, cfg)
    frames = {c: align_preceding_hour(f, wcfg["preceding_hour_variables"]) for c, f in frames.items()}
    weights = {c: v["weight"] for c, v in wcfg["cities"].items()}
    nat = national_average(frames, weights)
    nat = nat.asfreq("1h")
    missing_before = int(nat.isna().any(axis=1).sum())
    nat = nat.interpolate(method="time", limit=3, limit_area="inside")
    report = {"rows": len(nat), "start": nat.index[0].isoformat(), "end": nat.index[-1].isoformat(),
              "hours_with_gaps_before_fill": missing_before,
              "hours_with_gaps_after_fill": int(nat.isna().any(axis=1).sum())}
    return nat, report


def forecast_to_long(wide: pd.DataFrame, variables: list[str], lead_days: list[int]) -> pd.DataFrame:
    """`temperature_2m_previous_day3` columns -> rows keyed by (time, lead_day)."""
    parts = []
    for d in lead_days:
        part = wide[[f"{v}_previous_day{d}" for v in variables]].copy()
        part.columns = variables
        part["lead_day"] = d
        parts.append(part)
    long = pd.concat(parts).reset_index().set_index(["time", "lead_day"]).sort_index()
    return long.dropna(how="all")


# --------------------------------------------------------------------------- main

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config()
    out = processed_dir()
    report: dict = {}

    raw = load_raw_demand(sorted(raw_dir("neso").glob("demanddata_*.csv")))
    half, report["demand"] = clean_demand(raw, cfg)
    hourly = to_hourly(half)
    report["demand"]["hourly_rows"] = len(hourly)
    report["demand"]["hourly_missing"] = int(hourly["nd_mw"].isna().sum())
    report["demand"]["start"] = hourly.index[0].isoformat()
    report["demand"]["end"] = hourly.index[-1].isoformat()
    half.to_parquet(out / "demand_halfhourly.parquet")
    hourly.to_parquet(out / "demand_hourly.parquet")
    log.info("demand: %s", {k: v for k, v in report["demand"].items() if k != "outlier_times"})

    variables = cfg["weather"]["variables"]
    actual, report["weather_actual"] = clean_weather_source("archive", cfg)
    actual[variables].to_parquet(out / "weather_actual.parquet")

    hist_fc, report["weather_hist_forecast"] = clean_weather_source("historical_forecast", cfg)
    hist_fc[variables].to_parquet(out / "weather_hist_forecast.parquet")

    fc_wide, report["weather_forecast"] = clean_weather_source("previous_runs", cfg)
    fc = forecast_to_long(fc_wide, variables, cfg["weather"]["lead_days"])
    report["weather_forecast"]["rows_by_lead_day"] = fc.groupby("lead_day").size().to_dict()
    fc.to_parquet(out / "weather_forecast.parquet")
    for name in ("weather_actual", "weather_hist_forecast", "weather_forecast"):
        log.info("%s: %s", name, report[name])

    (out / "cleaning_report.json").write_text(json.dumps(report, indent=2, default=str))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
