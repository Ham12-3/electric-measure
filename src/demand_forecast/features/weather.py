"""Weather features, and choosing which weather forecast is legal for each horizon.

Timing rules (all times UTC, hours labelled by their start):
* A forecast made at origin t for horizon h targets the hour T = t + h - 1 (it ends at t + h).
* Open-Meteo's `previous_dayN` value for valid time V came from a model run started about
  24*N hours before V, which became downloadable `publication_delay` hours later. So it
  was available at V - 24*N + delay.
* Radiation for hour T is stamped at its end (V = T + 1), and is the later of the two
  timestamps a feature for T looks at. Requiring V - 24*N + delay <= t gives:

      N >= (h + delay) / 24   ->   lead_day(h) = ceil((h + delay) / 24), capped at 7

  h=1 -> day 1, h=24 -> day 2, h=168 -> day 8 -> capped to 7. The 7-day horizon is the one
  place we're slightly optimistic: its forecast was published up to ~8h after the origin.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MAX_LEAD_DAY = 7


def lead_day_for_horizon(h: int, publication_delay_hours: int) -> int:
    return min(MAX_LEAD_DAY, max(1, math.ceil((h + publication_delay_hours) / 24)))


def forecast_weather_for_horizon(forecast_long: pd.DataFrame, h: int,
                                 publication_delay_hours: int,
                                 variables: list[str]) -> pd.DataFrame:
    """Hourly weather series that a forecaster at horizon h could legally have had.

    Uses lead day N = lead_day_for_horizon(h). Where N is missing, fall back to OLDER
    forecasts (N+1 ... 7), which were published even earlier, so they're also legal.
    Never fall back to newer ones.
    """
    n = lead_day_for_horizon(h, publication_delay_hours)
    by_lead = forecast_long[variables].unstack("lead_day")
    out = by_lead.xs(n, axis=1, level="lead_day").copy()
    for older in range(n + 1, MAX_LEAD_DAY + 1):
        out = out.fillna(by_lead.xs(older, axis=1, level="lead_day"))
    return out.asfreq("1h")


def weather_features(weather: pd.DataFrame, variables: list[str],
                     hdd_base: float, cdd_base: float) -> pd.DataFrame:
    """Features for target hour T from a weather series (actual OR one horizon's forecast).

    The 24h mean temperature looks back over hours T-23..T *of the same series*, so in
    forecast mode it's built entirely from forecasts. (Mixing in observed weather from
    after the origin would leak.) Buildings respond to recent temperature, not just the
    current hour.
    """
    t = weather["temperature_2m"]
    out = weather[variables].copy()
    out["temp_mean_24h"] = t.rolling(24, min_periods=18).mean()
    out["hdd"] = np.maximum(0.0, hdd_base - t)
    out["cdd"] = np.maximum(0.0, t - cdd_base)
    return out
