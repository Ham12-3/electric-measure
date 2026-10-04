"""Lagged-demand features that respect the forecast horizon.

For horizon h, the newest demand we know is from hour T - h (see weather.py for the
timing convention). So:
* a lag of k hours is allowed only if k >= h,
* rolling means are taken over windows that END at T - h.

A 7-day-ahead forecast therefore can't see yesterday's demand. It only gets demand from
a week or more before the target.
"""

from __future__ import annotations

import pandas as pd


def lag_features(demand: pd.Series, h: int, lags: list[int], windows: list[int]) -> pd.DataFrame:
    out = {}
    for k in sorted(set(lags) | {h}):  # always include the most recent legal value
        if k >= h:
            out[f"lag_{k}"] = demand.shift(k)
    latest_known = demand.shift(h)
    for w in windows:
        out[f"roll_mean_{w}"] = latest_known.rolling(w, min_periods=int(w * 0.75)).mean()
    return pd.DataFrame(out, index=demand.index)
