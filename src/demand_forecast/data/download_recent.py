"""Near-real-time demand from Elexon BMRS, to fill the ~3-week gap after NESO's file.

NESO's historic demand file is published with a ~3 week lag, so it can't drive a live
"next 7 days" forecast. Elexon publishes Initial National Demand Outturn (INDO) within
~30 minutes. Over 1,200 overlapping half-hours (Aug-Sep 2026) INDO matched NESO ND to
0.10% on average (corr 1.0000), so we treat it as the same series. NESO values always
win where both exist, because they're the settled figures.

API: https://data.elexon.co.uk/bmrs/api/v1/demand/outturn (no key; max 28 days/request).
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
import requests

log = logging.getLogger(__name__)

URL = "https://data.elexon.co.uk/bmrs/api/v1/demand/outturn"
MAX_DAYS = 27


def fetch_indo(start: dt.date, end: dt.date) -> pd.Series:
    """Half-hourly INDO (MW) indexed by UTC period start."""
    values: dict[pd.Timestamp, float] = {}
    day = start
    while day <= end:
        chunk_end = min(end, day + dt.timedelta(days=MAX_DAYS))
        resp = requests.get(URL, params={"settlementDateFrom": day.isoformat(),
                                         "settlementDateTo": chunk_end.isoformat(),
                                         "format": "json"}, timeout=60)
        resp.raise_for_status()
        for row in resp.json()["data"]:
            values[pd.Timestamp(row["startTime"])] = float(row["initialDemandOutturn"])
        day = chunk_end + dt.timedelta(days=1)
    s = pd.Series(values, dtype=float).sort_index()
    s.index = pd.DatetimeIndex(s.index).tz_convert("UTC")
    s.index.name = "time"
    return s


def extend_hourly_demand(hourly: pd.Series, end: dt.date | None = None) -> pd.Series:
    """Append Elexon hours after the last NESO hour. Only complete hours are kept."""
    last = hourly.dropna().index[-1]
    end = end or dt.date.today()
    recent = fetch_indo(last.date(), end)
    recent = recent[recent.index > last + pd.Timedelta("30min")]
    if recent.empty:
        return hourly
    grouped = recent.resample("1h")
    rec_hourly = grouped.mean()[grouped.count() == 2]
    log.info("Elexon: added %d hours (%s .. %s)", len(rec_hourly), rec_hourly.index[0], rec_hourly.index[-1])
    return pd.concat([hourly.dropna(), rec_hourly]).sort_index().asfreq("1h")
