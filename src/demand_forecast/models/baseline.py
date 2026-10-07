"""Baseline: "same hour last week".

Demand for Tuesday 18:00 is predicted as last Tuesday's 18:00. It gets the daily and
weekly shape right for free, and it's legal for every horizon up to 7 days. Any model
that can't beat it isn't worth having.

Note: the lag is 168 hours in UTC, so in the week after a clock change it compares hours
that are one hour apart in local time. That's a small, honest handicap.
"""

from __future__ import annotations

import pandas as pd


class SameHourLastWeek:
    name = "baseline"

    def fit(self, train: pd.DataFrame, h: int) -> None:
        if h > 168:
            raise ValueError("same-hour-last-week is only available for horizons <= 168h")

    def predict(self, test: pd.DataFrame, h: int, context: pd.DataFrame | None = None) -> pd.DataFrame:
        return pd.DataFrame({"y_pred": test["lag_168"]}, index=test.index)
