"""Prophet: a curve-fitting model, demand = trend + seasonal cycles + weather effects.

Prophet fits a smooth long-term trend, repeating daily/weekly/yearly patterns, and linear
effects of extra regressors (here heating/cooling degrees, wind, sunshine, holidays).
It does NOT use recent demand (no lags). Its forecast for next Tuesday 18:00 is the same
whether it's made an hour or a week in advance; only the weather input differs. So one
model per fold serves all horizons, and we expect it to be weak at 1h, where recent
demand is the best clue.

Daily shape: a single daily cycle can't describe both a winter weekday (big 17:00 peak)
and a summer Sunday. So we fit four daily cycles and switch between them with
conditions: {GMT, BST} x {working day, day off}. Using GMT/BST as the switch also fixes
clock alignment, since Prophet works on clock time and we feed it UTC. In BST the whole
daily pattern happens an hour earlier in UTC, and the BST cycle learns that.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from demand_forecast.config import load_config

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

REGRESSORS = ["hdd", "cdd", "temp_mean_24h", "wind_speed_10m", "shortwave_radiation",
              "is_holiday_ew", "is_holiday_sct", "is_christmas_period"]


def _prophet_frame(df: pd.DataFrame) -> pd.DataFrame:
    utc = df.index
    is_bst = utc.tz_convert("Europe/London").map(lambda t: t.dst() != pd.Timedelta(0)).to_numpy(bool)
    day_off = (df["is_weekend"] == 1) | (df["is_holiday_ew"] == 1) | (df["is_christmas_period"] == 1)
    out = pd.DataFrame({"ds": utc.tz_localize(None)}, index=df.index)
    for clock, clock_mask in (("gmt", ~is_bst), ("bst", is_bst)):
        out[f"{clock}_workday"] = clock_mask & ~day_off.to_numpy()
        out[f"{clock}_dayoff"] = clock_mask & day_off.to_numpy()
    for r in REGRESSORS:
        out[r] = df[r].astype(float)
    if "y" in df:
        out["y"] = df["y"]
    return out


class ProphetForecaster:
    name = "prophet"
    horizon_independent = True

    def __init__(self):
        cfg = load_config()
        self.params = cfg["models"]["prophet"]
        self.seed = cfg["seed"]

    def fit(self, train: pd.DataFrame, h: int) -> None:
        from prophet import Prophet

        p = self.params
        m = Prophet(daily_seasonality=False, weekly_seasonality=p["weekly_fourier"],
                    yearly_seasonality=p["yearly_fourier"],
                    changepoint_prior_scale=p["changepoint_prior_scale"],
                    uncertainty_samples=0)  # we only need the point forecast; much faster
        for cond in ("gmt_workday", "gmt_dayoff", "bst_workday", "bst_dayoff"):
            m.add_seasonality(name=f"daily_{cond}", period=1, fourier_order=p["daily_fourier"],
                              condition_name=cond)
        for r in REGRESSORS:
            m.add_regressor(r)
        frame = _prophet_frame(train.dropna(subset=REGRESSORS + ["y"]))
        np.random.seed(self.seed)
        m.fit(frame.reset_index(drop=True), seed=self.seed)
        self.model = m

    def predict(self, test: pd.DataFrame, h: int, context: pd.DataFrame | None = None) -> pd.DataFrame:
        frame = _prophet_frame(test)
        ok = frame[REGRESSORS].notna().all(axis=1).to_numpy()
        y_pred = np.full(len(frame), np.nan)
        if ok.any():
            # Prophet sorts by ds internally; our UTC timestamps are unique, so map back by ds.
            fc = self.model.predict(frame[ok].reset_index(drop=True))
            y_pred[ok] = fc.set_index("ds")["yhat"].reindex(frame.loc[ok, "ds"]).to_numpy()
        return pd.DataFrame({"y_pred": y_pred}, index=test.index)
