"""Live inputs for the API: latest demand and the current weather forecast.

* Demand: cleaned NESO history (data/processed) extended to ~now with Elexon INDO.
* Weather: Open-Meteo's live forecast API for the same 5 cities and the same ECMWF model
  used in the backtest, from 7 days ago (for the 24h rolling temperature) to 9 days ahead.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time

import pandas as pd
import requests

from demand_forecast.config import load_config, processed_dir
from demand_forecast.data.clean import align_preceding_hour, national_average
from demand_forecast.data.download_recent import extend_hourly_demand
from demand_forecast.serving import storage
from demand_forecast.serving.production import ProductionModel

log = logging.getLogger(__name__)
LIVE_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def live_weather(cfg: dict, past_days: int = 7, forecast_days: int = 9) -> pd.DataFrame:
    w = cfg["weather"]
    variables = cfg["features"]["weather_vars"]
    frames = {}
    for city, c in w["cities"].items():
        resp = requests.get(LIVE_FORECAST_URL, params={
            "latitude": c["lat"], "longitude": c["lon"], "hourly": ",".join(variables),
            "models": w["forecast_model"], "timezone": "UTC",
            "past_days": past_days, "forecast_days": forecast_days}, timeout=60)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json()["hourly"])
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC")
        frames[city] = align_preceding_hour(df.set_index("time"), w["preceding_hour_variables"])
    nat = national_average(frames, {c: v["weight"] for c, v in w["cities"].items()})
    # ECMWF open data is 3/6-hourly far out; fill hourly gaps by interpolation.
    return nat.asfreq("1h").interpolate(method="time", limit_area="inside").ffill()


def recent_demand(extend: bool = True) -> pd.Series:
    hourly = pd.read_parquet(processed_dir() / "demand_hourly.parquet")["nd_mw"].asfreq("1h")
    if extend:
        try:
            hourly = extend_hourly_demand(hourly)
        except requests.RequestException as e:  # serve a slightly older forecast rather than fail
            log.warning("Elexon unavailable (%s); using NESO data only", e)
    return hourly


class ForecastService:
    """Computes the live forecast, caching it so repeated API calls are instant."""

    REFRESH_SECONDS = 6 * 3600  # how often to look for a newly retrained model in S3

    def __init__(self, ttl_seconds: int = 1800):
        self.cfg = load_config()
        storage.pull()  # no-op unless ARTIFACT_BUCKET is set
        self._pulled_at = time.time()
        self.model = ProductionModel()
        self.ttl = ttl_seconds
        self._cache: tuple[float, dict] | None = None
        self._lock = threading.Lock()

    def _maybe_refresh_model(self) -> None:
        if not storage.enabled() or time.time() - self._pulled_at < self.REFRESH_SECONDS:
            return
        self._pulled_at = time.time()
        try:
            storage.pull()
            model = ProductionModel()
            if model.meta["trained_at"] != self.model.meta["trained_at"]:
                log.info("loaded retrained model (data until %s)", model.meta["trained_until"])
                self.model, self._cache = model, None
        except Exception:  # keep serving the current model if S3 is unavailable
            log.exception("model refresh failed; keeping current model")

    def latest(self) -> dict:
        with self._lock:
            self._maybe_refresh_model()
            if self._cache and time.time() - self._cache[0] < self.ttl:
                return self._cache[1]
            demand = recent_demand()
            origin = demand.dropna().index[-1] + pd.Timedelta("1h")
            past_days = min(92, max(7, (pd.Timestamp.now(tz="UTC") - origin).days + 2))
            fc = self.model.forecast(demand, live_weather(self.cfg, past_days=past_days), self.cfg)
            result = {
                "model": "xgboost",
                "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "origin": origin.isoformat(),
                "model_trained_until": self.model.meta["trained_until"],
                "unit": "MW",
                "points": [{"time": t.isoformat(), "horizon_h": int(r.horizon_h),
                            "forecast_mw": round(float(r.forecast_mw), 1)} for t, r in fc.iterrows()],
            }
            self._cache = (time.time(), result)
            return result

    def actuals(self, days: int) -> list[dict]:
        d = recent_demand().dropna()
        d = d[d.index > d.index[-1] - pd.Timedelta(days=days)]
        return [{"time": t.isoformat(), "demand_mw": round(float(v), 1)} for t, v in d.items()]
