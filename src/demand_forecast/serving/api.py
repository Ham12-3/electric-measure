"""FastAPI service: the next 7 days of hourly GB demand forecasts.

Run locally:
    uv run uvicorn demand_forecast.serving.api:app --reload
Endpoints:
    GET /health    model status and training date
    GET /forecast  next 168 hourly forecasts from the production XGBoost models
    GET /actuals   recent hourly demand (NESO + Elexon), ?days=14
    GET /metrics   backtest comparison of all models vs the baseline
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request

from demand_forecast.config import PROJECT_ROOT

log = logging.getLogger(__name__)


def _default_service():
    from demand_forecast.serving.live import ForecastService
    return ForecastService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "service", None) is None:
        try:
            app.state.service = _default_service()
        except FileNotFoundError as e:
            log.error("no production model found (%s); run demand_forecast.serving.production", e)
            app.state.service = None
    yield


app = FastAPI(title="GB electricity demand forecast", version="1.0", lifespan=lifespan)


def _service(request: Request):
    svc = getattr(request.app.state, "service", None)
    if svc is None:
        raise HTTPException(503, "model not loaded - run `python -m demand_forecast.serving.production`")
    return svc


@app.get("/health")
def health(request: Request) -> dict:
    svc = getattr(request.app.state, "service", None)
    if svc is None:
        return {"status": "no_model"}
    meta = svc.model.meta
    return {"status": "ok", "model": meta["model"], "trained_until": meta["trained_until"],
            "trained_at": meta["trained_at"], "model_horizons_h": sorted(int(h) for h in meta["horizons"])}


@app.get("/forecast")
def forecast(request: Request) -> dict:
    try:
        return _service(request).latest()
    except HTTPException:
        raise
    except Exception as e:  # upstream data source down, etc.
        log.exception("forecast failed")
        raise HTTPException(502, f"could not build forecast: {e}") from e


@app.get("/actuals")
def actuals(request: Request, days: int = Query(14, ge=1, le=90)) -> dict:
    return {"unit": "MW", "points": _service(request).actuals(days)}


@app.get("/metrics")
def metrics() -> dict:
    files = sorted((PROJECT_ROOT / "reports" / "metrics").glob("*__forecast.csv"))
    if not files:
        raise HTTPException(404, "no backtest metrics found")
    t = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cols = [c for c in ["model", "horizon", "MAE", "RMSE", "MAPE", "pinball_q10", "pinball_q90", "coverage_80"] if c in t]
    t = t[cols].round(3).astype(object).where(t[cols].notna(), None)  # NaN isn't valid JSON
    return {"weather": "forecast", "rows": t.to_dict(orient="records")}
