"""Streamlit dashboard: recent demand, the latest 7-day forecast, and model comparison.

Run locally (with the API running):
    uv run streamlit run dashboard/app.py
Set API_URL to point at the API (default http://localhost:8000).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
FIGURES = Path(__file__).resolve().parents[1] / "reports" / "figures"

st.set_page_config(page_title="GB demand forecast", layout="wide")
st.title("GB electricity demand forecast")
st.caption("National Demand (NESO ND), hourly. Forecast: XGBoost, the best model in a "
           "walk-forward backtest using only information available at forecast time.")


@st.cache_data(ttl=900)
def get(path: str, **params) -> dict:
    r = requests.get(f"{API_URL}{path}", params=params, timeout=120)
    r.raise_for_status()
    return r.json()


try:
    fc = get("/forecast")
    act = get("/actuals", days=14)
except requests.RequestException as e:
    st.error(f"Can't reach the forecast API at {API_URL}: {e}")
    st.stop()

forecast = pd.DataFrame(fc["points"]).assign(time=lambda d: pd.to_datetime(d["time"]))
actuals = pd.DataFrame(act["points"]).assign(time=lambda d: pd.to_datetime(d["time"]))

peak = forecast.loc[forecast["forecast_mw"].idxmax()]
c1, c2, c3 = st.columns(3)
c1.metric("Next-7-day peak", f"{peak.forecast_mw / 1000:.1f} GW",
          help=f"at {peak.time.tz_convert('Europe/London'):%a %d %b %H:%M} UK time")
c2.metric("Latest actual", f"{actuals['demand_mw'].iloc[-1] / 1000:.1f} GW",
          help=f"hour starting {actuals['time'].iloc[-1].tz_convert('Europe/London'):%a %d %b %H:%M} UK time")
c3.metric("Forecast made from", pd.Timestamp(fc["origin"]).tz_convert("Europe/London").strftime("%d %b %H:%M"),
          help=f"model trained on data up to {fc['model_trained_until'][:10]}")

st.subheader("Last 14 days and next 7 days")
chart = pd.concat([
    actuals.set_index("time")["demand_mw"].rename("Actual"),
    forecast.set_index("time")["forecast_mw"].rename("Forecast"),
], axis=1) / 1000
chart.index = chart.index.tz_convert("Europe/London").tz_localize(None)
st.line_chart(chart, y_label="GW", color=["#0b0b0b", "#eb6834"])

st.subheader("Model comparison (backtest, Apr 2024 – Sep 2026)")
try:
    m = pd.DataFrame(get("/metrics")["rows"])
    table = m.pivot_table(index="model", columns="horizon", values="MAPE").reindex(
        ["baseline", "prophet", "xgboost", "tft"]).dropna(how="all")
    table.columns = [f"MAPE {h}h (%)" for h in table.columns]
    st.dataframe(table.style.format("{:.2f}").highlight_min(axis=0, color="#fde3d6"), use_container_width=True)
except requests.RequestException:
    st.info("Backtest metrics unavailable.")

cols = st.columns(2)
for col, name, caption in [
    (cols[0], "error_by_horizon.png", "Error by horizon"),
    (cols[1], "error_by_hour.png", "Error by hour of day"),
    (cols[0], "sample_week.png", "24h-ahead forecasts vs actual, sample week"),
    (cols[1], "weather_cheating.png", "Forecast vs actual weather"),
]:
    if (FIGURES / name).exists():
        col.image(str(FIGURES / name), caption=caption, use_container_width=True)

with st.expander("Forecast data"):
    st.dataframe(forecast, use_container_width=True)
