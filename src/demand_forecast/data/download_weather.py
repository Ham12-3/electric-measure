"""Download hourly weather for each city from three Open-Meteo APIs into data/raw/weather/.

Why three sources? A forecast must only use weather that was *known when it was made*.

* archive             - reanalysis ("what actually happened"). Used for training, and to
                        show how much scores improve if you cheat with real future weather.
* previous_runs       - real forecasts at fixed lead times: `<var>_previous_day1` is the
                        value predicted ~24h before the valid time, `_previous_day7` ~168h
                        before. This is what makes 24h and 7-day backtests honest.
                        Full coverage starts 2024-03-13.
* historical_forecast - first few hours of each model run stitched together, i.e. a
                        0-6h-ahead forecast. More accurate than even a day-1 forecast, so
                        it's only honest for very short horizons. Kept for comparison.

Both forecast sources use the same model (config: weather.forecast_model) so their
differences reflect lead time, not model choice.

Requests are cached as one JSON file per (source, city, year). Open-Meteo's free tier
counts a request as max(1, n_vars/10) * max(1, n_days/14) "calls" with a 600/min limit,
so we pace requests by that weight.

Usage:
    uv run python -m demand_forecast.data.download_weather [--force] [--source archive]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import time
from pathlib import Path

import requests

from demand_forecast.config import load_config, raw_dir

log = logging.getLogger(__name__)

SOURCES = ("archive", "previous_runs", "historical_forecast")
CALLS_PER_MINUTE_BUDGET = 400  # stay well below the 600/min free-tier limit


def request_weight(n_vars: int, n_days: int) -> float:
    return max(1.0, n_vars / 10) * max(1.0, n_days / 14)


def source_settings(source: str, cfg: dict) -> tuple[str, dt.date, dt.date, list[str], dict]:
    """Return (url, first_date, last_date, hourly_variables, extra_params) for a source."""
    w = cfg["weather"]
    today = dt.date.today()
    variables = w["variables"]
    if source == "archive":
        # ERA5 lags real time by ~5 days; stop a week back so chunks are complete.
        return (w["archive_url"], dt.date.fromisoformat(cfg["dates"]["start"]),
                today - dt.timedelta(days=7), variables, {})
    if source == "previous_runs":
        lead_vars = [f"{v}_previous_day{d}" for v in variables for d in w["lead_days"]]
        return (w["previous_runs_url"], dt.date.fromisoformat(cfg["dates"]["forecast_weather_start"]),
                today - dt.timedelta(days=1), lead_vars, {"models": w["forecast_model"]})
    if source == "historical_forecast":
        return (w["historical_forecast_url"], dt.date.fromisoformat(cfg["dates"]["forecast_weather_start"]),
                today - dt.timedelta(days=1), variables, {"models": w["forecast_model"]})
    raise ValueError(f"unknown source {source!r}")


def fetch(url: str, params: dict, retries: int = 5) -> dict:
    for attempt in range(retries):
        resp = requests.get(url, params=params, timeout=300)
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = 65 * (attempt + 1)
            log.warning("HTTP %s (%s) - waiting %ss", resp.status_code, resp.text[:120], wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise RuntimeError(f"Open-Meteo error: {data.get('reason')}")
        return data
    raise RuntimeError(f"giving up on {url} after {retries} attempts")


def download_source(source: str, force: bool = False) -> list[Path]:
    cfg = load_config()
    url, first, last, variables, extra = source_settings(source, cfg)
    out_dir = raw_dir("weather", source)
    paths = []
    for city, c in cfg["weather"]["cities"].items():
        for year in range(first.year, last.year + 1):
            start = max(first, dt.date(year, 1, 1))
            end = min(last, dt.date(year, 12, 31))
            dest = out_dir / f"{city}_{year}.json"
            is_open_year = end < dt.date(year, 12, 31)  # still growing -> refresh
            if dest.exists() and not force and not is_open_year:
                paths.append(dest)
                continue
            params = {
                "latitude": c["lat"], "longitude": c["lon"],
                "start_date": start.isoformat(), "end_date": end.isoformat(),
                "hourly": ",".join(variables), "timezone": "UTC", **extra,
            }
            weight = request_weight(len(variables), (end - start).days + 1)
            log.info("fetch   %-20s %-11s %s..%s (~%.0f calls)", source, city, start, end, weight)
            data = fetch(url, params)
            tmp = dest.with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(dest)
            paths.append(dest)
            time.sleep(60 * weight / CALLS_PER_MINUTE_BUDGET)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download cached files")
    parser.add_argument("--source", choices=SOURCES, action="append",
                        help="limit to one or more sources (default: all)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for source in args.source or SOURCES:
        paths = download_source(source, force=args.force)
        log.info("%s: %d files", source, len(paths))


if __name__ == "__main__":
    main()
