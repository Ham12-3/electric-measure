"""Daily job: refresh data, rebuild clean files, retrain the production models.

Designed to be run once a day by a scheduler (cron, or EventBridge -> ECS on AWS).
Downloads are incremental: only the current year / latest weather chunk is refetched.

Usage:
    uv run python -m demand_forecast.serving.pipeline
"""

from __future__ import annotations

import logging

from demand_forecast.config import load_config
from demand_forecast.data import clean
from demand_forecast.data.dataset import load_dataset
from demand_forecast.data.download_demand import download_demand
from demand_forecast.data.download_weather import download_source
from demand_forecast.serving import storage
from demand_forecast.serving.production import train_production

log = logging.getLogger(__name__)


def run() -> dict:
    storage.pull(storage.PIPELINE_DIRS)  # no-op unless ARTIFACT_BUCKET is set
    download_demand()
    for source in ("archive", "previous_runs", "historical_forecast"):
        download_source(source)
    clean.main()
    ds = load_dataset()
    meta = train_production(ds.demand, ds.weather_actual, load_config())
    storage.push(storage.PIPELINE_DIRS)
    return meta


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    meta = run()
    log.info("retrained on data up to %s", meta["trained_until"])


if __name__ == "__main__":
    main()
