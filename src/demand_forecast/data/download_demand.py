"""Download NESO historic demand CSVs (one file per year) into data/raw/neso/.

NESO publishes on a CKAN portal. We ask the CKAN API for the dataset's file list at
runtime instead of hardcoding links, because the current-year file
("demanddataupdate_YYYY.csv") is replaced as new data arrives.

Usage:
    uv run python -m demand_forecast.data.download_demand [--force]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
from pathlib import Path

import requests

from demand_forecast.config import load_config, raw_dir

log = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"demanddata(?:update)?_(\d{4})\.csv$", re.IGNORECASE)


def list_year_files(package_url: str) -> dict[int, str]:
    """Return {year: download_url} for every yearly demand CSV on the portal."""
    resp = requests.get(package_url, timeout=60)
    resp.raise_for_status()
    files: dict[int, str] = {}
    for res in resp.json()["result"]["resources"]:
        match = _YEAR_RE.search(res.get("url", ""))
        if match:
            files[int(match.group(1))] = res["url"]
    return files


def download_file(url: str, dest: Path) -> None:
    # Download links redirect to a signed storage URL, so redirects must be followed.
    resp = requests.get(url, timeout=300, allow_redirects=True)
    resp.raise_for_status()
    if b"SETTLEMENT_DATE" not in resp.content[:200]:  # header may or may not be quoted
        raise ValueError(f"{url} did not return a demand CSV (got {resp.content[:80]!r})")
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(resp.content)
    tmp.replace(dest)  # atomic: never leave a half-written CSV in the cache


def download_demand(force: bool = False) -> list[Path]:
    cfg = load_config()
    start_year = dt.date.fromisoformat(cfg["dates"]["start"]).year
    current_year = dt.date.today().year
    out_dir = raw_dir("neso")

    files = list_year_files(cfg["demand"]["ckan_package_url"])
    paths = []
    for year in sorted(y for y in files if y >= start_year):
        dest = out_dir / f"demanddata_{year}.csv"
        # The current year's file keeps growing, so always refresh it.
        if dest.exists() and not force and year < current_year:
            log.info("cached  %s", dest.name)
        else:
            log.info("fetch   %s", dest.name)
            download_file(files[year], dest)
        paths.append(dest)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download cached files")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = download_demand(force=args.force)
    log.info("%d demand files in %s", len(paths), paths[0].parent if paths else "-")


if __name__ == "__main__":
    main()
