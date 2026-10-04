"""Load configs/config.yaml and resolve paths relative to the project root."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def raw_dir(*parts: str) -> Path:
    path = PROJECT_ROOT / load_config()["paths"]["raw"] / Path(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def processed_dir() -> Path:
    path = PROJECT_ROOT / load_config()["paths"]["processed"]
    path.mkdir(parents=True, exist_ok=True)
    return path
