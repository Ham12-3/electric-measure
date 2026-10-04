"""The interface every model implements so the backtester can treat them identically."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class Forecaster(Protocol):
    name: str

    def fit(self, train: pd.DataFrame, h: int) -> None:
        """Train on rows whose target is before the fold cutoff (built by features.build)."""

    def predict(self, test: pd.DataFrame, h: int) -> pd.DataFrame:
        """Return a frame indexed like `test` with `y_pred` (optionally `q10`, `q90`)."""
