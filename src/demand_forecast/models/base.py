"""The interface every model implements so the backtester can treat them identically."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class Forecaster(Protocol):
    """Optional class attributes the backtester understands:

    horizon_independent: bool - one fitted model serves every horizon in a fold
        (Prophet: no lags; TFT: forecasts all 168 steps at once). Default False.
    refit_every_folds: int - retrain only every k-th fold and reuse the previous fold's
        (older, so still legal) model in between. Default 1. Used to keep TFT laptop-sized.
    """

    name: str

    def fit(self, train: pd.DataFrame, h: int) -> None:
        """Train on rows whose target is before the fold cutoff (built by features.build)."""

    def predict(self, test: pd.DataFrame, h: int, context: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return a frame indexed like `test` with `y_pred` (optionally `q10`, `q90`).

        `context` is the full feature table for this horizon and weather mode. Sequence
        models use it for the look-back window before each origin. They must only read
        demand up to each forecast's origin (Darts' historical_forecasts guarantees this).
        """
