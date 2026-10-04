"""Forecast error metrics.

* MAE  - average size of the error, in MW. Easy to read: "off by 900 MW on average".
* RMSE - like MAE but punishes big misses more. Matters for the grid, where one large
         miss is worse than several small ones.
* MAPE - error as a % of demand, so it's comparable across seasons and with other studies.
         Safe here because national demand never gets near zero.
* Pinball loss - scores a quantile forecast (e.g. "90% chance demand is below X").
         Being on the wrong side of the quantile costs more than being on the right side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mape(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs((y - p) / y)) * 100)


def pinball(y: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    diff = y - q_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def score(df: pd.DataFrame) -> pd.Series:
    """Metrics for a predictions frame with columns y, y_pred (and optional q10/q90)."""
    df = df.dropna(subset=["y", "y_pred"])
    y, p = df["y"].to_numpy(), df["y_pred"].to_numpy()
    out = {"n": len(df), "MAE": mae(y, p), "RMSE": rmse(y, p), "MAPE": mape(y, p)}
    for col, q in (("q10", 0.1), ("q90", 0.9)):
        if col in df and df[col].notna().any():
            out[f"pinball_{col}"] = pinball(y, df[col].to_numpy(), q)
    if {"q10", "q90"} <= set(df.columns) and df["q10"].notna().any():
        out["coverage_80"] = float(((y >= df["q10"]) & (y <= df["q90"])).mean() * 100)
    return pd.Series(out)


def score_table(preds: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return preds.groupby(by).apply(score, include_groups=False)
