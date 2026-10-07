"""Temporal Fusion Transformer (TFT) via Darts, with 10th/50th/90th percentile forecasts.

TFT is a neural network built for multi-horizon forecasting. It reads a look-back window
of past demand (the "encoder"), plus known-in-advance inputs (calendar, weather
forecasts) for both the look-back and the forecast window. It outputs all 168 future
hours at once. Instead of one number per hour, it predicts three quantiles: the 10th and
90th percentiles bracket the range demand should fall in 80% of the time.

Leakage-safe forecast weather: for horizon h we feed the horizon's legal weather series
W_h for the whole window (see features/weather.py). TFT's decoder is causal: an LSTM
running forward plus masked attention. So the step-h output only depends on inputs at
steps <= h, and those are all legally available from W_h. `tests/test_tft.py` checks this
by changing later steps and confirming the step-h forecast doesn't move. It also means
padding the covariates past the last target (Darts wants a full 168-step window) can't
affect anything.

Laptop budget (no GPU): small network, 1-week look-back, a capped number of training
windows (the most recent ones), and refits every `refit_every_folds` quarters.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from demand_forecast.config import load_config

for name in ("pytorch_lightning", "lightning", "lightning.pytorch", "darts"):
    logging.getLogger(name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", ".*does not have many workers.*")

COVARIATES = ["hour_sin", "hour_cos", "dow", "is_weekend", "is_holiday_ew", "is_holiday_sct",
              "is_christmas_period", "doy_sin", "doy_cos",
              "temperature_2m", "temp_mean_24h", "wind_speed_10m", "shortwave_radiation"]
QUANTILES = [0.1, 0.5, 0.9]


def _covariate_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return out[COVARIATES].astype("float32")


def _naive_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_convert("UTC").tz_localize(None)


class TFTForecaster:
    name = "tft"
    horizon_independent = True
    refit_every_folds = load_config()["models"]["tft"]["refit_every_folds"]

    def __init__(self, params: dict | None = None):
        cfg = load_config()
        self.p = {**cfg["models"]["tft"], **(params or {})}
        self.seed = cfg["seed"]

    def _ts(self, values: pd.DataFrame | pd.Series):
        from darts import TimeSeries
        df = values.to_frame() if isinstance(values, pd.Series) else values
        df = df.set_axis(_naive_utc(df.index))
        return TimeSeries.from_dataframe(df.astype("float32"), freq="h")

    def fit(self, train: pd.DataFrame, h: int) -> None:
        import torch
        from darts.models import TFTModel
        from darts.utils.likelihood_models import QuantileRegression

        torch.set_num_threads(self.p["threads"])
        cov = _covariate_frame(train)
        train = train[cov.notna().all(axis=1) & train["y"].notna()]
        first = train.index[0]
        train = train.loc[first:]  # one contiguous block
        if len(train) != (train.index[-1] - first) / pd.Timedelta("1h") + 1:
            raise ValueError("TFT needs a gap-free hourly training series")
        cov = _covariate_frame(train)

        # Standardise with TRAINING statistics only.
        self.y_mean, self.y_std = float(train["y"].mean()), float(train["y"].std())
        self.c_mean, self.c_std = cov.mean(), cov.std().replace(0, 1)

        self.model = TFTModel(
            input_chunk_length=self.p["input_chunk_length"],
            output_chunk_length=self.p["output_chunk_length"],
            hidden_size=self.p["hidden_size"],
            lstm_layers=1,
            num_attention_heads=self.p["attention_heads"],
            hidden_continuous_size=self.p["hidden_continuous_size"],
            dropout=self.p["dropout"],
            batch_size=self.p["batch_size"],
            n_epochs=self.p["n_epochs"],
            optimizer_kwargs={"lr": self.p["learning_rate"]},
            likelihood=QuantileRegression(quantiles=QUANTILES),
            full_attention=False,  # causal attention: step s never looks at steps after s
            add_relative_index=False,
            random_state=self.seed,
            save_checkpoints=False,
            pl_trainer_kwargs={"accelerator": self.p.get("accelerator", "cpu"), "enable_progress_bar": False,
                               "enable_model_summary": False, "logger": False,
                               "enable_checkpointing": False,
                               "gradient_clip_val": 0.5},
        )
        self.model.fit(
            series=self._ts((train["y"] - self.y_mean) / self.y_std),
            future_covariates=self._ts((cov - self.c_mean) / self.c_std),
            max_samples_per_ts=self.p["max_training_windows"],  # most recent windows
            dataloader_kwargs={"num_workers": 0},
            verbose=False,
        )

    def predict(self, test: pd.DataFrame, h: int, context: pd.DataFrame | None = None) -> pd.DataFrame:
        if context is None:
            raise ValueError("TFT needs the context frame for its look-back window")
        icl, ocl = self.p["input_chunk_length"], self.p["output_chunk_length"]
        first_origin = test["origin"].min()
        last_target = test.index.max()
        hours = pd.Timedelta("1h")
        start = first_origin - icl * hours

        y = context["y"].loc[start:last_target]
        if y.isna().any():
            raise ValueError("gap in demand inside the TFT look-back/test window")
        # Covariates must reach (last origin + ocl - 1). Steps after the target can't affect
        # it (causal decoder), so padding them by repeating the last value is harmless.
        cov_index = pd.date_range(start, test["origin"].max() + (ocl - 1) * hours, freq="1h")
        cov = _covariate_frame(context.reindex(cov_index).ffill())
        cov = ((cov - self.c_mean) / self.c_std).astype("float32")

        fc = self.model.historical_forecasts(
            series=self._ts((y - self.y_mean) / self.y_std),
            future_covariates=self._ts(cov),
            start=first_origin.tz_convert("UTC").tz_localize(None),
            start_format="value",
            forecast_horizon=h,
            stride=1,
            retrain=False,
            last_points_only=True,
            predict_likelihood_parameters=True,
            verbose=False,
        )
        q = fc.to_dataframe() if hasattr(fc, "to_dataframe") else fc.pd_dataframe()
        q = q * self.y_std + self.y_mean
        q.index = q.index.tz_localize("UTC")
        qs = np.sort(q.to_numpy(), axis=1)  # guard against crossing quantiles
        out = pd.DataFrame(qs, index=q.index, columns=["q10", "y_pred", "q90"]).reindex(test.index)
        missing = out["y_pred"].isna().sum()
        if missing:
            raise RuntimeError(f"TFT produced no forecast for {missing} test targets (h={h})")
        return out
