"""TFT leakage checks on a tiny model (a few seconds on CPU).

Our honest-weather scheme for TFT relies on its decoder being causal: the forecast for
step h must not depend on covariates at later steps. If that ever broke (e.g. someone
set full_attention=True), forecast weather that's too fresh for step h could leak in.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("darts")

from demand_forecast.features.build import build_features  # noqa: E402
from demand_forecast.models.tft import COVARIATES, TFTForecaster, _covariate_frame  # noqa: E402

TINY = {"input_chunk_length": 24, "output_chunk_length": 24, "hidden_size": 8,
        "attention_heads": 1, "hidden_continuous_size": 4, "n_epochs": 1,
        "max_training_windows": 200, "batch_size": 32, "threads": 2}


def _fit(synthetic_ds, cfg):
    frame = build_features(synthetic_ds.demand, synthetic_ds.weather_actual, 1, cfg)
    train = frame.iloc[: 24 * 25]
    m = TFTForecaster(TINY)
    m.fit(train, 1)
    return m, frame


def _step_forecast(m, frame, origin_pos, h, cov_override=None):
    from darts import TimeSeries
    y = frame["y"].iloc[origin_pos - 24: origin_pos]
    cov = _covariate_frame(frame.iloc[origin_pos - 24: origin_pos + 24])
    if cov_override is not None:
        cov = cov_override(cov)
    cov = (cov - m.c_mean) / m.c_std
    to_ts = lambda d: TimeSeries.from_dataframe(  # noqa: E731
        (d.to_frame() if isinstance(d, pd.Series) else d).set_axis(d.index.tz_localize(None)).astype("float32"), freq="h")
    pred = m.model.predict(n=24, series=to_ts((y - m.y_mean) / m.y_std), future_covariates=to_ts(cov),
                           predict_likelihood_parameters=True, verbose=False)
    vals = pred.values()
    return vals[h - 1]


def test_tft_step_h_ignores_covariates_after_step_h(synthetic_ds, cfg):
    m, frame = _fit(synthetic_ds, cfg)
    pos, h = 24 * 28, 6
    base = _step_forecast(m, frame, pos, h)

    def scramble_after_h(cov):  # change all decoder steps after h
        cov = cov.copy()
        cov.iloc[24 + h:, :] = cov.iloc[24 + h:, :] + 25.0
        return cov

    def scramble_at_h(cov):  # control: change step h itself -> must move the forecast
        cov = cov.copy()
        cov.iloc[24 + h - 1, COVARIATES.index("temperature_2m")] += 25.0
        return cov

    np.testing.assert_allclose(_step_forecast(m, frame, pos, h, scramble_after_h), base, rtol=1e-5, atol=1e-6)
    assert not np.allclose(_step_forecast(m, frame, pos, h, scramble_at_h), base)


def test_tft_backtest_predict_shapes(synthetic_ds, cfg):
    m, frame = _fit(synthetic_ds, cfg)
    test = frame.iloc[24 * 26: 24 * 30].dropna(subset=["y"])
    out = m.predict(test, 1, context=frame)
    assert list(out.columns) == ["q10", "y_pred", "q90"]
    assert out["y_pred"].notna().all() and (out["q10"] <= out["y_pred"]).all() and (out["y_pred"] <= out["q90"]).all()
