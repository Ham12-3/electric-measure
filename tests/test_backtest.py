import numpy as np
import pandas as pd
import pytest

from demand_forecast.evaluation.backtest import FeatureCache, make_folds, run_backtest, split_fold
from demand_forecast.evaluation.metrics import mae, mape, pinball, rmse
from demand_forecast.models.baseline import SameHourLastWeek


def test_folds_are_contiguous_and_ordered():
    end = pd.Timestamp("2026-09-13 22:00", tz="UTC")
    folds = make_folds("2024-04-01", end, 3)
    assert folds[0].cutoff == pd.Timestamp("2024-04-01", tz="UTC")
    for a, b in zip(folds, folds[1:]):
        assert a.end == b.cutoff and a.cutoff < a.end
    assert folds[-1].end > end


def test_first_fold_starts_at_test_start_even_mid_month():
    folds = make_folds("2025-01-22", pd.Timestamp("2025-03-10", tz="UTC"), 1)
    assert [f.cutoff.date().isoformat() for f in folds] == ["2025-01-22", "2025-02-01", "2025-03-01"]


@pytest.mark.parametrize("h", [1, 24, 168])
def test_train_is_strictly_before_test_origins(synthetic_ds, cfg, h):
    cfg = {**cfg, "backtest": {**cfg["backtest"], "test_start": "2025-01-22"}}
    end = synthetic_ds.demand.index[-1]
    fold = make_folds("2025-01-22", end, 1)[0]
    train, test = split_fold(FeatureCache(synthetic_ds, cfg), fold, h, "forecast",
                             pd.Timestamp("2025-01-01", tz="UTC"))
    assert train.index.max() < fold.cutoff <= test["origin"].min()
    assert train.index.is_monotonic_increasing and test.index.is_monotonic_increasing  # never shuffled
    # Every input a training row uses is from before the cutoff, so it was known in time.
    assert (train.index - pd.Timedelta(hours=h) < fold.cutoff).all()


def test_baseline_predicts_last_week(synthetic_ds, cfg):
    cfg = {**cfg, "backtest": {**cfg["backtest"], "test_start": "2025-01-22", "fold_months": 1,
                               "train_start": "2025-01-01"}}
    preds = run_backtest(SameHourLastWeek, synthetic_ds, cfg, "forecast", horizons=[24])
    d = synthetic_ds.demand
    expected = d.shift(168).reindex(pd.DatetimeIndex(preds["target"]))
    np.testing.assert_allclose(preds["y_pred"].to_numpy(), expected.to_numpy())


def test_metrics_known_values():
    y, p = np.array([100.0, 200.0]), np.array([110.0, 180.0])
    assert mae(y, p) == 15.0
    assert rmse(y, p) == pytest.approx(np.sqrt((100 + 400) / 2))
    assert mape(y, p) == pytest.approx(10.0)
    # Under-forecasting the 90th percentile costs 9x more than over-forecasting.
    assert pinball(np.array([100.0]), np.array([90.0]), 0.9) == pytest.approx(9.0)
    assert pinball(np.array([100.0]), np.array([110.0]), 0.9) == pytest.approx(1.0)
