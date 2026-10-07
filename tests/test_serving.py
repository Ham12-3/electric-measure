import pandas as pd
import pytest
from fastapi.testclient import TestClient

from demand_forecast.serving.api import app
from demand_forecast.serving.production import ProductionModel, model_horizon_for, train_production

TINY_XGB = {"n_estimators": 50, "early_stopping_rounds": 10, "validation_weeks": 1, "n_jobs": 1}


@pytest.mark.parametrize("h,expected", [(1, 1), (2, 6), (6, 6), (7, 24), (25, 72), (168, 168)])
def test_model_horizon_for_never_uses_a_shorter_horizon(h, expected):
    assert model_horizon_for(h, [1, 6, 24, 72, 168]) == expected


def test_model_horizon_for_rejects_uncovered_horizon():
    with pytest.raises(ValueError):
        model_horizon_for(200, [1, 24, 168])


def test_production_forecast_covers_168_hours_and_ignores_future(synthetic_ds, cfg, tmp_path):
    cfg = {**cfg, "serving": {"model_horizons": [1, 24, 168]}, "backtest": {**cfg["backtest"], "train_start": "2025-01-01"}}
    train_production(synthetic_ds.demand, synthetic_ds.weather_actual, cfg, tmp_path, TINY_XGB)
    model = ProductionModel(tmp_path)
    history = synthetic_ds.demand.iloc[: 24 * 28]
    weather = synthetic_ds.weather_actual  # covers the 7 days after the history
    fc = model.forecast(history, weather, cfg)
    assert len(fc) == 168 and fc["horizon_h"].tolist() == list(range(1, 169))
    assert fc.index[0] == history.index[-1] + pd.Timedelta("1h")
    assert fc["forecast_mw"].notna().all()
    # Demand after the origin must not matter (here: the forecast never receives it).
    longer = synthetic_ds.demand.copy()
    longer.iloc[24 * 28:] = float("nan")
    pd.testing.assert_frame_equal(fc, model.forecast(longer, weather, cfg))


class FakeService:
    model = type("M", (), {"meta": {"model": "xgboost", "trained_until": "2026-09-13T22:00:00+00:00",
                                    "trained_at": "2026-10-07T00:00:00+00:00",
                                    "horizons": {"1": {}, "24": {}, "168": {}}}})()

    def latest(self):
        start = pd.Timestamp("2026-10-07 00:00", tz="UTC")
        return {"model": "xgboost", "origin": start.isoformat(), "unit": "MW",
                "generated_at": start.isoformat(), "model_trained_until": "2026-09-13",
                "points": [{"time": (start + pd.Timedelta(hours=i)).isoformat(), "horizon_h": i + 1,
                            "forecast_mw": 25000.0} for i in range(168)]}

    def actuals(self, days):
        return [{"time": "2026-10-06T23:00:00+00:00", "demand_mw": 24000.0}]


@pytest.fixture
def client():
    app.state.service = FakeService()
    with TestClient(app) as c:
        yield c
    app.state.service = None


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_horizons_h"] == [1, 24, 168]


def test_forecast_returns_168_hourly_points(client):
    r = client.get("/forecast")
    assert r.status_code == 200
    pts = r.json()["points"]
    assert len(pts) == 168 and pts[0]["horizon_h"] == 1 and pts[-1]["horizon_h"] == 168
    times = pd.to_datetime([p["time"] for p in pts])
    assert (times[1:] - times[:-1] == pd.Timedelta("1h")).all()


def test_actuals_validates_days(client):
    assert client.get("/actuals?days=7").status_code == 200
    assert client.get("/actuals?days=0").status_code == 422


def test_metrics_endpoint_is_valid_json(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert {row["model"] for row in rows} >= {"baseline", "xgboost"}


def test_forecast_503_without_model():
    app.state.service = None
    with TestClient(app) as c:
        app.state.service = None  # lifespan may have tried to load; force "no model"
        assert c.get("/forecast").status_code == 503
