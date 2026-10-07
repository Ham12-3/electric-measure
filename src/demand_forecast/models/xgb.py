"""XGBoost: gradient-boosted decision trees on the tabular features.

"Direct" strategy: a separate model per horizon, each trained only on the features legal
for that horizon. The 168h model never sees yesterday's demand, even during training,
so it learns to rely on calendar, weather and week-old demand.

Trees capture non-linear effects (the heating curve) and interactions (cold *and* 18:00
*and* a weekday) without us spelling them out. Their weakness is that they can't
extrapolate beyond values seen in training, e.g. a level of demand lower than ever before.
The lag features soften this by anchoring predictions to recent demand.

Number of trees: the last 8 weeks of each training set are held out (still before the
cutoff, so this is legal) to find where validation error stops improving. Then we refit
on all training data with that many trees, so the newest weeks still get learned.
"""

from __future__ import annotations

import pandas as pd
import xgboost as xgb

from demand_forecast.config import load_config
from demand_forecast.features.build import feature_columns


class XGBForecaster:
    name = "xgboost"

    def __init__(self, params: dict | None = None):
        cfg = load_config()
        self.params = {**cfg["models"]["xgboost"], **(params or {})}
        self.seed = cfg["seed"]
        self.model: xgb.XGBRegressor | None = None
        self.columns: list[str] = []

    def _regressor(self, n_estimators: int, early_stopping: int | None = None) -> xgb.XGBRegressor:
        p = {k: v for k, v in self.params.items() if k not in ("n_estimators", "early_stopping_rounds", "validation_weeks")}
        return xgb.XGBRegressor(n_estimators=n_estimators, early_stopping_rounds=early_stopping,
                                random_state=self.seed, tree_method="hist", **p)

    def fit(self, train: pd.DataFrame, h: int) -> None:
        self.columns = feature_columns(train)
        X, y = train[self.columns], train["y"]
        split = train.index.max() - pd.Timedelta(weeks=self.params["validation_weeks"])
        fit_mask = train.index <= split
        probe = self._regressor(self.params["n_estimators"], self.params["early_stopping_rounds"])
        probe.fit(X[fit_mask], y[fit_mask], eval_set=[(X[~fit_mask], y[~fit_mask])], verbose=False)
        self.best_n = probe.best_iteration + 1
        self.model = self._regressor(self.best_n)
        self.model.fit(X, y, verbose=False)
        gain = self.model.get_booster().get_score(importance_type="gain")
        self.importance = pd.Series(gain).reindex(self.columns).fillna(0.0)

    def predict(self, test: pd.DataFrame, h: int, context: pd.DataFrame | None = None) -> pd.DataFrame:
        return pd.DataFrame({"y_pred": self.model.predict(test[self.columns])}, index=test.index)
