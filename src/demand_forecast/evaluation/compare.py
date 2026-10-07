"""Combine per-model metric files (reports/metrics/*.csv) into one comparison table.

"skill vs baseline" = how much lower a model's MAE is than same-hour-last-week's,
in %. 0% = no better than the naive guess; 50% = errors half the size.

Usage:
    uv run python -m demand_forecast.evaluation.compare [--weather forecast]
"""

from __future__ import annotations

import argparse

import pandas as pd

from demand_forecast.config import PROJECT_ROOT

MODEL_ORDER = ["baseline", "prophet", "xgboost", "tft"]


def comparison_table(weather: str = "forecast") -> pd.DataFrame:
    files = sorted((PROJECT_ROOT / "reports" / "metrics").glob(f"*__{weather}.csv"))
    table = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    base = table[table["model"] == "baseline"].set_index("horizon")["MAE"]
    table["skill_vs_baseline_%"] = (1 - table["MAE"] / table["horizon"].map(base)) * 100
    table["order"] = table["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return table.sort_values(["horizon", "order"]).drop(columns="order").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather", default="forecast")
    args = parser.parse_args()
    t = comparison_table(args.weather)
    cols = [c for c in ["horizon", "model", "MAE", "RMSE", "MAPE", "skill_vs_baseline_%",
                        "pinball_q10", "pinball_q90", "coverage_80"] if c in t]
    with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", 140):
        print(t[cols].to_string(index=False))


if __name__ == "__main__":
    main()
