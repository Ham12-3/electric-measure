"""Phase 4: results table and charts from saved backtest predictions.

Reads artifacts/backtests/*.parquet and writes:
    reports/results.md                  results table (markdown, pasted into README)
    reports/figures/sample_week.png     forecast vs actual, one winter week, 24h-ahead
    reports/figures/error_by_horizon.png
    reports/figures/error_by_hour.png
    reports/figures/xgb_importance.png
    reports/figures/weather_cheating.png  actual vs forecast weather (XGBoost)

Usage:
    uv run python -m demand_forecast.evaluation.report
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from demand_forecast.config import PROJECT_ROOT  # noqa: E402
from demand_forecast.evaluation.metrics import score_table  # noqa: E402

BT = PROJECT_ROOT / "artifacts" / "backtests"
FIG = PROJECT_ROOT / "reports" / "figures"
MODELS = ["baseline", "prophet", "xgboost", "tft"]
LABEL = {"baseline": "Same hour last week", "prophet": "Prophet", "xgboost": "XGBoost", "tft": "TFT"}
# Reference palette categorical slots 1-3 for models; neutral grey for the baseline.
COLOR = {"baseline": "#8a8984", "prophet": "#2a78d6", "xgboost": "#eb6834", "tft": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlecolor": INK, "axes.titlesize": 12, "font.size": 10,
    "legend.frameon": False, "lines.linewidth": 2,
})


def load(weather: str = "forecast") -> pd.DataFrame:
    frames = [pd.read_parquet(BT / f"{m}__{weather}.parquet") for m in MODELS
              if (BT / f"{m}__{weather}.parquet").exists()]
    df = pd.concat(frames, ignore_index=True)
    df["target"] = pd.to_datetime(df["target"], utc=True)
    return df


def present(df: pd.DataFrame) -> list[str]:
    return [m for m in MODELS if m in set(df["model"])]


def results_table(df: pd.DataFrame) -> pd.DataFrame:
    t = score_table(df, ["horizon", "model"]).reset_index()
    base = t[t["model"] == "baseline"].set_index("horizon")["MAE"]
    t["Skill vs baseline"] = (1 - t["MAE"] / t["horizon"].map(base)) * 100
    t["order"] = t["model"].map(MODELS.index)
    return t.sort_values(["horizon", "order"]).drop(columns="order")


def to_markdown(t: pd.DataFrame) -> str:
    has_q = "pinball_q10" in t
    head = "| Horizon | Model | MAE (MW) | RMSE (MW) | MAPE | Skill vs baseline |"
    head += " Pinball q10 / q90 (MW) | 80% interval coverage |" if has_q else ""
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for _, r in t.iterrows():
        best = r["MAE"] == t.loc[t["horizon"] == r["horizon"], "MAE"].min()
        name = f"**{LABEL[r['model']]}**" if best else LABEL[r["model"]]
        row = (f"| {int(r['horizon'])}h | {name} | {r['MAE']:,.0f} | {r['RMSE']:,.0f} | "
               f"{r['MAPE']:.2f}% | {r['Skill vs baseline']:.0f}% |")
        if has_q:
            row += (f" {r['pinball_q10']:,.0f} / {r['pinball_q90']:,.0f} | {r['coverage_80']:.0f}% |"
                    if pd.notna(r.get("pinball_q10")) else " – | – |")
        lines.append(row)
    return "\n".join(lines)


def fig_sample_week(df: pd.DataFrame, start: str = "2026-01-12", h: int = 24) -> None:
    lo, hi = pd.Timestamp(start, tz="Europe/London"), pd.Timestamp(start, tz="Europe/London") + pd.Timedelta(days=7)
    d = df[(df["horizon"] == h) & (df["target"] >= lo) & (df["target"] < hi)]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    actual = d[d["model"] == present(d)[0]].set_index("target")["y"] / 1000
    t = d[d["model"] == "tft"].set_index("target")
    if not t.empty:
        ax.fill_between(t.index, t["q10"] / 1000, t["q90"] / 1000, color=COLOR["tft"], alpha=0.15,
                        lw=0, label="TFT 10–90% range")
    for m in present(d):
        s = d[d["model"] == m].set_index("target")["y_pred"] / 1000
        ax.plot(s.index, s, color=COLOR[m], lw=1.6, label=LABEL[m], ls="--" if m == "baseline" else "-")
    ax.plot(actual.index, actual, color=INK, lw=2.2, label="Actual")
    ax.set_ylabel("GW"); ax.set_title(f"{h}h-ahead forecasts vs actual, week of {lo:%d %b %Y}")
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.savefig(FIG / "sample_week.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_error_by_horizon(t: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    models = [m for m in MODELS if m in set(t["model"])]
    hs = sorted(t["horizon"].unique())
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        v = t[t["model"] == m].set_index("horizon").reindex(hs)["MAPE"]
        x = [j + (i - (len(models) - 1) / 2) * width for j in range(len(hs))]
        bars = ax.bar(x, v, width=width * 0.92, color=COLOR[m], label=LABEL[m])
        ax.bar_label(bars, fmt="%.1f", fontsize=8, color=MUTED, padding=2)
    ax.set_xticks(range(len(hs)), [f"{h}h" for h in hs]); ax.set_ylabel("MAPE (%)")
    ax.set_title("Error by forecast horizon (lower is better)"); ax.legend(ncol=4)
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG / "error_by_horizon.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_error_by_hour(df: pd.DataFrame) -> None:
    d = df.copy()
    d["hour"] = d["target"].dt.tz_convert("Europe/London").dt.hour
    d["ape"] = (d["y"] - d["y_pred"]).abs() / d["y"] * 100
    hs = sorted(d["horizon"].unique())
    fig, axes = plt.subplots(1, len(hs), figsize=(13, 3.8), sharey=True)
    for ax, h in zip(axes, hs):
        g = d[d["horizon"] == h].groupby(["model", "hour"])["ape"].mean()
        for m in present(d):
            ax.plot(g.loc[m].index, g.loc[m], color=COLOR[m], lw=1.8, label=LABEL[m],
                    ls="--" if m == "baseline" else "-")
        ax.set_title(f"{h}h ahead"); ax.set_xticks(range(0, 24, 6)); ax.set_xlabel("hour (UK time)")
    axes[0].set_ylabel("MAPE (%)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Error by hour of day", color=INK)
    fig.savefig(FIG / "error_by_hour.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_importance(top: int = 12) -> pd.DataFrame | None:
    path = BT / "xgboost__forecast__importance.parquet"
    if not path.exists():
        return None
    imp = pd.read_parquet(path)
    share = imp.groupby(["horizon", "feature"])["gain"].mean()
    share = share / share.groupby(level=0).transform("sum") * 100
    hs = sorted(imp["horizon"].unique())
    fig, axes = plt.subplots(1, len(hs), figsize=(13, 4.5))
    for ax, h in zip(axes, hs):
        s = share.loc[h].sort_values().tail(top)
        ax.barh(s.index, s.values, color=COLOR["xgboost"], height=0.7)
        for y, v in enumerate(s.values):
            ax.text(v + 0.5, y, f"{v:.0f}%", va="center", fontsize=8, color=MUTED)
        ax.set_title(f"{h}h model"); ax.set_xlabel("share of total gain (%)"); ax.grid(axis="y", visible=False)
    fig.suptitle("XGBoost feature importance (gain, averaged over folds)", color=INK)
    fig.tight_layout(); fig.savefig(FIG / "xgb_importance.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    return share.unstack(0)


def fig_weather_cheating() -> pd.DataFrame | None:
    paths = {w: BT / f"xgboost__{w}.parquet" for w in ("forecast", "actual")}
    if not all(p.exists() for p in paths.values()):
        return None
    rows = []
    for w, p in paths.items():
        s = score_table(pd.read_parquet(p), ["horizon"])["MAPE"]
        rows.append(s.rename(w))
    t = pd.concat(rows, axis=1)
    t["inflation_%"] = (1 - t["actual"] / t["forecast"]) * 100
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = range(len(t))
    for i, (w, c, lab) in enumerate([("forecast", COLOR["xgboost"], "forecast weather (honest)"),
                                     ("actual", "#8a8984", "actual weather (cheating)")]):
        bars = ax.bar([j + (i - 0.5) * 0.38 for j in x], t[w], width=0.35, color=c, label=lab)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, color=MUTED, padding=2)
    ax.set_xticks(list(x), [f"{h}h" for h in t.index]); ax.set_ylabel("MAPE (%)")
    ax.set_title("XGBoost: how much does knowing the real weather flatter the score?")
    ax.legend(); ax.grid(axis="x", visible=False)
    fig.savefig(FIG / "weather_cheating.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    return t


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = load("forecast")
    t = results_table(df)
    md = to_markdown(t)
    fig_sample_week(df)
    fig_error_by_horizon(t)
    fig_error_by_hour(df)
    imp = fig_importance()
    cheat = fig_weather_cheating()
    parts = ["## Results (forecast weather)\n", md]
    if cheat is not None:
        parts += ["\n\n## XGBoost: actual vs forecast weather (MAPE %)\n",
                  cheat.round(2).to_markdown()]
    if imp is not None:
        parts += ["\n\n## XGBoost top features (% of gain)\n",
                  imp.round(1).sort_values(imp.columns[0], ascending=False).head(12).to_markdown()]
    (PROJECT_ROOT / "reports" / "results.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(md)
    if cheat is not None:
        print(cheat.round(2))


if __name__ == "__main__":
    main()
