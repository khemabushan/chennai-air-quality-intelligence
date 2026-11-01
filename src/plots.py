"""Static evaluation figures for the report/README (the dashboard uses interactive Plotly)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, STATIONS

sns.set_theme(style="whitegrid", context="notebook")
ORDER = ["persistence", "seasonal_naive", "random_forest", "xgboost", "lstm"]
COLORS = {"persistence": "#9aa5b1", "seasonal_naive": "#c3cad3", "xgboost": "#1f6f9f",
          "random_forest": "#5fa8d3", "lstm": "#e07a2f"}


def main(split: str = "test") -> None:
    ov = pd.read_csv(METRICS_DIR / f"overall_{split}.csv")
    ps = pd.read_csv(METRICS_DIR / f"per_station_{split}.csv")
    common = pd.read_csv(PREDICTIONS_DIR / f"common_{split}.csv.gz", parse_dates=["target_date"])
    order = [m for m in ORDER if m in set(ov.model)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, met in zip(axes, ["MAE", "RMSE", "R2"]):
        d = ov.set_index("model").loc[order]
        ax.bar(d.index, d[met], color=[COLORS[m] for m in order])
        ax.set_title(f"{met} ({split}, n={int(d.n.iloc[0])})")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.barplot(data=ps, x="station", y="MAE", hue="model", hue_order=order, palette=COLORS, ax=ax)
    ax.set_title(f"MAE per station ({split})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "station_mae.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(len(STATIONS), 1, figsize=(12, 15), sharex=True)
    for ax, st in zip(axes, STATIONS):
        g = common[common.station == st]
        a = g[g.model == "persistence"].sort_values("target_date")
        ax.plot(a.target_date, a.y_true, color="#2b2d42", lw=1.6, label="actual")
        for m in ["xgboost", "lstm"]:
            gm = g[g.model == m].sort_values("target_date")
            ax.plot(gm.target_date, gm.y_pred, color=COLORS[m], lw=1.1, label=m)
        ax.set_title(st, loc="left", fontsize=10)
    axes[0].legend(ncol=3, loc="upper right")
    fig.suptitle(f"Actual vs predicted next-day PM2.5 ({split})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(order), figsize=(4 * len(order), 3.6), sharey=True)
    for ax, m in zip(axes, order):
        g = common[common.model == m]
        ax.scatter(g.y_pred, g.y_true - g.y_pred, s=6, alpha=0.4, color=COLORS[m])
        ax.axhline(0, color="black", lw=0.8)
        ax.set(title=m, xlabel="predicted")
    axes[0].set_ylabel("residual (actual − predicted)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "residuals.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
