"""Metrics and model comparison on a common set of test targets."""
import json

import numpy as np
import pandas as pd

from src.config import METRICS_DIR, PREDICTIONS_DIR

KEY = ["station", "origin_date", "target_date"]


def smape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    denom = (np.abs(y) + np.abs(p)) / 2
    return float(100 * np.mean(np.where(denom == 0, 0, np.abs(y - p) / denom)))


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = y - p
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(len(y)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "sMAPE": smape(y, p),
    }


def save_predictions(df: pd.DataFrame, model: str) -> None:
    """df needs station, origin_date, target_date, split, y_true, y_pred."""
    out = df[KEY + ["split", "y_true", "y_pred"]].copy()
    out.insert(0, "model", model)
    out.to_csv(PREDICTIONS_DIR / f"{model}.csv", index=False)


def load_all_predictions() -> pd.DataFrame:
    frames = [pd.read_csv(p, parse_dates=["origin_date", "target_date"])
              for p in sorted(PREDICTIONS_DIR.glob("*.csv")) if p.stem != "all_predictions"]
    return pd.concat(frames, ignore_index=True)


def common_rows(preds: pd.DataFrame, split: str) -> pd.DataFrame:
    """Restrict to (station, origin) rows predicted by every model in this split."""
    p = preds[preds["split"] == split]
    n_models = p["model"].nunique()
    counts = p.groupby(KEY)["model"].nunique()
    keep = counts[counts == n_models].index
    return p.set_index(KEY).loc[keep].reset_index()


def comparison_tables(preds: pd.DataFrame, split: str):
    common = common_rows(preds, split)
    overall = pd.DataFrame([{"model": m, **metrics(g.y_true, g.y_pred)} for m, g in common.groupby("model")])
    pers = overall.set_index("model").loc["persistence"]
    overall["MAE_improvement_vs_persistence_%"] = 100 * (1 - overall["MAE"] / pers["MAE"])
    overall["RMSE_improvement_vs_persistence_%"] = 100 * (1 - overall["RMSE"] / pers["RMSE"])
    per_station = pd.DataFrame([{"model": m, "station": s, **metrics(g.y_true, g.y_pred)}
                                for (m, s), g in common.groupby(["model", "station"])])
    coverage = (preds[preds["split"] == split].groupby("model").size().rename("own_rows").reset_index())
    return overall.sort_values("MAE"), per_station, coverage, common


def run(split: str = "test") -> dict:
    preds = load_all_predictions()
    out = {}
    for sp in ["val", split]:
        overall, per_station, coverage, common = comparison_tables(preds, sp)
        overall.round(4).to_csv(METRICS_DIR / f"overall_{sp}.csv", index=False)
        per_station.round(4).to_csv(METRICS_DIR / f"per_station_{sp}.csv", index=False)
        coverage.to_csv(METRICS_DIR / f"coverage_{sp}.csv", index=False)
        common.to_csv(PREDICTIONS_DIR / f"common_{sp}.csv.gz", index=False)
        out[sp] = overall
    return out


if __name__ == "__main__":
    res = run()
    for sp, t in res.items():
        print(f"\n=== {sp} (common rows) ===")
        print(t.round(3).to_string(index=False))
    (METRICS_DIR / "summary.json").write_text(json.dumps(
        {sp: t.round(4).to_dict(orient="records") for sp, t in res.items()}, indent=2))
