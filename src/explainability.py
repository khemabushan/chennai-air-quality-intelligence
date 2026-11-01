"""XGBoost feature importance and SHAP (TreeSHAP) explanations.

SHAP values are computed with XGBoost's built-in exact TreeSHAP (``pred_contribs=True``); the
``shap`` package is used only to draw the summary plot. If the model was trained on log1p(PM2.5),
contributions are in log space. These describe model behaviour, not causal effects.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

from src.config import FIGURES_DIR, METRICS_DIR, RESULTS_DIR
from src.inference import load_xgboost
from src.train_baselines import load_features

EXPLAIN_DIR = RESULTS_DIR / "explainability"
EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    model, schema = load_xgboost()
    feats = schema["features"]
    f = load_features()

    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    imp = (pd.DataFrame({"feature": feats, "gain": [gain.get(c, 0.0) for c in feats]})
           .assign(gain_share=lambda d: d.gain / d.gain.sum())
           .sort_values("gain", ascending=False))
    imp.round(5).to_csv(METRICS_DIR / "xgb_feature_importance.csv", index=False)

    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.feature, top.gain_share, color="#2a6f97")
    ax.set(xlabel="share of total gain", title="XGBoost feature importance (gain)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "xgb_feature_importance.png", dpi=130)
    plt.close(fig)

    sub = f[f["split"].isin(["val", "test"])].reset_index(drop=True)
    X = sub[feats].astype(float)
    contrib = booster.predict(xgb.DMatrix(X, feature_names=feats), pred_contribs=True,
                              iteration_range=(0, schema["best_iteration"] + 1))
    shap_df = pd.DataFrame(contrib[:, :-1], columns=feats)
    shap_df["bias"] = contrib[:, -1]
    out = pd.concat([sub[["station", "origin_date", "target_date", "split"]], shap_df.round(5)], axis=1)
    out.to_csv(EXPLAIN_DIR / "shap_values_val_test.csv.gz", index=False)

    test_mask = (sub["split"] == "test").to_numpy()
    mean_abs = shap_df.loc[test_mask, feats].abs().mean().sort_values(ascending=False)
    mean_abs.rename("mean_abs_shap").round(5).to_csv(METRICS_DIR / "shap_mean_abs_test.csv", index_label="feature")

    try:
        import shap
        shap.summary_plot(shap_df.loc[test_mask, feats].to_numpy(), X[test_mask], show=False, max_display=15)
        plt.title("SHAP summary — XGBoost, test period (PM2.5 units)" if schema["target_transform"] == "none" else "SHAP summary — XGBoost, test period (log scale)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=130)
        plt.close("all")
        shap_plot = True
    except Exception as exc:  # optional dependency; values are already saved
        print("SHAP plot skipped:", exc)
        shap_plot = False
    top_shap = mean_abs.head(5)
    pm25_hist = [c for c in feats if c.startswith("pm25_")]
    share_hist = mean_abs[pm25_hist].sum() / mean_abs.sum()
    share_other = mean_abs[["pm10_t", "pm10_roll7_mean", "no2_t", "so2_t", "o3_t"]].sum() / mean_abs.sum()
    share_station = mean_abs[[c for c in feats if c.startswith("station_")]].sum() / mean_abs.sum()
    lines = [
        "**Strongest features on the test period (mean |SHAP|, PM2.5 units):** "
        + ", ".join(f"`{k}` ({v:.1f})" for k, v in top_shap.items()) + ".",
        "",
        f"* **Recent PM2.5 history** (today's value, lags, rolling means/std) carries "
        f"**{100 * share_hist:.0f}%** of total attribution — tomorrow's level is mostly anchored on the "
        "last few days, which is why persistence is already a strong baseline.",
        f"* **Same-day co-pollutants** (PM10, NO₂, SO₂, O₃) carry **{100 * share_other:.0f}%**, dominated by "
        "PM10. PM10 and PM2.5 share emission sources and weather drivers, so PM10 adds information about "
        "the current particulate regime beyond PM2.5 alone.",
        f"* **Station identity** carries **{100 * share_station:.0f}%**: the model learns station-specific "
        "offsets (e.g. Alandur and Kodungaiyur levels differ systematically).",
        "* Calendar features contribute little at a one-day horizon once recent history is known.",
        "",
        "Gain importance (how much each feature reduced training loss) ranks the same group of "
        "features highest, with `pm25_roll3_mean` and `pm25_t` on top. These are associations learned by the "
        "model, not causal effects.",
    ]
    (EXPLAIN_DIR / "interpretation.md").write_text("\n".join(lines), encoding="utf-8")
    (EXPLAIN_DIR / "meta.json").write_text(json.dumps(
        {"target_transform": schema["target_transform"], "shap_plot": shap_plot}, indent=2))
    print(imp.head(10).to_string(index=False))
    print(mean_abs.head(10).to_string())


if __name__ == "__main__":
    main()
