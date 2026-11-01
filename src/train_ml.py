"""XGBoost (main ML model) and Random Forest (secondary) with small validation-based tuning.

Models are fit on the training split only. Validation is used for early stopping and
configuration selection. The test split is only predicted, never used for any choice.
"""
import itertools
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from src.config import METRICS_DIR, MODELS_DIR, SEED
from src.evaluate import metrics, save_predictions
from src.feature_engineering import FEATURES, TARGET_COL
from src.train_baselines import load_features

TRANSFORMS = {
    "none": (lambda y: y, lambda y: y),
    "log1p": (np.log1p, np.expm1),
}


def split_xy(f: pd.DataFrame, split: str):
    d = f[f["split"] == split]
    return d, d[FEATURES].astype(float), d[TARGET_COL].to_numpy(float)


def tune_xgboost(f: pd.DataFrame):
    _, Xtr, ytr = split_xy(f, "train")
    _, Xva, yva = split_xy(f, "val")
    grid = itertools.product([3, 4, 6], [0.03, 0.08], [1, 5], list(TRANSFORMS))
    log = []
    best = None
    for depth, lr, mcw, tname in grid:
        fwd, inv = TRANSFORMS[tname]
        params = dict(n_estimators=2000, max_depth=depth, learning_rate=lr, min_child_weight=mcw,
                      subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="reg:squarederror",
                      early_stopping_rounds=100, random_state=SEED, n_jobs=4, tree_method="hist")
        m = xgb.XGBRegressor(**params)
        m.fit(Xtr, fwd(ytr), eval_set=[(Xva, fwd(yva))], verbose=False)
        pred = inv(m.predict(Xva))
        score = metrics(yva, pred)
        row = {"max_depth": depth, "learning_rate": lr, "min_child_weight": mcw, "target_transform": tname,
               "best_iteration": int(m.best_iteration), **{f"val_{k}": v for k, v in score.items()}}
        log.append(row)
        if best is None or score["MAE"] < best[0]["val_MAE"]:
            best = (row, m)
    pd.DataFrame(log).sort_values("val_MAE").round(4).to_csv(METRICS_DIR / "xgb_tuning.csv", index=False)
    return best


def tune_random_forest(f: pd.DataFrame):
    _, Xtr, ytr = split_xy(f, "train")
    _, Xva, yva = split_xy(f, "val")
    log, best = [], None
    for depth, leaf in itertools.product([None, 12], [2, 5]):
        m = RandomForestRegressor(n_estimators=300, max_depth=depth, min_samples_leaf=leaf,
                                  max_features=0.5, random_state=SEED, n_jobs=-1)
        m.fit(Xtr, ytr)
        score = metrics(yva, m.predict(Xva))
        row = {"max_depth": depth, "min_samples_leaf": leaf, **{f"val_{k}": v for k, v in score.items()}}
        log.append(row)
        if best is None or score["MAE"] < best[0]["val_MAE"]:
            best = (row, m)
    pd.DataFrame(log).sort_values("val_MAE").round(4).to_csv(METRICS_DIR / "rf_tuning.csv", index=False)
    return best


def predict_all(f: pd.DataFrame, predict_fn) -> pd.DataFrame:
    out = f[["station", "origin_date", "target_date", "split"]].copy()
    out["y_true"] = f[TARGET_COL].to_numpy()
    out["y_pred"] = predict_fn(f[FEATURES].astype(float))
    return out


def main() -> None:
    f = load_features()

    row, model = tune_xgboost(f)
    inv = TRANSFORMS[row["target_transform"]][1]
    print("XGBoost best config:", row)
    model.save_model(MODELS_DIR / "xgboost_model.json")
    schema = {"features": FEATURES, "target": TARGET_COL, "target_transform": row["target_transform"],
              "best_iteration": row["best_iteration"],
              "params": {k: row[k] for k in ["max_depth", "learning_rate", "min_child_weight"]},
              "trained_on": "train split (target dates 2022-01-02 .. 2023-12-31)",
              "xgboost_version": xgb.__version__}
    (MODELS_DIR / "xgboost_schema.json").write_text(json.dumps(schema, indent=2))
    save_predictions(predict_all(f, lambda X: inv(model.predict(X))), "xgboost")

    row, rf = tune_random_forest(f)
    print("Random Forest best config:", row)
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib", compress=3)
    save_predictions(predict_all(f, rf.predict), "random_forest")


if __name__ == "__main__":
    main()
