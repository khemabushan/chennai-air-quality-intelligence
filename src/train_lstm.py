"""Station-aware LSTM for next-day PM2.5.

Inputs for origin t: the last L calendar days (t-L+1 .. t) of the five pollutants plus an
"observed" mask, and static context (station one-hot, target-day calendar).

Missing days inside a window are causally forward-filled (at most 3 days, never backward);
windows that still contain gaps are skipped. Scaling (log1p + standardisation) is fitted on
training-period observations only. Labels are never imputed.
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import (DATA_PROCESSED, FIGURES_DIR, METRICS_DIR, MODELS_DIR, POLLUTANTS, SEED,
                        STATIONS, TRAIN_END)
from src.evaluate import metrics, save_predictions
from src.feature_engineering import TARGET_COL
from src.train_baselines import load_features

FFILL_LIMIT = 3
STATIC_COLS = [f"station_{s}" for s in STATIONS] + ["target_doy_sin", "target_doy_cos", "target_is_weekend"]


def fit_scaler(daily: pd.DataFrame) -> dict:
    tr = daily[daily["date"] <= pd.Timestamp(TRAIN_END)]
    logv = np.log1p(tr[POLLUTANTS])
    return {"mean": logv.mean().to_dict(), "std": logv.std().to_dict()}


def scale(values: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    return (np.log1p(values) - pd.Series(scaler["mean"])) / pd.Series(scaler["std"])


def build_sequences(daily: pd.DataFrame, feats: pd.DataFrame, scaler: dict, seq_len: int):
    seq_by_station = {}
    for st, g in daily.groupby("station"):
        g = g.sort_values("date").set_index("date")
        mask = g["PM2.5"].notna().astype(float)
        filled = g[POLLUTANTS].ffill(limit=FFILL_LIMIT)
        arr = np.column_stack([scale(filled, scaler).to_numpy(), mask.to_numpy()])
        seq_by_station[st] = (g.index, arr)

    X_seq, X_static, keep = [], [], []
    for i, row in enumerate(feats.itertuples(index=False)):
        dates, arr = seq_by_station[row.station]
        end = dates.get_loc(row.origin_date)
        start = end - seq_len + 1
        if start < 0:
            continue
        window = arr[start:end + 1]
        if np.isnan(window).any():
            continue
        X_seq.append(window)
        keep.append(i)
    sub = feats.iloc[keep].reset_index(drop=True)
    X_static = sub[STATIC_COLS].to_numpy(float)
    return np.asarray(X_seq, dtype="float32"), X_static.astype("float32"), sub


def build_model(seq_len: int, n_feat: int, n_static: int, units: int) -> tf.keras.Model:
    seq_in = tf.keras.Input(shape=(seq_len, n_feat), name="sequence")
    static_in = tf.keras.Input(shape=(n_static,), name="static")
    h = tf.keras.layers.LSTM(units, name="lstm")(seq_in)
    h = tf.keras.layers.Concatenate()([h, static_in])
    h = tf.keras.layers.Dense(16, activation="relu", name="dense_16")(h)
    out = tf.keras.layers.Dense(1, activation="linear", name="pm25_next")(h)
    model = tf.keras.Model([seq_in, static_in], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def main() -> None:
    daily = pd.read_csv(DATA_PROCESSED / "daily.csv", parse_dates=["date"])
    feats = load_features()
    scaler = fit_scaler(daily)
    mu, sd = scaler["mean"]["PM2.5"], scaler["std"]["PM2.5"]
    to_z = lambda y: (np.log1p(y) - mu) / sd
    from_z = lambda z: np.expm1(z * sd + mu)

    log, best = [], None
    for seq_len in [7, 14]:
        Xs, Xc, sub = build_sequences(daily, feats, scaler, seq_len)
        y = to_z(sub[TARGET_COL].to_numpy(float)).astype("float32")
        idx = {s: (sub["split"] == s).to_numpy() for s in ["train", "val", "test"]}
        for units in [32, 64]:
            tf.keras.utils.set_random_seed(SEED)
            model = build_model(seq_len, Xs.shape[2], Xc.shape[1], units)
            es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
            hist = model.fit([Xs[idx["train"]], Xc[idx["train"]]], y[idx["train"]],
                             validation_data=([Xs[idx["val"]], Xc[idx["val"]]], y[idx["val"]]),
                             epochs=200, batch_size=64, callbacks=[es], verbose=0, shuffle=True)
            pred_val = from_z(model.predict([Xs[idx["val"]], Xc[idx["val"]]], verbose=0).ravel())
            score = metrics(sub.loc[idx["val"], TARGET_COL], pred_val)
            row = {"seq_len": seq_len, "units": units, "epochs_run": len(hist.history["loss"]),
                   "best_epoch": int(np.argmin(hist.history["val_loss"]) + 1),
                   **{f"val_{k}": v for k, v in score.items()}}
            print(row)
            log.append(row)
            if best is None or score["MAE"] < best[0]["val_MAE"]:
                best = (row, model, hist.history, Xs, Xc, sub)

    pd.DataFrame(log).sort_values("val_MAE").round(4).to_csv(METRICS_DIR / "lstm_tuning.csv", index=False)
    row, model, history, Xs, Xc, sub = best
    print("LSTM best config:", row)
    model.save(MODELS_DIR / "lstm_model.keras")
    (MODELS_DIR / "lstm_metadata.json").write_text(json.dumps(
        {"config": row, "scaler": scaler, "static_cols": STATIC_COLS, "ffill_limit": FFILL_LIMIT,
         "sequence_channels": POLLUTANTS + ["pm25_observed_mask"], "tensorflow_version": tf.__version__},
        indent=2))
    pd.DataFrame(history).to_csv(METRICS_DIR / "lstm_history.csv", index_label="epoch")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["loss"], label="training loss")
    ax.plot(history["val_loss"], label="validation loss")
    ax.axvline(row["best_epoch"] - 1, color="grey", ls="--", lw=1, label="restored best epoch")
    ax.set(xlabel="epoch", ylabel="MSE (scaled log PM2.5)",
           title=f"LSTM loss (seq_len={row['seq_len']}, units={row['units']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "lstm_loss_curve.png", dpi=130)
    plt.close(fig)

    out = sub[["station", "origin_date", "target_date", "split"]].copy()
    out["y_true"] = sub[TARGET_COL].to_numpy()
    out["y_pred"] = from_z(model.predict([Xs, Xc], verbose=0).ravel())
    save_predictions(out, "lstm")


if __name__ == "__main__":
    main()
