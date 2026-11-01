"""Persistence and seasonal-naive baselines."""
import pandas as pd

from src.config import DATA_PROCESSED
from src.evaluate import save_predictions
from src.feature_engineering import TARGET_COL


def load_features() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / "features.csv", parse_dates=["origin_date", "target_date"])


def main() -> None:
    f = load_features()
    base = f[["station", "origin_date", "target_date", "split"]].assign(y_true=f[TARGET_COL])

    # Persistence: PM2.5(t+1) = PM2.5(t)
    save_predictions(base.assign(y_pred=f["pm25_t"]), "persistence")

    # Seasonal naive (weekly): PM2.5(t+1) = PM2.5(t+1-7) = lag 6 from the origin.
    # If that reference day is missing there is no forecast (row excluded, not imputed).
    sn = base.assign(y_pred=f["pm25_lag6"]).dropna(subset=["y_pred"])
    save_predictions(sn, "seasonal_naive")
    print(f"persistence rows={len(base)} seasonal_naive rows={len(sn)}")


if __name__ == "__main__":
    main()
