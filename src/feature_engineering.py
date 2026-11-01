"""Leakage-safe features for next-day PM2.5 forecasting.

Convention: at the end of forecast-origin day t we predict PM2.5 on target day t+1.
Every feature for a row uses only values dated <= t. All shifts and rolling windows run on
the station's complete daily calendar, so "lag 7" really means seven calendar days earlier
(NaN if that day was not recorded) and stations never mix.

Missing-data policy:
* An origin is eligible only if PM2.5 was observed on day t AND on day t+1 (the label).
  Labels are never imputed.
* Other missing lag/rolling inputs stay NaN; tree models handle NaN natively.
* Rolling statistics are trailing windows ending at t with a minimum-observation requirement.
"""
import numpy as np
import pandas as pd

from src.config import STATIONS
from src.preprocessing import assign_split

PM_LAGS = [1, 2, 3, 6, 7, 14]
ROLL_MEAN_WINDOWS = {3: 2, 7: 4, 14: 7}  # window -> min_periods

FEATURES = (
    ["pm25_t"]
    + [f"pm25_lag{k}" for k in PM_LAGS]
    + [f"pm25_roll{w}_mean" for w in ROLL_MEAN_WINDOWS]
    + ["pm25_roll7_std", "pm10_t", "no2_t", "so2_t", "o3_t", "pm10_roll7_mean", "obs_count_7d",
       "target_dow", "target_month", "target_doy_sin", "target_doy_cos", "target_is_weekend"]
    + [f"station_{s}" for s in STATIONS]
)
ID_COLUMNS = ["station", "origin_date", "target_date", "split"]
TARGET_COL = "y_next_pm25"


def _station_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    if not g["date"].diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("station calendar must be contiguous daily rows")
    pm = g["PM2.5"]
    f = pd.DataFrame({"station": g["station"], "origin_date": g["date"]})
    f["target_date"] = f["origin_date"] + pd.Timedelta(days=1)
    f["pm25_t"] = pm
    for k in PM_LAGS:
        f[f"pm25_lag{k}"] = pm.shift(k)
    for w, mp in ROLL_MEAN_WINDOWS.items():
        f[f"pm25_roll{w}_mean"] = pm.rolling(w, min_periods=mp).mean()
    f["pm25_roll7_std"] = pm.rolling(7, min_periods=4).std()
    f["pm10_t"] = g["PM10"]
    f["no2_t"] = g["NO2"]
    f["so2_t"] = g["SO2"]
    f["o3_t"] = g["O3"]
    f["pm10_roll7_mean"] = g["PM10"].rolling(7, min_periods=4).mean()
    f["obs_count_7d"] = pm.notna().astype(int).rolling(7, min_periods=1).sum()
    f[TARGET_COL] = pm.shift(-1)
    return f


def build_feature_table(daily: pd.DataFrame, drop_ineligible: bool = True) -> pd.DataFrame:
    frames = [_station_features(daily[daily["station"] == s]) for s in STATIONS]
    f = pd.concat(frames, ignore_index=True)
    td = f["target_date"]
    # Calendar attributes of the target day are known in advance, so they are not leakage.
    f["target_dow"] = td.dt.dayofweek
    f["target_month"] = td.dt.month
    doy = td.dt.dayofyear
    f["target_doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["target_doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["target_is_weekend"] = (td.dt.dayofweek >= 5).astype(int)
    for s in STATIONS:
        f[f"station_{s}"] = (f["station"] == s).astype(int)
    f["split"] = assign_split(td)
    f = f[f["split"] != "out_of_range"]
    if drop_ineligible:
        f = f[f["pm25_t"].notna() & f[TARGET_COL].notna()]
    return f[ID_COLUMNS + FEATURES + [TARGET_COL]].reset_index(drop=True)
