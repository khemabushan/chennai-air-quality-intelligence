"""Lightweight inference used by the dashboard (XGBoost only; no TensorFlow at runtime)."""
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import MODELS_DIR
from src.feature_engineering import build_feature_table

INVERSE = {"none": lambda z: z, "log1p": np.expm1}


def load_xgboost():
    schema = json.loads((MODELS_DIR / "xgboost_schema.json").read_text())
    model = xgb.XGBRegressor()
    model.load_model(MODELS_DIR / "xgboost_model.json")
    return model, schema


def predict_xgboost(model, schema, X: pd.DataFrame) -> np.ndarray:
    X = X[schema["features"]].astype(float)
    z = model.predict(X, iteration_range=(0, schema["best_iteration"] + 1))
    return INVERSE[schema["target_transform"]](z)


def forecast_from_daily(daily: pd.DataFrame, station: str, origin_date, model=None, schema=None) -> dict:
    """Recompute features for one (station, origin) from the daily calendar and predict t+1."""
    if model is None:
        model, schema = load_xgboost()
    origin_date = pd.Timestamp(origin_date)
    hist = daily[(daily["station"] == station) & (daily["date"] <= origin_date)]
    # Only history up to the origin is passed in, so the forecast cannot see the future.
    feats = build_feature_table(hist, drop_ineligible=False)
    row = feats[(feats["station"] == station) & (feats["origin_date"] == origin_date)]
    if row.empty or row["pm25_t"].isna().any():
        raise ValueError("PM2.5 was not recorded on the origin date; no forecast is issued")
    pred = float(predict_xgboost(model, schema, row)[0])
    return {"station": station, "origin_date": origin_date, "target_date": origin_date + pd.Timedelta(days=1),
            "prediction": pred, "features": row[schema["features"]].iloc[0].to_dict()}
