import numpy as np
import pandas as pd
import pytest

from src.config import DATA_PROCESSED, MODELS_DIR, PREDICTIONS_DIR

pytestmark = pytest.mark.skipif(not (MODELS_DIR / "xgboost_model.json").exists(),
                                reason="run scripts/run_pipeline.py first")


@pytest.fixture(scope="module")
def daily():
    return pd.read_csv(DATA_PROCESSED / "daily.csv", parse_dates=["date"])


def test_xgboost_inference_shape_and_reproducibility():
    from src.inference import load_xgboost, predict_xgboost
    from src.train_baselines import load_features
    model, schema = load_xgboost()
    f = load_features()
    test = f[f.split == "test"]
    pred = predict_xgboost(model, schema, test)
    assert pred.shape == (len(test),)
    assert np.isfinite(pred).all()
    saved = pd.read_csv(PREDICTIONS_DIR / "xgboost.csv", parse_dates=["origin_date"])
    saved = saved[saved.split == "test"].reset_index(drop=True)
    np.testing.assert_allclose(pred, saved.y_pred.to_numpy(), rtol=1e-4, atol=1e-3)


def test_live_forecast_uses_only_past_data(daily):
    from src.inference import forecast_from_daily
    origin = pd.Timestamp("2024-09-10")
    st = "Alandur"
    if pd.isna(daily[(daily.station == st) & (daily.date == origin)]["PM2.5"]).all():
        pytest.skip("origin not observed")
    a = forecast_from_daily(daily, st, origin)
    tampered = daily.copy()
    tampered.loc[tampered.date > origin, "PM2.5"] = 5000
    b = forecast_from_daily(tampered, st, origin)
    assert a["prediction"] == pytest.approx(b["prediction"])
    assert a["target_date"] == origin + pd.Timedelta(days=1)


def test_prediction_files_are_complete():
    for m in ["persistence", "seasonal_naive", "xgboost", "random_forest", "lstm"]:
        p = pd.read_csv(PREDICTIONS_DIR / f"{m}.csv")
        assert {"station", "origin_date", "target_date", "split", "y_true", "y_pred"} <= set(p.columns)
        assert p.y_pred.notna().all()
        assert set(p.split) <= {"train", "val", "test"}


def test_lstm_scaler_fitted_on_training_period_only(daily):
    import json
    from src.train_lstm import fit_scaler
    meta = json.loads((MODELS_DIR / "lstm_metadata.json").read_text())
    expected = fit_scaler(daily)
    assert meta["scaler"]["mean"]["PM2.5"] == pytest.approx(expected["mean"]["PM2.5"])
    full = np.log1p(daily["PM2.5"]).mean()
    assert meta["scaler"]["mean"]["PM2.5"] != pytest.approx(full, abs=1e-6)
