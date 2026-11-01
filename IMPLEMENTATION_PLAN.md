# Implementation Plan — Chennai Air Quality Intelligence

## Goal
Forecast next-day (t+1) PM2.5 at six Chennai CAAQMS stations from pollutant history up to day t,
compare baselines, XGBoost, Random Forest and an LSTM on a chronological hold-out, explain XGBoost,
and serve everything in a Streamlit dashboard deployed on Streamlit Community Cloud.

## Architecture

```
data/raw/*.csv (original, hash-verified, read-only)
      │  src/data_loader.py, src/data_validation.py, src/audit.py  → reports/data_audit.md, EDA figures
      ▼
src/preprocessing.py   full daily calendar per station (gaps explicit, nothing imputed)
      ▼
src/feature_engineering.py   lags / trailing rolls / pollutants / target-day calendar / station one-hot
      ▼   data/processed/{daily,features}.csv   (split assigned by target date)
 ┌────────────────┬──────────────────┬───────────────────┐
 train_baselines   train_ml (XGB, RF)  train_lstm (Keras)
 └────────────────┴────────┬─────────┴───────────────────┘
                 results/predictions/<model>.csv
                           ▼
 src/evaluate.py (common-row metrics) · src/plots.py · src/explainability.py (gain + TreeSHAP)
                           ▼
 app.py (Streamlit; reads artifacts only, XGBoost live re-inference, no TensorFlow at runtime)
```

## Key design decisions
| Decision | Choice | Why |
|---|---|---|
| Forecast convention | end of day t → PM2.5(t+1) | matches an operational daily bulletin |
| Missing days | reindex calendar; origin needs PM2.5(t) and PM2.5(t+1) observed; other inputs NaN | no future info, labels never imputed |
| Rolling windows | trailing, includes t, `min_periods` | causal |
| Split | by target date: train ≤2023-12-31, val 2024H1, test 2024H2 | chronological, no random split |
| Tuning | small grids, val MAE, early stopping on val | deadline; test untouched |
| LSTM gaps | causal ffill ≤3 days inside windows + observed-mask channel; windows with longer gaps skipped | sequences need dense input |
| LSTM scaling | log1p + z-score fitted on training-period data only | no scaler leakage |
| Fair comparison | metrics on the intersection of rows all models predict | identical test targets |
| Deployment | `requirements.txt` without TensorFlow; LSTM predictions pre-computed | Community Cloud resource limits |

## Schedule (4 days)
1. **Day 1** – workspace, dataset verification, audit/EDA, preprocessing, features, baselines, tests.
2. **Day 2** – XGBoost, Random Forest, LSTM, evaluation, plots, SHAP.
3. **Day 3** – Streamlit dashboard, inference integration, dashboard tests.
4. **Day 4** – README, academic report, viva material, GitHub push, Streamlit deployment, verification.

## Reproduce
```
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python scripts\run_pipeline.py
.venv\Scripts\python -m pytest -q
.venv\Scripts\streamlit run app.py
```
