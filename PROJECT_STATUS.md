# Project Status — Chennai Air Quality Intelligence

_Last updated: 2026-11-01_

**Current phase:** Phase 12 — deployment. Everything up to and including the local Git commit is
done and verified. Publishing to GitHub and deploying on Streamlit Community Cloud need the owner's
GitHub account.

## Definition of done
| Requirement | Status |
|---|---|
| Dataset verified; original CSVs unchanged (SHA-256) | ✅ |
| Reproducible preprocessing | ✅ full pipeline re-run gives identical metrics |
| Leakage-safe features (tested) | ✅ |
| Baselines evaluated | ✅ |
| XGBoost trained & evaluated | ✅ |
| LSTM trained & evaluated | ✅ |
| Genuine test metrics | ✅ `results/metrics/overall_test.csv` |
| Dashboard works locally | ✅ all 6 pages checked in a browser and by AppTest |
| Automated tests executed | ✅ 22 passed, 0 failed |
| README complete | ✅ (live URL to add after deploy) |
| GitHub repository | 🟡 local repo committed on `main`; not pushed (needs a GitHub repo and the owner's login) |
| Deployment verified | ⏳ not started (needs the owner's Streamlit Community Cloud login) |
| Academic documentation | ✅ `reports/academic_report.md`, `reports/viva_and_presentation.md` |

## Completed
- Data audit + 6 EDA figures → `reports/data_audit.md`.
- Findings: 392 missing station-days; values are US EPA AQI sub-indices from WAQI (explains PM2.5 > PM10 in 84.3% of rows);
  Kodungaiyur 126-day constant PM2.5 run (masked by a documented QC rule); 49-day shared gap in Nov–Dec 2023.
- Daily calendar preprocessing, 28 causal features, split by target date (train 3,831 / val 1,082 / test 1,063).
- Models: persistence, seasonal naive, XGBoost (24-config grid), Random Forest (4 configs), LSTM (4 configs).
- Evaluation on common rows, per-station metrics, figures, gain importance, TreeSHAP and an interpretation text.
- Streamlit app (6 pages): no training at startup, no TensorFlow at runtime, live XGBoost re-inference.

## Actual results — test (Jul–Dec 2024, 1,046 common station-days)
| Model | MAE | RMSE | R² | sMAPE | ΔMAE vs persistence |
|---|---:|---:|---:|---:|---:|
| LSTM (7-day window, 64 units) | 11.01 | 35.43 | 0.608 | 13.48 | +39.7% |
| XGBoost (depth 4, lr 0.03, 314 trees) | 13.03 | 36.95 | 0.573 | 16.69 | +28.6% |
| Random Forest | 13.77 | 37.95 | 0.550 | 18.21 | +24.6% |
| Persistence | 18.26 | 38.94 | 0.526 | 23.48 | — |
| Seasonal naive | 34.22 | 69.55 | −0.511 | 40.64 | −87.4% |

Caveats: RMSE improves only 5–9% (extreme spikes). At Velachery, persistence has a higher R² than all learned models.

## Commands executed
```
py -3.11 -m venv C:\venvs\caqi
C:\venvs\caqi\Scripts\python -m pip install pandas numpy scikit-learn xgboost tensorflow==2.17.* matplotlib seaborn plotly streamlit shap pytest joblib
python scripts/run_pipeline.py          # ~3.7 min on CPU
python -m pytest -q                      # 22 passed
streamlit run app.py
git init -b main; git add -A; git commit
```

## Tests
22 passed / 0 failed (`tests/test_data.py` 6, `test_features.py` 6, `test_models.py` 4, `test_app.py` 6).

## Known issues / notes
- The venv is at `C:\venvs\caqi`. Inside the project folder, the long path hits the
  Windows MAX_PATH limit and TensorFlow/Pillow DLLs fail to load ("filename or extension is too long").
- `models/random_forest.joblib` (5 MB) is git-ignored: its predictions are saved and the app doesn't need it.
- The `gh` CLI is not installed. Push with plain `git` after creating an empty GitHub repository.

## Deployment status
Not deployed. Runtime `requirements.txt` holds only streamlit, pandas, numpy, plotly, xgboost and
scikit-learn. Repository size is about 12 MB.

## Next actions
1. Create an empty GitHub repository `chennai-air-quality-intelligence` (owner action).
2. `git remote add origin https://github.com/<user>/chennai-air-quality-intelligence.git` and `git push -u origin main`.
3. share.streamlit.io → New app → repo, branch `main`, file `app.py`, Python 3.11 → Deploy.
4. Open the public URL, check all 6 pages, then put the URL in README.md and `GITHUB_URL` in app.py.
