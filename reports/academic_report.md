# Air Quality Prediction Using Machine Learning
## Chennai Air Quality Intelligence: Explainable Multi-Station PM2.5 Forecasting Using Machine Learning and Deep Learning

*Final-year B.Tech project — Computer Science (Artificial Intelligence and Machine Learning)*

---

## 1. Abstract
We forecast next-day PM2.5 concentration at six Chennai monitoring stations using three years
(2022–2024) of daily pollutant data from a public Mendeley dataset. A data audit revealed missing
dates and a 126-day constant filler run, and established from the upstream source (WAQI) that the
values are US EPA AQI sub-indices rather than concentrations. We built a
leakage-safe pipeline (daily calendar reindexing, causal lag and rolling features, labels never
imputed) and compared two baselines (persistence, weekly seasonal-naive) with XGBoost, Random Forest
and a station-aware LSTM, using a chronological split by target date. On the held-out test period
(July–December 2024, 1,046 common station-days) the LSTM achieved MAE 11.01 AQI points (R² 0.61), a 39.7%
MAE reduction over persistence (18.26), followed by XGBoost (13.03, −28.6%). RMSE gains were modest
(5–9%) because rare extreme spikes dominate squared error. TreeSHAP showed that same-day PM10 and PM2.5
and short rolling means drive XGBoost forecasts. A Streamlit dashboard presents the historical data,
backtested forecasts, model comparison and explanations.

## 2. Introduction
Fine particulate matter (PM2.5, diameter ≤ 2.5 µm) penetrates deep into the lungs and is associated
with cardiovascular and respiratory disease. Chennai operates several Continuous Ambient Air Quality
Monitoring Stations (CAAQMS). A reliable next-day forecast lets residents and agencies plan
outdoor activity and short-term interventions. Time-series forecasting of pollutants is challenging
because of missing data, sensor problems, and dependence on unobserved drivers such as weather.

## 3. Problem statement
Given daily observations of PM2.5, PM10, NO2, SO2 and O3 up to and including day *t* at a station,
predict that station's PM2.5 on day *t+1*, and determine whether machine-learning and deep-learning
models improve meaningfully over simple baselines when evaluated honestly on unseen future data.

## 4. Objectives
1. Audit and document the dataset's quality.
2. Design a reproducible, leakage-free preprocessing and feature pipeline.
3. Implement persistence and seasonal-naive baselines.
4. Train and tune XGBoost (and Random Forest) and an LSTM network.
5. Evaluate with MAE, RMSE, R², sMAPE on identical test targets, overall and per station.
6. Explain the XGBoost model using feature importance and SHAP.
7. Deliver an interactive, deployable dashboard.

## 5. Literature / background
- **Persistence and seasonal-naive** forecasts are the standard reference for short-horizon
  forecasting; skilful models must beat them (Hyndman & Athanasopoulos, 2021).
- **Gradient-boosted trees** (XGBoost; Chen & Guestrin, 2016) are strong on tabular lag features and
  handle missing values natively; they are widely used for air-quality forecasting.
- **Recurrent neural networks**, particularly **LSTM** (Hochreiter & Schmidhuber, 1997), learn temporal
  patterns directly from sequences via gated memory cells and have been applied to PM2.5 forecasting
  in many Asian cities.
- **SHAP** (Lundberg & Lee, 2017) attributes each prediction to features using Shapley values;
  TreeSHAP computes them exactly for tree ensembles (Lundberg et al., 2020).
- The source dataset's authors proposed a hybrid Prophet-XGBoost model; we do **not** reuse their
  code or results and use the data independently.

## 6. Dataset description
| Property | Value |
|---|---|
| Source | Mendeley Data, DOI 10.17632/br5dyxm7c6.1 |
| Stations | Alandur, Kodungaiyur, Manali, Perungudi, Royapuram, Velachery |
| Period | 1 Jan 2022 – 31 Dec 2024 (daily) |
| Records | 6,184 of 6,576 possible station-days (392 missing) |
| Variables | PM2.5, PM10, NO2, SO2, O3 (no missing values within recorded rows; no negatives) |
| Units | US EPA AQI sub-index per pollutant (WAQI converts all species; "no raw concentrations") |
| Upstream | WAQI / aqicn.org historic dashboard; cleaned and imputed by the publisher |
| Licence | CC BY 4.0 |

Key issues (see `reports/data_audit.md`):
- PM2.5 > PM10 in 84.3% of rows (60.9% Kodungaiyur – 97.3% Alandur); median 71 vs 48. This would be
  impossible for concentrations but is expected for AQI sub-indices, because US EPA breakpoints are far
  stricter for PM2.5 (AQI 100 ≈ 35 µg/m³) than PM10 (AQI 100 ≈ 154 µg/m³). It confirmed the unit finding.
- A shared 49-day gap (12 Nov – 30 Dec 2023) at Kodungaiyur, Perungudi and Royapuram.
- Kodungaiyur PM2.5 = 116.0 for 126 consecutive days (Jan–May 2022) — masked as filler.
- Many constant runs in SO2 and O3 (likely instrument floors); extreme values up to 790.

## 7. Proposed methodology
```
 Raw CSVs (hash-verified, unchanged)
   → Audit & validation (report only)
   → Daily calendar per station + QC mask (gaps explicit, no imputation)
   → Feature engineering (origin t → target t+1, causal only)
   → Chronological split by target date
   → Baselines | XGBoost | Random Forest | LSTM
   → Evaluation on common test rows → SHAP explanations → Streamlit dashboard
```
**Missing-data policy.** A forecast origin is used only when PM2.5 was recorded on both *t* and *t+1*.
Missing lag/rolling inputs remain NaN for tree models. For the LSTM, gaps inside a window are
forward-filled causally for at most 3 days (with an "observed" mask channel); windows with longer
gaps are skipped. Labels are never imputed.

**Split.** Train: target dates 2 Jan 2022 – 31 Dec 2023 (3,831 samples). Validation: Jan–Jun 2024
(1,082). Test: Jul–Dec 2024 (1,063). History from earlier periods is used as model input where it
would genuinely be available.

## 8. Feature engineering (28 features)
| Group | Features |
|---|---|
| PM2.5 history | value at *t*; lags 1, 2, 3, 6, 7, 14 days |
| Rolling (trailing, ending at *t*) | mean over 3/7/14 days; std over 7 days |
| Co-pollutants | PM10, NO2, SO2, O3 at *t*; 7-day PM10 mean |
| Data quality | number of observed days in the last 7 |
| Target-day calendar | day of week, month, sin/cos day-of-year, weekend flag |
| Station | one-hot (6) |

Lag 6 from the origin equals "same weekday last week" relative to the target. Automated tests prove
that (i) the label equals PM2.5 on the next calendar day, (ii) overwriting all future values does not
change any feature of an origin, and (iii) changing one station's data does not change another's features.

## 9. XGBoost methodology
XGBoost fits an additive ensemble of regression trees by gradient boosting with regularisation.
We searched 24 configurations: max depth {3, 4, 6} × learning rate {0.03, 0.08} × min child weight
{1, 5} × target {raw, log1p}, with subsample = colsample = 0.8, up to 2,000 trees and early stopping
(100 rounds) on validation MAE. Selected: depth 4, learning rate 0.03, min child weight 1, raw target,
314 trees (validation MAE 9.92). The model and its feature schema are saved as JSON. A Random Forest
(300 trees, 4 configurations) was trained the same way for comparison.

## 10. LSTM architecture
```
 Sequence input: 7 days × 6 channels         Static input: 9 values
 (log1p z-scored PM2.5, PM10, NO2,            (station one-hot ×6, target-day
  SO2, O3 + observed mask)                     sin/cos day-of-year, weekend)
            │                                          │
       LSTM (64 units)                                 │
            └──────────────── Concatenate ─────────────┘
                                 │
                       Dense (16, ReLU)
                                 │
                       Dense (1, linear) → z-scored log PM2.5(t+1) → inverse transform
```
- **Why LSTM:** its input, forget and output gates let it retain relevant information across the
  window and learn temporal dynamics (rises, decays) without hand-crafted lags.
- **Scaling:** log1p then z-score, with means/stds fitted only on training-period observations.
- **Training:** Adam (lr 1e-3), MSE loss, batch 64, up to 200 epochs, early stopping on validation
  loss (patience 15, best weights restored), seed 42.
- **Tuning:** window {7, 14} × units {32, 64}; selected 7 days × 64 units (best epoch 7, validation MAE 8.76).
- Loss curves: `results/figures/lstm_loss_curve.png` (validation loss plateaus after ~5 epochs; training
  continues to fall, so early stopping prevents overfitting).

## 11. Experimental setup
Windows 11, Python 3.11, CPU only. pandas 3.0, scikit-learn 1.9, XGBoost 3.2, TensorFlow 2.17,
SHAP 0.49, Streamlit 1.60. Full pipeline: `python scripts/run_pipeline.py`; tests: `pytest` (22 tests).

## 12. Evaluation metrics
- **MAE** = mean |y − ŷ| (AQI points) — primary selection metric, robust to outliers.
- **RMSE** = √mean (y − ŷ)² — penalises large errors.
- **R²** = 1 − SS_res / SS_tot — share of variance explained.
- **sMAPE** = mean 2|y − ŷ| / (|y| + |ŷ|) × 100 %.
- **Improvement over persistence** = 1 − metric_model / metric_persistence.

All models are compared on the intersection of station-days that every model could forecast
(1,046 test rows; seasonal-naive and LSTM need slightly more history).

## 13. Results and discussion
**Test period (Jul–Dec 2024, n = 1,046):**

| Model | MAE | RMSE | R² | sMAPE % | ΔMAE vs persistence |
|---|---:|---:|---:|---:|---:|
| LSTM | 11.01 | 35.43 | 0.608 | 13.5 | +39.7% |
| XGBoost | 13.03 | 36.95 | 0.573 | 16.7 | +28.6% |
| Random Forest | 13.77 | 37.95 | 0.550 | 18.2 | +24.6% |
| Persistence | 18.26 | 38.94 | 0.526 | 23.5 | — |
| Seasonal naive | 34.22 | 69.55 | −0.511 | 40.6 | −87.4% |

**Validation (Jan–Jun 2024, n = 1,053):** LSTM MAE 8.79 (R² 0.869), XGBoost 9.94 (0.850),
RF 10.09 (0.836), persistence 14.22 (0.672), seasonal naive 24.69 (0.184).

**Per-station test MAE:** LSTM is best at every station (5.39 at Kodungaiyur to 19.34 at Velachery).

**Discussion.**
1. All learned models clearly beat persistence on MAE, and the ranking is consistent between
   validation and test, which supports the model selection.
2. The LSTM outperformed XGBoost. Plausible reasons: it sees the full recent trajectory of all five
   pollutants, and its log-scale target reduces the influence of extreme values on training.
3. RMSE improvements are small (5–9%): rare extreme spikes in the test period (up to 790) are not
   predictable from pollutant history alone and dominate squared error.
4. Velachery is hardest: learned models have lower MAE but **lower R² than persistence**
   (e.g. LSTM 0.29 vs 0.34) because of abrupt, large jumps.
5. Seasonal-naive is much worse than persistence: day-to-day autocorrelation dominates any weekly cycle.
6. **Explainability (XGBoost, TreeSHAP on test):** the largest mean |SHAP| values are same-day PM10 (16.7),
   same-day PM2.5 (15.4) and 7-day PM10 mean (7.1). Recent PM2.5 history and co-pollutants each carry
   about 45% of attribution; station identity about 5%. High 7-day PM10 means mostly receive *negative*
   contributions while high same-day PM10 receives positive ones, consistent with the model using
   "today relative to the recent level" as a signal.
   These are model associations, not causal effects.

## 14. Limitations
- The target is an AQI sub-index (piecewise-linear in concentration), not a concentration in µg/m³.
- No meteorological variables; a one-day horizon only; point forecasts without uncertainty.
- Single random seed for the LSTM; one chronological test window (six months).
- Publisher's cleaning/imputation is only briefly described; the 126-day filler run shows some values are not measurements.
- Historical backtest only — the data ends in December 2024.

## 15. Conclusion
A carefully validated pipeline shows that machine learning and deep learning give substantial,
genuine improvements over persistence for next-day PM2.5 in Chennai (MAE −29% for XGBoost, −40% for
the LSTM), while extreme pollution spikes remain hard to forecast without weather information. The
project contributes an honest data audit, leakage-tested features, a fair identical-target comparison,
SHAP-based interpretation, and a deployable dashboard. Future work: weather and festival features,
multi-seed ensembles with prediction intervals, and multi-day horizons.

## 16. References
1. Hybrid Prophet-XGBoost Forecasting for Urban Air Quality with Real-Time Data Integration for Chennai.
   Mendeley Data, V1, 2025. doi:10.17632/br5dyxm7c6.1
2. T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," *KDD*, 2016.
3. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, 9(8), 1997.
4. S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," *NeurIPS*, 2017.
5. S. M. Lundberg et al., "From local explanations to global understanding with explainable AI for
   trees," *Nature Machine Intelligence*, 2, 2020.
6. R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed., OTexts, 2021.
7. L. Breiman, "Random Forests," *Machine Learning*, 45, 2001.
8. World Air Quality Index Project, "Air Quality Open Data Platform", https://aqicn.org/data-platform/
9. U.S. EPA, *Technical Assistance Document for the Reporting of Daily Air Quality — the AQI*, 2018/2024.
