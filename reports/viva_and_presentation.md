# Viva, Presentation & Portfolio Material

## 1. Architecture diagram
```
┌──────────────────────┐   ┌──────────────────────┐   ┌───────────────────────────┐
│ Mendeley CSVs (6)    │──►│ Audit & validation   │──►│ Daily calendar + QC mask  │
│ hash-verified, R/O   │   │ reports/data_audit.md│   │ gaps explicit, no impute  │
└──────────────────────┘   └──────────────────────┘   └─────────────┬─────────────┘
                                                                     ▼
                                     ┌────────────────────────────────────────────┐
                                     │ Features: origin t → target t+1 (causal)   │
                                     │ split by target date: train/val/test       │
                                     └───────┬───────────┬────────────┬───────────┘
                                             ▼           ▼            ▼
                                       Baselines   XGBoost / RF   LSTM (Keras)
                                             └───────────┼────────────┘
                                                         ▼
                                  Predictions · metrics · TreeSHAP · figures
                                                         ▼
                                  Streamlit dashboard (artifacts only, XGBoost live check)
```

## 2. Short explanation for professors (≈150 words)
The project forecasts tomorrow's PM2.5 at six Chennai monitoring stations from today's and earlier
pollutant readings. I first audited the public Mendeley dataset and found missing dates, a 126-day
filler value and PM2.5 > PM10 in 84% of rows, which I documented rather than hid. I reindexed each
station to a daily calendar, built only causal features (lags, trailing rolling statistics,
co-pollutants, calendar, station), and split the data chronologically by target date: 2022–23 train,
Jan–Jun 2024 validation, Jul–Dec 2024 test. Tests prove there is no future leakage. I compared
persistence and seasonal-naive baselines with XGBoost, Random Forest and an LSTM, all on the same test
days. The LSTM reduced MAE by 40% versus persistence (11.0 vs 18.3), XGBoost by 29%. SHAP shows
recent PM2.5 and PM10 dominate. A Streamlit dashboard presents everything.

## 3. Two-minute presentation script
> **(0:00)** Good morning. My project is *Chennai Air Quality Intelligence* — forecasting tomorrow's
> PM2.5, the most harmful air pollutant, at six Chennai monitoring stations.
>
> **(0:15)** I used a public Mendeley dataset with three years of daily data. Before modelling I audited
> it, and found real problems: 392 missing station-days, one station repeating the same value for 126
> days, and PM2.5 larger than PM10 in 84 percent of rows. I documented all of this and masked only the
> obvious filler run.
>
> **(0:40)** The key design rule is no data leakage. At the end of day *t* the model may only use data
> up to *t*. I built lag and trailing rolling features per station, split the data by date — 2022–23 for
> training, early 2024 for validation, late 2024 for testing — and wrote automated tests that prove
> future values cannot affect features.
>
> **(1:00)** I compared five models on exactly the same test days. Persistence — "tomorrow equals today"
> — has an MAE of 18.3. XGBoost reduces this to 13.0, a 29 percent improvement. My LSTM network, which
> reads the last seven days of all pollutants, does best at 11.0, about 40 percent better than persistence.
> Honestly, extreme spikes are still hard — RMSE improves only 5 to 9 percent.
>
> **(1:30)** For explainability I used SHAP: today's PM10 and PM2.5 and short rolling averages drive the
> forecasts. Finally, everything is in a Streamlit dashboard with historical exploration, a forecast
> explorer with per-prediction explanations, and model comparison. Thank you.

## 4. Common viva questions and answers
**Q1. Why not a random train-test split?**
Random splitting lets the model train on days after the test days, leaking future information and
inflating accuracy. Forecasting must be evaluated on a later period, so we split by target date.

**Q2. How did you prevent data leakage?**
(a) Features use only values dated ≤ *t* (shifts and trailing windows on a daily calendar);
(b) no backward fill or centred windows; (c) LSTM scaler fitted on training-period data only;
(d) model choice used validation only; (e) tests overwrite all future values and verify features are unchanged.

**Q3. How did you handle missing data?**
Reindexed each station to a full calendar so gaps are explicit. An origin is used only if PM2.5 is
recorded on *t* and *t+1*. Labels are never imputed. Trees handle missing inputs natively; the LSTM
forward-fills at most 3 days inside a window with a mask channel and skips longer gaps.

**Q4. Why use persistence as a baseline?**
Air quality is strongly autocorrelated, so "tomorrow = today" is hard to beat. A model is useful only
if it beats it; ours beat it by 29–40% MAE.

**Q5. Why did the LSTM beat XGBoost?**
It sees the full 7-day trajectory of all five pollutants and learns dynamics directly, and it was trained
on a log-scaled target. With a single seed and one test window the margin should be treated cautiously.

**Q6. Explain the LSTM gates.**
The forget gate decides what to drop from the cell state, the input gate what new information to store,
and the output gate what part of the cell state to expose as the hidden state. This mitigates vanishing
gradients and lets the network keep information across time steps.

**Q7. What does SHAP tell you, and what does it not?**
It splits each prediction into additive feature contributions (Shapley values). It explains the
model's behaviour, not causation in the atmosphere.

**Q8. Why is RMSE improvement so small?**
RMSE squares errors, so a few extreme spikes dominate it. Those spikes are not predictable from
pollutant history alone (they depend on weather and events like festivals).

**Q9. What is the PM2.5 > PM10 problem?**
PM2.5 is a subset of PM10, so PM2.5 should never exceed PM10. It does in 84% of rows, suggesting a
labelling or processing issue in the published data. We use columns as labelled and flag this as the
main limitation.

**Q10. Why is the seasonal-naive baseline so poor?**
Day-to-day persistence is much stronger than any weekly cycle; values from a week ago are often in a
different pollution regime.

**Q11. Is the dashboard forecasting live air quality?**
No. The data ends in December 2024; the dashboard shows backtested next-day forecasts for historical dates.

**Q12. How would you improve the project?**
Add weather (wind, rain, humidity), festival indicators, multi-seed ensembles with prediction intervals,
multi-day horizons, and rolling-origin validation.

**Q13. What are MAE, RMSE, R², sMAPE?**
MAE: average absolute error. RMSE: square-root of mean squared error, penalises big misses.
R²: proportion of variance explained. sMAPE: symmetric percentage error.

**Q14. Why exclude TensorFlow from the deployed app?**
It is large and not needed at runtime; LSTM predictions are pre-computed, keeping deployment light.

## 5. Resume bullet points (truthful)
- Built an end-to-end next-day PM2.5 forecasting system for 6 Chennai monitoring stations (6,184 daily
  records) with leakage-safe time-series features, chronological validation and 22 automated pytest checks.
- Trained and compared XGBoost, Random Forest and a TensorFlow/Keras LSTM against persistence baselines;
  the LSTM reduced test MAE by 40% (11.0 vs 18.3 µg/m³) and XGBoost by 29% on identical held-out targets.
- Delivered an explainable Streamlit + Plotly dashboard with TreeSHAP per-forecast explanations and a
  documented data audit exposing source-data issues (PM2.5 > PM10 in 84% of rows, 126-day filler run).
