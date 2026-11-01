"""Chennai Air Quality Intelligence — Streamlit dashboard.

Reads only saved artifacts (processed data, saved predictions, metrics, XGBoost model).
No model is trained at startup and TensorFlow is not needed at runtime.
"""
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import DATA_PROCESSED, FIGURES_DIR, METRICS_DIR, POLLUTANTS, PREDICTIONS_DIR, RESULTS_DIR, STATIONS
from src.inference import forecast_from_daily, load_xgboost

GITHUB_URL = "https://github.com/"  # replaced with the repository URL once published

MODEL_LABELS = {
    "persistence": "Persistence (baseline)",
    "seasonal_naive": "Seasonal naive, 7-day (baseline)",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "lstm": "LSTM (deep learning)",
}
MODEL_COLORS = {"persistence": "#9aa5b1", "seasonal_naive": "#c3cad3", "xgboost": "#1f6f9f",
                "random_forest": "#5fa8d3", "lstm": "#e07a2f"}
STATION_COLORS = dict(zip(STATIONS, px.colors.qualitative.Safe))

st.set_page_config(page_title="Chennai Air Quality Intelligence", page_icon="🌫️", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1250px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  div[data-testid="stMetric"] {background: var(--secondary-background-color); border-radius: 10px;
      padding: 0.8rem 1rem; border: 1px solid rgba(128,128,128,0.15);}
  .caqi-note {border-left: 4px solid #e07a2f; padding: 0.6rem 1rem; border-radius: 6px;
      background: rgba(224,122,47,0.08); margin: 0.5rem 0 1rem 0;}
</style>
""", unsafe_allow_html=True)


# ---------- data ----------
@st.cache_data
def load_daily() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / "daily.csv", parse_dates=["date"])


@st.cache_data
def load_predictions() -> pd.DataFrame:
    frames = [pd.read_csv(PREDICTIONS_DIR / f"{m}.csv", parse_dates=["origin_date", "target_date"])
              for m in MODEL_LABELS if (PREDICTIONS_DIR / f"{m}.csv").exists()]
    return pd.concat(frames, ignore_index=True)


@st.cache_data
def load_csv(path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_shap() -> pd.DataFrame | None:
    p = RESULTS_DIR / "explainability" / "shap_values_val_test.csv.gz"
    return pd.read_csv(p, parse_dates=["origin_date", "target_date"]) if p.exists() else None


@st.cache_resource
def xgb_model():
    return load_xgboost()


def note(text: str) -> None:
    st.markdown(f'<div class="caqi-note">{text}</div>', unsafe_allow_html=True)


def fig_layout(fig, height=420):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), legend_title_text="",
                      hovermode="x unified")
    return fig


DATA_NOTE = ("<b>Historical backtest only.</b> The dataset ends on <b>31 December 2024</b>. Every forecast "
             "shown here is a next-day prediction for a past date, made using only data available up to the "
             "day before. This is not a live forecasting service.")

daily = load_daily()
preds = load_predictions()
overall_test = load_csv(METRICS_DIR / "overall_test.csv")
counts = json.loads((METRICS_DIR / "dataset_counts.json").read_text())

st.sidebar.title("🌫️ Chennai AQI")
st.sidebar.caption("Explainable next-day PM2.5 forecasting")
page = st.sidebar.radio("Navigate", ["Overview", "Historical Air Quality", "Forecast Explorer",
                                     "Model Comparison", "Explainability", "About"])
st.sidebar.divider()
st.sidebar.caption("Data: Jan 2022 – Dec 2024 · 6 stations\n\nTest period: Jul – Dec 2024")


# ---------- pages ----------
def page_overview():
    st.title("Chennai Air Quality Intelligence")
    st.markdown("##### Explainable multi-station PM2.5 forecasting using machine learning and deep learning")
    note(DATA_NOTE)
    c = st.columns(5)
    c[0].metric("Coverage", "2022–24")
    c[1].metric("Monitoring stations", len(STATIONS))
    c[2].metric("Station-days", f"{int(daily.observed.sum()):,}")
    c[3].metric("Forecast samples", f"{counts['eligible_origins']:,}")
    c[4].metric("Models compared", preds.model.nunique())

    st.subheader("What this project does")
    st.markdown("""
At the end of each day *t*, the system predicts the **next day's (t+1) PM2.5 concentration** at six
Chennai monitoring stations — Alandur, Kodungaiyur, Manali, Perungudi, Royapuram and Velachery —
using only pollutant observations available up to day *t*.

* **Baselines:** persistence and 7-day seasonal-naive
* **Machine learning:** XGBoost (primary) and Random Forest
* **Deep learning:** station-aware LSTM (TensorFlow/Keras)
* **Evaluation:** chronological split — train 2022–2023, validation Jan–Jun 2024, test Jul–Dec 2024
* **Explainability:** gain importance and TreeSHAP for XGBoost
""")
    st.subheader("Test-period results (identical targets for all models)")
    t = overall_test.copy()
    t["model"] = t["model"].map(MODEL_LABELS)
    st.dataframe(t[["model", "n", "MAE", "RMSE", "R2", "sMAPE", "MAE_improvement_vs_persistence_%"]]
                 .rename(columns={"MAE_improvement_vs_persistence_%": "MAE gain vs persistence (%)"})
                 .style.format(precision=2), hide_index=True, width="stretch")

    st.subheader("Important dataset limitations")
    audit = json.loads((METRICS_DIR / "audit_summary.json").read_text())
    st.markdown(f"""
* **{audit['missing_calendar_days']:,} missing station-days** on the 2022–2024 calendar, including long gaps at some stations.
* **PM2.5 exceeds PM10 in {audit['pm25_gt_pm10_pct']}% of records** — physically implausible, indicating
  measurement or processing issues in the source data.
* Repeated constant readings and apparent instrument floors (especially SO₂ / O₃).
* The publisher does not state whether values were cleaned or imputed, so this is not raw sensor data.
* No meteorological variables (wind, rain, humidity) are available, which limits achievable accuracy.
""")


def page_history():
    st.title("Historical Air Quality")
    note("Gaps in the lines are days with no record in the source dataset — they are not filled in.")
    c1, c2, c3 = st.columns([2, 2, 1])
    stations = c1.multiselect("Stations", STATIONS, default=["Alandur", "Velachery"])
    dmin, dmax = daily.date.min().date(), daily.date.max().date()
    rng = c2.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
    pol = c3.selectbox("Pollutant", POLLUTANTS)
    if not stations or len(rng) != 2:
        st.info("Select at least one station and a start and end date.")
        return
    d = daily[daily.station.isin(stations) & daily.date.between(pd.Timestamp(rng[0]), pd.Timestamp(rng[1]))]
    smooth = st.toggle("Show 7-day trailing mean", value=False)
    plot_d = d.copy()
    if smooth:
        plot_d[pol] = plot_d.groupby("station")[pol].transform(lambda s: s.rolling(7, min_periods=4).mean())
    fig = px.line(plot_d, x="date", y=pol, color="station", color_discrete_map=STATION_COLORS,
                  title=f"Daily {pol}" + (" (7-day trailing mean)" if smooth else ""))
    st.plotly_chart(fig_layout(fig), width="stretch")

    a, b = st.columns(2)
    m = d.assign(month=d.date.dt.to_period("M").dt.to_timestamp()).groupby(["month", "station"])[pol].mean().reset_index()
    a.plotly_chart(fig_layout(px.bar(m, x="month", y=pol, color="station", barmode="group",
                                     color_discrete_map=STATION_COLORS, title=f"Monthly mean {pol}"), 380),
                   width="stretch")
    b.plotly_chart(fig_layout(px.box(d, x="station", y=pol, color="station", color_discrete_map=STATION_COLORS,
                                     title=f"{pol} distribution"), 380).update_layout(showlegend=False),
                   width="stretch")

    st.subheader("All pollutants for one station")
    s = st.selectbox("Station", stations)
    long = d[d.station == s].melt(id_vars="date", value_vars=POLLUTANTS, var_name="pollutant")
    st.plotly_chart(fig_layout(px.line(long, x="date", y="value", color="pollutant", title=s)), width="stretch")
    st.dataframe(d[d.station == s][POLLUTANTS].describe().T.style.format(precision=1), width="stretch")


def page_forecast():
    st.title("Forecast Explorer")
    note(DATA_NOTE)
    c1, c2, c3 = st.columns(3)
    station = c1.selectbox("Monitoring station", STATIONS)
    available = [m for m in MODEL_LABELS if m in preds.model.unique()]
    model = c2.selectbox("Model", available, index=available.index("xgboost") if "xgboost" in available else 0,
                         format_func=MODEL_LABELS.get)
    split = c3.selectbox("Period", ["test", "val"], format_func={"test": "Test (Jul–Dec 2024)",
                                                                 "val": "Validation (Jan–Jun 2024)"}.get)
    p = preds[(preds.model == model) & (preds.station == station) & (preds.split == split)].sort_values("origin_date")
    if p.empty:
        st.warning("No saved forecasts for this selection.")
        return
    dates = p.origin_date.dt.date.tolist()
    origin = st.select_slider("Forecast made at the end of (origin date)", options=dates, value=dates[len(dates) // 2])
    row = p[p.origin_date.dt.date == origin].iloc[0]
    pers = preds[(preds.model == "persistence") & (preds.station == station) & (preds.origin_date == row.origin_date)]

    k = st.columns(4)
    k[0].metric(f"Target date ({row.target_date.year})", f"{row.target_date.day} {row.target_date:%b}")
    k[1].metric(f"Predicted PM2.5 ({MODEL_LABELS[model].split(' (')[0]})", f"{row.y_pred:.1f}")
    k[2].metric("Actual PM2.5", f"{row.y_true:.1f}", delta=f"error {row.y_pred - row.y_true:+.1f}", delta_color="off")
    if not pers.empty:
        k[3].metric("Persistence forecast", f"{pers.y_pred.iloc[0]:.1f}")

    hist = daily[(daily.station == station) & daily.date.between(row.origin_date - pd.Timedelta(days=21), row.target_date)]
    fig = go.Figure()
    known = hist[hist.date <= row.origin_date]
    fig.add_scatter(x=known.date, y=known["PM2.5"], mode="lines+markers", name="observed (known at origin)",
                    line=dict(color="#6c7a89"))
    fig.add_scatter(x=[row.target_date], y=[row.y_true], mode="markers", name="actual next day",
                    marker=dict(size=13, color="#2b2d42", symbol="circle-open", line=dict(width=3)))
    fig.add_scatter(x=[row.target_date], y=[row.y_pred], mode="markers", name=f"{MODEL_LABELS[model]} forecast",
                    marker=dict(size=13, color=MODEL_COLORS[model], symbol="diamond"))
    fig.add_vline(x=row.origin_date, line_dash="dot", line_color="grey")
    fig.update_layout(title="Three weeks of context and the next-day forecast", yaxis_title="PM2.5")
    st.plotly_chart(fig_layout(fig, 380), width="stretch")

    if model == "xgboost":
        with st.expander("Live XGBoost re-computation and explanation", expanded=True):
            m, schema = xgb_model()
            live = forecast_from_daily(daily, station, row.origin_date, m, schema)
            st.caption(f"Features rebuilt from data up to {row.origin_date:%d %b %Y} only → live prediction "
                       f"**{live['prediction']:.2f}** (saved: {row.y_pred:.2f}).")
            shap = load_shap()
            if shap is not None:
                srow = shap[(shap.station == station) & (shap.origin_date == row.origin_date)]
                if not srow.empty:
                    contrib = srow[schema["features"]].iloc[0].sort_values(key=abs, ascending=False).head(10)[::-1]
                    vals = pd.Series(live["features"])[contrib.index]
                    labels = [f"{f} = {vals[f]:.1f}" if pd.notna(vals[f]) else f"{f} = missing" for f in contrib.index]
                    fig = go.Figure(go.Bar(x=contrib.values, y=labels, orientation="h",
                                           marker_color=["#c0392b" if v > 0 else "#2a6f97" for v in contrib.values]))
                    unit = "log(1+PM2.5)" if schema["target_transform"] == "log1p" else "PM2.5"
                    fig.update_layout(title=f"Top SHAP contributions for this forecast ({unit} scale)",
                                      xaxis_title="contribution (red pushes forecast up, blue down)")
                    st.plotly_chart(fig_layout(fig, 400).update_layout(hovermode="closest"), width="stretch")

    st.subheader(f"All {split} forecasts — {station}")
    fig = go.Figure()
    fig.add_scatter(x=p.target_date, y=p.y_true, name="actual", line=dict(color="#2b2d42", width=1.5))
    fig.add_scatter(x=p.target_date, y=p.y_pred, name=MODEL_LABELS[model], line=dict(color=MODEL_COLORS[model], width=1.5))
    st.plotly_chart(fig_layout(fig, 360), width="stretch")


def page_comparison():
    st.title("Model Comparison")
    split = st.radio("Evaluation period", ["test", "val"], horizontal=True,
                     format_func={"test": "Test (Jul–Dec 2024)", "val": "Validation (Jan–Jun 2024)"}.get)
    ov = load_csv(METRICS_DIR / f"overall_{split}.csv")
    ps = load_csv(METRICS_DIR / f"per_station_{split}.csv")
    note(f"All models are scored on the <b>same {int(ov.n.iloc[0]):,} target days</b> (station-days every model "
         "could forecast). Lower MAE/RMSE/sMAPE is better; higher R² is better.")
    ov_l = ov.assign(label=ov.model.map(MODEL_LABELS))
    st.dataframe(ov_l[["label", "MAE", "RMSE", "R2", "sMAPE", "MAE_improvement_vs_persistence_%",
                       "RMSE_improvement_vs_persistence_%"]].rename(columns={"label": "model"})
                 .style.format(precision=2).highlight_min(subset=["MAE", "RMSE", "sMAPE"], color="#d4edda")
                 .highlight_max(subset=["R2"], color="#d4edda"), hide_index=True, width="stretch")

    a, b = st.columns(2)
    long = ov_l.melt(id_vars=["model", "label"], value_vars=["MAE", "RMSE"], var_name="metric")
    a.plotly_chart(fig_layout(px.bar(long, x="label", y="value", color="model", facet_col="metric",
                                     color_discrete_map=MODEL_COLORS, title="Error by model")
                              .for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                              .update_layout(showlegend=False).update_xaxes(title=""), 400), width="stretch")
    b.plotly_chart(fig_layout(px.bar(ov_l, x="label", y="R2", color="model", color_discrete_map=MODEL_COLORS,
                                     title="R² by model").update_layout(showlegend=False).update_xaxes(title=""), 400),
                   width="stretch")

    st.subheader("Station-wise performance")
    metric = st.selectbox("Metric", ["MAE", "RMSE", "R2", "sMAPE"])
    st.plotly_chart(fig_layout(px.bar(ps.assign(label=ps.model.map(MODEL_LABELS)), x="station", y=metric, color="model",
                                      barmode="group", color_discrete_map=MODEL_COLORS,
                                      title=f"{metric} per station").for_each_trace(
        lambda t: t.update(name=MODEL_LABELS.get(t.name, t.name))), 420), width="stretch")

    st.subheader("Actual vs predicted")
    common = pd.read_csv(PREDICTIONS_DIR / f"common_{split}.csv.gz", parse_dates=["target_date"])
    c1, c2 = st.columns([1, 2])
    station = c1.selectbox("Station", STATIONS, key="cmp_station")
    models = c2.multiselect("Models", list(ov.model), default=["persistence", "xgboost", "lstm"],
                            format_func=MODEL_LABELS.get)
    cs = common[common.station == station]
    fig = go.Figure()
    act = cs[cs.model == "persistence"].sort_values("target_date")
    fig.add_scatter(x=act.target_date, y=act.y_true, name="actual", line=dict(color="#2b2d42", width=2))
    for m in models:
        g = cs[cs.model == m].sort_values("target_date")
        fig.add_scatter(x=g.target_date, y=g.y_pred, name=MODEL_LABELS[m], line=dict(color=MODEL_COLORS[m], width=1.3))
    st.plotly_chart(fig_layout(fig, 420), width="stretch")

    a, b = st.columns(2)
    sel = common[common.model.isin(models)].assign(residual=lambda d: d.y_true - d.y_pred,
                                                    label=lambda d: d.model.map(MODEL_LABELS))
    sc = px.scatter(sel, x="y_true", y="y_pred", color="model", opacity=0.45, color_discrete_map=MODEL_COLORS,
                    title="Predicted vs actual (all stations)", labels={"y_true": "actual", "y_pred": "predicted"})
    mx = float(sel[["y_true", "y_pred"]].max().max())
    sc.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dash", color="grey"), name="perfect")
    a.plotly_chart(fig_layout(sc, 420).update_layout(hovermode="closest"), width="stretch")
    b.plotly_chart(fig_layout(px.histogram(sel, x="residual", color="model", barmode="overlay", opacity=0.55, nbins=60,
                                           color_discrete_map=MODEL_COLORS, title="Residuals (actual − predicted)"), 420)
                   .update_layout(hovermode="closest"), width="stretch")

    if (FIGURES_DIR / "lstm_loss_curve.png").exists():
        with st.expander("LSTM training and validation loss"):
            st.image(str(FIGURES_DIR / "lstm_loss_curve.png"))


def page_explain():
    st.title("Explainability")
    note("Feature importance and SHAP describe <b>how the XGBoost model uses its inputs</b>. They are not "
         "evidence of causal relationships between pollutants.")
    imp = load_csv(METRICS_DIR / "xgb_feature_importance.csv")
    shap_mean = load_csv(METRICS_DIR / "shap_mean_abs_test.csv")
    a, b = st.columns(2)
    top = imp.head(15).iloc[::-1]
    a.plotly_chart(fig_layout(px.bar(top, x="gain_share", y="feature", orientation="h",
                                     title="Gain importance (share of total)", color_discrete_sequence=["#1f6f9f"]), 480)
                   .update_layout(hovermode="closest"), width="stretch")
    top = shap_mean.head(15).iloc[::-1]
    b.plotly_chart(fig_layout(px.bar(top, x="mean_abs_shap", y="feature", orientation="h",
                                     title="Mean |SHAP| on the test period", color_discrete_sequence=["#e07a2f"]), 480)
                   .update_layout(hovermode="closest"), width="stretch")
    if (FIGURES_DIR / "shap_summary.png").exists():
        st.subheader("SHAP summary (beeswarm)")
        st.image(str(FIGURES_DIR / "shap_summary.png"), width=760)
        st.caption("Each dot is one test-period forecast. Colour = feature value (red high, blue low); "
                   "x-position = how much that feature pushed the forecast up or down.")
    st.subheader("Interpretation")
    st.markdown((RESULTS_DIR / "explainability" / "interpretation.md").read_text(encoding="utf-8")
                if (RESULTS_DIR / "explainability" / "interpretation.md").exists() else "")


def page_about():
    st.title("About")
    st.markdown(f"""
### Dataset
*Hybrid Prophet-XGBoost Forecasting for Urban Air Quality with Real-Time Data Integration for Chennai*,
Mendeley Data, DOI [10.17632/br5dyxm7c6.1](https://data.mendeley.com/datasets/br5dyxm7c6/1).
Only the six historical station CSVs (2022–2024) are used; their SHA-256 hashes are verified on every
pipeline run. The original archive's code, models and reported metrics were **not** reused.

### Methodology
1. **Audit** — coverage, gaps, duplicates, outliers, PM2.5 > PM10, constant runs (documented, not altered).
2. **Preprocessing** — each station reindexed to a full daily calendar; no backward fill, interpolation or
   centred windows; labels never imputed.
3. **Features** (origin day *t* → target *t+1*) — PM2.5 today and lags 1/2/3/6/7/14, trailing rolling
   means (3/7/14) and std (7), same-day PM10/NO₂/SO₂/O₃, target-day calendar, station one-hot.
4. **Split by target date** — train 2022–2023, validation Jan–Jun 2024, test Jul–Dec 2024.
5. **Models** — persistence, 7-day seasonal naive, XGBoost, Random Forest and a station-aware LSTM.
   Hyper-parameters chosen on validation only; the test set was used once for final scoring.
6. **Explainability** — XGBoost gain importance and exact TreeSHAP.

### Limitations
* No meteorological inputs; only five pollutants are available.
* Source data quality issues (PM2.5 > PM10, constant runs, gaps; undocumented pre-processing by the publisher).
* Results are a historical backtest; the data ends in December 2024.
* Forecasts are point estimates without uncertainty intervals.

### Technology stack
Python 3.11 · pandas · NumPy · scikit-learn · XGBoost · TensorFlow/Keras · SHAP · Plotly · Streamlit · pytest

### Source code
[{GITHUB_URL}]({GITHUB_URL})
""")


{"Overview": page_overview, "Historical Air Quality": page_history, "Forecast Explorer": page_forecast,
 "Model Comparison": page_comparison, "Explainability": page_explain, "About": page_about}[page]()
