"""An honest, fast historical replay of the frozen week-10 experiment."""

from pathlib import Path
import json
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from app.operations_panel import render_decision_lab, render_quality_lab
st.set_page_config(page_title="Tetouan GridWatch", page_icon="⚡", layout="wide")
st.markdown("""<style>
.stApp {background:#f5f7fb} h1,h2,h3 {color:#15243b}
[data-testid="stMetric"] {background:white;padding:18px;border-radius:10px;border:1px solid #e2e8f0}
[data-testid="stSidebar"] {background:#edf2f7}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load_experiment():
    summary = json.loads((ROOT / "artifacts/experiment_summary.json").read_text(encoding="utf-8"))
    replay = pd.read_csv(ROOT / "artifacts/demo_data.csv", parse_dates=["origin", "target_time"])
    return summary, replay


if not (ROOT / "artifacts/experiment_summary.json").exists():
    st.error("The experiment has not been run. Follow the README to generate verified results first.")
    st.stop()

summary, replay = load_experiment()
st.title("Tetouan GridWatch")
st.caption("Week 10 research prototype · Forecasts, decision policies and input-quality resilience · Historical data")

with st.sidebar:
    st.header("Original benchmark replay")
    zone_label = st.selectbox("Distribution zone", ["Zone 1", "Zone 2", "Zone 3"])
    zone = zone_label.lower().replace(" ", "_")
    dates = sorted(replay.origin.dt.date.unique())
    day = st.selectbox("Forecast day", dates, format_func=lambda value: value.strftime("%d %b %Y"))
    available = replay[replay.origin.dt.date == day]
    time_label = st.select_slider("Forecast issued at", options=list(available.origin.dt.strftime("%H:%M")), value=list(available.origin.dt.strftime("%H:%M"))[len(available) // 2])
    row = available[available.origin.dt.strftime("%H:%M") == time_label].iloc[0]
    show_later_observations = st.checkbox("Reveal observations recorded later", value=False)
    st.caption("Only current and earlier measurements were used to issue each forecast. Reveal later observations to inspect performance.")
    st.caption("Zone selection applies to every tab. These date/time controls affect Forecast replay only; the new laboratories have their own development-period controls.")
    st.divider()
    st.write("**Research high-demand threshold**")
    st.write(f"{row[f'{zone}_high_threshold']:,.0f} source units")
    st.caption("Training 90th percentile. It is not the grid's physical capacity.")
    st.write("**Alert operating point**")
    st.caption("Selected on validation for at most 5% false positives. The test false-positive rate can differ.")

overview, decision_lab, quality_lab, evidence, roadmap = st.tabs([
    "Forecast replay", "Decision lab", "Data quality", "Verified experiment results", "Next five weeks"])

with overview:
    st.subheader(f"{zone_label} · Forecast issued {row.origin:%d %b %Y, %H:%M}")
    st.write(f"Target time: **{row.target_time:%H:%M}**, 30 minutes after the forecast origin.")
    first, second, third = st.columns(3)
    current = row[f"{zone}_current"]
    forecast = row[f"{zone}_forecast"]
    first.metric("Observed at forecast origin", f"{current:,.0f}", help="Original measurement units; UCI does not document a unit in its variables table.")
    second.metric("Forecast for 30 minutes later", f"{forecast:,.0f}", delta=f"{forecast-current:+,.0f} from current", delta_color="off")
    direct = bool(row[f"{zone}_direct_alert"])
    alert_validated = summary["selected_classifiers"][zone] != "Always normal"
    third.metric("Direct high-demand alert", ("Attention" if direct else "No alert") if alert_validated else "Not validated")
    if not alert_validated:
        st.warning("No high-demand events occurred in this zone's validation period. Alert sensitivity cannot be estimated; the always-normal placeholder must not be treated as a reliable alert model.")
    elif direct:
        st.warning("The selected classifier flags a high-demand event at the target time. This is a research alert, not evidence of an outage or grid overload.")
    else:
        st.info("The selected classifier does not flag a high-demand event at this operating point. Missed events are measured in the results tab.")
    origin = row.origin
    past = replay[(replay.origin >= origin - pd.Timedelta(hours=6)) & (replay.origin <= origin)]
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=past.origin, y=past[f"{zone}_current"], name="Observed by forecast origin", line={"color":"#64748b","width":2}))
    chart.add_trace(go.Scatter(x=[origin, row.target_time], y=[current, forecast], name="30-minute forecast", line={"color":"#2563eb","dash":"dash","width":3}, mode="lines+markers"))
    chart.add_hline(y=row[f"{zone}_high_threshold"], line_dash="dot", line_color="#d97706", annotation_text="Training high-demand threshold")
    if show_later_observations:
        later = replay[(replay.origin > origin) & (replay.origin <= row.target_time)]
        chart.add_trace(go.Scatter(x=later.origin, y=later[f"{zone}_current"], name="Revealed later measurements", line={"color":"#059669","width":2}, mode="lines+markers"))
        st.success(f"Later observation at the target time: {row[f'{zone}_actual']:,.0f}. Absolute forecast error: {abs(row[f'{zone}_actual']-forecast):,.0f} source units.")
    chart.update_layout(height=390, template="plotly_white", yaxis_title="Demand (source units)", xaxis_title="Source timestamp", legend={"orientation":"h","y":1.12}, margin={"t":50,"b":40,"l":50,"r":30})
    st.plotly_chart(chart, width="stretch")
    with st.expander("What can and cannot be claimed at this stage"):
        st.write("The original benchmark replays historical test observations and uses forecasts computed before seeing their targets. The Decision lab and Data quality tabs add separate development experiments within validation. Basic rule checks and recent-measurement fallbacks are implemented there. Adaptive regimes, calibrated forecast intervals and advanced anomaly detectors remain planned.")

with decision_lab:
    render_decision_lab(ROOT, zone)

with quality_lab:
    render_quality_lab(ROOT, zone)

with evidence:
    st.caption("Frozen initial benchmark. New decision and basic quality experiments are reported in their own tabs.")
    st.subheader("Model choice was fixed using validation")
    validation = pd.read_csv(ROOT / "reports/regression_validation.csv")
    st.dataframe(validation[validation.zone == "average"][["model", "mae", "rmse", "r2", "train_seconds"]].sort_values("mae"), hide_index=True, width="stretch")
    st.write(f"Selected forecast model: **{summary['selected_regressor']}**")
    st.subheader("Held-out forecasting results")
    test_table = pd.read_csv(ROOT / "reports/regression_test.csv")
    st.dataframe(test_table[["model", "zone", "mae", "rmse", "r2", "peak_mae", "peak_samples"]], hide_index=True, width="stretch")
    st.subheader("High-demand alerts: misses and false alarms")
    alerts = pd.read_csv(ROOT / "reports/alerts_test.csv")
    st.dataframe(alerts[alerts.zone == zone][["strategy", "precision", "recall", "false_positive_rate", "tp", "fp", "fn", "tn"]], hide_index=True, width="stretch")
    st.caption("Both score-based operating points target the same 5% validation false-positive budget. They need not meet that budget on a later test period. A high-load threshold comparator is reported separately.")
    st.subheader("Does each component help?")
    ablations = pd.read_csv(ROOT / "reports/ablation_validation.csv")
    st.dataframe(ablations[ablations.zone == "average"][["model", "mae", "rmse", "r2"]], hide_index=True, width="stretch")
    st.caption("Ablations were run on validation, with identical sample boundaries. This initial benchmark uses one chronological holdout.")
    for limit in summary["limitations"]:
        st.write(f"• {limit}")

with roadmap:
    st.subheader("Planned modules have no fabricated results")
    st.dataframe(pd.DataFrame([
        {"Course chapter":"11", "Module":"Small MLP/LSTM benchmark", "Status":"Planned"},
        {"Course chapter":"12", "Module":"Regime clustering and adaptive model selection", "Status":"Planned"},
        {"Course chapter":"13", "Module":"Awaiting the course material", "Status":"Pending material"},
        {"Course chapter":"14", "Module":"Feature selection / PCA and runtime comparison", "Status":"Planned"},
        {"Course chapter":"15", "Module":"Advanced anomaly detector; basic rule gate already implemented", "Status":"Advanced methods planned"},
    ]), hide_index=True, width="stretch")
    st.write("Each new module must outperform an appropriate comparator under the same forecasting setup, or be documented as an unsuccessful hypothesis. No automatic claim of novelty or improvement is made.")

st.caption("Source: UCI Power Consumption of Tetouan City · Split provisional until coordinated with peer groups · Original measurement units retained")
