"""Show why an alert policy changes and how input faults affect forecasts."""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gridwatch.decision_support import choose_policy
from gridwatch.experiment import alert_metrics
from gridwatch.quality import FAULT_SCENARIOS


@st.cache_data
def load_operations(root, signature):
    summary = json.loads((root / "artifacts/operations_summary.json").read_text(encoding="utf-8"))
    candidates = pd.read_csv(root / "artifacts/policy_candidates.csv", float_precision="round_trip")
    policy = pd.read_csv(root / "artifacts/policy_replay.csv", parse_dates=["origin", "target_time"], float_precision="round_trip")
    quality = pd.read_csv(root / "artifacts/quality_replay.csv", parse_dates=["origin", "target_time"], float_precision="round_trip")
    quality_metrics = pd.read_csv(root / "reports/quality_assessment.csv")
    return summary, candidates, policy, quality, quality_metrics


def operations_data(root):
    files = [root / "artifacts" / name for name in
             ["operations_summary.json", "policy_candidates.csv", "policy_replay.csv", "quality_replay.csv"]]
    files.append(root / "reports/quality_assessment.csv")
    if not all(path.exists() for path in files):
        st.info("Run .\\run_operations.ps1 to prepare the decision and data-quality laboratories.")
        return None
    return load_operations(root, tuple(path.stat().st_mtime_ns for path in files))


def render_decision_lab(root, zone):
    st.subheader("From forecasts to a decision worth making")
    st.write("An operator may prepare agreed flexible loads or available storage before high demand. "
             "This laboratory studies the alert decision; it does not control equipment or estimate real savings.")
    data = operations_data(root)
    if data is None:
        return
    summary, candidates, replay, _, _ = data
    st.caption(f"Earlier calibration: {summary['calibration_rows']:,} samples; later assessment: "
               f"{summary['assessment_rows']:,}; boundary: {summary['boundary']}; {summary['purged_rows']} boundary samples excluded. "
               "These are development windows within previously inspected validation data. Original test results are unchanged.")
    if not summary["zones"][zone]["policy_supported"]:
        st.warning("No high-demand calibration examples for this zone. A useful alert threshold cannot be validated; "
                   "no automatic alert policy is offered. You can still inspect the data-quality laboratory.")
        return
    first, second, third = st.columns(3)
    model_name = first.selectbox("Alert score", ["Logistic", "Linear score"], key="policy_model")
    miss_penalty = second.number_input("Penalty per missed high sample", min_value=1, max_value=1000, value=20, key="miss_penalty")
    false_penalty = third.number_input("Penalty per false alarm", min_value=1, max_value=1000, value=1, key="false_penalty")
    st.caption("Hypothetical penalty points per 10-minute sample, not money or independent peak episodes. "
               "Logistic scores are not verified calibrated probabilities.")
    options = candidates[(candidates.zone == zone) & (candidates.model == model_name)]
    selected = choose_policy(options, miss_penalty, false_penalty)
    assessed = replay[(replay.zone == zone) & (replay.model == model_name) & (replay.period == "assessment")]
    counts = alert_metrics(assessed.actual_high, assessed.score >= selected.threshold)
    penalty = miss_penalty * counts["fn"] + false_penalty * counts["fp"]
    a, b, c = st.columns(3)
    a.metric("Later assessment misses", counts["fn"])
    b.metric("Later assessment false alarms", counts["fp"])
    c.metric("Later assessment penalty points", f"{penalty:,}")
    st.write(f"**Penalty = {miss_penalty} × misses + {false_penalty} × false alarms.** "
             f"The earlier calibration selected threshold **{selected.threshold:.8g}**, "
             f"with {int(selected.fn)} misses and {int(selected.fp)} false alarms.")
    fixed_cutoff = .5 if model_name == "Logistic" else summary["zones"][zone]["high_demand_threshold"]
    rows = []
    for strategy, decisions in {
        "Cost-selected threshold": assessed.score >= selected.threshold,
        "Fixed 0.5 / training high threshold": assessed.score >= fixed_cutoff,
        "Always normal": np.zeros(len(assessed), dtype=bool),
        "Always alert": np.ones(len(assessed), dtype=bool),
    }.items():
        result = alert_metrics(assessed.actual_high, decisions)
        rows.append({"Policy": strategy, "Misses": result["fn"], "False alarms": result["fp"],
                     "Penalty points": miss_penalty * result["fn"] + false_penalty * result["fp"],
                     "Recall": result["recall"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=options.fp, y=options.fn, mode="lines", name="Earlier calibration choices"))
    chart.add_trace(go.Scatter(x=[selected.fp], y=[selected.fn], mode="markers", name="Chosen earlier",
                              marker={"size":13,"color":"#d97706"}))
    chart.update_layout(height=290, template="plotly_white", xaxis_title="False alarms on calibration",
                        yaxis_title="Misses on calibration", margin={"t":30,"b":40})
    st.plotly_chart(chart, width="stretch")
    with st.expander("Replay one real decision", expanded=True):
        labels = list(assessed.origin.dt.strftime("%d %b %H:%M"))
        sample_label = st.select_slider("Forecast origin in later assessment", options=labels,
                                        value=labels[0], key="policy_sample")
        sample = assessed[assessed.origin.dt.strftime("%d %b %H:%M") == sample_label].iloc[0]
        alert = sample.score >= selected.threshold
        st.write(f"At **{sample.origin:%d %b %H:%M}**, current demand is **{sample.current:,.2f}**; "
                 f"Linear forecasts **{sample.forecast:,.2f}** for **{sample.target_time:%H:%M}** (source units).")
        st.write(f"{model_name} score **{sample.score:.8g}** versus selected cutoff **{selected.threshold:.8g}**.")
        if alert:
            st.warning("Prepare an operator review of agreed flexible loads or available storage. "
                       "This high-demand alert does not establish overload and does not authorize a shutdown.")
        else:
            st.info("No preparation alert at this operating point. Continue monitoring; misses remain possible.")
        if st.checkbox("Reveal the later actual demand", key="policy_reveal"):
            st.write(f"Later measurement: **{sample.actual:,.2f}**; high-demand label: **{int(sample.actual_high)}**. "
                     f"Original CSV row at forecast origin: **{int(sample.source_row_number)}**.")
    st.caption("Changing the assumptions changes the preferred tradeoff. Lower calibration penalty does not guarantee "
               "lower later penalty. Model comparisons shown here are development evidence.")


def render_quality_lab(root, zone):
    st.subheader("A forecast service that exposes input problems")
    data = operations_data(root)
    if data is None:
        return
    _, _, _, replay, metrics = data
    st.write("The basic gate checks every required value and monitors sensor inputs against training-only envelopes. "
             "It uses a recent acceptable measurement if the model cannot be used, or explicitly reports no forecast.")
    scenario = st.selectbox("Prepared input-packet scenario", FAULT_SCENARIOS, key="quality_scenario")
    st.caption("Fault scenarios are synthetic edits to already prepared feature packets. They do not simulate an entire "
               "sensor outage and history reconstruction. Original-envelope breaches are not proof of sensor faults.")
    table = metrics[(metrics.zone == zone) & (metrics.scenario == scenario)]
    shown = table[["strategy", "available_rows", "coverage", "mae_available", "fallback_rows"]].copy()
    shown.columns = ["Strategy", "Available predictions", "Coverage", "MAE on available predictions", "Fallbacks"]
    st.dataframe(shown, hide_index=True, width="stretch")
    st.caption("Compare accuracy together with coverage. An unavailable forecast has no MAE; it does not have zero error.")
    samples = replay[(replay.zone == zone) & (replay.scenario == scenario)]
    labels = list(samples.origin.dt.strftime("%d %b %H:%M"))
    sample_label = st.select_slider("Inspect a real origin with this simulated packet", options=labels,
                                   value=labels[0], key="quality_sample")
    sample = samples[samples.origin.dt.strftime("%d %b %H:%M") == sample_label].iloc[0]
    display = lambda value: f"{value:,.2f}" if pd.notna(value) else "Unavailable"
    st.write(f"Origin **{sample.origin:%d %b %H:%M}** → target **{sample.target_time:%H:%M}**. "
             f"Received current demand: **{display(sample.current_received)}**; "
             f"temperature: **{display(sample.temperature_received)}**; "
             f"10-minute-old zone measurement: **{display(sample.previous_measurement)}**.")
    a, b = st.columns(2)
    a.metric("Forecast without gate", display(sample.unguarded_forecast))
    b.metric("Forecast with gate", display(sample.guarded_forecast))
    st.write(f"**Method:** {sample.method}. **Input check:** {sample.reason}.")
    if sample.method == "Unavailable":
        st.error("No acceptable recent measurement. Request data verification; no forecast or model alert is issued.")
    elif sample.method != "Linear model":
        st.warning("Fallback mode: a simple recent-measurement forecast is shown for review. "
                   "Direct classifier alerts are withheld for this faulty packet; forecast accuracy can worsen.")
    else:
        st.success("Input checks passed; the Linear forecast is used.")
    if st.checkbox("Reveal later measurement and forecast error", key="quality_reveal"):
        st.write(f"Later measurement: **{sample.actual:,.2f}** source units.")
        if pd.notna(sample.guarded_forecast):
            st.write(f"Absolute error of the guarded forecast: **{abs(sample.actual - sample.guarded_forecast):,.2f}**.")
    with st.expander("Measured quality results for all four scenarios"):
        st.dataframe(metrics[metrics.zone == zone], hide_index=True, width="stretch")
    st.caption("Envelope = training minimum/maximum plus 25% of their range, with elementary physical bounds. "
               "The fixed 25% margin is an engineering assumption. Advanced anomaly detection remains planned for chapter 15.")
