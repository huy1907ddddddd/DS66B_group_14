"""Read the immutable UCI source and audit it before modelling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE_URL = "https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city"
ZONES = ["zone_1", "zone_2", "zone_3"]
WEATHER = ["temperature", "humidity", "wind_speed", "general_diffuse", "diffuse"]


def load_source(root: Path = ROOT) -> pd.DataFrame:
    archive = root / "data/raw/uci_tetouan.zip"
    destination = root / "data/raw/Tetuan City power consumption.csv"
    if not destination.exists():
        with ZipFile(archive) as source:
            names = [name for name in source.namelist() if name.lower().endswith(".csv")]
            if len(names) != 1:
                raise ValueError("Expected exactly one CSV in the UCI archive.")
            destination.write_bytes(source.read(names[0]))
    frame = pd.read_csv(destination)
    normalized = {column.strip().lower().replace(" ", ""): column for column in frame.columns}
    mapping = {
        "datetime": "timestamp", "temperature": "temperature", "humidity": "humidity",
        "windspeed": "wind_speed", "generaldiffuseflows": "general_diffuse",
        "diffuseflows": "diffuse", "zone1powerconsumption": "zone_1",
        "zone2powerconsumption": "zone_2", "zone3powerconsumption": "zone_3",
    }
    missing = set(mapping) - set(normalized)
    if missing:
        raise ValueError(f"Source schema changed: missing {sorted(missing)}")
    frame = frame.rename(columns={normalized[key]: value for key, value in mapping.items()})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="%m/%d/%Y %H:%M", errors="raise")
    for name in WEATHER + ZONES:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    return frame


def audit_source(frame: pd.DataFrame, root: Path = ROOT) -> dict:
    numeric = frame[WEATHER + ZONES]
    intervals = frame.timestamp.diff().dropna()
    expected = pd.date_range(frame.timestamp.min(), frame.timestamp.max(), freq="10min")
    stats = numeric.describe(percentiles=[.1, .5, .9, .95]).round(3)
    result = {
        "source": SOURCE_URL,
        "source_csv_sha256": hashlib.sha256((root / "data/raw/Tetuan City power consumption.csv").read_bytes()).hexdigest(),
        "rows": int(len(frame)), "columns": list(frame.columns),
        "start": str(frame.timestamp.min()), "end": str(frame.timestamp.max()),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_timestamps": int(frame.timestamp.duplicated().sum()),
        "timestamps_sorted": bool(frame.timestamp.is_monotonic_increasing),
        "irregular_intervals": int((intervals != pd.Timedelta(minutes=10)).sum()),
        "missing_expected_timestamps": int(len(expected.difference(pd.DatetimeIndex(frame.timestamp)))),
        "nonfinite_numeric_cells": int((~np.isfinite(numeric.to_numpy())).sum()),
        "negative_target_cells": int((frame[ZONES] < 0).sum().sum()),
        "target_units": "Not specified in the UCI variables table; use original source units.",
        "uci_metadata_instances": 52417,
        "metadata_count_note": "The CSV row count is authoritative for this experiment; UCI metadata may include the header.",
    }
    if result["missing_cells"] or result["duplicate_timestamps"] or result["nonfinite_numeric_cells"] or not result["timestamps_sorted"]:
        raise ValueError(f"Resolve source data quality issues before training: {result}")
    (root / "artifacts/data_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    stats.to_csv(root / "reports/descriptive_statistics.csv")
    return result


def make_audit_figures(frame: pd.DataFrame, root: Path = ROOT) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140})
    colors = ["#2563eb", "#059669", "#d97706"]
    indexed = frame.set_index("timestamp")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    daily = indexed[ZONES].resample("D").mean()
    for zone, color in zip(ZONES, colors):
        axes[0].plot(daily.index, daily[zone], label=zone.replace("_", " ").title(), color=color, lw=1.3)
    axes[0].set(title="Daily average demand across the three zones", ylabel="Demand (source units)")
    axes[0].legend(ncol=3, frameon=False)
    hourly = indexed[ZONES].groupby(indexed.index.hour).mean()
    for zone, color in zip(ZONES, colors):
        axes[1].plot(hourly.index, hourly[zone], label=zone.replace("_", " ").title(), color=color, marker="o", ms=3)
    axes[1].set(title="Average intraday demand pattern", xlabel="Hour in source timestamps", ylabel="Demand (source units)", xticks=range(0, 24, 2))
    fig.savefig(root / "reports/figures/data_overview.png")
    plt.close(fig)
    corr = frame[WEATHER + ZONES].corr()
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    heatmap = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [name.replace("_", " ") for name in corr.columns]
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(corr.iloc[i, j]) > .6 else "#111827")
    ax.set_title("Correlation is descriptive, not evidence of causation")
    fig.colorbar(heatmap, ax=ax, shrink=.7)
    fig.savefig(root / "reports/figures/correlation.png")
    plt.close(fig)


def main() -> None:
    frame = load_source()
    audit = audit_source(frame)
    make_audit_figures(frame)
    daily = frame.set_index("timestamp")[ZONES].resample("D").mean()
    paragraphs = [
        "# Tetouan GridWatch — Source Data Audit\n",
        "## Established facts\n",
        f"- Source CSV: {audit['rows']:,} observations and {len(frame.columns)} columns.",
        f"- Observed period: {audit['start']} to {audit['end']}.",
        f"- Missing cells: {audit['missing_cells']}; duplicate timestamps: {audit['duplicate_timestamps']}; irregular ten-minute intervals: {audit['irregular_intervals']}.",
        f"- UCI reports 52,417 instances; this file contains {audit['rows']:,} data rows. The discrepancy is documented rather than adding a fabricated record.",
        "- Input: current and historical weather, current and historical demand, and known calendar information.",
        "- Target: demand at a timestamp exactly 30 minutes after the forecast origin, separately for each zone.",
        "- Original measurement units are not documented in the UCI variables table. Charts retain source units and do not label the targets as kWh.",
        "- Timestamps have no timezone offset in the CSV. Calendar features use the source timestamps unchanged.",
        "\n## Decisions before modelling\n",
        "- Use chronological development and held-out evaluation. The split is provisional until coordinated with other groups using this dataset.",
        "- Exclude future measured weather from prediction inputs.",
        "- Define high-demand thresholds using training data only. They are research thresholds, not grid capacity limits.",
        "- Retain high but valid demand values. Peak demand is a target of interest, not automatically an outlier to delete.",
        "- This stage establishes the data and problem. It does not establish forecasting accuracy or a novel method.",
        "\n## Initial descriptive observations\n",
        f"- Zone-level daily mean ranges: " + "; ".join(f"{zone}: {daily[zone].min():,.0f}–{daily[zone].max():,.0f}" for zone in ZONES) + ".",
        "- The annual and intraday plots will inform feature design. Correlations alone do not establish causal effects of weather.",
        f"\nSource: {SOURCE_URL}\n",
        "![Annual and intraday patterns](figures/data_overview.png)",
        "![Correlations](figures/correlation.png)",
    ]
    (ROOT / "reports/Data_audit.md").write_text("\n".join(paragraphs), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
