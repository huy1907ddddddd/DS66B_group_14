"""A basic input-quality gate with explicit, recent-measurement fallbacks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import WEATHER, ZONES


def fit_quality_bounds(training, margin=.25):
    """Training-only envelopes; the margin is a fixed engineering assumption."""
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("The envelope margin must be nonnegative and finite.")
    columns = WEATHER + ZONES + [f"{zone}_lag_1" for zone in ZONES]
    observed = training[columns]
    if not np.isfinite(observed.to_numpy()).all():
        raise ValueError("Quality bounds require finite training measurements.")
    minimum, maximum = observed.min(), observed.max()
    span = (maximum - minimum).clip(lower=1e-9)
    bounds = pd.DataFrame({"lower": minimum - margin * span, "upper": maximum + margin * span})
    nonnegative = [name for name in columns if name != "temperature"]
    bounds.loc[nonnegative, "lower"] = bounds.loc[nonnegative, "lower"].clip(lower=0)
    bounds.loc["humidity", "upper"] = min(100., bounds.loc["humidity", "upper"])
    return bounds


def guarded_forecast(model, features, bounds):
    """Never send invalid packets into the model or reuse a flagged backup."""
    expected = list(model.feature_names_in_)
    if list(features.columns) != expected:
        raise ValueError("Input feature names and order must match the fitted model.")
    finite = np.isfinite(features.to_numpy(dtype=float))
    monitor_columns = WEATHER + ZONES + [f"{zone}_lag_1" for zone in ZONES]
    values = features[monitor_columns]
    within = values.ge(bounds.loc[monitor_columns, "lower"], axis=1) & values.le(bounds.loc[monitor_columns, "upper"], axis=1)
    acceptable = finite.all(axis=1) & within.all(axis=1).to_numpy()
    forecasts = np.full((len(features), len(ZONES)), np.nan)
    methods = np.full(forecasts.shape, "Unavailable", dtype=object)
    reasons = np.full(len(features), "Input checks passed", dtype=object)
    for position in np.flatnonzero(~acceptable):
        missing = list(features.columns[~finite[position]])
        outside = [name for name in monitor_columns
                   if np.isfinite(values.iloc[position][name]) and not within.iloc[position][name]]
        details = []
        if missing:
            details.append("Missing/nonfinite: " + ", ".join(missing))
        if outside:
            details.append("Outside training/physical envelope: " + ", ".join(outside))
        reasons[position] = "; ".join(details)
    if acceptable.any():
        predictions = np.maximum(model.predict(features.loc[acceptable]), 0)
        positions = np.flatnonzero(acceptable)
        model_finite = np.isfinite(predictions).all(axis=1)
        forecasts[positions[model_finite]] = predictions[model_finite]
        methods[positions[model_finite]] = "Linear model"
        acceptable[positions[~model_finite]] = False
        reasons[positions[~model_finite]] = "Model returned a nonfinite forecast"
    blocked = ~acceptable
    for zone_index, zone in enumerate(ZONES):
        for source_column, method in [(zone, "Persistence: current"),
                                      (f"{zone}_lag_1", "Persistence: 10 minutes earlier")]:
            measurements = features[source_column].to_numpy(dtype=float)
            safe = (np.isfinite(measurements)
                    & (measurements >= bounds.loc[source_column, "lower"])
                    & (measurements <= bounds.loc[source_column, "upper"]))
            use = blocked & ~np.isfinite(forecasts[:, zone_index]) & safe
            forecasts[use, zone_index] = measurements[use]
            methods[use, zone_index] = method
    return forecasts, methods, reasons


def fault_packet(features, scenario):
    """Synthetic faults in prepared feature packets, not a raw-sensor simulation."""
    packet = features.copy()
    if scenario == "Clean packets":
        pass
    elif scenario == "Temperature missing":
        packet["temperature"] = np.nan
    elif scenario == "Zone 1 current multiplied by 10":
        packet["zone_1"] *= 10
    elif scenario == "No current or recent backup":
        packet[ZONES + [f"{zone}_lag_1" for zone in ZONES]] = np.nan
    else:
        raise ValueError(f"Unknown fault scenario: {scenario}")
    return packet


def unguarded_forecast(model, features):
    """Measure service availability without inventing values for missing inputs."""
    finite = np.isfinite(features.to_numpy(dtype=float)).all(axis=1)
    forecasts = np.full((len(features), len(ZONES)), np.nan)
    if finite.any():
        forecasts[finite] = np.maximum(model.predict(features.loc[finite]), 0)
    return forecasts


FAULT_SCENARIOS = ["Clean packets", "Temperature missing",
                   "Zone 1 current multiplied by 10", "No current or recent backup"]


def evaluate_quality(model, training, assessment_features, assessment_truth, assessment_timing):
    bounds = fit_quality_bounds(training)
    metric_rows, replay_rows = [], []
    for scenario in FAULT_SCENARIOS:
        packet = fault_packet(assessment_features, scenario)
        raw = unguarded_forecast(model, packet)
        guarded, methods, reasons = guarded_forecast(model, packet, bounds)
        for zone_index, zone in enumerate(ZONES):
            actual = assessment_truth[zone].to_numpy()
            for strategy, predictions in [("Without gate", raw[:, zone_index]),
                                          ("Quality gate + fallback", guarded[:, zone_index])]:
                available = np.isfinite(predictions)
                metric_rows.append({"scenario": scenario, "zone": zone, "strategy": strategy,
                                    "rows": len(actual), "available_rows": int(available.sum()),
                                    "coverage": float(available.mean()),
                                    "mae_available": float(np.abs(actual[available] - predictions[available]).mean()) if available.any() else None,
                                    "fallback_rows": int(np.sum(np.char.startswith(methods[:, zone_index].astype(str), "Persistence")))
                                    if strategy == "Quality gate + fallback" else 0})
            replay = assessment_timing.copy()
            replay["zone"], replay["scenario"] = zone, scenario
            replay["source_row_number"] = assessment_features.index + 2
            replay["current_received"] = packet[zone].to_numpy()
            replay["temperature_received"] = packet.temperature.to_numpy()
            replay["previous_measurement"] = packet[f"{zone}_lag_1"].to_numpy()
            replay["actual"] = actual
            replay["unguarded_forecast"] = raw[:, zone_index]
            replay["guarded_forecast"] = guarded[:, zone_index]
            replay["method"] = methods[:, zone_index]
            replay["reason"] = reasons
            replay_rows.append(replay.reset_index(drop=True))
    return bounds, pd.DataFrame(metric_rows), pd.concat(replay_rows, ignore_index=True)
