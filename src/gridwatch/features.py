"""Build only information available at the forecast origin."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .data import ROOT, WEATHER, ZONES


def prepare_features(frame: pd.DataFrame, horizon: int = 3, *, save_manifest: bool = False):
    if not 0 < horizon < 144:
        raise ValueError("This baseline supports positive intraday horizons only.")
    timestamp = frame.timestamp
    target_time = timestamp.shift(-horizon)
    future_clock = timestamp + pd.Timedelta(minutes=10 * horizon)
    X = frame[WEATHER + ZONES].copy()
    for zone in ZONES:
        for lag in (1, 3, 6, 18, 144, 1008):
            X[f"{zone}_lag_{lag}"] = frame[zone].shift(lag)
        for window in (6, 18):
            X[f"{zone}_mean_{window}"] = frame[zone].rolling(window, min_periods=window).mean()
            X[f"{zone}_std_{window}"] = frame[zone].rolling(window, min_periods=window).std()
        X[f"{zone}_daily_reference"] = frame[zone].shift(144 - horizon)
    minute = future_clock.dt.hour * 60 + future_clock.dt.minute
    X["clock_sin"] = np.sin(2 * np.pi * minute / 1440)
    X["clock_cos"] = np.cos(2 * np.pi * minute / 1440)
    X["weekday_sin"] = np.sin(2 * np.pi * future_clock.dt.dayofweek / 7)
    X["weekday_cos"] = np.cos(2 * np.pi * future_clock.dt.dayofweek / 7)
    X["year_sin"] = np.sin(2 * np.pi * future_clock.dt.dayofyear / 365.25)
    X["year_cos"] = np.cos(2 * np.pi * future_clock.dt.dayofyear / 365.25)
    X["is_weekend"] = (future_clock.dt.dayofweek >= 5).astype(float)
    y = frame[ZONES].shift(-horizon)
    exact_horizon = (target_time - timestamp) == pd.Timedelta(minutes=10 * horizon)
    valid = X.notna().all(axis=1) & y.notna().all(axis=1) & exact_horizon
    X, y = X.loc[valid], y.loc[valid]
    timing = pd.DataFrame({"origin": timestamp.loc[valid], "target_time": target_time.loc[valid]})
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    validation_start = timestamp.iloc[int(len(frame) * config["chronological_train_fraction"])]
    test_start = timestamp.iloc[int(len(frame) * (config["chronological_train_fraction"] + config["chronological_validation_fraction"]))]
    masks = {
        "train": timing.target_time.lt(validation_start),
        "validation": timing.origin.ge(validation_start) & timing.target_time.lt(test_start),
        "test": timing.origin.ge(test_start),
    }
    split_manifest = {
        "status": config["split_status"], "horizon_minutes": horizon * 10,
        "validation_start": str(validation_start), "test_start": str(test_start),
        "features": list(X.columns), "feature_count": int(X.shape[1]),
        "removed_initial_history_rows": int((~valid & (timestamp < validation_start)).sum()),
        "samples": {},
    }
    for part, mask in masks.items():
        if not mask.any():
            raise ValueError(f"Empty {part} split")
        selected = timing.loc[mask]
        split_manifest["samples"][part] = {
            "rows": int(mask.sum()), "first_origin": str(selected.origin.min()),
            "last_origin": str(selected.origin.max()), "last_target": str(selected.target_time.max()),
        }
    assert timing.loc[masks["train"], "target_time"].max() < validation_start
    assert timing.loc[masks["validation"], "target_time"].max() < test_start
    assert not (masks["train"] & masks["validation"]).any()
    assert not (masks["validation"] & masks["test"]).any()
    if save_manifest:
        (ROOT / "artifacts/split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    return X, y, timing, masks, split_manifest
