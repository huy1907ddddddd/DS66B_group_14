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

