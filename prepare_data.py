"""Run the reviewed preparation stage without training a model."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from gridwatch.data import load_source
from gridwatch.features import prepare_features


def main():
    (ROOT / "artifacts").mkdir(exist_ok=True)
    frame = load_source()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    X, y, timing, masks, manifest = prepare_features(
        frame, horizon=config["forecast_horizon_minutes"] // 10, save_manifest=True
    )
    example_time = pd.Timestamp("2017-01-08 00:00")
    selected = timing.origin.eq(example_time)
    examples = pd.concat([
        timing.loc[selected],
        X.loc[selected, ["temperature", "zone_1", "zone_1_lag_1"]],
        y.loc[selected, ["zone_1"]].rename(columns={"zone_1": "actual_zone_1_30min_later"}),
    ], axis=1)
    examples.to_csv(ROOT / "artifacts/real_example.csv", index=False)
    print(f"Source rows: {len(frame):,}; input columns: {X.shape[1]}")
    for part, details in manifest["samples"].items():
        print(f"{part}: {details['rows']:,} samples")
    print("\nActual source example (not a prediction):")
    print(examples.to_string(index=False))


if __name__ == "__main__":
    main()
