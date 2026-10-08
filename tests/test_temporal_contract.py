"""Test the forecasting contract rather than implementation details."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gridwatch.data import WEATHER, ZONES
from gridwatch.features import prepare_features


class TemporalContract(unittest.TestCase):
    def frame(self):
        frame = pd.DataFrame({"timestamp": pd.date_range("2017-01-01", periods=5000, freq="10min")})
        for i, name in enumerate(WEATHER + ZONES):
            frame[name] = np.arange(len(frame), dtype=float) + i
        return frame

    def test_targets_and_boundaries(self):
        frame = self.frame()
        X, y, times, masks, _ = prepare_features(frame)
        self.assertTrue(((times.target_time - times.origin) == pd.Timedelta(minutes=30)).all())
        self.assertTrue(np.allclose(y.zone_1 - X.zone_1, 3))
        self.assertLess(times.loc[masks["train"], "target_time"].max(), times.loc[masks["validation"], "origin"].min())
        self.assertLess(times.loc[masks["validation"], "target_time"].max(), times.loc[masks["test"], "origin"].min())

    def test_future_changes_cannot_change_current_features(self):
        frame = self.frame()
        baseline, _, _, _, _ = prepare_features(frame)
        changed = frame.copy()
        changed.loc[3001:, WEATHER + ZONES] += 1_000_000
        revised, _, _, _, _ = prepare_features(changed)
        pd.testing.assert_frame_equal(baseline.loc[:3000], revised.loc[:3000])


if __name__ == "__main__":
    unittest.main()
