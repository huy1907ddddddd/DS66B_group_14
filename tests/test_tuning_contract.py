"""Check temporal safety and the actual alpha-selection objective."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gridwatch.tuning import chronological_folds, forecast_mae, search_alpha


class TuningContract(unittest.TestCase):
    def timing(self, rows=120):
        origin = pd.Series(pd.date_range("2017-01-01", periods=rows, freq="10min"))
        return pd.DataFrame({"origin": origin, "target_time": origin + pd.Timedelta(minutes=30)})

    def test_gap_prevents_training_labels_crossing_each_boundary(self):
        timing = self.timing()
        for learn, evaluate in chronological_folds(timing):
            self.assertLess(timing.iloc[learn].target_time.max(), timing.iloc[evaluate].origin.min())
        with self.assertRaises(ValueError):
            chronological_folds(timing, horizon_steps=0)

    def test_score_matches_nonnegative_forecast_contract(self):
        self.assertEqual(forecast_mae([[1, 4]], [[-10, 2]]), 1.5)

    def test_selected_alpha_minimizes_the_recorded_fold_scores(self):
        values = np.arange(120, dtype=float)
        X = pd.DataFrame({"history": values, "clock": np.sin(values)})
        y = pd.DataFrame({f"zone_{i}": (i + 1) * values + 5 for i in range(1, 4)})
        folds = chronological_folds(self.timing())
        model, scores, choice = search_alpha("Ridge", X, y, folds, alphas=[.1, 100], expand=False, refine=False)
        expected = scores.sort_values(["cv_mae", "alpha"]).iloc[0]
        self.assertEqual(choice["alpha"], expected.alpha)
        self.assertEqual(model.named_steps["standardscaler"].n_samples_seen_, len(X))
        np.testing.assert_allclose(model.named_steps["standardscaler"].mean_, X.mean())


if __name__ == "__main__":
    unittest.main()
