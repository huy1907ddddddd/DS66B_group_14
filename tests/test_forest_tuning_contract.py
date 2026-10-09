"""Check random-forest selection and the displayed ensemble average."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gridwatch.forest_tuning import forest_sample_explanation, make_forest, select_forest
from gridwatch.tuning import chronological_folds


class ForestTuningContract(unittest.TestCase):
    def timing(self, rows=120):
        origin = pd.Series(pd.date_range("2017-01-01", periods=rows, freq="10min"))
        return pd.DataFrame({"origin": origin, "target_time": origin + pd.Timedelta(minutes=30)})

    def test_selection_records_the_smallest_fold_mae(self):
        values = np.arange(120, dtype=float)
        features = pd.DataFrame({"history": values, "clock": np.sin(values)})
        targets = pd.DataFrame({f"zone_{index}": (index + 1) * values + 5 for index in range(1, 4)})
        model, table, selected = select_forest(features, targets, chronological_folds(self.timing()), n_trees=3)
        winner = table.iloc[0]
        self.assertEqual(selected["max_depth"], winner.max_depth)
        self.assertEqual(selected["min_samples_leaf"], winner.min_samples_leaf)
        self.assertEqual(model.get_params()["n_estimators"], 3)

    def test_average_of_displayed_trees_matches_forest_forecast(self):
        features = pd.DataFrame({"history": np.arange(30, dtype=float), "weather": np.tile([0., 1., 2.], 10)})
        targets = pd.DataFrame({f"zone_{index}": (index + 1) * features.history for index in range(1, 4)})
        timing = self.timing(rows=len(features))
        model = make_forest(depth=3, leaf_size=3, n_trees=5).fit(features, targets)
        per_tree, summary, importance = forest_sample_explanation(model, features, targets, timing, features.iloc[[10]], 10)
        self.assertEqual(len(per_tree), 5)
        self.assertAlmostEqual(float(importance.importance.sum()), 1.0)
        np.testing.assert_allclose(summary.forest_prediction, summary.mean_of_tree_predictions)


if __name__ == "__main__":
    unittest.main()
