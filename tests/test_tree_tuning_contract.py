"""Check the decision-tree selection and displayed leaf explanation."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gridwatch.tree_tuning import make_tree, path_rows, select_tree
from gridwatch.tuning import chronological_folds


class TreeTuningContract(unittest.TestCase):
    def timing(self, rows=120):
        origin = pd.Series(pd.date_range("2017-01-01", periods=rows, freq="10min"))
        return pd.DataFrame({"origin": origin, "target_time": origin + pd.Timedelta(minutes=30)})

    def test_selection_records_the_smallest_fold_mae(self):
        values = np.arange(120, dtype=float)
        features = pd.DataFrame({"history": values, "clock": np.sin(values)})
        targets = pd.DataFrame({f"zone_{index}": (index + 1) * values + 5 for index in range(1, 4)})
        model, table, selected = select_tree(features, targets, chronological_folds(self.timing()))
        winner = table.iloc[0]
        self.assertEqual(selected["max_depth"], winner.max_depth)
        self.assertEqual(selected["min_samples_leaf"], winner.min_samples_leaf)
        self.assertEqual(model.get_params()["max_depth"], winner.max_depth)

    def test_displayed_leaf_mean_matches_tree_prediction(self):
        features = pd.DataFrame({"history": np.arange(30, dtype=float), "weather": np.tile([0., 1., 2.], 10)})
        targets = pd.DataFrame({f"zone_{index}": (index + 1) * features.history for index in range(1, 4)})
        timing = self.timing(rows=len(features))
        model = make_tree(depth=3, leaf_size=3).fit(features, targets)
        path, leaf_id, answers, prediction = path_rows(model, features, targets, timing, features.iloc[[10]], 10)
        self.assertGreaterEqual(len(path), 1)
        self.assertGreaterEqual(leaf_id, 0)
        self.assertGreaterEqual(len(answers), 3)
        np.testing.assert_allclose(answers.mean().to_numpy(), prediction)


if __name__ == "__main__":
    unittest.main()
