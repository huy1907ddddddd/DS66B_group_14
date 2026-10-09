"""Protect tied-score decisions and the temporal calibration boundary."""

import unittest

import numpy as np
import pandas as pd

from gridwatch.decision_support import chronological_policy_split, threshold_candidates, choose_policy


class DecisionSupportContract(unittest.TestCase):
    def test_tied_scores_match_every_actual_decision(self):
        scores, truth = np.array([.2, .2, .7, .8, .8]), np.array([0, 1, 0, 1, 1])
        candidates = threshold_candidates(scores, truth)
        for row in candidates.itertuples():
            alerts = scores >= row.threshold
            self.assertEqual(row.fp, int(((truth == 0) & alerts).sum()))
            self.assertEqual(row.fn, int(((truth == 1) & ~alerts).sum()))
        self.assertEqual(int(candidates.iloc[0].tp + candidates.iloc[0].fp), len(truth))
        self.assertEqual(int(candidates.iloc[-1].tp + candidates.iloc[-1].fp), 0)

    def test_expensive_misses_change_the_optimal_decision(self):
        candidates = threshold_candidates([.1, .4, .5, .6, .8], [0, 1, 0, 0, 1])
        equal = choose_policy(candidates, 1, 1)
        expensive = choose_policy(candidates, 10, 1)
        self.assertEqual(equal.fn, 1)
        self.assertEqual(equal.fp, 0)
        self.assertEqual(expensive.fn, 0)
        self.assertEqual(expensive.fp, 2)

    def test_calibration_targets_do_not_cross_assessment_origins(self):
        origin = pd.date_range("2017-01-01", periods=20, freq="10min")
        timing = pd.DataFrame({"origin": origin, "target_time": origin + pd.Timedelta(minutes=30)})
        calibration, assessment, _ = chronological_policy_split(timing)
        self.assertEqual(int((~calibration & ~assessment).sum()), 3)
        self.assertLess(timing.loc[calibration, "target_time"].max(), timing.loc[assessment, "origin"].min())


if __name__ == "__main__":
    unittest.main()
