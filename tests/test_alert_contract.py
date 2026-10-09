import unittest

import numpy as np

from gridwatch.experiment import alert_metrics, fit_operating_threshold


class AlertContract(unittest.TestCase):
    def test_no_positive_events_does_not_report_perfect_recall(self):
        result = alert_metrics([0, 0, 0], [0, 0, 0])
        self.assertIsNone(result["recall"])
        self.assertIsNone(result["miss_rate"])
        self.assertEqual(result["positive_events"], 0)

    def test_ties_cannot_break_validation_false_alarm_budget(self):
        scores = np.array([0.] * 90 + [1.] * 10)
        threshold = fit_operating_threshold(scores, np.zeros(100))
        self.assertLessEqual(np.mean(scores >= threshold), .05)


if __name__ == "__main__":
    unittest.main()
