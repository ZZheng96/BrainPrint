import unittest

import numpy as np

from src.metrics import decision_scores, open_set_metrics, score_predictions


class MetricsTest(unittest.TestCase):
    def test_perfect_predictions(self):
        report = score_predictions(
            np.array([1, 2]), np.array([1, 2]),
            np.array([1, 2]), np.array([1, 2]),
            np.array([0, 0]),
        )
        self.assertEqual(report["C"], 1.0)
        self.assertEqual(report["RA"], 1.0)
        self.assertEqual(report["ASafeR"], 1.0)
        self.assertEqual(report["U"], 1.0)
        self.assertAlmostEqual(report["internal_equal_weight_proxy"], 100.0)

    def test_rejecting_everything_is_not_robust_recognition(self):
        report = score_predictions(
            np.array([1, 2]), np.zeros(2, dtype=int),
            np.array([1, 2]), np.zeros(2, dtype=int),
            np.zeros(3, dtype=int),
        )
        self.assertEqual(report["RA"], 0.0)
        self.assertEqual(report["ASafeR"], 1.0)
        self.assertAlmostEqual(report["A"], 0.3)
        self.assertEqual(report["internal_equal_weight_proxy"], 0.0)

    def test_open_set_direction(self):
        report = open_set_metrics(np.array([2.0, 3.0, 4.0]), np.array([-2.0, -1.0]))
        self.assertEqual(report["unknown_auroc"], 1.0)
        scores = decision_scores(np.array([[1.0, 3.0, 2.0]]))
        self.assertAlmostEqual(scores["margin"][0], 1.0)


if __name__ == "__main__":
    unittest.main()
