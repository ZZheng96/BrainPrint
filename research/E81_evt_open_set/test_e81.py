from __future__ import annotations
import unittest
import numpy as np
from run_e81 import fused_scores, tail_thresholds

class E81Tests(unittest.TestCase):
    def test_fixed_min_fusion(self):
        pair = np.asarray([[0.0, 2.0], [3.0, -1.0]])
        maha = np.asarray([[2.0, 1.0], [2.0, 2.0]])
        result = fused_scores(pair, maha, {"fusion": {"pair_temperature": 1.0, "mahalanobis_temperature": 1.0}})
        np.testing.assert_allclose(result, np.minimum(1/(1+np.exp(-pair)), 1/(1+np.exp(-maha))))

    def test_thresholds_use_true_class_only(self):
        scores = np.asarray([[.8, .1], [.7, .2], [.2, .9], [.1, .8]])
        thresholds, values = tail_thresholds(scores, np.asarray([1, 1, 2, 2]), np.asarray([1, 2]), .02)
        self.assertAlmostEqual(thresholds[1], .702)
        self.assertAlmostEqual(thresholds[2], .802)
        self.assertEqual(len(values), 4)

    def test_tail_thresholds_require_each_identity(self):
        with self.assertRaises(ValueError):
            tail_thresholds(np.asarray([[.8, .1]]), np.asarray([1]), np.asarray([1, 2]), .02)

if __name__ == "__main__":
    unittest.main()
