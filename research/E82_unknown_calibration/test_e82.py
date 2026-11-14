from __future__ import annotations
import unittest
import numpy as np
from run_e82 import fused_scores, unknown_thresholds

class E81Tests(unittest.TestCase):
    def test_fixed_min_fusion(self):
        pair = np.asarray([[0.0, 2.0], [3.0, -1.0]])
        maha = np.asarray([[2.0, 1.0], [2.0, 2.0]])
        result = fused_scores(pair, maha, {"fusion": {"pair_temperature": 1.0, "mahalanobis_temperature": 1.0}})
        np.testing.assert_allclose(result, np.minimum(1/(1+np.exp(-pair)), 1/(1+np.exp(-maha))))

    def test_unknown_upper_quantiles_are_fixed(self):
        scores = np.asarray([[.1, .2], [.3, .4], [.5, .6], [.7, .8]])
        thresholds, confidence = unknown_thresholds(scores, [0.01, 0.02])
        self.assertEqual(len(confidence), 4)
        self.assertGreaterEqual(thresholds["0.01"], thresholds["0.02"])

    def test_unknown_thresholds_use_only_unknown_scores(self):
        thresholds, _ = unknown_thresholds(np.asarray([[.2, .8], [.4, .6]]), [0.02])
        self.assertAlmostEqual(thresholds["0.02"], .796)

if __name__ == "__main__":
    unittest.main()
