from __future__ import annotations
import unittest
import numpy as np
import torch
from run_e80 import fused_scores, maha_confidence

class E80Tests(unittest.TestCase):
    def test_fixed_min_fusion(self):
        pair = np.asarray([[0.0, 2.0], [3.0, -1.0]])
        maha = np.asarray([[2.0, 1.0], [2.0, 2.0]])
        result = fused_scores(pair, maha, {"fusion": {"pair_temperature": 1.0, "mahalanobis_temperature": 1.0}})
        expected = np.minimum(1 / (1 + np.exp(-pair)), 1 / (1 + np.exp(-maha)))
        np.testing.assert_allclose(result, expected)

    def test_classwise_median_mad_confidence(self):
        embedding = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        stats = {"means": np.asarray([[0.0, 0.0], [2.0, 0.0]]), "invcov": np.asarray([np.eye(2), np.eye(2)]), "medians": np.asarray([1.0, 1.0]), "mads": np.asarray([1.0, 1.0])}
        result = maha_confidence(embedding, stats, {"fusion": {"mahalanobis_temperature": 1.0}})
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(result[0, 0], result[0, 1])

    def test_q02_is_known_only(self):
        known = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
        threshold = float(np.quantile(known, 0.02, method="linear"))
        self.assertAlmostEqual(threshold, 0.108)
        unknown = np.asarray([0.109, 0.2])
        self.assertEqual(float(np.mean(unknown >= threshold)), 1.0)

if __name__ == "__main__":
    unittest.main()
