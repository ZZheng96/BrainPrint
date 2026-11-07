import unittest

import numpy as np
import torch

from src.raw_eeg_anchor.anchor import (
    CleanAnchorModel,
    anchor_features,
    learned_propagation,
    propagation_audit,
)


class RawEEGAnchorTest(unittest.TestCase):
    def test_anchor_features_have_expected_shape_and_finite_float32_values(self):
        rng = np.random.default_rng(42)
        scores = [rng.normal(size=(7, 3)).astype(np.float32) for _ in range(3)]
        operating_points = [
            {"score_name": name, "threshold": threshold}
            for name, threshold in zip(("max_logit", "margin", "energy"), (0.5, 0.2, 1.0))
        ]
        features = anchor_features(scores, np.array([2, 5, 9]), operating_points)
        self.assertEqual(features.shape, (7, 22))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())

    def test_clean_anchor_model_fits_separable_data(self):
        rng = np.random.default_rng(7)
        clean = rng.normal(loc=2.0, scale=0.25, size=(40, 4))
        adversarial = rng.normal(loc=-2.0, scale=0.25, size=(25, 4))
        features = np.concatenate([clean, adversarial]).astype(np.float32)
        labels = np.concatenate([np.ones(len(clean)), np.zeros(len(adversarial))])
        model = CleanAnchorModel(torch.device("cpu")).fit(features, labels)
        probability = model.predict_proba(features)
        self.assertEqual(probability.shape, (len(features),))
        self.assertTrue(np.isfinite(probability).all())
        self.assertGreater(probability[labels == 1].mean(), 0.9)
        self.assertLess(probability[labels == 0].mean(), 0.1)

    def test_learned_propagation_direction_and_guards(self):
        predictions = np.array([5, 0, 0, 7, 0, 8, 5, 6])
        probability = np.array([0.9, 0.1, 0.1, 0.9, 0.9, 0.1, 0.8, 0.7])
        neighbors = {"nearest": np.array([1, 0, 3, 2, 5, 4, 7, 6])}

        def pair_positions(_neighbors, gap):
            self.assertEqual(gap, 0.005)
            return np.array([0, 2, 4, 6])

        actual = learned_propagation(predictions, probability, neighbors, pair_positions)
        np.testing.assert_array_equal(actual, [5, 5, 7, 7, 0, 8, 5, 6])

    def test_propagation_audit_separates_clean_adv_and_unknown(self):
        before = np.array([1, 0, 2, 2, 0, 3])
        after = np.array([2, 0, 1, 0, 3, 3])
        audit = propagation_audit(before, after, np.array([1, 2]), n_known=2)
        self.assertEqual(audit, {
            "clean": {"changed": 1, "correct_before": 1, "correct_after": 0},
            "adv": {"changed": 2, "correct_before": 1, "correct_after": 1},
            "unknown": {"changed": 1, "correct_before": 1, "correct_after": 0},
        })


if __name__ == "__main__":
    unittest.main()
