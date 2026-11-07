import unittest

import numpy as np

from src.raw_eeg_defense.run import (
    aggregate,
    asymmetric_propagation,
    competition_metrics,
    multiview_predictions,
    transform_view,
)


class RawEEGDefenseTest(unittest.TestCase):
    def test_asymmetric_propagation_only_changes_lower_confidence_receiver(self):
        neighbors = {
            "nearest": np.array([1, 0, 0]),
            "similarity": np.array([0.999, 0.999, 0.999]),
            "second_similarity": np.array([0.90, 0.90, 0.90]),
            "reciprocal": np.array([True, True, False]),
        }
        actual = asymmetric_propagation(
            np.array([5, 0, 7]), np.array([2.0, 0.5, 3.0]), neighbors, 1.0
        )
        np.testing.assert_array_equal(actual, [5, 5, 7])

    def test_asymmetric_propagation_rejects_weak_anchor(self):
        neighbors = {
            "nearest": np.array([1, 0]),
            "similarity": np.array([0.999, 0.999]),
            "second_similarity": np.array([0.90, 0.90]),
            "reciprocal": np.array([True, True]),
        }
        actual = asymmetric_propagation(
            np.array([5, 0]), np.array([0.5, 0.1]), neighbors, 1.0
        )
        np.testing.assert_array_equal(actual, [5, 0])

    def test_multiview_requires_two_nonzero_votes(self):
        predictions = [
            np.array([5, 0, 7]),
            np.array([5, 6, 0]),
            np.array([0, 6, 0]),
        ]
        np.testing.assert_array_equal(multiview_predictions(predictions), [5, 6, 0])

    def test_competition_metrics_include_rejection_and_unknown_f1(self):
        report = competition_metrics(
            np.array([1, 2]),
            np.array([1, 0]),
            np.array([1, 0]),
            np.array([0, 2]),
            np.array([2.0, 1.0]),
            np.array([0.0, 3.0]),
        )
        self.assertEqual(report["C"], 0.5)
        self.assertEqual(report["RA"], 0.5)
        self.assertEqual(report["ASafeR"], 1.0)
        self.assertEqual(report["U"], 0.5)
        self.assertEqual(report["unknown_f1"], 0.5)
        self.assertAlmostEqual(report["A"], 0.65)

    def test_aggregate_is_unweighted_fold_macro_average(self):
        reports = [
            {name: 0.5 for name in (
                "C", "RA", "ASafeR", "A", "U", "official_score", "unknown_auroc",
                "known_aupr", "unknown_aupr", "eer", "clean_macro_f1_with_reject",
                "adversarial_macro_f1_with_reject", "unknown_f1",
            )} | {"n_known": 10, "n_unknown": 5},
            {name: 1.0 for name in (
                "C", "RA", "ASafeR", "A", "U", "official_score", "unknown_auroc",
                "known_aupr", "unknown_aupr", "eer", "clean_macro_f1_with_reject",
                "adversarial_macro_f1_with_reject", "unknown_f1",
            )} | {"n_known": 1000, "n_unknown": 500},
        ]
        result = aggregate(reports)
        self.assertEqual(result["C"], 0.75)
        self.assertEqual(result["official_score"], 0.75)
        self.assertEqual(result["n_known"], 1010)

    def test_views_keep_original_shape_and_finite_values(self):
        data = np.random.default_rng(42).normal(size=(1200, 18))
        for view in ("original", "lowpass40", "resample150"):
            transformed = transform_view(data, view)
            self.assertEqual(transformed.shape, data.shape)
            self.assertTrue(np.isfinite(transformed).all())


if __name__ == "__main__":
    unittest.main()
