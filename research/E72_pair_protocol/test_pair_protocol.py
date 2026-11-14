from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pair_protocol import (
    calibrate_q05, confidence_values, continuous_masks, evaluate_queries,
    fit_spaces, inner_splits, make_template_pairs, outer_groups, predict_spaces,
    subject_roles,
)


class PairProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))

    def test_subject_roles_and_continuous_blocks(self):
        metadata = pd.DataFrame({
            "subject": np.repeat(np.arange(1, 21), 10),
            "task": "EC",
            "trial": np.tile(np.arange(1, 11), 20),
        })
        subjects = np.arange(1, 21)
        outer = outer_groups(subjects, self.config)[0]
        known_subjects = np.setdiff1d(subjects, outer)
        splits = inner_splits(known_subjects, 1, self.config)
        roles = subject_roles(subjects, outer, splits[0])
        self.assertEqual([len(roles[key]) for key in ("outer_unknown", "enrolled", "calibration_unknown")], [4, 8, 8])
        self.assertFalse(set(roles["enrolled"]) & set(roles["calibration_unknown"]))
        masks = continuous_masks(metadata)
        enrolled_mask = np.isin(metadata["subject"], roles["enrolled"])
        fit = np.flatnonzero(enrolled_mask & masks["inner_fit"])
        known = np.flatnonzero(enrolled_mask & masks["inner_known"])
        forward = np.flatnonzero(enrolled_mask & masks["outer_known"])
        self.assertEqual((len(fit), len(known), len(forward)), (48, 16, 16))
        self.assertFalse(np.intersect1d(fit, known).size)
        self.assertFalse(np.intersect1d(known, forward).size)

    def test_query_template_pairs(self):
        scores = np.asarray([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
        classes = np.asarray([1, 2])
        pairs = make_template_pairs(scores, classes, np.asarray([1, 2, 0]))
        self.assertEqual(len(pairs["same"]), 6)
        self.assertEqual(int(pairs["same"].sum()), 2)
        self.assertEqual(int((1 - pairs["same"]).sum()), 4)
        unknown = make_template_pairs(scores[:1], classes, np.asarray([0]))
        self.assertFalse(unknown["same"].any())

    def test_confidence_and_known_only_q05(self):
        scores = np.asarray([[0.8, 0.2], [0.7, 0.4]])
        np.testing.assert_allclose(confidence_values(scores, "max_cosine"), [0.8, 0.7])
        np.testing.assert_allclose(confidence_values(scores, "margin"), [0.6, 0.3])
        np.testing.assert_allclose(confidence_values(scores, "distance_ratio"), [4.0, 2.0])
        runs = [{
            "known_scores": {"wccn": scores},
            "unknown_scores": {"wccn": np.asarray([[100.0, -100.0]])},
        }]
        expected = np.quantile(confidence_values(scores, "max_cosine"), 0.05, method="linear")
        self.assertAlmostEqual(calibrate_q05(runs, "wccn", "max_cosine"), expected)
        runs[0]["unknown_scores"]["wccn"][:] = -1000
        self.assertAlmostEqual(calibrate_q05(runs, "wccn", "max_cosine"), expected)

    def test_open_set_metrics_at_working_threshold(self):
        known_scores = np.asarray([[0.9, 0.1], [0.2, 0.8]])
        unknown_scores = np.asarray([[0.7, 0.3]])
        metrics, records = evaluate_queries(
            np.asarray([1, 2]), known_scores, unknown_scores,
            np.asarray([1, 2]), "max_cosine", 0.85,
        )
        self.assertEqual(metrics["direct_C"], 1.0)
        self.assertEqual(metrics["C"], 0.5)
        self.assertEqual(metrics["U"], 1.0)
        self.assertEqual(metrics["U_at_95"], 1.0)
        self.assertEqual(metrics["AUROC"], 1.0)
        self.assertEqual(metrics["N_same_pairs"], 2)
        self.assertEqual(metrics["N_different_pairs"], 4)
        np.testing.assert_array_equal(records["known_pred"], [1, 0])
        np.testing.assert_array_equal(records["unknown_pred"], [0])

    def test_template_models_are_finite(self):
        x = torch.asarray([
            [1.0, 0.0], [0.9, 0.1], [1.1, -0.1],
            [0.0, 1.0], [0.1, 0.9], [-0.1, 1.1],
        ], dtype=torch.float32)
        y = np.asarray([1, 1, 1, 2, 2, 2])
        classes = np.asarray([1, 2])
        models = fit_spaces(x, y, classes, self.config)
        scores = predict_spaces(models, x, np.arange(6), 2)
        self.assertEqual(set(scores), {"cosine", "wccn"})
        for values in scores.values():
            self.assertEqual(values.shape, (6, 2))
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue(np.all(values.argmax(axis=1) == np.repeat([0, 1], 3)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
