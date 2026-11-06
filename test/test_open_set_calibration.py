import unittest

import numpy as np
import pandas as pd
import torch

from src.open_set_calibration.core import (
    _target_zscore,
    aggregate_reports,
    build_submission,
    predict_calibration_scores,
)
from src.open_set_calibration.run_calibration import choose_score_name
from src.open_set_improvement.core import TaskSpecificModel


class OpenSetCalibrationTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.x = rng.normal(size=(36, 6)).astype(np.float32)
        self.window = rng.normal(size=(36, 2, 6)).astype(np.float32)
        self.subjects = np.tile([1, 2, 3], 12)
        self.tasks = np.repeat(["EC", "EO"], 18)
        self.model = TaskSpecificModel(torch.device("cpu")).fit(
            self.x, self.window, self.subjects, self.tasks,
            np.ones(len(self.subjects), dtype=bool),
        )
        self.target_x = rng.normal(size=(12, 6)).astype(np.float32) + 0.5
        self.target_window = rng.normal(size=(12, 2, 6)).astype(np.float32) - 0.5
        self.target_tasks = np.repeat(["EC", "EO"], 6)

    def test_no_alignment_matches_direct_model_scores(self):
        outputs, diagnostics = predict_calibration_scores(
            self.model, self.target_x, self.target_window, self.target_tasks, "none"
        )
        self.assertEqual(diagnostics, {})
        for task in ("EC", "EO"):
            indices = np.flatnonzero(self.target_tasks == task)
            for variant in ("full", "base"):
                with torch.no_grad():
                    expected = self.model.scores(
                        torch.as_tensor(self.target_x[indices]),
                        torch.as_tensor(self.target_window[indices]),
                        task,
                        variant,
                    ).numpy()
                np.testing.assert_allclose(
                    outputs[f"{variant}_none"][indices], expected, atol=1e-6, rtol=1e-6
                )

    def test_target_zscore_is_permutation_invariant_without_labels(self):
        first, _ = predict_calibration_scores(
            self.model, self.target_x, self.target_window, self.target_tasks,
            "target_zscore",
        )
        permutation = np.random.default_rng(7).permutation(len(self.target_x))
        second, _ = predict_calibration_scores(
            self.model, self.target_x[permutation], self.target_window[permutation],
            self.target_tasks[permutation], "target_zscore",
        )
        for config in first:
            np.testing.assert_allclose(
                first[config][permutation], second[config], atol=2e-5, rtol=2e-5
            )

    def test_row_score_normalization_applies_after_feature_alignment(self):
        raw, _ = predict_calibration_scores(
            self.model, self.target_x, self.target_window, self.target_tasks,
            "target_zscore",
        )
        normalized, _ = predict_calibration_scores(
            self.model, self.target_x, self.target_window, self.target_tasks,
            "target_zscore", score_normalization="row_zscore",
        )
        for variant in ("full", "base"):
            raw_scores = torch.as_tensor(raw[f"{variant}_target_zscore"])
            expected = (raw_scores - raw_scores.mean(dim=1, keepdim=True)) / raw_scores.std(
                dim=1, keepdim=True, unbiased=False
            ).clamp_min(1e-6)
            np.testing.assert_allclose(
                normalized[f"{variant}_target_zscore_row_zscore"],
                expected.numpy(), atol=1e-6, rtol=1e-6,
            )

    def test_target_zscore_maps_moments_to_source(self):
        target = torch.tensor([
            [0.0, 0.0], [2.0, 4.0], [4.0, 8.0], [6.0, 12.0]
        ])
        aligned, clip_fraction = _target_zscore(
            target,
            source_mean=torch.tensor([5.0, 6.0]),
            source_scale=torch.tensor([2.0, 4.0]),
        )
        self.assertEqual(clip_fraction, 0.0)
        torch.testing.assert_close(aligned.mean(dim=0), torch.tensor([5.0, 6.0]))
        torch.testing.assert_close(
            aligned.std(dim=0, unbiased=False), torch.tensor([2.0, 4.0])
        )

    def test_auc_aggregation_is_outer_fold_macro_average(self):
        records = {
            "is_known": np.array([True, True, False, False]),
            "true": np.array([1, 2, 0, 0]),
            "pred": np.array([1, 2, 0, 0]),
            "confidence": np.array([0.9, 0.8, 0.2, 0.1]),
        }
        folds = [
            {
                "unknown_auroc": 0.5, "known_aupr": 0.6, "unknown_aupr": 0.7,
                "eer": 0.2, "direct_C_without_rejection": 1.0,
                "known_acceptance": 0.9, "n_known": 2, "score_name": "margin",
                "threshold": 0.1,
            },
            {
                "unknown_auroc": 1.0, "known_aupr": 0.8, "unknown_aupr": 0.9,
                "eer": 0.0, "direct_C_without_rejection": 1.0,
                "known_acceptance": 1.0, "n_known": 2, "score_name": "margin",
                "threshold": 0.2,
            },
        ]
        report = aggregate_reports(records, folds)
        self.assertEqual(report["unknown_auroc"], 0.75)
        self.assertEqual(report["unknown_aupr"], 0.8)
        self.assertEqual(report["C"], 1.0)
        self.assertEqual(report["macro_f1"], 1.0)

    def test_submission_preserves_sample_order_and_label_range(self):
        template = pd.DataFrame({"Sample": ["b", "a"], "Label": [0, 0]})
        metadata = pd.DataFrame({"sample": ["a", "b"]})
        submission = build_submission(template, metadata, np.array([3, 0]))
        self.assertEqual(submission["Sample"].tolist(), ["b", "a"])
        self.assertEqual(submission["Label"].tolist(), [0, 3])
        with self.assertRaises(ValueError):
            build_submission(template, metadata, np.array([21, 0]))

    def test_candidate_score_selection_uses_only_inner_summaries(self):
        selections = [
            {"score_name": "margin", "inner_detection_macro_f1": 0.8},
            {"score_name": "margin", "inner_detection_macro_f1": 0.9},
            {"score_name": "energy", "inner_detection_macro_f1": 0.99},
        ]
        self.assertEqual(choose_score_name(selections), "margin")


if __name__ == "__main__":
    unittest.main()
