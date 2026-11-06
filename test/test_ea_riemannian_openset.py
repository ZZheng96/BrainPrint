import unittest

import numpy as np
import pandas as pd
import torch

from src.ea_riemannian_openset.core import (
    N_BANDS,
    N_CHANNELS,
    SPDRecenter,
    log_features_to_spd,
    riemannian_mean,
    spd_to_log_features,
)
from src.ea_riemannian_openset.run import CONFIGS, fit_predict_split, time_holdouts
from src.open_set_calibration.core import predict_calibration_scores
from src.open_set_improvement.core import TaskSpecificModel


def random_spd_features(rng: np.random.Generator, n: int) -> np.ndarray:
    matrices = rng.normal(size=(n, N_BANDS, N_CHANNELS, N_CHANNELS)) / 4
    covariances = matrices @ matrices.transpose(0, 1, 3, 2)
    covariances += np.eye(N_CHANNELS)[None, None]
    features = spd_to_log_features(torch.as_tensor(covariances), batch_size=8)
    return features.numpy().astype(np.float32)


class EARiemannianOpenSetTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261107)
        self.features = random_spd_features(self.rng, 12)
        self.tasks = np.repeat(["EC", "EO"], 6)

    def test_log_spd_roundtrip(self):
        original = log_features_to_spd(self.features, batch_size=5)
        recovered_features = spd_to_log_features(original, batch_size=5)
        recovered = log_features_to_spd(recovered_features, batch_size=5)
        torch.testing.assert_close(
            recovered_features,
            torch.as_tensor(self.features, dtype=torch.float64),
            atol=2e-6,
            rtol=2e-6,
        )
        torch.testing.assert_close(recovered, original, atol=2e-6, rtol=2e-6)

    def test_ea_recentered_arithmetic_mean_is_identity(self):
        aligner = SPDRecenter("ea", "cpu", batch_size=5).fit(
            self.features, self.tasks
        )
        aligned = aligner.transform(self.features, self.tasks)
        covariances = log_features_to_spd(aligned, batch_size=5)
        identity = torch.eye(N_CHANNELS, dtype=torch.float64).expand(
            N_BANDS, N_CHANNELS, N_CHANNELS
        )
        for task in ("EC", "EO"):
            task_covariances = covariances[self.tasks == task]
            torch.testing.assert_close(
                task_covariances.mean(dim=0), identity, atol=2e-5, rtol=2e-5
            )

    def test_riemannian_mean_is_spd(self):
        covariances = log_features_to_spd(self.features, batch_size=4)
        mean, iterations = riemannian_mean(
            covariances, max_iter=40, tol=1e-9, return_iterations=True
        )
        self.assertGreater(iterations, 0)
        torch.testing.assert_close(
            mean, mean.transpose(-1, -2), atol=1e-10, rtol=1e-10
        )
        self.assertTrue(torch.all(torch.linalg.eigvalsh(mean) > 0))

    def test_alignment_does_not_depend_on_subject_or_session(self):
        subjects = np.tile(np.arange(1, 4), 4)
        sessions = np.array(["session1"] * len(self.tasks), dtype=object)
        first = SPDRecenter("ea", "cpu", batch_size=4).fit(
            self.features, self.tasks
        ).transform(self.features, self.tasks)
        subjects[:] = 99
        sessions[:] = "changed_session"
        second = SPDRecenter("ea", "cpu", batch_size=4).fit(
            self.features, self.tasks
        ).transform(self.features, self.tasks)
        np.testing.assert_array_equal(first, second)

    def test_forward_temporal_masks_match_expected_intervals(self):
        metadata = pd.DataFrame({
            "subject": np.repeat([1, 2], 20),
            "task": np.tile(np.repeat(["EC", "EO"], 10), 2),
            "trial": np.tile(np.arange(1, 11), 4),
        })
        outer, inner = time_holdouts(metadata, "forward")
        self.assertTrue(np.all(outer == (metadata.trial.to_numpy() >= 9)))
        self.assertTrue(np.all(inner == np.isin(metadata.trial.to_numpy(), [7, 8])))
        self.assertFalse(np.any(outer & inner))

    def test_reverse_temporal_masks_are_forward_mirror_without_leakage(self):
        metadata = pd.DataFrame({
            "subject": np.repeat([1, 2], 20),
            "task": np.tile(np.repeat(["EC", "EO"], 10), 2),
            "trial": np.tile(np.arange(1, 11), 4),
        })
        reverse_outer, reverse_inner = time_holdouts(metadata, "reverse")
        mirrored_metadata = metadata.copy()
        mirrored_metadata["trial"] = 11 - mirrored_metadata["trial"]
        mirrored_outer, mirrored_inner = time_holdouts(mirrored_metadata, "forward")
        np.testing.assert_array_equal(reverse_outer, mirrored_outer)
        np.testing.assert_array_equal(reverse_inner, mirrored_inner)
        self.assertTrue(np.all(reverse_outer == (metadata.trial.to_numpy() <= 2)))
        self.assertTrue(np.all(reverse_inner == np.isin(metadata.trial.to_numpy(), [3, 4])))
        inner_fit = ~reverse_outer & ~reverse_inner
        self.assertFalse(np.any(inner_fit & reverse_outer))
        self.assertFalse(np.any(inner_fit & reverse_inner))

    def test_small_pipeline_preserves_indices_and_has_no_nan(self):
        subjects = np.repeat(np.arange(1, 5), 8)
        tasks = np.tile(np.repeat(["EC", "EO"], 4), 4)
        features = random_spd_features(self.rng, len(subjects))
        window = self.rng.normal(size=(len(subjects), 2, features.shape[1])).astype(
            np.float32
        )
        fit_mask = np.tile([True, True, False, False] * 2, 4)
        eval_indices = np.flatnonzero(~fit_mask)
        outputs, _, classes = fit_predict_split(
            features,
            window,
            subjects,
            tasks,
            fit_mask,
            eval_indices,
            torch.device("cpu"),
            batch_size=4,
            riemannian_max_iter=3,
            riemannian_tol=1e-6,
        )
        self.assertEqual(set(outputs), set(CONFIGS))
        for values in outputs.values():
            self.assertEqual(values.shape, (len(eval_indices), len(classes)))
            self.assertTrue(np.isfinite(values).all())

        baseline_model = TaskSpecificModel(torch.device("cpu")).fit(
            features, window, subjects, tasks, fit_mask
        )
        for alignment in ("none", "target_zscore"):
            expected, _ = predict_calibration_scores(
                baseline_model,
                features[eval_indices],
                window[eval_indices],
                tasks[eval_indices],
                alignment,
                batch_size=4,
            )
            np.testing.assert_allclose(
                outputs[f"full_{alignment}"], expected[f"full_{alignment}"],
                atol=1e-6, rtol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
