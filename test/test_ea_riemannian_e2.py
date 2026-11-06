import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from src.ea_riemannian_openset.e2_models import (
    MODEL_NAMES,
    E2TaskSpecificModel,
    TorchCosinePrototype,
    TorchShrinkageLDA,
)
from src.ea_riemannian_openset.run import time_holdouts
from src.ea_riemannian_openset.run_e2 import nested_loso_groups


class EARiemannianE2Test(unittest.TestCase):
    def test_lda_matches_sklearn_lsqr_auto_decision_scores(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(60, 6))
        y = np.repeat([1, 2, 4], 20)
        x[y == 2] += np.array([1.5, -0.5, 0, 0.5, 0, 1])
        x[y == 4] += np.array([-1, 1, 0.5, 0, -0.5, 0])
        classes = np.unique(y)
        model = TorchShrinkageLDA(torch.device("cpu")).fit(x, y, classes)
        scores = model(torch.as_tensor(x, dtype=torch.float64)).numpy()
        reference = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"
        ).fit(StandardScaler().fit_transform(x), y)
        np.testing.assert_allclose(
            scores, reference.decision_function(StandardScaler().fit_transform(x)),
            atol=1e-8, rtol=1e-8,
        )
        self.assertTrue(np.isfinite(scores).all())

    def test_lda_is_finite_for_small_high_dimensional_data(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(12, 40))
        y = np.repeat([1, 2, 3], 4)
        model = TorchShrinkageLDA(torch.device("cpu")).fit(x, y, np.unique(y))
        scores = model(torch.as_tensor(x)).numpy()
        self.assertEqual(scores.shape, (12, 3))
        self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_array_equal(model.classes, [1, 2, 3])

    def test_cosine_prototype_learns_simple_gaussians(self):
        rng = np.random.default_rng(8)
        centers = np.array([[-3, 0, 0], [0, 3, 0], [0, 0, 3]], dtype=float)
        y = np.repeat([1, 2, 3], 100)
        x = np.concatenate([
            rng.normal(loc=center, scale=0.2, size=(100, 3)) for center in centers
        ])
        model = TorchCosinePrototype(torch.device("cpu")).fit(x, y, np.unique(y))
        scores = model(torch.as_tensor(x, dtype=torch.float32)).numpy()
        accuracy = np.mean(model.classes[scores.argmax(axis=1)] == y)
        self.assertGreater(accuracy, 0.99)
        self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_array_equal(model.classes, [1, 2, 3])

    def test_forward_reverse_and_loso_masks_do_not_leak(self):
        metadata = pd.DataFrame({
            "subject": np.repeat(np.arange(1, 6), 20),
            "task": np.tile(np.repeat(["EC", "EO"], 10), 5),
            "trial": np.tile(np.arange(1, 11), 10),
        })
        subjects = metadata.subject.to_numpy()
        for protocol in ("forward", "reverse"):
            outer_time, inner_time = time_holdouts(metadata, protocol)
            self.assertFalse(np.any(outer_time & inner_time))
            for unknown, known, inner_groups in nested_loso_groups(subjects, 2, 2):
                outer_fit = np.isin(subjects, known) & ~outer_time
                self.assertFalse(np.any(outer_fit & np.isin(subjects, unknown)))
                self.assertFalse(np.any(outer_fit & outer_time))
                for inner_unknown in inner_groups:
                    enrolled = np.setdiff1d(known, inner_unknown)
                    inner_known = np.isin(subjects, enrolled)
                    inner_fit = inner_known & ~outer_time & ~inner_time
                    inner_eval_known = inner_known & inner_time
                    inner_eval_unknown = np.isin(subjects, inner_unknown) & ~outer_time
                    self.assertFalse(np.any(inner_fit & inner_eval_known))
                    self.assertFalse(np.any(inner_fit & inner_eval_unknown))
                    self.assertFalse(np.any(inner_eval_known & inner_eval_unknown))

    def test_task_models_and_alignment_interfaces_are_session_independent(self):
        rng = np.random.default_rng(9)
        subjects = np.repeat([1, 2], 12)
        tasks = np.tile(["EC"] * 6 + ["EO"] * 6, 2)
        x = rng.normal(size=(24, 4)).astype(np.float32)
        x[subjects == 2] += 1
        sessions = np.array(["session1"] * 24)
        first = E2TaskSpecificModel(torch.device("cpu")).fit(
            x, subjects, tasks, np.ones(24, dtype=bool)
        )
        sessions[:] = "another_session"
        second = E2TaskSpecificModel(torch.device("cpu")).fit(
            x, subjects, tasks, np.ones(24, dtype=bool)
        )
        self.assertEqual(set(first.models), {"EC", "EO"})
        for task in first.models:
            for name in MODEL_NAMES:
                a = first.models[task][name](torch.as_tensor(x)).numpy()
                b = second.models[task][name](torch.as_tensor(x)).numpy()
                np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
