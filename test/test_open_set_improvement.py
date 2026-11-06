import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.open_set_improvement.core import (
    TorchRidge,
    open_set_report,
    select_operating_point,
)
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout


class OpenSetImprovementTest(unittest.TestCase):
    def test_torch_ridge_matches_existing_pipeline(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(60, 12)).astype(np.float32)
        y = np.repeat([2, 4, 8], 20)
        reference = make_pipeline(StandardScaler(), RidgeClassifier(alpha=100.0, solver="lsqr"))
        reference.fit(x, y)
        model = TorchRidge(torch.device("cpu")).fit(x, y, np.array([2, 4, 8]))
        with torch.no_grad():
            actual = model(torch.as_tensor(x)).numpy()
        np.testing.assert_allclose(actual, reference.decision_function(x), atol=2e-4, rtol=2e-4)

    def test_dual_ridge_matches_existing_pipeline(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(12, 30)).astype(np.float32)
        y = np.repeat([1, 3, 9], 4)
        reference = make_pipeline(StandardScaler(), RidgeClassifier(alpha=100.0, solver="lsqr"))
        reference.fit(x, y)
        model = TorchRidge(torch.device("cpu")).fit(x, y, np.array([1, 3, 9]))
        with torch.no_grad():
            actual = model(torch.as_tensor(x)).numpy()
        np.testing.assert_allclose(actual, reference.decision_function(x), atol=2e-4, rtol=2e-4)

    def test_inner_selection_and_outer_metrics(self):
        known = np.array([[4.0, 1.0, 0.0], [1.0, 4.0, 0.0]] * 20)
        unknown = np.array([[0.0, 0.1, 0.2], [0.2, 0.0, 0.1]] * 20)
        selected = select_operating_point([known], [unknown])
        report, _ = open_set_report(
            np.tile([1, 2], 20), known, unknown, np.array([1, 2, 3]),
            selected["score_name"], selected["threshold"],
        )
        self.assertEqual(report["C"], 1.0)
        self.assertEqual(report["U"], 1.0)
        self.assertEqual(report["unknown_auroc"], 1.0)
        self.assertEqual(report["unknown_f1"], 1.0)
        self.assertEqual(report["macro_f1"], 1.0)

    def test_unknown_f1_penalizes_accepted_unknowns(self):
        known = np.array([[3.0, 1.0], [3.0, 1.0]])
        unknown = np.array([[3.0, 1.0], [0.0, 1.0]])
        report, _ = open_set_report(
            np.array([1, 1]), known, unknown, np.array([1, 2]), "margin", 1.5
        )
        self.assertEqual(report["U"], 0.5)
        self.assertAlmostEqual(report["unknown_f1"], 2 / 3)

    def test_nested_time_splits_are_disjoint(self):
        metadata = pd.DataFrame({
            "subject": np.ones(10, dtype=int),
            "task": ["EC"] * 10,
            "trial": np.arange(1, 11),
        })
        outer = forward_holdout(metadata)
        inner = inner_holdout(metadata)
        self.assertFalse(np.any(outer & inner))
        np.testing.assert_array_equal(np.flatnonzero(inner), [6, 7])
        np.testing.assert_array_equal(np.flatnonzero(outer), [8, 9])
        np.testing.assert_array_equal(np.flatnonzero(~outer & ~inner), np.arange(6))


if __name__ == "__main__":
    unittest.main()
