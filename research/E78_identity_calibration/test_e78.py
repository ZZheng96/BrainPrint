from __future__ import annotations
import json
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
from run_e78 import fit_head, thresholds, metrics

class E78Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).parent
        cls.config = json.loads((cls.root / "config.json").read_text(encoding="utf-8"))

    def test_preregistered_protocol(self):
        self.assertEqual(self.config["data"]["train_runs"], [0, 1, 2])
        self.assertEqual(self.config["data"]["prototype_run"], 3)
        self.assertEqual(self.config["data"]["known_calibration_run"], 4)
        self.assertEqual(self.config["data"]["outer_evaluation_run"], 5)
        self.assertIsNone(self.config["threshold"]["global_floor"])

    def test_per_identity_thresholds_and_metrics(self):
        torch.manual_seed(1)
        classes = np.array([1, 2]); x = torch.randn(12, 4); y = np.repeat(classes, 6)
        means, inv = fit_head(x, y, x, y, classes)
        scores = (-torch.cdist(x, means)).numpy()
        ts = thresholds(scores, y, classes, .05)
        result = metrics(scores, scores - 2, y, classes, ts)
        self.assertEqual(set(ts), {"1", "2"})
        self.assertTrue(np.isfinite(inv.cpu().numpy()).all())
        self.assertTrue(0 <= result["AUROC"] <= 1)

    def test_unknown_is_not_threshold_input(self):
        self.assertEqual(self.config["threshold"]["calibration_unknown"], "never used; diagnostic only")
        self.assertEqual(self.config["forbidden_inputs"][0], "E71")

if __name__ == "__main__": unittest.main()
