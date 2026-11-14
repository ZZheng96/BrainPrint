from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

E75_DIR = Path(__file__).resolve().parent
ROOT = E75_DIR.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "E73_external_multisession"))
from external_data import subject_folds  # noqa: E402
from run_e75 import confidence  # noqa: E402
from src.eeg_image_dinov2.image_encoders import encode  # noqa: E402


class E75Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((E75_DIR / "config.json").read_text(encoding="utf-8"))

    def test_image_encoder_shape(self):
        image = encode(np.random.default_rng(0).normal(size=(1200, 18)).astype(np.float32), "topomap5")
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertTrue(np.isfinite(image).all())
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)

    def test_subject_roles_and_run_protocol(self):
        folds = subject_folds(self.config)
        self.assertEqual(len(folds), 3)
        for roles in folds:
            self.assertEqual(set(np.concatenate(list(roles.values())).tolist()), set(range(1, 10)))
            self.assertEqual([len(roles[key]) for key in ("enrolled", "calibration_unknown", "outer_unknown")], [4, 2, 3])
            self.assertTrue(set(roles["enrolled"]).isdisjoint(roles["calibration_unknown"]))
            self.assertTrue(set(roles["enrolled"]).isdisjoint(roles["outer_unknown"]))

    def test_margin_confidence(self):
        scores = np.asarray([[0.8, 0.2, 0.1], [0.2, 0.3, 0.1]])
        np.testing.assert_allclose(confidence(scores), [0.6, 0.1])


if __name__ == "__main__":
    unittest.main()
