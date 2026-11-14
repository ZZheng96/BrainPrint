from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch

from multiscale_model import MultiScaleNative22
from run_e74 import cross_session_supcon, training_groups
from external_data import ExternalData, subject_folds


class E74Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8-sig"))

    def test_model_shape_finite_cpu_or_cuda(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiScaleNative22(self.config).to(device).eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model.embed(torch.randn(2, 22, 1200, device=device))
        self.assertEqual(tuple(out.shape), (2, 64))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_cross_session_loss(self):
        z = torch.randn(4, 64, requires_grad=True)
        loss = cross_session_supcon(z, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1]), .07)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(z.grad)

    def test_roles_and_training_indices_isolated(self):
        folds = subject_folds(self.config)
        self.assertEqual(len(folds), 3)
        for roles in folds:
            sets = [set(roles[k]) for k in ("enrolled", "calibration_unknown", "outer_unknown")]
            self.assertEqual(set.union(*sets), set(range(1, 10)))
            self.assertEqual(sum(map(len, sets)), 9)
            subjects, sessions, runs = [], [], []
            for subject in range(1, 10):
                for session in (0, 1):
                    for run in range(6):
                        subjects.append(subject); sessions.append(session); runs.append(run)
            data = ExternalData(torch.empty(0), np.array(subjects), np.array(sessions), np.array(runs),
                                np.zeros(len(subjects)), np.zeros(len(subjects), dtype=bool),
                                np.array(["synthetic"] * len(subjects)), np.ones(22, dtype=bool))
            groups = training_groups(data, roles["enrolled"], self.config)
            selected = np.concatenate(groups)
            self.assertTrue(set(data.subject[selected]).issubset(set(roles["enrolled"])))
            self.assertTrue(set(data.run[selected]).issubset({0, 1, 2}))


if __name__ == "__main__":
    unittest.main()
