from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch

from external_data import ExternalData, subject_folds
from native_model import NativeEEGNetEmbedding
from pair_head import SampleTemplatePairHead, same_different_labels, symmetric_pair_features
from pretrain import cross_session_supcon, pair_metrics
from run_e54 import EEGNetEmbedding
from run_pair import training_index_plan


class E73ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))

    def test_subject_roles_are_disjoint_and_cover_dataset(self):
        folds = subject_folds(self.config)
        self.assertEqual(len(folds), 3)
        for roles in folds:
            self.assertEqual([len(roles[key]) for key in ("enrolled", "calibration_unknown", "outer_unknown")], [4, 2, 3])
            sets = [set(roles[key]) for key in ("enrolled", "calibration_unknown", "outer_unknown")]
            self.assertEqual(set.union(*sets), set(range(1, 10)))
            self.assertEqual(sum(map(len, sets)), len(set.union(*sets)))

    def test_supcon_uses_only_opposite_session_positives(self):
        embedding = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], requires_grad=True)
        subject = torch.tensor([0, 0, 1, 1])
        session = torch.tensor([0, 1, 0, 1])
        loss = cross_session_supcon(embedding, subject, session, 0.07)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(embedding.grad)

    def test_supcon_rejects_missing_cross_session_positive(self):
        with self.assertRaises(ValueError):
            cross_session_supcon(
                torch.eye(2), torch.tensor([0, 0]), torch.tensor([0, 0]), 0.07
            )

    def test_pair_auroc_orientation(self):
        classes = np.asarray([1, 2])
        known_scores = np.asarray([[0.9, 0.1], [0.1, 0.9]])
        unknown_scores = np.asarray([[0.5, 0.5]])
        result = pair_metrics(known_scores, unknown_scores, np.asarray([1, 2]), classes)
        self.assertGreaterEqual(result["pair_AUROC"], 0.0)
        self.assertLessEqual(result["pair_AUROC"], 1.0)

    def test_encoder_shape_and_zero_channels(self):
        model = EEGNetEmbedding(n_classes=1, config=self.config).eval()
        raw = torch.randn(2, 18, 1200)
        raw[:, 8:] = 0
        with torch.no_grad():
            embedding = model.embed(raw)
        self.assertEqual(tuple(embedding.shape), (2, 64))
        self.assertTrue(torch.isfinite(embedding).all())
        self.assertTrue(torch.allclose(embedding.norm(dim=1), torch.ones(2), atol=1e-5))

    def test_native_22_channel_encoder(self):
        config = json.loads((Path(__file__).parent / "config_native22.json").read_text(encoding="utf-8"))
        model = NativeEEGNetEmbedding(config).eval()
        with torch.no_grad():
            embedding = model.embed(torch.randn(2, 22, 1200))
        self.assertEqual(tuple(embedding.shape), (2, 64))
        self.assertTrue(torch.isfinite(embedding).all())

    def test_pair_features_are_symmetric_and_finite(self):
        query = torch.randn(2, 64)
        template = torch.randn(4, 64)
        forward = symmetric_pair_features(query, template)
        reverse = symmetric_pair_features(template, query).transpose(0, 1)
        self.assertEqual(tuple(forward.shape), (2, 4, 129))
        self.assertTrue(torch.isfinite(forward).all())
        self.assertTrue(torch.allclose(forward, reverse, atol=1e-6))

    def test_pair_head_backpropagates_only_to_head(self):
        head = SampleTemplatePairHead()
        query = torch.randn(3, 64)
        template = torch.randn(4, 64)
        labels = same_different_labels(np.asarray([1, 2, 9]), np.asarray([1, 2, 3, 4]), query.device)
        self.assertEqual(labels.sum(dim=1).tolist(), [1.0, 1.0, 0.0])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(query, template), labels)
        loss.backward()
        self.assertIsNotNone(head.network[0].weight.grad)
        self.assertIsNone(query.grad)
        self.assertIsNone(template.grad)

    def test_pair_training_plan_excludes_unknown_and_evaluation_runs(self):
        subjects, sessions, runs = [], [], []
        for subject in (1, 2, 3):
            for session in (0, 1):
                for run in range(6):
                    subjects.append(subject)
                    sessions.append(session)
                    runs.append(run)
        data = ExternalData(
            raw=torch.empty(0), subject=np.asarray(subjects), session=np.asarray(sessions),
            run=np.asarray(runs), task=np.zeros(len(subjects)), artifact=np.zeros(len(subjects), dtype=bool),
            source_file=np.asarray(["synthetic"] * len(subjects)), channel_mask=np.ones(22, dtype=bool),
        )
        roles = {"enrolled": np.asarray([1, 2]), "calibration_unknown": np.asarray([], dtype=int),
                 "outer_unknown": np.asarray([3])}
        plan = training_index_plan(data, roles, self.config)
        self.assertEqual(len(plan), 6)
        for item in plan:
            selected = np.concatenate([item["query"], item["template"]])
            self.assertTrue(set(data.subject[selected]).issubset({1, 2}))
            self.assertTrue(set(data.run[selected]).issubset({0, 1, 2}))
            self.assertNotEqual(item["source"], item["target"])


if __name__ == "__main__":
    unittest.main()
