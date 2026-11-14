from __future__ import annotations

import unittest
from pathlib import Path

from data_loader import load_bi_data
from protocol import CONFIG, session_directions, subject_folds, training_plan


class BI2015aProtocolTest(unittest.TestCase):
    def test_folds_are_disjoint_and_cover_all_subjects(self):
        folds = subject_folds(CONFIG)
        all_subjects = set(CONFIG["subjects"])
        for fold in folds:
            self.assertEqual(len(fold["enrolled"]), 24)
            self.assertEqual(len(fold["calibration_unknown"]), 5)
            self.assertEqual(len(fold["outer_unknown"]), 14)
            roles = [set(fold[key]) for key in ("enrolled", "calibration_unknown", "outer_unknown")]
            self.assertEqual(set.union(*roles), all_subjects)
            self.assertEqual(sum(map(len, roles)), 43)

    def test_training_plan_excludes_unknown_and_non_training_runs(self):
        for fold in subject_folds(CONFIG):
            plan = training_plan(fold["enrolled"], 1, 2, CONFIG)
            unknown = set(fold["calibration_unknown"]) | set(fold["outer_unknown"])
            self.assertTrue(all(subject not in unknown for subject, _, _ in plan))
            self.assertTrue(all(run in (0, 1, 2) for _, _, run in plan))
            self.assertTrue(all(session in (1, 2) for _, session, _ in plan))

    def test_fixed_directions(self):
        self.assertEqual(session_directions(CONFIG), ((1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)))

    def test_missing_data_is_not_fabricated(self):
        derived = Path(__file__).parent / "cache" / "derived"
        if not derived.exists():
            with self.assertRaises(FileNotFoundError):
                load_bi_data(1, 1, device="cpu", derived_root=derived)


if __name__ == "__main__":
    unittest.main()
