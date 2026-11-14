from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from run_e83 import points_for, summarize


class E83AuditTests(unittest.TestCase):
    def test_gate_requires_one_same_operating_point(self):
        points = [
            {"label": "low_fpr", "threshold": 1.0, "tpr": 0.94, "fpr": 0.01},
            {"label": "high_tpr", "threshold": 0.5, "tpr": 0.96, "fpr": 0.03},
        ]
        result = summarize(points, 0.95, 0.02)
        self.assertFalse(result["simultaneous_gate_feasible"])
        self.assertEqual(result["TPR_at_FPR_2pct"], 0.94)
        self.assertEqual(result["FPR_at_TPR_95pct"], 0.03)

    def test_gate_accepts_same_point(self):
        result = summarize([{"label": "p", "threshold": 0.0, "tpr": 0.95, "fpr": 0.02}], 0.95, 0.02)
        self.assertTrue(result["simultaneous_gate_feasible"])
        self.assertEqual(result["TPR_at_FPR_2pct"], 0.95)
        self.assertEqual(result["FPR_at_TPR_95pct"], 0.02)

    def test_e82_nested_outer_points_are_read(self):
        record = {"directions": {"T_to_E": {"outer": {
            "0.01": {"known_acceptance": 0.91, "unknown_false_acceptance": 0.01, "AUROC": 0.8},
            "0.02": {"known_acceptance": 0.96, "unknown_false_acceptance": 0.03, "AUROC": 0.8},
        }}}}
        points = points_for("E82", record, "T_to_E")
        self.assertEqual(len(points), 2)
        self.assertEqual(points[1]["tpr"], 0.96)
        self.assertEqual(points[1]["fpr"], 0.03)

    def test_config_is_local_and_test_is_not_an_input(self):
        config = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(set(config["source_results"]), {"E76", "E79", "E82"})
        self.assertNotIn("test", config["source_results"])


if __name__ == "__main__":
    unittest.main()
