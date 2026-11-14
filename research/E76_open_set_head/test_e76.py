from __future__ import annotations
import json, unittest
from pathlib import Path
import numpy as np
import torch
from run_e76 import fit_head, metrics
from external_data import subject_folds

class E76Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((Path(__file__).parent/"config.json").read_text(encoding="utf-8"))
    def test_roles_are_disjoint(self):
        base=json.loads((Path(__file__).parents[1]/"E74_multiscale_native22/config.json").read_text(encoding="utf-8-sig"))
        for roles in subject_folds(base):
            sets=[set(roles[k]) for k in ("enrolled","calibration_unknown","outer_unknown")]
            self.assertEqual(set.union(*sets),set(range(1,10)))
            self.assertEqual(sum(map(len,sets)),9)
    def test_mahalanobis_head_and_metrics(self):
        torch.manual_seed(1); classes=np.array([1,2]); x=torch.randn(12,4); y=np.repeat(classes,6)
        means,inv=fit_head(x,y,x,y,classes)
        s=(-torch.cdist(x,means)).numpy()
        result=metrics(s,s-2,y,classes,float(np.quantile(s.max(1),.05)))
        self.assertTrue(np.isfinite(inv.cpu().numpy()).all()); self.assertTrue(0<=result["AUROC"]<=1)
    def test_calibration_unknown_not_threshold_input(self):
        self.assertEqual(self.config["data"]["calibration_unknown_run"],5)
        self.assertIn("diagnostic only",self.config["protocol"]["threshold"])

if __name__=="__main__": unittest.main()
