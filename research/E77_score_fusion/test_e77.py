from __future__ import annotations
import json, unittest
from pathlib import Path
import numpy as np
import torch
from run_e77 import fit_head, fit_normalizer, fuse, metric

class E77Tests(unittest.TestCase):
    def test_preregistered_weights_and_thresholds(self):
        cfg=json.loads((Path(__file__).parent/"config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["fusion"]["weights"],{"cosine_prototype":0.5,"mahalanobis":0.5})
        self.assertEqual(cfg["threshold"]["quantiles"],[0.05,0.02])
        self.assertIn("diagnostic only",cfg["threshold"]["calibration_unknown"])
    def test_fusion_and_head_finite(self):
        torch.manual_seed(1); x=torch.randn(12,4); y=np.repeat([1,2],6); means,inv=fit_head(x,y,x,y,np.array([1,2]))
        c=np.ones((12,2)); m=np.ones((12,2))*2; n=fit_normalizer(c,m,y,np.array([1,2])); s=fuse(c,m,n)
        self.assertTrue(torch.isfinite(inv).all()); self.assertTrue(np.isfinite(s).all())
    def test_metrics_gate_fields(self):
        s=np.array([[2.,0.],[2.,0.],[2.,0.],[2.,0.]])
        out=metric(s,s-3,np.array([1,1,1,1]),np.array([1,2]),0.)
        self.assertIn("known_acceptance",out); self.assertIn("unknown_false_acceptance",out)

if __name__=="__main__": unittest.main()
