import tempfile
import unittest
from pathlib import Path
import numpy as np
from data_loader import _validate

class DynamicBidsProtocolTest(unittest.TestCase):
    def test_dynamic_run_counts_are_validated(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'session_1.npz'; counts=[2,3,4,5,6,7]; n=sum(counts); run=np.repeat(np.arange(6),counts)
            np.savez(p,raw=np.zeros((n,32,300),np.float32),run=run,trial_index=np.arange(n))
            with np.load(p,allow_pickle=False) as archive:
                raw, actual_run, trial = _validate(1,1,p,archive)
            self.assertEqual(raw.shape[0],n); self.assertEqual(actual_run.tolist(),run.tolist()); self.assertEqual(trial.tolist(),list(range(n)))
    def test_labels_are_not_required(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'session_1.npz'; n=6
            np.savez(p,raw=np.zeros((n,32,300),np.float32),run=np.arange(6),trial_index=np.arange(n))
            with np.load(p,allow_pickle=False) as archive:
                _, _, trial = _validate(1,1,p,archive)
            self.assertEqual(trial.tolist(),list(range(n)))

if __name__=='__main__': unittest.main()
