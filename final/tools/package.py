"""Zip final/team_hymn -> final/team_hymn.zip and verify the unpacked copy in isolation."""
import hashlib
import importlib.util
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TEAM = "team_hymn"
SRC = ROOT / "final" / TEAM
ZIP = ROOT / "final" / f"{TEAM}.zip"
FILES = ["model.py", "weights.pt", "team.json", "requirements.txt", "README.md"]

with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for name in FILES:
        z.write(SRC / name, f"{TEAM}/{name}")

tmp = Path(tempfile.mkdtemp())
try:
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
        assert all(n.startswith(f"{TEAM}/") for n in names), names
        z.extractall(tmp)
    mdir = tmp / TEAM
    torch.load(mdir / "weights.pt", weights_only=True)
    spec = importlib.util.spec_from_file_location("model_iso", mdir / "model.py")
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(mdir))
    spec.loader.exec_module(m)
    rng = np.random.default_rng(0)
    for dev in ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []):
        art = m.load_model(str(mdir), dev)
        xe = rng.standard_normal((60, 18, 1200)).astype(np.float32) * 10
        art = m.enroll(art, xe, [{"task": "Imagery"}] * 60, np.repeat([21, 22, 23], 20).astype(np.int64))
        for n in (1, 128):
            x = rng.standard_normal((n, 18, 1200)).astype(np.float32) * 10
            x0 = x.copy()
            s = m.predict_batch(art, x, [{"task": "EC"}] * n)
            assert s.shape == (n, 24) and not np.isnan(s).any() and np.isfinite(s).any(1).all()
            assert np.array_equal(x, x0)
        s = m.predict_batch(art, xe[:5], [{"task": "Imagery"}] * 5)
        print(dev, "enrolled-sample argmax", s.argmax(1).tolist())
    print("zip entries:", names)
    print("size", ZIP.stat().st_size, "sha256", hashlib.sha256(ZIP.read_bytes()).hexdigest())
finally:
    shutil.rmtree(tmp, ignore_errors=True)
