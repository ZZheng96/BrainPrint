"""E254k: interface and enrollment verification for the E254 package."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "final2/team_hymn"
sys.path.insert(0, str(PKG))
import model as M  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"


def run(device, n):
    art = M.load_model(str(PKG), device)
    x = np.random.randn(n, 18, 1200).astype(np.float32)
    before = x.copy()
    meta = [{"task": "EC"}] * n
    y = M.predict_batch(art, x, meta)
    assert y.shape == (n, 24), y.shape
    assert np.isfinite(y).any(axis=1).all()
    assert not np.isnan(y).any()
    assert np.array_equal(x, before, equal_nan=True)
    return bool(y.shape == (n, 24))


def main() -> None:
    report = {}
    for dev in ("cpu", "cuda:0"):
        for n in (1, 7, 128):
            report[f"{dev}_N{n}"] = run(dev, n)
    art = M.load_model(str(PKG), "cpu")
    report["meta_none"] = bool(M.predict_batch(
        art, np.random.randn(5, 18, 1200).astype(np.float32), None).shape == (5, 24))

    base = art["prototypes"].clone()
    valid = art["valid"].clone()
    x = np.random.randn(60, 18, 1200).astype(np.float32)
    labels = np.repeat([21, 22, 23], 20)
    M.enroll(art, x, [{"task": "EC"}] * 60, labels)
    report["base_unchanged"] = bool(torch.equal(base[:, :20], art["prototypes"][:, :20]))
    report["new_valid"] = bool(art["valid"][:, 20:23].all())
    report["new_changed"] = bool(not torch.equal(base, art["prototypes"]))
    p = M.predict_batch(art, np.random.randn(6, 18, 1200).astype(np.float32),
                        [{"task": "EC"}] * 6)
    report["predict_after_enroll_finite"] = bool(np.isfinite(p).any(axis=1).all())
    (OUT / "verify_e254.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
