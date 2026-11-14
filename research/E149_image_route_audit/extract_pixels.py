"""E149: pixel-space features of every cached image type.

The image route has two representations that can actually be deployed:
  A. the frozen DINOv2 embedding (768) of each image type
  B. the image itself, mean-pooled to a small grid, i.e. no DINOv2 at all

This script builds B for every image type that exists for BOTH Train and Test.
Types that only exist for Train (gaf, mtf, spectrogram, topomap, tokens, dinov3)
cannot be used and are reported separately.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
IMG = ROOT / "BrainPrint_cache" / "eeg_image_dinov2"
MW = ROOT / "BrainPrint_cache" / "e62_multiwindow"
GRID = 32

PAIRS = {
    "correlation": (IMG / "train_correlation.npy", IMG / "test_correlation.npy"),
    "covariance5": (IMG / "train_covariance5.npy", IMG / "test_covariance5.npy"),
    "covariance_hr": (IMG / "train_covariance_hr.npy", IMG / "test_covariance_hr.npy"),
    "topomap5": (IMG / "train_topomap5.npy", IMG / "test_topomap5.npy"),
    "mw_corr8": (MW / "train_multiwindow_corr8.npy", MW / "test_multiwindow_corr8.npy"),
    "mw_cov4": (MW / "train_multiwindow_cov4.npy", MW / "test_multiwindow_cov4.npy"),
    "mw_cov8": (MW / "train_multiwindow_cov8.npy", MW / "test_multiwindow_cov8.npy"),
}


def pool(path: Path) -> np.ndarray:
    a = np.load(path, mmap_mode="r")
    n = a.shape[0]
    out = np.empty((n, 3 * GRID * GRID), dtype=np.float32)
    step = 512
    for st in range(0, n, step):
        sp = min(st + step, n)
        x = torch.as_tensor(np.asarray(a[st:sp]), dtype=torch.float32)  # B x 3 x 224 x 224
        x = torch.nn.functional.adaptive_avg_pool2d(x, GRID)            # B x 3 x G x G
        out[st:sp] = x.reshape(sp - st, -1).numpy()
    return out


def main():
    meta = {}
    for name, (ptr, pte) in PAIRS.items():
        tr = pool(ptr)
        np.save(OUT / f"train_px_{name}.npy", tr)
        te = pool(pte)
        np.save(OUT / f"test_px_{name}.npy", te)
        mu = tr.mean(0)
        sd = tr.std(0) + 1e-6
        meta[name] = {"train": str(OUT / f"train_px_{name}.npy"),
                      "test": str(OUT / f"test_px_{name}.npy"),
                      "dim": int(tr.shape[1]),
                      "train_scale": float(tr.std()), "test_scale": float(te.std()),
                      "mean_shift_l2": float(np.linalg.norm((te.mean(0) - mu) / sd))}
        print(f"{name:14s} {tr.shape} {te.shape} "
              f"shift={meta[name]['mean_shift_l2']:.1f} sd", flush=True)
        del tr, te
    (OUT / "pixel_manifest.json").write_text(
        json.dumps({"grid": GRID, "types": meta}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
