"""E253 step 1: cache per-sub-window features for Train and Test.

The frozen E200 model averages the 6 sub-window vectors before PCA.  Keeping the
6 vectors separate lets us test robust aggregation (median / trimmed mean) and
sub-window level voting without recomputing the FFT/correlation pipeline.

Outputs (BrainPrint_cache/e253_subwindow/):
  train_feat.npy  (11304, 6, 855) float32
  test_feat.npy   (22251, 6, 855) float32
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "final/team_hymn"
DATA = ROOT / "Brainprint_Competition"
OUT = ROOT / "BrainPrint_cache/e253_subwindow"
sys.path.insert(0, str(PKG))
import model as M  # noqa: E402


def load(path: Path) -> np.ndarray:
    arr = np.asarray(loadmat(path, variable_names=["data"])["data"], dtype=np.float32).T
    return np.ascontiguousarray(arr)


def load_many(paths) -> np.ndarray:
    with ThreadPoolExecutor(8) as pool:
        return np.stack(list(pool.map(load, paths)))


def subwindow_features(art: dict, x: np.ndarray, chunk: int = 512) -> np.ndarray:
    cfg = art["feature"]
    count, length = int(cfg["n_subwindows"]), int(cfg["subwindow_length"])
    out = []
    for i in range(0, len(x), chunk):
        t = torch.from_numpy(x[i:i + chunk]).to(art["device"])
        parts = [M._subwindow_feature(t[..., w * length:(w + 1) * length], cfg)
                 for w in range(count)]
        out.append(torch.stack(parts, dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    art = M.load_model(str(PKG), "cuda:0" if torch.cuda.is_available() else "cpu")
    for split, meta_name, limit in (("train", "train_metadata.csv", None),
                                    ("test", "test_metadata.csv", None)):
        dst = OUT / f"{split}_feat.npy"
        if dst.is_file():
            print(f"{split}: cached {np.load(dst, mmap_mode='r').shape}")
            continue
        meta = pd.read_csv(ROOT / "BrainPrint_cache" / meta_name)
        paths = [DATA / p for p in meta["path"]]
        t0 = time.time()
        x = load_many(paths)
        print(f"{split}: loaded {x.shape} in {time.time() - t0:.0f}s", flush=True)
        t0 = time.time()
        feat = subwindow_features(art, x)
        np.save(dst, feat)
        print(f"{split}: features {feat.shape} in {time.time() - t0:.0f}s", flush=True)
        del x, feat


if __name__ == "__main__":
    main()
