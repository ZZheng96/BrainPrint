"""从训练4秒Trial提取两个非重叠2秒窗口特征。"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_features import N_FEATURES, extract_features, load_eeg

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Brainprint_Competition"
CACHE = ROOT.parent / "BrainPrint_cache"


def process_parent(path_string: str) -> tuple[np.ndarray, np.ndarray]:
    data, _ = load_eeg(path_string)
    first, _ = extract_features(data[:600])
    second, _ = extract_features(data[600:])
    return first, second


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="train")
    args = parser.parse_args()
    metadata = pd.read_csv(CACHE / f"{args.split}_metadata.csv")
    features = np.empty((len(metadata) * 2, N_FEATURES), dtype=np.float32)
    started = time.time()
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        for index, (first, second) in enumerate(
            pool.map(process_parent, metadata["path"].map(lambda p: str(DATA / p)), chunksize=16),
            1,
        ):
            features[2 * (index - 1)] = first
            features[2 * (index - 1) + 1] = second
            if index % 500 == 0 or index == len(metadata):
                print(f"parents: {index}/{len(metadata)} | {time.time()-started:.1f}s", flush=True)
    output = CACHE / f"{args.split}_window2_features.npy"
    np.save(output, features)
    print(f"saved {features.shape} -> {output}", flush=True)


if __name__ == "__main__":
    main()
