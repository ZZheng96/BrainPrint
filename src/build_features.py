"""提取并缓存训练/测试特征；原始 MAT 文件保持不变。"""
from __future__ import annotations

import argparse
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_features import N_FEATURES, extract_features, load_eeg

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Brainprint_Competition"
CACHE = ROOT.parent / "BrainPrint_cache"


def numeric_key(path: Path) -> int:
    return int(re.search(r"(\d+)", path.stem).group(1))


def process_file(path_string: str) -> tuple[np.ndarray, dict]:
    path = Path(path_string)
    data, metadata = load_eeg(path)
    features, quality = extract_features(data)
    record = {**metadata, **quality, "path": path.relative_to(DATA).as_posix()}
    return features, record


def build_split(split: str, files: list[Path], output_dir: Path, n_jobs: int) -> None:
    started = time.time()
    features = np.empty((len(files), N_FEATURES), dtype=np.float32)
    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for index, (feature, record) in enumerate(
            pool.map(process_file, map(str, files), chunksize=16), 1
        ):
            features[index - 1] = feature
            records.append(record)
            if index % 500 == 0 or index == len(files):
                elapsed = time.time() - started
                print(
                    f"{split}: {index}/{len(files)} | {elapsed:.1f}s | {index / elapsed:.1f} trials/s",
                    flush=True,
                )

    metadata = pd.DataFrame(records)
    if split == "train":
        metadata["subject"] = metadata["path"].str.extract(r"Subject_(\d+)").astype(int)
        metadata["task_folder"] = metadata["path"].str.extract(r"(Task\d+)")[0]
        metadata["trial"] = metadata["path"].str.extract(r"trial_(\d+)").astype(int)
    else:
        metadata["sample"] = metadata["path"].str.extract(r"(test_\d+)")[0]
    metadata.drop(columns=["channels"]).to_csv(output_dir / f"{split}_metadata.csv", index=False)
    np.save(output_dir / f"{split}_features.npy", features)
    print(f"saved {split}: {features.shape} -> {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="train,test", choices=["train", "test", "train,test"])
    parser.add_argument("--output-dir", type=Path, default=CACHE)
    parser.add_argument("--n-jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--limit", type=int, default=0, help="仅用于小样本冒烟检查")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits.split(","):
        if split == "train":
            files = sorted(
                (DATA / "Train").glob("Subject_*/Task*/trial_*.mat"),
                key=lambda p: (
                    int(re.search(r"Subject_(\d+)", str(p)).group(1)),
                    int(re.search(r"Task(\d+)", str(p)).group(1)),
                    numeric_key(p),
                ),
            )
        else:
            files = sorted((DATA / "Test").glob("test_*.mat"), key=numeric_key)
        if args.limit:
            files = files[: args.limit]
        build_split(split, files, args.output_dir, args.n_jobs)


if __name__ == "__main__":
    main()
