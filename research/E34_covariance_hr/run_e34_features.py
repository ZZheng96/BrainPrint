"""E34：生成高分辨率协方差图（covariance_hr）的 DINOv2 特征。

与 E20 的 covariance5 区别：每个频带 18×18 矩阵直接双线性放大到 224×224，
而不是 5 张 74×74 小图平铺到 224×224。信息密度提升约 9 倍。
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eeg_features import load_eeg
from src.eeg_image_dinov2.backbone import extract_features
from src.eeg_image_dinov2.image_encoders import encode

DATA = ROOT / "Brainprint_Competition"
CACHE = ROOT / "BrainPrint_cache"
DIRECTORY = CACHE / "eeg_image_dinov2"
METHOD = "covariance_hr"


def encode_path(path: str) -> np.ndarray:
    data, _ = load_eeg(path)
    return encode(data, METHOD)


def build(split: str, metadata: pd.DataFrame, device: torch.device) -> None:
    image_path = DIRECTORY / f"{split}_{METHOD}.npy"
    shape = (len(metadata), 3, 224, 224)
    if image_path.exists() and np.load(image_path, mmap_mode="r").shape == shape:
        print(f"reuse {image_path.name}", flush=True)
    else:
        images = np.empty(shape, dtype=np.uint8)
        paths = [str(DATA / p) for p in metadata["path"]]
        started = time.time()
        with ProcessPoolExecutor(max_workers=4) as pool:
            for index, image in enumerate(pool.map(encode_path, paths, chunksize=16), 1):
                images[index - 1] = np.clip(image * 255, 0, 255).astype(np.uint8)
                if index % 4000 == 0 or index == len(paths):
                    print(f"{split} image: {index}/{len(paths)} | {time.time()-started:.0f}s", flush=True)
        np.save(image_path, images)

    feature_path = DIRECTORY / f"{split}_{METHOD}_dinov2.npy"
    if feature_path.exists():
        print(f"reuse {feature_path.name}", flush=True)
        return
    images = np.load(image_path, mmap_mode="r")
    started = time.time()
    features = extract_features(images, device, 64)
    np.save(feature_path, features)
    print(f"{split} DINOv2 {features.shape} in {time.time()-started:.0f}s", flush=True)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA")
    device = torch.device("cuda")
    train_meta = pd.read_csv(CACHE / "train_metadata.csv", usecols=["path"])
    test_meta = pd.read_csv(CACHE / "test_metadata.csv", usecols=["path"])
    build("train", train_meta, device)
    build("test", test_meta, device)


if __name__ == "__main__":
    main()
