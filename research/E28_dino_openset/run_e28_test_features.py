"""生成 Test 的 topomap5 / correlation DINO 特征（纯 EEG，不读 session）。"""
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
METHODS = ("topomap5", "correlation")


def encode_path(payload) -> np.ndarray:
    path, method = payload
    data, _ = load_eeg(path)
    return encode(data, method)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA")
    device = torch.device("cuda")
    metadata = pd.read_csv(CACHE / "test_metadata.csv", usecols=["sample", "task", "path"])
    directory = CACHE / "eeg_image_dinov2"
    paths = [str(DATA / p) for p in metadata["path"]]

    for method in METHODS:
        image_path = directory / f"test_{method}.npy"
        shape = (len(metadata), 3, 224, 224)
        if image_path.exists() and np.load(image_path, mmap_mode="r").shape == shape:
            print(f"reuse {image_path.name}", flush=True)
        else:
            images = np.empty(shape, dtype=np.uint8)
            started = time.time()
            with ProcessPoolExecutor(max_workers=4) as pool:
                payloads = [(path, method) for path in paths]
                for index, image in enumerate(pool.map(encode_path, payloads, chunksize=16), 1):
                    images[index - 1] = np.clip(image * 255, 0, 255).astype(np.uint8)
                    if index % 4000 == 0 or index == len(paths):
                        print(f"{method}: {index}/{len(paths)} | {time.time()-started:.0f}s", flush=True)
            np.save(image_path, images)

        feature_path = directory / f"test_{method}_dinov2.npy"
        if feature_path.exists():
            print(f"reuse {feature_path.name}", flush=True)
            continue
        images = np.load(image_path, mmap_mode="r")
        started = time.time()
        features = extract_features(images, device, 64)
        np.save(feature_path, features)
        print(f"{method} DINOv2 {features.shape} in {time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
