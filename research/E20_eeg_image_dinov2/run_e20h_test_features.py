"""生成 Test 的 covariance5 图像与 DINOv2 特征（纯 EEG，不读 session）。"""
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
METHOD = "covariance5"


def encode_path(path: str) -> np.ndarray:
    data, _ = load_eeg(path)
    return encode(data, METHOD)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA")
    device = torch.device("cuda")
    metadata = pd.read_csv(CACHE / "test_metadata.csv", usecols=["sample", "task", "path"])
    directory = CACHE / "eeg_image_dinov2"
    directory.mkdir(parents=True, exist_ok=True)

    image_path = directory / f"test_{METHOD}.npy"
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
                if index % 2000 == 0 or index == len(paths):
                    print(f"{METHOD}: {index}/{len(paths)} | {time.time()-started:.0f}s", flush=True)
        np.save(image_path, images)

    images = np.load(image_path, mmap_mode="r")
    started = time.time()
    features = extract_features(images, device, 64)
    print(f"DINOv2 features {features.shape} in {time.time()-started:.0f}s", flush=True)
    np.save(directory / f"test_{METHOD}_dinov2.npy", features)
    print("done", flush=True)


if __name__ == "__main__":
    main()
