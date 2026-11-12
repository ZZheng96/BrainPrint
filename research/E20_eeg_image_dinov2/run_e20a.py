"""E20A：图像表示可行性——生成图像、提 DINOv2 特征、线性探针。

5 折留被试：每折 4 人作未知，16 人作注册；注册者后 20% Trial 作已知测试。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eeg_features import load_eeg
from src.eeg_image_dinov2.backbone import extract_features
from src.eeg_image_dinov2.image_encoders import ENCODERS, encode

DATA = ROOT / "Brainprint_Competition"
CACHE = ROOT / "BrainPrint_cache"
OUT = ROOT / "research" / "E20_eeg_image_dinov2"
METHODS = ("spectrogram", "correlation", "topomap", "gaf", "mtf")


def encode_path(args: tuple[str, str]) -> np.ndarray:
    path, method = args
    data, _ = load_eeg(path)
    return encode(data, method)


def build_image_cache(method: str, metadata: pd.DataFrame, workers: int) -> np.ndarray:
    """生成并缓存全部训练图像的 uint8 版本。"""
    directory = CACHE / "eeg_image_dinov2"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"train_{method}.npy"
    shape = (len(metadata), 3, 224, 224)
    if path.exists():
        cached = np.load(path, mmap_mode="r")
        if cached.shape == shape:
            print(f"image cache: reuse {path.name}", flush=True)
            return np.asarray(cached)
    images = np.empty(shape, dtype=np.uint8)
    paths = [(str(DATA / p), method) for p in metadata["path"]]
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, image in enumerate(pool.map(encode_path, paths, chunksize=16), 1):
            images[index - 1] = np.clip(image * 255, 0, 255).astype(np.uint8)
            if index % 1000 == 0 or index == len(paths):
                print(f"{method}: {index}/{len(paths)} | {time.time()-started:.0f}s", flush=True)
    np.save(path, images)
    return images


def forward_mask(metadata: pd.DataFrame) -> np.ndarray:
    max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    return metadata["trial"].to_numpy() > 0.8 * max_trial


def linear_probe(
    features: np.ndarray,
    subjects: np.ndarray,
    forward: np.ndarray,
    device: torch.device,
) -> dict:
    """5 折留被试线性探针。"""
    all_subjects = np.unique(subjects)
    groups = np.array_split(np.random.default_rng(42).permutation(all_subjects), 5)
    rows = []
    for fold, unknown_subjects in enumerate(groups, 1):
        known = np.isin(subjects, np.setdiff1d(all_subjects, unknown_subjects))
        fit = known & ~forward
        known_eval = np.flatnonzero(known & forward)
        unknown_eval = np.flatnonzero(np.isin(subjects, unknown_subjects))
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1),
        )
        model.fit(features[fit], subjects[fit])
        known_scores = model.decision_function(features[known_eval])
        unknown_scores = model.decision_function(features[unknown_eval])
        classes = model.classes_
        known_pred = classes[known_scores.argmax(axis=1)]
        unknown_pred = classes[unknown_scores.argmax(axis=1)]
        # 置信度：top1 - top2 margin
        def margin(scores):
            top = np.partition(scores, -2, axis=1)[:, -2:]
            return top[:, 1] - top[:, 0]
        known_conf = margin(known_scores)
        unknown_conf = margin(unknown_scores)
        threshold = float(np.quantile(known_conf, 0.05))
        known_pred = known_pred.copy()
        known_pred[known_conf < threshold] = 0
        unknown_pred = unknown_pred.copy()
        unknown_pred[unknown_conf < threshold] = 0
        detection_true = np.r_[np.ones(len(known_eval)), np.zeros(len(unknown_eval))]
        detection_conf = np.r_[known_conf, unknown_conf]
        rows.append({
            "fold": fold,
            "C": float(np.mean(known_pred == subjects[known_eval])),
            "identity_macro_f1": float(f1_score(
                subjects[known_eval], known_pred, average="macro", zero_division=0
            )),
            "U": float(np.mean(unknown_pred == 0)),
            "unknown_auroc": float(roc_auc_score(detection_true, detection_conf)),
            "unknown_f1": float(f1_score(
                detection_true, detection_conf >= threshold, pos_label=0, zero_division=0
            )),
            "threshold": threshold,
            "n_known": int(len(known_eval)),
            "n_unknown": int(len(unknown_eval)),
        })
        print(f"  fold {fold}: C={rows[-1]['C']:.4f} U={rows[-1]['U']:.4f} "
              f"AUROC={rows[-1]['unknown_auroc']:.4f}", flush=True)
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("C", "identity_macro_f1", "U", "unknown_auroc", "unknown_f1")
    }
    return {"folds": rows, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=OUT / "e20a_linear_probe.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("本实验要求 CUDA GPU")
    device = torch.device("cuda")
    metadata = pd.read_csv(CACHE / "train_metadata.csv")
    subjects = metadata["subject"].to_numpy()
    forward = forward_mask(metadata)
    print(f"train={len(metadata)} subjects={len(np.unique(subjects))} "
          f"known_eval={int((forward & np.isin(subjects, np.unique(subjects))).sum())}", flush=True)

    results = {}
    for method in args.methods:
        print(f"\n=== {method} ===", flush=True)
        images = build_image_cache(method, metadata, args.workers)
        started = time.time()
        features = extract_features(
            images.astype(np.float32) / 255.0, device, args.batch_size
        )
        print(f"DINOv2 features {features.shape} in {time.time()-started:.0f}s", flush=True)
        np.save(CACHE / "eeg_image_dinov2" / f"train_{method}_dinov2.npy", features)
        results[method] = linear_probe(features, subjects, forward, device)
        print(f"{method} summary: {results[method]['summary']}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "protocol": {
            "outer": "5-fold subject holdout; 4 unknown, 16 registered",
            "known_eval": "last 20% trials per registered subject/task",
            "backbone": "frozen vit_base_patch14_dinov2.lvd142m, img_size=224",
            "classifier": "StandardScaler + LogisticRegression(C=1.0)",
            "rejection": "top1-top2 margin, 5th percentile of known confidence",
            "gpu": torch.cuda.get_device_name(0),
        },
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved {args.output}", flush=True)


if __name__ == "__main__":
    main()
