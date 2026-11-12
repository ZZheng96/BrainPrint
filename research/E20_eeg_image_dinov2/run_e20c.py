"""E20C：五频带图像表示 + 图像/协方差特征融合。

1. topomap5 / covariance5 两种保留全部 5 频带的表示，与 E20A 同协议线性探针
2. 图像特征与既往协方差特征拼接，检验互补性
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
from src.eeg_image_dinov2.image_encoders import encode
from src.open_set_improvement.core import COV_COLUMNS

DATA = ROOT / "Brainprint_Competition"
CACHE = ROOT / "BrainPrint_cache"
OUT = ROOT / "research" / "E20_eeg_image_dinov2"
NEW_METHODS = ("topomap5", "covariance5")


def encode_path(args: tuple[str, str]) -> np.ndarray:
    path, method = args
    data, _ = load_eeg(path)
    return encode(data, method)


def build_image_cache(method: str, metadata: pd.DataFrame, workers: int) -> np.ndarray:
    directory = CACHE / "eeg_image_dinov2"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"train_{method}.npy"
    shape = (len(metadata), 3, 224, 224)
    if path.exists():
        cached = np.load(path, mmap_mode="r")
        if cached.shape == shape:
            print(f"image cache: reuse {path.name}", flush=True)
            return cached
    images = np.empty(shape, dtype=np.uint8)
    paths = [(str(DATA / p), method) for p in metadata["path"]]
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, image in enumerate(pool.map(encode_path, paths, chunksize=16), 1):
            images[index - 1] = np.clip(image * 255, 0, 255).astype(np.uint8)
            if index % 2000 == 0 or index == len(paths):
                print(f"{method}: {index}/{len(paths)} | {time.time()-started:.0f}s", flush=True)
    np.save(path, images)
    return np.load(path, mmap_mode="r")


def forward_mask(metadata: pd.DataFrame) -> np.ndarray:
    max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    return metadata["trial"].to_numpy() > 0.8 * max_trial


def probe(features: np.ndarray, subjects: np.ndarray, forward: np.ndarray) -> dict:
    all_subjects = np.unique(subjects)
    groups = np.array_split(np.random.default_rng(42).permutation(all_subjects), 5)
    rows = []
    for fold, unknown_subjects in enumerate(groups, 1):
        known = np.isin(subjects, np.setdiff1d(all_subjects, unknown_subjects))
        fit = known & ~forward
        known_eval = np.flatnonzero(known & forward)
        unknown_eval = np.flatnonzero(np.isin(subjects, unknown_subjects))
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        model.fit(features[fit], subjects[fit])
        known_scores = model.decision_function(features[known_eval])
        unknown_scores = model.decision_function(features[unknown_eval])
        classes = model.classes_

        def margin(scores):
            top = np.partition(scores, -2, axis=1)[:, -2:]
            return top[:, 1] - top[:, 0]

        known_conf, unknown_conf = margin(known_scores), margin(unknown_scores)
        threshold = float(np.quantile(known_conf, 0.05))
        known_pred = classes[known_scores.argmax(axis=1)].copy()
        known_pred[known_conf < threshold] = 0
        unknown_pred = classes[unknown_scores.argmax(axis=1)].copy()
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
        })
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("C", "identity_macro_f1", "U", "unknown_auroc", "unknown_f1")
    }
    return {"folds": rows, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(NEW_METHODS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=OUT / "e20c_five_band_fusion.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("本实验要求 CUDA GPU")
    device = torch.device("cuda")
    metadata = pd.read_csv(CACHE / "train_metadata.csv")
    subjects = metadata["subject"].to_numpy()
    forward = forward_mask(metadata)

    # 既往协方差特征（1,026 维），作为融合对照
    cov_features = np.asarray(
        np.load(CACHE / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32
    )
    print(f"covariance features {cov_features.shape}", flush=True)

    results = {"covariance_only": probe(cov_features, subjects, forward)}
    print(f"covariance_only: {results['covariance_only']['summary']}", flush=True)

    for method in args.methods:
        print(f"\n=== {method} ===", flush=True)
        images = build_image_cache(method, metadata, args.workers)
        started = time.time()
        features = extract_features(images, device, args.batch_size)
        print(f"DINOv2 features {features.shape} in {time.time()-started:.0f}s", flush=True)
        np.save(CACHE / "eeg_image_dinov2" / f"train_{method}_dinov2.npy", features)

        results[f"{method}_image_only"] = probe(features, subjects, forward)
        print(f"  image_only: {results[f'{method}_image_only']['summary']}", flush=True)

        fused = np.concatenate([features, cov_features], axis=1)
        results[f"{method}_fused"] = probe(fused, subjects, forward)
        print(f"  fused: {results[f'{method}_fused']['summary']}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "protocol": {
            "outer": "5-fold subject holdout; 4 unknown, 16 registered",
            "known_eval": "last 20% trials per registered subject/task",
            "backbone": "frozen vit_base_patch14_dinov2.lvd142m, img_size=224",
            "classifier": "StandardScaler + LogisticRegression(C=1.0)",
            "rejection": "top1-top2 margin, 5th percentile of known confidence",
            "covariance_features": "6-band log-covariance, 1026 dims (COV_COLUMNS)",
            "gpu": torch.cuda.get_device_name(0),
        },
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved {args.output}", flush=True)


if __name__ == "__main__":
    main()
