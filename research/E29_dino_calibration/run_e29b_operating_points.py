"""E29B：用正确的内层伪未知标定，重新扫描 DINO 开集工作点。

E29A 修正了 E28 的标定错误：内层伪未知的 AUROC 实际是 0.6369（不是 0.45–0.51），
与 Test 的 0.7065 只差 0.07，说明内层标定是可信的。

本脚本：
  1. 用「留出被试」构造内层伪未知，得到已知/未知的距离分布；
  2. 阈值取内层已知距离的 q 分位（q 越小越宽松）；
  3. 报告内层伪未知拒识率与 Test 上 S1 拒识率的一致性；
  4. 生成候选。

只用 DINO covariance5 特征，不使用 Test 标签。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

CACHE = ROOT / "BrainPrint_cache"
DINO = CACHE / "eeg_image_dinov2"
OUT = Path(__file__).resolve().parent
QUANTILES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)


def nearest_distance(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(data**2, axis=1, keepdims=True)
        - 2 * data @ centroids.T
        + np.sum(centroids**2, axis=1)[None, :]
    )
    return np.sqrt(np.maximum(squared.min(axis=1), 0.0))


def fit(train_x, labels):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300))
    model.fit(train_x, labels)
    return model


def main() -> None:
    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial"])
    subjects = metadata["subject"].to_numpy()
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    test_meta = pd.read_csv(CACHE / "test_metadata.csv", usecols=["sample", "session"])
    known = (test_meta["session"] == "session2").to_numpy()

    train_x = np.load(DINO / "train_covariance5_dinov2.npy").astype(np.float32)
    test_x = np.load(DINO / "test_covariance5_dinov2.npy").astype(np.float32)

    # 内层标定：5 折留被试，每折再分两半做伪未知
    inner_known_distance, inner_unknown_distance = [], []
    for fold, unknown in enumerate(np.array_split(np.random.default_rng(42).permutation(np.unique(subjects)), 5), 1):
        known_subjects = np.setdiff1d(np.unique(subjects), unknown)
        is_known = np.isin(subjects, known_subjects)
        fit_mask = is_known & ~forward
        for inner_unknown in np.array_split(np.random.default_rng(100 + fold).permutation(known_subjects), 2):
            enrolled = np.setdiff1d(known_subjects, inner_unknown)
            enrolled_mask = np.isin(subjects, enrolled)
            inner_fit = enrolled_mask & ~forward & ~inner
            model = fit(train_x[inner_fit], subjects[inner_fit])
            scaler = model.named_steps["standardscaler"]
            classifier = model.named_steps["logisticregression"]
            z = scaler.transform(train_x).astype(np.float32)
            centers = np.stack([z[inner_fit & (subjects == c)].mean(0) for c in classifier.classes_])
            inner_known_distance.append(nearest_distance(z[np.flatnonzero(enrolled_mask & inner)], centers))
            inner_unknown_distance.append(
                nearest_distance(z[np.flatnonzero(np.isin(subjects, inner_unknown) & ~forward)], centers)
            )
    inner_known_distance = np.concatenate(inner_known_distance)
    inner_unknown_distance = np.concatenate(inner_unknown_distance)
    inner_auroc = float(roc_auc_score(
        np.r_[np.ones(len(inner_known_distance)), np.zeros(len(inner_unknown_distance))],
        -np.r_[inner_known_distance, inner_unknown_distance],
    ))
    print(f"内层 AUROC = {inner_auroc:.4f} | 已知 p50={np.median(inner_known_distance):.2f} "
          f"伪未知 p50={np.median(inner_unknown_distance):.2f}", flush=True)

    # 全量训练 + Test 打分
    model = fit(train_x, subjects)
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    centroid_mask = ~inner & ~forward
    z_train = scaler.transform(train_x).astype(np.float32)
    centers = np.stack([z_train[centroid_mask & (subjects == c)].mean(0) for c in classifier.classes_])
    test_z = scaler.transform(test_x).astype(np.float32)
    test_distance = nearest_distance(test_z, centers)
    predicted = classifier.classes_[model.decision_function(test_x).argmax(1)]
    test_auroc = float(roc_auc_score(known, -test_distance))
    print(f"Test AUROC = {test_auroc:.4f} | S2 p50={np.median(test_distance[known]):.2f} "
          f"S1 p50={np.median(test_distance[~known]):.2f}", flush=True)

    report = {"inner_auroc": inner_auroc, "test_auroc": test_auroc, "points": {}}
    print("\n%-10s %10s %10s %10s %8s" % ("工作点", "内层伪未知拒识", "S2接受", "S1拒识", "0标签"))
    for q in QUANTILES:
        threshold = float(np.quantile(inner_known_distance, 1 - q))
        inner_reject = float(np.mean(inner_unknown_distance > threshold))
        keep = test_distance <= threshold
        candidate = np.where(keep, predicted, 0)
        entry = {
            "threshold": threshold,
            "inner_pseudo_unknown_reject": inner_reject,
            "s2_accept": float(keep[known].mean()),
            "s1_reject": float((~keep[~known]).mean()),
            "zero_count": int((candidate == 0).sum()),
        }
        report["points"][f"q{int(q*100):02d}"] = entry
        pd.DataFrame({"Sample": test_meta["sample"], "Label": candidate}).to_csv(
            OUT / f"candidate_q{int(q*100):02d}.csv", index=False
        )
        print("%-10s %10.3f %10.3f %10.3f %8d" % (
            f"q{int(q*100):02d}", inner_reject, entry["s2_accept"], entry["s1_reject"], entry["zero_count"]))

    (OUT / "e29b_operating_points.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
