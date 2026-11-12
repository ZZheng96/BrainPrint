"""E28B：DINO 特征上的开集拒识改进（无泄露版）。

修正 E28A 的两个问题：
  1. 信号选择不能用 Test AUROC（那是泄露）。改为只用 Train 内层伪未知的 AUROC 选择。
  2. correlation 视图 Test 距离 167 vs Train 21.8，跨会话偏移过大，单独排除。

核心发现（E28A）：DINO covariance5 特征上，S1 的距离中位数 54.27 远大于 S2 的 20.94，
差距（33.3）比协方差原始特征（34.33 vs 31.26，差距 3.1）大一个数量级。
这说明 DINO 特征对「未注册者」的判别力其实更强，只是 E20 用错了判据（margin）。

本脚本：
  - 只用 Train 内层伪未知选择信号与阈值；
  - 给出 cov5 / cov5_topo5 的完整工作点；
  - 生成候选。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

CACHE = ROOT / "BrainPrint_cache"
DINO = CACHE / "eeg_image_dinov2"
OUT = Path(__file__).resolve().parent


def nearest_distance(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(data**2, axis=1, keepdims=True)
        - 2 * data @ centroids.T
        + np.sum(centroids**2, axis=1)[None, :]
    )
    return np.sqrt(np.maximum(squared.min(axis=1), 0.0))


def within_class_whitening(train_z: np.ndarray, labels: np.ndarray, shrinkage: float = 0.1):
    classes = np.unique(labels)
    centered = np.zeros_like(train_z)
    for label in classes:
        mask = labels == label
        centered[mask] = train_z[mask] - train_z[mask].mean(0)
    scatter = centered.T @ centered / max(len(train_z) - len(classes), 1)
    scatter = (1 - shrinkage) * scatter + shrinkage * np.trace(scatter) / scatter.shape[0] * np.eye(scatter.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    return (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T


def main() -> None:
    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial"])
    subjects = metadata["subject"].to_numpy()
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    test_meta = pd.read_csv(CACHE / "test_metadata.csv", usecols=["sample", "session"])
    known = (test_meta["session"] == "session2").to_numpy()

    train_cov = np.load(DINO / "train_covariance5_dinov2.npy").astype(np.float32)
    test_cov = np.load(DINO / "test_covariance5_dinov2.npy").astype(np.float32)
    train_topo = np.load(DINO / "train_topomap5_dinov2.npy").astype(np.float32)
    test_topo = np.load(DINO / "test_topomap5_dinov2.npy").astype(np.float32)

    feature_sets = {
        "cov5": (train_cov, test_cov),
        "cov5_topo5": (np.concatenate([train_cov, train_topo], 1), np.concatenate([test_cov, test_topo], 1)),
    }

    report = {}
    for name, (train_x, test_x) in feature_sets.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300))
        model.fit(train_x, subjects)
        scaler = model.named_steps["standardscaler"]
        classifier = model.named_steps["logisticregression"]
        train_z = scaler.transform(train_x).astype(np.float32)
        test_z = scaler.transform(test_x).astype(np.float32)

        centroid_mask = ~inner & ~forward
        centroids = np.stack([train_z[centroid_mask & (subjects == c)].mean(0) for c in classifier.classes_])
        whitening = within_class_whitening(train_z, subjects)
        train_w = (train_z @ whitening).astype(np.float32)
        test_w = (test_z @ whitening).astype(np.float32)
        centroids_w = np.stack([train_w[centroid_mask & (subjects == c)].mean(0) for c in classifier.classes_])

        neighbors = NearestNeighbors(n_neighbors=10).fit(train_z)
        pca = PCA(n_components=min(64, train_z.shape[1]), random_state=0).fit(train_z)

        candidates = {
            "dist": (nearest_distance(train_z, centroids), nearest_distance(test_z, centroids)),
            "dist_wccn": (nearest_distance(train_w, centroids_w), nearest_distance(test_w, centroids_w)),
            "knn": (neighbors.kneighbors(train_z)[0].mean(1), neighbors.kneighbors(test_z)[0].mean(1)),
            "pca": (
                np.linalg.norm(train_z - pca.inverse_transform(pca.transform(train_z)), axis=1),
                np.linalg.norm(test_z - pca.inverse_transform(pca.transform(test_z)), axis=1),
            ),
        }

        # 只用 Train 内层伪未知选择信号：内层已知 vs 内层留出被试
        inner_index = np.flatnonzero(inner)
        inner_subjects = subjects[inner_index]
        unique_inner = np.unique(inner_subjects)
        pseudo_unknown = np.isin(inner_subjects, unique_inner[::2])
        selection = {}
        for signal_name, (train_value, _) in candidates.items():
            column = train_value[inner_index]
            selection[signal_name] = float(roc_auc_score((~pseudo_unknown).astype(int), -column))
        best = max(selection, key=selection.get)
        print(f"{name}: 内层选择 AUROC {({k: round(v,3) for k,v in selection.items()})} -> best={best}", flush=True)

        train_value, test_value = candidates[best]
        entry = {
            "selection_auroc": selection,
            "best_signal": best,
            "test_auroc_s2_vs_s1": float(roc_auc_score(known, -test_value)),
            "train_inner_p50": float(np.median(train_value[inner])),
            "test_s2_p50": float(np.median(test_value[known])),
            "test_s1_p50": float(np.median(test_value[~known])),
            "points": {},
        }
        predicted = classifier.classes_[model.decision_function(test_x).argmax(1)]
        for rate in (0.01, 0.02, 0.05, 0.10, 0.20, 0.30):
            threshold = float(np.quantile(train_value[inner], 1 - rate))
            keep = test_value <= threshold
            candidate = np.where(keep, predicted, 0)
            entry["points"][f"reject{int(rate*100)}"] = {
                "threshold": threshold,
                "s2_accept": float(keep[known].mean()),
                "s1_reject": float((~keep[~known]).mean()),
                "zero_count": int((candidate == 0).sum()),
            }
            pd.DataFrame({"Sample": test_meta["sample"], "Label": candidate}).to_csv(
                OUT / f"candidate_{name}_{best}_reject{int(rate*100)}.csv", index=False
            )
        report[name] = entry
        print(f"{name}: Test AUROC={entry['test_auroc_s2_vs_s1']:.4f} | "
              f"train {entry['train_inner_p50']:.2f} S2 {entry['test_s2_p50']:.2f} "
              f"S1 {entry['test_s1_p50']:.2f}", flush=True)
        for key, value in entry["points"].items():
            print(f"   {key:10s} S2 accept={value['s2_accept']:.3f} S1 reject={value['s1_reject']:.3f} "
                  f"zeros={value['zero_count']}", flush=True)

    (OUT / "e28b_dino_openset.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
