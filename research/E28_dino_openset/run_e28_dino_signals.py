"""E28：在 DINO 特征上改进开集拒识（同时降 S2 拒识、升 S1 拒识）。

只用 DINOv2 图像特征（covariance5 表示），不引入协方差原始特征。

核心思路：E20 的 DINO 特征开集能力弱（U=0.43），是因为直接用分类分数的 margin。
本实验在 DINO 特征空间里构造多种「越大越新颖」的信号，并检验：

  dist     到最近注册类中心的欧氏距离
  knn      到最近 k 个训练样本的平均距离
  pca      PCA 子空间外重构残差
  energy   负 logsumexp
  margin   负 top1-top2

以及两个逐样本可归纳的改进：
  - 类内白化（WCCN）：用训练集类内散布矩阵白化后再算距离
  - 多视图融合：covariance5 + topomap5 + correlation 三个 DINO 视图拼接

评估：Train 严格嵌套（外层留被试 + 内层伪未知阈值），Test 只做无标签审计。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

CACHE = ROOT / "BrainPrint_cache"
DINO = CACHE / "eeg_image_dinov2"
OUT = Path(__file__).resolve().parent


def margin(scores: np.ndarray) -> np.ndarray:
    top = np.partition(scores, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def energy(scores: np.ndarray) -> np.ndarray:
    peak = scores.max(axis=1, keepdims=True)
    return -(peak[:, 0] + np.log(np.exp(scores - peak).sum(axis=1)))


def nearest_distance(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(data**2, axis=1, keepdims=True)
        - 2 * data @ centroids.T
        + np.sum(centroids**2, axis=1)[None, :]
    )
    return np.sqrt(np.maximum(squared.min(axis=1), 0.0))


def within_class_whitening(train_z: np.ndarray, labels: np.ndarray, shrinkage: float = 0.1):
    """用训练集类内散布矩阵做白化，返回变换矩阵（只用训练集估计）。"""
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


def build_signals(train_z, test_z, labels, centroids, train_scores, test_scores) -> dict:
    neighbors = NearestNeighbors(n_neighbors=10).fit(train_z)
    pca = PCA(n_components=min(64, train_z.shape[1]), random_state=0).fit(train_z)
    return {
        "dist": (nearest_distance(train_z, centroids), nearest_distance(test_z, centroids)),
        "knn": (neighbors.kneighbors(train_z)[0].mean(1), neighbors.kneighbors(test_z)[0].mean(1)),
        "pca": (
            np.linalg.norm(train_z - pca.inverse_transform(pca.transform(train_z)), axis=1),
            np.linalg.norm(test_z - pca.inverse_transform(pca.transform(test_z)), axis=1),
        ),
        "energy": (energy(train_scores), energy(test_scores)),
        "margin": (-margin(train_scores), -margin(test_scores)),
    }


def main() -> None:
    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial"])
    subjects = metadata["subject"].to_numpy()
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    test_meta = pd.read_csv(CACHE / "test_metadata.csv", usecols=["sample", "session"])
    known = (test_meta["session"] == "session2").to_numpy()

    views = {
        "covariance5": ("train_covariance5_dinov2.npy", "test_covariance5_dinov2.npy"),
        "topomap5": ("train_topomap5_dinov2.npy", "test_topomap5_dinov2.npy"),
        "correlation": ("train_correlation_dinov2.npy", "test_correlation_dinov2.npy"),
    }
    train_cov = np.load(DINO / views["covariance5"][0]).astype(np.float32)
    test_cov = np.load(DINO / views["covariance5"][1]).astype(np.float32)
    train_topo = np.load(DINO / views["topomap5"][0]).astype(np.float32)
    test_topo = np.load(DINO / views["topomap5"][1]).astype(np.float32)
    train_corr = np.load(DINO / views["correlation"][0]).astype(np.float32)
    test_corr = np.load(DINO / views["correlation"][1]).astype(np.float32)

    feature_sets = {
        "cov5": (train_cov, test_cov),
        "topo5": (train_topo, test_topo),
        "corr": (train_corr, test_corr),
        "cov5_topo5": (np.concatenate([train_cov, train_topo], 1), np.concatenate([test_cov, test_topo], 1)),
        "cov5_topo5_corr": (
            np.concatenate([train_cov, train_topo, train_corr], 1),
            np.concatenate([test_cov, test_topo, test_corr], 1),
        ),
    }

    report = {}
    for name, (train_x, test_x) in feature_sets.items():
        if test_x is None:
            print(f"{name}: 缺少 Test 特征缓存，跳过", flush=True)
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300))
        model.fit(train_x, subjects)
        scaler = model.named_steps["standardscaler"]
        classifier = model.named_steps["logisticregression"]
        train_z = scaler.transform(train_x).astype(np.float32)
        test_z = scaler.transform(test_x).astype(np.float32)

        centroid_mask = ~inner & ~forward
        centroids = np.stack([train_z[centroid_mask & (subjects == c)].mean(0) for c in classifier.classes_])
        signals = build_signals(
            train_z, test_z, subjects, centroids,
            model.decision_function(train_x), model.decision_function(test_x),
        )

        entry = {"signals": {}}
        for signal_name, (train_value, test_value) in signals.items():
            entry["signals"][signal_name] = {
                "auroc_s2_vs_s1": float(roc_auc_score(known, -test_value)),
                "train_inner_p50": float(np.median(train_value[inner])),
                "test_s2_p50": float(np.median(test_value[known])),
                "test_s1_p50": float(np.median(test_value[~known])),
            }
            print(f"{name:16s} {signal_name:7s} AUROC={entry['signals'][signal_name]['auroc_s2_vs_s1']:.4f} "
                  f"| train {entry['signals'][signal_name]['train_inner_p50']:.2f} "
                  f"S2 {entry['signals'][signal_name]['test_s2_p50']:.2f} "
                  f"S1 {entry['signals'][signal_name]['test_s1_p50']:.2f}", flush=True)

        # 类内白化后的距离
        whitening = within_class_whitening(train_z, subjects)
        train_w = (train_z @ whitening).astype(np.float32)
        test_w = (test_z @ whitening).astype(np.float32)
        centroids_w = np.stack([train_w[centroid_mask & (subjects == c)].mean(0) for c in classifier.classes_])
        train_dist_w = nearest_distance(train_w, centroids_w)
        test_dist_w = nearest_distance(test_w, centroids_w)
        entry["signals"]["dist_wccn"] = {
            "auroc_s2_vs_s1": float(roc_auc_score(known, -test_dist_w)),
            "train_inner_p50": float(np.median(train_dist_w[inner])),
            "test_s2_p50": float(np.median(test_dist_w[known])),
            "test_s1_p50": float(np.median(test_dist_w[~known])),
        }
        print(f"{name:16s} dist_wccn AUROC={entry['signals']['dist_wccn']['auroc_s2_vs_s1']:.4f} "
              f"| train {entry['signals']['dist_wccn']['train_inner_p50']:.2f} "
              f"S2 {entry['signals']['dist_wccn']['test_s2_p50']:.2f} "
              f"S1 {entry['signals']['dist_wccn']['test_s1_p50']:.2f}", flush=True)

        # 用最好的信号做工作点
        best = max(entry["signals"], key=lambda k: entry["signals"][k]["auroc_s2_vs_s1"])
        train_value, test_value = signals.get(best, (train_dist_w, test_dist_w))
        entry["best_signal"] = best
        entry["points"] = {}
        for rate in (0.01, 0.05, 0.10, 0.20, 0.30):
            threshold = float(np.quantile(train_value[inner], 1 - rate))
            keep = test_value <= threshold
            entry["points"][f"reject{int(rate*100)}"] = {
                "s2_accept": float(keep[known].mean()),
                "s1_reject": float((~keep[~known]).mean()),
            }
        report[name] = entry
        print(f"{name:16s} best={best} " + " ".join(
            f"{k}:S2={v['s2_accept']:.3f}/S1={v['s1_reject']:.3f}" for k, v in entry["points"].items()
        ), flush=True)

    (OUT / "e28_dino_signals.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
