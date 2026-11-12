"""E23A：空间参考与通道混合层面的跨会话对齐检验。

E22 已排除「幅度/频带功率」类解释：相关矩阵对通道增益与频带缩放严格不变，但 Test 分数比仍 0.54。
本实验检验剩下的假设——跨会话偏移在**空间结构**（参考电极、通道间线性混合）。

做法（全部逐样本、归纳式，P 为固定的物理参考矩阵，不用 Test 统计）：
  1. 诊断：比较 Train 与 Test S2 的逐频带平均协方差，看差值是否低秩、是否由公共参考方向主导；
  2. 评估：对 I / CAR / Cz 参考 / 拉普拉斯 四种参考，重算 log 协方差特征，跑严格嵌套闭集 + Test 诊断。
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

from src.cross_session.features import (
    N_BANDS,
    N_CHANNELS,
    log_from_cov,
    to_matrices,
    vectorize,
)
from src.eeg_image_dinov2.image_encoders import CHANNEL_ORDER, CHANNEL_POSITIONS
from src.open_set_improvement.core import COV_COLUMNS
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

CACHE = ROOT / "BrainPrint_cache"
OUT = Path(__file__).resolve().parent
CHUNK = 3000


def reference_matrix(mode: str) -> np.ndarray:
    """返回 18×18 的参考矩阵 P，使变换后数据为 x' = P x。"""
    eye = np.eye(N_CHANNELS)
    if mode == "identity":
        return eye
    if mode == "car":
        return eye - np.ones((N_CHANNELS, N_CHANNELS)) / N_CHANNELS
    if mode == "cz":
        cz = CHANNEL_ORDER.index("Cz")
        selector = np.zeros((N_CHANNELS, N_CHANNELS))
        selector[:, cz] = 1.0
        return eye - selector
    if mode == "laplacian":
        positions = np.array([CHANNEL_POSITIONS[name] for name in CHANNEL_ORDER])
        distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        neighbors = np.argsort(distances, axis=1)[:, :3]
        adjacency = np.zeros((N_CHANNELS, N_CHANNELS))
        adjacency[np.arange(N_CHANNELS)[:, None], neighbors] = 1.0 / 3.0
        return eye - adjacency
    raise ValueError(mode)


def transform_vectors(vectors: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """对 log 协方差向量施加空间参考：C' = P C P^T，再取 log。"""
    matrices = to_matrices(vectors)
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    covariances = (eigenvectors * np.exp(eigenvalues)[..., None, :]) @ np.swapaxes(eigenvectors, -1, -2)
    covariances = matrix @ covariances @ matrix.T
    log_matrices = log_from_cov(covariances)
    return vectorize(log_matrices)


def build_features(raw: np.ndarray, mode: str) -> np.ndarray:
    matrix = reference_matrix(mode)
    chunks = []
    for start in range(0, len(raw), CHUNK):
        chunks.append(transform_vectors(raw[start : start + CHUNK], matrix))
    return np.concatenate(chunks)


def mean_covariance(raw: np.ndarray) -> np.ndarray:
    """逐频带平均协方差（按 trace 归一后平均），用于诊断。"""
    totals = np.zeros((N_BANDS, N_CHANNELS, N_CHANNELS), dtype=np.float64)
    for start in range(0, len(raw), CHUNK):
        matrices = to_matrices(raw[start : start + CHUNK])
        eigenvalues, eigenvectors = np.linalg.eigh(matrices)
        covariances = (eigenvectors * np.exp(eigenvalues)[..., None, :]) @ np.swapaxes(eigenvectors, -1, -2)
        trace = np.trace(covariances, axis1=-2, axis2=-1)[..., None, None]
        totals += np.sum(covariances / np.maximum(trace, 1e-12), axis=0)
    return totals / len(raw)


def margin(scores: np.ndarray) -> np.ndarray:
    top = np.partition(scores, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def fit(features: np.ndarray, labels: np.ndarray):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300))
    model.fit(features, labels)
    return model


def nested(features: np.ndarray, subjects: np.ndarray, metadata: pd.DataFrame) -> dict:
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    groups = np.array_split(np.random.default_rng(42).permutation(np.unique(subjects)), 5)
    rows = []
    for fold, unknown in enumerate(groups, 1):
        known_subjects = np.setdiff1d(np.unique(subjects), unknown)
        known = np.isin(subjects, known_subjects)
        fit_mask = known & ~forward
        inner_order = np.random.default_rng(100 + fold).permutation(known_subjects)
        confidences = []
        for inner_unknown in np.array_split(inner_order, 2):
            enrolled = np.setdiff1d(known_subjects, inner_unknown)
            mask_known = np.isin(subjects, enrolled)
            model = fit(features[mask_known & ~forward & ~inner], subjects[mask_known & ~forward & ~inner])
            confidences.append(margin(model.decision_function(features[np.flatnonzero(mask_known & inner)])))
        threshold = float(np.quantile(np.concatenate(confidences), 0.05))

        evaluate = np.flatnonzero(known & forward)
        unknown_eval = np.flatnonzero(np.isin(subjects, unknown))
        model = fit(features[fit_mask], subjects[fit_mask])
        classes = model.classes_
        scores = model.decision_function(features[evaluate])
        unknown_scores = model.decision_function(features[unknown_eval])
        conf, unknown_conf = margin(scores), margin(unknown_scores)
        pred = classes[scores.argmax(1)].copy()
        pred[conf < threshold] = 0
        unknown_pred = classes[unknown_scores.argmax(1)].copy()
        unknown_pred[unknown_conf < threshold] = 0
        detection_true = np.r_[np.ones(len(evaluate)), np.zeros(len(unknown_eval))]
        detection_conf = np.r_[conf, unknown_conf]
        rows.append({
            "closed_acc": float(np.mean(classes[scores.argmax(1)] == subjects[evaluate])),
            "errors": int(np.sum(classes[scores.argmax(1)] != subjects[evaluate])),
            "C": float(np.mean(pred == subjects[evaluate])),
            "U": float(np.mean(unknown_pred == 0)),
            "auroc": float(roc_auc_score(detection_true, detection_conf)),
        })
    summary = {key: float(np.mean([row[key] for row in rows])) for key in ("closed_acc", "C", "U", "auroc")}
    summary["errors"] = int(sum(row["errors"] for row in rows))
    return summary


def main() -> None:
    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial"])
    subjects = metadata["subject"].to_numpy()
    inner = inner_holdout(metadata)
    test_meta = pd.read_csv(CACHE / "test_metadata.csv", usecols=["session"])
    known = (test_meta["session"] == "session2").to_numpy()
    train_raw = np.asarray(np.load(CACHE / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32)
    test_raw = np.asarray(np.load(CACHE / "test_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32)

    # 1. 均值协方差差值诊断
    train_mean = mean_covariance(train_raw)
    test_mean = mean_covariance(test_raw[known])
    diagnostics = []
    for band in range(N_BANDS):
        delta = test_mean[band] - train_mean[band]
        eigenvalues, eigenvectors = np.linalg.eigh((delta + delta.T) / 2)
        order = np.argsort(-np.abs(eigenvalues))
        top = eigenvectors[:, order[0]]
        ones = np.ones(N_CHANNELS) / np.sqrt(N_CHANNELS)
        diagnostics.append({
            "band": band,
            "trace_ratio_test_over_train": float(np.trace(test_mean[band]) / np.trace(train_mean[band])),
            "relative_delta_norm": float(np.linalg.norm(delta) / np.linalg.norm(train_mean[band])),
            "top_eigenvalue_share": float(np.abs(eigenvalues[order[0]]) / np.sum(np.abs(eigenvalues))),
            "top_eigenvector_cosine_with_common_mode": float(abs(np.dot(top, ones))),
        })

    # 2. 四种参考的评估
    results = {}
    for mode in ("identity", "car", "cz", "laplacian"):
        train_features = build_features(train_raw, mode)
        test_features = build_features(test_raw, mode)
        nested_summary = nested(train_features, subjects, metadata)
        model = fit(train_features, subjects)
        inner_scores = model.decision_function(train_features[inner])
        inner_order = np.sort(inner_scores, axis=1)
        inner_max, inner_margin = inner_order[:, -1], inner_order[:, -1] - inner_order[:, -2]
        test_scores = model.decision_function(test_features)
        test_order = np.sort(test_scores, axis=1)
        test_max, test_margin = test_order[:, -1], test_order[:, -1] - test_order[:, -2]
        threshold = float(np.quantile(inner_margin, 0.05))
        results[mode] = {
            "nested": nested_summary,
            "max_ratio": float(np.median(test_max[known]) / np.median(inner_max)),
            "margin_ratio": float(np.median(test_margin[known]) / np.median(inner_margin)),
            "auroc_s2_vs_s1": float(roc_auc_score(known, test_max)),
            "s2_accept_q05": float(np.mean(test_margin[known] >= threshold)),
        }
        print(f"{mode:10s} closed={nested_summary['closed_acc']:.4f} err={nested_summary['errors']:3d} "
              f"C={nested_summary['C']:.4f} U={nested_summary['U']:.4f} AUROC={nested_summary['auroc']:.4f} | "
              f"max_ratio={results[mode]['max_ratio']:.2f} S2accept={results[mode]['s2_accept_q05']:.3f} "
              f"AUROC(S2vsS1)={results[mode]['auroc_s2_vs_s1']:.3f}", flush=True)

    (OUT / "e23a_spatial_reference.json").write_text(
        json.dumps({"mean_covariance_diagnostics": diagnostics, "references": results},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n均值协方差差值诊断")
    for row in diagnostics:
        print(f"band {row['band']}: trace_ratio={row['trace_ratio_test_over_train']:.3f} "
              f"rel_delta={row['relative_delta_norm']:.3f} top_share={row['top_eigenvalue_share']:.3f} "
              f"common_mode_cos={row['top_eigenvector_cosine_with_common_mode']:.3f}")


if __name__ == "__main__":
    main()
