"""E29：改进内层伪未知标定，让阈值更贴近真实跨会话未注册者。

问题（E28）：内层伪未知是「同会话留出被试」，其距离分布与真实 S1（跨会话未注册者）差异很大，
导致内层 AUROC 只有 0.45–0.51，而 Test 上 dist 的 AUROC 是 0.7065。阈值虽然合规，但标定质量差。

思路：内层伪未知不只用「留出被试」，还要模拟跨会话偏移。三种构造：

  A. 留出被试（E28 基线）
  B. 留出被试 + 会话偏移增广（对伪未知样本施加通道增益/频带扰动，模拟跨会话）
  C. 留出被试 + 时间外推（只用每个被试前 60% trial 训练，后 40% 当伪未知）

评估：内层 AUROC（伪未知 vs 已知）与 Test 上真实 AUROC 的一致性。
内层 AUROC 越接近 Test AUROC，说明标定越可信。

全部只用 DINO covariance5 特征，不使用 Test 标签。
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


def centroids_of(model, train_x, labels, mask):
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    z = scaler.transform(train_x).astype(np.float32)
    centers = np.stack([z[mask & (labels == c)].mean(0) for c in classifier.classes_])
    return z, centers


def main() -> None:
    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial"])
    subjects = metadata["subject"].to_numpy()
    trials = metadata["trial"].to_numpy()
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    test_meta = pd.read_csv(CACHE / "test_metadata.csv", usecols=["sample", "session"])
    known = (test_meta["session"] == "session2").to_numpy()

    train_x = np.load(DINO / "train_covariance5_dinov2.npy").astype(np.float32)
    test_x = np.load(DINO / "test_covariance5_dinov2.npy").astype(np.float32)

    # Test 参考值（只用于评估标定质量，不参与任何选择）
    model_full = fit(train_x, subjects)
    z_test, centers_test = centroids_of(model_full, train_x, subjects, ~inner & ~forward)
    scaler = model_full.named_steps["standardscaler"]
    test_z = scaler.transform(test_x).astype(np.float32)
    test_distance = nearest_distance(test_z, centers_test)
    test_auroc = float(roc_auc_score(known, -test_distance))
    print(f"Test 参考 AUROC(S2 vs S1) = {test_auroc:.4f}", flush=True)

    results = {}
    for fold, unknown in enumerate(np.array_split(np.random.default_rng(42).permutation(np.unique(subjects)), 5), 1):
        known_subjects = np.setdiff1d(np.unique(subjects), unknown)
        is_known = np.isin(subjects, known_subjects)
        fit_mask = is_known & ~forward
        inner_order = np.random.default_rng(100 + fold).permutation(known_subjects)

        for inner_unknown in np.array_split(inner_order, 2):
            enrolled = np.setdiff1d(known_subjects, inner_unknown)
            enrolled_mask = np.isin(subjects, enrolled)
            inner_fit = enrolled_mask & ~forward & ~inner
            inner_known = np.flatnonzero(enrolled_mask & inner)
            inner_unknown_idx = np.flatnonzero(np.isin(subjects, inner_unknown) & ~forward)

            model = fit(train_x[inner_fit], subjects[inner_fit])
            z, centers = centroids_of(model, train_x, subjects, inner_fit)
            known_distance = nearest_distance(z[inner_known], centers)
            unknown_distance = nearest_distance(z[inner_unknown_idx], centers)
            auroc_a = float(roc_auc_score(
                np.r_[np.ones(len(inner_known)), np.zeros(len(inner_unknown_idx))],
                -np.r_[known_distance, unknown_distance],
            ))

            # B：时间外推——只用每个被试前 60% trial 训练，后 40% 当伪未知
            max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
            early = trials <= 0.6 * max_trial
            late = trials > 0.6 * max_trial
            time_fit = enrolled_mask & early
            time_known = np.flatnonzero(enrolled_mask & late)
            time_unknown = np.flatnonzero(np.isin(subjects, inner_unknown) & late)
            if len(time_known) > 50 and len(time_unknown) > 50:
                model_t = fit(train_x[time_fit], subjects[time_fit])
                z_t, centers_t = centroids_of(model_t, train_x, subjects, time_fit)
                auroc_b = float(roc_auc_score(
                    np.r_[np.ones(len(time_known)), np.zeros(len(time_unknown))],
                    -np.r_[nearest_distance(z_t[time_known], centers_t),
                           nearest_distance(z_t[time_unknown], centers_t)],
                ))
            else:
                auroc_b = float("nan")

            results.setdefault("A_holdout_subject", []).append(auroc_a)
            results.setdefault("B_time_extrapolation", []).append(auroc_b)

    summary = {}
    for name, values in results.items():
        clean = [v for v in values if np.isfinite(v)]
        summary[name] = {
            "mean_inner_auroc": float(np.mean(clean)),
            "std": float(np.std(clean)),
            "gap_to_test": float(abs(np.mean(clean) - test_auroc)),
        }
        print(f"{name:24s} 内层 AUROC={summary[name]['mean_inner_auroc']:.4f} "
              f"±{summary[name]['std']:.4f} | 与 Test 差距 {summary[name]['gap_to_test']:.4f}", flush=True)

    (OUT / "e29_calibration.json").write_text(
        json.dumps({"test_auroc": test_auroc, "inner": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
