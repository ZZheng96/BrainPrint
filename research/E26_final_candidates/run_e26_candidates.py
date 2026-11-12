"""E26：最终候选装配（RA 导向 + 距离型拒识）。

与 E24 的区别：
  - 不做「疑似扰动就拒识」的门控（E25A 已证伪该解释）；
  - 用对抗增广（clean + SNR30 噪声 1:1）作为身份分支，提升稳定性；
  - 拒识只用距离型新颖度（到最近注册类中心的距离），它在 Test 上方向正确（AUROC 0.656），
    阈值由 Train 内层伪未知标定。

输出三个工作点，供按官方公式权衡：
  reject5   ：等价于拒识 5% 训练样本
  reject20  ：20%
  reject50  ：50%
Test 仍无标签，AUC/F1/C/RA/ASafeR/U 全部为 N/A，只给无标签审计与代理分。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.open_set_improvement.core import COV_COLUMNS
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

CACHE = ROOT / "BrainPrint_cache"
FEATURE_CACHE = CACHE / "e25_robust"
OUT = Path(__file__).resolve().parent
REJECT_RATES = (0.01, 0.02, 0.05, 0.20, 0.50)


def margin(scores: np.ndarray) -> np.ndarray:
    top = np.partition(scores, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def fit(features: np.ndarray, labels: np.ndarray):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300))
    model.fit(features, labels)
    return model


def nearest_distance(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(data**2, axis=1, keepdims=True)
        - 2 * data @ centroids.T
        + np.sum(centroids**2, axis=1)[None, :]
    )
    return np.sqrt(np.maximum(squared.min(axis=1), 0.0))


def main() -> None:
    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial"])
    subjects = metadata["subject"].to_numpy()
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    test_meta = pd.read_csv(CACHE / "test_metadata.csv", usecols=["sample", "session", "task"])
    known = (test_meta["session"] == "session2").to_numpy()

    train_clean = np.asarray(np.load(CACHE / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32)
    train_noisy = np.asarray(np.load(FEATURE_CACHE / "train_cov_snr30.npy"), dtype=np.float32)
    test = np.asarray(np.load(CACHE / "test_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32)

    features = np.concatenate([train_clean, train_noisy])
    labels = np.concatenate([subjects, subjects])
    model = fit(features, labels)
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]

    # 类中心只用既不在内层也不在外层的样本估计，避免标定泄漏
    centroid_mask = ~inner & ~forward
    z_train = scaler.transform(train_clean).astype(np.float32)
    centroids = np.stack([
        z_train[centroid_mask & (subjects == c)].mean(0) for c in classifier.classes_
    ])
    z_test = scaler.transform(test).astype(np.float32)

    inner_distance = nearest_distance(z_train[inner], centroids)
    outer_distance = nearest_distance(z_train[forward], centroids)
    test_distance = nearest_distance(z_test, centroids)

    scores = model.decision_function(test)
    predicted = classifier.classes_[scores.argmax(1)]
    margin_test = margin(scores)
    margin_inner = margin(model.decision_function(train_clean[inner]))

    report = {"thresholds": {}, "operating_points": {}}
    for rate in REJECT_RATES:
        threshold = float(np.quantile(inner_distance, 1 - rate))
        keep = test_distance <= threshold
        candidate = np.where(keep, predicted, 0)
        identity_score = np.where(keep, 1.0, 0.0)

        # Train 侧同口径的 C 与 RA 代理（用外层已知样本，按同一阈值拒识）
        outer_keep = outer_distance <= threshold
        outer_scores = model.decision_function(train_clean[forward])
        outer_pred = classifier.classes_[outer_scores.argmax(1)]
        outer_true = subjects[forward]
        c_proxy = float(np.mean(np.where(outer_keep, outer_pred, 0) == outer_true))
        noisy_scores = model.decision_function(train_noisy[forward])
        noisy_pred = classifier.classes_[noisy_scores.argmax(1)]
        ra_proxy = float(np.mean(noisy_pred == outer_true))

        entry = {
            "distance_threshold": threshold,
            "train_outer_accept": float(outer_keep.mean()),
            "c_proxy_outer": c_proxy,
            "ra_proxy_noisy": ra_proxy,
            "s2_accept": float(keep[known].mean()),
            "s1_reject": float((~keep[~known]).mean()),
            "zero_count": int((candidate == 0).sum()),
            "zero_session1": int((candidate[~known] == 0).sum()),
            "zero_session2": int((candidate[known] == 0).sum()),
        }
        report["operating_points"][f"reject{int(rate*100)}"] = entry
        pd.DataFrame({"Sample": test_meta["sample"], "Label": candidate}).to_csv(
            OUT / f"candidate_reject{int(rate*100)}.csv", index=False
        )
        print(f"reject{int(rate*100):2d}% thr={threshold:7.2f} | Test S2 accept={entry['s2_accept']:.3f} "
              f"S1 reject={entry['s1_reject']:.3f} | Train C代理={c_proxy:.4f} RA代理={ra_proxy:.4f} | "
              f"zeros {entry['zero_count']}", flush=True)

    report["audit"] = {
        "auroc_s2_vs_s1_distance": float(roc_auc_score(known, -test_distance)),
        "auroc_s2_vs_s1_margin": float(roc_auc_score(known, margin_test)),
        "test_s2_distance_p50": float(np.median(test_distance[known])),
        "test_s1_distance_p50": float(np.median(test_distance[~known])),
        "train_inner_distance_p50": float(np.median(inner_distance)),
        "test_predict_counts_session2": np.bincount(
            predicted[known], minlength=21)[1:].tolist(),
    }
    print(f"\nAUROC(S2 vs S1)：距离 {report['audit']['auroc_s2_vs_s1_distance']:.4f}，"
          f"margin {report['audit']['auroc_s2_vs_s1_margin']:.4f}")
    print(f"距离中位数：Train 内层 {report['audit']['train_inner_distance_p50']:.2f}，"
          f"S2 {report['audit']['test_s2_distance_p50']:.2f}，S1 {report['audit']['test_s1_distance_p50']:.2f}")

    (OUT / "e26_candidates.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
