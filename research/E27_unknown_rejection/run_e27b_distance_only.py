"""E27B：只用距离型新颖度，直接优化 S1 拒识率。

E27A 结论：多信号秩融合反而把 AUROC 从 0.758 拉到 0.467，因为权重由内层伪未知确定，
而内层伪未知（同会话留出被试）与真实 S1（跨会话未注册者）分布不同。
因此只用唯一方向正确的 dist 信号。

本脚本给出 S1 拒识率与 S2 接受率的完整权衡曲线，并生成候选。
阈值仍只由 Train 内层伪未知的距离分位决定。
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

from src.open_set_improvement.core import COV_COLUMNS
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

CACHE = ROOT / "BrainPrint_cache"
FEATURE_CACHE = CACHE / "e25_robust"
OUT = Path(__file__).resolve().parent


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
    test_meta = pd.read_csv(CACHE / "test_metadata.csv", usecols=["sample", "session"])
    known = (test_meta["session"] == "session2").to_numpy()

    train_clean = np.asarray(np.load(CACHE / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32)
    train_noisy = np.asarray(np.load(FEATURE_CACHE / "train_cov_snr30.npy"), dtype=np.float32)
    test = np.asarray(np.load(CACHE / "test_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32)

    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300))
    model.fit(np.concatenate([train_clean, train_noisy]), np.concatenate([subjects, subjects]))
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]

    train_z = scaler.transform(train_clean).astype(np.float32)
    test_z = scaler.transform(test).astype(np.float32)
    centroid_mask = ~inner & ~forward
    centroids = np.stack([train_z[centroid_mask & (subjects == c)].mean(0) for c in classifier.classes_])

    inner_distance = nearest_distance(train_z[inner], centroids)
    outer_distance = nearest_distance(train_z[forward], centroids)
    test_distance = nearest_distance(test_z, centroids)
    predicted = classifier.classes_[model.decision_function(test).argmax(1)]

    print(f"AUROC(S2 vs S1) = {roc_auc_score(known, -test_distance):.4f}")
    print(f"距离中位数：Train 内层 {np.median(inner_distance):.2f} | "
          f"Test S2 {np.median(test_distance[known]):.2f} | Test S1 {np.median(test_distance[~known]):.2f}")

    report = {"auroc": float(roc_auc_score(known, -test_distance)), "points": {}}
    print("\n%-10s %8s %8s %8s %8s" % ("工作点", "S2接受", "S1拒识", "0标签", "Train接受"))
    for rate in (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
        threshold = float(np.quantile(inner_distance, 1 - rate))
        keep = test_distance <= threshold
        candidate = np.where(keep, predicted, 0)
        entry = {
            "threshold": threshold,
            "s2_accept": float(keep[known].mean()),
            "s1_reject": float((~keep[~known]).mean()),
            "zero_count": int((candidate == 0).sum()),
            "train_outer_accept": float((outer_distance <= threshold).mean()),
        }
        report["points"][f"reject{int(rate*100)}"] = entry
        pd.DataFrame({"Sample": test_meta["sample"], "Label": candidate}).to_csv(
            OUT / f"candidate_reject{int(rate*100)}.csv", index=False
        )
        print("%-10s %8.3f %8.3f %8d %8.3f" % (
            f"reject{int(rate*100)}%", entry["s2_accept"], entry["s1_reject"],
            entry["zero_count"], entry["train_outer_accept"]))

    (OUT / "e27b_distance_only.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
