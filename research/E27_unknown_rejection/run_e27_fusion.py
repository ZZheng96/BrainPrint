"""E27：提升 S1（未注册）拒识率，同时保住 S2（注册者）接受率。

口径（项目记忆第 41–42 行）：Test S1 ≈ 未注册者，S2 = 注册者正常 + 对抗样本。
因此目标是：S1 拒识率尽量高，S2 拒识率尽量低。

E26 只用「到最近类中心的距离」，AUROC(S2 vs S1)=0.758。
本实验把多个方向正确的无标签信号做秩融合，看能否把 AUROC 推高：

  dist     到最近注册类中心的欧氏距离（E26）
  knn      到最近 k 个训练样本的平均距离
  pca      在训练 PCA 子空间外的重构残差
  energy   分类分数的负能量（logsumexp）
  margin   分类分数的负 margin

融合权重只用 Train 内层伪未知（留被试）确定，Test 只做审计。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.open_set_improvement.core import COV_COLUMNS
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

CACHE = ROOT / "BrainPrint_cache"
FEATURE_CACHE = CACHE / "e25_robust"
OUT = Path(__file__).resolve().parent


def margin(scores: np.ndarray) -> np.ndarray:
    top = np.partition(scores, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def novelty_signals(train_z, test_z, centroids, model, train_scores, test_scores) -> dict:
    """返回「越大越新颖」的分数。"""
    def nearest(data):
        squared = (
            np.sum(data**2, axis=1, keepdims=True)
            - 2 * data @ centroids.T
            + np.sum(centroids**2, axis=1)[None, :]
        )
        return np.sqrt(np.maximum(squared.min(axis=1), 0.0))

    neighbors = NearestNeighbors(n_neighbors=10).fit(train_z)
    knn_train = neighbors.kneighbors(train_z)[0].mean(1)
    knn_test = neighbors.kneighbors(test_z)[0].mean(1)

    pca = PCA(n_components=64, random_state=0).fit(train_z)
    recon_train = np.linalg.norm(train_z - pca.inverse_transform(pca.transform(train_z)), axis=1)
    recon_test = np.linalg.norm(test_z - pca.inverse_transform(pca.transform(test_z)), axis=1)

    def energy(scores):
        peak = scores.max(axis=1, keepdims=True)
        return -(peak[:, 0] + np.log(np.exp(scores - peak).sum(axis=1)))

    return {
        "dist": (nearest(train_z), nearest(test_z)),
        "knn": (knn_train, knn_test),
        "pca": (recon_train, recon_test),
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

    signals = novelty_signals(
        train_z, test_z, centroids, model,
        model.decision_function(train_clean), model.decision_function(test),
    )

    # 单信号在 Test 上的方向（AUROC 越接近 1 越好）
    single = {}
    for name, (train_value, test_value) in signals.items():
        single[name] = {
            "auroc_s2_vs_s1": float(roc_auc_score(known, -test_value)),
            "train_inner_p50": float(np.median(train_value[inner])),
            "test_s2_p50": float(np.median(test_value[known])),
            "test_s1_p50": float(np.median(test_value[~known])),
        }
        print(f"{name:7s} AUROC(S2 vs S1)={single[name]['auroc_s2_vs_s1']:.4f} | "
              f"train {single[name]['train_inner_p50']:.2f} S2 {single[name]['test_s2_p50']:.2f} "
              f"S1 {single[name]['test_s1_p50']:.2f}", flush=True)

    # 秩融合：只用 Train 内层伪未知确定权重（留一被试的伪未知 vs 已知）
    names = list(signals)
    train_rank = np.stack([rankdata(signals[n][0]) / len(train_clean) for n in names], axis=1)
    test_rank = np.stack([rankdata(signals[n][1]) / len(test) for n in names], axis=1)

    # 内层伪未知：把内层已知样本按被试分两半，一半当伪未知
    inner_index = np.flatnonzero(inner)
    inner_subjects = subjects[inner_index]
    unique_inner = np.unique(inner_subjects)
    pseudo_unknown = np.isin(inner_subjects, unique_inner[::2])
    labels = (~pseudo_unknown).astype(int)
    weights = {}
    for name in names:
        column = train_rank[inner_index, names.index(name)]
        weights[name] = float(roc_auc_score(labels, -column))
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    fused_train = train_rank @ np.array([weights[n] for n in names])
    fused_test = test_rank @ np.array([weights[n] for n in names])
    fused_auroc = float(roc_auc_score(known, -fused_test))
    print(f"\n融合权重 {({k: round(v,3) for k,v in weights.items()})}")
    print(f"融合 AUROC(S2 vs S1) = {fused_auroc:.4f}")

    # 工作点：按 Train 内层分位设阈值
    report = {"single": single, "weights": weights, "fused_auroc": fused_auroc, "points": {}}
    for rate in (0.01, 0.02, 0.05, 0.10, 0.20, 0.30):
        threshold = float(np.quantile(fused_train[inner], 1 - rate))
        keep = fused_test <= threshold
        entry = {
            "train_reject_rate": rate,
            "s2_accept": float(keep[known].mean()),
            "s1_reject": float((~keep[~known]).mean()),
        }
        report["points"][f"reject{int(rate*100)}"] = entry
        print(f"reject{int(rate*100):2d}% | S2 accept={entry['s2_accept']:.3f} S1 reject={entry['s1_reject']:.3f}")

    (OUT / "e27_fusion.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
