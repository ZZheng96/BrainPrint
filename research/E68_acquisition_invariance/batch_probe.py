"""E68 诊断 2：Test 是否为「批次 × 分组」交叉设计？

E68 诊断 1 发现 S1 和 S2 各自双峰（silhouette 0.8054 / 0.5674），
且双峰方向与 S1/S2 方向余弦 -0.61。

本诊断检验：
  Q1  S1 的双峰与 S2 的双峰是否对应同一组批次？
      （用 L2 特征做 4 类聚类，看是否形成 2×2 交叉）
  Q2  若存在批次，S1/S2 的差异在「同批次内」还剩多少？
      （这是关键：若同批次内 S1/S2 仍可分，说明差异是身份属性；
        若同批次内不可分，说明差异是批次混杂）
  Q3  批次能否用逐样本特征识别？（若能，可做逐样本批次归一化）

全部只用无标签结构 + 离线 session 审计。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CACHE = ROOT / "BrainPrint_cache"


def cv_auroc(features: np.ndarray, labels: np.ndarray, folds: int = 5) -> float:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    oof = np.zeros(len(labels), dtype=np.float64)
    for train_idx, test_idx in skf.split(features, labels):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        model.fit(features[train_idx], labels[train_idx])
        oof[test_idx] = model.decision_function(features[test_idx])
    return float(roc_auc_score(labels, oof))


def main() -> None:
    test_meta = pd.read_csv(CACHE / "test_metadata.csv")
    session = test_meta["session"].astype(str).str.strip().str.casefold()
    codes = session.map({"1": "S1", "s1": "S1", "session1": "S1",
                         "2": "S2", "s2": "S2", "session2": "S2"})
    test_meta["grp"] = codes
    is_s1 = (test_meta["grp"] == "S1").to_numpy()
    is_s2 = (test_meta["grp"] == "S2").to_numpy()

    layers = dict(np.load(CACHE / "e63_root_cause" / "test_layers.npz"))
    l2 = layers["L2_log_cov"]
    norms = np.linalg.norm(l2, axis=1)

    report = {}

    # ---------- Q1 4 类聚类：是否形成 2×2 交叉 ----------
    print("Q1 用范数做 4 类聚类，看 S1/S2 与批次的交叉")
    km4 = KMeans(n_clusters=4, n_init=20, random_state=42).fit(norms.reshape(-1, 1))
    order = np.argsort(km4.cluster_centers_.ravel())
    cross = pd.crosstab(km4.labels_, test_meta["grp"]).reindex(order)
    cross.index = [f"batch{i+1}(c={km4.cluster_centers_.ravel()[o]:.1f})" for i, o in enumerate(order)]
    print(cross)
    report["q1_cross_4cluster"] = cross.to_dict()
    sil4 = float(silhouette_score(norms.reshape(-1, 1), km4.labels_))
    report["q1_silhouette_4cluster"] = sil4
    print(f"  silhouette = {sil4:.4f}")

    # ---------- Q2 同批次内 S1/S2 是否仍可分 ----------
    print("\nQ2 同批次内 S1/S2 可分性（关键）")
    # 用 2 类聚类定义批次（低/高范数）
    km2 = KMeans(n_clusters=2, n_init=20, random_state=42).fit(norms.reshape(-1, 1))
    low_batch = km2.labels_ == int(np.argmin(km2.cluster_centers_.ravel()))
    high_batch = ~low_batch
    report["q2_within_batch"] = {}
    for name, mask in (("low_batch", low_batch), ("high_batch", high_batch)):
        sub_s1 = mask & is_s1
        sub_s2 = mask & is_s2
        if sub_s1.sum() < 50 or sub_s2.sum() < 50:
            print(f"  {name}: 样本不足（S1={sub_s1.sum()}, S2={sub_s2.sum()}）")
            continue
        x = np.concatenate([l2[sub_s2], l2[sub_s1]])
        y = np.concatenate([np.ones(sub_s2.sum()), np.zeros(sub_s1.sum())])
        auroc = cv_auroc(x, y)
        report["q2_within_batch"][name] = {
            "n_s1": int(sub_s1.sum()), "n_s2": int(sub_s2.sum()),
            "auroc_s2_vs_s1": auroc,
        }
        print(f"  {name}: N(S1)={sub_s1.sum()} N(S2)={sub_s2.sum()} "
              f"AUROC(S2 vs S1)={auroc:.4f}")

    # 对照：全量
    x_all = np.concatenate([l2[is_s2], l2[is_s1]])
    y_all = np.concatenate([np.ones(is_s2.sum()), np.zeros(is_s1.sum())])
    auroc_all = cv_auroc(x_all, y_all)
    report["q2_full_auroc"] = auroc_all
    print(f"  对照 全量 AUROC = {auroc_all:.4f}")

    # ---------- Q3 批次能否用逐样本特征识别 ----------
    print("\nQ3 批次（低/高范数）能否用逐样本特征识别")
    l0 = layers["L0_raw_stats"]
    l1 = layers["L1_band_power"]
    for name, x in (("L0_raw_stats", l0), ("L1_band_power", l1)):
        auroc = cv_auroc(x, low_batch.astype(int))
        report[f"q3_batch_auroc_{name}"] = auroc
        print(f"  {name}: AUROC(批次) = {auroc:.4f}")

    # ---------- 补充：批次内 S1/S2 的范数分布 ----------
    print("\n补充：各批次内 S1/S2 范数中位")
    for name, mask in (("low_batch", low_batch), ("high_batch", high_batch)):
        s1_med = float(np.median(norms[mask & is_s1])) if (mask & is_s1).any() else float("nan")
        s2_med = float(np.median(norms[mask & is_s2])) if (mask & is_s2).any() else float("nan")
        report[f"norm_{name}"] = {"s1_median": s1_med, "s2_median": s2_med}
        print(f"  {name}: S1 中位 {s1_med:.2f}，S2 中位 {s2_med:.2f}")

    (OUT / "batch_probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved {OUT / 'batch_probe.json'}", flush=True)


if __name__ == "__main__":
    main()
