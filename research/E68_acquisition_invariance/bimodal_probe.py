"""E68 诊断 1：S2 双峰的两簇在哪些特征上不同？据此设计采集条件模拟。

E67 发现 S2 范数双峰（中心 23.4 / 28.1，计数 7,106 / 12,112）。
本诊断回答：
  Q1  两簇在 L0/L1/L2 各层上的差异有多大？（效应量）
  Q2  两簇在任务、样本顺序上是否均衡？（排除任务/时间混杂）
  Q3  两簇的差异方向是否与 S1/S2 差异方向一致？（若一致，说明双峰就是 S1/S2 差异的来源）
  Q4  S1 落在哪一簇附近？（若 S1 更接近低范数簇，说明"未注册"与"低质量采集"混淆）

全部只用无标签结构 + 离线 session 审计。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CACHE = ROOT / "BrainPrint_cache"


def effect_size(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐维 Cohen's d"""
    mean_diff = a.mean(0) - b.mean(0)
    pooled = np.sqrt((a.var(0) + b.var(0)) / 2).clip(1e-12)
    return mean_diff / pooled


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

    # S2 内部 2-means
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(norms[is_s2].reshape(-1, 1))
    centers = km.cluster_centers_.ravel()
    low_cluster = int(np.argmin(centers))
    s2_idx = np.flatnonzero(is_s2)
    s2_low = s2_idx[km.labels_ == low_cluster]
    s2_high = s2_idx[km.labels_ != low_cluster]

    report = {}
    print(f"S2 低范数簇 N={len(s2_low)}，高范数簇 N={len(s2_high)}")

    # Q1 各层效应量
    print(f"\nQ1 两簇在各层的差异（|Cohen's d| 均值 / 最大）")
    for layer in ("L0_raw_stats", "L1_band_power", "L2_log_cov"):
        x = layers[layer]
        d = effect_size(x[s2_high], x[s2_low])
        report[f"q1_{layer}"] = {
            "mean_abs_d": float(np.abs(d).mean()),
            "max_abs_d": float(np.abs(d).max()),
            "n_dims_abs_d_gt_1": int((np.abs(d) > 1).sum()),
            "n_dims": int(len(d)),
        }
        print(f"  {layer:16s} mean|d|={np.abs(d).mean():.4f} max|d|={np.abs(d).max():.4f} "
              f"|d|>1 维数={int((np.abs(d)>1).sum())}/{len(d)}")

    # Q2 任务与顺序均衡
    print(f"\nQ2 两簇的任务分布与样本顺序")
    task_low = test_meta["task"].to_numpy()[s2_low]
    task_high = test_meta["task"].to_numpy()[s2_high]
    task_dist = pd.DataFrame({
        "low": pd.Series(task_low).value_counts(normalize=True),
        "high": pd.Series(task_high).value_counts(normalize=True),
    }).fillna(0).round(4)
    print(task_dist)
    report["q2_task_distribution"] = task_dist.to_dict()
    report["q2_row_index"] = {
        "low_median": float(np.median(s2_low)),
        "high_median": float(np.median(s2_high)),
        "low_q25_q75": [float(np.quantile(s2_low, 0.25)), float(np.quantile(s2_low, 0.75))],
        "high_q25_q75": [float(np.quantile(s2_high, 0.25)), float(np.quantile(s2_high, 0.75))],
    }
    print(f"  行号中位：低簇 {np.median(s2_low):.0f}，高簇 {np.median(s2_high):.0f}")

    # Q3 双峰差异方向 vs S1/S2 差异方向
    print(f"\nQ3 双峰差异方向与 S1/S2 差异方向的余弦相似度")
    d_bimodal = l2[s2_high].mean(0) - l2[s2_low].mean(0)
    d_s1s2 = l2[is_s2].mean(0) - l2[is_s1].mean(0)
    cos = float(d_bimodal @ d_s1s2 / (np.linalg.norm(d_bimodal) * np.linalg.norm(d_s1s2) + 1e-12))
    report["q3_cosine_bimodal_vs_s1s2"] = cos
    print(f"  cos = {cos:.4f}")

    # Q4 S1 落在哪一簇附近
    print(f"\nQ4 S1 与两簇的距离")
    s1_norms = norms[is_s1]
    report["q4_s1_norm"] = {
        "median": float(np.median(s1_norms)),
        "q25": float(np.quantile(s1_norms, 0.25)),
        "q75": float(np.quantile(s1_norms, 0.75)),
    }
    report["q4_s2_low_norm"] = {
        "median": float(np.median(norms[s2_low])),
        "q25": float(np.quantile(norms[s2_low], 0.25)),
        "q75": float(np.quantile(norms[s2_low], 0.75)),
    }
    report["q4_s2_high_norm"] = {
        "median": float(np.median(norms[s2_high])),
        "q25": float(np.quantile(norms[s2_high], 0.25)),
        "q75": float(np.quantile(norms[s2_high], 0.75)),
    }
    print(f"  S1 范数中位 {np.median(s1_norms):.2f} "
          f"[{np.quantile(s1_norms,0.25):.2f}, {np.quantile(s1_norms,0.75):.2f}]")
    print(f"  S2 低簇中位 {np.median(norms[s2_low]):.2f} "
          f"[{np.quantile(norms[s2_low],0.25):.2f}, {np.quantile(norms[s2_low],0.75):.2f}]")
    print(f"  S2 高簇中位 {np.median(norms[s2_high]):.2f} "
          f"[{np.quantile(norms[s2_high],0.25):.2f}, {np.quantile(norms[s2_high],0.75):.2f}]")

    # 补充：S1 内部是否也双峰
    km_s1 = KMeans(n_clusters=2, n_init=10, random_state=42).fit(s1_norms.reshape(-1, 1))
    from sklearn.metrics import silhouette_score
    sil_s1 = float(silhouette_score(s1_norms.reshape(-1, 1), km_s1.labels_))
    report["q4_s1_bimodality_silhouette"] = sil_s1
    print(f"  S1 内部双峰 silhouette = {sil_s1:.4f}")

    (OUT / "bimodal_probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved {OUT / 'bimodal_probe.json'}", flush=True)


if __name__ == "__main__":
    main()
