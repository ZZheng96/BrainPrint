"""E67 诊断 2：检验 Test 内部结构，判断 S1/S2 标签假设是否成立。

三个问题：
  Q1  S2 内部是否双峰？（E23 曾发现低范数 7,985 / 高范数 11,233）
      若双峰，说明 S2 混入了两类样本，S1/S2 的二分假设可能不完整。
  Q2  Test 的样本能否按「被试」聚类？若 S2 的 20 个注册者各自成簇，
      说明身份信息存在，只是拒识判据没抓到。
  Q3  S1 与 S2 在「任务 × 被试」交叉表上是否均衡？
      若 S1 只覆盖部分任务，AUROC 会被任务差异污染。

全部只用无标签结构分析 + 离线 session 审计，不参与任何决策。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CACHE = ROOT / "BrainPrint_cache"


def main() -> None:
    test_meta = pd.read_csv(CACHE / "test_metadata.csv")
    session = test_meta["session"].astype(str).str.strip().str.casefold()
    codes = session.map({"1": "S1", "s1": "S1", "session1": "S1",
                         "2": "S2", "s2": "S2", "session2": "S2"})
    if codes.isna().any():
        raise ValueError("未知 Session 取值")
    test_meta["grp"] = codes

    report = {}

    # ---------- Q3 任务 × 分组交叉表 ----------
    cross = pd.crosstab(test_meta["task"], test_meta["grp"])
    cross_pct = pd.crosstab(test_meta["task"], test_meta["grp"], normalize="columns")
    report["task_by_group_counts"] = cross.to_dict()
    report["task_by_group_share"] = cross_pct.round(4).to_dict()
    print("Q3 任务 × 分组交叉表（行=任务，列=分组）")
    print(cross)
    print("\n列内占比：")
    print(cross_pct.round(4))

    # ---------- Q1 S2 内部双峰 ----------
    layers = np.load(CACHE / "e63_root_cause" / "test_layers.npz")
    l2 = layers["L2_log_cov"]
    is_s2 = (test_meta["grp"] == "S2").to_numpy()
    is_s1 = (test_meta["grp"] == "S1").to_numpy()

    # 用 L2 特征的范数作为「样本强度」代理
    norms = np.linalg.norm(l2, axis=1)
    s2_norms = norms[is_s2]
    s1_norms = norms[is_s1]
    # 双峰检测：对 S2 范数做 2-means，看两簇是否分离
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(s2_norms.reshape(-1, 1))
    centers = np.sort(km.cluster_centers_.ravel())
    counts = np.bincount(km.labels_)
    sil = float(silhouette_score(s2_norms.reshape(-1, 1), km.labels_))
    report["q1_s2_norm_bimodality"] = {
        "s2_norm_median": float(np.median(s2_norms)),
        "s1_norm_median": float(np.median(s1_norms)),
        "s2_two_cluster_centers": centers.tolist(),
        "s2_two_cluster_counts": counts.tolist(),
        "s2_silhouette": sil,
        "bimodal": bool(sil > 0.5),
    }
    print(f"\nQ1 S2 范数双峰检测：中心 {centers.round(2).tolist()}，"
          f"计数 {counts.tolist()}，silhouette={sil:.4f} → "
          f"{'双峰' if sil > 0.5 else '单峰'}")

    # ---------- Q2 Test 能否按被试聚类 ----------
    # 用 L2 特征做 PCA 后 KMeans，看 S2 内部是否形成 20 个簇
    pca = PCA(n_components=50, random_state=42).fit(l2)
    reduced = pca.transform(l2)
    report["q2_pca_explained"] = float(pca.explained_variance_ratio_.sum())
    for k in (5, 10, 20, 25):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(reduced)
        sil = float(silhouette_score(reduced, km.labels_, sample_size=5000, random_state=42))
        # 每个簇里 S1/S2 的比例
        s2_share = [float((test_meta["grp"].to_numpy()[km.labels_ == c] == "S2").mean())
                    for c in range(k)]
        report[f"q2_kmeans_k{k}"] = {
            "silhouette": sil,
            "s2_share_per_cluster": [round(v, 3) for v in s2_share],
            "s2_share_std": float(np.std(s2_share)),
        }
        print(f"Q2 KMeans k={k:2d}: silhouette={sil:.4f}，"
              f"各簇 S2 占比 std={np.std(s2_share):.4f}")

    # ---------- 补充：S1/S2 在 L2 特征上的均值差是否低秩 ----------
    mean_diff = l2[is_s2].mean(0) - l2[is_s1].mean(0)
    # 用 S2 的协方差做白化，看差异方向是否被少数主成分解释
    cov = np.cov(l2[is_s2], rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    whitened = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T @ mean_diff
    energy = np.sort(whitened ** 2)[::-1]
    cum = np.cumsum(energy) / energy.sum()
    report["mean_diff_rank"] = {
        "top1_energy": float(cum[0]),
        "top5_energy": float(cum[4]),
        "top10_energy": float(cum[9]),
        "top20_energy": float(cum[19]),
        "n_dims": int(len(cum)),
    }
    print(f"\n补充：S1/S2 均值差（白化后）能量集中度："
          f"top1={cum[0]:.3f} top5={cum[4]:.3f} top10={cum[9]:.3f} top20={cum[19]:.3f}")

    (OUT / "structure.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved {OUT / 'structure.json'}", flush=True)


if __name__ == "__main__":
    main()
