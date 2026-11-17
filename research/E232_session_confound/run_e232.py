"""E232: Test 的 session 划分是否与身份/注册状态混叠的诊断。

只回答一个问题：Test 的 session1 是否只由少数几个（未注册）被试构成，
session2 是否由 20 名注册用户构成。用 train 拟合的标准化 + PCA 空间，
在 Test 上做无监督聚类，比较两段的簇数与轮廓系数。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
TASKS = ["EC", "EO", "Imagery", "SSVEP"]
SEED = 20261117


def pca_fit(x, dim):
    mu = x.mean(0)
    _, _, vt = np.linalg.svd(x - mu, full_matrices=False)
    basis = vt[:dim].T
    return mu, basis


def kmeans(x, k, seed, iters=60):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        d = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1) if len(x) <= 4000 else None
        if d is None:  # 分块避免爆内存
            assign = np.empty(len(x), dtype=np.int32)
            for i in range(0, len(x), 2000):
                blk = x[i:i + 2000]
                assign[i:i + 2000] = ((blk[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
        else:
            assign = d.argmin(1)
        for j in range(k):
            m = assign == j
            if m.any():
                c[j] = x[m].mean(0)
    return assign, c


def silhouette(x, assign, max_n=1200, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), min(max_n, len(x)), replace=False)
    xs, a = x[idx], assign[idx]
    ks = np.unique(a)
    if len(ks) < 2:
        return -1.0
    d = np.sqrt(((xs[:, None, :] - xs[None, :, :]) ** 2).sum(-1))
    s = []
    for i in range(len(xs)):
        same = a == a[i]
        same[i] = False
        if same.sum() == 0:
            continue
        ai = d[i, same].mean()
        bi = min(d[i, a == k].mean() for k in ks if k != a[i] and (a == k).any())
        s.append((bi - ai) / max(ai, bi) if max(ai, bi) > 0 else 0.0)
    return float(np.mean(s))


def main():
    meta_tr = pd.read_csv("BrainPrint_cache/train_metadata.csv")
    meta_te = pd.read_csv("BrainPrint_cache/test_metadata.csv")
    xtr = np.load("BrainPrint_cache/e65_correlation_spd/train_correlation_logspd.npy").astype(np.float32)
    xte = np.load("research/E121_pca_sweep/test_logspd.npy").astype(np.float32)

    tr_t = meta_tr.task.to_numpy(); te_t = meta_te.task.to_numpy()
    sess = meta_te.session.astype(str).str.casefold().to_numpy()
    is1 = sess == "session1"; is2 = sess == "session2"

    # 只用 train 拟合 per-task 标准化 + 全局 PCA96
    parts_tr, parts_te = [], []
    for task in TASKS:
        m = tr_t == task
        mu, sd = xtr[m].mean(0), xtr[m].std(0) + 1e-6
        ztr = (xtr[m] - mu) / sd
        zte = (xte[te_t == task] - mu) / sd
        order = np.flatnonzero(te_t == task)
        parts_tr.append(ztr); parts_te.append((order, zte))
    ztr_all = np.concatenate(parts_tr)
    mu, basis = pca_fit(ztr_all, 96)
    proj_te = np.full((len(xte), 96), np.nan, np.float32)
    for order, z in parts_te:
        proj_te[order] = (z - mu) @ basis

    result = {"pca_dim": 96, "seed": SEED, "session1": {}, "session2": {}}
    for name, mask, kmax in [("session1", is1, 14), ("session2", is2, 34)]:
        x = proj_te[mask].astype(np.float64)
        rng = np.random.default_rng(SEED)
        if len(x) > 3000:
            x = x[rng.choice(len(x), 3000, replace=False)]
        rows = []
        best = (-1.0, None)
        for k in range(2, kmax + 1):
            assign, _ = kmeans(x, k, SEED + k)
            s = silhouette(x, assign)
            sizes = np.bincount(assign, minlength=k)
            rows.append({"k": k, "silhouette": round(s, 4),
                         "min_cluster": int(sizes.min()), "max_cluster": int(sizes.max()),
                         "size_cv": round(float(sizes.std() / max(sizes.mean(), 1e-9)), 3)})
            if s > best[0]:
                best = (s, k)
        result[name] = {"n_total": int(mask.sum()), "n_used": int(len(x)),
                        "best_k_by_silhouette": best[1], "best_silhouette": round(best[0], 4),
                        "curve": rows}
        print(name, "best k =", best[1], "silhouette", round(best[0], 4), flush=True)

    (OUT / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = ["# E232 Test session 与注册状态混叠诊断", "",
             "只用 Train 拟合 per-task 标准化 + PCA96，然后在 Test 两段各自做无监督 k-means（3000 行子样本，固定种子），",
             "比较轮廓系数最优簇数与簇大小分布。若 session1 的最优簇数接近 5、session2 接近 20，则说明 session 划分与",
             "“未注册 5 人 / 注册 20 人”高度混叠，S1/S2 不是纯生物特征指标。", ""]
    for name in ("session1", "session2"):
        r = result[name]
        lines += [f"## {name}", "",
                  f"- 样本数 {r['n_total']}，子样本 {r['n_used']}",
                  f"- 轮廓系数最优：k={r['best_k_by_silhouette']}，silhouette={r['best_silhouette']}", "",
                  "|k|silhouette|最小簇|最大簇|簇大小CV|", "|---:|---:|---:|---:|---:|"]
        for row in r["curve"]:
            lines.append(f"|{row['k']}|{row['silhouette']}|{row['min_cluster']}|{row['max_cluster']}|{row['size_cv']}|")
        lines.append("")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
