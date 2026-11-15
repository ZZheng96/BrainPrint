"""E159：E143 单一分数的 W 上界分析（不使用任何 Test 统计量做模型，只做上限证明）。

目的：回答"W 是否还能靠阈值/工作点调整达到 0.80"。
结论只依赖一个事实：单一分数 + 单调阈值 => (S1 拒识, S2 接受) 只能沿同一条 ROC 移动。
因此把阈值扫满全范围，取 W 的最大值，即为该分数的可达上界。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TASKS = CFG["tasks"]


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def pca_basis(x, dim):
    mu = x.mean(0)
    xc = x - mu
    w, v = np.linalg.eigh(xc.T @ xc / max(len(xc) - 1, 1))
    return mu, v[:, np.argsort(-w)[: min(dim, x.shape[1])]]


def kmeans_t(x, k, seed, iters=20):
    if len(x) <= k:
        return x
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g, device=DEVICE)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        s = torch.zeros_like(c).index_add_(0, a, x)
        n = torch.zeros(k, device=DEVICE).index_add_(0, a, torch.ones(len(x), device=DEVICE))
        z = torch.where(n[:, None] > 0, s / n.clamp_min(1)[:, None], c)
        if torch.allclose(z, c, atol=1e-6):
            c = z
            break
        c = z
    return c


def build_prototypes(feat, subjects, tasks, mask, classes, k):
    protos, keys = [], []
    for task in TASKS:
        for c in classes:
            sel = mask & (tasks == task) & (subjects == c)
            if not sel.any():
                continue
            block = torch.as_tensor(np.asarray(feat[sel], dtype=np.float32), device=DEVICE)
            for centre in kmeans_t(block, k, int(c) * 17).cpu().numpy():
                protos.append(centre)
                keys.append((task, int(c)))
    return np.stack(protos), keys


def neg_d1(feat, protos, keys, tasks):
    out = np.full(len(feat), np.nan, dtype=np.float32)
    for task in TASKS:
        idx = np.flatnonzero(tasks == task)
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(idx) == 0 or not cols:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        for st in range(0, len(idx), 2048):
            ch = idx[st:st + 2048]
            q = torch.as_tensor(np.asarray(feat[ch], dtype=torch.float32), device=DEVICE)
            out[ch] = (-torch.cdist(q, p)).max(1).values.cpu().numpy()
    return out


def auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    meta = pd.read_csv(ROOT / CFG["data"]["train_metadata"])
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)
    fit_rows = f <= CFG["fit_fraction"]
    blocks5 = np.clip((f * 5).astype(int), 0, 4)
    tm = pd.read_csv(ROOT / CFG["data"]["test_metadata"])
    is_s1 = tm.session.astype(str).str.casefold().eq("session1").to_numpy()
    test_tasks = tm.task.to_numpy()

    a = np.load(ROOT / CFG["data"]["train_features"]).astype(np.float32)
    b = np.load(ROOT / CFG["data"]["test_features"]).astype(np.float32)
    for t in TASKS:
        m = tasks == t
        mu, sd = a[m].mean(0), a[m].std(0) + 1e-6
        a[m] = (a[m] - mu) / sd
        q = test_tasks == t
        b[q] = (b[q] - mu) / sd
    mu, V = pca_basis(np.asarray(a[fit_rows], dtype=np.float64), CFG["pca_dim"])
    a = ((a.astype(np.float64) - mu) @ V).astype(np.float32)
    b = ((b.astype(np.float64) - mu) @ V).astype(np.float32)

    k = int(CFG["prototype_k"])
    protos, keys = build_prototypes(a, subjects, tasks, np.ones(len(a), bool), CLASSES, k)
    te = neg_d1(b, protos, keys, test_tasks)

    oof = np.full(len(a), np.nan, dtype=np.float32)
    for blk in range(5):
        hold = blocks5 == blk
        p2, k2 = build_prototypes(a, subjects, tasks, ~hold, CLASSES, k)
        oof[hold] = neg_d1(a, p2, k2, tasks)[hold]
    ref = oof[np.isfinite(oof)]

    loso = []
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        pm = np.isin(subjects, enrolled)
        ki = pm & (f > .6) & (f <= .8)
        ui = (subjects == held) & (f > .6) & (f <= .8)
        if ui.sum() == 0:
            continue
        p2, k2 = build_prototypes(a, subjects, tasks, pm & fit_rows, enrolled, k)
        s = neg_d1(a, p2, k2, tasks)
        loso.append((s[ki], s[ui]))

    y = np.r_[np.zeros(int(is_s1.sum())), np.ones(int((~is_s1).sum()))]
    diag_auc = auc(y, np.r_[te[is_s1], te[~is_s1]])

    grid = np.unique(np.quantile(np.r_[ref, te], np.linspace(0.001, 0.999, 2000)))
    rows = []
    for th in grid:
        s2 = float(np.mean(te[~is_s1] >= th))
        s1 = float(np.mean(te[is_s1] < th))
        a_rate = float(np.mean(ref >= th))
        lr = float(np.mean([np.mean(u < np.quantile(kn, 1 - a_rate)) for kn, u in loso])) \
            if 0 < a_rate < 1 else 0.0
        rows.append({"threshold": float(th), "accept_rate": a_rate,
                     "LOSO_rejection": lr, "S1_rejection": s1,
                     "S2_acceptance": s2,
                     "W": 0.25 * lr + 0.25 * s1 + 0.5 * s2})
    rows.sort(key=lambda r: -r["W"])
    best = rows[0]
    loso_max = float(max(r["LOSO_rejection"] for r in rows))
    # 乐观上界：LOSO 固定取历史最好值 0.7936，只在 Test 上取最优工作点
    optimistic = max(0.25 * 0.7936 + 0.25 * r["S1_rejection"] + 0.5 * r["S2_acceptance"] for r in rows)
    # 若 S1/S2 必须同时达到 0.80，需要的诊断 AUC 参考点
    sol = [r for r in rows if abs(r["S1_rejection"] - 0.80) < 0.01]
    at80 = min(sol, key=lambda r: abs(r["S2_acceptance"] - 0.80)) if sol else None

    out = {"experiment": "E159_ceiling_analysis", "clean": True, "k": k,
           "diagnostic_AUC": diag_auc,
           "oracle_max_W_single_score": best["W"], "oracle_point": best,
           "LOSO_rejection_max_over_thresholds": loso_max,
           "optimistic_upper_bound_W": float(optimistic),
           "point_with_S1_rejection_0.80": at80,
           "curve_head": rows[:20]}
    (OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2,
                                                 default=lambda o: float(o)), encoding="utf-8")

    lines = ["# E159 单一分数的 W 上界分析", "",
             f"设备：{DEVICE}；E143 协议（E65 → task 标准化 → PCA-{CFG['pca_dim']} → k={k} 原型）。",
             "本实验只做上限证明：不使用任何 Test 统计量建模，"
             "只在已冻结分数上把阈值扫满，看单条 ROC 能到多高。", "",
             f"- Test S1-vs-S2 诊断 AUC：**{diag_auc:.4f}**",
             f"- 扫满全部阈值后的最优 W：**{best['W']:.4f}**"
             f"（a={best['accept_rate']:.3f}，LOSO {best['LOSO_rejection']:.4f}，"
             f"S1 {best['S1_rejection']:.4f}，S2 {best['S2_acceptance']:.4f}）",
             f"- 乐观上界（LOSO 固定 0.7936，Test 工作点任选）：**{float(optimistic):.4f}**",
             f"- S1 拒识恰为 0.80 时，S2 接受最高只有 "
             f"{'(%.4f)' % at80['S2_acceptance'] if at80 else 'N/A'}",
             "", "项目 Test AUC/F1 = N/A（无真实标签）。", "",
             "结论：若乐观上界仍 < 0.80，则该分数无论怎么调阈值都不可能达标，"
             "必须提升排序能力（诊断 AUC），而不是继续换阈值或换归一化。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2, default=lambda o: float(o)))


if __name__ == "__main__":
    main()
