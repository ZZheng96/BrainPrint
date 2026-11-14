"""E139: feature-level multi-view fusion + PCA dimension / prototype count sweep.

Clean protocol (unchanged from E136-E138):
  - per-task standardisation and PCA basis use Train only (fit block, frac<=0.6)
  - deployed prototypes use the whole Train set
  - the reference distribution is Train out-of-fold (5 blocks), the threshold is a
    quantile of it; the LOSO threshold is a quantile of the enrolled known block
  - Test is only ever scored

Objective: (LOSO rejection + Test S1 rejection + Test S2 acceptance)/3 > 0.85
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

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
    w, V = np.linalg.eigh(xc.T @ xc / max(len(xc) - 1, 1))
    return mu, V[:, np.argsort(-w)][:, : min(dim, x.shape[1])]


def kmeans_t(x, k, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    if len(x) <= k:
        return x.clone()
    c = x[rng.choice(len(x), size=k, replace=False)].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        new = torch.stack([x[a == j].mean(0) if (a == j).any() else c[j] for j in range(k)])
        if torch.allclose(new, c, atol=1e-6):
            break
        c = new
    return c


def build_prototypes(feat, subjects, tasks, mask, classes, k):
    protos, keys = [], []
    for task in TASKS:
        for c in classes:
            sel = mask & (tasks == task) & (subjects == c)
            if sel.sum() == 0:
                continue
            block = torch.as_tensor(np.asarray(feat[sel], dtype=np.float32), device=DEVICE)
            for centre in kmeans_t(block, k, seed=int(c) * 17).cpu().numpy():
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
            q = torch.as_tensor(np.asarray(feat[ch], dtype=np.float32), device=DEVICE)
            out[ch] = (-torch.cdist(q, p)).max(1).values.cpu().numpy()
    return out


def main():
    meta = pd.read_csv(ROOT / CFG["data"]["train_metadata"])
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)
    tm = pd.read_csv(ROOT / CFG["data"]["test_metadata"])
    is_s1 = tm.session.astype(str).str.casefold().eq("session1").to_numpy()
    test_tasks = tm.task.to_numpy()

    fit_rows = f <= CFG["fit_fraction"]
    blocks5 = np.clip((f * 5).astype(int), 0, 4)
    y_test = np.r_[np.zeros(int(is_s1.sum())), np.ones(int((~is_s1).sum()))]

    # load + per-task standardise every view once
    cache = {}
    for name, (ptr, pte) in CFG["views"].items():
        a = np.load(ROOT / ptr).astype(np.float32)
        b = np.load(ROOT / pte).astype(np.float32)
        for t in TASKS:
            m = tasks == t
            mu, sd = a[m].mean(0), a[m].std(0) + 1e-6
            a[m] = (a[m] - mu) / sd
            q = test_tasks == t
            b[q] = (b[q] - mu) / sd
        cache[name] = (a, b)
        print(f"loaded {name} {a.shape} {b.shape}", flush=True)

    rows = []
    for set_name, views in CFG["sets"].items():
        A = np.concatenate([cache[v][0] for v in views], axis=1)
        B = np.concatenate([cache[v][1] for v in views], axis=1)
        print(f"{set_name}: {A.shape[1]} dims", flush=True)
        for dim in CFG["pca_dims"]:
            mu, V = pca_basis(np.asarray(A[fit_rows], dtype=np.float64), dim)
            trp = ((A.astype(np.float64) - mu) @ V).astype(np.float32)
            tep = ((B.astype(np.float64) - mu) @ V).astype(np.float32)
            for k in CFG["prototype_k"]:
                protos, keys = build_prototypes(trp, subjects, tasks,
                                                np.ones(len(trp), bool), CLASSES, k)
                te = neg_d1(tep, protos, keys, test_tasks)
                ref = np.full(len(trp), np.nan, dtype=np.float32)
                for b in range(5):
                    hold = blocks5 == b
                    p2, k2 = build_prototypes(trp, subjects, tasks, ~hold, CLASSES, k)
                    s = neg_d1(trp, p2, k2, tasks)
                    ref[hold] = s[hold]
                loso = []
                for held in CLASSES:
                    enrolled = np.setdiff1d(CLASSES, [held])
                    pm = np.isin(subjects, enrolled)
                    ki = pm & (f > .6) & (f <= .8)
                    ui = (subjects == held) & (f > .6) & (f <= .8)
                    if ui.sum() == 0:
                        continue
                    p2, k2 = build_prototypes(trp, subjects, tasks, pm, enrolled, k)
                    s = neg_d1(trp, p2, k2, tasks)
                    loso.append((s[ki], s[ui]))
                auc = float(roc_auc_score(y_test, np.r_[te[is_s1], te[~is_s1]]))
                shift = float(np.nanmedian(te) - np.nanmedian(ref)) / (float(np.nanstd(ref)) + 1e-9)
                for a in CFG["acceptance_grid"]:
                    th = float(np.nanquantile(ref, 1 - a))
                    s2 = float(np.mean(te[~is_s1] >= th))
                    s1 = float(np.mean(te[is_s1] < th))
                    rejs = [float(np.mean(u < np.quantile(kn, 1 - a))) for kn, u in loso]
                    rows.append({"set": set_name, "pca_dim": dim, "k": k,
                                 "acceptance_rate": a, "test_auc": auc,
                                 "median_shift_sd": shift,
                                 "LOSO_rejection": float(np.mean(rejs)),
                                 "Test_S1_rejection": s1, "Test_S2_acceptance": s2,
                                 "three_way_mean": (float(np.mean(rejs)) + s1 + s2) / 3.0})
                print(f"  dim={dim} k={k} AUC {auc:.4f} best "
                      f"{max(r['three_way_mean'] for r in rows):.4f}", flush=True)
            del trp, tep
        del A, B

    rows.sort(key=lambda r: -r["three_way_mean"])
    best = rows[0]
    (OUT / "results.json").write_text(json.dumps(
        {"experiment": CFG["experiment"], "clean": True, "best": best, "curve": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")

    seen, lines = set(), ["# E139 Feature-level fusion, PCA dimension and prototype count", "",
                          "目标：三项均值 (LOSO + Test S1 拒识 + Test S2 接受)/3 > 0.85，全程不使用 Test 侧统计。",
                          "特征级融合：多视图按任务标准化后拼接，PCA 基底与原型只用 Train（原型用全量 Train）。", "",
                          f"**当前最好：{best['set']} / PCA {best['pca_dim']} / k={best['k']} / "
                          f"接受率 {best['acceptance_rate']:.2f} → 均值 {best['three_way_mean']:.4f}**", "",
                          "| 视图集 | PCA 维 | k | 接受率 | 诊断 AUC | LOSO 拒识 | Test S1 拒识 | Test S2 接受 | 三项均值 | 中位偏移 |",
                          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows[:24]:
        lines.append(f"| {r['set']} | {r['pca_dim']} | {r['k']} | {r['acceptance_rate']:.2f} | "
                     f"{r['test_auc']:.4f} | {r['LOSO_rejection']:.4f} | "
                     f"{r['Test_S1_rejection']:.4f} | {r['Test_S2_acceptance']:.4f} | "
                     f"{r['three_way_mean']:.4f} | {r['median_shift_sd']:+.2f} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]

    # per-set summary of the best configuration
    lines += ["", "## 每个视图集的最好结果", "",
              "| 视图集 | 最好架构 | 诊断 AUC | LOSO | S1 拒识 | S2 接受 | 三项均值 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for s in CFG["sets"]:
        sub = [r for r in rows if r["set"] == s]
        if not sub:
            continue
        b = sub[0]
        lines.append(f"| {s} | PCA {b['pca_dim']} / k={b['k']} / a={b['acceptance_rate']:.2f} | "
                     f"{b['test_auc']:.4f} | {b['LOSO_rejection']:.4f} | "
                     f"{b['Test_S1_rejection']:.4f} | {b['Test_S2_acceptance']:.4f} | "
                     f"{b['three_way_mean']:.4f} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
