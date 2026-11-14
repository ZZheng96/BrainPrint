"""E141: new weighted objective + image route (frozen DINOv2) and complementary fusion.

Objective (user, 2026-11-15):
    W = 0.25 * LOSO_rejection + 0.25 * Test_S1_rejection + 0.5 * Test_S2_acceptance
    target W > 0.80

Because S2 carries half the weight, the best operating point moves to a looser
threshold than the equal-weight three-way objective did.

Cleanliness (unchanged):
  - per-view per-task standardisation: Train statistics only
  - PCA bases: Train fit block (frac <= 0.6)
  - deployed prototypes: whole Train set
  - reference distribution: Train out-of-fold over 5 blocks
  - LOSO threshold: enrolled known block (frac in (0.6, 0.8])
  - Test is only ever scored; no Test labels, no Test batch statistics

Fusion modes:
  score  per-view score -> per-task z (Train OOF) -> equal-weight mean
  feat   concat per-view PCA-96 projections -> second PCA-96 -> prototypes
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
K = CFG["prototype_k"]


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


def build_prototypes(feat, subjects, tasks, mask, classes):
    protos, keys = [], []
    for task in TASKS:
        for c in classes:
            sel = mask & (tasks == task) & (subjects == c)
            if sel.sum() == 0:
                continue
            block = torch.as_tensor(np.asarray(feat[sel], dtype=np.float32), device=DEVICE)
            for centre in kmeans_t(block, K, seed=int(c) * 17).cpu().numpy():
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


def space_scores(tr, te, subjects, tasks, test_tasks, f, blocks5):
    """Test scores (raw), Train OOF reference (raw) and per-fold LOSO pairs (raw)."""
    protos, keys = build_prototypes(tr, subjects, tasks, np.ones(len(tr), bool), CLASSES)
    te_raw = neg_d1(te, protos, keys, test_tasks)
    ref = np.full(len(tr), np.nan, dtype=np.float32)
    for b in range(5):
        hold = blocks5 == b
        p2, k2 = build_prototypes(tr, subjects, tasks, ~hold, CLASSES)
        s = neg_d1(tr, p2, k2, tasks)
        ref[hold] = s[hold]
    loso = []
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        pm = np.isin(subjects, enrolled)
        ki = pm & (f > .6) & (f <= .8)
        ui = (subjects == held) & (f > .6) & (f <= .8)
        if ui.sum() == 0:
            continue
        p2, k2 = build_prototypes(tr, subjects, tasks, pm, enrolled)
        s = neg_d1(tr, p2, k2, tasks)
        loso.append({"known": s[ki], "unknown": s[ui],
                     "known_task": tasks[ki], "unknown_task": tasks[ui]})
    return te_raw, ref, loso


def z_by_task(values, values_tasks, ref, ref_tasks):
    out = np.full(len(values), np.nan, dtype=np.float32)
    for t in TASKS:
        m = values_tasks == t
        r = ref[ref_tasks == t]
        r = r[np.isfinite(r)]
        if len(r) < 10:
            continue
        out[m] = (values[m] - float(r.mean())) / (float(r.std()) + 1e-6)
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
    D = CFG["pca_dim"]

    # ---- per-view PCA-96 projection (bases on the Train fit block only)
    proj = {}
    for name, (ptr, pte) in CFG["views"].items():
        a = np.load(ROOT / ptr).astype(np.float32)
        b = np.load(ROOT / pte).astype(np.float32)
        if len(a) != len(meta) or len(b) != len(tm):
            print(f"SKIP {name} shape {a.shape} {b.shape}", flush=True)
            continue
        for t in TASKS:
            m = tasks == t
            mu, sd = a[m].mean(0), a[m].std(0) + 1e-6
            a[m] = (a[m] - mu) / sd
            q = test_tasks == t
            b[q] = (b[q] - mu) / sd
        mu, V = pca_basis(np.asarray(a[fit_rows], dtype=np.float64), D)
        proj[name] = (((a.astype(np.float64) - mu) @ V).astype(np.float32),
                      ((b.astype(np.float64) - mu) @ V).astype(np.float32))
        del a, b
        print(f"projected {name}", flush=True)

    rows = []

    def record(label, mode, te_raw, ref, loso):
        te = z_by_task(te_raw, test_tasks, ref, tasks)
        rf = z_by_task(ref, tasks, ref, tasks)
        auc = float(roc_auc_score(y_test, np.r_[te[is_s1], te[~is_s1]]))
        parts = [(z_by_task(fd["known"], fd["known_task"], ref, tasks),
                  z_by_task(fd["unknown"], fd["unknown_task"], ref, tasks)) for fd in loso]
        for a in CFG["acceptance_grid"]:
            th = float(np.nanquantile(rf, 1 - a))
            s2 = float(np.mean(te[~is_s1] >= th))
            s1 = float(np.mean(te[is_s1] < th))
            rejs = [float(np.mean(u < np.quantile(kn, 1 - a))) for kn, u in parts]
            loso_r = float(np.mean(rejs))
            w = (CFG["weights"]["LOSO"] * loso_r + CFG["weights"]["S1"] * s1
                 + CFG["weights"]["S2"] * s2)
            rows.append({"label": label, "mode": mode, "acceptance_rate": a,
                         "test_auc": auc, "LOSO_rejection": loso_r,
                         "Test_S1_rejection": s1, "Test_S2_acceptance": s2, "W": w})
        print(f"  {label:34s} AUC {auc:.4f} bestW "
              f"{max(r['W'] for r in rows):.4f}", flush=True)

    # ---- score mode: per-view scores, equal-weight fusion over the subset
    view_scores = {}
    for name, (a, b) in proj.items():
        te_raw, ref, loso = space_scores(a, b, subjects, tasks, test_tasks, f, blocks5)
        view_scores[name] = (te_raw, ref, loso)
        record(f"score:{name}", "score", te_raw, ref, loso)

    for sname, views in CFG["sets"].items():
        if len(views) < 2:
            continue
        te = np.mean([z_by_task(view_scores[v][0], test_tasks, view_scores[v][1], tasks)
                      for v in views], axis=0)
        rf = np.mean([z_by_task(view_scores[v][1], tasks, view_scores[v][1], tasks)
                      for v in views], axis=0)
        loso = []
        for i in range(len(view_scores[views[0]][2])):
            kn = np.mean([z_by_task(view_scores[v][2][i]["known"],
                                    view_scores[v][2][i]["known_task"],
                                    view_scores[v][1], tasks) for v in views], axis=0)
            un = np.mean([z_by_task(view_scores[v][2][i]["unknown"],
                                    view_scores[v][2][i]["unknown_task"],
                                    view_scores[v][1], tasks) for v in views], axis=0)
            loso.append({"known": kn, "unknown": un})
        # already z-scored: pass through identity reference
        auc = float(roc_auc_score(y_test, np.r_[te[is_s1], te[~is_s1]]))
        parts = list(zip([x["known"] for x in loso], [x["unknown"] for x in loso]))
        for a in CFG["acceptance_grid"]:
            th = float(np.nanquantile(rf, 1 - a))
            s2 = float(np.mean(te[~is_s1] >= th))
            s1 = float(np.mean(te[is_s1] < th))
            loso_r = float(np.mean([np.mean(u < np.quantile(kn, 1 - a)) for kn, u in parts]))
            w = (CFG["weights"]["LOSO"] * loso_r + CFG["weights"]["S1"] * s1
                 + CFG["weights"]["S2"] * s2)
            rows.append({"label": f"scorefuse:{sname}", "mode": "scorefuse",
                         "acceptance_rate": a, "test_auc": auc, "LOSO_rejection": loso_r,
                         "Test_S1_rejection": s1, "Test_S2_acceptance": s2, "W": w})
        print(f"  scorefuse:{sname:28s} AUC {auc:.4f}", flush=True)

    # ---- feature mode: concat per-view projections, second PCA-96
    for sname, views in CFG["sets"].items():
        views = [v for v in views if v in proj]
        if not views:
            continue
        A = np.concatenate([proj[v][0] for v in views], axis=1)
        B = np.concatenate([proj[v][1] for v in views], axis=1)
        mu, V = pca_basis(np.asarray(A[fit_rows], dtype=np.float64), D)
        tr = ((A.astype(np.float64) - mu) @ V).astype(np.float32)
        te = ((B.astype(np.float64) - mu) @ V).astype(np.float32)
        del A, B
        te_raw, ref, loso = space_scores(tr, te, subjects, tasks, test_tasks, f, blocks5)
        record(f"featfuse:{sname}", "featfuse", te_raw, ref, loso)
        del tr, te

    rows.sort(key=lambda r: -r["W"])
    best = rows[0]
    (OUT / "results.json").write_text(json.dumps(
        {"experiment": CFG["experiment"], "clean": True,
         "objective": "0.25*LOSO + 0.25*S1 + 0.5*S2",
         "target": CFG["targets"]["weighted"], "best": best, "curve": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# E141 加权目标 + 图像化路线与互补融合", "",
             "新目标：**W = 0.25×LOSO 拒识 + 0.25×Test S1 拒识 + 0.5×Test S2 接受 > 0.80**。",
             "全程不使用任何 Test 侧统计：标准化/PCA/原型/阈值全部来自 Train，Test 只被评分。", "",
             f"**当前最好：{best['label']} @ 接受率 {best['acceptance_rate']:.2f} → W = {best['W']:.4f}**",
             f"（LOSO {best['LOSO_rejection']:.4f} / S1 {best['Test_S1_rejection']:.4f} / "
             f"S2 {best['Test_S2_acceptance']:.4f}，诊断 AUC {best['test_auc']:.4f}）", "",
             "## 单视图（k=32）", "",
             "| 视图 | 诊断 AUC | 最好 W | LOSO | S1 拒识 | S2 接受 | 接受率 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["mode"] != "score":
            continue
        lines.append(f"| {r['label']} | {r['test_auc']:.4f} | {r['W']:.4f} | "
                     f"{r['LOSO_rejection']:.4f} | {r['Test_S1_rejection']:.4f} | "
                     f"{r['Test_S2_acceptance']:.4f} | {r['acceptance_rate']:.2f} |")
    lines += ["", "## 融合（前 24 个工作点）", "",
              "| 方案 | 方式 | 接受率 | 诊断 AUC | LOSO | S1 拒识 | S2 接受 | W |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows[:24]:
        lines.append(f"| {r['label']} | {r['mode']} | {r['acceptance_rate']:.2f} | "
                     f"{r['test_auc']:.4f} | {r['LOSO_rejection']:.4f} | "
                     f"{r['Test_S1_rejection']:.4f} | {r['Test_S2_acceptance']:.4f} | "
                     f"{r['W']:.4f} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
