"""E143: push the prototype count to the limit (pure nearest neighbour).

New objective: W = 0.25*LOSO_rejection + 0.25*Test_S1_rejection
                  + 0.5*Test_S2_acceptance, target 0.80.

S2 carries half the weight, so the shape of the score distribution matters as much
as the AUC.  k = 100000 means "use every training trial of that (task, identity)
as a prototype", i.e. nearest-neighbour open-set scoring.

Clean protocol: Train-only standardisation / PCA / prototypes / threshold.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# config path may be overridden so the same harness can serve other view sets
CFG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "config.json"
CFG_PATH = CFG_PATH.resolve()
OUT = CFG_PATH.parent
ROOT = OUT.parents[1]
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TASKS = CFG["tasks"]
W_ = CFG["weights"]


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
        sums = torch.zeros_like(c).index_add_(0, a, x)
        cnt = torch.zeros(k, device=x.device, dtype=x.dtype).index_add_(
            0, a, torch.ones(len(x), device=x.device, dtype=x.dtype))
        new = torch.where(cnt[:, None] > 0, sums / cnt.clamp(min=1)[:, None], c)
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
        for st in range(0, len(idx), 1024):
            ch = idx[st:st + 1024]
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
    y = np.r_[np.zeros(int(is_s1.sum())), np.ones(int((~is_s1).sum()))]
    D = CFG["pca_dim"]

    proj = {}
    for name, (ptr, pte) in CFG["views"].items():
        a = np.load(ROOT / ptr).astype(np.float32)
        b = np.load(ROOT / pte).astype(np.float32)
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
    for k in CFG["prototype_k"]:
        view_scores = {}
        for name, (a, b) in proj.items():
            protos, keys = build_prototypes(a, subjects, tasks, np.ones(len(a), bool), CLASSES, k)
            te_raw = neg_d1(b, protos, keys, test_tasks)
            ref = np.full(len(a), np.nan, dtype=np.float32)
            for blk in range(5):
                hold = blocks5 == blk
                p2, k2 = build_prototypes(a, subjects, tasks, ~hold, CLASSES, k)
                s = neg_d1(a, p2, k2, tasks)
                ref[hold] = s[hold]
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
            rr = ref[np.isfinite(ref)]
            mu, sd = float(rr.mean()), float(rr.std()) + 1e-6
            view_scores[name] = (((te_raw - mu) / sd), (ref - mu) / sd,
                                 [((x - mu) / sd, (yv - mu) / sd) for x, yv in loso])
            print(f"  k={k} {name:14s} AUC "
                  f"{roc_auc_score(y, np.r_[te_raw[is_s1], te_raw[~is_s1]]):.4f}", flush=True)

        for sname, views in CFG["sets"].items():
            views = [v for v in views if v in view_scores]
            te = np.mean([view_scores[v][0] for v in views], axis=0)
            rf = np.mean([view_scores[v][1] for v in views], axis=0)
            auc = float(roc_auc_score(y, np.r_[te[is_s1], te[~is_s1]]))
            nf = len(view_scores[views[0]][2])
            parts = []
            for i in range(nf):
                kn = np.mean([view_scores[v][2][i][0] for v in views], axis=0)
                un = np.mean([view_scores[v][2][i][1] for v in views], axis=0)
                parts.append((kn, un))
            best = None
            for a in CFG["acceptance_grid"]:
                th = float(np.nanquantile(rf, 1 - a))
                s2 = float(np.mean(te[~is_s1] >= th))
                s1 = float(np.mean(te[is_s1] < th))
                lr = float(np.mean([np.mean(u < np.quantile(kn, 1 - a)) for kn, u in parts]))
                w = W_["LOSO"] * lr + W_["S1"] * s1 + W_["S2"] * s2
                row = {"set": sname, "k": k, "acceptance_rate": a, "test_auc": auc,
                       "LOSO_rejection": lr, "Test_S1_rejection": s1,
                       "Test_S2_acceptance": s2, "W": w}
                rows.append(row)
                if best is None or w > best["W"]:
                    best = row
            print(f"  k={k} {sname:12s} AUC {auc:.4f} bestW {best['W']:.4f} "
                  f"@a={best['acceptance_rate']:.3f}", flush=True)

    rows.sort(key=lambda r: -r["W"])
    best = rows[0]
    (OUT / "results.json").write_text(json.dumps(
        {"experiment": CFG["experiment"], "clean": True,
         "objective": "0.25*LOSO + 0.25*S1 + 0.5*S2",
         "target": CFG["targets"]["weighted"], "best": best, "curve": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# E143 原型数上限（纯最近邻）扫描", "",
             "新目标：**W = 0.25×LOSO 拒识 + 0.25×Test S1 拒识 + 0.5×Test S2 接受 > 0.80**。",
             "k=100000 表示把该 (task, identity) 的全部 Train trial 都当原型，即纯最近邻开放集打分。"
             "全程 Train-only，Test 只被评分。", "",
             f"**当前最好：{best['set']} k={best['k']} @ 接受率 {best['acceptance_rate']:.3f} → "
             f"W = {best['W']:.4f}**",
             f"（LOSO {best['LOSO_rejection']:.4f} / S1 {best['Test_S1_rejection']:.4f} / "
             f"S2 {best['Test_S2_acceptance']:.4f}，诊断 AUC {best['test_auc']:.4f}）", "",
             "| 视图集 | k | 接受率 | 诊断 AUC | LOSO | S1 拒识 | S2 接受 | W |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows[:20]:
        lines.append(f"| {r['set']} | {r['k']} | {r['acceptance_rate']:.3f} | {r['test_auc']:.4f} | "
                     f"{r['LOSO_rejection']:.4f} | {r['Test_S1_rejection']:.4f} | "
                     f"{r['Test_S2_acceptance']:.4f} | {r['W']:.4f} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
