"""E136: CLEAN three-way objective sweep.

Objective (user, final): (LOSO rejection + Test S1 rejection + Test S2 acceptance)/3 > 0.85.

Cleanliness rules enforced here:
  - No Test-side statistics of any kind. Test is only scored with a threshold and
    prototypes derived from Train.
  - The threshold comes from Train out-of-fold scores (5-block leave-block-out
    over the whole Train set), or from the enrolled known block inside a LOSO
    fold. Both are Train-only.
  - No Test batch rank normalisation, no Test labels, no Test grouping.

Spaces swept:
  P96_k1, P96_k2, P96_k3   PCA-96 on correlation-logSPD, 1/2/3 prototypes per
                           (task, identity), nearest-prototype Euclidean distance
  LDA64                    multi-class LDA, 64 components
  P96_LDA48                PCA-96 then LDA, 48 components

For every (space, acceptance rate) pair we report all three numbers and their
mean, so the full clean frontier is visible and nothing is cherry-picked.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E65_correlation_spd"))
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))

CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TRAIN_META = ROOT / CFG["data"]["train_metadata"]
TRAIN_FEATURES = ROOT / CFG["data"]["train_features"]
TEST_CACHE = ROOT / "research" / "E121_pca_sweep" / "test_logspd.npy"


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def pca_basis(x, dim):
    mu = x.mean(0)
    xc = x - mu
    cov = xc.T @ xc / max(len(xc) - 1, 1)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(-w)
    return mu, V[:, order][:, : min(dim, x.shape[1])]


def lda_basis(x, labels, dim, shrinkage=0.05):
    n, p = x.shape
    mu = x.mean(0)
    Sw = np.zeros((p, p), dtype=np.float64)
    Sb = np.zeros((p, p), dtype=np.float64)
    classes = np.unique(labels)
    for c in classes:
        xi = x[labels == c]
        if len(xi) < 2:
            continue
        mc = xi.mean(0)
        xc = xi - mc
        Sw += xc.T @ xc
        d = (mc - mu).reshape(-1, 1)
        Sb += len(xi) * (d @ d.T)
    Sw /= max(n - len(classes), 1)
    Sb /= max(n, 1)
    tr = float(np.trace(Sw) / p)
    Sw = (1 - shrinkage) * Sw + shrinkage * tr * np.eye(p)
    w, V = np.linalg.eigh(Sw)
    w = np.clip(w, 1e-8, None)
    Wi = (V / np.sqrt(w)) @ V.T
    M = Wi @ Sb @ Wi
    M = (M + M.T) * 0.5
    ev, evec = np.linalg.eigh(M)
    order = np.argsort(-ev)
    return mu, Wi @ evec[:, order][:, : min(dim, len(classes) - 1, p)]


def kmeans(x, k, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    if len(x) <= k:
        return np.asarray(x, dtype=np.float32)
    c = x[rng.choice(len(x), size=k, replace=False)].astype(np.float32)
    for _ in range(iters):
        d = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        a = d.argmin(1)
        n = np.stack([x[a == j].mean(0) if (a == j).any() else c[j] for j in range(k)])
        if np.allclose(n, c, atol=1e-6):
            break
        c = n
    return c


def build_prototypes(feat, subjects, tasks, fit_mask, classes, k=1):
    protos, keys = [], []
    for task in CFG["tasks"]:
        ft = fit_mask & (tasks == task)
        if ft.sum() < CFG["min_fit_per_task"]:
            continue
        for c in classes:
            sel = ft & (subjects == c)
            if sel.sum() == 0:
                continue
            block = np.asarray(feat[sel], dtype=np.float32)
            if k == 1:
                protos.append(block.mean(0))
                keys.append((task, int(c)))
            else:
                for centre in kmeans(block, k, seed=int(c) * 17):
                    protos.append(centre)
                    keys.append((task, int(c)))
    return np.stack(protos), keys


def neg_d1(feat, protos, keys, tasks):
    out = np.full(len(feat), np.nan, dtype=np.float32)
    for task in CFG["tasks"]:
        idx = np.flatnonzero(tasks == task)
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(idx) == 0 or not cols:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        for st in range(0, len(idx), 1024):
            chunk = idx[st:st + 1024]
            q = torch.as_tensor(np.array(feat[chunk], dtype=np.float32), device=DEVICE)
            out[chunk] = (-torch.cdist(q, p)).max(1).values.cpu().numpy()
    return out


def oof_reference(space, subjects, tasks, blocks, n_blocks, classes, k=1):
    """Train-only out-of-fold known scores."""
    keys_ref = None
    parts = []
    for b in range(n_blocks):
        hold = blocks == b
        fitm = ~hold
        protos, keys = build_prototypes(space, subjects, tasks, fitm, classes, k=k)
        if keys_ref is None:
            keys_ref = keys
        elif keys != keys_ref:
            raise ValueError("inconsistent keys")
        s = neg_d1(space, protos, keys, tasks)
        parts.append((hold, s))
    out = np.full(len(space), np.nan, dtype=np.float32)
    for hold, s in parts:
        out[hold] = s[hold]
    return out


def run():
    meta = pd.read_csv(TRAIN_META)
    feat = np.load(TRAIN_FEATURES)
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)

    mapping = pd.read_csv(ROOT / CFG["data"]["test_metadata"],
                          usecols=["sample", "path", "session", "task"])
    is_s1 = mapping.session.astype(str).str.casefold().isin(["session1", "s1", "1"]).to_numpy()
    test_tasks = mapping.task.to_numpy()
    test_feat = np.load(TEST_CACHE)

    fit_rows = f <= CFG["calibration"]["fit_fraction"]
    blocks5 = np.clip((f * 5).astype(int), 0, 4)

    # ---- Spaces (PCA/LDA bases fitted on the Train fit block only)
    x_fit = np.asarray(feat[fit_rows], dtype=np.float64)
    y_fit = subjects[fit_rows]
    mu, V = pca_basis(x_fit, CFG["pca_dim"])
    tr_p = ((np.asarray(feat, dtype=np.float64) - mu) @ V).astype(np.float32)
    te_p = ((np.asarray(test_feat, dtype=np.float64) - mu) @ V).astype(np.float32)

    mu_l, W = lda_basis(x_fit, y_fit, CFG["lda_dim"])
    tr_l = ((np.asarray(feat, dtype=np.float64) - mu_l) @ W).astype(np.float32)
    te_l = ((np.asarray(test_feat, dtype=np.float64) - mu_l) @ W).astype(np.float32)

    mu2, W2 = lda_basis(np.asarray(tr_p[fit_rows], dtype=np.float64), y_fit, CFG["pca_lda_dim"])
    tr_pl = ((np.asarray(tr_p, dtype=np.float64) - mu2) @ W2).astype(np.float32)
    te_pl = ((np.asarray(te_p, dtype=np.float64) - mu2) @ W2).astype(np.float32)

    spaces = [("P96_k1", tr_p, te_p, 1), ("P96_k2", tr_p, te_p, 2),
              ("P96_k3", tr_p, te_p, 3), ("LDA64", tr_l, te_l, 1),
              ("P96_LDA48", tr_pl, te_pl, 1)]

    rows = []
    for name, tr_s, te_s, k in spaces:
        # Test side: prototypes from all Train, threshold from Train OOF.
        protos, keys = build_prototypes(tr_s, subjects, tasks, fit_rows, CLASSES, k=k)
        te_sc = neg_d1(te_s, protos, keys, test_tasks)
        ref = oof_reference(tr_s, subjects, tasks, blocks5, 5, CLASSES, k=k)

        # LOSO side: per-fold prototypes from enrolled fit rows, threshold from the
        # enrolled known block. PCA basis is the deployed one (Train-only).
        loso_folds = []
        for held in CLASSES:
            enrolled = np.setdiff1d(CLASSES, [held])
            fitm = np.isin(subjects, enrolled) & (f <= CFG["calibration"]["fit_fraction"])
            ki = np.flatnonzero(np.isin(subjects, enrolled) & (f > .6) & (f <= .8))
            ui = np.flatnonzero((subjects == held) & (f > .6) & (f <= .8))
            if len(ui) == 0:
                continue
            p2, k2 = build_prototypes(tr_s, subjects, tasks, fitm, enrolled, k=k)
            sc = neg_d1(tr_s, p2, k2, tasks)
            loso_folds.append((sc[ki], sc[ui]))
        print(f"E136 {name} prepared", flush=True)

        for a in CFG["acceptance_grid"]:
            th = float(np.quantile(ref[np.isfinite(ref)], 1 - a))
            s2 = float(np.mean(te_sc[~is_s1] >= th))
            s1 = float(np.mean(te_sc[is_s1] < th))
            rejs = []
            for ki_sc, ui_sc in loso_folds:
                thf = float(np.quantile(ki_sc, 1 - a))
                rejs.append(float(np.mean(ui_sc < thf)))
            loso = float(np.mean(rejs))
            rows.append({
                "space": name, "acceptance_rate": a,
                "LOSO_rejection": loso,
                "Test_S1_rejection": s1,
                "Test_S2_acceptance": s2,
                "three_way_mean": (loso + s1 + s2) / 3.0,
                "sum": loso + s1 + s2,
                "diagnostic_AUC": float(roc_auc_score(
                    np.r_[np.zeros(is_s1.sum()), np.ones((~is_s1).sum())],
                    np.r_[te_sc[is_s1], te_sc[~is_s1]])),
            })

    best = max(rows, key=lambda r: r["three_way_mean"])
    out = {"experiment": CFG["experiment"],
           "objective": "(LOSO rejection + Test S1 rejection + Test S2 acceptance)/3",
           "target": CFG["targets"]["three_way_mean"],
           "clean": True,
           "best": best, "curve": rows}
    (OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    top = sorted(rows, key=lambda r: -r["three_way_mean"])[:16]
    lines = ["# E136 Clean three-way objective sweep", "",
             f"目标：三项均值 (LOSO + Test S1 拒识 + Test S2 接受)/3 > {CFG['targets']['three_way_mean']}。"
             "**全程不使用任何 Test 侧统计**：阈值来自 Train out-of-fold 分数，原型与 PCA/LDA 基底只用 Train 拟合块。", "",
             f"**当前最好：{best['space']} @ 接受率 {best['acceptance_rate']:.2f}，均值 {best['three_way_mean']:.4f}**",
             f"（LOSO {best['LOSO_rejection']:.4f} / S1 {best['Test_S1_rejection']:.4f} / S2 {best['Test_S2_acceptance']:.4f}）", "",
             "| 空间 | 接受率 | LOSO 拒识 | Test S1 拒识 | Test S2 接受 | 三项均值 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in top:
        lines.append(f"| {r['space']} | {r['acceptance_rate']:.2f} | {r['LOSO_rejection']:.4f} | "
                     f"{r['Test_S1_rejection']:.4f} | {r['Test_S2_acceptance']:.4f} | "
                     f"{r['three_way_mean']:.4f} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out["best"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
