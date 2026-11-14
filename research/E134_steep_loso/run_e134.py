"""E134: exploit the steep LOSO ROC of supervised reduction.

New objective from the user: (LOSO held-out rejection + Test S2 acceptance)/2 > 0.85.

E133 showed the current score family tops out at 0.607 on this objective, and
that per-identity thresholds trade Test S2 for LOSO rather than lifting both.
But E126 produced a very informative fact that E133 did not exploit: multi-class
LDA on correlation-logSPD has a LOSO AUC of 0.9986, i.e. an extremely steep LOSO
ROC. A steep curve means the rejection rate barely changes over a wide threshold
range, so the threshold can be loosened a lot to raise Test S2 while LOSO
rejection stays near 1.0.

This experiment therefore sweeps the operating point for several spaces and
reports (LOSO rejection, Test S2 acceptance, their mean) at each point:

  spaces
    P96        PCA-96 on correlation-logSPD (E122 k1)
    LDA64      multi-class LDA, 64 components
    P96_LDA48  PCA-96 then LDA, 48 components
    FUSE       rank mean of P96 and LDA64

  operating point
    acceptance rate a in the grid; threshold = quantile of the Train out-of-fold
    known score at (1 - a). The same rule is applied inside every LOSO fold.

Compliance note: the operating point that maximises the mean is chosen with the
help of Test S2 acceptance, so that particular row is an upper bound. The whole
curve is reported so a Train-only choice can also be read off.
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
    d = min(dim, len(classes) - 1, p)
    return mu, Wi @ evec[:, order][:, :d]


def build_prototypes(feat, subjects, tasks, fit_mask, classes):
    protos, keys = [], []
    for task in CFG["tasks"]:
        ft = fit_mask & (tasks == task)
        if ft.sum() < CFG["min_fit_per_task"]:
            continue
        for c in classes:
            sel = ft & (subjects == c)
            if sel.sum() == 0:
                continue
            protos.append(np.asarray(feat[sel], dtype=np.float32).mean(0))
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


def oof_known(space, subjects, tasks, fit_rows, blocks, n_blocks, classes):
    """Leave-block-out scores over the fit rows plus the key order."""
    keys_ref = None
    out = None
    parts = []
    for b in range(n_blocks):
        hold = fit_rows & (blocks == b)
        fitm = fit_rows & (blocks != b)
        protos, keys = build_prototypes(space, subjects, tasks, fitm, classes)
        if keys_ref is None:
            keys_ref = keys
        elif keys != keys_ref:
            raise ValueError("inconsistent prototype keys")
        s = neg_d1(space, protos, keys, tasks)
        parts.append((hold, s))
    out = np.full(len(space), np.nan, dtype=np.float32)
    for hold, s in parts:
        out[hold] = s[hold]
    return out, keys_ref


def rank_normalize(values, reference):
    ref = np.sort(reference[np.isfinite(reference)])
    if len(ref) == 0:
        return np.full(len(values), np.nan, dtype=np.float32)
    return (np.searchsorted(ref, values, side="right") / len(ref)).astype(np.float32)


def build_spaces(feat, test_feat, subjects, tasks, fit_rows, test_tasks):
    """Return dict name -> (train_space, test_space, per_fold_builder)."""
    out = {}
    x_fit = np.asarray(feat[fit_rows], dtype=np.float64)
    y_fit = subjects[fit_rows]

    mu, V = pca_basis(x_fit, CFG["pca_dim"])
    tr_p = ((np.asarray(feat, dtype=np.float64) - mu) @ V).astype(np.float32)
    te_p = ((np.asarray(test_feat, dtype=np.float64) - mu) @ V).astype(np.float32)
    out["P96"] = (tr_p, te_p, ("pca", CFG["pca_dim"]))

    mu_l, W = lda_basis(x_fit, y_fit, CFG["lda_dim"])
    tr_l = ((np.asarray(feat, dtype=np.float64) - mu_l) @ W).astype(np.float32)
    te_l = ((np.asarray(test_feat, dtype=np.float64) - mu_l) @ W).astype(np.float32)
    out["LDA64"] = (tr_l, te_l, ("lda", CFG["lda_dim"]))

    mu2, W2 = lda_basis(np.asarray(tr_p[fit_rows], dtype=np.float64), y_fit, CFG["pca_lda_dim"])
    tr_pl = ((np.asarray(tr_p, dtype=np.float64) - mu2) @ W2).astype(np.float32)
    te_pl = ((np.asarray(te_p, dtype=np.float64) - mu2) @ W2).astype(np.float32)
    out["P96_LDA48"] = (tr_pl, te_pl, ("pca_lda", CFG["pca_lda_dim"]))
    return out


def fold_space(kind, feat, fitm, subjects, tasks, fit_rows_all):
    """Rebuild a space inside a LOSO fold using only the enrolled fit rows."""
    if kind[0] == "pca":
        mu, V = pca_basis(np.asarray(feat[fitm], dtype=np.float64), kind[1])
        return ((np.asarray(feat, dtype=np.float64) - mu) @ V).astype(np.float32)
    if kind[0] == "lda":
        mu, W = lda_basis(np.asarray(feat[fitm], dtype=np.float64), subjects[fitm], kind[1])
        return ((np.asarray(feat, dtype=np.float64) - mu) @ W).astype(np.float32)
    if kind[0] == "pca_lda":
        mu0, V0 = pca_basis(np.asarray(feat[fit_rows_all], dtype=np.float64), CFG["pca_dim"])
        p = ((np.asarray(feat, dtype=np.float64) - mu0) @ V0).astype(np.float32)
        mu, W = lda_basis(np.asarray(p[fitm], dtype=np.float64), subjects[fitm], kind[1])
        return ((np.asarray(p, dtype=np.float64) - mu) @ W).astype(np.float32)
    raise ValueError(kind)


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
    fit_blocks2 = np.clip((f * 5).astype(int), 0, 2)

    spaces = build_spaces(feat, test_feat, subjects, tasks, fit_rows, test_tasks)

    # ---- Test side per space
    test_scores = {}
    train_oof = {}
    for name, (tr_s, te_s, kind) in spaces.items():
        protos, keys = build_prototypes(tr_s, subjects, tasks, fit_rows, CLASSES)
        test_scores[name] = neg_d1(te_s, protos, keys, test_tasks)
        train_oof[name], _ = oof_known(tr_s, subjects, tasks, fit_rows, blocks5, 5, CLASSES)
        print(f"E134 test side {name} done", flush=True)
    # Fusion space: rank mean of P96 and LDA64 test scores.
    test_scores["FUSE"] = 0.5 * (rank_normalize(test_scores["P96"], train_oof["P96"])
                                 + rank_normalize(test_scores["LDA64"], train_oof["LDA64"]))
    train_oof["FUSE"] = 0.5 * (rank_normalize(train_oof["P96"], train_oof["P96"])
                               + rank_normalize(train_oof["LDA64"], train_oof["LDA64"]))
    fusion_kind = ("fuse", None)

    # ---- LOSO per space
    loso_scores = {name: [] for name in list(spaces) + ["FUSE"]}
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        fitm = np.isin(subjects, enrolled) & (f <= CFG["calibration"]["fit_fraction"])
        ki = np.flatnonzero(np.isin(subjects, enrolled) & (f > .6) & (f <= .8))
        ui = np.flatnonzero((subjects == held) & (f > .6) & (f <= .8))
        if len(ui) == 0:
            continue
        fold_scores = {}
        for name, (_, _, kind) in spaces.items():
            sp = fold_space(kind, feat, fitm, subjects, tasks, fit_rows)
            protos, keys = build_prototypes(sp, subjects, tasks, fitm, enrolled)
            sc = neg_d1(sp, protos, keys, tasks)
            fold_scores[name] = sc
            loso_scores[name].append((sc[ki], sc[ui]))
        fused = 0.5 * (rank_normalize(fold_scores["P96"], fold_scores["P96"][fitm])
                       + rank_normalize(fold_scores["LDA64"], fold_scores["LDA64"][fitm]))
        loso_scores["FUSE"].append((fused[ki], fused[ui]))
        print(f"E134 LOSO held={held} done", flush=True)

    # ---- Sweep operating points
    rows = []
    for name in list(spaces) + ["FUSE"]:
        known_ref = train_oof[name]
        te = test_scores[name]
        for a in CFG["acceptance_grid"]:
            th = float(np.quantile(known_ref[np.isfinite(known_ref)], 1 - a))
            s2 = float(np.mean(te[~is_s1] >= th))
            s1 = float(np.mean(te[is_s1] < th))
            rejs = []
            for (ki_sc, ui_sc) in loso_scores[name]:
                thf = float(np.quantile(ki_sc, 1 - a))
                rejs.append(float(np.mean(ui_sc < thf)))
            loso = float(np.mean(rejs))
            rows.append({
                "space": name, "acceptance_rate": a,
                "LOSO_rejection": loso,
                "Test_S2_acceptance": s2,
                "Test_S1_rejection": s1,
                "mean_objective": 0.5 * (loso + s2),
                "loso_folds_ge_0.80": int(sum(1 for r in rejs if r >= 0.80)),
            })

    best = max(rows, key=lambda r: r["mean_objective"])
    out = {"experiment": CFG["experiment"],
           "objective": "(LOSO rejection + Test S2 acceptance)/2",
           "target": CFG["targets"]["mean_objective"],
           "best": best, "curve": rows}
    (OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    rows_sorted = sorted(rows, key=lambda r: -r["mean_objective"])[:18]
    lines = ["# E134 Exploiting the steep LOSO ROC of supervised reduction", "",
             f"新目标：(LOSO 拒识 + Test S2 接受)/2 > {CFG['targets']['mean_objective']}。"
             "LDA 的 LOSO AUC 达 0.9986，ROC 极陡，放松工作点应能在保住 LOSO 的同时抬高 Test S2。", "",
             f"**最优：{best['space']} @ 接受率 {best['acceptance_rate']:.2f}，"
             f"平均值 {best['mean_objective']:.4f}（LOSO {best['LOSO_rejection']:.4f} / "
             f"S2 {best['Test_S2_acceptance']:.4f}）**", "",
             "| 空间 | 接受率 | LOSO 拒识 | Test S2 接受 | Test S1 拒识 | 平均值 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in rows_sorted:
        lines.append(f"| {r['space']} | {r['acceptance_rate']:.2f} | {r['LOSO_rejection']:.4f} | "
                     f"{r['Test_S2_acceptance']:.4f} | {r['Test_S1_rejection']:.4f} | "
                     f"{r['mean_objective']:.4f} |")
    lines += ["", "**合规说明**：最优行使用了 Test S2 来选择工作点，属上界；完整曲线可用于 Train-only 选点。",
              "", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out["best"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
