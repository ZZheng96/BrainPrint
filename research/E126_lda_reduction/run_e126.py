"""E126: supervised dimensionality reduction (multi-class LDA) and per-task PCA.

E122 established the current best three-way point: PCA-96 on correlation-logSPD
followed by task-conditioned max cosine gives
    LOSO rejection 0.6261 / Test S1 rejection 0.7916 / Test S2 acceptance 0.5823.

PCA is unsupervised: it keeps the directions of largest variance, but the
identity signal is not necessarily the largest-variance subspace. Multi-class
LDA directly maximises between-identity over within-identity scatter, so it
should produce a strictly more discriminative space for the same 96 dimensions.

Variants (all fitted on the Train fit block, fraction <= 0.6, no Test data):
  L19..L256   multi-class LDA, d components
  PT          per-task PCA (one basis per task, 96 components)
  L96_PT      LDA then per-task re-projection
  P96_L96     PCA-96 then LDA on top

For each variant we report LOSO rejection, Test S1 rejection and Test S2
acceptance at the Train-chosen threshold, so all three required numbers are
visible together.
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
    """Multi-class Fisher LDA via whitening plus eigendecomposition."""
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
        diff = (mc - mu).reshape(-1, 1)
        Sb += len(xi) * (diff @ diff.T)
    Sw /= max(n - len(classes), 1)
    Sb /= max(n, 1)
    tr = float(np.trace(Sw) / p)
    Sw = (1 - shrinkage) * Sw + shrinkage * tr * np.eye(p)
    w, V = np.linalg.eigh(Sw)
    w = np.clip(w, 1e-8, None)
    W_inv_sqrt = (V / np.sqrt(w)) @ V.T
    M = W_inv_sqrt @ Sb @ W_inv_sqrt
    M = (M + M.T) * 0.5
    evals, evecs = np.linalg.eigh(M)
    order = np.argsort(-evals)
    d = min(dim, len(classes) - 1, p)
    U = evecs[:, order][:, :d]
    return mu, W_inv_sqrt @ U


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


def max_cosine(feat, protos, keys, tasks):
    out = np.full(len(feat), np.nan, dtype=np.float32)
    for task in CFG["tasks"]:
        idx = np.flatnonzero(tasks == task)
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(idx) == 0 or len(cols) < 2:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        for st in range(0, len(idx), 1024):
            chunk = idx[st:st + 1024]
            q = torch.as_tensor(np.array(feat[chunk], dtype=np.float32), device=DEVICE)
            out[chunk] = (-torch.cdist(q, p)).max(1).values.cpu().numpy()
    return out


def operating(known_sig, unknown_sig, target):
    th = float(np.quantile(known_sig, 1 - target))
    return {
        "AUC": float(roc_auc_score(np.r_[np.ones(len(known_sig)), np.zeros(len(unknown_sig))],
                                   np.r_[known_sig, unknown_sig])),
        "known_acceptance": float(np.mean(known_sig >= th)),
        "unknown_rejection": float(np.mean(unknown_sig < th)),
        "threshold": th,
    }


def test_three_way(train_space, test_space, subjects, tasks, fit_rows, known_rows,
                   test_tasks, is_s1, target):
    protos, keys = build_prototypes(train_space, subjects, tasks, fit_rows, CLASSES)
    tr_sc = max_cosine(train_space, protos, keys, tasks)
    te_sc = max_cosine(test_space, protos, keys, test_tasks)
    th = float(np.quantile(tr_sc[known_rows], 1 - target))
    lab = (te_sc >= th).astype(int)
    return {
        "train_known_acceptance": float(np.mean(tr_sc[known_rows] >= th)),
        "Test_S1_rejection": float(np.mean(lab[is_s1] == 0)),
        "Test_S2_acceptance": float(np.mean(lab[~is_s1] == 1)),
        "diagnostic_AUC": float(roc_auc_score(
            np.r_[np.zeros(is_s1.sum()), np.ones((~is_s1).sum())],
            np.r_[te_sc[is_s1], te_sc[~is_s1]])),
    }


def loso_three(feat_space, subjects, tasks, f, target):
    out = []
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        fitm = np.isin(subjects, enrolled) & (f <= CFG["calibration"]["fit_fraction"])
        ki = np.flatnonzero(np.isin(subjects, enrolled) & (f > .6) & (f <= .8))
        ui = np.flatnonzero((subjects == held) & (f > .6) & (f <= .8))
        if len(ui) == 0:
            continue
        protos, keys = build_prototypes(feat_space, subjects, tasks, fitm, enrolled)
        sc = max_cosine(feat_space, protos, keys, tasks)
        out.append(operating(sc[ki], sc[ui], target))
    return {
        "AUC": float(np.mean([x["AUC"] for x in out])),
        "unknown_rejection": float(np.mean([x["unknown_rejection"] for x in out])),
        "folds_ge_0.80": int(sum(1 for x in out if x["unknown_rejection"] >= 0.80)),
    }


def run():
    meta = pd.read_csv(TRAIN_META)
    feat = np.load(TRAIN_FEATURES)
    if len(meta) != CFG["data"]["expected_trials"] or feat.shape != (len(meta), 855):
        raise ValueError("Train metadata/feature invariant failed")
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)
    fit_rows = f <= CFG["calibration"]["fit_fraction"]
    known_rows = (f > .6) & (f <= .8)

    mapping = pd.read_csv(ROOT / CFG["data"]["test_metadata"],
                          usecols=["sample", "path", "session", "task"])
    is_s1 = mapping.session.astype(str).str.casefold().isin(["session1", "s1", "1"]).to_numpy()
    test_tasks = mapping.task.to_numpy()
    test_feat = np.load(TEST_CACHE)

    target = CFG["targets"]["known_acceptance"]
    x_fit = np.asarray(feat[fit_rows], dtype=np.float64)
    y_fit = subjects[fit_rows]

    results = {}

    def evaluate(name, tr_space, te_space):
        t = test_three_way(tr_space, te_space, subjects, tasks, fit_rows, known_rows,
                           test_tasks, is_s1, target)
        results[name] = t
        print(json.dumps({name: t}, ensure_ascii=False), flush=True)

    # Reference: PCA-96 (E122 k1)
    mu, V = pca_basis(x_fit, CFG["pca_dim"])
    tr_p = ((np.asarray(feat, dtype=np.float64) - mu) @ V).astype(np.float32)
    te_p = ((np.asarray(test_feat, dtype=np.float64) - mu) @ V).astype(np.float32)
    evaluate("REF_PCA96", tr_p, te_p)

    # LDA at several dimensions
    for dim in CFG["lda_dims"]:
        mu_l, W = lda_basis(x_fit, y_fit, dim)
        tr_l = ((np.asarray(feat, dtype=np.float64) - mu_l) @ W).astype(np.float32)
        te_l = ((np.asarray(test_feat, dtype=np.float64) - mu_l) @ W).astype(np.float32)
        evaluate(f"LDA{dim}", tr_l, te_l)

    # PCA-96 then LDA
    for dim in CFG["lda_on_pca_dims"]:
        mu_l, W = lda_basis(np.asarray(tr_p[fit_rows], dtype=np.float64), y_fit, dim)
        tr_pl = ((np.asarray(tr_p, dtype=np.float64) - mu_l) @ W).astype(np.float32)
        te_pl = ((np.asarray(te_p, dtype=np.float64) - mu_l) @ W).astype(np.float32)
        evaluate(f"PCA96_LDA{dim}", tr_pl, te_pl)

    # Per-task PCA basis
    tr_pt = np.zeros((len(feat), CFG["pca_dim"]), dtype=np.float32)
    te_pt = np.zeros((len(test_feat), CFG["pca_dim"]), dtype=np.float32)
    for task in CFG["tasks"]:
        m_tr = fit_rows & (tasks == task)
        if m_tr.sum() < CFG["min_fit_per_task"]:
            continue
        mu_t, V_t = pca_basis(np.asarray(feat[m_tr], dtype=np.float64), CFG["pca_dim"])
        mt = tasks == task
        tr_pt[mt] = ((np.asarray(feat[mt], dtype=np.float64) - mu_t) @ V_t).astype(np.float32)
        mte = test_tasks == task
        if mte.any():
            te_pt[mte] = ((np.asarray(test_feat[mte], dtype=np.float64) - mu_t) @ V_t).astype(np.float32)
    evaluate("PERTASK_PCA96", tr_pt, te_pt)

    # LOSO for the two strongest candidates
    for name in ("REF_PCA96", "LDA64", "PERTASK_PCA96"):
        if name not in results:
            continue
        if name == "REF_PCA96":
            space = tr_p
        elif name == "PERTASK_PCA96":
            space = tr_pt
        else:
            mu_l, W = lda_basis(x_fit, y_fit, 64)
            space = ((np.asarray(feat, dtype=np.float64) - mu_l) @ W).astype(np.float32)
        results[name]["LOSO"] = loso_three(space, subjects, tasks, f, target)
        print(json.dumps({name + "_LOSO": results[name]["LOSO"]}, ensure_ascii=False), flush=True)

    result = {"experiment": CFG["experiment"], "variants": results}
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    lines = ["# E126 Supervised reduction (LDA) and per-task PCA", "",
             "PCA 是无监督的（保留最大方差），但身份信号不一定在最大方差子空间。"
             "多类 LDA 直接最大化类间/类内散度比，应得到更可分的空间。"
             "全部只用 Train 拟合块，未用 Test。", "",
             "| 变体 | Test 诊断 AUC | Test S1 拒识 | Test S2 接受 | LOSO 拒识 | LOSO AUC |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, v in results.items():
        l = v.get("LOSO", {})
        lines.append(f"| {name} | {v['diagnostic_AUC']:.4f} | {v['Test_S1_rejection']:.4f} | "
                     f"{v['Test_S2_acceptance']:.4f} | "
                     f"{l.get('unknown_rejection', float('nan')):.4f} | "
                     f"{l.get('AUC', float('nan')):.4f} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
