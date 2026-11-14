"""E128: distance-form search inside the winning PCA-96 space.

Two facts motivate this:
  1. E121/E122 showed PCA-96 on correlation-logSPD raises the Test S1-vs-S2
     diagnostic AUC to 0.7658.
  2. E127 revealed that the metric matters a lot: nearest-prototype Euclidean
     distance scores 0.766 while cosine similarity scores only 0.539 on the same
     space. So the *magnitude* of the distance carries real signal, and cosine
     throws it away.

This experiment therefore searches distance forms inside the PCA-96 space, all
with task-conditioned prototypes and the Train-only threshold rule.

Variants:
  V1 neg_d1          -distance to nearest prototype (E122 baseline)
  V2 neg_d1_mahal    PCs standardised by Train std, then -nearest distance
  V3 ratio_d2_d1     d2 / d1
  V4 margin_ratio    (d2 - d1) / d1
  V5 neg_d1_rel      -(d1 - mean_d) / std_d, within task
  V6 fused           mean of Train-rank-normalised V1 and V3
  V7 fused2          mean of Train-rank-normalised V1 and V5

Reported together per variant: LOSO rejection, Test S1 rejection, Test S2
acceptance, and the Test diagnostic AUC.
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
    w = w[order][: min(dim, x.shape[1])]
    V = V[:, order][:, : min(dim, x.shape[1])]
    return mu, V, np.sqrt(np.clip(w, 1e-9, None))


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


def per_task_distances(feat, protos, keys, tasks):
    """Return d1, d2 per sample within its own task, plus within-task distance stats."""
    n = len(feat)
    d1 = np.full(n, np.nan, dtype=np.float32)
    d2 = np.full(n, np.nan, dtype=np.float32)
    dmean = np.full(n, np.nan, dtype=np.float32)
    dstd = np.full(n, np.nan, dtype=np.float32)
    for task in CFG["tasks"]:
        idx = np.flatnonzero(tasks == task)
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(idx) == 0 or len(cols) < 2:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        for st in range(0, len(idx), 1024):
            chunk = idx[st:st + 1024]
            q = torch.as_tensor(np.array(feat[chunk], dtype=np.float32), device=DEVICE)
            d = torch.cdist(q, p).cpu().numpy()
            order = np.argsort(d, axis=1)
            d1[chunk] = d[np.arange(len(d)), order[:, 0]]
            d2[chunk] = d[np.arange(len(d)), order[:, 1]]
            dmean[chunk] = d.mean(1)
            dstd[chunk] = d.std(1) + 1e-9
    return d1, d2, dmean, dstd


def rank_normalize(values, reference):
    ref = np.sort(reference[np.isfinite(reference)])
    if len(ref) == 0:
        return np.full(len(values), np.nan, dtype=np.float32)
    return (np.searchsorted(ref, values, side="right") / len(ref)).astype(np.float32)


def three_way(train_sig, test_sig, known_rows, is_s1, target):
    th = float(np.quantile(train_sig[known_rows], 1 - target))
    lab = (test_sig >= th).astype(int)
    return {
        "train_known_acceptance": float(np.mean(train_sig[known_rows] >= th)),
        "Test_S1_rejection": float(np.mean(lab[is_s1] == 0)),
        "Test_S2_acceptance": float(np.mean(lab[~is_s1] == 1)),
        "diagnostic_AUC": float(roc_auc_score(
            np.r_[np.zeros(is_s1.sum()), np.ones((~is_s1).sum())],
            np.r_[test_sig[is_s1], test_sig[~is_s1]])),
    }


def variants_from(d1, d2, dmean, dstd):
    return {
        "neg_d1": -d1,
        "ratio_d2_d1": d2 / np.maximum(d1, 1e-9),
        "margin_ratio": (d2 - d1) / np.maximum(d1, 1e-9),
        "neg_d1_rel": -(d1 - dmean) / dstd,
    }


def loso_eval(space, subjects, tasks, f, target, build_sig):
    out = {}
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        fitm = np.isin(subjects, enrolled) & (f <= CFG["calibration"]["fit_fraction"])
        ki = np.flatnonzero(np.isin(subjects, enrolled) & (f > .6) & (f <= .8))
        ui = np.flatnonzero((subjects == held) & (f > .6) & (f <= .8))
        if len(ui) == 0:
            continue
        protos, keys = build_prototypes(space, subjects, tasks, fitm, enrolled)
        sigs = build_sig(space, protos, keys, tasks, fitm, ki)
        for name, sig in sigs.items():
            th = float(np.quantile(sig[ki], 1 - target))
            out.setdefault(name, []).append({
                "AUC": float(roc_auc_score(np.r_[np.ones(len(ki)), np.zeros(len(ui))],
                                           np.r_[sig[ki], sig[ui]])),
                "unknown_rejection": float(np.mean(sig[ui] < th)),
            })
    return {name: {"AUC": float(np.mean([x["AUC"] for x in v])),
                   "unknown_rejection": float(np.mean([x["unknown_rejection"] for x in v])),
                   "folds_ge_0.80": int(sum(1 for x in v if x["unknown_rejection"] >= 0.80))}
            for name, v in out.items()}


def run():
    meta = pd.read_csv(TRAIN_META)
    feat = np.load(TRAIN_FEATURES)
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

    mu, V, sd = pca_basis(np.asarray(feat[fit_rows], dtype=np.float64), CFG["pca_dim"])
    tr_p = ((np.asarray(feat, dtype=np.float64) - mu) @ V).astype(np.float32)
    te_p = ((np.asarray(test_feat, dtype=np.float64) - mu) @ V).astype(np.float32)
    tr_m = tr_p / sd.astype(np.float32)
    te_m = te_p / sd.astype(np.float32)

    results = {}

    # Standard space
    protos, keys = build_prototypes(tr_p, subjects, tasks, fit_rows, CLASSES)
    d1, d2, dm, ds = per_task_distances(tr_p, protos, keys, tasks)
    t1, t2, tm, ts = per_task_distances(te_p, protos, keys, test_tasks)
    tv = variants_from(t1, t2, tm, ts)
    trv = variants_from(d1, d2, dm, ds)
    for name in tv:
        results[name] = three_way(trv[name], tv[name], known_rows, is_s1, target)
        print(json.dumps({name: results[name]}, ensure_ascii=False), flush=True)

    # Mahalanobis-style standardised space
    protos_m, keys_m = build_prototypes(tr_m, subjects, tasks, fit_rows, CLASSES)
    t1m, t2m, tmm, tsm = per_task_distances(te_m, protos_m, keys_m, test_tasks)
    d1m, d2m, dmm, dsm_ = per_task_distances(tr_m, protos_m, keys_m, tasks)
    results["neg_d1_mahal"] = three_way(-d1m, -t1m, known_rows, is_s1, target)
    print(json.dumps({"neg_d1_mahal": results["neg_d1_mahal"]}, ensure_ascii=False), flush=True)

    # Rank fusions, reference = Train fit block
    for a, b, label in (("neg_d1", "ratio_d2_d1", "fuse_d1_ratio"),):
        fa_tr = 0.5 * (rank_normalize(trv[a], trv[a][fit_rows])
                       + rank_normalize(trv[b], trv[b][fit_rows]))
        fa_te = 0.5 * (rank_normalize(tv[a], trv[a][fit_rows])
                       + rank_normalize(tv[b], trv[b][fit_rows]))
        results[label] = three_way(fa_tr, fa_te, known_rows, is_s1, target)
        print(json.dumps({label: results[label]}, ensure_ascii=False), flush=True)

    # LOSO on the standard space for the same variants.
    def build_sig(space, protos_, keys_, tasks_, fitm, ki):
        a1, a2, am, asd = per_task_distances(space, protos_, keys_, tasks_)
        return variants_from(a1, a2, am, asd)

    l = loso_eval(tr_p, subjects, tasks, f, target, build_sig)
    for name, v in l.items():
        if name in results:
            results[name]["LOSO"] = v

    result = {"experiment": CFG["experiment"], "variants": results}
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    lines = ["# E128 Distance-form search in the PCA-96 space", "",
             "E127 发现度量方式影响巨大：最近原型欧氏距离 AUC 0.766，余弦只有 0.539。"
             "本实验在 PCA-96 空间里系统搜索距离形态。全部只用 Train 拟合块。", "",
             "| 变体 | Test 诊断 AUC | Test S1 拒识 | Test S2 接受 | LOSO 拒识 | LOSO AUC |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, v in results.items():
        lo = v.get("LOSO", {})
        lines.append(f"| {name} | {v['diagnostic_AUC']:.4f} | {v['Test_S1_rejection']:.4f} | "
                     f"{v['Test_S2_acceptance']:.4f} | "
                     f"{lo.get('unknown_rejection', float('nan')):.4f} | "
                     f"{lo.get('AUC', float('nan')):.4f} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
