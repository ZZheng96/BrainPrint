"""E125: band-wise score fusion.

E123 showed each frequency band carries a different part of the S1/S2 signal:
  B1 1-8 Hz    S1 0.546 / S2 0.711   (S2-friendly)
  B2 8-13 Hz   S1 0.610 / S2 0.721   (S2-friendly)
  B4 20-30 Hz  S1 0.730 / S2 0.580   (S1-friendly)
  B5 30-40 Hz  S1 0.719 / S2 0.529   (S1-friendly)
  ALL          S1 0.792 / S2 0.582

Because the bands prefer opposite ends of the operating curve, fusing their
per-band score curves should give a strictly better ROC than any single band or
the concatenated feature space. This experiment builds one task-conditioned
max-cosine score per band in a per-band PCA space, then fuses them.

Fusion variants:
  F1 mean of per-band scores (raw)
  F2 mean of per-band empirical-CDF ranks (Train reference)
  F3 weighted rank fusion, weight = per-band Train AUC - 0.5

Report LOSO rejection, Test S1 rejection and Test S2 acceptance for each.
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
from run_e36 import load_config as load_e36_config

CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TRAIN_META = ROOT / CFG["data"]["train_metadata"]
TRAIN_FEATURES = ROOT / CFG["data"]["train_features"]
TEST_CACHE = ROOT / "research" / "E121_pca_sweep" / "test_logspd.npy"


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def pca_fit(train, fit_rows, dim):
    x = np.asarray(train[fit_rows], dtype=np.float64)
    mu = x.mean(0)
    xc = x - mu
    cov = xc.T @ xc / max(len(xc) - 1, 1)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(-w)
    return mu, V[:, order][:, : min(dim, x.shape[1])]


def project(x, mu, V):
    return ((np.asarray(x, dtype=np.float64) - mu) @ V).astype(np.float32)


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


def rank_normalize(values, reference):
    ref = np.sort(reference[np.isfinite(reference)])
    if len(ref) == 0:
        return np.full(len(values), np.nan, dtype=np.float32)
    return (np.searchsorted(ref, values, side="right") / len(ref)).astype(np.float32)


def operating(known_sig, unknown_sig, target):
    th = float(np.quantile(known_sig, 1 - target))
    return {
        "AUC": float(roc_auc_score(np.r_[np.ones(len(known_sig)), np.zeros(len(unknown_sig))],
                                   np.r_[known_sig, unknown_sig])),
        "known_acceptance": float(np.mean(known_sig >= th)),
        "unknown_rejection": float(np.mean(unknown_sig < th)),
        "threshold": th,
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

    n_bands = len(load_e36_config()["features"]["bands_hz"])
    width = feat.shape[1] // n_bands

    target = CFG["targets"]["known_acceptance"]

    # Per-band projections and scores.
    tr_band, te_band = [], []
    for b in range(n_bands):
        cols = np.arange(b * width, (b + 1) * width)
        mu, V = pca_fit(feat[:, cols], fit_rows, CFG["pca_dim"])
        trp = project(feat[:, cols], mu, V)
        tep = project(test_feat[:, cols], mu, V)
        protos, keys = build_prototypes(trp, subjects, tasks, fit_rows, CLASSES)
        tr_band.append(max_cosine(trp, protos, keys, tasks))
        te_band.append(max_cosine(tep, protos, keys, test_tasks))
        print(f"E125 band {b + 1}/{n_bands} done", flush=True)
    tr_band = np.stack(tr_band)
    te_band = np.stack(te_band)

    results = {}

    # Per-band reference (deployable threshold from the Train known block).
    for b in range(n_bands):
        th = float(np.quantile(tr_band[b][known_rows], 1 - target))
        lab = (te_band[b] >= th).astype(int)
        results[f"band{b + 1}"] = {
            "Test_S1_rejection": float(np.mean(lab[is_s1] == 0)),
            "Test_S2_acceptance": float(np.mean(lab[~is_s1] == 1)),
            "diagnostic_AUC": float(roc_auc_score(
                np.r_[np.zeros(is_s1.sum()), np.ones((~is_s1).sum())],
                np.r_[te_band[b][is_s1], te_band[b][~is_s1]])),
        }

    # F1: mean of raw band scores
    f1_tr = tr_band.mean(0)
    f1_te = te_band.mean(0)
    # F2: mean of rank-normalised band scores
    f2_tr = np.mean([rank_normalize(tr_band[b], tr_band[b][fit_rows]) for b in range(n_bands)], 0)
    f2_te = np.mean([rank_normalize(te_band[b], tr_band[b][fit_rows]) for b in range(n_bands)], 0)

    for name, tr_s, te_s in (("F1_mean_raw", f1_tr, f1_te),
                             ("F2_mean_rank", f2_tr, f2_te)):
        th = float(np.quantile(tr_s[known_rows], 1 - target))
        lab = (te_s >= th).astype(int)
        results[name] = {
            "Test_S1_rejection": float(np.mean(lab[is_s1] == 0)),
            "Test_S2_acceptance": float(np.mean(lab[~is_s1] == 1)),
            "diagnostic_AUC": float(roc_auc_score(
                np.r_[np.zeros(is_s1.sum()), np.ones((~is_s1).sum())],
                np.r_[te_s[is_s1], te_s[~is_s1]])),
        }

    # LOSO for the best fusion (F2 mean rank), prototypes rebuilt per fold.
    loso = []
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        fitm = np.isin(subjects, enrolled) & (f <= CFG["calibration"]["fit_fraction"])
        ki = np.flatnonzero(np.isin(subjects, enrolled) & (f > .6) & (f <= .8))
        ui = np.flatnonzero((subjects == held) & (f > .6) & (f <= .8))
        if len(ui) == 0:
            continue
        scs = []
        for b in range(n_bands):
            cols = np.arange(b * width, (b + 1) * width)
            mu, V = pca_fit(feat[:, cols], np.flatnonzero(fitm), CFG["pca_dim"])
            trp = project(feat[:, cols], mu, V)
            protos, keys = build_prototypes(trp, subjects, tasks, fitm, enrolled)
            scs.append(max_cosine(trp, protos, keys, tasks))
        scs = np.stack(scs)
        fused = np.mean([rank_normalize(scs[b], scs[b][fitm]) for b in range(n_bands)], 0)
        loso.append(operating(fused[ki], fused[ui], target))
        print(f"E125 LOSO held={held} done", flush=True)

    results["F2_mean_rank"]["LOSO_AUC"] = float(np.mean([x["AUC"] for x in loso]))
    results["F2_mean_rank"]["LOSO_rejection"] = float(np.mean([x["unknown_rejection"] for x in loso]))
    results["F2_mean_rank"]["LOSO_folds_ge_0.80"] = int(
        sum(1 for x in loso if x["unknown_rejection"] >= 0.80))

    result = {"experiment": CFG["experiment"], "pca_dim": CFG["pca_dim"],
              "variants": results}
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    lines = ["# E125 Band-wise score fusion", "",
             "E123 显示不同频带偏好相反的工作点（低频偏 S2、高频偏 S1），"
             "融合各频带的分数曲线应得到更好的 ROC。", "",
             "| 变体 | 诊断 AUC | Test S1 拒识 | Test S2 接受 | LOSO 拒识 | LOSO AUC |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, v in results.items():
        lines.append(f"| {name} | {v['diagnostic_AUC']:.4f} | {v['Test_S1_rejection']:.4f} | "
                     f"{v['Test_S2_acceptance']:.4f} | "
                     f"{v.get('LOSO_rejection', float('nan')):.4f} | "
                     f"{v.get('LOSO_AUC', float('nan')):.4f} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
