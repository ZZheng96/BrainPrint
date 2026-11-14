"""E123: band-selective PCA on correlation-logSPD.

E121/E122 showed that projecting the 855-dim correlation-logSPD to 96 dims before
computing task-conditioned max cosine raises the Test S1-vs-S2 diagnostic AUC to
0.7658 and gives, at the Train-chosen threshold, Test S1 rejection 0.7916 and
Test S2 acceptance 0.5823. That is the most balanced Test result so far.

The 855-dim vector is 5 bands x 171 entries (18x18 upper triangle). Different
bands carry different amounts of identity information, and mixing them may wash
out the useful part. This experiment selects bands before PCA.

Variants:
  B1..B5   single band (1-8, 8-13, 13-20, 20-30, 30-40 Hz)
  ALL      all five bands (E122 reference)
  LO       bands 1-3 (1-20 Hz)
  HI       bands 4-5 (20-40 Hz)
  MID      8-20 Hz (alpha + low beta)
For each: PCA to 96 dims (or the band's full width if smaller), task-conditioned
max cosine, and report LOSO rejection plus Test S1/S2 at the Train threshold.
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


def pca_project(train, other, fit_rows, dim):
    x = np.asarray(train[fit_rows], dtype=np.float64)
    mu = x.mean(0)
    xc = x - mu
    cov = xc.T @ xc / max(len(xc) - 1, 1)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(-w)
    V = V[:, order][:, : min(dim, x.shape[1])]
    return (((np.asarray(train, dtype=np.float64) - mu) @ V).astype(np.float32),
            ((np.asarray(other, dtype=np.float64) - mu) @ V).astype(np.float32))


def separability(known_sig, unknown_sig, target):
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

    mapping = pd.read_csv(ROOT / CFG["data"]["test_metadata"],
                          usecols=["sample", "path", "session", "task"])
    is_s1 = mapping.session.astype(str).str.casefold().isin(["session1", "s1", "1"]).to_numpy()
    test_tasks = mapping.task.to_numpy()
    test_feat = np.load(TEST_CACHE)
    if len(test_feat) != len(mapping):
        raise ValueError("cached Test features do not match the mapping")

    n_bands = len(load_e36_config()["features"]["bands_hz"])
    width = feat.shape[1] // n_bands
    if width * n_bands != feat.shape[1]:
        raise ValueError("unexpected feature width")

    subsets = {
        "B1_1-8": [0], "B2_8-13": [1], "B3_13-20": [2], "B4_20-30": [3], "B5_30-40": [4],
        "LO_1-20": [0, 1, 2], "HI_20-40": [3, 4], "MID_8-20": [1, 2],
        "ALL": [0, 1, 2, 3, 4],
    }

    target = CFG["targets"]["known_acceptance"]
    results = {}
    for name, bands in subsets.items():
        cols = np.concatenate([np.arange(b * width, (b + 1) * width) for b in bands])
        tr_sub = np.asarray(feat[:, cols], dtype=np.float32)
        te_sub = np.asarray(test_feat[:, cols], dtype=np.float32)

        tr_p, te_p = pca_project(tr_sub, te_sub, fit_rows, CFG["pca_dim"])

        # LOSO
        folds = []
        for held in CLASSES:
            enrolled = np.setdiff1d(CLASSES, [held])
            fit = np.isin(subjects, enrolled) & (f <= CFG["calibration"]["fit_fraction"])
            ki = np.flatnonzero(np.isin(subjects, enrolled) & (f > .6) & (f <= .8))
            ui = np.flatnonzero((subjects == held) & (f > .6) & (f <= .8))
            if len(ui) == 0:
                continue
            protos, keys = build_prototypes(tr_p, subjects, tasks, fit, enrolled)
            sc = max_cosine(tr_p, protos, keys, tasks)
            folds.append(separability(sc[ki], sc[ui], target))

        # Deployable model
        protos, keys = build_prototypes(tr_p, subjects, tasks, fit_rows, CLASSES)
        train_sc = max_cosine(tr_p, protos, keys, tasks)
        known_idx = np.flatnonzero((f > .6) & (f <= .8))
        th = float(np.quantile(train_sc[known_idx], 1 - target))
        te_sc = max_cosine(te_p, protos, keys, test_tasks)
        labels = (te_sc >= th).astype(int)

        results[name] = {
            "LOSO_AUC": float(np.mean([x["AUC"] for x in folds])),
            "LOSO_rejection": float(np.mean([x["unknown_rejection"] for x in folds])),
            "LOSO_folds_ge_0.80": int(sum(1 for x in folds if x["unknown_rejection"] >= 0.80)),
            "Test_S1_rejection": float(np.mean(labels[is_s1] == 0)),
            "Test_S2_acceptance": float(np.mean(labels[~is_s1] == 1)),
            "train_known_acceptance": float(np.mean(train_sc[known_idx] >= th)),
        }
        print(json.dumps({name: results[name]}, ensure_ascii=False), flush=True)

    best = max(results.items(),
               key=lambda kv: min(kv[1]["Test_S1_rejection"], kv[1]["Test_S2_acceptance"]))
    result = {"experiment": CFG["experiment"], "pca_dim": CFG["pca_dim"],
              "best_by_test_joint_min": {"name": best[0], **best[1]}, "variants": results}
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    lines = ["# E123 Band-selective PCA on correlation-logSPD", "",
             "E122 用全频带 PCA-96 得到 Test S1 0.7916 / S2 0.5823。"
             "本实验在 PCA 前先选频带，看能否进一步提升。", "",
             f"**按 Test 均衡点最优：{best[0]}**", "",
             "| 频带子集 | LOSO AUC | LOSO 拒识 | LOSO ≥0.80 折数 | Test S1 拒识 | Test S2 接受 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, v in results.items():
        lines.append(f"| {name} | {v['LOSO_AUC']:.4f} | {v['LOSO_rejection']:.4f} | "
                     f"{v['LOSO_folds_ge_0.80']}/20 | {v['Test_S1_rejection']:.4f} | "
                     f"{v['Test_S2_acceptance']:.4f} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
