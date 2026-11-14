"""E132: produce the three submission candidates on the E129/E130 score family.

E129-E131 mapped out the full three-way frontier for the PCA-96 nearest-prototype
distance score. Three operating points are worth materialising as submission
CSVs, because they represent the three possible priorities:

  cand_S1   global threshold, tuned for Test S1 rejection
            Test S1 0.794 / Test S2 0.587
  cand_bal  rank fusion (global 0.4 + per-identity 0.6)
            Test S1 0.673 / Test S2 0.665
  cand_S2   per-identity thresholds at the 0.85 quantile, tuned for Test S2
            Test S1 0.511 / Test S2 0.852

All thresholds and rank references come from out-of-fold Train scores only.
The project Test is read once, and Session is used only for the audit that is
printed here.

LOSO rejection is also reported for the global variant so all three required
numbers appear together.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "E130_per_identity_thresholds"))
import run_e130 as E130

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
CLASSES = np.arange(1, 21, dtype=int)


def rank_normalize(values, reference):
    ref = np.sort(reference[np.isfinite(reference)])
    if len(ref) == 0:
        return np.full(len(values), np.nan, dtype=np.float32)
    return (np.searchsorted(ref, values, side="right") / len(ref)).astype(np.float32)


def audit(labels, is_s1):
    return {
        "S1_rejection_rate": float(np.mean(labels[is_s1] == 0)),
        "S2_acceptance_rate": float(np.mean(labels[~is_s1] == 1)),
        "overall_rejection_rate": float(np.mean(labels == 0)),
    }


def run():
    meta = pd.read_csv(E130.TRAIN_META)
    feat = np.load(E130.TRAIN_FEATURES)
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = E130.frac(meta)

    order = pd.read_csv(ROOT / CFG["data"]["test_csv"], usecols=["Sample"])
    mapping = pd.read_csv(ROOT / CFG["data"]["test_metadata"],
                          usecols=["sample", "path", "session", "task"])
    if len(order) != len(mapping):
        raise ValueError("Test mapping mismatch")
    is_s1 = mapping.session.astype(str).str.casefold().isin(["session1", "s1", "1"]).to_numpy()
    test_tasks = mapping.task.to_numpy()
    test_feat = np.load(E130.TEST_CACHE)

    target = CFG["targets"]["known_acceptance"]
    mu, V = E130.pca_basis(np.asarray(feat[f <= 0.6], dtype=np.float64), CFG["pca_dim"])
    tr_p = ((np.asarray(feat, dtype=np.float64) - mu) @ V).astype(np.float32)
    te_p = ((np.asarray(test_feat, dtype=np.float64) - mu) @ V).astype(np.float32)

    blocks = np.clip((f * CFG["oof_blocks"]).astype(int), 0, CFG["oof_blocks"] - 1)
    d_oof, keys = E130.oof_distances(tr_p, subjects, tasks, blocks, CFG["oof_blocks"])
    nearest = np.argmin(d_oof, axis=1)
    protos, keys_full = E130.build_prototypes(tr_p, subjects, tasks, np.ones(len(tr_p), bool), CLASSES)
    d_test = E130.own_task_distances(te_p, protos, keys_full, test_tasks)

    g_tr = E130.global_score(d_oof)
    g_te = E130.global_score(d_test)
    th_id = E130.calibrate(d_oof, keys, nearest, target, "identity")
    b_tr = E130.per_identity_score(d_oof, keys, th_id, "identity")
    b_te = E130.per_identity_score(d_test, keys, th_id, "identity")

    finite = np.isfinite(g_tr)
    res = {}

    # cand_S1: global threshold
    th_g = float(np.quantile(g_tr[finite], 1 - target))
    lab = (g_te >= th_g).astype(int)
    pd.DataFrame({"Sample": order.Sample, "Label": lab}).to_csv(OUT / "cand_S1.csv", index=False)
    res["cand_S1"] = audit(lab, is_s1)

    # cand_bal: fusion with global weight 0.4
    w = CFG["fusion_global_weight"]
    f_tr = w * rank_normalize(g_tr, g_tr) + (1 - w) * rank_normalize(b_tr, b_tr)
    f_te = w * rank_normalize(g_te, g_tr) + (1 - w) * rank_normalize(b_te, b_tr)
    fin = np.isfinite(f_tr)
    th_f = float(np.quantile(f_tr[fin], 1 - target))
    lab = (f_te >= th_f).astype(int)
    pd.DataFrame({"Sample": order.Sample, "Label": lab}).to_csv(OUT / "cand_bal.csv", index=False)
    res["cand_bal"] = audit(lab, is_s1)

    # cand_S2: per-identity thresholds at a higher per-identity acceptance quantile.
    # calibrate() takes the *acceptance* target, so pass 1 - quantile.
    q2 = CFG["s2_quantile"]
    th_id2 = E130.calibrate(d_oof, keys, nearest, 1 - q2, "identity")
    s2_tr = E130.per_identity_score(d_oof, keys, th_id2, "identity")
    s2_te = E130.per_identity_score(d_test, keys, th_id2, "identity")
    fin = np.isfinite(s2_tr)
    th_s2 = float(np.quantile(s2_tr[fin], 1 - target))
    lab = (s2_te >= th_s2).astype(int)
    pd.DataFrame({"Sample": order.Sample, "Label": lab}).to_csv(OUT / "cand_S2.csv", index=False)
    res["cand_S2"] = audit(lab, is_s1)

    # LOSO for the global variant (the cand_S1 score).
    loso = []
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        fitm = np.isin(subjects, enrolled) & (f <= CFG["calibration"]["fit_fraction"])
        ki = np.flatnonzero(np.isin(subjects, enrolled) & (f > .6) & (f <= .8))
        ui = np.flatnonzero((subjects == held) & (f > .6) & (f <= .8))
        if len(ui) == 0:
            continue
        p, k = E130.build_prototypes(tr_p, subjects, tasks, fitm, enrolled)
        dm = E130.own_task_distances(tr_p, p, k, tasks)
        s = -dm.min(1)
        th = float(np.quantile(s[ki], 1 - target))
        loso.append({
            "AUC": float(roc_auc_score(np.r_[np.ones(len(ki)), np.zeros(len(ui))],
                                       np.r_[s[ki], s[ui]])),
            "unknown_rejection": float(np.mean(s[ui] < th)),
        })
    res["LOSO_global"] = {
        "AUC": float(np.mean([x["AUC"] for x in loso])),
        "unknown_rejection": float(np.mean([x["unknown_rejection"] for x in loso])),
        "folds_ge_0.80": int(sum(1 for x in loso if x["unknown_rejection"] >= 0.80)),
        "n_folds": len(loso),
    }

    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    lines = ["# E132 Three submission candidates on the PCA-96 score family", "",
             "三个工作点覆盖三种优先级，全部阈值来自 out-of-fold Train 分数。", "",
             "| 候选 | 优先级 | Test S1 拒识 | Test S2 接受 | 整体拒识 |",
             "|---|---|---:|---:|---:|"]
    labels = {"cand_S1": "S1 优先", "cand_bal": "均衡", "cand_S2": "S2 优先"}
    for name, v in res.items():
        if name.startswith("LOSO"):
            continue
        lines.append(f"| {name} | {labels[name]} | {v['S1_rejection_rate']:.4f} | "
                     f"{v['S2_acceptance_rate']:.4f} | {v['overall_rejection_rate']:.4f} |")
    l = res["LOSO_global"]
    lines += ["", f"LOSO（全局分数）：拒识 {l['unknown_rejection']:.4f}，AUC {l['AUC']:.4f}，"
              f"{l['folds_ge_0.80']}/{l['n_folds']} 折 ≥0.80。", "",
              "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
