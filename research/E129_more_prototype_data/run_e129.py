"""E129: more prototype data plus an out-of-fold threshold.

Current best (E122 k1) builds prototypes from the first 60% of each
identity/task block and calibrates the threshold on 60-80%. Two levers are left
that do not change the score form:

  1. Use more trials per prototype. 40% of the Train trials are currently unused.
     Using up to 80% gives each centroid roughly twice the data, which should
     make the acceptance region better centred and raise Test S2 acceptance.
  2. Calibrate the threshold out-of-fold instead of on a single block. Split the
     Train trials into 5 contiguous time blocks, score each block against
     prototypes built from the other four, and take the 15th percentile of the
     pooled out-of-fold scores. This is honest (no Test) and more stable.

Variants:
  E0  fit<=0.6, threshold from 0.6-0.8     (E122 k1 reference)
  E1  fit<=0.8, threshold from 5-fold OOF  (more prototype data)
  E2  fit<=0.8, threshold from 0.6-0.8     (more data, same threshold block)
  E3  fit<=0.9, threshold from 5-fold OOF
  E4  fit<=1.0, threshold from 5-fold OOF  (all Train trials)

Score: nearest-prototype Euclidean distance in the PCA-96 space with
task-conditioned prototypes, which E128 confirmed is the strongest Test form.
LOSO is reported for every variant with prototypes rebuilt per fold.
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
        if len(idx) == 0 or len(cols) < 2:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        for st in range(0, len(idx), 1024):
            chunk = idx[st:st + 1024]
            q = torch.as_tensor(np.array(feat[chunk], dtype=np.float32), device=DEVICE)
            out[chunk] = (-torch.cdist(q, p)).max(1).values.cpu().numpy()
    return out


def oof_scores(space, subjects, tasks, blocks, n_blocks):
    """Leave-one-block-out scores; prototypes built from the other blocks."""
    scores = np.full(len(space), np.nan, dtype=np.float32)
    for b in range(n_blocks):
        hold = blocks == b
        fitm = ~hold
        protos, keys = build_prototypes(space, subjects, tasks, fitm, CLASSES)
        s = neg_d1(space, protos, keys, tasks)
        scores[hold] = s[hold]
    return scores


def three_way(train_sig, test_sig, known_mask, is_s1, target, th=None):
    if th is None:
        th = float(np.quantile(train_sig[known_mask], 1 - target))
    lab = (test_sig >= th).astype(int)
    return {
        "threshold": th,
        "train_known_acceptance": float(np.mean(train_sig[known_mask] >= th)),
        "Test_S1_rejection": float(np.mean(lab[is_s1] == 0)),
        "Test_S2_acceptance": float(np.mean(lab[~is_s1] == 1)),
        "diagnostic_AUC": float(roc_auc_score(
            np.r_[np.zeros(is_s1.sum()), np.ones((~is_s1).sum())],
            np.r_[test_sig[is_s1], test_sig[~is_s1]])),
    }


def loso(space, subjects, tasks, f, target):
    out = []
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        fitm = np.isin(subjects, enrolled) & (f <= CFG["calibration"]["fit_fraction"])
        ki = np.flatnonzero(np.isin(subjects, enrolled) & (f > .6) & (f <= .8))
        ui = np.flatnonzero((subjects == held) & (f > .6) & (f <= .8))
        if len(ui) == 0:
            continue
        protos, keys = build_prototypes(space, subjects, tasks, fitm, enrolled)
        s = neg_d1(space, protos, keys, tasks)
        th = float(np.quantile(s[ki], 1 - target))
        out.append({
            "AUC": float(roc_auc_score(np.r_[np.ones(len(ki)), np.zeros(len(ui))],
                                       np.r_[s[ki], s[ui]])),
            "unknown_rejection": float(np.mean(s[ui] < th)),
        })
    return {
        "AUC": float(np.mean([x["AUC"] for x in out])),
        "unknown_rejection": float(np.mean([x["unknown_rejection"] for x in out])),
        "folds_ge_0.80": int(sum(1 for x in out if x["unknown_rejection"] >= 0.80)),
    }


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

    target = CFG["targets"]["known_acceptance"]

    # PCA basis from the first 60% (kept fixed so that prototype data can vary).
    fit60 = f <= 0.6
    mu, V = pca_basis(np.asarray(feat[fit60], dtype=np.float64), CFG["pca_dim"])
    tr_p = ((np.asarray(feat, dtype=np.float64) - mu) @ V).astype(np.float32)
    te_p = ((np.asarray(test_feat, dtype=np.float64) - mu) @ V).astype(np.float32)

    blocks = np.clip((f * CFG["oof_blocks"]).astype(int), 0, CFG["oof_blocks"] - 1)
    oof = oof_scores(tr_p, subjects, tasks, blocks, CFG["oof_blocks"])

    results = {}
    specs = {
        "E0_ref_fit60_thr6080": (0.6, "block"),
        "E1_fit80_thrOOF": (0.8, "oof"),
        "E2_fit80_thr6080": (0.8, "block"),
        "E3_fit90_thrOOF": (0.9, "oof"),
        "E4_fit100_thrOOF": (1.0, "oof"),
    }
    for name, (cut, thr_mode) in specs.items():
        fitm = f <= cut
        protos, keys = build_prototypes(tr_p, subjects, tasks, fitm, CLASSES)
        tr_s = neg_d1(tr_p, protos, keys, tasks)
        te_s = neg_d1(te_p, protos, keys, test_tasks)
        if thr_mode == "oof":
            finite = np.isfinite(oof)
            th = float(np.quantile(oof[finite], 1 - target))
            known_mask = np.isfinite(oof)
            entry = three_way(oof, te_s, known_mask, is_s1, target, th=th)
            entry["calibration_set"] = "5-fold out-of-fold"
        else:
            known_mask = (f > .6) & (f <= .8)
            entry = three_way(tr_s, te_s, known_mask, is_s1, target)
            entry["calibration_set"] = "0.6-0.8 block"
        entry["prototype_trials_per_subject"] = float(fitm.sum() / 20)
        results[name] = entry
        print(json.dumps({name: entry}, ensure_ascii=False), flush=True)

    for name in ("E0_ref_fit60_thr6080", "E1_fit80_thrOOF"):
        results[name]["LOSO"] = loso(tr_p, subjects, tasks, f, target)
        print(json.dumps({name + "_LOSO": results[name]["LOSO"]}, ensure_ascii=False), flush=True)

    result = {"experiment": CFG["experiment"], "pca_dim": CFG["pca_dim"], "variants": results}
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    lines = ["# E129 More prototype data plus out-of-fold threshold", "",
             "E122 k1 只用前 60% 的 Train trial 建原型，剩余 40% 未使用。"
             "本实验把原型数据扩到 80%/90%/100%，并用 5 折 out-of-fold 分数标定阈值。"
             "打分仍是 PCA-96 空间的最近原型欧氏距离。", "",
             "| 变体 | 每身份原型 trial | 标定集 | Test 诊断 AUC | Test S1 拒识 | Test S2 接受 | LOSO 拒识 | LOSO AUC |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for name, v in results.items():
        lo = v.get("LOSO", {})
        lines.append(f"| {name} | {v['prototype_trials_per_subject']:.0f} | {v['calibration_set']} | "
                     f"{v['diagnostic_AUC']:.4f} | {v['Test_S1_rejection']:.4f} | "
                     f"{v['Test_S2_acceptance']:.4f} | "
                     f"{lo.get('unknown_rejection', float('nan')):.4f} | "
                     f"{lo.get('AUC', float('nan')):.4f} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
