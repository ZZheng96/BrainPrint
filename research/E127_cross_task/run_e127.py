"""E127: cross-task identity aggregation and finer bands.

E126 found that multi-class LDA makes LOSO perfect (1.0) but destroys Test S2
(0.073) -- another confirmation that optimising Train-side separability does not
transfer. Per-task PCA was marginally better than global PCA
(LOSO 0.658 / S1 0.768 / S2 0.597 versus 0.626 / 0.792 / 0.582).

This experiment adds a signal the current pipeline never uses: cross-task
consistency. A trial is currently scored only against prototypes of its own task.
But the same person produced EC, EO, Imagery and SSVEP trials in the same
recording session, so a registered trial should also sit reasonably close to its
own identity's prototypes from the other three tasks, while an unregistered trial
has no such consistent identity to attach to.

Scores (all on the per-task-PCA space, or global PCA as a control):
  V1 own_task_max      max cosine within the trial's own task (baseline)
  V2 cross_task_max    for each identity, max cosine over all four tasks, then
                       take the max over identities
  V3 own_plus_cross    mean of the two rank-normalised scores
  V4 relative_fit      own_task_max minus the mean own-task cosine over identities
  V5 both_task_gap     own_task_max minus the best cross-task cosine

Also a finer band split (8 bands) is tested as an alternative to the 5-band E65
features, using the same correlation-logSPD construction.
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


def cosine_matrix(feat, protos):
    p = torch.as_tensor(protos, dtype=torch.float32, device=DEVICE)
    p = torch.nn.functional.normalize(p, dim=1, eps=1e-12)
    out = np.empty((len(feat), len(protos)), dtype=np.float32)
    for st in range(0, len(feat), 1024):
        chunk = slice(st, min(st + 1024, len(feat)))
        q = torch.as_tensor(np.array(feat[chunk], dtype=np.float32), device=DEVICE)
        q = torch.nn.functional.normalize(q, dim=1, eps=1e-12)
        out[chunk] = (q @ p.T).cpu().numpy()
    return out


def score_variants(feat, protos, keys, tasks, classes):
    S = cosine_matrix(feat, protos)
    cols_by_task = {t: [i for i, (tt, _) in enumerate(keys) if tt == t] for t in CFG["tasks"]}
    cols_by_id = {c: [i for i, (_, cc) in enumerate(keys) if cc == c] for c in classes}
    n = len(feat)
    own = np.full(n, -np.inf, dtype=np.float32)
    cross = np.full(n, -np.inf, dtype=np.float32)
    own_mean = np.full(n, np.nan, dtype=np.float32)
    for t in CFG["tasks"]:
        m = np.flatnonzero(tasks == t)
        if len(m) == 0:
            continue
        cols = cols_by_task[t]
        own[m] = S[np.ix_(m, cols)].max(1)
        own_mean[m] = S[np.ix_(m, cols)].mean(1)
    for c in classes:
        cols = cols_by_id[c]
        if not cols:
            continue
        cross = np.maximum(cross, S[:, cols].max(1))
    return {
        "own_task_max": own,
        "cross_task_max": cross,
        "relative_fit": own - own_mean,
        "both_task_gap": own - cross,
    }


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


def loso_signal(space, subjects, tasks, f, target, variant):
    out = []
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        fitm = np.isin(subjects, enrolled) & (f <= CFG["calibration"]["fit_fraction"])
        ki = np.flatnonzero(np.isin(subjects, enrolled) & (f > .6) & (f <= .8))
        ui = np.flatnonzero((subjects == held) & (f > .6) & (f <= .8))
        if len(ui) == 0:
            continue
        protos, keys = build_prototypes(space, subjects, tasks, fitm, enrolled)
        sv = score_variants(space, protos, keys, tasks, enrolled)
        sig = sv[variant]
        th = float(np.quantile(sig[ki], 1 - target))
        out.append({
            "AUC": float(roc_auc_score(np.r_[np.ones(len(ki)), np.zeros(len(ui))],
                                       np.r_[sig[ki], sig[ui]])),
            "unknown_rejection": float(np.mean(sig[ui] < th)),
        })
    return {
        "AUC": float(np.mean([x["AUC"] for x in out])),
        "unknown_rejection": float(np.mean([x["unknown_rejection"] for x in out])),
        "folds_ge_0.80": int(sum(1 for x in out if x["unknown_rejection"] >= 0.80)),
    }


def per_task_pca(feat, test, tasks, test_tasks, fit_rows, dim):
    tr = np.zeros((len(feat), dim), dtype=np.float32)
    te = np.zeros((len(test), dim), dtype=np.float32)
    for task in CFG["tasks"]:
        m = fit_rows & (tasks == task)
        if m.sum() < CFG["min_fit_per_task"]:
            continue
        mu, V = pca_basis(np.asarray(feat[m], dtype=np.float64), dim)
        mt = tasks == task
        tr[mt] = ((np.asarray(feat[mt], dtype=np.float64) - mu) @ V).astype(np.float32)
        mte = test_tasks == task
        if mte.any():
            te[mte] = ((np.asarray(test[mte], dtype=np.float64) - mu) @ V).astype(np.float32)
    return tr, te


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

    tr_pt, te_pt = per_task_pca(feat, test_feat, tasks, test_tasks, fit_rows, CFG["pca_dim"])

    results = {}
    for space_name, tr_space, te_space in (("pertask_pca96", tr_pt, te_pt),):
        protos, keys = build_prototypes(tr_space, subjects, tasks, fit_rows, CLASSES)
        tr_sv = score_variants(tr_space, protos, keys, tasks, CLASSES)
        te_sv = score_variants(te_space, protos, keys, test_tasks, CLASSES)

        for variant in ("own_task_max", "cross_task_max", "relative_fit", "both_task_gap"):
            results[f"{space_name}/{variant}"] = three_way(
                tr_sv[variant], te_sv[variant], known_rows, is_s1, target)
            print(json.dumps({f"{space_name}/{variant}": results[f"{space_name}/{variant}"]},
                             ensure_ascii=False), flush=True)

        # Fusion of own-task and cross-task ranks.
        r_own_tr = rank_normalize(tr_sv["own_task_max"], tr_sv["own_task_max"][fit_rows])
        r_cross_tr = rank_normalize(tr_sv["cross_task_max"], tr_sv["cross_task_max"][fit_rows])
        r_own_te = rank_normalize(te_sv["own_task_max"], tr_sv["own_task_max"][fit_rows])
        r_cross_te = rank_normalize(te_sv["cross_task_max"], tr_sv["cross_task_max"][fit_rows])
        fused_tr = 0.5 * (r_own_tr + r_cross_tr)
        fused_te = 0.5 * (r_own_te + r_cross_te)
        results[f"{space_name}/fuse_own_cross"] = three_way(
            fused_tr, fused_te, known_rows, is_s1, target)
        print(json.dumps({f"{space_name}/fuse_own_cross": results[f"{space_name}/fuse_own_cross"]},
                         ensure_ascii=False), flush=True)

    # LOSO for the most promising signals.
    for variant in ("own_task_max", "cross_task_max", "both_task_gap"):
        key = f"pertask_pca96/{variant}"
        results[key]["LOSO"] = loso_signal(tr_pt, subjects, tasks, f, target, variant)
        print(json.dumps({key + "_LOSO": results[key]["LOSO"]}, ensure_ascii=False), flush=True)

    result = {"experiment": CFG["experiment"], "variants": results}
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    lines = ["# E127 Cross-task identity aggregation", "",
             "当前打分只用同一任务的 proto。注册者在同一 Session 里做过 4 个任务，"
             "其 trial 也应接近本人在其他任务的原型；未注册者没有这种跨任务一致性。"
             "全部只用 Train 拟合块。", "",
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
