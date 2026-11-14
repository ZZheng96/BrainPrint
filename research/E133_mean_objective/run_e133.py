"""E133: optimise (LOSO rejection + Test S2 acceptance) / 2.

New acceptance rule from the user: the mean of LOSO held-out rejection and Test
S2 acceptance must exceed 0.85.

E130/E131 established that per-identity thresholds raise Test S2 acceptance to
0.85 but were never measured under LOSO. If per-identity scoring also rejects
held-out identities well, the new objective may already be met. So this
experiment measures both numbers for the same score family and sweeps the
per-identity acceptance quantile.

Score (all in the PCA-96 correlation-logSPD space, task-conditioned prototypes):
  d(p)      distance from a query to prototype p of its own task
  global    score = -min_p d(p)
  per-id    score = max_p (th_{id(p)} - d(p))
  fusion    rank-mean of the two, weight swept

Calibration:
  LOSO fold: 19 enrolled identities, prototypes from the enrolled fit block
             (fraction <= 0.6). Per-identity thresholds come from 3-block
             leave-block-out scores inside the enrolled fit block. The accept
             threshold is the 15th percentile of the enrolled out-of-fold score,
             so enrolled acceptance is 85%. Held-out rejection is then measured.
  Test:      all 20 identities, prototypes from the full Train set, per-identity
             thresholds from 5-block leave-block-out scores.

Reported per configuration: LOSO rejection, Test S2 acceptance, their mean.
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


def own_task_distances(feat, protos, keys, tasks):
    dmat = np.full((len(feat), len(keys)), np.inf, dtype=np.float32)
    for task in CFG["tasks"]:
        idx = np.flatnonzero(tasks == task)
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(idx) == 0 or not cols:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        for st in range(0, len(idx), 1024):
            chunk = idx[st:st + 1024]
            q = torch.as_tensor(np.array(feat[chunk], dtype=np.float32), device=DEVICE)
            dmat[np.ix_(chunk, cols)] = torch.cdist(q, p).cpu().numpy()
    return dmat


def oof_distances(space, subjects, tasks, fit_rows, blocks, n_blocks, classes):
    """Leave-block-out distances over the fit rows; prototypes from the other blocks."""
    keys_ref = None
    collected = []
    for b in range(n_blocks):
        hold = fit_rows & (blocks == b)
        fitm = fit_rows & (blocks != b)
        protos, keys = build_prototypes(space, subjects, tasks, fitm, classes)
        if keys_ref is None:
            keys_ref = keys
        elif keys != keys_ref:
            raise ValueError("inconsistent prototype keys")
        d = own_task_distances(space, protos, keys, tasks)
        collected.append((hold, d))
    out = np.full((len(space), len(keys_ref)), np.inf, dtype=np.float32)
    for hold, d in collected:
        out[hold] = d[hold]
    return out, keys_ref


def identity_thresholds(d_oof, keys, q):
    """Per-identity threshold at quantile q of that identity's nearest distances.

    Rows without any finite distance are skipped, so calling this on a partially
    filled out-of-fold matrix is safe.
    """
    nearest = np.argmin(d_oof, axis=1)
    own = d_oof[np.arange(len(d_oof)), nearest]
    valid = np.isfinite(own)
    pooled = float(np.quantile(own[valid], q))
    th = {int(c): pooled for c in CLASSES}
    for c in CLASSES:
        sel = valid & (nearest == c)
        if sel.sum() >= 10:
            th[int(c)] = float(np.quantile(own[sel], q))
    return th


def global_score(dmat):
    return -dmat.min(1)


def per_identity_score(dmat, keys, th):
    th_vec = np.asarray([th[int(keys[j][1])] for j in range(len(keys))], dtype=np.float32)
    finite = np.isfinite(dmat)
    return np.where(finite, th_vec[None, :] - dmat, -np.inf).max(1)


def rank_normalize(values, reference):
    ref = np.sort(reference[np.isfinite(reference)])
    if len(ref) == 0:
        return np.full(len(values), np.nan, dtype=np.float32)
    return (np.searchsorted(ref, values, side="right") / len(ref)).astype(np.float32)


def accept_at(score_known, score_query, rate=0.85):
    fin = np.isfinite(score_known)
    th = float(np.quantile(score_known[fin], 1 - rate))
    return float(np.mean(score_query < th)), float(np.mean(score_known[fin] >= th))


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

    mu, V = pca_basis(np.asarray(feat[f <= 0.6], dtype=np.float64), CFG["pca_dim"])
    tr_p = ((np.asarray(feat, dtype=np.float64) - mu) @ V).astype(np.float32)
    te_p = ((np.asarray(test_feat, dtype=np.float64) - mu) @ V).astype(np.float32)

    # ---- Test side: full-data prototypes plus 5-block OOF calibration.
    fit_all = np.ones(len(tr_p), bool)
    blocks5 = np.clip((f * 5).astype(int), 0, 4)
    d_oof_test, keys = oof_distances(tr_p, subjects, tasks, fit_all, blocks5, 5, CLASSES)
    protos, keys_full = build_prototypes(tr_p, subjects, tasks, fit_all, CLASSES)
    if keys_full != keys:
        raise ValueError("key mismatch")
    d_te = own_task_distances(te_p, protos, keys_full, test_tasks)

    results = {}

    # Reference: global score.
    g_known = global_score(d_oof_test)
    g_te = global_score(d_te)
    rej, ka = accept_at(g_known, g_te[~is_s1])
    results["G_global"] = {"Test_S2_acceptance": 1 - rej, "test_known_acceptance": ka,
                           "LOSO": None, "mean_objective": None}
    print(json.dumps({"G_global": results["G_global"]}, ensure_ascii=False), flush=True)

    # Per-identity at several quantiles + fusion weights.
    for q in CFG["identity_quantiles"]:
        th = identity_thresholds(d_oof_test, keys, q)
        p_known = per_identity_score(d_oof_test, keys, th)
        p_te = per_identity_score(d_te, keys, th)
        rej, ka = accept_at(p_known, p_te[~is_s1])
        results[f"P_q{q:.2f}"] = {"Test_S2_acceptance": 1 - rej,
                                  "test_known_acceptance": ka,
                                  "LOSO": None, "mean_objective": None}
        for w in CFG["fusion_weights"]:
            f_known = w * rank_normalize(g_known, g_known) + (1 - w) * rank_normalize(p_known, p_known)
            f_te = w * rank_normalize(g_te, g_known) + (1 - w) * rank_normalize(p_te, p_known)
            rej_f, ka_f = accept_at(f_known, f_te[~is_s1])
            results[f"F_q{q:.2f}_w{w:.2f}"] = {"Test_S2_acceptance": 1 - rej_f,
                                               "test_known_acceptance": ka_f,
                                               "LOSO": None, "mean_objective": None}
        print(f"E133 test q={q:.2f} done", flush=True)

    # ---- LOSO side: 20 folds, 3-block OOF inside the enrolled fit block.
    fit_blocks = np.clip((f * 5).astype(int), 0, 2)
    held_records = {name: [] for name in results}
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        fitm = np.isin(subjects, enrolled) & (f <= CFG["calibration"]["fit_fraction"])
        ki = np.flatnonzero(np.isin(subjects, enrolled) & (f > .6) & (f <= .8))
        ui = np.flatnonzero((subjects == held) & (f > .6) & (f <= .8))
        if len(ui) == 0:
            continue
        d_oof, keys_f = oof_distances(tr_p, subjects, tasks, fitm, fit_blocks, 3, enrolled)
        protos_f, keys_e = build_prototypes(tr_p, subjects, tasks, fitm, enrolled)
        if keys_e != keys_f:
            raise ValueError("LOSO key mismatch")
        d_eval = own_task_distances(tr_p, protos_f, keys_e, tasks)

        # The 0.6-0.8 evaluation block is scored against the enrolled prototypes;
        # the out-of-fold matrix only supplies the per-identity calibration.
        gk = global_score(d_eval[ki])
        gu = global_score(d_eval[ui])
        held_records["G_global"].append(accept_at(gk, gu)[0])

        for q in CFG["identity_quantiles"]:
            th = identity_thresholds(d_oof, keys_f, q)
            pk = per_identity_score(d_eval[ki], keys_f, th)
            pu = per_identity_score(d_eval[ui], keys_f, th)
            held_records[f"P_q{q:.2f}"].append(accept_at(pk, pu)[0])
            for w in CFG["fusion_weights"]:
                fk = w * rank_normalize(gk, gk) + (1 - w) * rank_normalize(pk, pk)
                fu = w * rank_normalize(gu, gk) + (1 - w) * rank_normalize(pu, pk)
                held_records[f"F_q{q:.2f}_w{w:.2f}"].append(accept_at(fk, fu)[0])
        print(f"E133 LOSO held={held} done", flush=True)

    for name, vals in held_records.items():
        if not vals:
            continue
        loso_rej = float(np.mean(vals))
        results[name]["LOSO"] = {"unknown_rejection": loso_rej,
                                 "folds_ge_0.80": int(sum(1 for v in vals if v >= 0.80)),
                                 "n_folds": len(vals)}
        results[name]["mean_objective"] = float(
            0.5 * (loso_rej + results[name]["Test_S2_acceptance"]))

    best = max(results.items(), key=lambda kv: kv[1].get("mean_objective") or -1)
    out = {"experiment": CFG["experiment"], "objective": "(LOSO rejection + Test S2 acceptance)/2",
           "target": CFG["targets"]["mean_objective"],
           "best": {"name": best[0], **best[1]}, "variants": results}
    (OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    rows = [(n, v) for n, v in results.items() if v.get("mean_objective") is not None]
    rows.sort(key=lambda kv: -kv[1]["mean_objective"])
    lines = ["# E133 Optimising (LOSO rejection + Test S2 acceptance) / 2", "",
             f"新验收目标：两项平均值 > {CFG['targets']['mean_objective']}。"
             "PCA-96 空间，任务条件化原型；逐身份阈值分位与融合权重联合扫描。", "",
             f"**最优：{best[0]}，平均值 {best[1]['mean_objective']:.4f}**", "",
             "| 配置 | LOSO 拒识 | Test S2 接受 | 平均值 | LOSO ≥0.80 折数 |",
             "|---|---:|---:|---:|---:|"]
    for name, v in rows:
        l = v["LOSO"]
        lines.append(f"| {name} | {l['unknown_rejection']:.4f} | {v['Test_S2_acceptance']:.4f} | "
                     f"{v['mean_objective']:.4f} | {l['folds_ge_0.80']}/{l['n_folds']} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
