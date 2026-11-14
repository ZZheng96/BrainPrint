"""E130: per-identity thresholds from out-of-fold Train scores.

E122-E129 all use one global distance threshold. But identities differ a lot in
how tightly their trials cluster, so a single threshold systematically
over-rejects the diffuse identities and under-rejects the tight ones. Since Test
S2 acceptance is the largest remaining gap, equalising per-identity acceptance
inside Train is the natural fix, and it stays fully Train-only.

Mechanism:
  1. Score every Train trial against prototypes built from the other four time
     blocks (5-fold leave-block-out), recording the distance and the nearest
     identity.
  2. For identity c, set threshold_c to the 15th percentile of the distances of
     the trials whose nearest identity is c, so each identity keeps 85%
     acceptance out of fold.
  3. Accept a query if any prototype p satisfies d(p) <= threshold_{id(p)}.
     Equivalently score = max_p (threshold_{id(p)} - d(p)) and accept if >= 0.

Variants:
  G  global threshold (E4 reference: fit<=1.0, OOF threshold)
  P1 per-identity thresholds
  P2 per-prototype (task, identity) thresholds
  P3 per-task thresholds

Reported together: LOSO rejection, Test S1 rejection, Test S2 acceptance, Test
diagnostic AUC.
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
    """Distance to every prototype of the sample's own task plus the column index."""
    n = len(feat)
    dmat = np.full((n, len(keys)), np.inf, dtype=np.float32)
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


def global_score(dmat):
    return -dmat.min(1)


def per_identity_score(dmat, keys, thresholds, mode):
    """max over prototypes of (threshold - distance); accept when >= 0."""
    cols = len(keys)
    if mode == "identity":
        th_vec = np.asarray([thresholds[int(keys[j][1])] for j in range(cols)], dtype=np.float32)
    elif mode == "prototype":
        th_vec = np.asarray([thresholds[j] for j in range(cols)], dtype=np.float32)
    elif mode == "task":
        th_vec = np.asarray([thresholds[keys[j][0]] for j in range(cols)], dtype=np.float32)
    else:
        raise ValueError(mode)
    finite = np.isfinite(dmat)
    s = np.where(finite, th_vec[None, :] - dmat, -np.inf)
    return s.max(1)


def oof_distances(space, subjects, tasks, blocks, n_blocks):
    """Leave-one-block-out distance matrix; prototypes from the other blocks."""
    dmat = np.full((len(space), 0), np.inf, dtype=np.float32)
    templates = None
    collected = []
    for b in range(n_blocks):
        hold = blocks == b
        fitm = ~hold
        protos, keys = build_prototypes(space, subjects, tasks, fitm, CLASSES)
        d = own_task_distances(space, protos, keys, tasks)
        collected.append((b, hold, d, protos, keys))
        templates = keys
        print(f"E130 OOF block {b + 1}/{n_blocks} done", flush=True)
    out = np.full((len(space), len(templates)), np.inf, dtype=np.float32)
    for b, hold, d, protos, keys in collected:
        if keys != templates:
            raise ValueError("prototype key order differs across folds")
        out[hold] = d[hold]
    return out, templates


def calibrate(d_oof, keys, nearest, target, mode):
    q = 1 - target
    if mode == "identity":
        th = {}
        for c in CLASSES:
            sel = nearest == c
            if sel.sum() >= 10:
                th[int(c)] = float(np.quantile(d_oof[np.arange(len(d_oof)), nearest][sel], q))
        # Fallback for identities with too few samples.
        pooled = float(np.quantile(d_oof[np.arange(len(d_oof)), nearest], q))
        for c in CLASSES:
            th.setdefault(int(c), pooled)
        return th
    if mode == "prototype":
        used = np.unique(nearest)
        pooled = float(np.quantile(d_oof[np.arange(len(d_oof)), nearest], q))
        return {int(j): (float(np.quantile(d_oof[nearest == j, j], q))
                         if (nearest == j).sum() >= 10 else pooled) for j in range(len(keys))}
    if mode == "task":
        pooled = float(np.quantile(d_oof[np.arange(len(d_oof)), nearest], q))
        th = {}
        for j, (t, _) in enumerate(keys):
            th.setdefault(t, [])
            th[t].append(float(np.quantile(d_oof[nearest == j, j], q))
                         if (nearest == j).sum() >= 10 else pooled)
        return {t: float(np.mean(v)) for t, v in th.items()}
    raise ValueError(mode)


def evaluate(train_sig, test_sig, is_s1, target, label):
    finite = np.isfinite(train_sig)
    th = float(np.quantile(train_sig[finite], 1 - target))
    lab = (test_sig >= th).astype(int)
    return {
        "label": label,
        "threshold": th,
        "train_known_acceptance": float(np.mean(train_sig[finite] >= th)),
        "Test_S1_rejection": float(np.mean(lab[is_s1] == 0)),
        "Test_S2_acceptance": float(np.mean(lab[~is_s1] == 1)),
        "diagnostic_AUC": float(roc_auc_score(
            np.r_[np.zeros(is_s1.sum()), np.ones((~is_s1).sum())],
            np.r_[test_sig[is_s1], test_sig[~is_s1]])),
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
    mu, V = pca_basis(np.asarray(feat[f <= 0.6], dtype=np.float64), CFG["pca_dim"])
    tr_p = ((np.asarray(feat, dtype=np.float64) - mu) @ V).astype(np.float32)
    te_p = ((np.asarray(test_feat, dtype=np.float64) - mu) @ V).astype(np.float32)

    blocks = np.clip((f * CFG["oof_blocks"]).astype(int), 0, CFG["oof_blocks"] - 1)
    d_oof, keys = oof_distances(tr_p, subjects, tasks, blocks, CFG["oof_blocks"])
    nearest = np.argmin(d_oof, axis=1)

    # Final prototypes from all Train trials.
    protos, keys_full = build_prototypes(tr_p, subjects, tasks, np.ones(len(tr_p), bool), CLASSES)
    if keys_full != keys:
        raise ValueError("full-data prototype keys differ from OOF order")
    d_test = own_task_distances(te_p, protos, keys_full, test_tasks)

    results = {}
    results["G_global"] = evaluate(global_score(d_oof), global_score(d_test), is_s1, target,
                                   "global threshold")
    print(json.dumps({"G_global": results["G_global"]}, ensure_ascii=False), flush=True)

    for mode, name in (("identity", "P1_per_identity"),
                       ("prototype", "P2_per_prototype"),
                       ("task", "P3_per_task")):
        th = calibrate(d_oof, keys, nearest, target, mode)
        tr_s = per_identity_score(d_oof, keys, th, mode)
        te_s = per_identity_score(d_test, keys, th, mode)
        results[name] = evaluate(tr_s, te_s, is_s1, target, mode)
        print(json.dumps({name: results[name]}, ensure_ascii=False), flush=True)
        np.savez(OUT / f"thresholds_{mode}.npz",
                 th=np.asarray([th[k] for k in sorted(th, key=str)], dtype=np.float64),
                 keys=np.asarray([str(k) for k in sorted(th, key=str)]))

    result = {"experiment": CFG["experiment"], "pca_dim": CFG["pca_dim"], "variants": results}
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    lines = ["# E130 Per-identity thresholds from out-of-fold scores", "",
             "身份之间的紧致程度差异很大，单一全局阈值会系统性过度拒绝松散的identity。"
             "本实验用 5 折 leave-block-out 的 Train 分数为每个身份（或原型/任务）单独标定阈值。"
             "全部只用 Train。", "",
             "| 变体 | 阈值类型 | Test 诊断 AUC | Test S1 拒识 | Test S2 接受 |",
             "|---|---|---:|---:|---:|"]
    for name, v in results.items():
        lines.append(f"| {name} | {v['label']} | {v['diagnostic_AUC']:.4f} | "
                     f"{v['Test_S1_rejection']:.4f} | {v['Test_S2_acceptance']:.4f} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
