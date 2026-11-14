"""E137: multi-view score fusion under the clean protocol.

Goal: raise Test diagnostic AUC from ~0.78 to ~0.86 so that
(S1 rejection + S2 acceptance) can reach ~1.58 and the clean three-way mean
(LOSO + Test S1 rejection + Test S2 acceptance)/3 exceeds 0.85.

Cleanliness:
  - PCA bases, prototypes, per-task score standardisation and thresholds are all
    fitted on Train only (fit block = frac <= 0.6 for bases/prototypes,
    5-block out-of-fold over the whole Train set for the score reference).
  - Test is only ever *scored*; no Test label, no Test batch statistic, no
    Test rank/CDF calibration.
  - View selection uses LOSO AUC only (Train-side).

Per view and prototype count k we build, per task:
  score = -min Euclidean distance to the (task, identity) prototypes
then map to a z-score with the Train out-of-fold mean/std of that task.
Fusion = equal-weight mean of the view z-scores.
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
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TASKS = CFG["tasks"]


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def pca_basis(x, dim):
    mu = x.mean(0)
    xc = x - mu
    w, V = np.linalg.eigh(xc.T @ xc / max(len(xc) - 1, 1))
    return mu, V[:, np.argsort(-w)][:, : min(dim, x.shape[1])]


def kmeans_t(x, k, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    if len(x) <= k:
        return x.clone()
    idx = rng.choice(len(x), size=k, replace=False)
    c = x[idx].clone()
    for _ in range(iters):
        d = torch.cdist(x, c)
        a = d.argmin(1)
        new = torch.stack([x[a == j].mean(0) if (a == j).any() else c[j] for j in range(k)])
        if torch.allclose(new, c, atol=1e-6):
            break
        c = new
    return c


def task_standardize(tr, te, tasks, test_tasks):
    """Per-task z-score with TRAIN statistics only."""
    tr = tr.astype(np.float32).copy()
    te = te.astype(np.float32).copy()
    for t in TASKS:
        m = tasks == t
        mu = tr[m].mean(0)
        sd = tr[m].std(0) + 1e-6
        tr[m] = (tr[m] - mu) / sd
        q = test_tasks == t
        te[q] = (te[q] - mu) / sd
    return tr, te


def build_prototypes(feat, subjects, tasks, fit_mask, classes, k=1,
                     proto_mask=None):
    """Prototypes per (task, identity). proto_mask defaults to fit_mask."""
    pm = fit_mask if proto_mask is None else proto_mask
    protos, keys = [], []
    for task in TASKS:
        for c in classes:
            sel = pm & (tasks == task) & (subjects == c)
            if sel.sum() == 0:
                continue
            block = torch.as_tensor(np.asarray(feat[sel], dtype=np.float32), device=DEVICE)
            if k == 1:
                protos.append(block.mean(0).cpu().numpy())
                keys.append((task, int(c)))
            else:
                for centre in kmeans_t(block, k, seed=int(c) * 17).cpu().numpy():
                    protos.append(centre)
                    keys.append((task, int(c)))
    return np.stack(protos), keys


def neg_d1(feat, protos, keys, tasks):
    out = np.full(len(feat), np.nan, dtype=np.float32)
    for task in TASKS:
        idx = np.flatnonzero(tasks == task)
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(idx) == 0 or not cols:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        for st in range(0, len(idx), 2048):
            chunk = idx[st:st + 2048]
            q = torch.as_tensor(np.asarray(feat[chunk], dtype=np.float32), device=DEVICE)
            out[chunk] = (-torch.cdist(q, p)).max(1).values.cpu().numpy()
    return out


def z_by_task(values, tasks, ref, ref_tasks=None):
    """z-score each task using the Train out-of-fold reference of that task."""
    rt = tasks if ref_tasks is None else ref_tasks
    out = np.full(len(values), np.nan, dtype=np.float32)
    for t in TASKS:
        m = tasks == t
        r = ref[rt == t]
        r = r[np.isfinite(r)]
        if len(r) < 10:
            continue
        mu, sd = float(r.mean()), float(r.std()) + 1e-6
        out[m] = (values[m] - mu) / sd
    return out


def main():
    meta = pd.read_csv(ROOT / CFG["data"]["train_metadata"])
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)
    tm = pd.read_csv(ROOT / CFG["data"]["test_metadata"])
    is_s1 = tm.session.astype(str).str.casefold().eq("session1").to_numpy()
    test_tasks = tm.task.to_numpy()

    fit_rows = f <= CFG["fit_fraction"]
    blocks5 = np.clip((f * 5).astype(int), 0, 4)

    y_test = np.r_[np.zeros(int(is_s1.sum())), np.ones(int((~is_s1).sum()))]

    # ---------- per-view scores ----------
    store = {}  # (view, k) -> dict of z-scores
    for view, (ptr, pte) in CFG["views"].items():
        xtr = np.load(ROOT / ptr)
        xte = np.load(ROOT / pte)
        if len(xtr) != len(meta) or len(xte) != len(tm):
            print(f"SKIP {view}: shape mismatch {xtr.shape} {xte.shape}", flush=True)
            continue
        xtr, xte = task_standardize(xtr, xte, tasks, test_tasks)
        mu, V = pca_basis(np.asarray(xtr[fit_rows], dtype=np.float64), CFG["pca_dim"])
        trp = ((xtr.astype(np.float64) - mu) @ V).astype(np.float32)
        tep = ((xte.astype(np.float64) - mu) @ V).astype(np.float32)

        for k in CFG["prototype_k"]:
            # Deployed: prototypes from the whole Train set (clean, more data).
            protos, keys = build_prototypes(trp, subjects, tasks, fit_mask=fit_rows,
                                            classes=CLASSES, k=k,
                                            proto_mask=np.ones(len(trp), bool))
            te_raw = neg_d1(tep, protos, keys, test_tasks)

            # Train out-of-fold reference (5 blocks, prototype excludes the block).
            ref = np.full(len(trp), np.nan, dtype=np.float32)
            for b in range(5):
                hold = blocks5 == b
                p2, k2 = build_prototypes(trp, subjects, tasks, fit_mask=~hold,
                                          classes=CLASSES, k=k, proto_mask=~hold)
                s = neg_d1(trp, p2, k2, tasks)
                ref[hold] = s[hold]

            # LOSO folds: enrolled identities only, threshold side = known block.
            loso = []
            for held in CLASSES:
                enrolled = np.setdiff1d(CLASSES, [held])
                pm = np.isin(subjects, enrolled)
                ki = pm & (f > .6) & (f <= .8)
                ui = (subjects == held) & (f > .6) & (f <= .8)
                if ui.sum() == 0:
                    continue
                p2, k2 = build_prototypes(trp, subjects, tasks, fit_mask=pm,
                                          classes=enrolled, k=k, proto_mask=pm)
                s = neg_d1(trp, p2, k2, tasks)
                loso.append({"known": s[ki], "unknown": s[ui],
                             "known_task": tasks[ki], "unknown_task": tasks[ui]})

            te_z = z_by_task(te_raw, test_tasks, ref, tasks)
            ref_z = z_by_task(ref, tasks, ref)
            store[(view, k)] = dict(te=te_z, ref=ref_z, raw_loso=loso, ref_raw=ref)
            auc = roc_auc_score(np.r_[np.zeros(int(is_s1.sum())), np.ones(int((~is_s1).sum()))],
                                np.r_[te_z[is_s1], te_z[~is_s1]])
            print(f"{view:14s} k={k}  Test AUC {auc:.4f}", flush=True)
            store[(view, k)]["auc"] = float(auc)
    import pickle
    with open(OUT / "view_scores.pkl", "wb") as fh:
        pickle.dump({"store": store,
                     "is_s1": is_s1, "test_tasks": test_tasks,
                     "subjects": subjects, "tasks": tasks, "f": f}, fh)
    print("saved view scores", flush=True)


if __name__ == "__main__":
    main()
