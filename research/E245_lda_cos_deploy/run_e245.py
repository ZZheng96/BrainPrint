"""E245: deployable LDA-cosine open-set scorer.

(a) inner-LOSO threshold: per outer fold enrol all 19 outer-train ids; threshold = median over the 19
    inner leave-one-identity-out models of the 90% quantile of the left-out identity's eval scores.
(b) sensitivity: shrinkage {0.01,0.05,0.1,0.3} x eval-batch task centering {off,on}; max / tnorm.
(c) final model on all 20 Train ids, Test scored descriptively (no selection on Test).
All selection uses Train LOSO only. Score convention: higher = more "known".
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E243_de_deep_fusion"))
import run_e243 as e243  # noqa: E402

DEVICE, TASKS = e243.DEVICE, e243.TASKS
SHRINK = [0.01, 0.05, 0.1, 0.3]
CENTER = [False, True]
REDUCE = ["max", "tnorm"]
Q = 0.90


def fit(x, subj, tasks, mask, ids):
    """Protos per (task, id) and eigendecomposition of pooled within-class cov (shrinkage applied later)."""
    xt = torch.as_tensor(x, device=DEVICE)
    protos, resid = {}, []
    for t in TASKS:
        rows = []
        for s in ids:
            m = torch.as_tensor(np.flatnonzero(mask & (tasks == t) & (subj == s)), device=DEVICE)
            mu = xt[m].mean(0); rows.append(mu); resid.append(xt[m] - mu)
        protos[t] = torch.stack(rows)
    r = torch.cat(resid).double()
    cov = r.T @ r / max(len(r) - 1, 1)
    w, v = torch.linalg.eigh(cov)
    fit_mean = {t: xt[torch.as_tensor(np.flatnonzero(mask & (tasks == t)), device=DEVICE)].mean(0) for t in TASKS}
    return protos, (w, v, float(w.mean())), fit_mean


def whitener(eig, lam):
    w, v, tr = eig
    return (v / ((1 - lam) * w + lam * tr).sqrt()).float()


def score(x, tasks, rows, protos, W, fit_mean, center, groups=None):
    """groups: optional array (len rows) of batch ids for centering (e.g. Test session); default one batch."""
    xt = torch.as_tensor(x[rows], device=DEVICE).clone(); tr = tasks[rows]
    g = np.zeros(len(rows), int) if groups is None else groups
    if center:
        for t in TASKS:
            for b in np.unique(g):
                idx = torch.as_tensor(np.flatnonzero((tr == t) & (g == b)), device=DEVICE)
                if len(idx):
                    xt[idx] = xt[idx] - xt[idx].mean(0) + fit_mean[t]
    k = next(iter(protos.values())).shape[0]
    sim = torch.full((len(rows), k), -2.0, device=DEVICE)
    for t in TASKS:
        idx = torch.as_tensor(np.flatnonzero(tr == t), device=DEVICE)
        if len(idx):
            a = torch.nn.functional.normalize(xt[idx] @ W, dim=1)
            p = torch.nn.functional.normalize(protos[t] @ W, dim=1)
            sim[idx] = a @ p.T
    s = sim.sort(1, descending=True).values
    rest = s[:, 1:]
    out = {"max": s[:, 0], "tnorm": (s[:, 0] - rest.mean(1)) / (rest.std(1) + 1e-6)}
    return {k_: v.cpu().numpy().astype(np.float64) for k_, v in out.items()}, sim.argmax(1).cpu().numpy()


def run_config_scores(raw, subj, tasks, fit_mask, eval_mask, ids, dim):
    """Build features on enrolled ids, return {cfg: scores on eval rows} and the eval row index."""
    tr = fit_mask & np.isin(subj, ids)
    x, _ = e243.standardize_and_pca(raw, tasks, tr, dim)
    protos, eig, fm = fit(x, subj, tasks, tr, ids)
    rows = np.flatnonzero(eval_mask)
    res = {}
    for lam in SHRINK:
        W = whitener(eig, lam)
        for c in CENTER:
            sc, _ = score(x, tasks, rows, protos, W, fm, c)
            for r in REDUCE:
                res[(lam, c, r)] = sc[r]
    return res, rows


def f1(k_s, u_s, th):
    tp = int((u_s < th).sum()); fp = int((k_s < th).sum())
    return float(2 * tp / max(2 * tp + fp + len(u_s) - tp, 1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, default=20)
    ap.add_argument("--inner", type=int, default=19, help="number of inner left-out ids per outer fold")
    args = ap.parse_args(); t0 = time.time()
    for d in ["models", "scores"]:
        (OUT / d).mkdir(exist_ok=True)
    cfg = e243.CFG
    meta = pd.read_csv(ROOT / cfg["data"]["train_metadata"])
    subj, tasks, fr = meta.subject.to_numpy(int), meta.task.astype(str).to_numpy(), e243.frac(meta)
    raw = np.load(ROOT / cfg["data"]["eeg"]).astype(np.float32)
    dim = int(cfg["pca_dim"])
    fit_all = fr <= float(cfg["fit_fraction"])
    eval_all = (fr > float(cfg["fit_fraction"])) & (fr <= float(cfg["score_fraction"][1]))
    rows_out, q_held = [], {}
    for held in range(1, args.folds + 1):
        ids = np.array([s for s in range(1, 21) if s != held])
        outer, rows = run_config_scores(raw, subj, tasks, fit_all, eval_all, ids, dim)
        is_unk = subj[rows] == held
        # inner LOSO thresholds
        inner_q = {c: [] for c in outer}
        for j in ids[: args.inner]:
            ids_in = ids[ids != j]
            em = eval_all & (subj != held)  # same batch composition (enrolled + 1 unknown) as deployment
            res, r_in = run_config_scores(raw, subj, tasks, fit_all, em, ids_in, dim)
            pj = subj[r_in] == j
            for c, v in res.items():
                inner_q[c].append(float(np.quantile(v[pj], Q)))
        for c, v in outer.items():
            k_s, u_s = v[~is_unk], v[is_unk]
            q_held[(c, held)] = float(np.quantile(u_s, Q))
            row = {"shrink": c[0], "center": c[1], "reduce": c[2], "held": held,
                   "AUC": e243.auc(-u_s, -k_s), "TPR_FAR1": float((u_s < np.quantile(k_s, .01)).mean()),
                   "TPR_FAR5": float((u_s < np.quantile(k_s, .05)).mean())}
            for name, th in [("median", float(np.median(inner_q[c]))), ("mean", float(np.mean(inner_q[c])))]:
                row[f"{name}_th"] = th
                row[f"{name}_known_acc"] = float((k_s >= th).mean())
                row[f"{name}_unknown_rej"] = float((u_s < th).mean())
                row[f"{name}_F1"] = f1(k_s, u_s, th)
            rows_out.append(row)
        print(f"held={held} done {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows_out); df.to_csv(OUT / "per_fold.csv", index=False)
    keys = ["shrink", "center", "reduce"]
    summ = df.groupby(keys).mean(numeric_only=True).drop(columns="held").reset_index()
    summ = summ.sort_values("median_F1", ascending=False); summ.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ[keys + ["AUC", "TPR_FAR1", "TPR_FAR5", "median_known_acc", "median_unknown_rej", "median_F1",
                       "mean_unknown_rej", "mean_F1"]].to_string(index=False, float_format="%.4f"))

    # ---- selection (Train LOSO only): best median_F1 ----
    best = summ.iloc[0]; lam, ctr, red = float(best.shrink), bool(best.center), str(best.reduce)
    # ---- final model: all 20 ids, fit rows (frac<=0.6) so the threshold calibration matches ----
    allm = fit_all; ids = np.arange(1, 21)
    x, states = e243.standardize_and_pca(raw, tasks, allm, dim)
    protos, eig, fm = fit(x, subj, tasks, allm, ids)
    W = whitener(eig, lam)
    # threshold for the 20-id model: inner LOSO on all 20 ids with the same fit/eval windows
    th_list = []
    for j in ids:
        res, r_in = run_config_scores(raw, subj, tasks, fit_all, eval_all, ids[ids != j], dim)
        th_list.append(float(np.quantile(res[(lam, ctr, red)][subj[r_in] == j], Q)))
    th = float(np.median(th_list))
    torch.save({"protos": {t: p.cpu() for t, p in protos.items()}, "whiten": W.cpu(), "fit_mean": {t: v.cpu() for t, v in fm.items()},
                "states": states, "enrolled": ids, "shrink": lam, "center": ctr, "reduce": red, "threshold": th,
                "inner_thresholds": th_list}, OUT / "models" / "final_lda_cos.pt")
    # ---- Test (descriptive only) ----
    tmeta = pd.read_csv(ROOT / cfg["data"]["test_metadata"])
    traw = np.load(ROOT / cfg["data"]["eeg_test"]).astype(np.float32)
    assert traw.shape[1] == raw.shape[1] and len(traw) == len(tmeta), (traw.shape, raw.shape, len(tmeta))
    ttasks = tmeta.task.astype(str).to_numpy()
    xt = e243.apply_states(traw, ttasks, {"e": states}, "e")
    sess = tmeta.session.astype(str).to_numpy()
    _, gidx = np.unique(sess, return_inverse=True)
    sc, arg = score(xt, ttasks, np.arange(len(xt)), protos, W, fm, ctr, groups=gidx)
    s = sc[red]
    pred = np.where(s >= th, ids[arg], -1)
    out = pd.DataFrame({"path": tmeta.path, "session": sess, "task": ttasks, "score": s, "nearest_id": ids[arg], "pred": pred})
    out.to_csv(OUT / "scores" / "test_scores.csv", index=False)
    desc = out.groupby("session").agg(n=("score", "size"), accept_rate=("pred", lambda p: float((p != -1).mean())),
                                      score_median=("score", "median")).reset_index()
    print(desc.to_string(index=False))
    # train-in-sample sanity: accept rate of same model on its own eval window (optimistic)
    (OUT / "results.json").write_text(json.dumps({
        "device": str(DEVICE), "folds": args.folds, "inner": args.inner, "runtime_s": time.time() - t0,
        "selected": {"shrink": lam, "center": ctr, "reduce": red}, "final_threshold": th,
        "final_inner_thresholds": th_list, "test_descriptive": desc.to_dict("records"),
        "summary": summ.to_dict("records")}, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
