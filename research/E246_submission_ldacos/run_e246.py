"""E246: LDA-whitened cosine vs Euclidean on the submission's own 855-d features (Train only).

Protocol = final/tools/selftest.py simulation (seed 20261116, 20 reps): 3 new users (20 enrollment
samples from frac<=0.6, tested on frac>0.8), 3 unregistered (all rows), 14 registered (tested frac>0.8).
Regimes:
  frozen : submission weights (task stats + PCA saw all 20 ids) -> optimistic, = README numbers.
  honest : task stats + PCA96 + whitening refit on the 14 registered ids only (new/unknown unseen).
Scorers: euclid (score=-dist) and ldacos(shrink) (score=cos sim in LDA-whitened space).
Centering: none / batch128 (random batches of 128 eval rows, per-task mean shift to fit mean).
Threshold: frozen tau (euclid only) and inner-LOSO over the 14 registered ids
(median over left-out ids of the 90% quantile of the left-out id's max score).
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "final/team_hymn"))
import model as M  # noqa: E402

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SHRINK = [0.01, 0.1]
CENTER = ["none", "batch128"]
Q = 0.90
DIM = 96


def auc(pos, neg):
    """P(pos > neg)."""
    a = np.concatenate([pos, neg]); r = pd.Series(a).rank().to_numpy()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class Space:
    """Per-task z-score + shared PCA (frozen from artifact, or refit on given rows)."""

    def __init__(self, feats, tasks, art=None, stat_rows=None, pca_rows=None):
        if art is not None:
            self.mu, self.sd = art["task_mu"], art["task_sd"]
            self.pm, self.pb = art["pca_mean"], art["pca_basis"]
            return
        mu, sd = [], []
        for t in range(4):
            f = feats[torch.as_tensor(np.flatnonzero(stat_rows & (tasks == t)), device=DEV)]
            mu.append(f.mean(0)); sd.append(f.std(0).clamp_min(1e-6))
        self.mu, self.sd = torch.stack(mu), torch.stack(sd)
        z = self.task_z(feats, tasks, pca_rows)
        self.pm = z.mean(0)
        _, _, vh = torch.linalg.svd(z - self.pm, full_matrices=False)
        self.pb = vh[:DIM].T.contiguous()

    def task_z(self, feats, tasks, rows):
        idx = np.flatnonzero(rows); z = torch.empty(len(idx), feats.shape[1], device=DEV)
        for t in range(4):
            m = tasks[idx] == t
            if m.any():
                z[torch.as_tensor(np.flatnonzero(m), device=DEV)] = (feats[torch.as_tensor(idx[m], device=DEV)] - self.mu[t]) / self.sd[t]
        return z

    def proj(self, f, t):
        return ((f - self.mu[t]) / self.sd[t] - self.pm) @ self.pb


def build(feats, tasks, subj, frac, space, known, enroll_idx, enroll_lab):
    """Prototypes (4, 23, D), valid mask, fit means per task, LDA eigendecomposition (registered ids)."""
    protos = torch.zeros(4, 23, DIM, device=DEV); valid = torch.zeros(23, dtype=torch.bool, device=DEV)
    resid, fit_mean = [], []
    for t in range(4):
        fm = []
        for s in known:
            r = torch.as_tensor(np.flatnonzero((subj == s) & (tasks == t) & (frac <= 0.6)), device=DEV)
            p = space.proj(feats[r], t); mu = p.mean(0)
            protos[t, s - 1] = mu; resid.append(p - mu); fm.append(p)
        fit_mean.append(torch.cat(fm).mean(0))
    valid[torch.as_tensor(known - 1, device=DEV)] = True
    fe = feats[torch.as_tensor(enroll_idx, device=DEV)]
    for uid in np.unique(enroll_lab):
        m = torch.as_tensor(enroll_lab == uid, device=DEV)
        for t in range(4):
            protos[t, uid - 1] = space.proj(fe[m], t).mean(0)
        valid[uid - 1] = True
    r = torch.cat(resid).double(); cov = r.T @ r / (len(r) - 1)
    w, v = torch.linalg.eigh(cov)
    return protos, valid, torch.stack(fit_mean), (w, v, float(w.mean()))


def whiten(eig, lam):
    w, v, tr = eig
    return (v / ((1 - lam) * w + lam * tr).sqrt()).float()


def score(feats, tasks, rows, space, protos, valid, fit_mean, W, center, rng):
    """Returns max score (higher = more known) and argmax identity (1..23) for eval rows."""
    idx = np.flatnonzero(rows)
    P = torch.empty(len(idx), DIM, device=DEV)
    for t in range(4):
        m = tasks[idx] == t
        if m.any():
            P[torch.as_tensor(np.flatnonzero(m), device=DEV)] = space.proj(feats[torch.as_tensor(idx[m], device=DEV)], t)
    if center == "batch128":
        order = rng.permutation(len(idx)); tt = tasks[idx]
        for b in range(0, len(order), 128):
            ob = order[b:b + 128]
            for t in range(4):
                sel = ob[tt[ob] == t]
                if len(sel) >= 8:
                    st = torch.as_tensor(sel, device=DEV)
                    P[st] = P[st] - P[st].mean(0) + fit_mean[t]
    out = {}
    for t_name, Wt in W.items():
        S = torch.full((len(idx), 23), -float("inf"), device=DEV)
        for t in range(4):
            m = torch.as_tensor(np.flatnonzero(tasks[idx] == t), device=DEV)
            if not len(m):
                continue
            if Wt is None:
                s = -torch.cdist(P[m], protos[t])
            else:
                a = torch.nn.functional.normalize(P[m] @ Wt, dim=1)
                p = torch.nn.functional.normalize(protos[t] @ Wt, dim=1)
                s = a @ p.T
            S[m] = s.masked_fill(~valid[None], -float("inf"))
        mx, am = S.max(1)
        out[t_name] = (mx.cpu().numpy().astype(np.float64), am.cpu().numpy() + 1)
    return out


def main():
    t0 = time.time()
    tm = pd.read_csv(ROOT / "BrainPrint_cache/train_metadata.csv")
    tm["frac"] = tm.trial / tm.groupby(["subject", "task"])["trial"].transform("max")  # = E200 bundle
    feats = torch.as_tensor(np.load(ROOT / "BrainPrint_cache/e201_final_selftest/train_raw_scores_inputs.npy"), device=DEV)
    tasks = M._task_indices([{"task": t} for t in tm.task], len(tm))
    subj, frac = tm.subject.to_numpy(), tm.frac.to_numpy()
    art = M.load_model(str(ROOT / "final/team_hymn"), str(DEV))
    tau = art["tau"]
    scorers = ["euclid"] + [f"ldacos{l}" for l in SHRINK]
    rng = np.random.default_rng(20261116)
    rows = []
    for rep in range(20):
        perm = rng.permutation(np.arange(1, 21))
        new_s, unk_s, known = perm[:3], perm[3:6], np.sort(perm[6:])
        mapping = {int(s): 21 + k for k, s in enumerate(new_s)}
        pick = np.concatenate([rng.choice(np.flatnonzero((subj == s) & (frac <= 0.6)), 20, replace=False) for s in new_s])
        lab = np.array([mapping[int(s)] for s in subj[pick]])
        late = frac > 0.8
        r_known, r_new, r_unk = late & np.isin(subj, known), late & np.isin(subj, new_s), np.isin(subj, unk_s)
        truth_known, truth_new = subj[r_known], np.array([mapping[int(s)] for s in subj[r_new]])
        for regime in ["frozen", "honest"]:
            stat = np.isin(subj, known)
            space = Space(feats, tasks, art=art) if regime == "frozen" else Space(feats, tasks, stat_rows=stat, pca_rows=stat & (frac <= 0.6))
            protos, valid, fm, eig = build(feats, tasks, subj, frac, space, known, pick, lab)
            W = {"euclid": None, **{f"ldacos{l}": whiten(eig, l) for l in SHRINK}}
            # inner LOSO thresholds over the 14 registered ids (same space; prototype of j removed,
            # whitening refit without j)
            inner = {(s_, c): [] for s_ in scorers for c in CENTER}
            for j in known:
                kj = known[known != j]
                if regime == "honest":  # j must be unseen by stats/PCA too, like a real unknown
                    sj = np.isin(subj, kj)
                    space_j = Space(feats, tasks, stat_rows=sj, pca_rows=sj & (frac <= 0.6))
                else:
                    space_j = space
                pj, vj, fmj, eigj = build(feats, tasks, subj, frac, space_j, kj, pick, lab)
                Wj = {"euclid": None, **{f"ldacos{l}": whiten(eigj, l) for l in SHRINK}}
                rj = late & (subj == j)
                for c in CENTER:
                    # batch composition: the left-out id's rows mixed with other registered eval rows
                    rows_j = late & np.isin(subj, known)
                    res = score(feats, tasks, rows_j, space_j, pj, vj, fmj, Wj, c, rng)
                    isj = subj[np.flatnonzero(rows_j)] == j
                    for s_ in scorers:
                        inner[(s_, c)].append(float(np.quantile(res[s_][0][isj], Q)))
                del rj
            ev = r_known | r_new | r_unk
            ev_idx = np.flatnonzero(ev)
            gk, gn, gu = r_known[ev_idx], r_new[ev_idx], r_unk[ev_idx]
            for c in CENTER:
                res = score(feats, tasks, ev, space, protos, valid, fm, W, c, rng)
                for s_ in scorers:
                    mx, am = res[s_]
                    ths = {"loso": float(np.median(inner[(s_, c)]))}
                    if s_ == "euclid":
                        ths["tau"] = -tau
                    for th_name, th in ths.items():
                        pred = np.where(mx >= th, am, 0)
                        tp = int((pred[gu] == 0).sum()); fp = int((pred[~gu] == 0).sum()); fn = int(gu.sum()) - tp
                        rows.append({"rep": rep, "regime": regime, "scorer": s_, "center": c, "thr": th_name,
                                     "AUC": auc(mx[~gu], mx[gu]),
                                     "acc_known": float((pred[gk] == truth_known).mean()),
                                     "acc_new": float((pred[gn] == truth_new).mean()),
                                     "rej_unknown": float((pred[gu] == 0).mean()),
                                     "F1_reject": 2 * tp / max(2 * tp + fp + fn, 1),
                                     "closed_acc": float(np.concatenate([am[gk] == truth_known, am[gn] == truth_new]).mean())})
        print(f"rep {rep} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "per_rep.csv", index=False)
    keys = ["regime", "scorer", "center", "thr"]
    g = df.groupby(keys)
    summ = g.mean(numeric_only=True).drop(columns="rep")
    summ["acc_new_min"] = g.acc_new.min(); summ["rej_unknown_min"] = g.rej_unknown.min()
    summ = summ.reset_index(); summ.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False, float_format="%.3f"))
    (OUT / "results.json").write_text(json.dumps({"device": str(DEV), "runtime_s": time.time() - t0,
                                                  "summary": summ.to_dict("records")}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
