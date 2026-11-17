"""E253f: refine the ranking projection (whitening strength / dimension / cosine).

Ranking and rejection are decoupled:
  ranking  = nearest prototype after within-class whitening (LDA geometry)
  rejection= frozen Euclidean nearest-prototype distance with a Train-calibrated tau

Selection evidence is Train-only (forward split: fit on frac<=0.6, evaluate on
frac>0.8).  Test is a single audit per configuration.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "final/team_hymn"
CACHE = ROOT / "BrainPrint_cache/e253_subwindow"
OUT = ROOT / "research/E253_session_robust/results"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(Path(__file__).parent))
import model as M  # noqa: E402
from sweep import load_artifacts, project  # noqa: E402

N_BASE = 20
DIMS = (96, 128, 192)
SHRINKS = (0.5, 0.7, 0.9, 0.99)
SEED = 20261118


def fit_pca(x, dim):
    mu = x.mean(0)
    _, _, vt = torch.linalg.svd(x - mu, full_matrices=False)
    return mu, vt[:dim].T.contiguous()


def fit_whiten(x, y, shrink):
    mu = torch.stack([x[y == s].mean(0) for s in range(1, N_BASE + 1)])
    resid = torch.cat([x[y == s] - mu[s - 1] for s in range(1, N_BASE + 1)])
    cov = resid.T @ resid / max(len(resid) - N_BASE, 1)
    cov = shrink * cov + (1.0 - shrink) * torch.diagonal(cov).mean() * torch.eye(cov.shape[0])
    vals, vecs = torch.linalg.eigh(cov)
    return (vecs * vals.clamp_min(1e-6).rsqrt()) @ vecs.T


def main() -> None:
    art = load_artifacts()
    tr_feat = np.load(CACHE / "train_feat.npy")
    te_feat = np.load(CACHE / "test_feat.npy")
    tr = pd.read_csv(ROOT / "BrainPrint_cache/train_metadata.csv")
    te = pd.read_csv(ROOT / "BrainPrint_cache/test_metadata.csv")
    lab = pd.read_csv(ROOT / "Brainprint_Competition/test.csv")
    te = te.merge(lab, left_on="sample", right_on="Sample", how="left")
    tr_tasks = M._task_indices([{"task": t} for t in tr["task"]], len(tr))
    te_tasks = M._task_indices([{"task": t} for t in te["task"]], len(te))
    tr_subj = tr["subject"].to_numpy()
    g = tr.groupby(["subject", "task"])
    frac = ((g.cumcount() + 1) / g["trial"].transform("size")).to_numpy()
    early, late = frac <= 0.6, frac > 0.8
    te_label = te["Label"].to_numpy()
    is_reg = te_label > 0

    p_tr = project(art, tr_feat, tr_tasks).mean(1)
    p_te = project(art, te_feat, te_tasks).mean(1)

    # rejection metric: frozen euclid geometry, tau from Train forward genuine distances
    protos_e = torch.stack([torch.stack([
        p_tr[early & (tr_tasks == t) & (tr_subj == s)].mean(0)
        for s in range(1, N_BASE + 1)]) for t in range(4)])

    def dist(proj, tasks, protos):
        out = torch.full((len(proj), N_BASE), float("inf"))
        for t in range(4):
            r = np.nonzero(tasks == t)[0]
            if r.size:
                out[r] = torch.cdist(proj[r], protos[t])
        return out

    tau = float(np.quantile(dist(p_tr[late], tr_tasks[late], protos_e)[
        np.arange(int(late.sum())), tr_subj[late] - 1].numpy(), 0.90))
    te_dmin = dist(p_te, te_tasks, protos_e).min(1).values.numpy()
    accept = te_dmin <= tau
    print("rejection tau", round(tau, 3), "accept", round(accept[is_reg].mean(), 3),
          "unknown_rej", round((~accept)[~is_reg].mean(), 3), flush=True)

    report = {"tau": tau}
    for dim in DIMS:
        mu, basis = fit_pca(p_tr[early], dim)
        q_tr, q_te = (p_tr - mu) @ basis, (p_te - mu) @ basis
        for shrink in SHRINKS:
            W = fit_whiten(q_tr[early], tr_subj[early], shrink)
            w_tr, w_te = q_tr @ W, q_te @ W
            protos = torch.stack([torch.stack([
                w_tr[early & (tr_tasks == t) & (tr_subj == s)].mean(0)
                for s in range(1, N_BASE + 1)]) for t in range(4)])
            # Train forward ranking accuracy
            sc = torch.full((int(late.sum()), N_BASE), float("inf"))
            for t in range(4):
                r = np.nonzero(tr_tasks[late] == t)[0]
                if r.size:
                    sc[r] = torch.cdist(w_tr[late][r], protos[t])
            tr_acc = float((sc.argmin(1).numpy() + 1 == tr_subj[late]).mean())
            # Test audit
            sc_te = dist(w_te, te_tasks, protos)
            pred = sc_te.argmin(1).numpy() + 1
            e = {"dim": dim, "shrink": shrink, "train_forward_acc": tr_acc,
                 "test_top1": float((pred == te_label)[is_reg].mean()),
                 "test_acc_given_accept": float((pred == te_label)[is_reg & accept].mean()),
                 "test_e2e": float(((pred == te_label) & accept)[is_reg].mean()),
                 "test_unknown_rejection": float((~accept)[~is_reg].mean())}
            e["test_J"] = (3 * e["test_e2e"] + e["test_unknown_rejection"]) / 4
            report[f"dim{dim}_shrink{shrink}"] = e
            print(f"dim{dim} shrink{shrink} train_fwd {tr_acc:.3f} test_top1 "
                  f"{e['test_top1']:.3f} e2e {e['test_e2e']:.3f} J {e['test_J']:.3f}",
                  flush=True)
    (OUT / "ranking.json").write_text(json.dumps(report, indent=2, default=float),
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
