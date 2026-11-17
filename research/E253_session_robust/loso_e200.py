"""E253n: run the strict LOSO gates on the ORIGINAL final package.

The rejection head is byte-identical between E200 and E253 (same task_mu/sd, PCA,
prototypes; only tau moved from 19.2638 to 19.1706, and the gates use a quantile
of the genuine distances rather than tau).  So #5/#6/#11 must come out identical.
This script computes them for E200 to make that explicit instead of asserting it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "BrainPrint_cache/e253_subwindow"
OUT = ROOT / "research/E253_session_robust/results"
sys.path.insert(0, str(ROOT / "research/E253_session_robust"))
sys.path.insert(0, str(ROOT / "final/team_hymn"))
import model as M  # noqa: E402
from gates11 import DEV, N_BASE, build_protos, decide, fit_head  # noqa: E402


def main() -> None:
    feat = torch.from_numpy(np.load(CACHE / "train_feat.npy")).float().mean(1).to(DEV)
    tr = pd.read_csv(ROOT / "BrainPrint_cache/train_metadata.csv")
    tasks = torch.from_numpy(M._task_indices([{"task": t} for t in tr["task"]], len(tr))).to(DEV)
    subj = torch.from_numpy(tr["subject"].to_numpy()).to(DEV)
    g = tr.groupby(["subject", "task"])
    frac = ((g.cumcount() + 1) / g["trial"].transform("size")).to_numpy()
    early = torch.from_numpy(frac <= 0.6).to(DEV)
    late = torch.from_numpy(frac > 0.8).to(DEV)

    def zscore(rows):
        mu = torch.zeros(4, feat.shape[-1], device=DEV)
        sd = torch.ones(4, feat.shape[-1], device=DEV)
        for t in range(4):
            r = rows & (tasks == t)
            mu[t] = feat[r].mean(0)
            sd[t] = feat[r].std(0).clamp_min(1e-6)
        return (feat - mu[tasks]) / sd[tasks]

    ks, us = [], []
    for held_id in range(1, N_BASE + 1):
        ids = [s for s in range(1, N_BASE + 1) if s != held_id]
        fit = early & (subj != held_id)
        z = zscore(fit)
        mu, basis, _ = fit_head(z, fit, tasks, subj)  # W unused for euclid rejection
        p = (z - mu) @ basis
        protos, valid = build_protos(p, tasks, subj, fit, ids)
        dk, *_ = decide(p[late & (subj != held_id)], tasks[late & (subj != held_id)],
                        protos, valid, float("inf"))
        du, *_ = decide(p[late & (subj == held_id)], tasks[late & (subj == held_id)],
                        protos, valid, float("inf"))
        ks.append(dk.cpu().numpy())
        us.append(du.cpu().numpy())
    ks, us = np.concatenate(ks), np.concatenate(us)
    order = np.argsort(np.concatenate([ks, us]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    n_k, n_u = len(ks), len(us)
    auc = 1.0 - float((ranks[:n_k].sum() - n_k * (n_k + 1) / 2) / (n_k * n_u))
    res = {"5_strict_loso_auroc": auc,
           "11_strict_loso_unknown_rej_at_90": float((us > np.quantile(ks, 0.90)).mean()),
           "6_strict_loso_u_at_95": float((us > np.quantile(ks, 0.95)).mean()),
           "median_known": float(np.median(ks)), "median_unknown": float(np.median(us)),
           "package": "final/team_hymn (E200)"}
    print(json.dumps(res, indent=2))
    (OUT / "gates11_e200.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
