"""E254f: freeze the selected LDA-cos registration model on the full Train set.

Config (pre-registered by the Train-only LOSO grid):
  features : E253 6x200 sub-window 855-d mean
  PCA      : 256
  metric   : within-class whitening (shrink 0.95) + cosine
  proto    : 4 trial-order chunks per (task, identity)
  gate     : s = s1 + 0.5 * (s1 - s2) < tau -> unregistered
  tau      : 10th percentile of cross-fitted genuine late scores (90% acceptance)

Writes research/E254_loso_registration/frozen/weights.pt and manifest.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "BrainPrint_cache"
OUT = Path(__file__).resolve().parent / "frozen"
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DIM = 256
SHRINK = 0.95
K = 4
BETA = 0.5
N_ID = 20
ACCEPT = 0.90


def build_protos(q, rows, subj, tasks, order, ids, k):
    protos = torch.zeros(4, N_ID, k, q.shape[-1], device=q.device)
    valid = torch.zeros(N_ID, dtype=torch.bool, device=q.device)
    for s in ids:
        valid[s - 1] = True
        for t in range(4):
            idx = torch.nonzero(rows & (subj == s) & (tasks == t)).flatten()
            idx = idx[torch.argsort(order[idx])]
            for j in range(k):
                chunk = idx[j * len(idx) // k:(j + 1) * len(idx) // k]
                protos[t, s - 1, j] = q[chunk].mean(0)
    return protos, valid


def score_full(q, tasks, protos, valid, mask):
    idx = torch.nonzero(mask).flatten()
    out = torch.full((len(idx), N_ID), -float("inf"), device=q.device)
    for t in range(4):
        sel = tasks[idx] == t
        if not bool(sel.any()):
            continue
        x = q[idx[sel]]
        p = protos[t]
        xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        pn = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        out[sel] = (xn[:, None, None, :] * pn[None]).sum(-1).max(2).values.masked_fill(
            ~valid[None], -float("inf"))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(CACHE / "train_metadata.csv")
    feat = torch.from_numpy(
        np.load(CACHE / "e253_subwindow/train_feat.npy").astype(np.float32).mean(1)).to(DEV)
    sys.path.insert(0, str(ROOT / "final2/team_hymn"))
    import model as M  # noqa: E402

    tasks = torch.as_tensor(M._task_indices([{"task": t} for t in meta["task"]], len(meta)), device=DEV)
    subj = torch.as_tensor(meta["subject"].to_numpy(), device=DEV)
    order = torch.as_tensor(meta.groupby(["subject", "task"]).cumcount().to_numpy(), device=DEV)
    g = meta.groupby(["subject", "task"])
    frac = torch.as_tensor(((g.cumcount() + 1) / g["trial"].transform("size")).to_numpy(), device=DEV)
    ids = list(range(1, N_ID + 1))
    every = torch.ones(len(feat), dtype=torch.bool, device=DEV)

    mu = torch.stack([feat[tasks == t].mean(0) for t in range(4)])
    sd = torch.stack([feat[tasks == t].std(0).clamp_min(1e-6) for t in range(4)])
    z = torch.empty_like(feat)
    for t in range(4):
        m = tasks == t
        z[m] = (feat[m] - mu[t]) / sd[t]
    pm = z.mean(0)
    xc = z - pm
    cov = xc.T @ xc / max(len(z) - 1, 1)
    vals, vecs = torch.linalg.eigh(cov)
    basis = vecs[:, torch.argsort(vals, descending=True)[:DIM]].contiguous()
    q0 = (z - pm) @ basis

    resid = []
    for s in ids:
        r = subj == s
        resid.append(q0[r] - q0[r].mean(0))
    r = torch.cat(resid)
    covw = r.T @ r / max(len(r) - len(ids), 1)
    covw = SHRINK * covw + (1 - SHRINK) * torch.diagonal(covw).mean() * torch.eye(DIM, device=DEV)
    wv, we = torch.linalg.eigh(covw)
    W = (we * wv.clamp_min(1e-6).rsqrt()) @ we.T
    q = q0 @ W

    protos, valid = build_protos(q, every, subj, tasks, order, ids, K)
    pe, ve = build_protos(q, frac <= 0.4, subj, tasks, order, ids, K)
    pmd, vmd = build_protos(q, frac <= 0.6, subj, tasks, order, ids, K)

    def gate(mat):
        top = mat.topk(2, dim=1).values
        return top[:, 0] + BETA * (top[:, 0] - top[:, 1])

    early = gate(score_full(q, tasks, pe, ve, (frac > 0.4) & (frac <= 0.6))).cpu().numpy()
    late = gate(score_full(q, tasks, pmd, vmd, frac > 0.6)).cpu().numpy()
    tau_early = float(np.quantile(early, 1.0 - ACCEPT))
    tau_late = float(np.quantile(late, 1.0 - ACCEPT))
    tau = tau_late

    bundle = {
        "feature": {
            "channels": 18, "sampling_rate_hz": 300, "subwindow_length": 200, "n_subwindows": 6,
            "bands_hz": [[1.0, 8.0], [8.0, 13.0], [13.0, 20.0], [20.0, 30.0], [30.0, 40.0]],
            "channel_std_ddof": 1, "channel_std_epsilon": 1e-6, "covariance_shrinkage": 0.05,
            "eigenvalue_floor": 1e-8, "off_diagonal_multiplier": 1.4142135623730951,
            "zband_epsilon": 1e-6,
        },
        "task_mu": mu.cpu(), "task_sd": sd.cpu(), "pca_mean": pm.cpu(), "pca_basis": basis.cpu(),
        "whiten": W.cpu(), "prototypes": protos.cpu(), "valid": valid.cpu(),
        "tau_score": tau, "beta": BETA, "dim": DIM, "shrink": SHRINK, "k": K,
        "protocol": "E254 nested-LOSO LDA-cos k4 d256 shrink0.95",
        "calibration": {"tau_late": tau_late, "tau_early": tau_early,
                        "acc_late_at_tau": float((late >= tau).mean()),
                        "acc_early_at_tau": float((early >= tau).mean())},
    }
    torch.save(bundle, OUT / "weights.pt")
    (OUT / "manifest.json").write_text(json.dumps({
        "dim": DIM, "shrink": SHRINK, "k": K, "beta": BETA,
        "tau": tau, "calibration": bundle["calibration"],
        "n_train_rows": int(len(feat)), "identities": ids,
    }, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in bundle["calibration"].items()}, indent=2))
    print("tau", tau)


if __name__ == "__main__":
    main()
