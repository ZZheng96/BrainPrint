"""E254c: score normalisation + larger dims for strict identity-LOSO.

Ranking is LDA-cos (within-class whitening + cosine).  Three normalisations:
  none    : raw cosine
  id      : (s_c - mu_c) / sd_c, mu/sd from the identity's calibration genuine scores
  task    : (s_c - mu_task) / sd_task, pooled over identities per task
Calibration rows (0.4 < frac <= 0.6) never enter their own prototype.
Evaluation: known rows frac > 0.6 vs all rows of the held identity.
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
OUT = Path(__file__).resolve().parent / "results"
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DIMS = (96, 128, 160)
SHRINKS = (0.7, 0.9)
KS = (2, 3)
BETAS = (0.0, 0.5)
NORMS = ("none", "id", "task")
N_ID = 20


def auroc(known, unknown):
    x = np.concatenate([known, unknown])
    ranks = pd.Series(x).rank(method="average").to_numpy()
    return float((ranks[:len(known)].sum() - len(known) * (len(known) + 1) / 2)
                 / (len(known) * len(unknown)))


def score_full(q, tasks, protos, valid, mask):
    """Full (n, 20) LDA-cos score matrix for the selected rows."""
    idx = torch.nonzero(mask).flatten()
    out = torch.full((len(idx), N_ID), -float("inf"), device=q.device)
    for t in range(4):
        sel = tasks[idx] == t
        if not bool(sel.any()):
            continue
        sub = idx[sel]
        x = q[sub]
        p = protos[t]
        xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        pn = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        out[sel] = (xn[:, None, None, :] * pn[None]).sum(-1).max(2).values.masked_fill(
            ~valid[None], -float("inf"))
    return out, idx


def build_protos(q, rows, subj, tasks, order, ids, k):
    protos = torch.zeros(4, N_ID, k, q.shape[1], device=q.device)
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

    results = {}
    for held in range(1, N_ID + 1):
        keep = subj != held
        rows_e = keep & (frac <= 0.4)
        rows_f = keep & (frac <= 0.6)
        cal = keep & (frac > 0.4) & (frac <= 0.6)
        evk = keep & (frac > 0.6)
        evu = subj == held
        mu = torch.stack([feat[rows_e & (tasks == t)].mean(0) for t in range(4)])
        sd = torch.stack([feat[rows_e & (tasks == t)].std(0).clamp_min(1e-6) for t in range(4)])
        z = torch.empty_like(feat)
        for t in range(4):
            m = tasks == t
            z[m] = (feat[m] - mu[t]) / sd[t]
        x_fit = z[rows_e]
        pm = x_fit.mean(0)
        xc = x_fit - pm
        cov = xc.T @ xc / max(len(x_fit) - 1, 1)
        vals, vecs = torch.linalg.eigh(cov)
        basis = vecs[:, torch.argsort(vals, descending=True)[:max(DIMS)]].contiguous()
        ids = [s for s in range(1, N_ID + 1) if s != held]
        sub_cal, sub_k, sub_u = subj[cal], subj[evk], subj[evu]
        task_cal, task_k = tasks[cal], tasks[evk]

        for dim in DIMS:
            q0 = (z - pm) @ basis[:, :dim]
            for shrink in SHRINKS:
                resid = []
                for s in ids:
                    r = rows_e & (subj == s)
                    resid.append(q0[r] - q0[r].mean(0))
                r = torch.cat(resid)
                covw = r.T @ r / max(len(r) - len(ids), 1)
                covw = shrink * covw + (1 - shrink) * torch.diagonal(covw).mean() * torch.eye(
                    dim, device=DEV)
                wv, we = torch.linalg.eigh(covw)
                W = (we * wv.clamp_min(1e-6).rsqrt()) @ we.T
                q = q0 @ W
                y = q
                for k in KS:
                    pe, ve = build_protos(y, rows_e, subj, tasks, order, ids, k)
                    pf, vf = build_protos(y, rows_f, subj, tasks, order, ids, k)
                    S_c, _ = score_full(y, tasks, pe, ve, cal)
                    S_k, _ = score_full(y, tasks, pf, vf, evk)
                    S_u, _ = score_full(y, tasks, pf, vf, evu)
                    # per-identity / per-task normalisation statistics from calibration rows
                    mu_id = torch.zeros(N_ID, device=DEV)
                    sd_id = torch.ones(N_ID, device=DEV)
                    for s in range(1, N_ID + 1):
                        rows = sub_cal == s
                        if bool(rows.any()):
                            col = S_c[rows, s - 1]
                            mu_id[s - 1] = col.mean()
                            sd_id[s - 1] = col.std().clamp_min(1e-4)
                    mu_t = torch.zeros(4, device=DEV)
                    sd_t = torch.ones(4, device=DEV)
                    for t in range(4):
                        rows = task_cal == t
                        if bool(rows.any()):
                            col = S_c[rows][torch.arange(int(rows.sum()), device=DEV),
                                           sub_cal[rows] - 1]
                            mu_t[t] = col.mean()
                            sd_t[t] = col.std().clamp_min(1e-4)
                    variants = {}
                    variants["none"] = (S_k, S_u)
                    variants["id"] = ((S_k - mu_id) / sd_id, (S_u - mu_id) / sd_id)
                    variants["task"] = (
                        (S_k - mu_t[task_k][:, None]) / sd_t[task_k][:, None],
                        (S_u - mu_t[tasks[evu]][:, None]) / sd_t[tasks[evu]][:, None])
                    for norm in NORMS:
                        Sk, Su = variants[norm]
                        top_k = Sk.topk(2, dim=1).values
                        top_u = Su.topk(2, dim=1).values
                        for beta in BETAS:
                            kk = (top_k[:, 0] + beta * (top_k[:, 0] - top_k[:, 1])).cpu().numpy()
                            uu = (top_u[:, 0] + beta * (top_u[:, 0] - top_u[:, 1])).cpu().numpy()
                            key = f"d{dim}_s{shrink}_k{k}_{norm}_b{beta}"
                            folds = results.setdefault(key, [])
                            e = {"held": held, "auroc": auroc(kk, uu)}
                            for a in (0.90, 0.95):
                                tau = float(np.quantile(kk, 1.0 - a))
                                e[f"rej{a:.2f}"] = float((uu < tau).mean())
                            folds.append(e)
        print(f"held {held:2d} done", flush=True)

    summary = {}
    for key, folds in results.items():
        au = np.array([f["auroc"] for f in folds])
        r90 = np.array([f["rej0.90"] for f in folds])
        r95 = np.array([f["rej0.95"] for f in folds])
        summary[key] = {"auroc": float(au.mean()), "auroc_min": float(au.min()),
                        "rej90": float(r90.mean()), "rej90_min": float(r90.min()),
                        "u95": float(r95.mean()), "u95_min": float(r95.min()),
                        "worst5_u95": float(np.sort(r95)[:5].mean())}
    (OUT / "loso_norm.json").write_text(json.dumps({"grid": summary}, indent=2), encoding="utf-8")
    ranked = sorted(summary.items(), key=lambda kv: (
        kv[1]["auroc"] + kv[1]["u95"] + kv[1]["rej90"]), reverse=True)
    for key, v in ranked[:12]:
        print(f"{key:26s} auc {v['auroc']:.4f} u95 {v['u95']:.4f} rej90 {v['rej90']:.4f} "
              f"min90 {v['rej90_min']:.3f} min95 {v['u95_min']:.3f}")


if __name__ == "__main__":
    main()
