"""E254e: nested identity-LOSO selection check.

For every outer held identity h:
  * inner loop over the remaining 19 identities (leave-one-out inside the outer
    training pool) scores each short-listed configuration
  * the configuration is chosen from inner metrics only
  * the chosen configuration is then evaluated on h and on the outer known rows

This verifies that the LOSO gain is not an artefact of selecting on the same
folds that are reported.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "BrainPrint_cache"
OUT = Path(__file__).resolve().parent / "results"
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DIMS = (192, 256)
SHRINKS = (0.9, 0.95)
KS = (3, 4)
BETAS = (0.0, 0.5)
N_ID = 20


def auroc(known, unknown):
    x = np.concatenate([known, unknown])
    ranks = pd.Series(x).rank(method="average").to_numpy()
    return float((ranks[:len(known)].sum() - len(known) * (len(known) + 1) / 2)
                 / (len(known) * len(unknown)))


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


def fold_metrics(feat, tasks, subj, frac, order, fit_ids, unknown_id):
    """Return {config: (auroc, u95, rej90)} for one identity-LOSO fold."""
    fit = torch.zeros(len(feat), dtype=torch.bool, device=DEV)
    for s in fit_ids:
        fit |= subj == s
    rows_e = fit & (frac <= 0.4)
    rows_f = fit & (frac <= 0.6)
    evk = fit & (frac > 0.6)
    evu = subj == unknown_id

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

    out = {}
    for dim in DIMS:
        q0 = (z - pm) @ basis[:, :dim]
        for shrink in SHRINKS:
            resid = []
            for s in fit_ids:
                r = rows_e & (subj == s)
                resid.append(q0[r] - q0[r].mean(0))
            r = torch.cat(resid)
            covw = r.T @ r / max(len(r) - len(fit_ids), 1)
            covw = shrink * covw + (1 - shrink) * torch.diagonal(covw).mean() * torch.eye(
                dim, device=DEV)
            wv, we = torch.linalg.eigh(covw)
            q = q0 @ ((we * wv.clamp_min(1e-6).rsqrt()) @ we.T)
            for k in KS:
                pf, vf = build_protos(q, rows_f, subj, tasks, order, fit_ids, k)
                Sk = score_full(q, tasks, pf, vf, evk).topk(2, dim=1).values
                Su = score_full(q, tasks, pf, vf, evu).topk(2, dim=1).values
                for beta in BETAS:
                    kk = (Sk[:, 0] + beta * (Sk[:, 0] - Sk[:, 1])).cpu().numpy()
                    uu = (Su[:, 0] + beta * (Su[:, 0] - Su[:, 1])).cpu().numpy()
                    r90 = float((uu < np.quantile(kk, 0.10)).mean())
                    r95 = float((uu < np.quantile(kk, 0.05)).mean())
                    out[f"d{dim}_s{shrink}_k{k}_b{beta}"] = {
                        "auroc": auroc(kk, uu), "u95": r95, "rej90": r90}
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

    chosen, outer = Counter(), []
    for held in range(1, N_ID + 1):
        pool = [s for s in range(1, N_ID + 1) if s != held]
        inner = {}
        for held_in in pool:
            fit_ids = [s for s in pool if s != held_in]
            inner[held_in] = fold_metrics(feat, tasks, subj, frac, order, fit_ids, held_in)
        agg = {}
        for key in inner[pool[0]]:
            agg[key] = {m: float(np.mean([inner[s][key][m] for s in pool]))
                        for m in ("auroc", "u95", "rej90")}
        feasible = [k for k, v in agg.items()
                    if v["auroc"] >= 0.98 and v["u95"] >= 0.95 and v["rej90"] > 0.90]
        if feasible:
            best = max(feasible, key=lambda k: agg[k]["rej90"])
        else:
            best = max(agg, key=lambda k: agg[k]["auroc"] + agg[k]["u95"] + agg[k]["rej90"])
        chosen[best] += 1
        outer.append({"held": held, "config": best, "inner": agg[best],
                      "outer": fold_metrics(feat, tasks, subj, frac, order, pool, held)[best]})
        print(f"held {held:2d} -> {best:24s} outer auc {outer[-1]['outer']['auroc']:.4f} "
              f"u95 {outer[-1]['outer']['u95']:.4f} rej90 {outer[-1]['outer']['rej90']:.4f}",
              flush=True)

    au = np.array([r["outer"]["auroc"] for r in outer])
    u95 = np.array([r["outer"]["u95"] for r in outer])
    r90 = np.array([r["outer"]["rej90"] for r in outer])
    report = {
        "protocol": "E254e nested identity-LOSO, inner selection",
        "chosen": dict(chosen),
        "outer_auroc_mean": float(au.mean()), "outer_auroc_min": float(au.min()),
        "outer_u95_mean": float(u95.mean()), "outer_u95_min": float(u95.min()),
        "outer_rej90_mean": float(r90.mean()), "outer_rej90_min": float(r90.min()),
        "pass_auroc": bool((au >= 0.98).all()), "pass_u95": bool((u95 >= 0.95).all()),
        "pass_rej90": bool((r90 > 0.90).all()),
        "folds": outer,
    }
    (OUT / "loso_nested.json").write_text(json.dumps(report, indent=2, default=float),
                                          encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "folds"}, indent=2, default=float))


if __name__ == "__main__":
    main()
