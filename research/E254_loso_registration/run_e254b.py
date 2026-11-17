"""E254b: progressive-prototype strict identity-LOSO.

Improves on run_e254.py:
  * calibration scores use prototypes built from frac<=0.4 (mid rows are queries)
  * evaluation scores use prototypes built from frac<=0.6 (late rows are queries)
  -> twice the prototype data, still no query leaks into its own prototype
  * oracle thresholds reported (quantile of eval genuine scores) so #6/#11 match
    the gates11 definition, plus deployment thresholds from calibration rows
  * feature source selectable: E253 sub-window mean (855d) or E201 full window

Usage: python run_e254b.py [sub|full]
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

DIMS = (64, 96, 128)
SHRINKS = (0.3, 0.5, 0.7)
PROTOS = ("k1", "k2")
BETAS = (0.0, 0.5)
N_ID = 20


def auroc(known: np.ndarray, unknown: np.ndarray) -> float:
    x = np.concatenate([known, unknown])
    ranks = pd.Series(x).rank(method="average").to_numpy()
    return float((ranks[:len(known)].sum() - len(known) * (len(known) + 1) / 2)
                 / (len(known) * len(unknown)))


def fit_stats(feat, tasks, rows):
    mu = torch.stack([feat[rows & (tasks == t)].mean(0) for t in range(4)])
    sd = torch.stack([feat[rows & (tasks == t)].std(0).clamp_min(1e-6) for t in range(4)])
    return mu, sd


def zscore(feat, tasks, mu, sd):
    z = torch.empty_like(feat)
    for t in range(4):
        m = tasks == t
        z[m] = (feat[m] - mu[t]) / sd[t]
    return z


def fit_pca(z, rows, max_dim):
    x = z[rows]
    mu = x.mean(0)
    xc = x - mu
    cov = xc.T @ xc / max(len(x) - 1, 1)
    vals, vecs = torch.linalg.eigh(cov)
    order = torch.argsort(vals, descending=True)[:max_dim]
    return mu, vecs[:, order].contiguous()


def fit_whiten(q, rows, subj, ids, shrink):
    resid = []
    for s in ids:
        r = rows & (subj == s)
        resid.append(q[r] - q[r].mean(0))
    r = torch.cat(resid)
    cov = r.T @ r / max(len(r) - len(ids), 1)
    cov = shrink * cov + (1.0 - shrink) * torch.diagonal(cov).mean() * torch.eye(
        cov.shape[0], device=cov.device, dtype=cov.dtype)
    vals, vecs = torch.linalg.eigh(cov)
    return (vecs * vals.clamp_min(1e-6).rsqrt()) @ vecs.T


def build_protos(q, rows, subj, tasks, order, ids, k):
    dim = q.shape[1]
    protos = torch.zeros(4, N_ID, k, dim, device=q.device)
    valid = torch.zeros(N_ID, dtype=torch.bool, device=q.device)
    for s in ids:
        valid[s - 1] = True
        for t in range(4):
            sel = rows & (subj == s) & (tasks == t)
            idx = torch.nonzero(sel).flatten()
            idx = idx[torch.argsort(order[idx])]
            if k == 1:
                protos[t, s - 1, 0] = q[idx].mean(0)
            else:
                half = max(len(idx) // 2, 1)
                protos[t, s - 1, 0] = q[idx[:half]].mean(0)
                protos[t, s - 1, 1] = q[idx[half:]].mean(0)
    return protos, valid


def score_matrix(q, tasks, protos, valid, metric, mask):
    idx = torch.nonzero(mask).flatten()
    n = len(idx)
    best = torch.full((n, N_ID), -float("inf"), device=q.device)
    for t in range(4):
        sel = tasks[idx] == t
        if not bool(sel.any()):
            continue
        sub = idx[sel]
        x = q[sub]
        p = protos[t]
        if metric == "euclid":
            d = torch.cdist(x, p.reshape(-1, p.shape[-1])).reshape(len(sub), N_ID, p.shape[1])
            best[sel] = -d.min(2).values.masked_fill(~valid[None], float("inf"))
        else:
            xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            pn = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            s = (xn[:, None, None, :] * pn[None]).sum(-1).max(2).values
            best[sel] = s.masked_fill(~valid[None], -float("inf"))
    top2 = best.topk(2, dim=1).values
    return top2[:, 0], top2[:, 1]


def load_feature(mode: str):
    meta = pd.read_csv(CACHE / "train_metadata.csv")
    if mode == "sub":
        feat = np.load(CACHE / "e253_subwindow/train_feat.npy").astype(np.float32).mean(1)
    else:
        feat = np.load(CACHE / "e201_final_selftest/train_raw_scores_inputs.npy").astype(np.float32)
    assert feat.shape == (len(meta), 855), feat.shape
    return meta, torch.from_numpy(feat).to(DEV)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "sub"
    OUT.mkdir(parents=True, exist_ok=True)
    meta, feat = load_feature(mode)
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
        assert not bool((rows_f)[evk].any()) and not bool(rows_e[cal].any())

        mu, sd = fit_stats(feat, tasks, rows_e)
        z = zscore(feat, tasks, mu, sd)
        pm, basis = fit_pca(z, rows_e, max(DIMS))
        ids = [s for s in range(1, N_ID + 1) if s != held]

        for dim in DIMS:
            q = (z - pm) @ basis[:, :dim]
            variants = {"euclid": q, "cosine": q}
            for shrink in SHRINKS:
                variants[f"ldacos{shrink}"] = q @ fit_whiten(q, rows_e, subj, ids, shrink)
            for metric, qq in variants.items():
                base = "ldacos" if metric.startswith("ldacos") else metric
                for k, proto_name in ((1, "k1"), (2, "k2")):
                    pe, ve = build_protos(qq, rows_e, subj, tasks, order, ids, k)
                    pf, vf = build_protos(qq, rows_f, subj, tasks, order, ids, k)
                    c1, c2 = score_matrix(qq, tasks, pe, ve, base, cal)
                    k1, k2 = score_matrix(qq, tasks, pf, vf, base, evk)
                    u1, u2 = score_matrix(qq, tasks, pf, vf, base, evu)
                    for beta in BETAS:
                        cc = (c1 + beta * (c1 - c2)).cpu().numpy()
                        kk = (k1 + beta * (k1 - k2)).cpu().numpy()
                        uu = (u1 + beta * (u1 - u2)).cpu().numpy()
                        key = f"dim{dim}_{metric}_{proto_name}_b{beta}"
                        folds = results.setdefault(key, [])
                        entry = {"held": held, "auroc": auroc(kk, uu)}
                        for a in (0.90, 0.95):
                            tau_o = float(np.quantile(kk, 1.0 - a))
                            tau_c = float(np.quantile(cc, 1.0 - a))
                            entry[f"rej{a:.2f}_oracle"] = float((uu < tau_o).mean())
                            entry[f"rej{a:.2f}_deploy"] = float((uu < tau_c).mean())
                            entry[f"acc{a:.2f}_deploy"] = float((kk >= tau_c).mean())
                        folds.append(entry)
        print(f"{mode} held {held:2d} " + " ".join(
            f"{k}:{v[-1]['auroc']:.3f}/{v[-1]['rej0.90_oracle']:.3f}"
            for k, v in list(results.items())[-4:]), flush=True)

    summary = {}
    for key, folds in results.items():
        au = np.array([f["auroc"] for f in folds])
        r90 = np.array([f["rej0.90_oracle"] for f in folds])
        r95 = np.array([f["rej0.95_oracle"] for f in folds])
        d90 = np.array([f["rej0.90_deploy"] for f in folds])
        a90 = np.array([f["acc0.90_deploy"] for f in folds])
        summary[key] = {
            "auroc_mean": float(au.mean()), "auroc_min": float(au.min()),
            "rej90_mean": float(r90.mean()), "rej90_min": float(r90.min()),
            "u95_mean": float(r95.mean()), "u95_min": float(r95.min()),
            "rej90_deploy_mean": float(d90.mean()),
            "acc90_deploy_mean": float(a90.mean()),
            "worst5_rej90": float(np.sort(r90)[:5].mean()),
        }
    (OUT / f"loso_{mode}.json").write_text(json.dumps(
        {"protocol": "E254b progressive prototypes, Train-only", "grid": summary}, indent=2),
        encoding="utf-8")
    ranked = sorted(summary.items(), key=lambda kv: (
        0.5 * kv[1]["auroc_mean"] + 0.5 * kv[1]["u95_mean"]), reverse=True)
    for key, v in ranked[:10]:
        print(f"{key:28s} auc {v['auroc_mean']:.4f} u95 {v['u95_mean']:.4f} "
              f"rej90 {v['rej90_mean']:.4f} min {v['rej90_min']:.4f} "
              f"w5 {v['worst5_rej90']:.4f} | deploy acc {v['acc90_deploy_mean']:.3f} "
              f"rej {v['rej90_deploy_mean']:.4f}")


if __name__ == "__main__":
    main()
