"""E254: strict identity-LOSO with match-score ranking and an open-set gate.

Protocol (Train only, never reads Test):
  outer fold h  : identity h is pseudo-unknown; the other 19 identities are known
  fit rows      : known identities, trial fraction <= 0.4
  calibration   : known identities, 0.4 < frac <= 0.6  (genuine score quantiles)
  eval (known)  : known identities, frac > 0.6
  eval (unknown): every trial of h

Everything (task mean/std, PCA, WCCN whitening, prototypes, thresholds) is fitted
on the fit rows of the fold only, so the held identity never enters any estimate.

Grid (pre-registered, finite):
  dim    : 32 48 64 96
  metric : euclid | cosine | ldacos
  proto  : k1 mean | k2 trial-order halves
  gate   : s = s1 + beta * (s1 - s2), beta in {0.0, 0.5}

Output: results/loso_grid.json with per-fold and aggregate metrics.
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
OUT = Path(__file__).resolve().parent / "results"
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DIMS = (32, 48, 64, 96)
METRICS = ("euclid", "cosine", "ldacos")
PROTOS = ("k1", "k2")
BETAS = (0.0, 0.5)
TARGETS = (0.90, 0.95)
N_ID = 20
SEED = 20261118


def auroc(known: np.ndarray, unknown: np.ndarray) -> float:
    if not len(known) or not len(unknown):
        return float("nan")
    x = np.concatenate([known, unknown])
    ranks = pd.Series(x).rank(method="average").to_numpy()
    return float((ranks[:len(known)].sum() - len(known) * (len(known) + 1) / 2)
                 / (len(known) * len(unknown)))


def fit_stats(feat: torch.Tensor, tasks: torch.Tensor, rows: torch.Tensor):
    mu = torch.stack([feat[rows & (tasks == t)].mean(0) for t in range(4)])
    sd = torch.stack([feat[rows & (tasks == t)].std(0).clamp_min(1e-6) for t in range(4)])
    return mu, sd


def zscore(feat: torch.Tensor, tasks: torch.Tensor, mu, sd):
    z = torch.empty_like(feat)
    for t in range(4):
        m = tasks == t
        z[m] = (feat[m] - mu[t]) / sd[t]
    return z


def fit_pca(z: torch.Tensor, rows: torch.Tensor, max_dim: int):
    x = z[rows]
    mu = x.mean(0)
    xc = x - mu
    cov = xc.T @ xc / max(len(x) - 1, 1)
    vals, vecs = torch.linalg.eigh(cov)
    order = torch.argsort(vals, descending=True)[:max_dim]
    return mu, vecs[:, order].contiguous()


def fit_whiten(q: torch.Tensor, rows: torch.Tensor, subj: torch.Tensor, ids):
    resid = []
    for s in ids:
        r = rows & (subj == s)
        resid.append(q[r] - q[r].mean(0))
    r = torch.cat(resid)
    cov = r.T @ r / max(len(r) - len(ids), 1)
    cov = 0.5 * cov + 0.5 * torch.diagonal(cov).mean() * torch.eye(
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


def score_matrix(q, tasks, protos, valid, metric, rows_mask=None):
    """Return (best_score, dmin, s1, s2) for the selected rows."""
    idx = torch.nonzero(rows_mask).flatten() if rows_mask is not None else \
        torch.arange(len(q), device=q.device)
    n = len(idx)
    best = torch.full((n, N_ID), -float("inf"), device=q.device)
    dmin = torch.full((n, N_ID), float("inf"), device=q.device)
    for t in range(4):
        m = tasks[idx] == t
        if not bool(m.any()):
            continue
        sub = idx[m]
        x = q[sub]
        p = protos[t]  # (N_ID, k, dim)
        if metric == "euclid":
            d = torch.cdist(x, p.reshape(-1, p.shape[-1])).reshape(len(sub), N_ID, p.shape[1])
            dm = d.min(2).values
            dm = dm.masked_fill(~valid[None], float("inf"))
            best[m] = -dm
            dmin[m] = dm
        else:
            xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            pn = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            s = (xn[:, None, None, :] * pn[None]).sum(-1).max(2).values
            s = s.masked_fill(~valid[None], -float("inf"))
            best[m] = s
            d = torch.cdist(x, p.reshape(-1, p.shape[-1])).reshape(len(sub), N_ID, p.shape[1])
            dmin[m] = d.min(2).values.masked_fill(~valid[None], float("inf"))
    top2 = best.topk(2, dim=1).values
    return top2[:, 0], top2[:, 1], dmin.min(1).values


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    meta = pd.read_csv(ROOT / "BrainPrint_cache/train_metadata.csv")
    feat = np.load(CACHE / "train_feat.npy")
    assert len(feat) == len(meta) and feat.shape[1] == 6
    assert meta.subject.nunique() == N_ID and meta.session.astype(str).unique().tolist() == ["session1"]
    sys.path.insert(0, str(ROOT / "final2/team_hymn"))
    import model as M  # noqa: E402

    tasks_np = M._task_indices([{"task": t} for t in meta["task"]], len(meta))
    subj_np = meta["subject"].to_numpy()
    order_np = meta.groupby(["subject", "task"]).cumcount().to_numpy()
    g = meta.groupby(["subject", "task"])
    frac_np = ((g.cumcount() + 1) / g["trial"].transform("size")).to_numpy()

    feat = torch.from_numpy(feat.astype(np.float32)).to(DEV).mean(1)  # (N, 855)
    tasks = torch.as_tensor(tasks_np, device=DEV)
    subj = torch.as_tensor(subj_np, device=DEV)
    frac = torch.as_tensor(frac_np, device=DEV)
    order = torch.as_tensor(order_np, device=DEV)

    results = {}
    for held in range(1, N_ID + 1):
        keep = subj != held
        fit_rows = keep & (frac <= 0.4)
        cal_rows = keep & (frac > 0.4) & (frac <= 0.6)
        ev_known = keep & (frac > 0.6)
        ev_unknown = subj == held
        assert not bool((fit_rows | cal_rows | ev_known)[ev_unknown].any())

        mu, sd = fit_stats(feat, tasks, fit_rows)
        z = zscore(feat, tasks, mu, sd)
        pm, basis = fit_pca(z, fit_rows, max(DIMS))
        ids = [s for s in range(1, N_ID + 1) if s != held]

        for dim in DIMS:
            q = (z - pm) @ basis[:, :dim]
            q = q.reshape(len(q), -1) if q.dim() > 2 else q
            spaces = {"euclid": q, "cosine": q}
            W = fit_whiten(q, fit_rows, subj, ids)
            spaces["ldacos"] = q @ W
            for metric, qq in spaces.items():
                for k, proto_name in ((1, "k1"), (2, "k2")):
                    protos, valid = build_protos(qq, fit_rows, subj, tasks, order, ids, k)
                    s1_c, s2_c, _ = score_matrix(qq, tasks, protos, valid, metric, cal_rows)
                    s1_k, s2_k, _ = score_matrix(qq, tasks, protos, valid, metric, ev_known)
                    s1_u, s2_u, _ = score_matrix(qq, tasks, protos, valid, metric, ev_unknown)
                    for beta in BETAS:
                        c = s1_c + beta * (s1_c - s2_c)
                        kk = s1_k + beta * (s1_k - s2_k)
                        uu = s1_u + beta * (s1_u - s2_u)
                        ca = c.cpu().numpy()
                        kv = kk.cpu().numpy()
                        uv = uu.cpu().numpy()
                        key = f"dim{dim}_{metric}_{proto_name}_b{beta}"
                        entry = results.setdefault(key, {"folds": []})
                        for target in TARGETS:
                            tau = float(np.quantile(ca, 1.0 - target))
                            entry["folds"].append({
                                "held": held, "target": target,
                                "auroc": auroc(kv, uv),
                                "known_accept": float((kv >= tau).mean()),
                                "unknown_rejection": float((uv < tau).mean()),
                                "tau": tau,
                            })
                        print(f"held {held:2d} dim{dim:2d} {metric:6s} {proto_name} b{beta} "
                              f"auroc {entry['folds'][-2]['auroc']:.4f} "
                              f"rej90 {entry['folds'][-2]['unknown_rejection']:.4f}",
                              flush=True)

    summary = {}
    for key, entry in results.items():
        per_target = {}
        for target in TARGETS:
            rows = [f for f in entry["folds"] if f["target"] == target]
            rej = np.array([f["unknown_rejection"] for f in rows])
            au = np.array([f["auroc"] for f in rows])
            per_target[f"a{int(target * 100)}"] = {
                "auroc_mean": float(au.mean()), "auroc_min": float(au.min()),
                "unknown_rej_mean": float(rej.mean()), "unknown_rej_min": float(rej.min()),
                "unknown_rej_std": float(rej.std()),
                "known_accept_mean": float(np.mean([f["known_accept"] for f in rows])),
            }
        summary[key] = per_target
    (OUT / "loso_grid.json").write_text(json.dumps(
        {"protocol": "E254 identity-LOSO, Train-only", "grid": summary}, indent=2),
        encoding="utf-8")

    ranked = sorted(summary.items(),
                    key=lambda kv: (kv[1]["a90"]["auroc_mean"] + kv[1]["a90"]["unknown_rej_mean"]),
                    reverse=True)
    for key, val in ranked[:12]:
        a = val["a95"]
        b = val["a90"]
        print(f"{key:34s} a95 auc {a['auroc_mean']:.4f} rej {a['unknown_rej_mean']:.4f} | "
              f"a90 rej {b['unknown_rej_mean']:.4f} min {b['unknown_rej_min']:.4f}")


if __name__ == "__main__":
    main()
