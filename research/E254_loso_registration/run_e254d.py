"""E254d: extend the LDA-cos grid (dims / shrinkage / prototypes) and test
sub-window score aggregation for the strongest settings.

Everything remains Train-only: fit rows frac<=0.4, calibration 0.4-0.6,
evaluation known frac>0.6, unknown = all rows of the held identity.
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

DIMS = (160, 192, 256)
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(CACHE / "train_metadata.csv")
    feat_sw = np.load(CACHE / "e253_subwindow/train_feat.npy").astype(np.float32)
    feat = torch.from_numpy(feat_sw.mean(1)).to(DEV)
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
        sw_t = torch.from_numpy(feat_sw).to(DEV)  # (N, 6, 855)
        z_sw = torch.empty_like(sw_t)
        for t in range(4):
            m = tasks == t
            z_sw[m] = (sw_t[m] - mu[t]) / sd[t]
        z = z_sw.mean(1)
        x_fit = z[rows_e]
        pm = x_fit.mean(0)
        xc = x_fit - pm
        cov = xc.T @ xc / max(len(x_fit) - 1, 1)
        vals, vecs = torch.linalg.eigh(cov)
        basis = vecs[:, torch.argsort(vals, descending=True)[:max(DIMS)]].contiguous()
        ids = [s for s in range(1, N_ID + 1) if s != held]

        for dim in DIMS:
            q0 = (z - pm) @ basis[:, :dim]
            q0_sw = (z_sw - pm) @ basis[:, :dim]  # (N, 6, dim)
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
                q_sw = q0_sw @ W
                for k in KS:
                    pe, ve = build_protos(q, rows_e, subj, tasks, order, ids, k)
                    pf, vf = build_protos(q, rows_f, subj, tasks, order, ids, k)
                    Sk = score_full(q, tasks, pf, vf, evk)
                    Su = score_full(q, tasks, pf, vf, evu)
                    Sk_sw = torch.stack(
                        [score_full(q_sw[:, w], tasks, pf, vf, evk) for w in range(6)]).mean(0)
                    Su_sw = torch.stack(
                        [score_full(q_sw[:, w], tasks, pf, vf, evu) for w in range(6)]).mean(0)
                    for agg, (Sk_, Su_) in (("mean", (Sk, Su)), ("sw", (Sk_sw, Su_sw))):
                        tk = Sk_.topk(2, dim=1).values
                        tu = Su_.topk(2, dim=1).values
                        proto_ok = True
                        for beta in BETAS:
                            kk = (tk[:, 0] + beta * (tk[:, 0] - tk[:, 1])).cpu().numpy()
                            uu = (tu[:, 0] + beta * (tu[:, 0] - tu[:, 1])).cpu().numpy()
                            key = f"d{dim}_s{shrink}_k{k}_{agg}_b{beta}"
                            folds = results.setdefault(key, [])
                            e = {"held": held, "auroc": auroc(kk, uu)}
                            for a in (0.90, 0.95):
                                tau = float(np.quantile(kk, 1.0 - a))
                                e[f"rej{a:.2f}"] = float((uu < tau).mean())
                            folds.append(e)
                        del proto_ok
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
    (OUT / "loso_ext.json").write_text(json.dumps({"grid": summary}, indent=2), encoding="utf-8")
    ranked = sorted(summary.items(), key=lambda kv: (
        kv[1]["auroc"] + kv[1]["u95"] + kv[1]["rej90"]), reverse=True)
    for key, v in ranked[:12]:
        print(f"{key:30s} auc {v['auroc']:.4f} u95 {v['u95']:.4f} rej90 {v['rej90']:.4f} "
              f"min95 {v['u95_min']:.3f} min90 {v['rej90_min']:.3f}")


if __name__ == "__main__":
    main()
