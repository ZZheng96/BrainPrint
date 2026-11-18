"""E254m: feature-component sensitivity under the strict identity-LOSO protocol.

Ablations (all cached, no re-extraction):
  bands      : all / no 30-40 / no 1-8 / only 1-20 / only 8-30 / only 20-40
  sub-window : mean / median / first three / last three

Fixed decision rule: PCA (<=256), within-class whitening shrink 0.95,
k=4 trial-order prototypes, gate s1 + 0.5*(s1-s2) at a=0.90.
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
N_ID = 20
SHRINK, K, BETA = 0.95, 4, 0.5

BANDS = {
    "all": [0, 1, 2, 3, 4],
    "no30_40": [0, 1, 2, 3],
    "no1_8": [1, 2, 3, 4],
    "low1_20": [0, 1, 2],
    "mid8_30": [1, 2, 3],
    "high20_40": [3, 4],
}
AGG = {"mean": "mean", "median": "median", "first3": "first3", "last3": "last3"}


def auroc(known, unknown):
    x = np.concatenate([known, unknown])
    ranks = pd.Series(x).rank(method="average").to_numpy()
    return float((ranks[:len(known)].sum() - len(known) * (len(known) + 1) / 2)
                 / (len(known) * len(unknown)))


def build_feature(sw: torch.Tensor, bands, agg: str):
    x = sw.reshape(len(sw), 6, 5, 171)[:, :, bands, :].reshape(len(sw), 6, -1)
    if agg == "mean":
        return x.mean(1)
    if agg == "median":
        return x.median(1).values
    if agg == "first3":
        return x[:, :3].mean(1)
    return x[:, 3:].mean(1)


def run_variant(feat, tasks, subj, order, frac, dim_cap):
    known_all, unknown_all, folds = [], [], []
    for held in range(1, N_ID + 1):
        keep = subj != held
        ids = [s for s in range(1, N_ID + 1) if s != held]
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
        dim = min(dim_cap, feat.shape[1], len(x_fit))
        basis = vecs[:, torch.argsort(vals, descending=True)[:dim]].contiguous()
        q0 = (z - pm) @ basis
        resid = [q0[rows_e & (subj == s)] - q0[rows_e & (subj == s)].mean(0) for s in ids]
        r = torch.cat(resid)
        covw = r.T @ r / max(len(r) - len(ids), 1)
        covw = SHRINK * covw + (1 - SHRINK) * torch.diagonal(covw).mean() * torch.eye(dim, device=DEV)
        wv, we = torch.linalg.eigh(covw)
        q = q0 @ ((we * wv.clamp_min(1e-6).rsqrt()) @ we.T)

        protos = torch.zeros(4, N_ID, K, dim, device=DEV)
        for s in ids:
            for t in range(4):
                idx = torch.nonzero(rows_f & (subj == s) & (tasks == t)).flatten()
                idx = idx[torch.argsort(order[idx])]
                for j in range(K):
                    protos[t, s - 1, j] = q[idx[j * len(idx) // K:(j + 1) * len(idx) // K]].mean(0)

        def gate_vals(mask):
            idx = torch.nonzero(mask).flatten()
            out = torch.full((len(idx), N_ID), -float("inf"), device=DEV)
            for t in range(4):
                m = tasks[idx] == t
                if not bool(m.any()):
                    continue
                x, p = q[idx[m]], protos[t]
                xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                pn = (p.reshape(-1, dim) / p.reshape(-1, dim).norm(dim=-1, keepdim=True).clamp_min(1e-8)).reshape(N_ID, K, -1)
                out[m] = (xn[:, None, None, :] * pn[None]).sum(-1).max(2).values
            top2 = out.topk(2, dim=1).values
            return (top2[:, 0] + BETA * (top2[:, 0] - top2[:, 1])).cpu().numpy()

        kk, uu = gate_vals(evk), gate_vals(evu)
        known_all.append(kk)
        unknown_all.append(uu)
        q90 = float(np.quantile(kk, 0.10))
        folds.append(float((uu < q90).mean()))
    kv, uv = np.concatenate(known_all), np.concatenate(unknown_all)
    return {"auroc": auroc(kv, uv),
            "u95": float((uv < np.quantile(kv, 0.05)).mean()),
            "rej90": float((uv < np.quantile(kv, 0.10)).mean()),
            "fold_rej90_min": float(np.min(folds)),
            "fold_rej90_mean": float(np.mean(folds))}


def main() -> None:
    sys.path.insert(0, str(ROOT / "final2/team_hymn"))
    import model as M  # noqa: E402

    meta = pd.read_csv(CACHE / "train_metadata.csv")
    sw = torch.from_numpy(
        np.load(CACHE / "e253_subwindow/train_feat.npy").astype(np.float32)).to(DEV)
    tasks = torch.as_tensor(M._task_indices([{"task": t} for t in meta["task"]], len(meta)), device=DEV)
    subj = torch.as_tensor(meta["subject"].to_numpy(), device=DEV)
    order = torch.as_tensor(meta.groupby(["subject", "task"]).cumcount().to_numpy(), device=DEV)
    g = meta.groupby(["subject", "task"])
    frac = torch.as_tensor(((g.cumcount() + 1) / g["trial"].transform("size")).to_numpy(), device=DEV)

    report = {}
    for name, bands in BANDS.items():
        feat = build_feature(sw, bands, "mean")
        report[f"band_{name}"] = run_variant(feat, tasks, subj, order, frac, 256)
        print(f"band_{name:10s} " + json.dumps(report[f"band_{name}"]), flush=True)
    for name, agg in AGG.items():
        feat = build_feature(sw, BANDS["all"], agg)
        report[f"agg_{name}"] = run_variant(feat, tasks, subj, order, frac, 256)
        print(f"agg_{name:10s} " + json.dumps(report[f"agg_{name}"]), flush=True)
    (OUT / "ablation_e254.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
