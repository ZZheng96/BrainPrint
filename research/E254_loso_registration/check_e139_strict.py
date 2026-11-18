"""E254l: re-check E139's k-means multi-prototype euclid under the strict
nested identity-LOSO protocol (per-fold refit, held identity excluded).

E139 reported LOSO rejection 0.9963 with a globally fitted PCA, so its number is
not comparable to E254.  This script keeps E139's decision rule (PCA-96,
k-means k=12 prototypes per identity, nearest-prototype euclidean distance,
90% Train acceptance) but refits everything inside each outer fold.
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

DIM, K = 96, 12
N_ID = 20


def auroc(known, unknown):
    x = np.concatenate([known, unknown])
    ranks = pd.Series(x).rank(method="average").to_numpy()
    return float((ranks[:len(known)].sum() - len(known) * (len(known) + 1) / 2)
                 / (len(known) * len(unknown)))


def kmeans(x: torch.Tensor, k: int, iters: int = 20, seed: int = 0):
    n = len(x)
    if n <= k:
        return x.clone()
    gen = torch.Generator(device=x.device).manual_seed(seed)
    idx = torch.randperm(n, generator=gen, device=x.device)[:k]
    c = x[idx].clone()
    for _ in range(iters):
        d = torch.cdist(x, c)
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            if bool(m.any()):
                c[j] = x[m].mean(0)
    return c


def main() -> None:
    sys.path.insert(0, str(ROOT / "final2/team_hymn"))
    import model as M  # noqa: E402

    meta = pd.read_csv(CACHE / "train_metadata.csv")
    feat = torch.from_numpy(
        np.load(CACHE / "e253_subwindow/train_feat.npy").astype(np.float32).mean(1)).to(DEV)
    tasks = torch.as_tensor(M._task_indices([{"task": t} for t in meta["task"]], len(meta)), device=DEV)
    subj = torch.as_tensor(meta["subject"].to_numpy(), device=DEV)
    order = torch.as_tensor(meta.groupby(["subject", "task"]).cumcount().to_numpy(), device=DEV)
    g = meta.groupby(["subject", "task"])
    frac = torch.as_tensor(((g.cumcount() + 1) / g["trial"].transform("size")).to_numpy(), device=DEV)

    known_all, unknown_all = [], []
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
        basis = vecs[:, torch.argsort(vals, descending=True)[:DIM]].contiguous()
        q = (z - pm) @ basis

        protos = {}
        for s in ids:
            for t in range(4):
                idx = torch.nonzero(rows_f & (subj == s) & (tasks == t)).flatten()
                idx = idx[torch.argsort(order[idx])]
                protos[(s, t)] = kmeans(q[idx], K)

        def score(mask, gen_seed):
            idx = torch.nonzero(mask).flatten()
            out = torch.full((len(idx), N_ID), float("inf"), device=DEV)
            for t in range(4):
                m = tasks[idx] == t
                if not bool(m.any()):
                    continue
                x = q[idx[m]]
                for s in ids:
                    out[m, s - 1] = torch.cdist(x, protos[(s, t)]).min(1).values
            return -out

        # calibration prototypes: same k-means but fitted on frac<=0.4 only
        protos_e = {}
        for s in ids:
            for t in range(4):
                idx = torch.nonzero(rows_e & (subj == s) & (tasks == t)).flatten()
                idx = idx[torch.argsort(order[idx])]
                protos_e[(s, t)] = kmeans(q[idx], K)
        protos, protos_save = protos_e, protos
        ck = score(cal, 0)
        protos = protos_save
        kk = score(evk, 1)
        uu = score(evu, 2)
        tau = float(np.quantile(ck.max(1).values.cpu().numpy(), 0.10))
        known_all.append(kk.max(1).values.cpu().numpy())
        unknown_all.append(uu.max(1).values.cpu().numpy())
        print(f"held {held:2d} tau {tau:.3f}", flush=True)

    kv, uv = np.concatenate(known_all), np.concatenate(unknown_all)
    report = {
        "protocol": "E254l strict nested identity-LOSO, E139 rule (PCA96, kmeans k=12, euclid)",
        "auroc": auroc(kv, uv),
        "u95": float((uv < np.quantile(kv, 0.05)).mean()),
        "rej90": float((uv < np.quantile(kv, 0.10)).mean()),
        "per_fold_rej90_min": float(np.min([
            (uv < q).mean() for q in [np.quantile(kv, 0.10)]])) if len(uv) else None,
    }
    folds = []
    for k_, u_ in zip(known_all, unknown_all):
        q90 = float(np.quantile(k_, 0.10))
        folds.append(float((u_ < q90).mean()))
    report["per_fold_rej90_mean"] = float(np.mean(folds))
    report["per_fold_rej90_min"] = float(np.min(folds))
    (OUT / "e139_strict_recheck.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
