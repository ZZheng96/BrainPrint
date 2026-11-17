"""E253c: representation sweep on the frozen 855-d logSPD features.

Everything is fitted on Train only (rows with frac <= 0.6, exactly like E200).
Swept: PCA dimension and shared-vs-per-task PCA.  Decision rule is fixed to the
frozen one (mean of 6 sub-window projections, Euclidean, 90% genuine OOF tau).

Reported:
  Train 5-block OOF closed-set accuracy (prototypes refitted per block)
  Train pseudo-unknown rejection at the calibrated tau (17 known / 3 unknown)
  Test registered end-to-end accuracy and unknown rejection (single audit)
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
import model as M  # noqa: E402

N_BASE = 20
DIMS = (64, 96, 128, 192, 256)
SCOPES = ("shared", "task")
SEED = 20261118


def load_artifacts():
    w = torch.load(PKG / "weights.pt", map_location="cpu", weights_only=True)
    return {"task_mu": w["task_mu"].float(), "task_sd": w["task_sd"].float()}


def zscore(art, feat, tasks):
    t = torch.from_numpy(np.ascontiguousarray(feat)).float()
    return (t - art["task_mu"][tasks]) / art["task_sd"][tasks] \
        if feat.ndim == 2 else (t - art["task_mu"][tasks][:, None, :]) / art["task_sd"][tasks][:, None, :]


def fit_pca(x: torch.Tensor, dim: int):
    mu = x.mean(0)
    xc = x - mu
    _, _, vt = torch.linalg.svd(xc, full_matrices=False)
    return mu, vt[:dim].T.contiguous()


def apply_pca(x, mu, basis):
    return (x - mu) @ basis


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    art = load_artifacts()
    train_feat = np.load(CACHE / "train_feat.npy")
    test_feat = np.load(CACHE / "test_feat.npy")
    tr = pd.read_csv(ROOT / "BrainPrint_cache/train_metadata.csv")
    te = pd.read_csv(ROOT / "BrainPrint_cache/test_metadata.csv")
    labels = pd.read_csv(ROOT / "Brainprint_Competition/test.csv")
    te = te.merge(labels, left_on="sample", right_on="Sample", how="left")
    tr_tasks = M._task_indices([{"task": t} for t in tr["task"]], len(tr))
    te_tasks = M._task_indices([{"task": t} for t in te["task"]], len(te))
    tr_subj = tr["subject"].to_numpy()
    g = tr.groupby(["subject", "task"])
    frac = ((g.cumcount() + 1) / g["trial"].transform("size")).to_numpy()
    te_label = te["Label"].to_numpy()
    early, late = frac <= 0.6, frac > 0.8
    block = np.clip((frac * 5).astype(int), 0, 4)
    is_reg = te_label > 0

    # mean over sub-windows, then per-task z-score
    tr_mean = torch.from_numpy(train_feat).float().mean(1).numpy()
    te_mean = torch.from_numpy(test_feat).float().mean(1).numpy()
    tr_z = zscore(art, tr_mean, tr_tasks)
    te_z = zscore(art, te_mean, te_tasks)

    rng = np.random.default_rng(SEED)
    splits = [rng.permutation(np.arange(1, N_BASE + 1))[:3] for _ in range(6)]
    report = {}
    for scope in SCOPES:
        for dim in DIMS:
            # ---- fit PCA on early rows
            if scope == "shared":
                mu, basis = fit_pca(tr_z[early], dim)
                p_tr, p_te = apply_pca(tr_z, mu, basis), apply_pca(te_z, mu, basis)
                protos = torch.stack([torch.stack([
                    p_tr[early & (tr_tasks == t) & (tr_subj == s)].mean(0)
                    for s in range(1, N_BASE + 1)]) for t in range(4)])
            else:
                p_tr = torch.zeros(len(tr_z), dim)
                p_te = torch.zeros(len(te_z), dim)
                protos = torch.zeros(4, N_BASE, dim)
                for t in range(4):
                    r = early & (tr_tasks == t)
                    mu_t, b_t = fit_pca(tr_z[r], dim)
                    p_tr[torch.from_numpy(tr_tasks == t)] = apply_pca(
                        tr_z[tr_tasks == t], mu_t, b_t)
                    p_te[torch.from_numpy(te_tasks == t)] = apply_pca(
                        te_z[te_tasks == t], mu_t, b_t)
                    for s in range(1, N_BASE + 1):
                        rr = early & (tr_tasks == t) & (tr_subj == s)
                        protos[t, s - 1] = p_tr[rr].mean(0)

            def scores(proj, tasks, proto):
                out = torch.full((len(proj), N_BASE), float("inf"))
                for t in range(4):
                    r = np.nonzero(tasks == t)[0]
                    if r.size:
                        out[r] = torch.cdist(proj[r], proto[t])
                return out

            # ---- Train: OOF closed-set accuracy + pseudo-unknown
            correct, total, dg_all, du_all = 0, 0, [], []
            for b in range(5):
                fit, held = block != b, block == b
                proto_b = protos.clone()
                for t in range(4):
                    for s in range(1, N_BASE + 1):
                        r = fit & (tr_tasks == t) & (tr_subj == s)
                        if r.sum() >= 3:
                            proto_b[t, s - 1] = p_tr[r].mean(0)
                sc = scores(p_tr[held], tr_tasks[held], proto_b)
                truth = tr_subj[held]
                correct += int((sc.argmin(1).numpy() + 1 == truth).sum())
                total += len(truth)
                dg_all.append(sc[np.arange(len(truth)), truth - 1].numpy())
            dg = np.concatenate(dg_all)
            tau = float(np.quantile(dg, 0.90))
            for unknown_ids in splits:
                known = np.setdiff1d(np.arange(1, N_BASE + 1), unknown_ids)
                r = np.isin(tr_subj, unknown_ids)
                sc = scores(p_tr[r], tr_tasks[r], protos)
                su = sc.masked_fill(torch.from_numpy(~np.isin(np.arange(1, N_BASE + 1), known)), float("inf"))
                du_all.append(su.min(1).values.numpy())
            du = np.concatenate(du_all)

            # ---- Test audit
            sc_te = scores(p_te, te_tasks, protos)
            pred = sc_te.argmin(1).numpy() + 1
            am = sc_te.min(1).values.numpy() <= tau
            entry = {
                "scope": scope, "dim": dim, "tau": tau,
                "train_oof_closed_acc": correct / total,
                "train_registered_accept": float((dg <= tau).mean()),
                "train_unknown_rejection": float((du > tau).mean()),
                "test_registered_top1": float((pred == te_label)[is_reg].mean()),
                "test_registered_accept": float(am[is_reg].mean()),
                "test_registered_e2e": float(((pred == te_label) & am)[is_reg].mean()),
                "test_unknown_rejection": float((~am)[~is_reg].mean())}
            entry["test_J"] = (3 * entry["test_registered_e2e"]
                               + entry["test_unknown_rejection"]) / 4
            entry["train_J"] = (3 * entry["train_oof_closed_acc"] * entry["train_registered_accept"]
                                + entry["train_unknown_rejection"]) / 4
            report[f"{scope}_{dim}"] = entry
            print(f"{scope}_{dim}", json.dumps(entry, default=float), flush=True)
    (OUT / "representation.json").write_text(json.dumps(report, indent=2, default=float),
                                             encoding="utf-8")


if __name__ == "__main__":
    main()
