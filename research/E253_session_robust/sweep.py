"""E253b: decision-rule and operating-point sweep, fitted on Train only.

Corrected semantics of the provided Test split:
  session1 (3033 rows) -> Label 0  == unregistered  (the unknown-rejection scene)
  session2 (19218 rows) -> Label 1..20 == registered users (the accuracy scenes)

Frozen pipeline (855-d logSPD -> per-task z -> PCA-96 -> mean-projection prototypes).
Only the decision rule and the operating point are swept:

  metric : euclid | cosine | maha   (maha = shrinkage within-class whitening)
  agg    : mean   -> average the 6 sub-window projections, then compare (baseline)
           median -> compare per sub-window, take the median distance

Train proxy for the final: 6 random 17-known / 3-unknown identity splits, genuine
rows = late trials (frac > 0.8) of the 17 known identities, unknown rows = every
trial of the 3 held-out identities.  Objective J = (3 * registered_e2e + rejection) / 4
mirrors the equal weighting of the four final scenes.
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
COMBOS = [("euclid", "mean"), ("cosine", "mean"), ("maha", "mean"),
          ("euclid", "median"), ("cosine", "median"), ("maha", "median")]
LEVELS = (0.60, 0.70, 0.80, 0.85, 0.90, 0.95)
SEED = 20261118


def load_artifacts():
    w = torch.load(PKG / "weights.pt", map_location="cpu", weights_only=True)
    return {"task_mu": w["task_mu"].float(), "task_sd": w["task_sd"].float(),
            "pca_mean": w["pca_mean"].float(), "pca_basis": w["pca_basis"].float(),
            "prototypes": w["prototypes"].float(), "tau": float(w["tau_dist"])}


def project(art, feat_sw, tasks, chunk=4096):
    feat_sw = np.asarray(feat_sw)
    out = []
    for i in range(0, len(feat_sw), chunk):
        t = torch.from_numpy(np.ascontiguousarray(feat_sw[i:i + chunk]))
        z = (t - art["task_mu"][tasks[i:i + chunk]][:, None, :]) \
            / art["task_sd"][tasks[i:i + chunk]][:, None, :]
        out.append((z - art["pca_mean"]) @ art["pca_basis"])
    return torch.cat(out)


def pair(x: torch.Tensor, proto: torch.Tensor, metric: str) -> torch.Tensor:
    """x (..., d), proto (k, d) -> (..., k) distance-like scores."""
    if metric == "cosine":
        xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        pn = proto / proto.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return -(xn @ pn.T)
    d2 = (x * x).sum(-1).unsqueeze(-1) + (proto * proto).sum(-1).view(*([1] * (x.dim() - 1)), -1) \
        - 2.0 * (x @ proto.T)
    return d2.clamp_min(0).sqrt()


def score(proj: torch.Tensor, proto: torch.Tensor, metric: str, agg: str) -> torch.Tensor:
    if agg == "mean":
        return pair(proj.mean(1), proto, metric)
    return pair(proj, proto, metric).median(1).values


def whiten(art, early, tr_tasks, tr_subj, p_tr):
    feats, labels = [], []
    for t in range(4):
        for s in range(1, N_BASE + 1):
            r = early & (tr_tasks == t) & (tr_subj == s)
            if r.sum() >= 3:
                feats.append(p_tr[r].mean(1))
                labels.append(np.full(int(r.sum()), s))
    x = torch.cat(feats)
    y = torch.cat([torch.from_numpy(l) for l in labels])
    mu = torch.stack([x[y == s].mean(0) for s in range(1, N_BASE + 1)])
    resid = torch.cat([x[y == s] - mu[s - 1] for s in range(1, N_BASE + 1)])
    cov = resid.T @ resid / max(len(resid) - N_BASE, 1)
    cov = 0.9 * cov + 0.1 * torch.diagonal(cov).mean() * torch.eye(cov.shape[0])
    vals, vecs = torch.linalg.eigh(cov)
    return (vecs * vals.clamp_min(1e-6).rsqrt()) @ vecs.T


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

    tr_proj = project(art, train_feat, tr_tasks)
    te_proj = project(art, test_feat, te_tasks)
    W = whiten(art, early, tr_tasks, tr_subj, tr_proj)

    def prep(proj, metric):
        return proj @ W if metric == "maha" else proj

    report = {"frozen_tau": art["tau"], "train": {}, "test": {}}
    rng = np.random.default_rng(SEED)
    splits = [rng.permutation(np.arange(1, N_BASE + 1))[:3] for _ in range(6)]

    for metric, agg in COMBOS:
        p_tr, p_te = prep(tr_proj, metric), prep(te_proj, metric)
        protos = torch.stack([torch.stack([
            p_tr[early & (tr_tasks == t) & (tr_subj == s)].mean(1).mean(0)
            for s in range(1, N_BASE + 1)]) for t in range(4)])

        dg_all, du_all, corr_all = [], [], []
        for unknown_ids in splits:
            known = np.setdiff1d(np.arange(1, N_BASE + 1), unknown_ids)
            proto = protos.clone()
            for t in range(4):
                for s in known:
                    r = early & (tr_tasks == t) & (tr_subj == s)
                    if r.sum() >= 3:
                        proto[t, s - 1] = p_tr[r].mean((0, 1))
            gen = late & np.isin(tr_subj, known)
            unk = np.isin(tr_subj, unknown_ids)
            sg = torch.full((int(gen.sum()), N_BASE), float("inf"))
            su = torch.full((int(unk.sum()), N_BASE), float("inf"))
            gt, ut = tr_tasks[gen], tr_tasks[unk]
            for t in range(4):
                r = np.nonzero(gt == t)[0]
                if r.size:
                    sg[r] = score(p_tr[gen][r], proto[t], metric, agg)
                r = np.nonzero(ut == t)[0]
                if r.size:
                    su[r] = score(p_tr[unk][r], proto[t], metric, agg)
            keep = np.zeros(N_BASE, bool)
            keep[known - 1] = True
            keep_t = torch.from_numpy(keep)
            sg = sg.masked_fill(~keep_t, float("inf"))
            su = su.masked_fill(~keep_t, float("inf"))
            dg_all.append(sg.min(1).values.numpy())
            du_all.append(su.min(1).values.numpy())
            corr_all.append(sg.argmin(1).numpy() + 1 == tr_subj[gen])
        dg, du, corr = (np.concatenate(dg_all), np.concatenate(du_all),
                        np.concatenate(corr_all))

        curve = []
        for a in LEVELS:
            tau = float(np.quantile(dg, a))
            am = dg <= tau
            e2e = float((corr & am).mean())
            rej = float((du > tau).mean())
            curve.append({"level": a, "tau": tau, "train_registered_e2e": e2e,
                          "train_unknown_rejection": rej, "J": (3 * e2e + rej) / 4})
        report["train"][f"{metric}_{agg}"] = curve
        print(f"{metric}_{agg}", [(c["level"], round(c["train_registered_e2e"], 3),
                                   round(c["train_unknown_rejection"], 3)) for c in curve],
              flush=True)

        te_score = torch.full((len(p_te), N_BASE), float("inf"))
        for t in range(4):
            r = np.nonzero(te_tasks == t)[0]
            if r.size:
                te_score[r] = score(p_te[r], protos[t], metric, agg)
        pred = te_score.argmin(1).numpy() + 1
        dmin = te_score.min(1).values.numpy()
        is_reg = te_label > 0
        audit = []
        for c in curve:
            am = dmin <= c["tau"]
            audit.append({
                "level": c["level"],
                "registered_e2e": float(((pred == te_label) & am)[is_reg].mean()),
                "registered_accept": float(am[is_reg].mean()),
                "registered_acc_given_accept": float((pred == te_label)[is_reg & am].mean())
                if (is_reg & am).any() else None,
                "registered_top1_no_reject": float((pred == te_label)[is_reg].mean()),
                "unknown_rejection": float((~am)[~is_reg].mean())})
        report["test"][f"{metric}_{agg}"] = audit
        print(f"{metric}_{agg} TEST", [(a["level"], round(a["registered_e2e"], 3),
                                        round(a["unknown_rejection"], 3)) for a in audit],
              flush=True)

    # exact frozen baseline
    base = torch.full((len(te_proj), N_BASE), float("inf"))
    for t in range(4):
        r = np.nonzero(te_tasks == t)[0]
        if r.size:
            base[r] = score(te_proj[r], art["prototypes"][t], "euclid", "mean")
    pred = base.argmin(1).numpy() + 1
    am = base.min(1).values.numpy() <= art["tau"]
    is_reg = te_label > 0
    report["frozen_baseline"] = {
        "registered_top1_no_reject": float((pred == te_label)[is_reg].mean()),
        "registered_accept": float(am[is_reg].mean()),
        "registered_e2e": float(((pred == te_label) & am)[is_reg].mean()),
        "registered_acc_given_accept": float((pred == te_label)[is_reg & am].mean()),
        "unknown_rejection": float((~am)[~is_reg].mean())}
    print("frozen", report["frozen_baseline"])
    (OUT / "sweep.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
