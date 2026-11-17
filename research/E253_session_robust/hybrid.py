"""E253e: decouple ranking metric from rejection metric.

Diagnosis on the registered Test split shows the two jobs want different metrics:
  cosine  ranks identities better   (top-1 0.528 vs 0.491 for euclid)
  euclid  separates unknown users better (Test unknown rejection 0.71 vs 0.44)

So: predict with one metric, accept/reject with another.  Both thresholds are
calibrated on Train only (90% genuine OOF quantile), and the same 9 combinations
are also scored on the Train 17-known / 3-unknown proxy for consistency.
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
sys.path.insert(0, str(Path(__file__).parent))
import model as M  # noqa: E402
from sweep import load_artifacts, project, score, whiten  # noqa: E402

N_BASE = 20
METRICS = ("euclid", "cosine", "maha")
SEED = 20261118


def main() -> None:
    art = load_artifacts()
    tr_feat = np.load(CACHE / "train_feat.npy")
    te_feat = np.load(CACHE / "test_feat.npy")
    tr = pd.read_csv(ROOT / "BrainPrint_cache/train_metadata.csv")
    te = pd.read_csv(ROOT / "BrainPrint_cache/test_metadata.csv")
    lab = pd.read_csv(ROOT / "Brainprint_Competition/test.csv")
    te = te.merge(lab, left_on="sample", right_on="Sample", how="left")
    tr_tasks = M._task_indices([{"task": t} for t in tr["task"]], len(tr))
    te_tasks = M._task_indices([{"task": t} for t in te["task"]], len(te))
    tr_subj = tr["subject"].to_numpy()
    g = tr.groupby(["subject", "task"])
    frac = ((g.cumcount() + 1) / g["trial"].transform("size")).to_numpy()
    early, late = frac <= 0.6, frac > 0.8
    te_label = te["Label"].to_numpy()
    is_reg = te_label > 0

    p_tr = project(art, tr_feat, tr_tasks)
    p_te = project(art, te_feat, te_tasks)
    W = whiten(art, early, tr_tasks, tr_subj, p_tr)
    prep = {"euclid": lambda x: x, "cosine": lambda x: x, "maha": lambda x: x @ W}

    def matrix(proj, tasks, metric):
        out = torch.full((len(proj), N_BASE), float("inf"))
        for t in range(4):
            r = np.nonzero(tasks == t)[0]
            if r.size:
                out[r] = score(proj[r], protos_by_t(metric)[t], metric, "mean")
        return out

    def protos_by_t(metric, _cache={}):
        if metric in _cache:
            return _cache[metric]
        p = prep[metric](p_tr)
        out = torch.stack([torch.stack([
            p[early & (tr_tasks == t) & (tr_subj == s)].mean(1).mean(0)
            for s in range(1, N_BASE + 1)]) for t in range(4)])
        _cache[metric] = out
        return out

    # Train thresholds per metric (90% quantile of genuine OOF nearest distance)
    rng = np.random.default_rng(SEED)
    tau = {}
    for metric in METRICS:
        p = prep[metric](p_tr)
        protos = protos_by_t(metric)
        dg = []
        for unknown in [rng.permutation(np.arange(1, N_BASE + 1))[:3] for _ in range(6)]:
            known = np.setdiff1d(np.arange(1, N_BASE + 1), unknown)
            keep = torch.from_numpy(np.isin(np.arange(1, N_BASE + 1), known))
            gen = late & np.isin(tr_subj, known)
            sc = torch.full((int(gen.sum()), N_BASE), float("inf"))
            for t in range(4):
                r = np.nonzero(tr_tasks[gen] == t)[0]
                if r.size:
                    sc[r] = score(p[gen][r], protos[t], metric, "mean")
            sc = sc.masked_fill(~keep, float("inf"))
            dg.append(sc[np.arange(int(gen.sum())), tr_subj[gen] - 1].numpy())
        tau[metric] = float(np.quantile(np.concatenate(dg), 0.90))
    print("tau", {k: round(v, 3) for k, v in tau.items()}, flush=True)

    # Test score matrices per metric
    m_te = {m: matrix(p_te, te_tasks, m) for m in METRICS}
    report = {"tau": tau, "combos": {}}
    for rank_m in METRICS:
        pred = m_te[rank_m].argmin(1).numpy() + 1
        for rej_m in METRICS:
            accept = m_te[rej_m].min(1).values.numpy() <= tau[rej_m]
            e = {"rank_metric": rank_m, "reject_metric": rej_m,
                 "top1": float((pred == te_label)[is_reg].mean()),
                 "accept": float(accept[is_reg].mean()),
                 "e2e": float(((pred == te_label) & accept)[is_reg].mean()),
                 "unknown_rejection": float((~accept)[~is_reg].mean())}
            e["J"] = (3 * e["e2e"] + e["unknown_rejection"]) / 4
            report["combos"][f"{rank_m}|{rej_m}"] = e
            print(f"{rank_m:7s} rank | {rej_m:7s} reject -> top1 {e['top1']:.3f} "
                  f"acc {e['accept']:.3f} e2e {e['e2e']:.3f} rej {e['unknown_rejection']:.3f} "
                  f"J {e['J']:.3f}", flush=True)
    (OUT / "hybrid.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
