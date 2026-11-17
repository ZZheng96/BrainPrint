"""E253g: is the cross-session failure a feature-norm collapse?

Checks the per-subject relationship between the projection norm and accuracy, then
tests ranking variants that restore the norm.  Rejection stays on the frozen
Euclidean geometry with a Train-calibrated tau.
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
from sweep import load_artifacts, project  # noqa: E402
from ranking import fit_whiten  # noqa: E402

N_BASE = 20


def normed(x, target=None):
    n = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return x / n if target is None else x * (target / n)


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

    p_tr = project(art, tr_feat, tr_tasks).mean(1)
    p_te = project(art, te_feat, te_tasks).mean(1)

    def protos_of(x, mask):
        return torch.stack([torch.stack([
            x[mask & (tr_tasks == t) & (tr_subj == s)].mean(0)
            for s in range(1, N_BASE + 1)]) for t in range(4)])

    def dist(proj, tasks, protos):
        out = torch.full((len(proj), N_BASE), float("inf"))
        for t in range(4):
            r = np.nonzero(tasks == t)[0]
            if r.size:
                out[r] = torch.cdist(proj[r], protos[t])
        return out

    # ---- norm vs accuracy (descriptive)
    sc_te0 = dist(p_te, te_tasks, protos_of(p_tr, early))
    pred0 = sc_te0.argmin(1).numpy() + 1
    df = pd.DataFrame({"subject": te_label[is_reg], "correct": pred0[is_reg] == te_label[is_reg],
                       "norm": p_te[torch.from_numpy(is_reg)].norm(dim=-1).numpy()})
    prof = df.groupby("subject").agg(acc=("correct", "mean"), norm=("norm", "median"))
    print(prof.round(3).to_string())
    print("corr(acc, norm) =", round(float(prof.acc.corr(prof.norm)), 4))

    tau = float(np.quantile(dist(p_tr[late], tr_tasks[late], protos_of(p_tr, early))[
        np.arange(int(late.sum())), tr_subj[late] - 1].numpy(), 0.90))
    accept = dist(p_te, te_tasks, protos_of(p_tr, early)).min(1).values.numpy() <= tau

    W = fit_whiten(p_tr[early], tr_subj[early], 0.7)
    target = float(p_tr[early].norm(dim=-1).median())

    variants = {
        "euclid": (p_tr, p_te),
        "cosine": (normed(p_tr), normed(p_te)),
        "whiten": (p_tr @ W, p_te @ W),
        "whiten_cos": (normed(p_tr @ W), normed(p_te @ W)),
        "norm_restore": (normed(p_tr, target), normed(p_te, target)),
        "whiten_norm_restore": (normed(p_tr @ W), normed(p_te @ W)),
    }
    report = {"tau": tau, "corr_acc_norm": float(prof.acc.corr(prof.norm))}
    for name, (a, b) in variants.items():
        prot = protos_of(a, early)
        tr_acc = float((dist(a[late], tr_tasks[late], prot).argmin(1).numpy() + 1
                        == tr_subj[late]).mean())
        pred = dist(b, te_tasks, prot).argmin(1).numpy() + 1
        e = {"train_forward_acc": tr_acc,
             "test_top1": float((pred == te_label)[is_reg].mean()),
             "test_acc_given_accept": float((pred == te_label)[is_reg & accept].mean()),
             "test_e2e": float(((pred == te_label) & accept)[is_reg].mean())}
        e["test_J"] = (3 * e["test_e2e"] + float((~accept)[~is_reg].mean())) / 4
        report[name] = e
        print(f"{name:20s} train_fwd {tr_acc:.3f} top1 {e['test_top1']:.3f} "
              f"e2e {e['test_e2e']:.3f} J {e['test_J']:.3f}", flush=True)
    (OUT / "norm.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
