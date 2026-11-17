"""E253h: stability sweep for the whitened-cosine ranking.

ranking  : (x - pca_mean) @ B, then within-class whitening W(shrink), then cosine
rejection: frozen Euclidean dmin <= tau (Train-calibrated)
Swept    : PCA dimension (frozen 96 vs refitted 160/256) x whitening shrinkage
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


def normed(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


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

    def protos_of(x):
        return torch.stack([torch.stack([
            x[early & (tr_tasks == t) & (tr_subj == s)].mean(0)
            for s in range(1, N_BASE + 1)]) for t in range(4)])

    def dist(proj, tasks, protos, cos=False):
        out = torch.full((len(proj), N_BASE), float("inf"))
        pr = normed(protos) if cos else protos
        pj = normed(proj) if cos else proj
        for t in range(4):
            r = np.nonzero(tasks == t)[0]
            if r.size:
                out[r] = torch.cdist(pj[r], pr[t])
        return out

    tau = float(np.quantile(dist(p_tr[late], tr_tasks[late], protos_of(p_tr))[
        np.arange(int(late.sum())), tr_subj[late] - 1].numpy(), 0.90))
    dmin = dist(p_te, te_tasks, protos_of(p_tr)).min(1).values.numpy()
    accept = dmin <= tau
    rej = float((~accept)[~is_reg].mean())
    print(f"tau {tau:.3f} accept {accept[is_reg].mean():.3f} unknown_rej {rej:.3f}", flush=True)

    # refitted PCA bases
    z_tr = torch.from_numpy(np.load(CACHE / "train_feat.npy")).float().mean(1).numpy()
    z_te = torch.from_numpy(np.load(CACHE / "test_feat.npy")).float().mean(1).numpy()
    from sweep import project as _  # noqa: F401

    report = {"tau": tau, "unknown_rejection": rej, "rows": {}}
    for dim in (96, 160, 256):
        if dim == 96:
            a, b = p_tr, p_te
        else:
            x = project(art, tr_feat, tr_tasks).mean(1)
            mu = x[early].mean(0)
            _, _, vt = torch.linalg.svd(x[early] - mu, full_matrices=False)
            basis = vt[:dim].T.contiguous()
            a = (x - mu) @ basis
            b = (project(art, te_feat, te_tasks).mean(1) - mu) @ basis
        for shrink in (0.3, 0.5, 0.7, 0.9):
            W = fit_whiten(a[early], tr_subj[early], shrink)
            wa, wb = a @ W, b @ W
            prot = protos_of(wa)
            tr_acc = float((dist(wa[late], tr_tasks[late], prot, cos=True).argmin(1).numpy()
                            + 1 == tr_subj[late]).mean())
            pred = dist(wb, te_tasks, prot, cos=True).argmin(1).numpy() + 1
            e = {"dim": dim, "shrink": shrink, "train_forward_acc": tr_acc,
                 "test_top1": float((pred == te_label)[is_reg].mean()),
                 "test_e2e": float(((pred == te_label) & accept)[is_reg].mean())}
            e["test_J"] = (3 * e["test_e2e"] + rej) / 4
            report["rows"][f"dim{dim}_shrink{shrink}"] = e
            print(f"dim{dim:3d} shrink{shrink} train_fwd {tr_acc:.3f} top1 "
                  f"{e['test_top1']:.3f} e2e {e['test_e2e']:.3f} J {e['test_J']:.3f}",
                  flush=True)
    (OUT / "refine.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
