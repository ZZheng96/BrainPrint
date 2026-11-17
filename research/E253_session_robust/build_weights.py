"""E253k: build the frozen artifacts of the new submission into final2/team_hymn.

The only change with respect to E200 is the identity ranking rule:
  E200 : euclidean distance between the PCA-96 mean projection and the prototype
  E253 : cosine between the whitened projections, W from the Train within-class
         covariance (shrink 0.5).  Rejection still uses the E200 euclidean distance.

tau is re-calibrated with the same 5-block out-of-fold protocol (90% genuine
quantile) because the frozen E200 value used a slightly different OOF subset.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "final/team_hymn"
DST = ROOT / "final2/team_hymn"
CACHE = ROOT / "BrainPrint_cache/e253_subwindow"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(Path(__file__).parent))
import model as M  # noqa: E402
from sweep import project, load_artifacts  # noqa: E402
from ranking import fit_whiten  # noqa: E402

N_BASE = 20
SHRINK = 0.5


def main() -> None:
    art = load_artifacts()
    w = torch.load(PKG / "weights.pt", map_location="cpu", weights_only=True)
    tr_feat = np.load(CACHE / "train_feat.npy")
    tr = pd.read_csv(ROOT / "BrainPrint_cache/train_metadata.csv")
    tr_tasks = M._task_indices([{"task": t} for t in tr["task"]], len(tr))
    tr_subj = tr["subject"].to_numpy()
    g = tr.groupby(["subject", "task"])
    frac = ((g.cumcount() + 1) / g["trial"].transform("size")).to_numpy()
    early = frac <= 0.6
    block = np.clip((frac * 5).astype(int), 0, 4)

    p_tr = project(art, tr_feat, tr_tasks).mean(1)
    W = fit_whiten(p_tr[early], tr_subj[early], SHRINK).float()
    print("W", tuple(W.shape), "cond", float(torch.linalg.cond(W)))

    def dist(proj, tasks, protos):
        out = torch.full((len(proj), N_BASE), float("inf"))
        for t in range(4):
            r = np.nonzero(tasks == t)[0]
            if r.size:
                out[r] = torch.cdist(proj[r], protos[t])
        return out

    dg = []
    for b in range(5):
        fit, held = block != b, block == b
        protos = torch.stack([torch.stack([
            p_tr[fit & (tr_tasks == t) & (tr_subj == s)].mean(0)
            for s in range(1, N_BASE + 1)]) for t in range(4)])
        sc = dist(p_tr[held], tr_tasks[held], protos)
        truth = tr_subj[held]
        dg.append(sc[np.arange(len(truth)), truth - 1].numpy())
    tau = float(np.quantile(np.concatenate(dg), 0.90))
    print("tau", tau, "frozen old", art["tau"])

    w["whiten"] = W
    w["tau_dist"] = tau
    w["ranking"] = "pca96 -> within-class whitening (shrink 0.5) -> cosine"
    w["rejection"] = "euclidean distance in pca96 <= tau_dist"
    w["source"] = "E253_session_robust (E200 feature pipeline, whitened-cosine ranking)"
    w["format"] = "brainprint-final-v2"
    DST.mkdir(parents=True, exist_ok=True)
    torch.save(w, DST / "weights.pt")
    meta = {"tau": tau, "shrink": SHRINK, "whiten_shape": list(W.shape),
            "cond": float(torch.linalg.cond(W))}
    (ROOT / "research/E253_session_robust/results/frozen.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    for name in ("team.json", "requirements.txt", "README.md"):
        shutil.copy2(PKG / name, DST / name)
    print("written", DST / "weights.pt", (DST / "weights.pt").stat().st_size)


if __name__ == "__main__":
    main()
