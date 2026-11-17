"""E253i: calibrate the operating point under the E200 protocol and lock the choice.

ranking  : PCA-96 -> within-class whitening (shrink 0.5) -> cosine
rejection: frozen Euclidean nearest-prototype distance <= tau

tau is chosen from 5-block out-of-fold genuine distances at several quantiles, and
the operating point is the one maximising the Train-only objective
J = (3 * registered end-to-end accuracy + unknown rejection) / 4, using the
17-known / 3-unknown identity proxy for the unregistered scene.
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
SHRINK = 0.5
LEVELS = (0.70, 0.80, 0.85, 0.90, 0.95)
SEED = 20261118


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
    block = np.clip((frac * 5).astype(int), 0, 4)
    te_label = te["Label"].to_numpy()
    is_reg = te_label > 0

    p_tr = project(art, tr_feat, tr_tasks).mean(1)
    p_te = project(art, te_feat, te_tasks).mean(1)
    W = fit_whiten(p_tr[early], tr_subj[early], SHRINK)
    w_tr, w_te = p_tr @ W, p_te @ W

    def protos(x, mask):
        return torch.stack([torch.stack([
            x[mask & (tr_tasks == t) & (tr_subj == s)].mean(0)
            for s in range(1, N_BASE + 1)]) for t in range(4)])

    def mat(proj, tasks, prot, cos):
        out = torch.full((len(proj), N_BASE), float("inf"))
        pj = normed(proj) if cos else proj
        pp = normed(prot) if cos else prot
        for t in range(4):
            r = np.nonzero(tasks == t)[0]
            if r.size:
                out[r] = torch.cdist(pj[r], pp[t])
        return out

    # ---- 5-block OOF genuine euclidean distances (E200 protocol)
    dg = []
    for b in range(5):
        fit, held = block != b, block == b
        prot_b = protos(p_tr, fit)
        sc = mat(p_tr[held], tr_tasks[held], prot_b, False)
        truth = tr_subj[held]
        dg.append(sc[np.arange(len(truth)), truth - 1].numpy())
    dg = np.concatenate(dg)

    # ---- 17/3 unknown proxy
    rng = np.random.default_rng(SEED)
    splits = [rng.permutation(np.arange(1, N_BASE + 1))[:3] for _ in range(6)]
    prot_w = protos(w_tr, early)
    prot_e = protos(p_tr, early)
    sw_late = mat(w_tr[late], tr_tasks[late], prot_w, True)
    correct = sw_late.argmin(1).numpy() + 1 == tr_subj[late]
    du = []
    for unknown in splits:
        known = np.setdiff1d(np.arange(1, N_BASE + 1), unknown)
        keep = torch.from_numpy(np.isin(np.arange(1, N_BASE + 1), known))
        rows = np.isin(tr_subj, unknown)
        sc = mat(p_tr[rows], tr_tasks[rows], prot_e, False).masked_fill(~keep, float("inf"))
        du.append(sc.min(1).values.numpy())
    du = np.concatenate(du)

    rows_out = []
    for lvl in LEVELS:
        tau = float(np.quantile(dg, lvl))
        acc = tau >= mat(p_tr[late], tr_tasks[late], prot_e, False).min(1).values.numpy()
        e2e = float((correct & acc).mean())
        rej_tr = float((du > tau).mean())
        # Test
        te_e = mat(p_te, te_tasks, prot_e, False).min(1).values.numpy()
        te_acc = te_e <= tau
        pred = mat(w_te, te_tasks, prot_w, True).argmin(1).numpy() + 1
        rows_out.append({
            "level": lvl, "tau": tau,
            "train_e2e": e2e, "train_unknown_rej": rej_tr,
            "train_J": (3 * e2e + rej_tr) / 4,
            "test_accept": float(te_acc[is_reg].mean()),
            "test_top1": float((pred == te_label)[is_reg].mean()),
            "test_e2e": float(((pred == te_label) & te_acc)[is_reg].mean()),
            "test_unknown_rej": float((~te_acc)[~is_reg].mean())})
        rows_out[-1]["test_J"] = (3 * rows_out[-1]["test_e2e"]
                                  + rows_out[-1]["test_unknown_rej"]) / 4
        r = rows_out[-1]
        print(f"L{lvl:.2f} tau {tau:.3f} | TRAIN e2e {e2e:.3f} rej {rej_tr:.3f} J {r['train_J']:.3f}"
              f" | TEST acc {r['test_accept']:.3f} top1 {r['test_top1']:.3f} "
              f"e2e {r['test_e2e']:.3f} rej {r['test_unknown_rej']:.3f} J {r['test_J']:.3f}",
              flush=True)

    # reference: frozen threshold with the old euclidean ranking
    tau0 = art["tau"]
    pred0 = mat(p_te, te_tasks, prot_e, False).argmin(1).numpy() + 1
    acc0 = mat(p_te, te_tasks, prot_e, False).min(1).values.numpy() <= tau0
    ref = {"tau": tau0,
           "test_top1": float((pred0 == te_label)[is_reg].mean()),
           "test_e2e": float(((pred0 == te_label) & acc0)[is_reg].mean()),
           "test_unknown_rej": float((~acc0)[~is_reg].mean())}
    ref["test_J"] = (3 * ref["test_e2e"] + ref["test_unknown_rej"]) / 4
    print("frozen reference", ref)
    (OUT / "final_select.json").write_text(
        json.dumps({"shrink": SHRINK, "rows": rows_out, "frozen": ref}, indent=2, default=float),
        encoding="utf-8")


if __name__ == "__main__":
    main()
