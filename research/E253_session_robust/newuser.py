"""E253j: new-user simulation for the whitened-cosine ranking vs the frozen euclid one.

Mirrors the final scene 3: 3 subjects act as new users (20 labelled enrollment
samples from early trials, tested on late trials), 3 others are unregistered and
must be rejected, the remaining 14 are registered users.
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
REPS = 30


def normed(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def main() -> None:
    art = load_artifacts()
    tr_feat = np.load(CACHE / "train_feat.npy")
    tr = pd.read_csv(ROOT / "BrainPrint_cache/train_metadata.csv")
    tr_tasks = M._task_indices([{"task": t} for t in tr["task"]], len(tr))
    tr_subj = tr["subject"].to_numpy()
    g = tr.groupby(["subject", "task"])
    frac = ((g.cumcount() + 1) / g["trial"].transform("size")).to_numpy()
    early, late = frac <= 0.6, frac > 0.8

    p_tr = project(art, tr_feat, tr_tasks).mean(1)
    W = fit_whiten(p_tr[early], tr_subj[early], SHRINK)
    w_tr = p_tr @ W

    def protos(x, mask, extra=None):
        """(4, 23, d) with 0-based identity index; extra = [(id, rows)]"""
        out = torch.zeros(4, 23, x.shape[-1])
        valid = torch.zeros(4, 23, dtype=torch.bool)
        for t in range(4):
            for s in range(1, N_BASE + 1):
                r = mask & (tr_tasks == t) & (tr_subj == s)
                if r.sum() >= 3:
                    out[t, s - 1] = x[r].mean(0)
                    valid[t, s - 1] = True
        if extra:
            for uid, rows in extra:
                for t in range(4):
                    rr = rows & (tr_tasks == t)
                    if rr.sum():
                        out[t, uid - 1] = x[rr].mean(0)
                        valid[t, uid - 1] = True
        return out, valid

    def decide(x, tasks, prot, valid, tau, cos):
        n = len(x)
        out = torch.full((n, 23), float("inf"))
        pj = normed(x) if cos else x
        pp = normed(prot) if cos else prot
        for t in range(4):
            r = np.nonzero(tasks == t)[0]
            if r.size:
                out[r] = torch.cdist(pj[r], pp[t]).masked_fill(~valid[t][None], float("inf"))
        return out

    # tau for euclid rejection under the new W is not needed: the frozen distance
    # geometry is unchanged, so reuse the 5-block OOF calibrated value.
    dg = []
    block = np.clip((frac * 5).astype(int), 0, 4)
    for b in range(5):
        fit, held = block != b, block == b
        prot, valid = protos(p_tr, fit)
        sc = decide(p_tr[held], tr_tasks[held], prot, valid, 0.0, False)
        truth = tr_subj[held]
        dg.append(sc[np.arange(len(truth)), truth - 1].numpy())
    tau = float(np.quantile(np.concatenate(dg), 0.90))

    rng = np.random.default_rng(7)
    res = []
    for rep in range(REPS):
        perm = rng.permutation(np.arange(1, N_BASE + 1))
        new_ids, unk_ids, known = perm[:3], perm[3:6], perm[6:]
        extra = []
        for k, s in enumerate(new_ids):
            pool = np.nonzero((tr_subj == s) & early)[0]
            pick = rng.choice(pool, min(20, len(pool)), replace=False)
            m = np.zeros(len(tr_subj), bool)
            m[pick] = True
            extra.append((21 + k, m))
        drop = np.isin(tr_subj, np.concatenate([new_ids, unk_ids]))
        known_mask = early & ~drop
        row = {"rep": rep}
        for name, x, cos in (("euclid", p_tr, False), ("whiten_cos", w_tr, True)):
            prot, valid = protos(x, known_mask, extra)
            # rejection always uses the frozen unwhitened Euclidean geometry
            prot_e, valid_e = protos(p_tr, known_mask, extra)
            te = late & np.isin(tr_subj, new_ids)
            tr_ = late & np.isin(tr_subj, known)
            un = np.isin(tr_subj, unk_ids)
            sc_new = decide(x[te], tr_tasks[te], prot, valid, tau, cos)
            truth_new = np.array([21 + list(new_ids).index(s) for s in tr_subj[te]])
            sc_known = decide(x[tr_], tr_tasks[tr_], prot, valid, tau, cos)
            acc_known = tau >= decide(p_tr[tr_], tr_tasks[tr_], prot_e, valid_e,
                                      tau, False).min(1).values.numpy()
            acc_new = tau >= decide(p_tr[te], tr_tasks[te], prot_e, valid_e, tau, False).min(1).values.numpy()
            unk_dmin = decide(p_tr[un], tr_tasks[un], prot_e, valid_e, tau, False).min(1).values.numpy()
            row[f"{name}_new_acc"] = float(((sc_new.argmin(1).numpy() + 1 == truth_new) & acc_new).mean())
            row[f"{name}_known_acc"] = float(((sc_known.argmin(1).numpy() + 1 == tr_subj[tr_]) & acc_known).mean())
            row[f"{name}_unk_rej"] = float((unk_dmin > tau).mean())
        res.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    df = pd.DataFrame(res)
    summary = {c: float(df[c].mean()) for c in df.columns if c != "rep"}
    print("\nMEAN", {k: round(v, 4) for k, v in summary.items()})
    (OUT / "newuser.json").write_text(json.dumps({"tau": tau, "reps": REPS, "mean": summary,
                                                  "runs": res}, indent=2, default=float),
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
