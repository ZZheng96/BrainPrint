"""E254n: pseudo-session nuisance removal (NAP) under the strict LOSO protocol.

Between-time-block differences inside Train are treated as a pseudo-session
shift.  The top-r directions that maximise between-block / within-block
variance are removed from the whitened projection, then the E254 match-score
gate is re-evaluated.

r = 0 is the baseline.  r is chosen on Train LOSO only; the frozen model is then
audited once on Test.
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
PKG = ROOT / "final2/team_hymn"
OUT = Path(__file__).resolve().parent / "results"
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_ID, RS = 20, (0, 4, 8, 16)
SHRINK, K, BETA, DIM = 0.95, 4, 0.5, 256


def auroc(known, unknown):
    x = np.concatenate([known, unknown])
    ranks = pd.Series(x).rank(method="average").to_numpy()
    return float((ranks[:len(known)].sum() - len(known) * (len(known) + 1) / 2)
                 / (len(known) * len(unknown)))


def nap_dirs(q, rows, subj, tasks, frac, fit_ids, r):
    """Top-r between-block directions in the whitened space."""
    if r == 0:
        return None
    diffs, resid = [], []
    for s in fit_ids:
        for t in range(4):
            e = rows & (subj == s) & (tasks == t) & (frac <= 0.3)
            l = rows & (subj == s) & (tasks == t) & (frac > 0.3)
            if int(e.sum()) < 2 or int(l.sum()) < 2:
                continue
            me, ml = q[e].mean(0), q[l].mean(0)
            diffs.append(ml - me)
            resid.append(q[e] - me)
            resid.append(q[l] - ml)
    d = torch.stack(diffs)
    cb = d.T @ d / len(d)
    res = torch.cat(resid)
    cw = res.T @ res / max(len(res) - 1, 1)
    cw = cw + (1e-4 * torch.trace(cw) / DIM) * torch.eye(DIM, device=q.device)
    vals, vecs = torch.linalg.eigh(cw)
    inv = (vecs * vals.clamp_min(1e-8).rsqrt()) @ vecs.T
    m = inv @ cb @ inv
    w, v = torch.linalg.eigh(m)
    order = torch.argsort(w, descending=True)[:r]
    u = inv @ v[:, order]
    u = u / u.norm(dim=0, keepdim=True).clamp_min(1e-8)
    return u


def main() -> None:
    sys.path.insert(0, str(PKG))
    import model as M  # noqa: E402

    meta = pd.read_csv(CACHE / "train_metadata.csv")
    feat = torch.from_numpy(
        np.load(CACHE / "e253_subwindow/train_feat.npy").astype(np.float32).mean(1)).to(DEV)
    tasks = torch.as_tensor(M._task_indices([{"task": t} for t in meta["task"]], len(meta)), device=DEV)
    subj = torch.as_tensor(meta["subject"].to_numpy(), device=DEV)
    order = torch.as_tensor(meta.groupby(["subject", "task"]).cumcount().to_numpy(), device=DEV)
    g = meta.groupby(["subject", "task"])
    frac = torch.as_tensor(((g.cumcount() + 1) / g["trial"].transform("size")).to_numpy(), device=DEV)

    known_all = {r: [] for r in RS}
    unknown_all = {r: [] for r in RS}
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
        q0 = (z - pm) @ basis
        resid = [q0[rows_e & (subj == s)] - q0[rows_e & (subj == s)].mean(0) for s in ids]
        r_ = torch.cat(resid)
        covw = r_.T @ r_ / max(len(r_) - len(ids), 1)
        covw = SHRINK * covw + (1 - SHRINK) * torch.diagonal(covw).mean() * torch.eye(DIM, device=DEV)
        wv, we = torch.linalg.eigh(covw)
        W = (we * wv.clamp_min(1e-6).rsqrt()) @ we.T
        q_full = q0 @ W

        protos = torch.zeros(4, N_ID, K, DIM, device=DEV)
        for s in ids:
            for t in range(4):
                idx = torch.nonzero(rows_f & (subj == s) & (tasks == t)).flatten()
                idx = idx[torch.argsort(order[idx])]
                for j in range(K):
                    protos[t, s - 1, j] = q_full[idx[j * len(idx) // K:(j + 1) * len(idx) // K]].mean(0)

        for r in RS:
            u = nap_dirs(q_full, rows_e | rows_f, subj, tasks, frac, ids, r)
            q = q_full if u is None else q_full - (q_full @ u) @ u.T
            p = protos if u is None else protos - ((protos.reshape(-1, DIM) @ u) @ u.T).reshape_as(protos)

            def gate_vals(mask):
                idx = torch.nonzero(mask).flatten()
                out = torch.full((len(idx), N_ID), -float("inf"), device=DEV)
                for t in range(4):
                    m = tasks[idx] == t
                    if not bool(m.any()):
                        continue
                    x = q[idx[m]]
                    xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    pn = (p[t].reshape(-1, DIM) / p[t].reshape(-1, DIM).norm(
                        dim=-1, keepdim=True).clamp_min(1e-8)).reshape(N_ID, K, -1)
                    out[m] = (xn[:, None, None, :] * pn[None]).sum(-1).max(2).values
                top2 = out.topk(2, dim=1).values
                return (top2[:, 0] + BETA * (top2[:, 0] - top2[:, 1])).cpu().numpy()

            known_all[r].append(gate_vals(evk))
            unknown_all[r].append(gate_vals(evu))
        print(f"held {held:2d} done", flush=True)

    loso = {}
    for r in RS:
        kv, uv = np.concatenate(known_all[r]), np.concatenate(unknown_all[r])
        folds = [float((u < np.quantile(k, 0.10)).mean()) for k, u in zip(known_all[r], unknown_all[r])]
        loso[r] = {"auroc": auroc(kv, uv),
                   "u95": float((uv < np.quantile(kv, 0.05)).mean()),
                   "rej90": float((uv < np.quantile(kv, 0.10)).mean()),
                   "fold_rej90_min": float(np.min(folds))}
        print(f"r={r:2d} " + json.dumps(loso[r]), flush=True)

    feasible = [r for r in RS if loso[r]["auroc"] >= 0.98 and loso[r]["u95"] >= 0.95]
    best = max(feasible, key=lambda r_: (loso[r_]["rej90"], -r_)) if feasible else \
        max(RS, key=lambda r_: loso[r_]["auroc"] + loso[r_]["u95"] + loso[r_]["rej90"])
    print("selected r =", best)

    # ---- frozen audit with the selected r ----
    art = M.load_model(str(PKG), str(DEV))
    raw_tr = torch.from_numpy(
        np.load(CACHE / "e253_subwindow/train_feat.npy").astype(np.float32).mean(1)).to(DEV)
    proj = torch.empty((len(raw_tr), art["pca_basis"].shape[-1]), device=DEV)
    for t in range(4):
        m = tasks == t
        proj[m] = M._project(art, raw_tr[m], t)
    q_tr = proj @ art["whiten"]
    rows_frozen = torch.ones(len(feat), dtype=torch.bool, device=DEV)
    u = nap_dirs(q_tr, rows_frozen, subj, tasks, frac, range(1, N_ID + 1), best)

    def apply(q):
        return q if u is None else q - (q @ u) @ u.T

    n_cols = art["prototypes"].shape[1]

    def build(qv, rows):
        p = torch.zeros(4, n_cols, art["n_proto"], DIM, device=DEV)
        for s in range(1, N_ID + 1):
            for t in range(4):
                idx = torch.nonzero(rows & (subj == s) & (tasks == t)).flatten()
                idx = idx[torch.argsort(order[idx])]
                for j in range(art["n_proto"]):
                    p[t, s - 1, j] = qv[idx[j * len(idx) // art["n_proto"]:
                                           (j + 1) * len(idx) // art["n_proto"]]].mean(0)
        return p

    te = pd.read_csv(CACHE / "test_metadata.csv")
    lab = pd.read_csv(ROOT / "Brainprint_Competition/test.csv")
    te = te.merge(lab, left_on="sample", right_on="Sample", how="left")
    te_feat = torch.from_numpy(
        np.load(CACHE / "e253_subwindow/test_feat.npy").astype(np.float32).mean(1)).to(DEV)
    te_tasks = torch.as_tensor(M._task_indices([{"task": t} for t in te["task"]], len(te)), device=DEV)
    q_te = torch.empty((len(te_feat), art["pca_basis"].shape[-1]), device=DEV)
    for t in range(4):
        m = te_tasks == t
        q_te[m] = M._project(art, te_feat[m], t)
    q_te = apply(q_te @ art["whiten"])

    def score(qv, tk, mask, protos):
        idx = torch.nonzero(mask).flatten()
        out = torch.full((len(idx), n_cols), -float("inf"), device=DEV)
        for t in range(4):
            m = tk[idx] == t
            if not bool(m.any()):
                continue
            x = qv[idx[m]]
            xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            pt = apply(protos[t].reshape(-1, DIM)).reshape(n_cols, art["n_proto"], -1)
            pn = pt / pt.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            out[m] = (xn[:, None, None, :] * pn[None]).sum(-1).max(2).values
        out[:, N_ID:] = -float("inf")
        return out

    # tau from the same late cross-fit used by freeze_e254: prototypes <=0.6, queries >0.6
    p_cal = build(q_tr, frac <= 0.6)
    sc = score(q_tr, tasks, frac > 0.6, p_cal)
    top2 = sc.topk(2, dim=1).values
    gate_cal = (top2[:, 0] + BETA * (top2[:, 0] - top2[:, 1])).cpu().numpy()
    tau = float(np.quantile(gate_cal, 0.10))

    full = build(q_tr, rows_frozen)
    st = score(q_te, te_tasks, torch.ones(len(te), dtype=torch.bool, device=DEV), full)
    top2 = st.topk(2, dim=1).values
    g_te = (top2[:, 0] + BETA * (top2[:, 0] - top2[:, 1])).cpu().numpy()
    accept = g_te >= tau
    label = te["Label"].to_numpy()
    reg, unknown = label > 0, label == 0
    pred = (st.argmax(1) + 1).cpu().numpy()
    test = {"r": best, "tau": tau,
            "registered_accept": float(accept[reg].mean()),
            "unregistered_reject": float((~accept[unknown]).mean()),
            "top1_given_accept": float((pred[reg & accept] == label[reg & accept]).mean())}
    print("TEST", json.dumps(test, indent=2))
    (OUT / "nap_e254.json").write_text(json.dumps(
        {"loso": {str(k): v for k, v in loso.items()}, "selected_r": best, "test": test},
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
