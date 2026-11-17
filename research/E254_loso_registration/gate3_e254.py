"""E254j: relative-distance gate.

Absolute distances shift with the session, so the gate uses the query's own
distance profile:

  reld = d_min / mean(d to every enrolled prototype)      (lower = more genuine)

Ranking stays LDA-cos (top-1 given acceptance is high).  The gate is compared
against the absolute LDA gate on the same strict identity-LOSO folds, then the
frozen package is audited once on Test.
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

DIM, SHRINK, K = 256, 0.95, 4
N_ID = 20


def auroc(known, unknown):
    x = np.concatenate([known, unknown])
    ranks = pd.Series(x).rank(method="average").to_numpy()
    return float((ranks[:len(known)].sum() - len(known) * (len(known) + 1) / 2)
                 / (len(known) * len(unknown)))


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

    keys = ("abs", "reld", "abs_and_reld", "abs_or_reld")
    known_all = {k: [] for k in keys}
    unknown_all = {k: [] for k in keys}
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
        r = torch.cat(resid)
        covw = r.T @ r / max(len(r) - len(ids), 1)
        covw = SHRINK * covw + (1 - SHRINK) * torch.diagonal(covw).mean() * torch.eye(DIM, device=DEV)
        wv, we = torch.linalg.eigh(covw)
        W = (we * wv.clamp_min(1e-6).rsqrt()) @ we.T
        q = q0 @ W

        protos = torch.zeros(4, N_ID, K, DIM, device=DEV)
        protos0 = torch.zeros(4, N_ID, K, DIM, device=DEV)
        valid = torch.zeros(N_ID, dtype=torch.bool, device=DEV)
        for s in ids:
            valid[s - 1] = True
            for t in range(4):
                idx = torch.nonzero(rows_f & (subj == s) & (tasks == t)).flatten()
                idx = idx[torch.argsort(order[idx])]
                for j in range(K):
                    chunk = idx[j * len(idx) // K:(j + 1) * len(idx) // K]
                    protos[t, s - 1, j] = q[chunk].mean(0)
                    protos0[t, s - 1, j] = q0[chunk].mean(0)

        def lda_scores(mask):
            idx = torch.nonzero(mask).flatten()
            out = torch.full((len(idx), N_ID), -float("inf"), device=DEV)
            for t in range(4):
                m = tasks[idx] == t
                if not bool(m.any()):
                    continue
                x, p = q[idx[m]], protos[t]
                xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                pn = (p.reshape(-1, DIM) / p.reshape(-1, DIM).norm(dim=-1, keepdim=True).clamp_min(1e-8)).reshape(N_ID, K, -1)
                out[m] = (xn[:, None, None, :] * pn[None]).sum(-1).max(2).values.masked_fill(
                    ~valid[None], -float("inf"))
            return out

        def dist_scores(mask):
            idx = torch.nonzero(mask).flatten()
            out = torch.full((len(idx), N_ID), float("inf"), device=DEV)
            for t in range(4):
                m = tasks[idx] == t
                if not bool(m.any()):
                    continue
                x, p = q0[idx[m]], protos0[t]
                out[m] = torch.cdist(x, p.reshape(-1, DIM)).reshape(len(x), N_ID, K).min(2).values.masked_fill(
                    ~valid[None], float("inf"))
            return out

        def gate_values(lda, dist):
            top2 = lda.topk(2, dim=1).values
            abs_ = top2[:, 0] + 0.5 * (top2[:, 0] - top2[:, 1])
            dmin = dist.min(1).values
            dmean = dist.mean(1)
            reld = -(dmin / dmean)
            return abs_, reld

        a_c, r_c = gate_values(lda_scores(cal), dist_scores(cal))
        a_k, r_k = gate_values(lda_scores(evk), dist_scores(evk))
        a_u, r_u = gate_values(lda_scores(evu), dist_scores(evu))
        ta = float(np.quantile(a_c.cpu().numpy(), 0.10))
        tr_ = float(np.quantile(r_c.cpu().numpy(), 0.10))
        vals = {
            "abs": (a_k, a_u, ta),
            "reld": (r_k, r_u, tr_),
            "abs_and_reld": (torch.minimum(a_k / ta, r_k / tr_), torch.minimum(a_u / ta, r_u / tr_), 1.0),
            "abs_or_reld": (torch.maximum(a_k / ta, r_k / tr_), torch.maximum(a_u / ta, r_u / tr_), 1.0),
        }
        for k, (kk, uu, _) in vals.items():
            known_all[k].append(kk.cpu().numpy())
            unknown_all[k].append(uu.cpu().numpy())
        print(f"held {held:2d} abs {ta:.3f} reld {tr_:.4f}", flush=True)

    summary = {}
    for k in keys:
        kv, uv = np.concatenate(known_all[k]), np.concatenate(unknown_all[k])
        summary[k] = {"auroc": auroc(kv, uv),
                      "u95": float((uv < np.quantile(kv, 0.05)).mean()),
                      "rej90": float((uv < np.quantile(kv, 0.10)).mean()),
                      "tau90": float(np.quantile(kv, 0.10))}
    for k, v in summary.items():
        print(f"TRAIN {k:12s} auc {v['auroc']:.4f} u95 {v['u95']:.4f} rej90 {v['rej90']:.4f}")

    # ---- frozen Test audit ----
    art = M.load_model(str(PKG), str(DEV))
    te = pd.read_csv(CACHE / "test_metadata.csv")
    lab = pd.read_csv(ROOT / "Brainprint_Competition/test.csv")
    te = te.merge(lab, left_on="sample", right_on="Sample", how="left")
    te_feat = torch.from_numpy(
        np.load(CACHE / "e253_subwindow/test_feat.npy").astype(np.float32).mean(1)).to(DEV)
    te_tasks = torch.as_tensor(M._task_indices([{"task": t} for t in te["task"]], len(te)), device=DEV)
    st = torch.full((len(te), art["prototypes"].shape[1]), -float("inf"), device=DEV)
    sd_ = torch.full((len(te), art["prototypes"].shape[1]), float("inf"), device=DEV)
    for t in range(4):
        m = te_tasks == t
        if not bool(m.any()):
            continue
        x = M._project(art, te_feat[m], t)
        p = art["prototypes"][t]
        xw = x @ art["whiten"]
        pw = p.reshape(-1, p.shape[-1]) @ art["whiten"]
        xn = xw / xw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        pn = (pw / pw.norm(dim=-1, keepdim=True).clamp_min(1e-8)).reshape(
            art["prototypes"].shape[1], art["n_proto"], -1)
        st[m] = ((xn[:, None, None, :] * pn[None]).sum(-1).max(2).values).masked_fill(
            ~art["valid"][t][None], -float("inf"))
        sd_[m] = torch.cdist(x, p.reshape(-1, p.shape[-1])).reshape(
            len(x), art["prototypes"].shape[1], art["n_proto"]).min(2).values.masked_fill(
            ~art["valid"][t][None], float("inf"))
    top2 = st.topk(2, dim=1).values
    a_te = top2[:, 0] + 0.5 * (top2[:, 0] - top2[:, 1])
    r_te = -(sd_.min(1).values / sd_.mean(1))
    label = te["Label"].to_numpy()
    reg, unknown = label > 0, label == 0
    pred = (st.argmax(1) + 1).cpu().numpy()
    test = {}
    for k, score in (("abs", a_te), ("reld", r_te),
                     ("abs_and_reld", torch.minimum(a_te / summary["abs"]["tau90"],
                                                    r_te / summary["reld"]["tau90"])),
                     ("abs_or_reld", torch.maximum(a_te / summary["abs"]["tau90"],
                                                   r_te / summary["reld"]["tau90"]))):
        s = score.cpu().numpy()
        accept = s >= summary[k]["tau90"]
        test[k] = {"accept": float(accept[reg].mean()),
                   "reject": float((~accept[unknown]).mean()),
                   "top1": float((pred[reg & accept] == label[reg & accept]).mean())}
        print(f"TEST {k:12s} acc {test[k]['accept']:.3f} rej {test[k]['reject']:.3f} "
              f"top1 {test[k]['top1']:.3f}")
    (OUT / "gate_relative.json").write_text(json.dumps(
        {"train": summary, "test": test}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
