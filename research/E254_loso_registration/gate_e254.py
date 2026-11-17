"""E254h: compare per-query open-set gates under strict identity-LOSO.

Every gate uses only the query's own score profile (no batch statistics), so it
is deployable and Test-safe:
  abs   : s1 + 0.5 * (s1 - s2)
  relz  : (s1 - mean(other 19)) / std(other 19)
  gapz  : (s1 - s2 - mu_gap) / sd_gap, statistics from Train calibration rows
  soft  : log softmax posterior of the best identity (temperature 0.05)

The gate is chosen from the Train-only LOSO table.  Test is then audited once
for every candidate and reported in full.
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


def gates(mat, gap_mu, gap_sd):
    top2 = mat.topk(2, dim=1).values
    s1, s2 = top2[:, 0], top2[:, 1]
    idx = mat.argmax(1, keepdim=True)
    others = mat.clone()
    others.scatter_(1, idx, -float("inf"))
    keep = torch.isfinite(others)
    mean_o = others[keep].reshape(len(mat), -1).mean(1)
    std_o = others[keep].reshape(len(mat), -1).std(1).clamp_min(1e-6)
    return {
        "abs": s1 + 0.5 * (s1 - s2),
        "relz": (s1 - mean_o) / std_o,
        "gapz": ((s1 - s2) - gap_mu) / gap_sd,
        "soft": torch.log_softmax(mat / 0.05, dim=1).max(1).values,
    }


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

    names = ("abs", "relz", "gapz", "soft")
    known_all = {n: [] for n in names}
    unknown_all = {n: [] for n in names}
    gap_mus, gap_sds = [], []
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
        q = q0 @ ((we * wv.clamp_min(1e-6).rsqrt()) @ we.T)

        protos = torch.zeros(4, N_ID, K, DIM, device=DEV)
        for s in ids:
            for t in range(4):
                idx = torch.nonzero(rows_f & (subj == s) & (tasks == t)).flatten()
                idx = idx[torch.argsort(order[idx])]
                for j in range(K):
                    protos[t, s - 1, j] = q[idx[j * len(idx) // K:(j + 1) * len(idx) // K]].mean(0)

        def score(mask):
            idx = torch.nonzero(mask).flatten()
            out = torch.full((len(idx), N_ID), -float("inf"), device=DEV)
            for t in range(4):
                m = tasks[idx] == t
                if not bool(m.any()):
                    continue
                x, p = q[idx[m]], protos[t]
                xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                pn = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                out[m] = (xn[:, None, None, :] * pn[None]).sum(-1).max(2).values
            return out

        cal_mat = score(cal)
        top_c = cal_mat.topk(2, dim=1).values
        gap = (top_c[:, 0] - top_c[:, 1]).cpu().numpy()
        gm, gs = float(gap.mean()), float(gap.std() + 1e-6)
        gap_mus.append(gm)
        gap_sds.append(gs)
        gk, gu = gates(score(evk), gm, gs), gates(score(evu), gm, gs)
        for n in names:
            known_all[n].append(gk[n].cpu().numpy())
            unknown_all[n].append(gu[n].cpu().numpy())
        print(f"held {held:2d} done", flush=True)

    train_summary = {}
    for n in names:
        kv, uv = np.concatenate(known_all[n]), np.concatenate(unknown_all[n])
        train_summary[n] = {
            "auroc": auroc(kv, uv),
            "u95": float((uv < np.quantile(kv, 0.05)).mean()),
            "rej90": float((uv < np.quantile(kv, 0.10)).mean()),
            "tau90": float(np.quantile(kv, 0.10)),
        }
    print("TRAIN strict LOSO (pooled)")
    for n, v in train_summary.items():
        print(f"  {n:5s} auc {v['auroc']:.4f} u95 {v['u95']:.4f} rej90 {v['rej90']:.4f}")

    # ---- one-shot Test audit: same frozen package, tau taken from Train ----
    art = M.load_model(str(PKG), str(DEV))
    te = pd.read_csv(CACHE / "test_metadata.csv")
    lab = pd.read_csv(ROOT / "Brainprint_Competition/test.csv")
    te = te.merge(lab, left_on="sample", right_on="Sample", how="left")
    te_feat = torch.from_numpy(
        np.load(CACHE / "e253_subwindow/test_feat.npy").astype(np.float32).mean(1)).to(DEV)
    te_tasks = torch.as_tensor(M._task_indices([{"task": t} for t in te["task"]], len(te)), device=DEV)
    st = torch.full((len(te), art["prototypes"].shape[1]), -float("inf"), device=DEV)
    for t in range(4):
        m = te_tasks == t
        if not bool(m.any()):
            continue
        x = M._project(art, te_feat[m], t)
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        p = art["prototypes"][t].reshape(-1, art["prototypes"].shape[-1]) @ art["whiten"]
        p = (p / p.norm(dim=-1, keepdim=True).clamp_min(1e-8)).reshape(
            art["prototypes"].shape[1], art["n_proto"], -1)
        s = (x[:, None, None, :] * p[None]).sum(-1).max(2).values
        st[m] = s.masked_fill(~art["valid"][t][None], -float("inf"))
    top_te = st.topk(2, dim=1).values
    gt = {"abs": top_te[:, 0] + 0.5 * (top_te[:, 0] - top_te[:, 1])}
    idx = st.argmax(1, keepdim=True)
    others = st.clone()
    others.scatter_(1, idx, -float("inf"))
    keep_t = torch.isfinite(others)
    gt["relz"] = (top_te[:, 0] - others[keep_t].reshape(len(st), -1).mean(1)) / \
        others[keep_t].reshape(len(st), -1).std(1).clamp_min(1e-6)
    gt["gapz"] = (top_te[:, 0] - top_te[:, 1] - float(np.mean(gap_mus))) / float(np.mean(gap_sds))
    gt["soft"] = torch.log_softmax(st / 0.05, dim=1).max(1).values
    label = te["Label"].to_numpy()
    reg, unknown = label > 0, label == 0
    pred_te = (st.argmax(1) + 1).cpu().numpy()
    test_summary = {}
    for n, v in gt.items():
        s = v.cpu().numpy()
        tau = train_summary[n]["tau90"]
        accept = s >= tau
        test_summary[n] = {
            "registered_accept": float(accept[reg].mean()),
            "unregistered_reject": float((~accept[unknown]).mean()),
            "top1_given_accept": float((pred_te[reg & accept] == label[reg & accept]).mean()),
        }
        print(f"TEST {n:5s} acc {test_summary[n]['registered_accept']:.3f} "
              f"rej {test_summary[n]['unregistered_reject']:.3f} "
              f"top1 {test_summary[n]['top1_given_accept']:.3f}")
    (OUT / "gate_candidates.json").write_text(json.dumps(
        {"train_loso": train_summary, "test": test_summary}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
