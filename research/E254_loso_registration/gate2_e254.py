"""E254i: two-branch open-set gate.

Branch 1 (identity evidence): LDA-cos profile gate  s1 + 0.5*(s1-s2)
Branch 2 (domain-safe recovery): non-whitened euclidean nearest-prototype
          distance passes AND the LDA profile is unambiguous (relz).

accept = branch1 OR (euclid_pass AND relz_pass)

Both thresholds are calibrated on Train calibration rows only.  The policy is
pre-registered: branch 2 exists to recover genuine trials whose absolute
whitened score collapses under session shift, while the relz condition keeps
pseudo-unknown identities out.
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


def stats(mat):
    s1, s2 = mat.topk(2, dim=1).values.T
    idx = mat.argmax(1, keepdim=True)
    others = mat.clone()
    others.scatter_(1, idx, -float("inf"))
    keep = torch.isfinite(others)
    mean_o = others[keep].reshape(len(mat), -1).mean(1)
    std_o = others[keep].reshape(len(mat), -1).std(1).clamp_min(1e-6)
    return {"s1": s1, "abs": s1 + 0.5 * (s1 - s2), "relz": (s1 - mean_o) / std_o,
            "s1v": s1, "s2v": s2}


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

    keys = ("abs", "euclid", "relz", "combo")
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
        q, q0w = q0 @ W, q0

        protos = torch.zeros(4, N_ID, K, DIM, device=DEV)
        protos0 = torch.zeros(4, N_ID, K, DIM, device=DEV)
        for s in ids:
            for t in range(4):
                idx = torch.nonzero(rows_f & (subj == s) & (tasks == t)).flatten()
                idx = idx[torch.argsort(order[idx])]
                for j in range(K):
                    chunk = idx[j * len(idx) // K:(j + 1) * len(idx) // K]
                    protos[t, s - 1, j] = q[chunk].mean(0)
                    protos0[t, s - 1, j] = q0w[chunk].mean(0)

        def score(protos, qv, mask, cosine=True):
            idx = torch.nonzero(mask).flatten()
            out = torch.full((len(idx), N_ID), -float("inf") if cosine else float("inf"), device=DEV)
            for t in range(4):
                m = tasks[idx] == t
                if not bool(m.any()):
                    continue
                x, p = qv[idx[m]], protos[t]
                flat = p.reshape(-1, DIM)
                if cosine:
                    xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    pn = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    out[m] = (xn[:, None, None, :] * pn.reshape(N_ID, K, -1)[None]).sum(-1).max(2).values
                else:
                    out[m] = torch.cdist(x, flat).reshape(len(x), N_ID, K).min(2).values
            return out

        c_lda, c_eu = score(protos, q, cal, True), score(protos0, q0, cal, False)
        k_lda, k_eu = score(protos, q, evk, True), score(protos0, q0, evk, False)
        u_lda, u_eu = score(protos, q, evu, True), score(protos0, q0, evu, False)
        tau_abs = float(np.quantile(stats(c_lda)["abs"].cpu().numpy(), 0.10))
        tau_relz = float(np.quantile(stats(c_lda)["relz"].cpu().numpy(), 0.10))
        tau_eu = float(np.quantile(c_eu.min(1).values.cpu().numpy(), 0.90))

        def gates(lda, eu):
            st = stats(lda)
            b1 = st["abs"] >= tau_abs
            b2 = (eu.min(1).values <= tau_eu) & (st["relz"] >= tau_relz)
            return {
                "abs": st["abs"],
                "euclid": -eu.min(1).values,
                "relz": st["relz"],
                "combo": torch.where(b1 | b2, torch.ones_like(st["abs"]),
                                     torch.zeros_like(st["abs"])),
            }

        gk, gu = gates(k_lda, k_eu), gates(u_lda, u_eu)
        for k in keys:
            known_all[k].append(gk[k].cpu().numpy())
            unknown_all[k].append(gu[k].cpu().numpy())
        print(f"held {held:2d} thresholds abs {tau_abs:.3f} relz {tau_relz:.2f} eu {tau_eu:.2f}",
              flush=True)

    train_summary = {}
    for k in keys:
        kv, uv = np.concatenate(known_all[k]), np.concatenate(unknown_all[k])
        train_summary[k] = {
            "auroc": auroc(kv, uv),
            "u95": float((uv < np.quantile(kv, 0.05)).mean()),
            "rej90": float((uv < np.quantile(kv, 0.10)).mean()),
            "known_accept": float((kv >= np.quantile(kv, 0.10)).mean()),
        }
    print("TRAIN strict LOSO (pooled)")
    for k, v in train_summary.items():
        print(f"  {k:6s} auc {v['auroc']:.4f} u95 {v['u95']:.4f} rej90 {v['rej90']:.4f}")

    # ---- Test audit with the frozen package ----
    art = M.load_model(str(PKG), str(DEV))
    te = pd.read_csv(CACHE / "test_metadata.csv")
    lab = pd.read_csv(ROOT / "Brainprint_Competition/test.csv")
    te = te.merge(lab, left_on="sample", right_on="Sample", how="left")
    te_feat = torch.from_numpy(
        np.load(CACHE / "e253_subwindow/test_feat.npy").astype(np.float32).mean(1)).to(DEV)
    te_tasks = torch.as_tensor(M._task_indices([{"task": t} for t in te["task"]], len(te)), device=DEV)
    st_lda = torch.full((len(te), art["prototypes"].shape[1]), -float("inf"), device=DEV)
    st_eu = torch.full((len(te), art["prototypes"].shape[1]), float("inf"), device=DEV)
    for t in range(4):
        m = te_tasks == t
        if not bool(m.any()):
            continue
        x = M._project(art, te_feat[m], t)
        xw = x @ art["whiten"]
        p = art["prototypes"][t]
        flat = p.reshape(-1, p.shape[-1])
        pw = flat @ art["whiten"]
        xn = xw / xw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        pn = pw / pw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        s = (xn[:, None, None, :] * pn.reshape(art["prototypes"].shape[1], art["n_proto"], -1)[None]).sum(-1).max(2).values
        st_lda[m] = s.masked_fill(~art["valid"][t][None], -float("inf"))
        d = torch.cdist(x, flat).reshape(len(x), art["prototypes"].shape[1], art["n_proto"]).min(2).values
        st_eu[m] = d.masked_fill(~art["valid"][t][None], float("inf"))
    te_st = stats(st_lda)
    # Train-calibrated thresholds are the pooled 10th/90th percentiles above
    tau_abs = float(np.quantile(np.concatenate(known_all["abs"]), 0.10))
    tau_relz = float(np.quantile(np.concatenate(known_all["relz"]), 0.10))
    tau_eu = float(np.quantile(-np.concatenate(known_all["euclid"]), 0.90))
    b1 = te_st["abs"] >= tau_abs
    b2 = (st_eu.min(1).values <= tau_eu) & (te_st["relz"] >= tau_relz)
    accept = (b1 | b2).cpu().numpy()
    label = te["Label"].to_numpy()
    reg, unknown = label > 0, label == 0
    pred = (st_lda.argmax(1) + 1).cpu().numpy()
    test_summary = {
        "accept_registered": float(accept[reg].mean()),
        "reject_unregistered": float((~accept[unknown]).mean()),
        "top1_given_accept": float((pred[reg & accept] == label[reg & accept]).mean()),
        "e2e": float(((pred == label) & accept)[reg].mean()),
        "branch1_only_accept": float(b1.cpu().numpy()[reg].mean()),
        "branch2_only_accept": float(b2.cpu().numpy()[reg].mean()),
    }
    print("TEST", json.dumps(test_summary, indent=2))
    (OUT / "gate_two_branch.json").write_text(json.dumps(
        {"train_loso": train_summary, "test": test_summary, "tau": {"abs": tau_abs,
         "relz": tau_relz, "euclid": tau_eu}}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
