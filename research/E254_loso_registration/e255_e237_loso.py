"""E255: raise E237's LOSO by transplanting the E254 decision layer.

E237 / E239 established that monotone transforms of a fixed score cannot change
the (LOSO, S2) pair: the only way to gain S2 at fixed LOSO is a higher LOSO AUC
("headroom").  This script keeps E237's 20-fold window, metadata order and
cached image-view scores, but replaces the EEG branch by the E254 rule
(PCA-256, WCCN shrink 0.95, k=4 prototypes, gate s1 + 0.5*(s1-s2)).

Reports:
  * LOSO AUC / rejection @ 90% acceptance for EEG-only and for score fusion
  * Test S1 rejection / S2 acceptance at the LOSO tau (raw and aligned views)
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
E237 = ROOT / "research/E237_image_view_fusion/cache"
OUT = Path(__file__).resolve().parent / "results"
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_ID, DIM, SHRINK, K, BETA = 20, 256, 0.95, 4, 0.5
VIEWS = ("X_mw_cov4", "X_covariance_hr", "X_correlation")


def auc(known, unknown):
    x = np.concatenate([known, unknown])
    r = pd.Series(x).rank(method="average").to_numpy()
    return float((r[:len(known)].sum() - len(known) * (len(known) + 1) / 2)
                 / (len(known) * len(unknown)))


def main() -> None:
    sys.path.insert(0, str(PKG))
    import model as M  # noqa: E402

    meta = pd.read_csv(CACHE / "train_metadata.csv")
    sw = torch.from_numpy(
        np.load(CACHE / "e253_subwindow/train_feat.npy").astype(np.float32)).to(DEV)
    feat = sw.mean(1)
    tasks = torch.as_tensor(M._task_indices([{"task": t} for t in meta["task"]], len(meta)), device=DEV)
    subj = torch.as_tensor(meta["subject"].to_numpy(), device=DEV)
    order = torch.as_tensor(meta.groupby(["subject", "task"]).cumcount().to_numpy(), device=DEV)
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    fr_np = meta.trial.to_numpy(float) / np.maximum(mx, 1.0)   # E237's frac
    fr = torch.as_tensor(fr_np, device=DEV)
    eval_idx = np.flatnonzero((fr_np > 0.6) & (fr_np <= 0.8))
    print("eval rows", len(eval_idx))

    eeg_mat = np.zeros((N_ID, len(eval_idx)))
    for held in range(1, N_ID + 1):
        keep = subj != held
        ids = [s for s in range(1, N_ID + 1) if s != held]
        fit = keep & (fr <= 0.6)
        mu = torch.stack([feat[fit & (tasks == t)].mean(0) for t in range(4)])
        sd = torch.stack([feat[fit & (tasks == t)].std(0).clamp_min(1e-6) for t in range(4)])
        z = torch.empty_like(feat)
        for t in range(4):
            m = tasks == t
            z[m] = (feat[m] - mu[t]) / sd[t]
        xf = z[fit]
        pm = xf.mean(0)
        xc = xf - pm
        cov = xc.T @ xc / max(len(xf) - 1, 1)
        vals, vecs = torch.linalg.eigh(cov)
        basis = vecs[:, torch.argsort(vals, descending=True)[:DIM]].contiguous()
        q0 = (z - pm) @ basis
        resid = [q0[fit & (subj == s)] - q0[fit & (subj == s)].mean(0) for s in ids]
        r_ = torch.cat(resid)
        covw = r_.T @ r_ / max(len(r_) - len(ids), 1)
        covw = SHRINK * covw + (1 - SHRINK) * torch.diagonal(covw).mean() * torch.eye(DIM, device=DEV)
        wv, we = torch.linalg.eigh(covw)
        q = q0 @ ((we * wv.clamp_min(1e-6).rsqrt()) @ we.T)
        protos = torch.zeros(4, N_ID, K, DIM, device=DEV)
        for s in ids:
            for t in range(4):
                ii = torch.nonzero(fit & (subj == s) & (tasks == t)).flatten()
                ii = ii[torch.argsort(order[ii])]
                for j in range(K):
                    protos[t, s - 1, j] = q[ii[j * len(ii) // K:(j + 1) * len(ii) // K]].mean(0)
        ev = torch.as_tensor(eval_idx, device=DEV)
        out = torch.full((len(ev), N_ID), -float("inf"), device=DEV)
        for t in range(4):
            m = tasks[ev] == t
            if not bool(m.any()):
                continue
            x = q[ev[m]]
            xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            pn = (protos[t].reshape(-1, DIM) / protos[t].reshape(-1, DIM).norm(
                dim=-1, keepdim=True).clamp_min(1e-8)).reshape(N_ID, K, -1)
            out[m] = (xn[:, None, None, :] * pn[None]).sum(-1).max(2).values
        top2 = out.topk(2, dim=1).values
        eeg_mat[held - 1] = (top2[:, 0] + BETA * (top2[:, 0] - top2[:, 1])).cpu().numpy()

    y = subj.cpu().numpy()[eval_idx]
    views = {v: np.load(E237 / f"view_{v}.npz")["fold_score"].astype(np.float64) for v in VIEWS}

    def zs(x):
        x = np.asarray(x, dtype=np.float64)
        return (x - x.mean()) / max(x.std(), 1e-8)

    report = {"eval_rows": int(len(eval_idx)), "views": list(VIEWS), "loso": {}}

    def loso_metrics(mat):
        fold_rej, known, unknown = [], [], []
        for h in range(N_ID):
            u = y == (h + 1)
            k = ~u
            q90 = float(np.quantile(mat[h][k], 0.10))
            fold_rej.append(float((mat[h][u] < q90).mean()))
            known.append(mat[h][k])
            unknown.append(mat[h][u])
        kv, uv = np.concatenate(known), np.concatenate(unknown)
        return {"auroc": auc(kv, uv), "rej90": float(np.mean(fold_rej)),
                "rej90_min": float(np.min(fold_rej))}

    report["loso"]["eeg_only"] = loso_metrics(eeg_mat)
    for v, arr in views.items():
        report["loso"][f"eeg_plus_{v}_50"] = loso_metrics(
            0.5 * np.stack([zs(eeg_mat[h]) for h in range(N_ID)])
            + 0.5 * np.stack([zs(arr[h]) for h in range(N_ID)]))
        report["loso"][f"eeg_plus_{v}_70"] = loso_metrics(
            0.7 * np.stack([zs(eeg_mat[h]) for h in range(N_ID)])
            + 0.3 * np.stack([zs(arr[h]) for h in range(N_ID)]))
        report["loso"][f"img_only_{v}"] = loso_metrics(
            np.stack([zs(arr[h]) for h in range(N_ID)]))
    for k_, v_ in report["loso"].items():
        print(f"LOSO {k_:24s} auc {v_['auroc']:.4f} rej90 {v_['rej90']:.4f} min {v_['rej90_min']:.3f}")

    # ---- Test: E254 frozen gate + E237 view Test scores ----
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
        x = M._project(art, te_feat[m], t) @ art["whiten"]
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        pt = art["prototypes"][t].reshape(-1, art["prototypes"].shape[-1]) @ art["whiten"]
        pt = (pt / pt.norm(dim=-1, keepdim=True).clamp_min(1e-8)).reshape(
            art["prototypes"].shape[1], art["n_proto"], -1)
        st[m] = ((x[:, None, None, :] * pt[None]).sum(-1).max(2).values).masked_fill(
            ~art["valid"][t][None], -float("inf"))
    top2 = st.topk(2, dim=1).values
    gate_te = (top2[:, 0] + BETA * (top2[:, 0] - top2[:, 1])).cpu().numpy()
    label = te["Label"].to_numpy()
    reg, unk = label > 0, label == 0
    known_pool = np.concatenate([eeg_mat[h][y != (h + 1)] for h in range(N_ID)])
    mu_kn, sd_kn = float(known_pool.mean()), float(known_pool.std() + 1e-8)
    tau = float(np.quantile(known_pool, 0.10))
    rep = {"tau": tau, "test": {}}
    acc = gate_te >= tau
    rep["test"]["eeg_only"] = {"S1": float((~acc)[unk].mean()), "S2": float(acc[reg].mean())}
    for v in VIEWS:
        arr = views[v]
        vm, vsd = float(arr.mean()), float(arr.std() + 1e-8)
        z = np.load(E237 / f"view_{v}.npz")
        vte = (z["test_raw"].astype(np.float64) - vm) / vsd
        for w in (0.5, 0.7):
            fused_tr = w * np.stack([zs(eeg_mat[h]) for h in range(N_ID)]) + \
                (1 - w) * np.stack([zs(arr[h]) for h in range(N_ID)])
            pooled = np.concatenate([fused_tr[h][y != (h + 1)] for h in range(N_ID)])
            tau_f = float(np.quantile(pooled, 0.10))
            fused = w * ((gate_te - mu_kn) / sd_kn) + (1 - w) * vte
            accf = fused >= tau_f
            rep["test"][f"{v}_raw_{int(w * 100)}"] = {"S1": float((~accf)[unk].mean()),
                                                      "S2": float(accf[reg].mean())}
    report["test"] = rep
    print("TEST", json.dumps(rep, indent=2))
    (OUT / "e255_e237_loso.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
