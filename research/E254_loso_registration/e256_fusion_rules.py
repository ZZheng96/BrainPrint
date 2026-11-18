"""E256: can the Test-good image fusions keep LOSO >= 0.98?

Same E237 window / cached view scores as E255.  The EEG branch is the E254
LDA-cos gate (LOSO 0.9935).  Rules:
  wXX   : XX * z_eeg + (1-XX) * z_view
  min   : min(z_eeg, z_view)
  soft  : min + 0.2 * max
  veto  : z_eeg, unless the view score is in its bottom 5% (then -2)
All thresholds and veto quantiles are calibrated on Train known rows only.
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
WS = (0.95, 0.90, 0.85, 0.80, 0.70)


def auc(known, unknown):
    x = np.concatenate([known, unknown])
    r = pd.Series(x).rank(method="average").to_numpy()
    return float((r[:len(known)].sum() - len(known) * (len(known) + 1) / 2)
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
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    fr_np = meta.trial.to_numpy(float) / np.maximum(mx, 1.0)
    fr = torch.as_tensor(fr_np, device=DEV)
    eval_idx = np.flatnonzero((fr_np > 0.6) & (fr_np <= 0.8))
    y = subj.cpu().numpy()[eval_idx]

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

    views = {v: np.load(E237 / f"view_{v}.npz")["fold_score"].astype(np.float64) for v in VIEWS}

    def zs(x):
        x = np.asarray(x, dtype=np.float64)
        return (x - x.mean()) / max(x.std(), 1e-8)

    eeg_z = np.stack([zs(eeg_mat[h]) for h in range(N_ID)])
    view_z = {v: np.stack([zs(views[v][h]) for h in range(N_ID)]) for v in VIEWS}

    def loso(mat):
        rej, rej95, kn, un = [], [], [], []
        for h in range(N_ID):
            u = y == (h + 1)
            k = ~u
            q90 = float(np.quantile(mat[h][k], 0.10))
            q95 = float(np.quantile(mat[h][k], 0.05))
            rej.append(float((mat[h][u] < q90).mean()))
            rej95.append(float((mat[h][u] < q95).mean()))
            kn.append(mat[h][k])
            un.append(mat[h][u])
        return {"auroc": auc(np.concatenate(kn), np.concatenate(un)),
                "u95": float(np.mean(rej95)),
                "rej90": float(np.mean(rej)), "min": float(np.min(rej))}

    def rule(name, ze, zi):
        if name.startswith("w"):
            w = float(name[1:]) / 100.0
            return w * ze + (1 - w) * zi
        if name == "min":
            return np.minimum(ze, zi)
        if name == "soft":
            return np.minimum(ze, zi) + 0.2 * np.maximum(ze, zi)
        raise ValueError(name)

    results = {"eeg_only": loso(eeg_mat)}
    mats = {}
    for v in VIEWS:
        for name in [f"w{int(w * 100)}" for w in WS] + ["min", "soft"]:
            m = rule(name, eeg_z, view_z[v])
            mats[(v, name)] = m
            results[f"{v}_{name}"] = loso(m)
        # veto: bottom 5% of the view score among Train known rows
        veto = np.empty_like(eeg_z)
        for h in range(N_ID):
            k = y != (h + 1)
            q05 = float(np.quantile(view_z[v][h][k], 0.05))
            veto[h] = np.where(view_z[v][h] >= q05, eeg_mat[h], eeg_mat[h] - 2.0)
        mats[(v, "veto")] = veto
        results[f"{v}_veto"] = loso(veto)

    for k_, v_ in results.items():
        print(f"LOSO {k_:28s} auc {v_['auroc']:.4f} u95 {v_['u95']:.4f} "
              f"rej90 {v_['rej90']:.4f} min {v_['min']:.3f}")

    # ---- Test ----
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
    ze_te = (gate_te - mu_kn) / sd_kn
    tau_eeg_z = float(np.quantile((known_pool - mu_kn) / sd_kn, 0.10))
    test = {"eeg_only": {"S1": float((ze_te < tau_eeg_z)[unk].mean()),
                         "S2": float((ze_te >= tau_eeg_z)[reg].mean())}}
    for v in VIEWS:
        arr = views[v]
        vm, vsd = float(arr.mean()), float(arr.std() + 1e-8)
        zi_te = (np.load(E237 / f"view_{v}.npz")["test_raw"].astype(np.float64) - vm) / vsd
        for name in [f"w{int(w * 100)}" for w in WS] + ["min", "soft"]:
            m = rule(name, eeg_z, view_z[v])
            pooled = np.concatenate([m[h][y != (h + 1)] for h in range(N_ID)])
            tau = float(np.quantile(pooled, 0.10))
            te_s = rule(name, ze_te, zi_te)
            acc = te_s >= tau
            test[f"{v}_{name}"] = {"S1": float((~acc)[unk].mean()), "S2": float(acc[reg].mean())}
        # veto threshold from Train known pooled
        tr_view_known = np.concatenate([view_z[v][h][y != (h + 1)] for h in range(N_ID)])
        qv = float(np.quantile(tr_view_known, 0.05))
        te_s = np.where(zi_te >= qv, ze_te, ze_te - 2.0)
        pooled = np.concatenate([mats[(v, "veto")][h][y != (h + 1)] for h in range(N_ID)])
        tau = float(np.quantile(pooled, 0.10))
        acc = te_s >= tau
        test[f"{v}_veto"] = {"S1": float((~acc)[unk].mean()), "S2": float(acc[reg].mean())}

    print("TEST")
    for k_, v_ in test.items():
        print(f"  {k_:28s} S1 {v_['S1']:.3f} S2 {v_['S2']:.3f}")
    (OUT / "e256_fusion_rules.json").write_text(json.dumps(
        {"loso": results, "test": test}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
