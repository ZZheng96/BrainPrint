"""E253m: the 11 acceptance metrics for the final2 scheme, same-protocol only.

Every number below is computed on this exact model (E200 features + whitened-cosine
ranking + euclidean rejection).  Nothing is borrowed from E249/E250/E251.

  #1  Train in-sample closed-set accuracy        (fit all 20 on frac<=0.6, eval same rows)
  #2  5-block OOF closed-set accuracy            (block = clip(int(frac*5),0,4))
  #3  5-block OOF identity macro-F1
  #4  forward block-out closed-set accuracy      (fit frac<=0.6, eval frac>0.6)
  #5  strict identity-LOSO pseudo-unknown AUROC  (held identity never enters z/PCA/whitening/prototypes)
  #6  U@95% (strict, tau = 95% quantile of held-out genuine distance)
  #9  Test registered acceptance                 (preliminary Test session2)
  #10 Test unregistered rejection                (preliminary Test session1, Label=0)
  #11 strict LOSO unknown rejection at a=0.90

#7/#8 (adversarial RA / ASafeR) are NOT re-measured here: the final rules state the
final round contains no adversarial samples, and re-using E251's attack numbers on a
different scoring rule would mix protocols.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "final2/team_hymn"
CACHE = ROOT / "BrainPrint_cache/e253_subwindow"
OUT = ROOT / "research/E253_session_robust/results"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(Path(__file__).parent))
import model as M  # noqa: E402

N_BASE = 20
DIM = 96
SHRINK = 0.5
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def fit_head(feat_z, rows, tasks, subj, dim=DIM):
    """Return (pca_mean, basis, W) fitted on `rows` only."""
    x = feat_z[rows]
    mu = x.mean(0)
    _, _, vt = torch.linalg.svd(x - mu, full_matrices=False)
    basis = vt[:dim].T.contiguous()
    p = (x - mu) @ basis
    y = subj[rows]
    present = [s for s in range(1, N_BASE + 1) if bool((y == s).any())]
    cm = torch.stack([p[y == s].mean(0) for s in present])
    resid = torch.cat([p[y == s] - cm[i] for i, s in enumerate(present)])
    cov = resid.T @ resid / max(len(resid) - len(cm), 1)
    cov = SHRINK * cov + (1 - SHRINK) * torch.diagonal(cov).mean() * torch.eye(
        dim, device=cov.device, dtype=cov.dtype)
    vals, vecs = torch.linalg.eigh(cov)
    W = (vecs * vals.clamp_min(1e-6).rsqrt()) @ vecs.T
    return mu, basis, W


def build_protos(p, tasks, subj, rows, ids):
    protos = torch.zeros(4, N_BASE, p.shape[-1], device=p.device, dtype=p.dtype)
    valid = torch.zeros(4, N_BASE, dtype=torch.bool, device=p.device)
    for t in range(4):
        for s in ids:
            r = rows & (tasks == t) & (subj == s)
            if r.sum() >= 3:
                protos[t, s - 1] = p[r].mean(0)
                valid[t, s - 1] = True
    return protos, valid


def decide(proj, tasks, protos, valid, tau):
    n = len(proj)
    d = torch.full((n, N_BASE), float("inf"), device=proj.device)
    for t in range(4):
        mask = tasks == t
        if not bool(mask.any()):
            continue
        dd = torch.cdist(proj[mask], protos[t]).masked_fill(~valid[t][None], float("inf"))
        d[mask] = dd
    dmin = d.min(1).values
    pred = d.argmin(1) + 1
    accept = dmin <= tau
    return dmin, pred, accept


def main() -> None:
    w = torch.load(PKG / "weights.pt", map_location="cpu", weights_only=True)
    feat = torch.from_numpy(np.load(CACHE / "train_feat.npy")).float().mean(1).to(DEV)
    tr = pd.read_csv(ROOT / "BrainPrint_cache/train_metadata.csv")
    tasks = torch.from_numpy(M._task_indices([{"task": t} for t in tr["task"]], len(tr))).to(DEV)
    subj = torch.from_numpy(tr["subject"].to_numpy()).to(DEV)
    g = tr.groupby(["subject", "task"])
    frac = ((g.cumcount() + 1) / g["trial"].transform("size")).to_numpy()
    early, fwd = frac <= 0.6, frac > 0.6
    block = np.clip((frac * 5).astype(int), 0, 4)

    out = {}

    def zscore(rows):
        mu = torch.zeros(4, feat.shape[-1], device=DEV)
        sd = torch.ones(4, feat.shape[-1], device=DEV)
        for t in range(4):
            r = rows & (tasks == t)
            if r.sum() > 1:
                mu[t] = feat[r].mean(0)
                sd[t] = feat[r].std(0).clamp_min(1e-6)
        return (feat - mu[tasks]) / sd[tasks]

    # ---------- #1 in-sample
    z = zscore(torch.from_numpy(early).to(DEV))
    rows = torch.from_numpy(early).to(DEV)
    mu, basis, W = fit_head(z, rows, tasks, subj)
    p = (z - mu) @ basis
    protos, valid = build_protos(p, tasks, subj, rows, list(range(1, N_BASE + 1)))
    _, pred, _ = decide(p[rows], tasks[rows], protos, valid, float("inf"))
    out["1_train_in_sample_acc"] = float((pred == subj[rows]).float().mean())

    # ---------- #2/#3 5-block OOF
    correct, total = 0, 0
    y_true, y_pred = [], []
    for b in range(5):
        fit = torch.from_numpy(block != b).to(DEV)
        held = torch.from_numpy(block == b).to(DEV)
        mu_b, basis_b, W_b = fit_head(z, fit, tasks, subj)
        p_b = (z - mu_b) @ basis_b
        protos_b, valid_b = build_protos(p_b, tasks, subj, fit, list(range(1, N_BASE + 1)))
        dmin, pred, _ = decide(p_b[held], tasks[held], protos_b, valid_b, float("inf"))
        y_true.append(subj[held].cpu().numpy())
        y_pred.append(pred.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    out["2_oof_closed_acc"] = float((y_true == y_pred).mean())
    f1s = []
    for s in range(1, N_BASE + 1):
        tp = np.sum((y_pred == s) & (y_true == s))
        fp = np.sum((y_pred == s) & (y_true != s))
        fn = np.sum((y_pred != s) & (y_true == s))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    out["3_oof_macro_f1"] = float(np.mean(f1s))

    # ---------- #4 forward block-out
    fit = torch.from_numpy(early).to(DEV)
    ev = torch.from_numpy(fwd).to(DEV)
    mu_f, basis_f, W_f = fit_head(z, fit, tasks, subj)
    p_f = (z - mu_f) @ basis_f
    protos_f, valid_f = build_protos(p_f, tasks, subj, fit, list(range(1, N_BASE + 1)))
    _, pred_f, _ = decide(p_f[ev], tasks[ev], protos_f, valid_f, float("inf"))
    out["4_forward_block_out_acc"] = float((pred_f == subj[ev]).float().mean())

    # ---------- #5/#6/#11 strict identity LOSO
    known_scores, unknown_scores = [], []
    auc_pairs = []
    for held_id in range(1, N_BASE + 1):
        ids = [s for s in range(1, N_BASE + 1) if s != held_id]
        fit = torch.from_numpy(early).to(DEV) & (subj != held_id)
        z_f = zscore(fit)
        mu_h, basis_h, W_h = fit_head(z_f, fit, tasks, subj)
        p_h = (z_f - mu_h) @ basis_h
        protos_h, valid_h = build_protos(p_h, tasks, subj, fit, ids)
        ev_k = torch.from_numpy(frac > 0.8).to(DEV) & (subj != held_id)
        ev_u = torch.from_numpy(frac > 0.8).to(DEV) & (subj == held_id)
        dk, *_ = decide(p_h[ev_k], tasks[ev_k], protos_h, valid_h, float("inf"))
        du, *_ = decide(p_h[ev_u], tasks[ev_u], protos_h, valid_h, float("inf"))
        known_scores.append(dk.cpu().numpy())
        unknown_scores.append(du.cpu().numpy())
    ks = np.concatenate(known_scores)
    us = np.concatenate(unknown_scores)
    # AUROC of the rejection score (smaller = known)
    order = np.argsort(np.concatenate([ks, us]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    n_k, n_u = len(ks), len(us)
    # AUC = P(genuine distance < pseudo-unknown distance); smaller distance = more genuine
    auc = 1.0 - float((ranks[:n_k].sum() - n_k * (n_k + 1) / 2) / (n_k * n_u))
    out["5_strict_loso_auroc"] = auc
    out["5_diag_median_known"] = float(np.median(ks))
    out["5_diag_median_unknown"] = float(np.median(us))
    tau90 = float(np.quantile(ks, 0.90))
    tau95 = float(np.quantile(ks, 0.95))
    out["11_strict_loso_unknown_rej_at_90"] = float((us > tau90).mean())
    out["6_strict_loso_u_at_95"] = float((us > tau95).mean())
    out["strict_loso_known_accept_at_90"] = float((ks <= tau90).mean())
    out["strict_loso_known_accept_at_95"] = float((ks <= tau95).mean())

    # ---------- #9/#10 preliminary Test, already audited by verify.py
    audit = json.loads((OUT / "verify_final2.json").read_text(encoding="utf-8"))["test_audit"]
    out["9_test_registered_accept"] = audit["registered_accept_rate"]
    out["9b_test_registered_top1_given_accept"] = audit["registered_top1_given_accept"]
    out["9c_test_registered_e2e"] = audit["registered_e2e"]
    out["10_test_unregistered_rejection"] = audit["unknown_rejection"]

    print(json.dumps(out, indent=2))
    (OUT / "gates11.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
