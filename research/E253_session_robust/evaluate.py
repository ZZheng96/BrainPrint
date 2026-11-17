"""E253 step 2: score-level variants on top of the frozen E200 feature pipeline.

Everything is fitted on Train only.  The Test labels are read once for the final
descriptive audit; no threshold, weight or prototype is fitted on Test.

Variants share the same 96-d PCA space and the same mean-projection prototypes,
so the comparison isolates the scoring rule:
  V0_mean    mean of the 6 sub-window Euclidean distances (current submission)
  V1_median  per-identity median over the 6 sub-window distances
  V2_cos     cosine distance of the mean projection
  V3_cos_med per-identity median cosine distance
  V4_trim    drop the single worst sub-window, mean of the remaining five
  V5_rank    rank fusion of V0 and V1, rejection = V0 accept AND V1 accept

Thresholds are re-calibrated per variant with 5-block out-of-fold genuine
distances at the same 90% operating point as the frozen model.
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
import model as M  # noqa: E402

N_BASE = 20
VARIANTS = ("V0_mean", "V1_median", "V2_cos", "V3_cos_med", "V4_trim")


def load_artifacts():
    w = torch.load(PKG / "weights.pt", map_location="cpu", weights_only=True)
    return {"task_mu": w["task_mu"].float(), "task_sd": w["task_sd"].float(),
            "pca_mean": w["pca_mean"].float(), "pca_basis": w["pca_basis"].float(),
            "prototypes": w["prototypes"].float(), "tau": float(w["tau_dist"])}


def project(art, feat_sw: np.ndarray, tasks: np.ndarray) -> torch.Tensor:
    """(n, 6, 855) -> (n, 6, 96) under the frozen per-task standardisation + PCA."""
    t = torch.from_numpy(np.ascontiguousarray(feat_sw))
    z = (t - art["task_mu"][tasks][:, None, :]) / art["task_sd"][tasks][:, None, :]
    return (z - art["pca_mean"]) @ art["pca_basis"]


def sqdist(proj: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
    """(n, 6, 96) x (20, 96) -> (n, 6, 20) Euclidean distances."""
    d2 = (proj * proj).sum(-1).unsqueeze(-1) + (proto * proto).sum(-1).view(1, 1, -1) \
        - 2.0 * (proj @ proto.T)
    return d2.clamp_min(0).sqrt()


def cosine(proj: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
    m = proj / proj.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    p = proto / proto.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return -(m @ p.T)


def distances(proj: torch.Tensor, proto: torch.Tensor, variant: str) -> torch.Tensor:
    if variant == "V0_mean":
        return sqdist(proj, proto).mean(1)
    if variant == "V1_median":
        return sqdist(proj, proto).median(1).values
    if variant == "V4_trim":
        d = sqdist(proj, proto).sort(dim=1).values
        return d[:, :-1].mean(1)
    if variant == "V2_cos":
        return cosine(proj.mean(1), proto)
    if variant == "V3_cos_med":
        return cosine(proj, proto).median(1).values
    raise ValueError(variant)


def score_matrix(art, feat_sw, tasks, variant, proj_all=None) -> torch.Tensor:
    n = len(feat_sw)
    out = torch.full((n, N_BASE), float("inf"))
    if proj_all is None:
        proj_all = project(art, feat_sw, tasks)
    for t in range(4):
        rows = np.nonzero(tasks == t)[0]
        if rows.size == 0:
            continue
        out[rows] = distances(proj_all[rows], art["prototypes"][t], variant)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    art = load_artifacts()
    train_feat = np.load(CACHE / "train_feat.npy")
    test_feat = np.load(CACHE / "test_feat.npy")
    tr = pd.read_csv(ROOT / "BrainPrint_cache/train_metadata.csv")
    te = pd.read_csv(ROOT / "BrainPrint_cache/test_metadata.csv")
    labels = pd.read_csv(ROOT / "Brainprint_Competition/test.csv")
    te = te.merge(labels, left_on="sample", right_on="Sample", how="left")
    tr_tasks = M._task_indices([{"task": t} for t in tr["task"]], len(tr))
    te_tasks = M._task_indices([{"task": t} for t in te["task"]], len(te))
    tr_subj = tr["subject"].to_numpy()
    g = tr.groupby(["subject", "task"])
    frac = ((g.cumcount() + 1) / g["trial"].transform("size")).to_numpy()
    te_label = te["Label"].to_numpy()
    session = te["session"].to_numpy()
    block = np.clip((frac * 5).astype(int), 0, 4)

    report = {"frozen_tau": art["tau"], "variants": {}}
    # ---- Train: 5-block OOF, prototypes refitted on the 4 remaining blocks
    for variant in VARIANTS:
        d_gen, correct, total = [], 0, 0
        for b in range(5):
            fit, held = block != b, block == b
            protos = art["prototypes"].clone()
            for t in range(4):
                for s in range(1, N_BASE + 1):
                    rows = fit & (tr_tasks == t) & (tr_subj == s)
                    if rows.sum() >= 3:
                        protos[t, s - 1] = project(art, train_feat[rows],
                                                   np.full(int(rows.sum()), t)).mean((0, 1))
            local = dict(art, prototypes=protos)
            sc = score_matrix(local, train_feat[held], tr_tasks[held], variant)
            truth = tr_subj[held]
            correct += int((sc.argmin(1).numpy() + 1 == truth).sum())
            total += len(truth)
            d_gen.append(sc[np.arange(len(truth)), truth - 1].numpy())
        d_gen = np.concatenate(d_gen)
        tau = float(np.quantile(d_gen, 0.90))
        entry = {"tau_oof90": tau, "train_oof_closed_acc": correct / total,
                 "train_genuine_accept_at_tau": float((d_gen <= tau).mean())}
        report["variants"][variant] = entry
        print(variant, "tau", round(tau, 4), "oof_acc", round(correct / total, 4), flush=True)

    # ---- Test: single descriptive audit (projection computed once)
    test_proj = project(art, test_feat, te_tasks)
    test_scores = {v: score_matrix(art, test_feat, te_tasks, v, proj_all=test_proj)
                   for v in VARIANTS}
    for variant, sc in test_scores.items():
        entry = report["variants"][variant]
        tau = entry["tau_oof90"]
        pred = sc.argmin(1).numpy() + 1
        dmin = sc.min(1).values.numpy()
        acc_mask = dmin <= tau
        for sess in ("session1", "session2"):
            m = session == sess
            am = m & acc_mask
            entry[f"test_{sess}"] = {
                "n": int(m.sum()),
                "top1_acc_no_reject": float((pred[m] == te_label[m]).mean()),
                "accept_rate": float(acc_mask[m].mean()),
                "acc_given_accept": float((pred[am] == te_label[am]).mean()) if am.any() else None,
                "end_to_end_acc": float(((pred == te_label) & acc_mask)[m].mean())}
        for lvl in (0.85, 0.70, 0.55):
            thr = np.quantile(dmin, lvl)
            a = dmin <= thr
            entry[f"test_top1_acc_at_accept{int(lvl * 100)}"] = float(
                (pred[a] == te_label[a]).mean())

    # ---- V5 rank fusion: accept only if both V0 and V1 accept
    entry = {"note": "rejection = V0 accept AND V1 accept; prediction = V0"}
    sc0 = test_scores["V0_mean"]
    sc1 = test_scores["V1_median"]
    t0 = report["variants"]["V0_mean"]["tau_oof90"]
    t1 = report["variants"]["V1_median"]["tau_oof90"]
    pred = sc0.argmin(1).numpy() + 1
    acc_mask = (sc0.min(1).values.numpy() <= t0) & (sc1.min(1).values.numpy() <= t1)
    for sess in ("session1", "session2"):
        m = session == sess
        am = m & acc_mask
        entry[f"test_{sess}"] = {
            "n": int(m.sum()),
            "top1_acc_no_reject": float((pred[m] == te_label[m]).mean()),
            "accept_rate": float(acc_mask[m].mean()),
            "acc_given_accept": float((pred[am] == te_label[am]).mean()) if am.any() else None,
            "end_to_end_acc": float(((pred == te_label) & acc_mask)[m].mean())}
    report["variants"]["V5_rank"] = entry

    (OUT / "variants.json").write_text(json.dumps(report, indent=2, default=float),
                                       encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk.startswith("test_session")}
                      for k, v in report["variants"].items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
