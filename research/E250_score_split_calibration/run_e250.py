"""E250-S1: train-only nested score/refusal calibration with strict identity LOSO."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RES = OUT / "results"
sys.path.insert(0, str(ROOT / "research" / "E247_train_temporal_loso"))
import run_e247 as R  # noqa: E402

DEV, DIM = R.DEV, R.DIM
FAR = 0.10
CFGS = ["C0", "C1_margin", "C2_margin_cohort_hubness"]
PROTOCOL = {
    "version": "E250-S1-v1-score-split-calibration",
    "outer_fit": "known identities, frac <= 0.6",
    "inner_fit": "remaining identities, frac <= 0.4",
    "inner_calibration": "held identity unknown and remaining identities known, 0.4 < frac <= 0.6",
    "evaluation": "outer held identity and known identities, frac > 0.8",
    "identity_decision": "top-1 strict LDA-cos prototype score; calibration never changes ranking",
    "C0": "raw top-1 max-cos score, E247 strict LDA-cos shrink 0.01",
    "C1_margin": "L2 logistic calibration of top-1 and top-1 minus top-2 margin",
    "C2_margin_cohort_hubness": "L2 logistic calibration of top-1, margin, cohort mean/std and predicted-prototype hubness",
    "standardization": "median/MAD fitted only on pooled inner calibration features",
    "logistic": "PyTorch L2 logistic, labels known=1 and pseudo-unknown=0",
    "threshold": "weighted identity-balanced FAR10 threshold from pseudo-unknown scores",
    "session": "not used; only the metadata session validation in E247 is retained",
    "test_data": "not read; no Test result participates in fitting or selection",
    "attack_logic": "not_used; E249 gates_attack.py is not imported",
}
FEATURE_NAMES = {
    "C0": ["top1"],
    "C1_margin": ["top1", "margin"],
    "C2_margin_cohort_hubness": ["top1", "margin", "cohort_mean", "cohort_std", "pred_hubness"],
}


def outer_plan(subj, frac, outer):
    known = np.arange(1, 21)[np.arange(1, 21) != outer]
    fit = np.isin(subj, known) & (frac <= .6)
    plan = {"known": known, "fit": fit, "late": frac > .8, "inner": []}
    for held in known:
        remain = known[known != held]
        ifit = np.isin(subj, remain) & (frac <= .4)
        plan["inner"].append({"held": int(held), "remain": remain, "fit": ifit,
                              "cal_unknown": (subj == held) & (frac > .4) & (frac <= .6),
                              "cal_known": np.isin(subj, remain) & (frac > .4) & (frac <= .6)})
    return plan


def validate_plan(subj, frac, outer, plan):
    assert not plan["fit"][subj == outer].any()
    assert not (plan["fit"] & (frac > .6)).any()
    assert not (plan["fit"] & plan["late"]).any()
    for inn in plan["inner"]:
        held = inn["held"]
        assert not inn["fit"][(subj == held) | (subj == outer)].any()
        assert not (inn["fit"] & (frac > .4)).any()
        assert not (inn["cal_unknown"] & inn["fit"]).any()
        assert not (inn["cal_known"] & inn["fit"]).any()
        assert not inn["cal_unknown"][subj != held].any()
        assert not inn["cal_known"][(subj == held) | (subj == outer)].any()
        assert not ((inn["cal_unknown"] | inn["cal_known"]) & (frac > .6)).any()


def project_all(space, feats, tasks):
    p = torch.empty(len(tasks), DIM, device=DEV)
    for t in range(4):
        ix = np.flatnonzero(tasks == t)
        if len(ix):
            ti = torch.as_tensor(ix, device=DEV)
            p[ti] = space.proj(feats[ti], tasks[ix])
    return p


def fit_base(feats, tasks, subj, rows, known):
    fitted = R.fit_model(feats, tasks, subj, rows, known)
    p = project_all(fitted["space"], feats, tasks)
    return fitted, p, R.whiten(fitted["eig"], .01)


def raw_features(P, ix, fitted, W):
    x = F.normalize(P[torch.as_tensor(ix, device=DEV)] @ W, dim=1)
    proto = F.normalize(fitted["protos"] @ W, dim=2)
    tk = torch.as_tensor(fitted["_tasks"][ix], device=DEV)
    all_sim = torch.einsum("nd,nsd->ns", x, proto[tk])
    all_sim[:, ~fitted["valid"]] = -float("inf")
    vals, ids = torch.topk(all_sim, 2, dim=1)
    top1, top2 = vals[:, 0], vals[:, 1]
    pred = ids[:, 0] + 1
    valid = fitted["valid"]
    cohort = all_sim[:, valid]
    finite = torch.isfinite(cohort)
    cmean = (torch.where(finite, cohort, 0).sum(1) / finite.sum(1).clamp_min(1)).double()
    cstd = torch.where(finite, cohort, cmean[:, None]).double().std(1, unbiased=False)
    proto_mean = proto.mean(0)
    proto_sim = proto_mean @ proto_mean.T
    proto_sim.fill_diagonal_(-float("inf"))
    hub = torch.bincount(proto_sim.argmax(1), minlength=20).to(DEV)[pred - 1].double()
    out = torch.stack([top1.double(), (top1 - top2).double(), cmean, cstd, hub], 1)
    return out.detach().cpu().numpy(), pred.detach().cpu().numpy().astype(np.int64)


def score_features(P, ix, fitted, W, tasks):
    fitted = dict(fitted)
    fitted["_tasks"] = tasks
    return raw_features(P, ix, fitted, W)


def standardize_fit(x):
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0) * 1.4826
    scale = np.where(mad > 1e-8, mad, 1.0)
    return med, scale


def logistic_fit(x, y, l2=1.0):
    med, scale = standardize_fit(x)
    z = torch.tensor((x - med) / scale, dtype=torch.float64, device=DEV)
    yy = torch.tensor(y, dtype=torch.float64, device=DEV)
    w = torch.zeros(z.shape[1], dtype=torch.float64, device=DEV, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=.5, max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z @ w + b, yy) + .5 * l2 * (w @ w) / max(len(x), 1)
        loss.backward()
        return loss
    opt.step(closure)
    return {"median": med.tolist(), "scale": scale.tolist(), "weight": w.detach().cpu().numpy().tolist(), "bias": float(b.detach().cpu())}


def calibrated(model, x):
    z = (x - np.asarray(model["median"])) / np.asarray(model["scale"])
    return (1 / (1 + np.exp(-(z @ np.asarray(model["weight"]) + model["bias"])))) .astype(np.float64)


def fit_calibrators(inner_records):
    models, thresholds, selection = {}, {}, {}
    for cfg in CFGS:
        cols = [FEATURE_NAMES[cfg].index(n) for n in FEATURE_NAMES[cfg]]
        known = np.concatenate([r[cfg]["known"][:, cols] for r in inner_records])
        unknown = np.concatenate([r[cfg]["unknown"][:, cols] for r in inner_records])
        if cfg == "C0":
            model = {"kind": "identity", "columns": cols}
            uk = unknown[:, 0]
            kk = known[:, 0]
        else:
            model = {"kind": "logistic", "columns": cols}
            model.update(logistic_fit(np.vstack([known, unknown]), np.r_[np.ones(len(known)), np.zeros(len(unknown))]))
            uk, kk = calibrated(model, unknown), calibrated(model, known)
        models[cfg] = model
        # Keep the inner held-identity vectors separate: weighted_far_threshold
        # must give each pseudo-unknown identity equal mass.
        unknown_by_identity = []
        known_by_identity = []
        for r in inner_records:
            u = r[cfg]["unknown"][:, cols]
            k = r[cfg]["known"][:, cols]
            unknown_by_identity.append(u[:, 0] if cfg == "C0" else calibrated(model, u))
            known_by_identity.append(k[:, 0] if cfg == "C0" else calibrated(model, k))
        thresholds[cfg] = float(R.weighted_far_threshold(unknown_by_identity, FAR))
        selection[cfg] = {"inner_known_acceptance": float(np.mean([float((v >= thresholds[cfg]).mean()) for v in known_by_identity])), "unknown_count": len(uk), "known_count": len(kk)}
    return models, thresholds, selection


def apply_model(model, x):
    cols = model["columns"]
    x = x[:, cols]
    return x[:, 0] if model["kind"] == "identity" else calibrated(model, x)


def metric(score, pred, subj, rix, outer, threshold):
    rows = np.zeros(len(subj), dtype=bool); rows[rix] = True
    return R.metrics(score, pred, subj, rows, outer, threshold)


def run_fold(feats, tasks, subj, frac, outer):
    plan = outer_plan(subj, frac, outer); validate_plan(subj, frac, outer, plan)
    inner = []
    for inn in plan["inner"]:
        fitted, P, W = fit_base(feats, tasks, subj, inn["fit"], inn["remain"])
        fitted["_tasks"] = tasks
        iu, pu = score_features(P, np.flatnonzero(inn["cal_unknown"]), fitted, W, tasks)
        ik, pk = score_features(P, np.flatnonzero(inn["cal_known"]), fitted, W, tasks)
        inner.append({c: {"unknown": iu, "known": ik} for c in CFGS})
    models, thresholds, selection = fit_calibrators(inner)
    fitted, P, W = fit_base(feats, tasks, subj, plan["fit"], plan["known"])
    ix = np.flatnonzero(plan["late"])
    feats_raw, pred = score_features(P, ix, fitted, W, tasks)
    rows = []
    for cfg in CFGS:
        score = apply_model(models[cfg], feats_raw)
        m = metric(score, pred, subj, ix, outer, thresholds[cfg])
        rows.append({"outer_subject": outer, "candidate": cfg, "endpoint": "P1", "threshold_value": thresholds[cfg], **m,
                     "inner_known_acceptance": selection[cfg]["inner_known_acceptance"]})
    fold_dir = RES / f"outer_{outer:02d}"; fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"protocol": PROTOCOL, "outer_subject": outer, "known": plan["known"].tolist(),
                "fitted": R.cpu_model(fitted), "calibrators": models, "thresholds": thresholds,
                "selection": selection}, fold_dir / "model.pt")
    saved = torch.load(fold_dir / "model.pt", map_location="cpu", weights_only=False)
    f2 = R.device_model(saved["fitted"]); p2 = project_all(f2["space"], feats, tasks)
    raw2, pred2 = score_features(p2, ix, f2, W, tasks)
    for cfg in CFGS:
        assert np.allclose(apply_model(models[cfg], feats_raw), apply_model(saved["calibrators"][cfg], raw2), rtol=1e-5, atol=1e-6)
    assert np.array_equal(pred, pred2)
    split = {"protocol_version": PROTOCOL["version"], "outer_subject": outer, "known": plan["known"].tolist(),
             "outer_fit_rows": int(plan["fit"].sum()), "late_rows": int(plan["late"].sum()), "reload_check": "passed",
             "metadata_sha256": hashlib.sha256(R.META_PATH.read_bytes()).hexdigest(),
             "feature_sha256": hashlib.sha256(R.FEATURE_PATH.read_bytes()).hexdigest()}
    (fold_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (fold_dir / "results.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    return rows


def write_outputs(rows):
    RES.mkdir(exist_ok=True); df = pd.DataFrame(rows); df.to_csv(RES / "per_fold.csv", index=False)
    summary = df.groupby("candidate", as_index=False).agg({"AUC": "mean", "F1_unknown_rejection": "mean", "known_acceptance": "mean", "unknown_rejection": "mean", "known_identification_accuracy": "mean"})
    summary.to_csv(RES / "summary.csv", index=False)
    (RES / "selection.json").write_text(json.dumps({"protocol": PROTOCOL, "summary": summary.to_dict("records")}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--validate-only", action="store_true"); ap.add_argument("--all-folds", action="store_true"); ap.add_argument("--outer-subject", type=int, nargs="+"); ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(); meta = pd.read_csv(R.META_PATH)
    feats = torch.as_tensor(np.array(np.load(R.FEATURE_PATH, mmap_mode="r"), dtype=np.float32, copy=True), device=DEV)
    R.validate(meta, feats); _, frac = R.temporal_blocks(meta); subj = meta.subject.to_numpy(); tasks = R.M._task_indices([{"task": t} for t in meta.task], len(meta))
    if args.validate_only:
        for outer in range(1, 21): validate_plan(subj, frac, outer, outer_plan(subj, frac, outer))
        print(json.dumps({"validate": "passed", "folds": 20, "candidates": CFGS, "device": str(DEV)})); return
    folds = list(range(1, 21)) if args.all_folds else (args.outer_subject or [])
    if not folds: raise SystemExit("choose --validate-only, --outer-subject N [N..], or --all-folds")
    rows = []; t0 = time.time()
    for outer in folds:
        fd = RES / f"outer_{outer:02d}"
        if args.resume and (fd / "results.json").exists(): rows += json.loads((fd / "results.json").read_text(encoding="utf-8"))["rows"]; continue
        rows += run_fold(feats, tasks, subj, frac, outer); print(f"outer={outer} seconds={time.time()-t0:.1f}", flush=True)
    write_outputs(rows)


if __name__ == "__main__": main()
