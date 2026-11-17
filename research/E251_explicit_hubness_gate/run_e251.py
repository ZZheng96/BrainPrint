"""E251 explicit hubness refusal gate with strict nested identity LOSO."""
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
LAMBDAS = (0.0, 0.02, 0.05, 0.10)
CFGS = [f"lambda_{x:.2f}" for x in LAMBDAS]
PROTOCOL = {
    "version": "E251-v1-explicit-hubness-gate",
    "outer_fit": "known identities, frac <= 0.6",
    "inner_fit": "remaining identities, frac <= 0.4",
    "inner_calibration": "held identity unknown and remaining identities known, 0.4 < frac <= 0.6",
    "evaluation": "outer held identity and known identities, frac > 0.8",
    "identity_decision": "top-1 strict LDA-cos prototype score; lambda changes refusal score only",
    "reject_score": "top1_cosine - lambda * standardized_predicted_prototype_hubness",
    "lambdas": list(LAMBDAS),
    "hubness": "prototype mutual-nearest-neighbor in fitted identity prototype space; standardization fit on fitted identities",
    "threshold": "weighted identity-balanced FAR10 threshold from inner pseudo-unknown scores",
    "session": "not used",
    "test_data": "not read; no Test result participates in fitting or selection",
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
            p[torch.as_tensor(ix, device=DEV)] = space.proj(feats[torch.as_tensor(ix, device=DEV)], tasks[ix])
    return p


def fit_base(feats, tasks, subj, rows, known):
    fitted = R.fit_model(feats, tasks, subj, rows, known)
    p = project_all(fitted["space"], feats, tasks)
    W = R.whiten(fitted["eig"], .01)
    proto = F.normalize(fitted["protos"] @ W, dim=2)
    valid_ix = torch.nonzero(fitted["valid"], as_tuple=False).flatten()
    mean_proto = proto[:, fitted["valid"]].mean(0)
    sim = mean_proto @ mean_proto.T
    sim.fill_diagonal_(-float("inf"))
    nn = sim.argmax(1)
    hub = torch.bincount(nn, minlength=len(valid_ix)).double().cpu().numpy()
    med = float(np.median(hub)); scale = float(np.median(np.abs(hub - med)) * 1.4826)
    fitted["hub_median"] = med
    fitted["hub_scale"] = scale if scale > 1e-8 else 1.0
    return fitted, p, W


def raw_features(P, ix, fitted, W, tasks):
    x = F.normalize(P[torch.as_tensor(ix, device=DEV)] @ W, dim=1)
    proto = F.normalize(fitted["protos"] @ W, dim=2)
    tk = torch.as_tensor(tasks[ix], device=DEV)
    all_sim = torch.einsum("nd,nsd->ns", x, proto[tk])
    all_sim[:, ~fitted["valid"]] = -float("inf")
    vals, ids = torch.topk(all_sim, 2, dim=1)
    top1, top2 = vals[:, 0], vals[:, 1]
    pred = ids[:, 0] + 1
    valid_ix = torch.nonzero(fitted["valid"], as_tuple=False).flatten()
    mean_proto = proto[:, fitted["valid"]].mean(0)
    proto_sim = mean_proto @ mean_proto.T
    proto_sim.fill_diagonal_(-float("inf"))
    hub_valid = torch.bincount(proto_sim.argmax(1), minlength=len(valid_ix)).double()
    hub_all = torch.zeros(20, dtype=torch.float64, device=DEV)
    hub_all[valid_ix] = hub_valid
    hub = (hub_all[pred - 1].cpu().numpy() - fitted["hub_median"]) / fitted["hub_scale"]
    out = np.column_stack((top1.detach().cpu().numpy(), hub, (top1 - top2).detach().cpu().numpy()))
    return out.astype(np.float64), pred.detach().cpu().numpy().astype(np.int64)


def score_features(P, ix, fitted, W, tasks):
    return raw_features(P, ix, fitted, W, tasks)


def reject_score(raw, lam):
    return raw[:, 0] - lam * raw[:, 1]


def metric(score, pred, subj, rix, outer, threshold):
    rows = np.zeros(len(subj), dtype=bool); rows[rix] = True
    return R.metrics(score, pred, subj, rows, outer, threshold)


def candidate_stats(records, subj, outer):
    out = {}
    for lam in LAMBDAS:
        unknown_by_id, known_by_id, uk, kk = [], [], [], []
        for r in records:
            u = reject_score(r["unknown_raw"], lam); k = reject_score(r["known_raw"], lam)
            unknown_by_id.append(u); known_by_id.append(k); uk.append(u); kk.append(k)
        threshold = R.weighted_far_threshold(unknown_by_id, FAR)
        uall, kall = np.concatenate(uk), np.concatenate(kk)
        accepted_k = np.concatenate([k >= threshold for k in known_by_id])
        accepted_u = np.concatenate([u >= threshold for u in unknown_by_id])
        tp = int((~accepted_u).sum()); fp = int((~accepted_k).sum()); fn = int(accepted_u.sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        out[lam] = {"threshold": float(threshold), "known_acceptance": float(accepted_k.mean()),
                    "unknown_rejection": float((~accepted_u).mean()), "AUC": R.auc(kall, uall),
                    "F1_unknown_rejection": float(f1), "identity_acceptance_min": float(min((v >= threshold).mean() for v in known_by_id))}
    return out


def choose_lambda(stats):
    base = stats[0.0]
    eligible = []
    for lam in LAMBDAS:
        s = stats[lam]
        if (s["unknown_rejection"] >= base["unknown_rejection"] - 1e-12 and
            s["known_acceptance"] >= base["known_acceptance"] - .01 - 1e-12 and
            s["AUC"] >= base["AUC"] - 1e-12 and s["F1_unknown_rejection"] >= base["F1_unknown_rejection"] - 1e-12 and
            s["identity_acceptance_min"] >= base["identity_acceptance_min"] - 1e-12):
            eligible.append(lam)
    return max(eligible, key=lambda x: (stats[x]["known_acceptance"], -x)) if eligible else 0.0


def run_fold(feats, tasks, subj, frac, outer):
    plan = outer_plan(subj, frac, outer); validate_plan(subj, frac, outer, plan)
    inner = []
    for inn in plan["inner"]:
        fitted, P, W = fit_base(feats, tasks, subj, inn["fit"], inn["remain"])
        iu, pu = score_features(P, np.flatnonzero(inn["cal_unknown"]), fitted, W, tasks)
        ik, pk = score_features(P, np.flatnonzero(inn["cal_known"]), fitted, W, tasks)
        inner.append({"unknown_raw": iu, "known_raw": ik, "unknown_pred": pu, "known_pred": pk})
    inner_stats = candidate_stats(inner, subj, outer)
    chosen = choose_lambda(inner_stats)
    fitted, P, W = fit_base(feats, tasks, subj, plan["fit"], plan["known"])
    ix = np.flatnonzero(plan["late"])
    raw, pred = score_features(P, ix, fitted, W, tasks)
    rows = []
    for lam in LAMBDAS:
        score = reject_score(raw, lam)
        m = metric(score, pred, subj, ix, outer, inner_stats[lam]["threshold"])
        rows.append({"outer_subject": outer, "candidate": f"lambda_{lam:.2f}", "lambda": lam,
                     "selected": bool(lam == chosen), "endpoint": "P1", "threshold_value": inner_stats[lam]["threshold"], **m,
                     "inner_known_acceptance": inner_stats[lam]["known_acceptance"]})
    fold_dir = RES / f"outer_{outer:02d}"; fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"protocol": PROTOCOL, "outer_subject": outer, "known": plan["known"].tolist(),
                "fitted": R.cpu_model(fitted), "hub_median": fitted["hub_median"], "hub_scale": fitted["hub_scale"],
                "thresholds": {str(k): v["threshold"] for k, v in inner_stats.items()}, "inner_stats": inner_stats, "selected_lambda": chosen}, fold_dir / "model.pt")
    saved = torch.load(fold_dir / "model.pt", map_location="cpu", weights_only=False)
    f2 = R.device_model(saved["fitted"]); f2["hub_median"] = saved["hub_median"]; f2["hub_scale"] = saved["hub_scale"]
    p2 = project_all(f2["space"], feats, tasks)
    raw2, pred2 = score_features(p2, ix, f2, W, tasks)
    for lam in LAMBDAS:
        assert np.allclose(reject_score(raw, lam), reject_score(raw2, lam), rtol=1e-5, atol=1e-6)
    assert len(pred) == len(pred2)
    for sample_idx, (identity_before, identity_after) in enumerate(zip(pred, pred2)):
        assert int(identity_before) == int(identity_after), f"top1 identity changed at sample {sample_idx}"
    top1_by_lambda = [pred.copy() for _ in LAMBDAS]
    for identity_by_lambda in top1_by_lambda[1:]:
        for sample_idx, (identity_before, identity_after) in enumerate(zip(top1_by_lambda[0], identity_by_lambda)):
            assert int(identity_before) == int(identity_after), f"top1 identity changed at sample {sample_idx}"
    split = {"protocol_version": PROTOCOL["version"], "outer_subject": outer, "known": plan["known"].tolist(),
             "outer_fit_rows": int(plan["fit"].sum()), "late_rows": int(plan["late"].sum()), "selected_lambda": chosen,
             "top1_identity_unchanged_per_sample": True, "reload_check": "passed",
             "metadata_sha256": hashlib.sha256(R.META_PATH.read_bytes()).hexdigest(), "feature_sha256": hashlib.sha256(R.FEATURE_PATH.read_bytes()).hexdigest()}
    (fold_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (fold_dir / "results.json").write_text(json.dumps({"rows": rows, "inner_stats": {str(k): v for k, v in inner_stats.items()}}, indent=2), encoding="utf-8")
    return rows


def write_outputs(rows):
    RES.mkdir(exist_ok=True); df = pd.DataFrame(rows); df.to_csv(RES / "per_fold.csv", index=False)
    summary = df.groupby("candidate", as_index=False).agg({"AUC": "mean", "F1_unknown_rejection": "mean", "known_acceptance": "mean", "unknown_rejection": "mean", "known_identification_accuracy": "mean"})
    summary.to_csv(RES / "summary.csv", index=False)
    summary.to_json(RES / "summary.json", orient="records", indent=2)
    (RES / "selection.json").write_text(json.dumps({"protocol": PROTOCOL, "summary": summary.to_dict("records")}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--validate-only", action="store_true"); ap.add_argument("--all-folds", action="store_true"); ap.add_argument("--outer-subject", type=int, nargs="+"); ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(); meta = pd.read_csv(R.META_PATH)
    feats = torch.as_tensor(np.array(np.load(R.FEATURE_PATH, mmap_mode="r"), dtype=np.float32, copy=True), device=DEV)
    R.validate(meta, feats); _, frac = R.temporal_blocks(meta); subj = meta.subject.to_numpy(); tasks = R.M._task_indices([{"task": t} for t in meta.task], len(meta))
    if args.validate_only:
        for outer in range(1, 21): validate_plan(subj, frac, outer, outer_plan(subj, frac, outer))
        print(json.dumps({"validate": "passed", "folds": 20, "lambdas": LAMBDAS, "device": str(DEV)})); return
    folds = list(range(1, 21)) if args.all_folds else (args.outer_subject or [])
    if not folds: raise SystemExit("choose --validate-only, --outer-subject N [N..], or --all-folds")
    rows = []; t0 = time.time()
    for outer in folds:
        fd = RES / f"outer_{outer:02d}"
        if args.resume and (fd / "results.json").exists(): rows += json.loads((fd / "results.json").read_text(encoding="utf-8"))["rows"]; continue
        rows += run_fold(feats, tasks, subj, frac, outer); print(f"outer={outer} seconds={time.time()-t0:.1f}", flush=True)
    write_outputs(rows)


if __name__ == "__main__": main()
