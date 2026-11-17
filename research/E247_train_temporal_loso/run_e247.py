"""E247: train-only trial-order temporal nested subject LOSO."""
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

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CACHE = ROOT / "BrainPrint_cache"
MODEL_DIR = ROOT / "final" / "team_hymn"
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DIM = 96
FARS = (0.01, 0.05, 0.10)
PROTOCOL_VERSION = "E247-v2-weighted-far-temporal-bounds"
FEATURE_PATH = CACHE / "e201_final_selftest" / "train_raw_scores_inputs.npy"
META_PATH = CACHE / "train_metadata.csv"

sys.path.insert(0, str(MODEL_DIR))
import model as M  # noqa: E402


def auc(known, unknown):
    known, unknown = np.asarray(known), np.asarray(unknown)
    if not len(known) or not len(unknown):
        return float("nan")
    x = np.concatenate([known, unknown])
    ranks = pd.Series(x).rank(method="average").to_numpy()
    return float((ranks[:len(known)].sum() - len(known) * (len(known) + 1) / 2) / (len(known) * len(unknown)))


def weighted_far_threshold(values_by_subject, far):
    """Return a tie-safe threshold whose accepted weighted tail is <= far."""
    if not values_by_subject:
        return float("nan")
    n_total = sum(len(v) for v in values_by_subject)
    if not n_total:
        return float("nan")
    n_subjects = len(values_by_subject)
    entries = [(float(x), 1.0 / (n_subjects * len(v))) for v in values_by_subject for x in v]
    entries.sort(key=lambda z: z[0])
    unique = np.unique(np.asarray([x for x, _ in entries], dtype=np.float64))
    weights = np.asarray([w for _, w in entries])
    values = np.asarray([x for x, _ in entries])
    cumulative = np.cumsum(weights)
    ends = np.searchsorted(values, unique, side="right") - 1
    tails = cumulative[-1] - cumulative[ends]
    eligible = np.flatnonzero(tails <= far + 1e-15)
    if len(eligible):
        return float(np.nextafter(unique[eligible[0]], np.inf))
    return float("inf")


class Space:
    def __init__(self, feats, tasks, rows, art=None):
        if art is not None:
            self.mu = art["task_mu"].to(DEV)
            self.sd = art["task_sd"].to(DEV)
            self.pm = art["pca_mean"].to(DEV)
            self.pb = art["pca_basis"][:, :DIM].to(DEV)
            return
        mu, sd = [], []
        for t in range(4):
            ix = np.flatnonzero(rows & (tasks == t))
            x = feats[torch.as_tensor(ix, device=DEV)]
            assert len(ix), f"empty fit task {t}"
            mu.append(x.mean(0))
            sd.append(x.std(0).clamp_min(1e-6))
        self.mu, self.sd = torch.stack(mu), torch.stack(sd)
        ix = np.flatnonzero(rows)
        z = self.z(feats[torch.as_tensor(ix, device=DEV)], tasks[ix])
        self.pm = z.mean(0)
        _, _, vh = torch.linalg.svd(z - self.pm, full_matrices=False)
        self.pb = vh[:DIM].T.contiguous()

    def z(self, x, task):
        out = torch.empty_like(x)
        for t in range(4):
            m = task == t
            if m.any():
                out[torch.as_tensor(np.flatnonzero(m), device=DEV)] = (x[m] - self.mu[t]) / self.sd[t]
        return out

    def proj(self, x, task):
        return (self.z(x, task) - self.pm) @ self.pb


def cpu_model(fitted):
    space = fitted["space"]
    return {
        "space": {k: getattr(space, k).detach().cpu() for k in ("mu", "sd", "pm", "pb")},
        "protos": fitted["protos"].detach().cpu(),
        "valid": fitted["valid"].detach().cpu(),
        "eig": tuple(x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in fitted["eig"]),
    }


def device_model(saved):
    s = saved["space"]
    space = Space.__new__(Space)
    for key, value in s.items():
        setattr(space, key, value.to(DEV))
    return {"space": space, "protos": saved["protos"].to(DEV), "valid": saved["valid"].to(DEV),
            "eig": tuple(x.to(DEV) if isinstance(x, torch.Tensor) else x for x in saved["eig"])}


def fit_model(feats, tasks, subjects, rows, known, art=None):
    space = Space(feats, tasks, rows, art=art)
    protos = torch.zeros(4, 20, DIM, device=DEV)
    valid = torch.zeros(20, dtype=torch.bool, device=DEV)
    residuals = []
    for t in range(4):
        for s in known:
            ix = np.flatnonzero(rows & (subjects == s) & (tasks == t))
            assert len(ix), f"empty prototype fit subject={s}, task={t}"
            p = space.proj(feats[torch.as_tensor(ix, device=DEV)], tasks[ix])
            mu = p.mean(0)
            protos[t, s - 1] = mu
            valid[s - 1] = True
            residuals.append(p - mu)
    r = torch.cat(residuals).double()
    cov = r.T @ r / max(len(r) - 1, 1)
    w, v = torch.linalg.eigh(cov)
    return {"space": space, "protos": protos, "valid": valid,
            "eig": (w.float(), v.float(), float(w.mean()))}


def whiten(eig, lam=0.01):
    w, v, tr = eig
    return (v / ((1 - lam) * w + lam * tr).clamp_min(1e-8).sqrt()).float()


def score(feats, tasks, rows, fitted, kind):
    ix = np.flatnonzero(rows)
    p = torch.empty(len(ix), DIM, device=DEV)
    for t in range(4):
        m = tasks[ix] == t
        if m.any():
            p[torch.as_tensor(np.flatnonzero(m), device=DEV)] = fitted["space"].proj(feats[torch.as_tensor(ix[m], device=DEV)], tasks[ix[m]])
    mat = torch.full((len(ix), 20), -float("inf"), device=DEV)
    if kind == "strict_ldacos_shrink001":
        w = whiten(fitted["eig"])
        p = torch.nn.functional.normalize(p @ w, dim=1)
        proto = torch.nn.functional.normalize(fitted["protos"] @ w, dim=2)
        for t in range(4):
            m = torch.as_tensor(tasks[ix] == t, device=DEV)
            if m.any():
                mat[m] = p[m] @ proto[t].T
    else:
        for t in range(4):
            m = torch.as_tensor(tasks[ix] == t, device=DEV)
            if m.any():
                mat[m] = -torch.cdist(p[m], fitted["protos"][t])
    mat[:, ~fitted["valid"]] = -float("inf")
    mx, who = mat.max(1)
    return mx.detach().cpu().numpy().astype(np.float64), (who.detach().cpu().numpy() + 1).astype(np.int64), ix


def temporal_blocks(meta):
    meta = meta.copy()
    meta["order"] = meta.groupby(["subject", "task"], sort=False)["trial"].rank(method="first")
    meta["n_group"] = meta.groupby(["subject", "task"], sort=False)["trial"].transform("count")
    frac = (meta["order"] / meta["n_group"]).to_numpy()
    meta["block"] = np.select([frac <= .6, frac <= .8], ["early", "middle"], default="late")
    return meta, frac


def thresholds(feats, tasks, subj, frac, known):
    out, calibration = {}, {}
    calibration = {"strict_euclid": {}, "strict_ldacos_shrink001": {}}
    vals_by_kind = {kind: {far: [] for far in FARS} for kind in calibration}
    for held in known:
        remain = known[known != held]
        fit_rows = np.isin(subj, remain) & (frac <= .4)
        assert not fit_rows[subj == held].any()
        inner = fit_model(feats, tasks, subj, fit_rows, remain)
        cal_rows = (subj == held) & (frac > .4) & (frac <= .6)
        assert not cal_rows[fit_rows].any()
        for kind in calibration:
            vals = score(feats, tasks, cal_rows, inner, kind)[0]
            calibration[kind][str(int(held))] = vals
            for far in FARS:
                vals_by_kind[kind][far].append(vals)
    out = {kind: {f"far_{int(far * 100)}": weighted_far_threshold(vals_by_kind[kind][far], far)
                  for far in FARS} for kind in calibration}
    return out, calibration


def metrics(score_vals, pred_id, subjects, rows, outer, threshold):
    ix = np.flatnonzero(rows)
    ysub = subjects[ix]
    unknown = ysub == outer
    accepted = score_vals >= threshold
    pred = np.where(accepted, pred_id, 0)
    known = ~unknown
    correct = pred[known] == ysub[known]
    known_acceptance = float(accepted[known].mean())
    unknown_rejection = float((~accepted[unknown]).mean())
    reject_tp = int((~accepted[unknown]).sum())
    reject_fp = int((~accepted[known]).sum())
    reject_fn = int(accepted[unknown].sum())
    f1 = 2 * reject_tp / max(2 * reject_tp + reject_fp + reject_fn, 1)
    return {
        "AUC": auc(score_vals[known], score_vals[unknown]),
        "F1_unknown_rejection": f1,
        "known_acceptance": known_acceptance,
        "unknown_rejection": unknown_rejection,
        "known_false_rejection": 1.0 - known_acceptance,
        "known_identification_accuracy": float(correct.mean()),
        "accepted_correct_accuracy": float(correct[accepted[known]].mean()) if accepted[known].any() else float("nan"),
        "actual_FAR": float(accepted[unknown].mean()),
        "accepted_count": int(accepted.sum()),
        "known_count": int(known.sum()),
        "unknown_count": int(unknown.sum()),
    }


def validate(meta, feats):
    assert len(meta) == len(feats), (len(meta), len(feats))
    assert set(meta.session.astype(str)) == {"session1"}
    assert meta[["subject", "task", "trial"]].duplicated().sum() == 0, "duplicate subject-task-trial rows"
    assert meta.subject.nunique() == 20 and set(meta.subject) == set(range(1, 21))
    for _, group in meta.groupby(["subject", "task"]):
        assert group.trial.is_monotonic_increasing, "metadata is not trial ordered"
    blocks, frac = temporal_blocks(meta)
    assert set(blocks.block) == {"early", "middle", "late"}
    masks = {"early": frac <= .6, "middle": (frac > .6) & (frac <= .8), "late": frac > .8}
    assert not (masks["early"] & masks["middle"]).any()
    assert not (masks["early"] & masks["late"]).any()
    assert not (masks["middle"] & masks["late"]).any()
    assert np.all(masks["early"] | masks["middle"] | masks["late"])
    assert bool(torch.isfinite(feats).all())
    print(json.dumps({"rows": len(meta), "feature_shape": list(feats.shape), "subjects": sorted(meta.subject.unique().tolist()),
                      "session": "session1", "blocks": blocks.block.value_counts().to_dict(), "duplicate_rows": 0,
                      "device": str(DEV), "feature_source": str(FEATURE_PATH)}, indent=2))


def run_fold(meta, feats, outer, root, art):
    blocks, frac = temporal_blocks(meta)
    subjects = meta.subject.to_numpy()
    tasks = M._task_indices([{"task": t} for t in meta.task], len(meta))
    known = np.arange(1, 21)[np.arange(1, 21) != outer]
    early = frac <= .6
    middle = (frac > .6) & (frac <= .8)
    late = frac > .8
    fit = early & np.isin(subjects, known)
    assert not fit[subjects == outer].any()
    strict = fit_model(feats, tasks, subjects, fit, known)
    frozen = fit_model(feats, tasks, subjects, fit, known, art=art)
    frozen["protos"] = art["prototypes"][:, :20].to(DEV)
    frozen["valid"] = torch.ones(20, dtype=torch.bool, device=DEV)
    frozen["valid"][outer - 1] = False
    th, calibration = thresholds(feats, tasks, subjects, frac, known)
    th["frozen_euclid_tau"] = {"frozen_tau": -float(art["tau"])}
    result, score_cache, pred_cache = [], {}, {}
    for kind, fitted in (("frozen_euclid_tau", frozen), ("strict_euclid", strict), ("strict_ldacos_shrink001", strict)):
        score_kind = "strict_ldacos_shrink001" if kind == "strict_ldacos_shrink001" else "strict_euclid"
        for block_name, block_rows in (("middle", middle), ("late", late)):
            assert not (block_rows & fit).any()
            sv, who, ix = score(feats, tasks, block_rows, fitted, score_kind)
            score_cache[f"{kind}_{block_name}_score"] = sv
            score_cache[f"{kind}_{block_name}_row_index"] = ix
            pred_cache[f"{kind}_{block_name}_pred_id"] = who
            for th_name, tv in th[kind].items():
                accepted = sv >= tv
                pred_cache[f"{kind}_{block_name}_{th_name}_pred"] = np.where(accepted, who, 0)
                result.append({"outer_subject": outer, "candidate": kind, "block": block_name, "threshold": th_name,
                               "threshold_value": tv, **metrics(sv, who, subjects, block_rows, outer, tv)})
    fold_dir = root / f"outer_{outer:02d}"
    (fold_dir / "models").mkdir(parents=True, exist_ok=True)
    torch.save({"protocol_version": PROTOCOL_VERSION, "strict": cpu_model(strict), "frozen": cpu_model(frozen),
                "thresholds": th, "known": known.tolist(), "outer_subject": outer}, fold_dir / "models" / "model.pt")
    reloaded = torch.load(fold_dir / "models" / "model.pt", map_location="cpu", weights_only=False)
    for kind, fitted in (("strict_euclid", device_model(reloaded["strict"])),
                         ("frozen_euclid_tau", device_model(reloaded["frozen"])),
                         ("strict_ldacos_shrink001", device_model(reloaded["strict"]))):
        check_kind = "strict_ldacos_shrink001" if kind == "strict_ldacos_shrink001" else "strict_euclid"
        for block_name, block_rows in (("middle", middle), ("late", late)):
            chk, _, _ = score(feats, tasks, block_rows, fitted, check_kind)
            assert np.allclose(chk, score_cache[f"{kind}_{block_name}_score"], rtol=1e-5, atol=1e-6)
    np.savez_compressed(fold_dir / "scores_and_rows.npz", **score_cache, **pred_cache)
    np.savez_compressed(fold_dir / "calibration_scores.npz", **{f"{kind}_subject_{s}": vals for kind, by_subject in calibration.items() for s, vals in by_subject.items()})
    config = {"protocol_version": PROTOCOL_VERSION, "outer_subject": outer, "known": known.tolist(),
              "early_fit_rule": "frac <= 0.6", "inner_fit_rule": "frac <= 0.4", "inner_calibration_rule": "0.4 < frac <= 0.6",
              "middle_rule": "0.6 < frac <= 0.8", "late_rule": "frac > 0.8", "far_rule": "weighted unique-score tail with nextafter",
              "early_rows": int(fit.sum()), "middle_rows": int(middle.sum()), "late_rows": int(late.sum()),
              "metadata_source": str(META_PATH), "feature_source": str(FEATURE_PATH),
              "metadata_sha256": hashlib.sha256(META_PATH.read_bytes()).hexdigest(),
              "feature_sha256": hashlib.sha256(FEATURE_PATH.read_bytes()).hexdigest()}
    (fold_dir / "split.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (fold_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def protocol_matches(fold_dir):
    try:
        return json.loads((fold_dir / "split.json").read_text(encoding="utf-8")).get("protocol_version") == PROTOCOL_VERSION
    except (OSError, json.JSONDecodeError):
        return False


def write_summary(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    metrics_cols = ["AUC", "F1_unknown_rejection", "known_acceptance", "unknown_rejection", "known_false_rejection",
                    "known_identification_accuracy", "accepted_correct_accuracy", "actual_FAR"]
    agg = {m: ["mean", "std", "min"] for m in metrics_cols}
    summary = df.groupby(["candidate", "block", "threshold"], as_index=False).agg(agg)
    summary.columns = ["_".join(x).strip("_") for x in summary.columns]
    summary.to_csv(OUT / "results" / "summary.csv", index=False)
    summary.to_json(OUT / "results" / "summary.json", orient="records", indent=2)
    piv = df.pivot_table(index=["outer_subject", "candidate", "threshold"], columns="block", values=metrics_cols)
    drops = pd.DataFrame(index=piv.index)
    for metric in metrics_cols:
        if (metric, "middle") in piv and (metric, "late") in piv:
            drops[f"late_minus_middle_{metric}"] = piv[(metric, "late")] - piv[(metric, "middle")]
    drops.reset_index().to_csv(OUT / "results" / "middle_late_paired.csv", index=False)
    pair = df[df.candidate.isin(["strict_euclid", "strict_ldacos_shrink001"])].pivot_table(
        index=["outer_subject", "block", "threshold"], columns="candidate", values=metrics_cols)
    cb = pd.DataFrame(index=pair.index)
    for metric in metrics_cols:
        if (metric, "strict_ldacos_shrink001") in pair and (metric, "strict_euclid") in pair:
            cb[f"C_minus_B_{metric}"] = pair[(metric, "strict_ldacos_shrink001")] - pair[(metric, "strict_euclid")]
    cb.reset_index().to_csv(OUT / "results" / "C_minus_B_paired.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--outer-subject", type=int)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    tm = pd.read_csv(META_PATH)
    raw = np.load(FEATURE_PATH, mmap_mode="r")
    feats = torch.as_tensor(np.array(raw, dtype=np.float32, copy=True), device=DEV)
    validate(tm, feats)
    if args.validate_only:
        return
    art = M.load_model(str(MODEL_DIR), str(DEV))
    if args.outer_subject:
        assert 1 <= args.outer_subject <= 20
        folds = [args.outer_subject]
    elif args.smoke:
        folds = [1]
    elif args.all_folds:
        folds = list(range(1, 21))
    else:
        raise SystemExit("choose --validate-only, --outer-subject N, --smoke, or --all-folds")
    all_rows = []
    for outer in folds:
        fold_dir = OUT / "results" / f"outer_{outer:02d}"
        if args.resume and (fold_dir / "results.json").exists() and protocol_matches(fold_dir):
            all_rows.extend(json.loads((fold_dir / "results.json").read_text(encoding="utf-8")))
            continue
        t0 = time.time()
        rows = run_fold(tm, feats, outer, OUT / "results", art)
        all_rows.extend(rows)
        print(f"outer={outer} rows={len(rows)} seconds={time.time()-t0:.1f}", flush=True)
    (OUT / "results").mkdir(exist_ok=True)
    pd.DataFrame(all_rows).to_csv(OUT / "results" / "per_fold.csv", index=False)
    write_summary(all_rows)


if __name__ == "__main__":
    main()
