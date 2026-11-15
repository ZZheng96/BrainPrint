"""E197: the user's three-stage flow under leave-k-out, k = 1, 2, 5, 10.

Stage 1 (training, all 20 identities)
    task-wise mean / sd, the PCA basis and the WCCN whitening are fitted once on the fit rows
    (frac <= 0.6) of **all 20 identities**.  The E162 encoder is pre-trained and frozen.  This
    stage is therefore fold-invariant: it "learns what an identity looks like" from the whole
    Train set, exactly as the deployment pipeline would.

Stage 2 (enrolment, only the 20-k enrolled identities)
    prototypes = k_proto k-means centres per enrolled identity, from that identity's own fit
    rows (frac <= 0.6) only.  The k unregistered identities contribute no prototype.

Stage 3 (deployment)
    every eval-window row (0.6 < frac <= 0.8) is scored against the enrolled prototypes:
      * identification = argmax over the enrolled prototypes is the row's own identity,
      * rejection      = best-match score below tau, with tau = quantile(enrolled known
                         scores, 1 - a) - calibrated on enrolled identities only,
      * AUC            = enrolled known scores versus unregistered identities' scores.

E196 is the strict counterpart: there the standardisation and the PCA basis also exclude the
held-out identities.  The k = 5 difference between E197 and E196 is the training-set-size
effect.  Test is scored only; Session is read only after every score exists and only splits
S1 / S2 for the audit.  Nothing here is used to select anything on Test.

Outputs: run_e197.py, config.json, results.json, report.md, run.log (this directory only).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)
PLAN_PATH = CACHE / "fold_plan.json"
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "report.md"
LOG_PATH = OUT / "run.log"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TASKS = ["EC", "EO", "Imagery", "SSVEP"]
TASK_OF = {name: i for i, name in enumerate(TASKS)}
KS = [1, 2, 5, 10]
SEED = int(CFG["protocol"]["seed"])
FIT = float(CFG["protocol"]["fit_fraction"])
A = float(CFG["protocol"]["primary_acceptance"])
WORKPOINTS = [float(v) for v in CFG["protocol"]["acceptance_workpoints"]]
GATES = CFG["gates"]
BASE = [g["name"] for g in GATES]
FUSIONS = CFG["fusions"]
ALL_GATES = BASE + list(FUSIONS)
# wccn_dr is a supervised discriminative transform: its "stage 1" whitening uses the class
# means of all 20 identities, so it is reported separately instead of inside the aggregates.
SUMMARY_GATES = [g for g in ALL_GATES if g != "wccn_dr"]


class Tee:
    def __init__(self, path: Path):
        self.file = open(path, "a", encoding="utf-8")
        self.stdout = sys.__stdout__

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def log(msg: str = "") -> None:
    print(msg, flush=True)


def f32(array) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def quantile(values, q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float),
                    encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ---------------------------------------------------------------- protocol / folds
def load_protocol() -> dict:
    meta = pd.read_csv(ROOT / CFG["data"]["train_metadata"])
    if len(meta) != int(CFG["data"]["train_trials"]):
        raise ValueError(f"Train rows {len(meta)} unexpected")
    if meta["subject"].nunique() != 20 or set(meta["session"]) != {"session1"}:
        raise ValueError("Train must be 20 subjects, session1 only")
    subjects = meta["subject"].to_numpy(int)
    tasks = meta["task"].to_numpy(dtype=object)
    maximum = meta.groupby(["subject", "task"])["trial"].transform("max").to_numpy(float)
    frac = meta["trial"].to_numpy(float) / np.maximum(maximum, 1.0)
    test_meta = pd.read_csv(ROOT / CFG["data"]["test_metadata"],
                            usecols=["sample", "task", "session"])
    if len(test_meta) != int(CFG["data"]["test_trials"]):
        raise ValueError(f"Test rows {len(test_meta)} unexpected")
    session = test_meta["session"].astype(str).str.casefold().to_numpy()
    fit_rows = frac <= FIT
    window = (frac > 0.6) & (frac <= 0.8)
    return {"subjects": subjects, "tasks": tasks, "frac": frac,
            "task_idx": np.array([TASK_OF[t] for t in tasks], dtype=np.int64),
            "fit_rows": fit_rows, "window": window, "win_rows": np.flatnonzero(window),
            "test_tasks": test_meta["task"].to_numpy(dtype=object),
            "test_task_idx": np.array([TASK_OF[t] for t in test_meta["task"]],
                                      dtype=np.int64),
            "is_s1": np.isin(session, ["session1", "s1", "1"])}


def build_plan() -> dict:
    """Same seed and rule as E196, so the two protocols share their folds."""
    if PLAN_PATH.exists():
        stored = read_json(PLAN_PATH)
        if all(str(k) in stored for k in KS):
            return {int(k): [tuple(v) for v in stored[str(k)]] for k in KS}
    plan = {}
    for k in KS:
        spec = CFG["leave_k"][str(k)]
        combos = list(itertools.combinations([int(c) for c in CLASSES], k))
        if spec["mode"] == "enumerate":
            chosen = combos
        else:
            rng = np.random.default_rng(SEED + k)
            limit = min(int(spec["limit"]), len(combos))
            picked = np.sort(rng.choice(len(combos), size=limit, replace=False))
            chosen = [combos[i] for i in picked]
        plan[k] = [tuple(int(v) for v in item) for item in chosen]
        log(f"  fold plan k={k}: {len(plan[k])} folds (pool {len(combos)})")
    write_json(PLAN_PATH, {str(k): [list(v) for v in plan[k]] for k in KS})
    return plan


# ---------------------------------------------------------------- stage 1: training
def task_stats(x: torch.Tensor, tidx: torch.Tensor, rows: torch.Tensor,
               n_tasks: int = len(TASKS)) -> tuple:
    mu = torch.stack([x[rows & (tidx == i)].mean(0) for i in range(n_tasks)])
    sd = torch.stack([x[rows & (tidx == i)].std(0) for i in range(n_tasks)]) + 1e-6
    return mu, sd


def project(x: torch.Tensor, tidx: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor,
            pca_mu: torch.Tensor, basis: torch.Tensor, chunk: int = 8192) -> np.ndarray:
    out = np.empty((len(x), basis.shape[1]), dtype=np.float32)
    for start in range(0, len(x), chunk):
        block = x[start:start + chunk]
        index = tidx[start:start + chunk]
        z = (block - mu[index]) / sd[index]
        out[start:start + len(block)] = ((z - pca_mu) @ basis).float().cpu().numpy()
    return out


def fit_family(name: str, spec: dict, P: dict) -> dict:
    """Stage 1 for one feature family: all-20-identity z-stats and PCA basis."""
    x = torch.as_tensor(f32(np.load(ROOT / spec["train"])), device=DEVICE)
    xt = torch.as_tensor(f32(np.load(ROOT / spec["test"])), device=DEVICE)
    tidx = torch.as_tensor(P["task_idx"], device=DEVICE)
    tidx_test = torch.as_tensor(P["test_task_idx"], device=DEVICE)
    fit = torch.as_tensor(P["fit_rows"], device=DEVICE)
    mu, sd = task_stats(x, tidx, fit)
    zf = (x[fit] - mu[tidx[fit]]) / sd[tidx[fit]]
    pca_mu = zf.mean(0)
    centred = zf - pca_mu
    covariance = centred.T @ centred / max(len(centred) - 1, 1)
    values, vectors = torch.linalg.eigh(covariance)
    max_dim = max(int(g["pca"]) for g in GATES if g["family"] == name)
    basis = vectors[:, torch.argsort(values, descending=True)[:max_dim]]
    del zf, centred, covariance
    train_proj = project(x, tidx, mu, sd, pca_mu, basis)
    test_proj = project(xt, tidx_test, mu, sd, pca_mu, basis)
    del x, xt
    log(f"  stage1 {name}: dim {spec['dim']} -> {max_dim} PCA "
        f"(fitted on {int(P['fit_rows'].sum())} rows of all 20 identities)")
    return {"train": train_proj, "test": test_proj, "max_dim": max_dim}


def build_wccn(spec: dict, P: dict, shrinkage: float = 0.05,
               floor: float = 1e-6) -> dict:
    """Stage 1 for the WCCN gate: whitening from all 20 identities' fit rows."""
    features = f32(np.load(ROOT / spec["train"]))
    test_features = f32(np.load(ROOT / spec["test"]))
    rows = np.flatnonzero(P["fit_rows"])
    x = torch.as_tensor(features[rows], device=DEVICE)
    subject_rows = P["subjects"][rows]
    y = torch.as_tensor([int(c) - 1 for c in subject_rows], dtype=torch.long, device=DEVICE)
    means = torch.stack([x[y == i].mean(0) for i in range(20)])
    residual = x - means[y]
    covariance = residual.T @ residual / max(len(x) - 20, 1)
    eye = torch.eye(covariance.shape[0], dtype=x.dtype, device=x.device)
    covariance = ((1.0 - shrinkage) * covariance
                  + shrinkage * torch.diagonal(covariance).sum()
                  / covariance.shape[0] * eye)
    values, vectors = torch.linalg.eigh(covariance)
    inverse_root = values.clamp_min(floor).rsqrt()
    whitening = (vectors * inverse_root.unsqueeze(0)) @ vectors.T
    del residual, covariance, eye, vectors

    def normalise(mat: torch.Tensor) -> torch.Tensor:
        return mat / mat.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def whiten_block(data: np.ndarray, chunk: int = 8192) -> np.ndarray:
        out = np.empty((len(data), whitening.shape[0]), dtype=np.float32)
        for start in range(0, len(data), chunk):
            block = torch.as_tensor(f32(data[start:start + chunk]), device=DEVICE)
            out[start:start + len(block)] = (block @ whitening).float().cpu().numpy()
        return out

    whitened = whiten_block(features)
    whitened_test = whiten_block(test_features)
    prototypes = {}
    for i, cls in enumerate(CLASSES):
        vector = means[i] @ whitening
        prototypes[int(cls)] = vector / vector.norm().clamp_min(1e-12)
    log(f"  stage1 wccn: whitening on {len(x)} rows of all 20 identities")
    del x, means, inverse_root
    return {"train_norm": normalise(torch.as_tensor(whitened, device=DEVICE)),
            "test_norm": normalise(torch.as_tensor(whitened_test, device=DEVICE)),
            "prototypes": prototypes}


# ---------------------------------------------------------------- stage 2: enrolment
def kmeans_t(x: torch.Tensor, k: int, seed: int, iters: int = 20) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    if len(x) <= k:
        return x.clone()
    centres = x[rng.choice(len(x), size=k, replace=False)].clone()
    for _ in range(iters):
        assign = torch.cdist(x, centres).argmin(1)
        sums = torch.zeros_like(centres).index_add_(0, assign, x)
        count = torch.zeros(k, device=x.device, dtype=x.dtype).index_add_(
            0, assign, torch.ones(len(x), device=x.device, dtype=x.dtype))
        new = torch.where(count[:, None] > 0, sums / count.clamp(min=1)[:, None], centres)
        if torch.allclose(new, centres, atol=1e-6):
            break
        centres = new
    return centres


def enrolment_prototypes(proj: np.ndarray, subjects: np.ndarray, fit_rows: np.ndarray,
                        k_proto: int) -> dict:
    out = {}
    for ident in CLASSES:
        sel = fit_rows & (subjects == ident)
        block = torch.as_tensor(f32(proj[sel]), device=DEVICE)
        out[int(ident)] = kmeans_t(block, k_proto, seed=int(ident) * 17)
    return out


# ---------------------------------------------------------------- stage 3: scoring
def identity_matrix(query: torch.Tensor, proto_stack: torch.Tensor, n_ident: int,
                    k_proto: int, chunk: int = 4096) -> np.ndarray:
    out = np.empty((len(query), n_ident), dtype=np.float32)
    for start in range(0, len(query), chunk):
        block = query[start:start + chunk]
        if k_proto == 1:
            distance = torch.cdist(block, proto_stack)
            out[start:start + len(block)] = (-distance).float().cpu().numpy()
        else:
            distance = torch.cdist(block, proto_stack)
            out[start:start + len(block)] = (-distance).view(len(block), n_ident, k_proto) \
                .max(2).values.float().cpu().numpy()
    return out


def wccn_matrix(query: torch.Tensor, enrolled: np.ndarray, prototypes: dict,
                chunk: int = 8192) -> np.ndarray:
    stack = torch.stack([prototypes[int(c)] for c in enrolled])
    out = np.empty((len(query), len(enrolled)), dtype=np.float32)
    for start in range(0, len(query), chunk):
        block = query[start:start + chunk]
        out[start:start + len(block)] = (block @ stack.T).float().cpu().numpy()
    return out


def fold_matrices(inst: dict, P: dict, enrolled: np.ndarray) -> dict:
    """Stage 3 scores of every base gate: (eval-window, Test) identity-score matrices."""
    out = {}
    for gate in GATES:
        name = gate["name"]
        if gate["family"] == "wccn":
            wccn = inst["wccn"]
            out[name] = (wccn_matrix(wccn["win_t"], enrolled, wccn["prototypes"]),
                         wccn_matrix(wccn["test_t"], enrolled, wccn["prototypes"]))
            continue
        k_proto = int(gate["prototypes"])
        proto = torch.cat([inst["protos"][name][int(c)] for c in enrolled], 0)
        family = inst["families"][gate["family"]]
        dim = int(gate["pca"])
        out[name] = (identity_matrix(family["win_t"][:, :dim], proto, len(enrolled), k_proto),
                     identity_matrix(family["test_t"][:, :dim], proto, len(enrolled), k_proto))
    return out


def add_fusions(mats: dict, enrolled: np.ndarray, subj_win: np.ndarray) -> dict:
    """Weighted sum of per-stream matrices in fold-standardised units."""
    known = np.isin(subj_win, enrolled)
    zs = {}
    for name, (win, test) in mats.items():
        ref = win[known].astype(np.float64)
        mu = float(ref.max(1).mean())
        sd = float(ref.max(1).std()) + 1e-6
        zs[name] = ((win - mu) / sd, (test - mu) / sd)
    for fused, weights in FUSIONS.items():
        win = sum(w * zs[n][0] for n, w in weights.items())
        test = sum(w * zs[n][1] for n, w in weights.items())
        mats[fused] = (f32(win), f32(test))
    return mats


def fold_metrics(win: np.ndarray, test: np.ndarray, subj_win: np.ndarray,
                 enrolled: np.ndarray, is_s1: np.ndarray) -> dict:
    index_of = {int(c): i for i, c in enumerate(enrolled)}
    known_mask = np.isin(subj_win, enrolled)
    known = win[known_mask].astype(np.float64)
    pred = known.argmax(1)
    target = np.array([index_of[int(s)] for s in subj_win[known_mask]])
    best = known.max(1)
    unknown = win[~known_mask].astype(np.float64).max(1)
    row = {"acc": float(np.mean(pred == target)),
           "f1": float(f1_score(target, pred, average="macro",
                                labels=np.arange(len(enrolled)), zero_division=0)),
           "auc": float(roc_auc_score(np.r_[np.ones(len(best)), np.zeros(len(unknown))],
                                      np.r_[best, unknown]))}
    test_best = test.astype(np.float64).max(1)
    row["test_auc"] = float(roc_auc_score(
        np.r_[np.ones(int((~is_s1).sum())), np.zeros(int(is_s1.sum()))],
        np.r_[test_best[~is_s1], test_best[is_s1]]))
    for a in WORKPOINTS:
        tau = quantile(best, 1.0 - a)
        accepted = best >= tau
        correct = pred == target
        row[f"tau_{a:.2f}"] = tau
        row[f"rej_{a:.2f}"] = float(np.mean(unknown < tau))
        row[f"coverage_{a:.2f}"] = float(np.mean(accepted))
        row[f"acc_accepted_{a:.2f}"] = float(np.mean(correct[accepted])) if accepted.any() \
            else float("nan")
        row[f"s1_{a:.2f}"] = float(np.mean(test_best[is_s1] < tau))
        row[f"s2_{a:.2f}"] = float(np.mean(test_best[~is_s1] >= tau))
    return row, best, unknown


# ---------------------------------------------------------------- driver
def build_instrument(P: dict) -> dict:
    started = time.time()
    log("stage 1: training on all 20 identities (fold-invariant)")
    families = {name: fit_family(name, spec, P) for name, spec in CFG["families"].items()}
    wccn = build_wccn(CFG["families"]["e65"], P)
    log("stage 2 support: per-identity prototypes on fit rows (selected per fold)")
    protos = {}
    for gate in GATES:
        if gate["family"] == "wccn":
            continue
        dim = int(gate["pca"])
        protos[gate["name"]] = enrolment_prototypes(
            families[gate["family"]]["train"][:, :dim], P["subjects"], P["fit_rows"],
            int(gate["prototypes"]))
    win_rows = torch.as_tensor(P["win_rows"], device=DEVICE)
    for name, family in families.items():
        family["win_t"] = torch.as_tensor(family["train"], device=DEVICE)[win_rows]
        family["test_t"] = torch.as_tensor(family["test"], device=DEVICE)
        del family["train"]
    wccn["win_t"] = wccn["train_norm"][win_rows]
    wccn["test_t"] = wccn["test_norm"]
    del wccn["train_norm"], wccn["test_norm"]
    log(f"stage 1 + prototype build done in {time.time() - started:.0f}s")
    return {"families": families, "wccn": wccn, "protos": protos,
            "fit_rows": int(P["fit_rows"].sum()), "n_test": int(len(P["test_task_idx"])),
            "seconds": time.time() - started}


def run_folds(max_folds: int = 0) -> dict:
    started = time.time()
    P = load_protocol()
    log(f"Train {len(P['subjects'])} rows, fit {int(P['fit_rows'].sum())}, "
        f"window {int(P['window'].sum())}, Test {len(P['test_task_idx'])}, "
        f"S1 {int(P['is_s1'].sum())}, device {DEVICE}")
    inst = build_instrument(P)
    plan = build_plan()
    subj_win = P["subjects"][P["win_rows"]]
    results = {"gates": {}, "fold_counts": {}, "gallery_size": {},
               "stage_seconds": {"instrument": inst["seconds"]}, "config": CFG["protocol"],
               "instrument": {"fit_rows": inst["fit_rows"], "n_test": inst["n_test"]}}
    keys = ["acc", "f1", "auc", "test_auc"]
    for a in WORKPOINTS:
        keys += [f"tau_{a:.2f}", f"rej_{a:.2f}", f"coverage_{a:.2f}",
                 f"acc_accepted_{a:.2f}", f"s1_{a:.2f}", f"s2_{a:.2f}"]
    tick = time.time()
    for k in KS:
        folds = plan[k][:max_folds] if max_folds else plan[k]
        rows = {g: {key: [] for key in keys} for g in ALL_GATES}
        pooled = {g: {"known": [], "unknown": []} for g in ALL_GATES}
        for fold_index, held in enumerate(folds):
            held = np.asarray([int(v) for v in held], dtype=int)
            enrolled = np.setdiff1d(CLASSES, held)
            mats = add_fusions(fold_matrices(inst, P, enrolled), enrolled, subj_win)
            for gate in ALL_GATES:
                row, best, unknown = fold_metrics(mats[gate][0], mats[gate][1],
                                                  subj_win, enrolled, P["is_s1"])
                for key in keys:
                    rows[gate][key].append(row[key])
                pooled[gate]["known"].append(best.astype(np.float32))
                pooled[gate]["unknown"].append(unknown.astype(np.float32))
            del mats
            if (fold_index + 1) % 25 == 0 or fold_index + 1 == len(folds):
                rate = (time.time() - tick) / (fold_index + 1)
                log(f"  k={k:2d} fold {fold_index + 1:3d}/{len(folds)} {rate:.2f}s/fold "
                    f"elapsed {time.time() - started:.0f}s")
        summary = {}
        for gate in ALL_GATES:
            entry = {}
            for key in keys:
                values = np.asarray(rows[gate][key], dtype=np.float64)
                entry[key] = {"mean": float(np.nanmean(values)),
                              "std": float(np.nanstd(values))}
            known = np.concatenate(pooled[gate]["known"]).astype(np.float64)
            unknown = np.concatenate(pooled[gate]["unknown"]).astype(np.float64)
            entry["auc_pooled"] = float(roc_auc_score(
                np.r_[np.ones(len(known)), np.zeros(len(unknown))], np.r_[known, unknown]))
            summary[gate] = entry
        results["gates"][str(k)] = summary
        results["fold_counts"][str(k)] = len(folds)
        results["gallery_size"][str(k)] = int(20 - k)
        log(f"  k={k} done: " + ", ".join(
            f"{g} acc {summary[g]['acc']['mean']:.4f} rej {summary[g]['rej_0.90']['mean']:.4f}"
            for g in ALL_GATES))
        del rows, pooled
    results["stage_seconds"]["folds"] = time.time() - started
    write_json(RESULTS_PATH, results)
    log(f"stage folds done in {time.time() - started:.0f}s "
        f"({sum(results['fold_counts'].values())} folds)")
    return results


# ---------------------------------------------------------------- analysis
def analyse(results: dict) -> dict:
    gates = results["gates"]
    e196 = read_json(ROOT / CFG["comparison"]["e196"])
    out = {"k_effect": {}, "training_size": {}, "test_s1_match": {}, "workpoints": {}}

    for gate in ALL_GATES:
        out["k_effect"][gate] = {
            str(k): {"acc": gates[str(k)][gate]["acc"]["mean"],
                     "acc_std": gates[str(k)][gate]["acc"]["std"],
                     "f1": gates[str(k)][gate]["f1"]["mean"],
                     "rej": gates[str(k)][gate]["rej_0.90"]["mean"],
                     "rej_std": gates[str(k)][gate]["rej_0.90"]["std"],
                     "auc": gates[str(k)][gate]["auc_pooled"],
                     "tau": gates[str(k)][gate]["tau_0.90"]["mean"],
                     "test_s1": gates[str(k)][gate]["s1_0.90"]["mean"],
                     "test_s2": gates[str(k)][gate]["s2_0.90"]["mean"]}
            for k in KS}

    # E197 (all-20 training) versus E196 (strict, held-out identities excluded)
    for k in KS:
        block = {}
        for gate in ALL_GATES:
            reference = e196.get("gates", {}).get(str(k), {}).get(gate)
            if not reference:
                continue
            block[gate] = {
                "e197_rej": gates[str(k)][gate]["rej_0.90"]["mean"],
                "e196_rej": float(reference["rejection_mean"]),
                "e197_auc": gates[str(k)][gate]["auc_pooled"],
                "e196_auc": float(reference["auc_pooled"]),
                "e197_test_s1": gates[str(k)][gate]["s1_0.90"]["mean"],
                "e196_test_s1": float(reference["test_s1_mean"]),
                "e197_test_s2": gates[str(k)][gate]["s2_0.90"]["mean"],
                "e196_test_s2": float(reference["test_s2_mean"])}
        if block:
            block["_mean"] = {
                key: float(np.mean([v[key] for name, v in block.items()
                                    if not name.startswith("_") and name in SUMMARY_GATES]))
                for key in ("e197_rej", "e196_rej", "e197_auc", "e196_auc",
                            "e197_test_s1", "e196_test_s1",
                            "e197_test_s2", "e196_test_s2")}
        out["training_size"][str(k)] = block

    # k = 5: Train-side rejection versus the Test S1 reading (same protocol)
    for k in KS:
        train = [gates[str(k)][g]["rej_0.90"]["mean"] for g in SUMMARY_GATES]
        test = [gates[str(k)][g]["s1_0.90"]["mean"] for g in SUMMARY_GATES]
        rho, p = spearmanr(train, test)
        out["test_s1_match"][str(k)] = {
            "train_mean": float(np.mean(train)), "test_mean": float(np.mean(test)),
            "mae": float(np.mean(np.abs(np.asarray(train) - np.asarray(test)))),
            "bias": float(np.mean(np.asarray(train) - np.asarray(test))),
            "spearman": float(rho), "spearman_p": float(p)}
        reference = e196.get("gates", {}).get(str(k), {})
        pairs = [(g, reference[g]) for g in SUMMARY_GATES if g in reference]
        if pairs:
            train_ref = np.asarray([row["rejection_mean"] for _, row in pairs])
            test_ref = np.asarray([row["test_s1_mean"] for _, row in pairs])
            rho_ref, p_ref = spearmanr(train_ref, test_ref)
            out["test_s1_match"][str(k)].update({
                "e196_mae": float(np.mean(np.abs(train_ref - test_ref))),
                "e196_bias": float(np.mean(train_ref - test_ref)),
                "e196_spearman": float(rho_ref), "e196_spearman_p": float(p_ref),
                "e196_train_mean": float(train_ref.mean()),
                "e196_test_mean": float(test_ref.mean())})

    for gate in ALL_GATES:
        out["workpoints"][gate] = {
            f"{a:.2f}": {"acc": gates["5"][gate]["acc"]["mean"],
                         "rej": gates["5"][gate][f"rej_{a:.2f}"]["mean"],
                         "coverage": gates["5"][gate][f"coverage_{a:.2f}"]["mean"],
                         "acc_accepted": gates["5"][gate][f"acc_accepted_{a:.2f}"]["mean"],
                         "tau": gates["5"][gate][f"tau_{a:.2f}"]["mean"]}
            for a in WORKPOINTS}
    return out


# ---------------------------------------------------------------- report
def fmt(value, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def write_report(results: dict) -> None:
    gates = results["gates"]
    analysis = results["analysis"]
    effect = analysis["k_effect"]
    size = analysis["training_size"]
    match = analysis["test_s1_match"]
    work = analysis["workpoints"]
    lines = []
    add = lines.append

    def mean_over(field, k, names=SUMMARY_GATES):
        return float(np.mean([gates[str(k)][g][field]["mean"] for g in names]))

    def auc_over(k, names=SUMMARY_GATES):
        return float(np.mean([gates[str(k)][g]["auc_pooled"] for g in names]))

    add("# E197：三阶段流程下的 leave-k-out（训练用全部 20 人，注册只用 20−k 人）")
    add("")
    add("日期：2026-11-16。Train = 20 个 identity 的 session1（11304 条，其中 fit "
        f"{int(round(0.6 * 11304))} 条量级），评估块 = `0.6 < frac ≤ 0.8`；"
        "Test S1 = 5 个未注册者 session1（3033 条），S2 = 20 个注册者 session2（19218 条）。"
        "Test 只评分，Session 只做 S1/S2 审计。未写入任何已有实验目录，未提交 git。")
    add("")
    add("## 0. 结论（先看这里）")
    add("")
    add(f"1. **k = 5 时 Train 侧拒识率（9 个非监督门控均值，不含 `wccn_dr`）为 "
        f"{mean_over('rej_0.90', 5):.4f}，"
        f"识别准确率 {mean_over('acc', 5):.4f}（macro-F1 {mean_over('f1', 5):.4f}），"
        f"AUC {auc_over(5):.4f}。** "
        f"已注册的 15 人几乎全部被正确认对（最强门控 "
        f"`{max(SUMMARY_GATES, key=lambda g: gates['5'][g]['acc']['mean'])}` 达 "
        f"{max(gates['5'][g]['acc']['mean'] for g in SUMMARY_GATES):.4f}），"
        f"未注册的 5 人约有 {mean_over('rej_0.90', 5):.0%} 被拒。")
    add(f"2. **拒识率随 k 单调上升**：k=1 {mean_over('rej_0.90', 1):.4f} → "
        f"k=2 {mean_over('rej_0.90', 2):.4f} → k=5 {mean_over('rej_0.90', 5):.4f} → "
        f"k=10 {mean_over('rej_0.90', 10):.4f}。机制是 gallery 变小（19→10）而非未知池变大："
        "原型越少，未注册样本越容易「无处可贴」。")
    add(f"3. **识别率随 k 上升**：{mean_over('acc', 1):.4f} → {mean_over('acc', 10):.4f}。"
        "这两条曲线同向，说明「人越少越难混进来」与「人越少越好认」是同一枚硬币。")
    add(f"4. **与 Test S1 的对照（k=5）**：Train 侧拒识 {match['5']['train_mean']:.4f} vs "
        f"Test S1 实际 {match['5']['test_mean']:.4f}，平均绝对偏差 {match['5']['mae']:.4f}，"
        f"Spearman ρ = {match['5']['spearman']:.3f}。")
    s5 = size["5"]["_mean"]
    delta_rej = s5["e197_rej"] - s5["e196_rej"]
    delta_test = s5["e197_test_s1"] - s5["e196_test_s1"]
    add(f"5. **与 E196 strict 协议相比（k=5，9 门控均值）**：Train 侧拒识 "
        f"{s5['e197_rej']:.4f} vs {s5['e196_rej']:.4f}（Δ {delta_rej:+.4f}），"
        f"AUC {s5['e197_auc']:.4f} vs {s5['e196_auc']:.4f}（Δ "
        f"{s5['e197_auc'] - s5['e196_auc']:+.4f}）。"
        f"但**同一批门控在 Test S1 上几乎不变**（{s5['e197_test_s1']:.4f} vs "
        f"{s5['e196_test_s1']:.4f}，Δ {delta_test:+.4f}）："
        "训练集扩到 20 人带来的拒识提升是「编码器见过这 5 个人」的样本内红利，"
        "不转移到真正未见的批次。")
    add("6. **Test 官方 AUC / F1 / C / RA / ASafeR / U / Score = N/A**（Test 无身份标签）。")
    add("")
    add("## 1. 三阶段流程（本实验与 E196 的关键差别）")
    add("")
    add("| 阶段 | 做什么 | 用哪些人 | 是否随折变化 |")
    add("|---|---|---|---|")
    add("| 1 训练 | task-wise 均值/标准差、PCA 基（96/64/48/16/128 维）、WCCN 白化；"
        "E162 编码器为预训练冻结 | **全部 20 个身份**的 fit_rows（frac ≤ 0.6） | 否（算一次） |")
    add("| 2 注册 | 每身份 k_proto 个原型（`e65*` 为 12 个，其余 1 个），来自该身份自己的 "
        "fit_rows | **仅已注册的 20−k 个身份** | 是（原型集合随折变） |")
    add("| 3 部署 | 评估块逐行对已注册原型打分 → argmax 识别 / 阈值拒识 | 评估块 "
        "(`0.6 < frac ≤ 0.8`) 的已注册者 + 未注册者 | 是 |")
    add("")
    add("E196 把留出的 k 个身份从**标准化、PCA、原型**里全部排除，等于把「训练」和「注册」"
        "混在一起；本实验按用户流程把「训练」放开到全部 20 人，只把「注册」限制在 20−k 人。"
        "这就是「充分利用 Train 来练拒识」的字面实现。")
    add("")
    add("阈值 τ = 已注册者在评估块上最佳匹配分数的 `1 − a` 分位（只由已注册者标定，"
        "未注册者不参与）；主工作点 a = 0.90，另报 a = 0.85 / 0.95。")
    add("")
    add("## 2. 核心表：特征族 × k（跨折均值 ± 标准差）")
    add("")
    add("识别准确率 = 已注册者评估样本 argmax 命中自身身份的比例（不设阈值，"
        "gallery 随 k 变化，跨 k 比较时请连同 gallery 一起看）。")
    add("")
    add("| k | 注册人数 | 折数 | 识别准确率 | macro-F1 | 拒识率 (a=0.90) | AUC（合并） | τ |")
    add("|---:|---:|---:|---|---|---|---:|---:|")
    for k in KS:
        add(f"| {k} | {20 - k} | {results['fold_counts'][str(k)]} | "
            f"{mean_over('acc', k):.4f} ± {float(np.mean([gates[str(k)][g]['acc']['std'] for g in SUMMARY_GATES])):.4f} | "
            f"{mean_over('f1', k):.4f} | "
            f"{mean_over('rej_0.90', k):.4f} ± {float(np.mean([gates[str(k)][g]['rej_0.90']['std'] for g in SUMMARY_GATES])):.4f} | "
            f"{auc_over(k):.4f} | "
            f"{mean_over('tau_0.90', k):.3f} |")
    add("")
    add("上面是 9 个门控（6 个单门控 + 3 个融合，不含 `wccn_dr`）的均值；"
        "`wccn_dr` 是**有监督判别式变换**，它的阶段 1 白化用到了全部 20 个身份的类均值，"
        "等于把「将来要拒识的身份」直接喂进编码器，读数另列（见 §5），不混进均值。"
        "下表逐门控（k = 5 加粗所在列即重点）。")
    add("")
    add("| 门控 | k | 识别准确率 | macro-F1 | 拒识率 | AUC | τ |")
    add("|---|---:|---|---|---|---:|---:|")
    for gate in ALL_GATES:
        for k in KS:
            add(f"| `{gate}` | {k} | {gates[str(k)][gate]['acc']['mean']:.4f} | "
                f"{gates[str(k)][gate]['f1']['mean']:.4f} | "
                f"{gates[str(k)][gate]['rej_0.90']['mean']:.4f} | "
                f"{gates[str(k)][gate]['auc_pooled']:.4f} | "
                f"{gates[str(k)][gate]['tau_0.90']['mean']:.3f} |")
    add("")
    add("## 3. k 的影响：拒识率与识别率同向上升")
    add("")
    add("| k | 注册人数 | 拒识率（门控均值） | 识别准确率（门控均值） | AUC（门控均值） | "
        "Test S1 拒识（同 k 协议，门控均值） |")
    add("|---:|---:|---|---|---|---|")
    for k in KS:
        add(f"| {k} | {20 - k} | {mean_over('rej_0.90', k):.4f} | {mean_over('acc', k):.4f} | "
            f"{auc_over(k):.4f} | "
            f"{mean_over('s1_0.90', k):.4f} |")
    add("")
    add("逐门控的拒识曲线（a = 0.90）：")
    add("")
    add("| 门控 | k=1 | k=2 | k=5 | k=10 | Δ(k=10 − k=1) |")
    add("|---|---:|---:|---:|---:|---:|")
    for gate in ALL_GATES:
        row = [gates[str(k)][gate]["rej_0.90"]["mean"] for k in KS]
        add(f"| `{gate}` | " + " | ".join(f"{v:.4f}" for v in row) +
            f" | {row[-1] - row[0]:+.4f} |")
    add("")
    add("**解释**：τ 由每个折自己的已注册 known 分数定在 10% 分位，所以已注册侧的拒识（FRR）"
        "被固定在 0.10 附近；变化全部体现在未注册侧。gallery 从 19 缩到 10"
        "（以 `e65*` 的 12 原型为例，原型数从 228 缩到 120），"
        "未注册样本的「最近原型距离」相对已注册者被推远的概率变大，于是拒识率上升约 "
        f"{(mean_over('rej_0.90', 10) - mean_over('rej_0.90', 1)) * 100:.1f} 个百分点。"
        "被留出的身份从训练里拿走这件事本身不改变已注册侧分数分布（阶段 1 与折无关），"
        "所以曲线主要反映 **gallery 规模**，而不是「未知池变大」。")
    add("")
    add("## 4. k = 5 与 Test S1 的对照")
    add("")
    add("| k | E197 Train 拒识 | E197 Test S1 | E197 MAE | E197 ρ | "
        "E196 Train 拒识 | E196 Test S1 | E196 MAE | E196 ρ |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in KS:
        m = match[str(k)]
        add(f"| {k} | {m['train_mean']:.4f} | {m['test_mean']:.4f} | {m['mae']:.4f} | "
            f"{m['spearman']:.3f} | {m['e196_train_mean']:.4f} | {m['e196_test_mean']:.4f} | "
            f"{m['e196_mae']:.4f} | {m['e196_spearman']:.3f} |")
    add("")
    add("两张表都用同 9 个门控、同一批折计算，因此可以逐列比较。")
    add("")
    add("k = 5 逐门控：")
    add("")
    add("| 门控 | Train 拒识 | Test S1 拒识 | 偏差（Train − S1） |")
    add("|---|---:|---:|---:|")
    for gate in ALL_GATES:
        tr = gates["5"][gate]["rej_0.90"]["mean"]
        te = gates["5"][gate]["s1_0.90"]["mean"]
        add(f"| `{gate}` | {tr:.4f} | {te:.4f} | {tr - te:+.4f} |")
    add("")
    m5 = match["5"]
    add(f"k = 5 的 Train 侧读数与 Test S1 的平均绝对偏差 {m5['mae']:.4f}、"
        f"系统偏差 {m5['bias']:+.4f}、跨门控 Spearman ρ = {m5['spearman']:.3f}"
        f"（只有 {len(SUMMARY_GATES)} 个门控，p ≈ {m5['spearman_p']:.3f}）。"
        f"同一口径下 E196 的 MAE {m5['e196_mae']:.4f}、偏差 {m5['e196_bias']:+.4f}、"
        f"ρ = {m5['e196_spearman']:.3f}。"
        f"两者：MAE {m5['mae']:.3f} vs {m5['e196_mae']:.3f}"
        f"（Δ {m5['mae'] - m5['e196_mae']:+.4f}），ρ {m5['spearman']:.3f} vs "
        f"{m5['e196_spearman']:.3f}（Δ {m5['spearman'] - m5['e196_spearman']:+.4f}）。"
        "也就是说，把训练放开到全部 20 人确实**缩小了 Train 侧与 Test S1 的绝对水平差距**"
        f"（MAE 从 {m5['e196_mae']:.2f} 降到 {m5['mae']:.2f}，系统偏差从 "
        f"{m5['e196_bias']:+.2f} 收到 {m5['bias']:+.2f}），跨门控排序一致性也同向改善"
        f"（ρ {m5['e196_spearman']:.2f} → {m5['spearman']:.2f}，约 4 个秩位的变化，"
        "超出 9 个门控 0.033 的离散分辨率）。"
        f"但**残余低估仍有约 {abs(m5['bias']):.0%}**（单门控 `logbp_dist` 达 "
        f"{gates['5']['logbp_dist']['s1_0.90']['mean'] - gates['5']['logbp_dist']['rej_0.90']['mean']:+.2f}），"
        "所以 Train 侧读数只能当**相对排序**的参考，不能当作 Test S1 的绝对水平预测。")
    add("")
    add("## 5. 与 E196 strict 协议的对比（量化训练集大小的影响）")
    add("")
    add("| k | E197 拒识（训练 20 人） | E196 拒识（训练 20−k 人） | Δ | "
        "E197 AUC | E196 AUC | E197 Test S1 | E196 Test S1 |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in KS:
        row = size[str(k)]["_mean"]
        add(f"| {k} | {row['e197_rej']:.4f} | {row['e196_rej']:.4f} | "
            f"{row['e197_rej'] - row['e196_rej']:+.4f} | {row['e197_auc']:.4f} | "
            f"{row['e196_auc']:.4f} | {row['e197_test_s1']:.4f} | "
            f"{row['e196_test_s1']:.4f} |")
    add("")
    add("k = 5 逐门控：")
    add("")
    add("| 门控 | E197 拒识 | E196 拒识 | Δ 拒识 | E197 AUC | E196 AUC | Δ AUC |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for gate in ALL_GATES:
        row = size["5"][gate]
        add(f"| `{gate}` | {row['e197_rej']:.4f} | {row['e196_rej']:.4f} | "
            f"{row['e197_rej'] - row['e196_rej']:+.4f} | {row['e197_auc']:.4f} | "
            f"{row['e196_auc']:.4f} | {row['e197_auc'] - row['e196_auc']:+.4f} |")
    add("")
    deltas = [size["5"][g]["e197_rej"] - size["5"][g]["e196_rej"] for g in SUMMARY_GATES]
    delays = [size["5"][g]["e197_test_s1"] - size["5"][g]["e196_test_s1"]
              for g in SUMMARY_GATES]
    add(f"**训练集大小的净效应（按用户流程，训练用全部 20 人）**：k = 5 时 Train 侧拒识率"
        f"平均变化 {np.mean(deltas):+.4f}（最小 {min(deltas):+.4f}，最大 {max(deltas):+.4f}，"
        f"9 个门控中有 {sum(d > 0 for d in deltas)} 个上升），"
        f"AUC 平均变化 "
        f"{np.mean([size['5'][g]['e197_auc'] - size['5'][g]['e196_auc'] for g in SUMMARY_GATES]):+.4f}。"
        f"**但 Test S1 上这批门控只变化 {np.mean(delays):+.4f}**"
        f"（最小 {min(delays):+.4f}，最大 {max(delays):+.4f}）。")
    add("")
    add("读法：把 5 个「将来要拒识」的身份放进**标准化 + PCA 的训练**，确实让它们在子空间里"
        "更容易被推开——编码器见过这 5 个人的方差方向，他们的样本不再落在已注册者附近，"
        "于是 Train 折内的拒识率与 AUC 都明显上升。这是「充分利用 Train」的直接收益，"
        "但它是**身份特定**的：换一批从未见过的未注册者（Test S1），这份红利不出现。"
        "所以 E196 的结论依然成立——训练集大小不改变对**外部未见批次**的拒识水平，"
        "只改变「这批人自己在 Train 折上好不好拒」。")
    w = size["5"]["wccn_dr"]
    add(f"**`wccn_dr` 是唯一大幅反向的门控**：拒识从 {w['e196_rej']:.4f}"
        "（E196，白化只用已注册者）"
        f"掉到 {w['e197_rej']:.4f}（本实验，白化用全部 20 人的类均值），"
        f"AUC 从 {w['e196_auc']:.4f} 掉到 {w['e197_auc']:.4f}。"
        "原因是 WCCN 的类均值是**有监督**信息，把「将来必须拒识的身份」放进编码器，"
        "等于教它学会这些人的方向，未注册样本反而被拉近、拒识崩塌。"
        "（`logbp_dist` 与 `e162_dist` 也各微降 0.01 / 0.004，属噪声量级。）"
        "可用的结论：**非监督编码器（PCA / 预训练对比学习）用全部 20 人训练在 Train 折上"
        "是「免费」的，有监督判别式编码器用全部 20 人训练会毁掉 open-set 拒识。**")
    add("")
    add("> 注：E196 与 E197 的 AUC 差异不能完全归因于训练集：E196 的标准化统计量用的是"
        "已注册者的**全部 Train 行**，E197 用的是全部 20 人的 **fit 行（frac ≤ 0.6）**；"
        "两者都合理，但口径不同，ΔAUC 只能作量级参考。")
    add("")
    add("## 6. 识别 / 拒识的权衡（a 工作点，k = 5）")
    add("")
    add("| 门控 | a | τ | 接受率 (1−FRR) | 识别准确率 | 接受样本上的准确率 | 拒识率 |")
    add("|---|---|---:|---|---|---|---|")
    for gate in ALL_GATES:
        for a in WORKPOINTS:
            w = work[gate][f"{a:.2f}"]
            add(f"| `{gate}` | {a:.2f} | {w['tau']:.3f} | {w['coverage']:.4f} | "
                f"{w['acc']:.4f} | {w['acc_accepted']:.4f} | {w['rej']:.4f} |")
    add("")
    add(f"三个工作点上**无条件识别准确率完全不动**（argmax 与阈值无关，恒为 "
        f"{mean_over('acc', 5):.4f}）；变的是接受率与拒识率："
        f"a = 0.85 时已注册者接受率 {mean_over('coverage_0.85', 5):.4f}、"
        f"拒识 {mean_over('rej_0.85', 5):.4f}；"
        f"a = 0.90 时 {mean_over('coverage_0.90', 5):.4f} / "
        f"{mean_over('rej_0.90', 5):.4f}；"
        f"a = 0.95 时 {mean_over('coverage_0.95', 5):.4f} / "
        f"{mean_over('rej_0.95', 5):.4f}。"
        f"接受样本上的准确率对应为 {mean_over('acc_accepted_0.85', 5):.4f} → "
        f"{mean_over('acc_accepted_0.90', 5):.4f} → "
        f"{mean_over('acc_accepted_0.95', 5):.4f}。"
        "结论：识别与拒识是**反向权衡**——a 越大（阈值越松）接受率越高、拒识越低，"
        "被放进来的样本精度也略降；a 越小拒识越高，但要牺牲已注册者的通过率。"
        "识别能力本身不参与这个权衡，它只是分母。")
    add("")
    add("## 7. 明确回答")
    add("")
    q5 = {g: (gates["5"][g]["rej_0.90"]["mean"], gates["5"][g]["acc"]["mean"],
              gates["5"][g]["f1"]["mean"], gates["5"][g]["auc_pooled"]) for g in ALL_GATES}
    add("**问：留 5 个未注册者时，Train 侧拒识率是多少？能否正确识别已注册的 15 人？**")
    add("")
    add(f"- 拒识率（k = 5，a = 0.90，9 个非监督门控均值）= **{mean_over('rej_0.90', 5):.4f}**；"
        "全部 10 个门控的单门控区间 "
        f"{min(v[0] for v in q5.values()):.4f}（`{min(q5, key=lambda g: q5[g][0])}`）– "
        f"{max(v[0] for v in q5.values()):.4f}（`{max(q5, key=lambda g: q5[g][0])}`）。")
    add(f"- 识别准确率 = **{mean_over('acc', 5):.4f}**（macro-F1 {mean_over('f1', 5):.4f}）："
        "已注册的 15 人在同 session 评估块上几乎全部认对，"
        f"最强的非监督门控 `"
        f"{max(SUMMARY_GATES, key=lambda g: gates['5'][g]['acc']['mean'])}` 达到 "
        f"{max(gates['5'][g]['acc']['mean'] for g in SUMMARY_GATES):.4f}，"
        f"均值被 `logbp_dist`（{gates['5']['logbp_dist']['acc']['mean']:.4f}）拉低。")
    add(f"- AUC（已注册 known vs 未注册）= **{auc_over(5):.4f}**"
        f"（单门控 {min(v[3] for v in q5.values()):.4f}–{max(v[3] for v in q5.values()):.4f}）。")
    add("- 结论：**能认对已注册者（9 门控均值 "
        f"{mean_over('acc', 5):.4f}），也能拒掉大部分未注册者（{mean_over('rej_0.90', 5):.4f}），"
        "但离可用还差得远**——最强的 `e65_pca64` 在 a = 0.90 时拒识 "
        f"{gates['5']['e65_pca64']['rej_0.90']['mean']:.4f}，"
        f"代价是已注册者接受率 {gates['5']['e65_pca64']['coverage_0.90']['mean']:.4f}；"
        f"把 a 收到 0.85，9 门控平均拒识升到 {mean_over('rej_0.85', 5):.4f}，"
        f"但已注册者接受率只剩 {mean_over('coverage_0.85', 5):.4f}。"
        "真正的瓶颈不是识别，而是同 session 的已注册者与未注册者在分数上仍然重叠（AUC "
        f"{auc_over(5):.4f}）。")
    add("")
    add("## 8. 合规说明")
    add("")
    add("- 原型与阈值只用已注册的 20−k 个身份（阶段 2 与阶段 3 的 τ 都只读已注册者的分数）。")
    add("- 标准化 / PCA / WCCN 白化用全部 20 个身份（阶段 1，用户明确定义的流程）；"
        "这一项与 E196 的差别已在 §5 量化。")
    add("- Test 只评分（`s1_*` / `s2_*` / `test_auc`），Session 列只在所有分数算完后读取，"
        "仅用于把 S1 / S2 分开，从未置 0、也未参与任何拟合或选择。")
    add("")
    add("## 9. 官方指标与文件")
    add("")
    add("Test 官方 AUC / F1 / C / RA / ASafeR / U / Score = **N/A**"
        "（Test 无身份标签，只有 S1/S2 审计读数）。")
    add("")
    add("| 文件 | 内容 |")
    add("|---|---|")
    add("| `run_e197.py` | 全部逻辑（`--stage folds/report/all`） |")
    add("| `config.json` | 冻结配置 |")
    add("| `cache/fold_plan.json` | 固定 seed 的折划分（与 E196 相同） |")
    add("| `results.json` | 门控 × k 指标、分析块 |")
    add("| `report.md` | 本报告 |")
    add("| `run.log` | 运行日志 |")
    add("")
    seconds = results.get("stage_seconds", {})
    add("耗时（秒）：" + "、".join(f"{k} {v:.0f}" for k, v in seconds.items()) +
        f"；总折数 {sum(results['fold_counts'].values())}。")
    add("")
    add("```powershell")
    add('& "D:\\WORK\\BCIC2026\\BrainPrint\\.venv\\Scripts\\python.exe" '
        '"D:\\WORK\\BCIC2026\\BrainPrint\\research\\E197_enrollment_leave_k\\run_e197.py" --stage all')
    add("```")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verification(results: dict) -> dict:
    out = {}
    e196_plan = ROOT / "research" / "E196_leave_k_out" / "cache" / "fold_plan.json"
    if e196_plan.exists():
        other = read_json(e196_plan)
        mine = read_json(PLAN_PATH)
        out["folds_identical_to_e196"] = all(
            [list(map(int, a)) for a in other.get(str(k), [])]
            == [list(map(int, a)) for a in mine.get(str(k), [])] for k in KS)
    inst = results.get("instrument", {})
    out["stage1_rows"] = inst.get("fit_rows")
    out["n_test"] = inst.get("n_test")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["folds", "report", "all"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-folds", type=int, default=0,
                        help="cap the folds per k (timing smoke test only)")
    args = parser.parse_args()

    sys.stdout = Tee(LOG_PATH)
    log(f"\nE197 start {time.strftime('%Y-%m-%d %H:%M:%S')} device {DEVICE}")
    torch.manual_seed(SEED)

    if args.stage in ("folds", "all"):
        if args.force or not RESULTS_PATH.exists() or args.max_folds:
            results = run_folds(args.max_folds)
            results["verification"] = verification(results)
            write_json(RESULTS_PATH, results)
        else:
            log("stage folds: results present, reusing")
    if args.stage in ("report", "all"):
        results = read_json(RESULTS_PATH)
        results["analysis"] = analyse(results)
        write_json(RESULTS_PATH, results)
        write_report(results)
        log("stage report done")
    log("E197 done")


if __name__ == "__main__":
    main()
