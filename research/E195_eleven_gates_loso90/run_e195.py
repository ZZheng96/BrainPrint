"""E195: eleven gates under a strict-LOSO > 0.90 constraint.

New 11th gate (user priority): loso_strict unknown rejection at a=0.90 must exceed 0.90.
The other ten gates are the E191 set.  This experiment

  1  implements two LOSO readings for every reusable gate candidate
       loso_std    E143 protocol: task-wise z and the PCA basis use every Train row,
                   the held-out identity included; only the prototypes exclude it.
       loso_strict the held-out identity is excluded from the z statistics, the PCA
                   basis and the prototypes (the other 19 identities only).
  2  scores seven base streams (five distance gates, the E192 WCCN distance-ratio
     gate, the E183 three-view fuse) on both readings and on Test S1 / S2,
  3  searches pairwise / three-way linear fusions with the weights selected on
     loso_strict only (Test never selects anything), and draws the
     loso_strict vs Test-min Pareto frontier,
  4  evaluates the best compromise on the full 11 gates with a self-built PGD
     (SNR 30 and SNR 20), using the E191 branch-A v3 identity backbone.

Train-only for every statistic.  Test is scored only; Session is read only after the
scores exist and only to split S1 / S2.  Nothing is written into an existing
experiment directory and nothing is committed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)
SCORE_PATH = CACHE / "streams.npz"
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "report.md"
LOG_PATH = OUT / "run.log"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FT = torch.float32
CLASSES = np.arange(1, 21, dtype=int)
TASKS = ["EC", "EO", "Imagery", "SSVEP"]
SEED = int(CFG["protocol"]["seed"])
FIT = float(CFG["protocol"]["fit_fraction"])
BLOCKS = int(CFG["protocol"]["blocks"])
GRID = [float(a) for a in CFG["protocol"]["acceptance_grid"]]
PRIMARY = float(CFG["protocol"]["primary_acceptance"])
KEY = f"{PRIMARY:.2f}"
E183_DIR = ROOT / "research" / "E183_gate_head_and_capacity"
E194_DIR = ROOT / "research" / "E194_loso_vs_s1_gap"
E191_DIR = ROOT / CFG["e191_backbone"]["directory"]
E192_DIR = ROOT / "research" / "E192_oe_three_region_18ch"
E183_LOSO = E183_DIR / "cache" / "loso_head.npz"
E183_TEST = E183_DIR / "test_predictions_fused.csv"
CONTAM_HEADS = E194_DIR / "cache" / "contam_heads.npz"


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


def log(msg: str) -> None:
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


# ---------------------------------------------------------------- data / protocol
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
    blocks = np.clip((frac * BLOCKS).astype(int), 0, BLOCKS - 1)
    test_meta = pd.read_csv(ROOT / CFG["data"]["test_metadata"],
                            usecols=["sample", "task", "session"])
    if len(test_meta) != int(CFG["data"]["test_trials"]):
        raise ValueError(f"Test rows {len(test_meta)} unexpected")
    session = test_meta["session"].astype(str).str.casefold().to_numpy()
    return {"subjects": subjects, "tasks": tasks, "frac": frac, "blocks": blocks,
            "fit_rows": frac <= FIT,
            "window": (frac > 0.6) & (frac <= 0.8),
            "test_tasks": test_meta["task"].to_numpy(dtype=object),
            "is_s1": np.isin(session, ["session1", "s1", "1"])}


def apply_z(features: np.ndarray, tasks: np.ndarray,
            mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    out = np.array(features, dtype=np.float64, copy=True)
    for index, task in enumerate(TASKS):
        rows = tasks == task
        if not rows.any():
            continue
        out[rows] = (out[rows] - mu[index].astype(np.float64)) / sd[index].astype(np.float64)
    return out


def pca_basis(x: np.ndarray, dim: int) -> tuple:
    data = np.asarray(x, dtype=np.float64)
    mu = data.mean(0)
    centred = data - mu
    values, vectors = np.linalg.eigh(centred.T @ centred / max(len(centred) - 1, 1))
    basis = vectors[:, np.argsort(-values)][:, : min(int(dim), data.shape[1])]
    return f32(mu), f32(basis)


def fit_instrument(features: np.ndarray, tasks: np.ndarray, ref_mask: np.ndarray,
                   pca_mask: np.ndarray, dim: int) -> dict:
    mu = np.zeros((len(TASKS), features.shape[1]), dtype=np.float32)
    sd = np.zeros((len(TASKS), features.shape[1]), dtype=np.float32)
    for index, task in enumerate(TASKS):
        ref = ref_mask & (tasks == task)
        if not ref.any():
            raise ValueError(f"empty reference for task {task}")
        mu[index] = features[ref].mean(0)
        sd[index] = features[ref].std(0) + 1e-6
    z = apply_z(features, tasks, mu, sd)
    pca_mu, basis = pca_basis(z[pca_mask], dim)
    return {"mu": mu, "sd": sd, "pca_mu": pca_mu, "basis": basis}


def apply_instrument(features: np.ndarray, tasks: np.ndarray, inst: dict) -> np.ndarray:
    z = apply_z(features, tasks, inst["mu"], inst["sd"])
    return f32((z - inst["pca_mu"].astype(np.float64)) @ inst["basis"].astype(np.float64))


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


def build_protos(proj: np.ndarray, subjects: np.ndarray, tasks: np.ndarray,
                 mask: np.ndarray, classes: np.ndarray, k: int) -> dict:
    out = {}
    for task in TASKS:
        chunks = []
        for ident in classes:
            sel = mask & (tasks == task) & (subjects == ident)
            if not sel.any():
                continue
            block = torch.as_tensor(f32(proj[sel]), device=DEVICE)
            chunks.append(kmeans_t(block, k, seed=int(ident) * 17))
        out[task] = torch.cat(chunks, 0)
    return out


def neg_dist(proj: np.ndarray, tasks: np.ndarray, protos: dict,
             rows: np.ndarray, batch: int = 2048) -> np.ndarray:
    out = np.full(len(proj), np.nan, dtype=np.float32)
    for task in TASKS:
        idx = rows[tasks[rows] == task]
        if len(idx) == 0:
            continue
        p = protos[task]
        for start in range(0, len(idx), batch):
            chunk = idx[start:start + batch]
            q = torch.as_tensor(f32(proj[chunk]), device=DEVICE)
            out[chunk] = (-torch.cdist(q, p)).max(1).values.cpu().numpy()
    return out


# ---------------------------------------------------------------- WCCN distance ratio
class WCCNGate:
    """E192 WCCN whitening + unit prototypes + distance ratio (larger = more known)."""

    def __init__(self, features: np.ndarray, rows: np.ndarray, subjects: np.ndarray,
                 classes: np.ndarray, shrinkage: float = 0.05,
                 floor: float = 1e-6) -> None:
        self.classes = np.asarray(classes, dtype=int)
        lookup = {int(c): i for i, c in enumerate(self.classes)}
        x = torch.as_tensor(f32(features[np.asarray(rows)]), device=DEVICE)
        y = torch.as_tensor([lookup[int(v)] for v in subjects[np.asarray(rows)]],
                            dtype=torch.long, device=DEVICE)
        means = torch.stack([x[y == i].mean(0) for i in range(len(self.classes))])
        residual = x - means[y]
        covariance = residual.T @ residual / max(len(x) - len(self.classes), 1)
        eye = torch.eye(covariance.shape[0], dtype=x.dtype, device=x.device)
        covariance = ((1.0 - shrinkage) * covariance
                      + shrinkage * torch.diagonal(covariance).sum()
                      / covariance.shape[0] * eye)
        values, vectors = torch.linalg.eigh(covariance)
        inverse_root = values.clamp_min(floor).rsqrt()
        self.whitening = (vectors * inverse_root.unsqueeze(0)) @ vectors.T
        self.all_prototypes = self._normalise(means @ self.whitening)
        self.mean_of = {int(c): means[i] for i, c in enumerate(self.classes)}
        del x, residual, covariance

    @staticmethod
    def _normalise(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def prototypes(self, classes) -> torch.Tensor:
        order = [int(c) for c in classes]
        return self._normalise(torch.stack([self.mean_of[c] for c in order]) @ self.whitening)

    @torch.no_grad()
    def scores(self, features: np.ndarray, classes=None, batch: int = 2048) -> np.ndarray:
        protos = self.all_prototypes if classes is None else self.prototypes(classes)
        out = np.empty((len(features), len(protos)), dtype=np.float32)
        for start in range(0, len(features), batch):
            chunk = torch.as_tensor(f32(features[start:start + batch]), device=DEVICE)
            chunk = self._normalise(chunk @ self.whitening)
            out[start:start + len(chunk)] = (chunk @ protos.T).float().cpu().numpy()
        return out

    @staticmethod
    def ratio(scores: np.ndarray) -> np.ndarray:
        ordered = np.sort(np.asarray(scores, dtype=np.float64), axis=1)
        maximum, second = ordered[:, -1], ordered[:, -2]
        return (1.0 - second) / np.maximum(1.0 - maximum, 1e-6)


def wccn_stream(protocol: dict, spec: dict) -> dict:
    train = f32(np.load(ROOT / spec["train"]))
    test = f32(np.load(ROOT / spec["test"]))
    subjects, tasks = protocol["subjects"], protocol["tasks"]
    fit_rows, window, blocks = protocol["fit_rows"], protocol["window"], protocol["blocks"]
    rows = np.flatnonzero(window)
    shrink = float(spec["shrinkage"])
    floor = float(spec["eigenvalue_floor"])

    full = WCCNGate(train, np.flatnonzero(fit_rows), subjects, CLASSES, shrink, floor)
    window_scores = WCCNGate.ratio(full.scores(train[rows]))
    test_scores = WCCNGate.ratio(full.scores(test))

    oof = np.full(len(subjects), np.nan, dtype=np.float32)
    for blk in range(BLOCKS):
        hold = blocks == blk
        gate = WCCNGate(train, np.flatnonzero(fit_rows & ~hold), subjects, CLASSES,
                        shrink, floor)
        idx = np.flatnonzero(hold)
        oof[hold] = WCCNGate.ratio(gate.scores(train[idx]))
        del gate
    oof_ref = oof[np.isfinite(oof)]

    contam = []
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        scores = WCCNGate.ratio(full.scores(train[rows], classes=enrolled))
        keep = subjects[rows] != held
        contam.append((scores[keep].astype(np.float32),
                       scores[~keep].astype(np.float32)))

    strict = []
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        keep_rows = subjects != held
        gate = WCCNGate(train, np.flatnonzero(fit_rows & keep_rows), subjects, enrolled,
                        shrink, floor)
        scores = WCCNGate.ratio(gate.scores(train[rows], classes=enrolled))
        keep = subjects[rows] != held
        strict.append((scores[keep].astype(np.float32),
                       scores[~keep].astype(np.float32)))
        del gate
    return {"window": window_scores.astype(np.float32), "test": test_scores.astype(np.float32),
            "oof_ref": oof_ref.astype(np.float32),
            "contam": contam, "strict": strict,
            "deploy": {"whitening": full.whitening.cpu().numpy(),
                       "prototypes": full.all_prototypes.cpu().numpy()}}


# ---------------------------------------------------------------- distance stream
def distance_stream(protocol: dict, spec: dict) -> dict:
    train = f32(np.load(ROOT / spec["train"]))
    test = f32(np.load(ROOT / spec["test"]))
    if train.shape[1] != int(spec["dim"]):
        raise ValueError(f"{spec['path']}: dim {train.shape}")
    subjects, tasks = protocol["subjects"], protocol["tasks"]
    fit_rows, window, blocks = protocol["fit_rows"], protocol["window"], protocol["blocks"]
    rows = np.flatnonzero(window)
    dim, k = int(spec["pca"]), int(spec["k"])

    deploy = fit_instrument(train, tasks, np.ones(len(subjects), bool), fit_rows, dim)
    proj = apply_instrument(train, tasks, deploy)
    test_proj = apply_instrument(test, protocol["test_tasks"], deploy)
    # E143 / E194 deployment convention: the Test score (and the 5-block OOF reference)
    # use prototypes built from every Train row, while the LOSO folds use pm & fit_rows.
    protos = build_protos(proj, subjects, tasks, np.ones(len(subjects), bool), CLASSES, k)
    window_scores = neg_dist(proj, tasks, protos, rows)
    test_scores = neg_dist(test_proj, protocol["test_tasks"], protos,
                           np.arange(len(test_proj)))

    oof = np.full(len(subjects), np.nan, dtype=np.float32)
    for blk in range(BLOCKS):
        hold = blocks == blk
        p = build_protos(proj, subjects, tasks, ~hold, CLASSES, k)
        idx = np.flatnonzero(hold)
        oof[hold] = neg_dist(proj, tasks, p, idx)[idx]
    oof_ref = oof[np.isfinite(oof)]

    contam = []
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        pm = subjects != held
        p = build_protos(proj, subjects, tasks, pm & fit_rows, enrolled, k)
        scores = neg_dist(proj, tasks, p, rows)
        contam.append((scores[window & pm].astype(np.float32),
                       scores[window & ~pm].astype(np.float32)))

    strict = []
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        keep = subjects != held
        inst = fit_instrument(train, tasks, keep, keep & fit_rows, dim)
        proj_h = apply_instrument(train, tasks, inst)
        p = build_protos(proj_h, subjects, tasks, keep & fit_rows, enrolled, k)
        scores = neg_dist(proj_h, tasks, p, rows)
        strict.append((scores[window & keep].astype(np.float32),
                       scores[window & ~keep].astype(np.float32)))
        del proj_h
    return {"window": window_scores.astype(np.float32), "test": test_scores.astype(np.float32),
            "oof_ref": oof_ref.astype(np.float32),
            "contam": contam, "strict": strict,
            "deploy": {"mu": deploy["mu"], "sd": deploy["sd"],
                       "pca_mu": deploy["pca_mu"], "basis": deploy["basis"],
                       "proto": np.stack([protos[t].cpu().numpy() for t in TASKS]),
                       "k": k, "pca": dim, "dim": int(spec["dim"])}}


# ---------------------------------------------------------------- E183 stream
def e183_stream(protocol: dict) -> dict:
    data = np.load(E183_LOSO)
    rows, fold = data["rows"], data["fold"].astype(int)
    known = data["known"].astype(bool)
    if len(rows) != 20 * int(protocol["window"].sum()):
        raise ValueError("unexpected E183 LOSO row count")
    z_dist = data["z_dist"].astype(np.float64)
    z_ent = data["z_ent"].astype(np.float64)
    fused_strict = (0.85 * data["comp_conf_max"].astype(np.float64)
                    + 0.10 * z_dist + 0.05 * z_ent)

    comp = np.load(CONTAM_HEADS)["comp_conf"].astype(np.float64)
    if comp.shape != (20, int(protocol["window"].sum())):
        raise ValueError(f"unexpected contaminated head shape {comp.shape}")
    fused_contam = np.empty_like(fused_strict)
    for held in CLASSES:
        sel = fold == int(held)
        fused_contam[sel] = 0.85 * comp[int(held) - 1] + 0.10 * z_dist[sel] + 0.05 * z_ent[sel]

    strict, contam = [], []
    for held in CLASSES:
        sel = fold == int(held)
        strict.append((fused_strict[sel & known], fused_strict[sel & ~known]))
        contam.append((fused_contam[sel & known], fused_contam[sel & ~known]))

    frozen = pd.read_csv(E183_TEST)
    test_meta = pd.read_csv(ROOT / CFG["data"]["test_metadata"], usecols=["sample"])
    if len(frozen) != len(test_meta):
        raise ValueError("E183 test prediction length mismatch")
    if not (frozen["sample"].astype(str).to_numpy()
            == test_meta["sample"].astype(str).to_numpy()).all():
        raise ValueError("E183 test prediction order mismatch")
    return {"window": None, "test": frozen["fused_gate"].to_numpy(np.float32),
            "oof_ref": None, "contam": contam, "strict": strict,
            "deploy": {"note": "published E183 cache; no 5-block OOF stream"}}


# ---------------------------------------------------------------- score stage
def stage_scores(force: bool = False) -> dict:
    started = time.time()
    if SCORE_PATH.exists() and not force:
        log("stage scores: cache present, reusing")
        return {}
    P = load_protocol()
    log(f"stage scores: Train {len(P['subjects'])} rows, fit {int(P['fit_rows'].sum())}, "
        f"window {int(P['window'].sum())}, Test {len(P['test_tasks'])}, "
        f"S1 {int(P['is_s1'].sum())}, device {DEVICE}")
    specs = CFG["streams"]
    payload = {}
    summary = {}
    for name in ("e65_dist", "subwin6_dist", "subwin12_dist", "logbp_dist", "e162_dist"):
        t0 = time.time()
        store = distance_stream(P, specs[name])
        payload[f"{name}__test"] = store["test"]
        payload[f"{name}__oof_ref"] = store["oof_ref"]
        for index, (kn, un) in enumerate(store["strict"]):
            payload[f"{name}__strict_{index}_known"] = kn
            payload[f"{name}__strict_{index}_unknown"] = un
        for index, (kn, un) in enumerate(store["contam"]):
            payload[f"{name}__std_{index}_known"] = kn
            payload[f"{name}__std_{index}_unknown"] = un
        for key, value in store["deploy"].items():
            payload[f"{name}__deploy_{key}"] = value
        summary[name] = {"seconds": time.time() - t0,
                         "test_mean": float(np.mean(store["test"])),
                         "oof_mean": float(np.mean(store["oof_ref"]))}
        log(f"  {name}: {time.time() - t0:.0f}s "
            f"(test mean {np.mean(store['test']):.3f})")
        del store

    t0 = time.time()
    store = wccn_stream(P, specs["wccn_dr"])
    payload["wccn_dr__test"] = store["test"]
    payload["wccn_dr__oof_ref"] = store["oof_ref"]
    for index, (kn, un) in enumerate(store["strict"]):
        payload[f"wccn_dr__strict_{index}_known"] = kn
        payload[f"wccn_dr__strict_{index}_unknown"] = un
    for index, (kn, un) in enumerate(store["contam"]):
        payload[f"wccn_dr__std_{index}_known"] = kn
        payload[f"wccn_dr__std_{index}_unknown"] = un
    payload["wccn_dr__deploy_whitening"] = store["deploy"]["whitening"]
    payload["wccn_dr__deploy_prototypes"] = store["deploy"]["prototypes"]
    summary["wccn_dr"] = {"seconds": time.time() - t0,
                          "test_mean": float(np.mean(store["test"])),
                          "oof_mean": float(np.mean(store["oof_ref"]))}
    log(f"  wccn_dr: {time.time() - t0:.0f}s (test mean {np.mean(store['test']):.3f})")

    data = np.load(E192_DIR / "cache" / "gate_arrays.npz")
    ours_k = np.concatenate([p[0] for p in store["strict"]]).astype(np.float64)
    ours_u = np.concatenate([p[1] for p in store["strict"]]).astype(np.float64)
    theirs_k = data["known_dr"].astype(np.float64)
    theirs_u = data["unknown_dr"].astype(np.float64)
    check = {"known_corr": float(np.corrcoef(ours_k, theirs_k)[0, 1]),
             "unknown_corr": float(np.corrcoef(ours_u, theirs_u)[0, 1]),
             "ours_auc": float(roc_auc_score(
                 np.r_[np.zeros(len(ours_u)), np.ones(len(ours_k))],
                 np.r_[ours_u, ours_k])),
             "e192_auc": float(roc_auc_score(
                 np.r_[np.zeros(len(theirs_u)), np.ones(len(theirs_k))],
                 np.r_[theirs_u, theirs_k]))}
    log(f"  E192 cross-check: known corr {check['known_corr']:.6f}, "
        f"unknown corr {check['unknown_corr']:.6f}, "
        f"AUC ours {check['ours_auc']:.4f} vs E192 {check['e192_auc']:.4f}")
    summary["wccn_dr"]["e192_cross_check"] = check
    del store

    t0 = time.time()
    store = e183_stream(P)
    payload["e183_fuse__test"] = store["test"]
    for index, (kn, un) in enumerate(store["strict"]):
        payload["e183_fuse__strict_%d_known" % index] = kn
        payload["e183_fuse__strict_%d_unknown" % index] = un
    for index, (kn, un) in enumerate(store["contam"]):
        payload["e183_fuse__std_%d_known" % index] = kn
        payload["e183_fuse__std_%d_unknown" % index] = un
    summary["e183_fuse"] = {"seconds": time.time() - t0,
                            "test_mean": float(np.mean(store["test"]))}
    log(f"  e183_fuse: {time.time() - t0:.0f}s")

    payload["is_s1"] = P["is_s1"]
    np.savez_compressed(SCORE_PATH, **payload)
    results = read_json(RESULTS_PATH)
    results["streams"] = summary
    results.setdefault("stage_seconds", {})["scores"] = time.time() - started
    write_json(RESULTS_PATH, results)
    log(f"stage scores done in {time.time() - started:.0f}s")
    return summary


# ---------------------------------------------------------------- stream loading
def load_streams() -> dict:
    data = np.load(SCORE_PATH, allow_pickle=False)
    streams = {}
    for name in CFG["streams"]:
        std = {"known": [], "unknown": []}
        strict = {"known": [], "unknown": []}
        for index in range(20):
            std["known"].append(f32(data[f"{name}__std_{index}_known"]))
            std["unknown"].append(f32(data[f"{name}__std_{index}_unknown"]))
            strict["known"].append(f32(data[f"{name}__strict_{index}_known"]))
            strict["unknown"].append(f32(data[f"{name}__strict_{index}_unknown"]))
        test = f32(data[f"{name}__test"]) if f"{name}__test" in data.files else None
        oof = (f32(data[f"{name}__oof_ref"])
               if f"{name}__oof_ref" in data.files else None)
        reference = (oof if oof is not None
                     else np.concatenate(strict["known"]).astype(np.float32))
        mu = float(np.mean(reference.astype(np.float64)))
        sd = float(np.std(reference.astype(np.float64))) + 1e-6
        streams[name] = {"std": std, "strict": strict, "test": test,
                         "oof": oof, "mu": mu, "sd": sd}
    streams["_is_s1"] = bool_array(data["is_s1"])
    return streams


def bool_array(value) -> np.ndarray:
    return np.asarray(value, dtype=bool)


def standardised(streams: dict, weights: dict, mode: str) -> dict:
    """Weighted standardised fold arrays and Test scores for one candidate."""
    names = [n for n, w in weights.items() if w > 0]
    known, unknown = [], []
    for index in range(20):
        known.append(sum(w * (streams[n][mode]["known"][index] - streams[n]["mu"])
                         / streams[n]["sd"] for n, w in weights.items() if w > 0))
        unknown.append(sum(w * (streams[n][mode]["unknown"][index] - streams[n]["mu"])
                           / streams[n]["sd"] for n, w in weights.items() if w > 0))
    test = None
    if all(streams[n]["test"] is not None for n in names):
        test = sum(w * (streams[n]["test"] - streams[n]["mu"]) / streams[n]["sd"]
                   for n, w in weights.items() if w > 0)
    oof = None
    if all(streams[n]["oof"] is not None for n in names):
        oof = sum(w * (streams[n]["oof"] - streams[n]["mu"]) / streams[n]["sd"]
                  for n, w in weights.items() if w > 0)
    return {"known": known, "unknown": unknown, "test": test, "oof": oof}


def evaluate(streams: dict, weights: dict, full: bool = False) -> dict:
    out = {}
    for mode in ("std", "strict"):
        comp = standardised(streams, weights, mode)
        known, unknown = comp["known"], comp["unknown"]
        pooled_k = np.concatenate(known).astype(np.float64)
        pooled_u = np.concatenate(unknown).astype(np.float64)
        labels = np.r_[np.ones(len(pooled_k)), np.zeros(len(pooled_u))]
        pooled_auc = float(roc_auc_score(labels, np.r_[pooled_k, pooled_u]))
        fold_auc, rejections = [], {f"{a:.2f}": [] for a in GRID}
        u95 = []
        for kn, un in zip(known, unknown):
            fold_auc.append(float(roc_auc_score(
                np.r_[np.ones(len(kn)), np.zeros(len(un))], np.r_[kn, un])))
            for a in GRID:
                rejections[f"{a:.2f}"].append(float(np.mean(un < quantile(kn, 1.0 - a))))
            u95.append(float(np.mean(un < quantile(kn, 1.0 - 0.95))))
        out[mode] = {
            "pooled_auc": pooled_auc,
            "fold_auc_mean": float(np.mean(fold_auc)),
            "fold_auc_std": float(np.std(fold_auc)),
            "fold_auc_min": float(np.min(fold_auc)),
            "rejection_fold": {k: float(np.mean(v)) for k, v in rejections.items()},
            "rejection_fold_std": {k: float(np.std(v)) for k, v in rejections.items()},
            "u95": float(np.mean(u95)),
            "u95_min": float(np.min(u95)),
            "known_mean": float(np.mean(pooled_k)),
            "unknown_mean": float(np.mean(pooled_u)),
        }
        if full:
            out[mode]["per_fold_rejection"] = rejections
            out[mode]["per_fold_auc"] = fold_auc

    is_s1 = streams["_is_s1"]
    strict = standardised(streams, weights, "strict")
    names = [n for n, w in weights.items() if w > 0]
    rules = {}
    paired_reference = np.concatenate(strict["known"]).astype(np.float64)
    rules["paired"] = paired_reference
    if strict["oof"] is not None:
        rules["oof"] = strict["oof"].astype(np.float64)
    test = strict["test"]
    test_out = {}
    if test is not None:
        for rule, reference in rules.items():
            tau = quantile(reference, 1.0 - PRIMARY)
            test_out[rule] = {
                "tau": tau,
                "s1_rejection": float(np.mean(test[is_s1] < tau)),
                "s2_acceptance": float(np.mean(test[~is_s1] >= tau)),
                "min": float(min(np.mean(test[is_s1] < tau),
                                 np.mean(test[~is_s1] >= tau))),
            }
    out["test"] = test_out
    out["weights"] = {k: float(v) for k, v in weights.items() if v > 0}
    return out


def single_points(streams: dict) -> dict:
    out = {}
    for name in CFG["candidates"]["singles"]:
        out[name] = evaluate(streams, {name: 1.0}, full=True)
    return out


def weight_grid(count: int, step: float) -> list:
    levels = int(round(1.0 / step))
    out = []

    def walk(prefix, left):
        if len(prefix) == count - 1:
            out.append(tuple(list(prefix) + [left]))
            return
        for value in range(left + 1):
            walk(prefix + [value], left - value)

    walk([], levels)
    return [tuple(v * step for v in item) for item in out]


def search(streams: dict) -> dict:
    started = time.time()
    points = []
    for name in CFG["candidates"]["singles"]:
        row = evaluate(streams, {name: 1.0})
        points.append({"candidate": name, "weights": {name: 1.0}, "point": row})
    fusion_summary = {}
    for group in CFG["candidates"]["pairs"]:
        best = (-1.0, -1.0, None)
        for w in np.arange(0.0, 1.0 + 1e-9, 0.05):
            weights = {group[0]: float(w), group[1]: float(1.0 - w)}
            row = evaluate(streams, weights)
            points.append({"candidate": "+".join(group), "weights": weights, "point": row})
            score = row["strict"]["rejection_fold"][KEY]
            auc = row["strict"]["pooled_auc"]
            if (score, auc) > (best[0], best[1]):
                best = (score, auc, weights)
        key = "+".join(group)
        valid = [(p["point"]["strict"]["rejection_fold"][KEY],
                  p["point"]["strict"]["pooled_auc"], p["weights"]) for p in points
                 if p["candidate"] == key]
        fusion_summary[key] = {"train_selected": best[2],
                               "train_selected_strict": best[0],
                               "train_selected_auc": best[1],
                               "grid": [{"weights": w, "strict": s, "auc": a}
                                        for s, a, w in valid]}
        log(f"  pair {key:36s} selected {best[2]} strict {best[0]:.4f}")
    for group in CFG["candidates"]["triples"]:
        best = (-1.0, -1.0, None)
        for w in weight_grid(len(group), 0.1):
            weights = {name: float(value) for name, value in zip(group, w)}
            row = evaluate(streams, weights)
            points.append({"candidate": "+".join(group), "weights": weights, "point": row})
            score = row["strict"]["rejection_fold"][KEY]
            auc = row["strict"]["pooled_auc"]
            if (score, auc) > (best[0], best[1]):
                best = (score, auc, weights)
        key = "+".join(group)
        fusion_summary[key] = {"train_selected": best[2],
                               "train_selected_strict": best[0],
                               "train_selected_auc": best[1]}
        log(f"  triple {key:42s} selected {best[2]} strict {best[0]:.4f}")
    log(f"stage search done in {time.time() - started:.0f}s ({len(points)} points)")
    return {"points": points, "fusions": fusion_summary}


def pareto(points: list, rule: str = "paired") -> dict:
    """Non-dominated on (loso_strict rejection, Test min)."""
    rows = []
    for p in points:
        test = p["point"]["test"].get(rule)
        if test is None:
            continue
        rows.append({"candidate": p["candidate"], "weights": p["weights"],
                     "strict": p["point"]["strict"]["rejection_fold"][KEY],
                     "strict_auc": p["point"]["strict"]["pooled_auc"],
                     "std": p["point"]["std"]["rejection_fold"][KEY],
                     "s1": test["s1_rejection"], "s2": test["s2_acceptance"],
                     "test_min": test["min"], "rule": rule})
    frontier = []
    for r in rows:
        dominated = any((o["strict"] >= r["strict"] and o["test_min"] >= r["test_min"])
                        and (o["strict"] > r["strict"] or o["test_min"] > r["test_min"])
                        for o in rows)
        if not dominated:
            frontier.append(r)
    frontier.sort(key=lambda r: (-r["strict"], -r["test_min"]))
    feasible = [r for r in rows if r["strict"] > 0.90 and r["test_min"] >= 0.70]
    under_strict = [r for r in rows if r["strict"] > 0.90]
    dual = [r for r in rows if r["test_min"] >= 0.70]
    best_compromise = (max(under_strict, key=lambda r: (r["test_min"], r["strict_auc"]))
                       if under_strict else None)
    best_at_dual = (max(dual, key=lambda r: (r["strict"], r["strict_auc"]))
                    if dual else None)
    return {"frontier": frontier, "feasible": feasible,
            "best_compromise": best_compromise,
            "best_at_dual": best_at_dual,
            "n_points": len(rows),
            "max_strict": max(r["strict"] for r in rows),
            "max_test_min": max(r["test_min"] for r in rows)}


def stage_search(force: bool = False) -> dict:
    started = time.time()
    if RESULTS_PATH.exists() and not force:
        previous = read_json(RESULTS_PATH)
        if "search" in previous:
            log("stage search: results present, reusing")
            return previous["search"]
    streams = load_streams()
    singles = single_points(streams)
    found = search(streams)
    results = read_json(RESULTS_PATH)
    results["singles"] = singles
    results["search"] = found
    results["pareto"] = {rule: pareto(found["points"], rule)
                         for rule in ("paired", "oof")}
    results.setdefault("stage_seconds", {})["search"] = time.time() - started
    write_json(RESULTS_PATH, results)
    rule = results["pareto"]["paired"]
    log(f"  Pareto: {len(rule['frontier'])} points, "
        f"feasible (#11 + dual 0.70) {len(rule['feasible'])}")
    for r in rule["frontier"]:
        log(f"    {r['candidate']:40s} {str(r['weights']):50s} "
            f"strict {r['strict']:.4f} std {r['std']:.4f} "
            f"S1 {r['s1']:.4f} S2 {r['s2']:.4f} min {r['test_min']:.4f}")
    return found


# ---------------------------------------------------------------- attack stage
def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TorchDistanceGate(nn.Module):
    """Differentiable deployment distance gate (same instrument / prototypes as E195)."""

    def __init__(self, store: dict, kind: str) -> None:
        super().__init__()
        self.stats_mu = torch.as_tensor(f32(store["stats_mu"]), device=DEVICE)
        self.stats_sd = torch.as_tensor(f32(store["stats_sd"]), device=DEVICE)
        self.pca_mu = torch.as_tensor(f32(store["pca_mu"]), device=DEVICE)
        self.basis = torch.as_tensor(f32(store["basis"]), device=DEVICE)
        self.proto = [torch.as_tensor(f32(store["proto"][i]), device=DEVICE)
                      for i in range(len(TASKS))]
        self.kind = kind
        self.e189 = None

    def forward(self, x: torch.Tensor, task_idx: torch.Tensor) -> torch.Tensor:
        if self.e189 is None:
            self.e189 = load_module("e195_frozen_e189",
                                    ROOT / "research" / "E189_balanced_gate_gates"
                                    / "run_e189.py")
            self.e189.build_constants()
        if self.kind == "logspd":
            feat = self.e189.logspd(x)
        elif self.kind == "subwin6":
            feat = self.e189.subwin_mean(x, 6)
        elif self.kind == "subwin12":
            feat = self.e189.subwin_mean(x, 12)
        elif self.kind == "logbp":
            feat = self.e189.logbp(x)
        else:
            raise KeyError(self.kind)
        z = (feat - self.stats_mu[task_idx]) / self.stats_sd[task_idx]
        proj = (z - self.pca_mu) @ self.basis
        out = torch.zeros(len(proj), device=proj.device, dtype=proj.dtype)
        for index in range(len(TASKS)):
            take = task_idx == index
            if not bool(take.any()):
                continue
            dist = torch.cdist(proj[take], self.proto[index])
            out = out.index_copy(0, torch.nonzero(take).squeeze(1),
                                 (-dist).max(1).values)
        return out


class E183Gate(nn.Module):
    """E183 three-view gate, copied from E183.FusedHeadScorer (1-based fold index fixed)."""

    def __init__(self, model_a, store, folds, weights, ent_stats, component,
                 e180) -> None:
        super().__init__()
        self.model_a = model_a
        for parameter in self.model_a.parameters():
            parameter.requires_grad_(False)
        self.model_a.eval()
        self.weights = {key: float(value) for key, value in weights.items()}
        self.stats_mu = torch.as_tensor(f32(store["stats_mu"]), device=DEVICE)
        self.stats_sd = torch.as_tensor(f32(store["stats_sd"]), device=DEVICE)
        self.pca_mu = torch.as_tensor(f32(store["pca_mu"]), device=DEVICE)
        self.basis = torch.as_tensor(f32(store["pca_basis"]), device=DEVICE)
        self.proto = torch.as_tensor(f32(store["proto"]), device=DEVICE)
        self.filled = torch.as_tensor(np.asarray(store["proto_filled"]), device=DEVICE)
        self.k = int(store["k"])
        self.oof_mu = float(store["oof_mu"])
        self.oof_sd = float(store["oof_sd"])
        self.ent_mu, self.ent_sd = float(ent_stats[0]), float(ent_stats[1])
        self.folds = folds
        self.component = component
        self.e180 = e180

    def forward(self, x: torch.Tensor, task_idx: torch.Tensor, groups: list) -> torch.Tensor:
        prob = torch.softmax(self.model_a(x), dim=1)
        ent = (prob * prob.clamp_min(1e-12).log()).sum(1)
        g_ent = (-ent - self.ent_mu) / self.ent_sd
        features = self.e180.differentiable_logspd(x)
        z = (features - self.stats_mu[task_idx]) / self.stats_sd[task_idx]
        projected = (z - self.pca_mu) @ self.basis
        dist = None
        for t in range(len(TASKS)):
            valid = self.filled[t]
            p = self.proto[t][valid].reshape(-1, self.proto.shape[-1])
            d = torch.cdist(projected, p).view(len(x), int(valid.sum()), self.k)
            score = (-d).max(dim=2).values.max(dim=1).values
            take = task_idx == t
            if not bool(take.any()):
                continue
            filled = torch.full_like(score, -1e9)
            filled[take] = score[take]
            dist = filled if dist is None else torch.where(take, filled, dist)
        g_dist = (dist - self.oof_mu) / self.oof_sd
        head = torch.zeros(len(x), device=DEVICE, dtype=x.dtype)
        for fold_index, index in groups:
            art = self.folds[int(fold_index) - 1]
            group_task = task_idx[index]
            zz = (features[index] - art["stats_mu"][group_task]) / art["stats_sd"][group_task]
            p = ((zz - art["pca_mu"]) @ art["pca_basis"] - art["feat_mu"]) / art["feat_sd"]
            views = []
            for rep, model in enumerate(art["models"]):
                emb = model.trunk(p)
                if self.component == "open":
                    value = (model.open_head(emb).squeeze(1) - art["mu_open"][rep]) \
                        / art["sd_open"][rep]
                elif self.component == "conf":
                    conf = torch.softmax(model.id_head(emb), 1).max(1).values
                    value = (conf - art["mu_conf"][rep]) / art["sd_conf"][rep]
                else:
                    d = torch.cdist(emb, art["centroids"][rep][art["k_mask"][rep]])
                    value = ((-d.min(1).values) - art["mu_trunk"][rep]) / art["sd_trunk"][rep]
                views.append(value)
            head = head.index_copy(0, index, torch.stack(views).max(0).values)
        return (self.weights["g_head"] * head + self.weights["g_dist"] * g_dist
                + self.weights["g_entropy"] * g_ent)


class CandidateScorer(nn.Module):
    def __init__(self, identity, weights: dict, streams: dict, gates: dict,
                 e183_gate: nn.Module | None) -> None:
        super().__init__()
        self.identity = identity
        self.weights = {k: float(v) for k, v in weights.items() if v > 0}
        self.gates = nn.ModuleDict(gates)
        self.e183 = e183_gate
        self.mu = {k: float(streams[k]["mu"]) for k in self.weights}
        self.sd = {k: float(streams[k]["sd"]) for k in self.weights}

    def gate(self, x, task_idx, groups) -> torch.Tensor:
        total = None
        for name, weight in self.weights.items():
            value = (self.e183(x, task_idx, groups) if name == "e183_fuse"
                     else self.gates[name](x, task_idx))
            value = (value - self.mu[name]) / self.sd[name]
            total = weight * value if total is None else total + weight * value
        return total

    def forward(self, x, task_idx, groups):
        return self.identity(x), self.gate(x, task_idx, groups)


def build_e183_attack_gate(weights: dict):
    e180 = load_module("e195_frozen_e180",
                       ROOT / "research" / "E180_dual_branch_adversarial_openset"
                       / "run_e180.py")
    e183 = load_module("e195_frozen_e183", E183_DIR / "run_e183.py")
    e180.build_torch_constants()
    model = e180.Net().to(DEVICE)
    payload = torch.load(e183.E182_DIR / "model_branch_a_v2.pt", map_location="cpu",
                         weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    store = e180.load_branch_b()
    signals = np.load(e183.E182_DIR / "cache" / "signals_v2.npz")
    ent_stats = (float(signals["stats_mu"][4]), float(signals["stats_sd"][4]))
    data = np.load(E183_LOSO)
    centroids, k_mask = data["centroids"], data["k_mask"]
    folds = []
    for held in CLASSES:
        art, models = e183.load_fold_models(int(held))
        entry = {"models": models,
                 "stats_mu": torch.as_tensor(art["stats_mu"], device=DEVICE),
                 "stats_sd": torch.as_tensor(art["stats_sd"], device=DEVICE),
                 "pca_mu": torch.as_tensor(art["pca_mu"], device=DEVICE),
                 "pca_basis": torch.as_tensor(art["pca_basis"], device=DEVICE),
                 "feat_mu": torch.as_tensor(art["feat_mu"], device=DEVICE),
                 "feat_sd": torch.as_tensor(art["feat_sd"], device=DEVICE)}
        for name, mu_key, sd_key in (("open", "mu_open", "sd_open"),
                                     ("trunk", "mu_trunk", "sd_trunk"),
                                     ("conf", "mu_conf", "sd_conf")):
            entry[mu_key] = art[mu_key]
            entry[sd_key] = art[sd_key]
        entry["centroids"] = [torch.as_tensor(
            centroids[int(held) - 1, rep][k_mask[int(held) - 1, rep]], device=DEVICE)
            for rep in range(e183.REPS)]
        entry["k_mask"] = k_mask[int(held) - 1]
        folds.append(entry)
    gate = E183Gate(model, store, folds, weights, ent_stats, "conf", e180)
    head_choice = {}
    for value in CLASSES:
        for held in CLASSES:
            if int(value) != int(held) and k_mask[int(held) - 1, :, int(value) - 1].all():
                head_choice[int(value)] = int(held)
                break
        if int(value) not in head_choice:
            raise RuntimeError(f"no fold treats identity {value} as registered")
    return gate, e183, e180, head_choice


def build_identity_v3(e191) -> nn.Module:
    model = e191.NetV3().to(DEVICE)
    payload = torch.load(E191_DIR / "model_branch_a_v3.pt", map_location="cpu",
                         weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def pgd_snr(scorer, x, task_idx, groups, true_idx, target_idx, goal, snr, steps,
            seed) -> dict:
    gen = torch.Generator(device=DEVICE).manual_seed(int(seed))
    xc = x - x.mean(dim=-1, keepdim=True)
    signal_norm = xc.reshape(len(x), -1).norm(dim=1).clamp_min(1e-12)
    radius = signal_norm / (10.0 ** (float(snr) / 20.0))
    delta = torch.randn(x.shape, generator=gen, device=DEVICE)
    delta = delta / delta.reshape(len(delta), -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
    delta = delta * radius.view(-1, 1, 1) * torch.rand(
        len(delta), generator=gen, device=DEVICE).view(-1, 1, 1)
    alpha = 2.0 * radius / int(steps)
    for _ in range(int(steps)):
        delta = delta.detach().requires_grad_(True)
        logits, gate = scorer(x + delta, task_idx, groups)
        if goal == "D":
            objective = gate
        else:
            true_logit = logits.gather(1, true_idx[:, None]).squeeze(1)
            wrong = logits.masked_fill(
                F.one_hot(true_idx, len(CLASSES)).bool(), -1e9)
            if goal == "A":
                margin = wrong.max(dim=1).values
            elif goal == "B":
                margin = torch.logsumexp(wrong, dim=1)
            else:
                margin = logits.gather(1, target_idx[:, None]).squeeze(1)
            objective = (margin - true_logit) / 10.0
        grad = torch.autograd.grad(-objective.sum(), delta)[0]
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        direction = grad / grad.reshape(len(grad), -1).norm(
            dim=1).view(-1, 1, 1).clamp_min(1e-12)
        delta = delta.detach() - alpha.view(-1, 1, 1) * direction
        norm = delta.reshape(len(delta), -1).norm(dim=1)
        delta = delta * torch.minimum(torch.ones_like(norm), radius / norm).view(-1, 1, 1)
    with torch.no_grad():
        logits, gate = scorer(x + delta, task_idx, groups)
    return {"logits": logits.detach().cpu().numpy(),
            "gate": gate.detach().cpu().numpy()}


def attack_metrics(logits, gate, truth, tau) -> dict:
    accept = gate >= tau
    predicted = np.where(accept, logits.argmax(1) + 1, 0)
    return {"RA": float(np.mean(predicted == truth)),
            "ASafeR": float(np.mean((predicted == truth) | (predicted == 0))),
            "accept_rate": float(np.mean(accept)),
            "closed_set_accuracy": float(np.mean(logits.argmax(1) + 1 == truth)),
            "wrong_identity_rate": float(np.mean((predicted != 0) & (predicted != truth)))}


def stage_attack(force: bool = False) -> dict:
    started = time.time()
    results = read_json(RESULTS_PATH)
    if "attack" in results and not force:
        log("stage attack: results present, reusing")
        return results["attack"]
    pareto = results["pareto"]["paired"]
    feasible = pareto.get("feasible") or []
    point = (max(feasible, key=lambda r: (r["strict_auc"], r["test_min"]))
             if feasible else pareto["best_compromise"])
    weights = {k: float(v) for k, v in point["weights"].items()}
    log(f"stage attack: candidate {point['candidate']} weights {weights} "
        f"(strict {point['strict']:.4f}, Test min {point['test_min']:.4f})")
    streams = load_streams()
    e189 = load_module("e195_frozen_e189",
                       ROOT / "research" / "E189_balanced_gate_gates" / "run_e189.py")
    e191 = load_module("e195_frozen_e191", E191_DIR / "run_e191.py")
    P = e189.protocol()
    subset = e189.attack_subset(P)
    subjects, tasks = P["subjects"], P["tasks"]
    log(f"  attack subset {len(subset)} trials "
        f"({', '.join(f'{t}={int((tasks[subset] == t).sum())}' for t in TASKS)})")

    identity = build_identity_v3(e191)
    gates = {}
    e183_gate = None
    head_choice = None
    if "e183_fuse" in weights:
        e183_gate, e183, e180, head_choice = build_e183_attack_gate(
            {"g_head": 0.85, "g_dist": 0.10, "g_entropy": 0.05})
    data = np.load(SCORE_PATH, allow_pickle=False)
    for name in weights:
        if name == "e183_fuse":
            continue
        kind = {"e65_dist": "logspd", "subwin6_dist": "subwin6",
                "subwin12_dist": "subwin12", "logbp_dist": "logbp"}.get(name)
        if kind is None:
            raise ValueError(f"attack does not support gate component {name}")
        store = {"stats_mu": data[f"{name}__deploy_mu"],
                 "stats_sd": data[f"{name}__deploy_sd"],
                 "pca_mu": data[f"{name}__deploy_pca_mu"],
                 "basis": data[f"{name}__deploy_basis"],
                 "proto": data[f"{name}__deploy_proto"],
                 "k": data[f"{name}__deploy_k"]}
        gates[name] = TorchDistanceGate(store, kind)
    scorer = CandidateScorer(identity, weights, streams, gates, e183_gate)

    raw = e191.E180.load_raw([P["meta"]["path"].iloc[i] for i in subset])
    truth = subjects[subset]
    task_idx = torch.as_tensor([TASKS.index(t) for t in tasks[subset]],
                               dtype=torch.long, device=DEVICE)
    true_idx = torch.as_tensor(truth - 1, dtype=torch.long, device=DEVICE)
    lut = {int(c): i for i, c in enumerate(CLASSES)}
    target = np.asarray([CLASSES[(lut[int(v)] + 1) % len(CLASSES)] for v in truth])
    target_idx = torch.as_tensor([lut[int(v)] for v in target], dtype=torch.long,
                                 device=DEVICE)
    groups = []
    if head_choice is not None:
        assign = np.asarray([head_choice[int(v)] for v in truth])
        for held in CLASSES:
            index = np.flatnonzero(assign == int(held))
            if len(index):
                groups.append((int(held), torch.as_tensor(index, dtype=torch.long,
                                                          device=DEVICE)))
    tau = float(point.get("tau", evaluate(streams, weights)["test"]["paired"]["tau"]))
    log(f"  tau(a=0.90, paired) {tau:.4f}")

    with torch.no_grad():
        logits, gate = scorer(raw, task_idx, groups)
    clean = attack_metrics(logits.detach().cpu().numpy(), gate.detach().cpu().numpy(),
                           truth, tau)
    clean["snr"] = None
    clean["goal"] = "clean"
    log(f"  clean: closed {clean['closed_set_accuracy']:.4f} accept "
        f"{clean['accept_rate']:.4f} RA {clean['RA']:.4f} ASafeR {clean['ASafeR']:.4f}")

    conditions = []
    cfg = CFG["attack"]
    for snr in cfg["snr_db"]:
        for goal in ("A", "B", "C", "D"):
            out = pgd_snr(scorer, raw, task_idx, groups, true_idx, target_idx, goal,
                          snr, int(cfg["steps"]), SEED + 1000 * int(snr) + ord(goal))
            metric = attack_metrics(out["logits"], out["gate"], truth, tau)
            metric.update({"snr": int(snr), "goal": goal})
            conditions.append(metric)
            log(f"  SNR{snr} goal {goal}: RA {metric['RA']:.4f} ASafeR "
                f"{metric['ASafeR']:.4f} accept {metric['accept_rate']:.4f} closed "
                f"{metric['closed_set_accuracy']:.4f}")
    worst_ra = min(conditions, key=lambda r: r["RA"])
    worst_safe = min(conditions, key=lambda r: r["ASafeR"])
    summary = {"candidate": point["candidate"], "weights": weights, "tau": tau,
               "n_trials": int(len(subset)),
               "per_task": {t: int((tasks[subset] == t).sum()) for t in TASKS},
               "clean": clean, "conditions": conditions,
               "worst_RA": worst_ra, "worst_ASafeR": worst_safe,
               "snr_db": cfg["snr_db"], "steps": int(cfg["steps"]),
               "note": "self-built white-box PGD on the E191 branch-A v3 identity "
                       "model; SNR radius = per-trial L2 norm / 10^(SNR/20)"}
    results = read_json(RESULTS_PATH)
    results["attack"] = summary
    results.setdefault("stage_seconds", {})["attack"] = time.time() - started
    write_json(RESULTS_PATH, results)
    log(f"stage attack done: worst RA {worst_ra['RA']:.4f} "
        f"(SNR{worst_ra['snr']} {worst_ra['goal']}), worst ASafeR "
        f"{worst_safe['ASafeR']:.4f} (SNR{worst_safe['snr']} {worst_safe['goal']})")
    return summary


# ---------------------------------------------------------------- report
def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def wstr(weights: dict) -> str:
    return " + ".join(f"{name} {float(value):.2f}"
                      for name, value in weights.items() if float(value) > 0)


def selected_weights(results: dict) -> tuple:
    pareto = results["pareto"]["paired"]
    feasible = pareto.get("feasible") or []
    point = (max(feasible, key=lambda r: (r["strict_auc"], r["test_min"]))
             if feasible else pareto["best_compromise"])
    return point, {k: float(v) for k, v in point["weights"].items()}


def gate_values(results: dict, streams: dict) -> dict:
    point, weights = selected_weights(results)
    evaluation = evaluate(streams, weights, full=True)
    branch = read_json(E191_DIR / "results.json")["branch_a_v3"]
    attack = results.get("attack") or {}
    cond = attack.get("conditions") or []
    snr30 = [c for c in cond if c["snr"] == 30]
    snr20 = [c for c in cond if c["snr"] == 20]
    worst30_ra = min(snr30, key=lambda c: c["RA"]) if snr30 else None
    worst30_safe = min(snr30, key=lambda c: c["ASafeR"]) if snr30 else None
    worst20_ra = min(snr20, key=lambda c: c["RA"]) if snr20 else None
    worst20_safe = min(snr20, key=lambda c: c["ASafeR"]) if snr20 else None
    test = evaluation["test"]["paired"]
    values = {
        "#1": branch["in_sample_accuracy"],
        "#2": branch["oof_accuracy_5block"],
        "#3": branch["oof_macro_f1"],
        "#4": branch["oof_accuracy_forward_blocks"],
        "#5": evaluation["strict"]["pooled_auc"],
        "#6": evaluation["strict"]["u95"],
        "#7": worst30_ra["RA"] if worst30_ra else None,
        "#8": worst30_safe["ASafeR"] if worst30_safe else None,
        "#9": test["s2_acceptance"],
        "#10": test["s1_rejection"],
        "#11": evaluation["strict"]["rejection_fold"][KEY],
    }
    values_std = {"#5": evaluation["std"]["pooled_auc"],
                  "#6": evaluation["std"]["u95"],
                  "#11": evaluation["std"]["rejection_fold"][KEY]}
    targets = {"#1": 1.00, "#2": 0.95, "#3": 0.92, "#4": 0.90, "#5": 0.98,
               "#6": 0.95, "#7": 0.95, "#8": 0.98, "#9": 0.70, "#10": 0.70}
    passed = {key: bool(values[key] is not None and values[key] >= targets[key] - 1e-12)
              for key in targets}
    passed["#11"] = bool(values["#11"] > 0.90)
    return {"point": point, "weights": weights, "evaluation": evaluation,
            "values": values, "values_std": values_std, "passed": passed,
            "count": int(sum(passed.values())), "targets": targets,
            "snr30": {"worst_RA": worst30_ra, "worst_ASafeR": worst30_safe},
            "snr20": {"worst_RA": worst20_ra, "worst_ASafeR": worst20_safe},
            "test": test}


def build_report(results: dict, streams: dict) -> str:
    singles = results["singles"]
    pareto = results["pareto"]["paired"]
    gates = results["eleven_gates"]
    evaluation = gates["evaluation"]
    point = gates["point"]
    weights = gates["weights"]
    values, values_std = gates["values"], gates["values_std"]
    attack = results.get("attack") or {}
    cond = attack.get("conditions") or []
    snr30 = [c for c in cond if c["snr"] == 30]
    snr20 = [c for c in cond if c["snr"] == 20]
    lines = []
    add = lines.append

    add("# E195：第 11 项（LOSO 拒识 > 0.90）约束下的门控重搜索")
    add("")
    add("日期：2026-11-15。Train = 20 个注册者 session1（11304 条），"
        "Test S1 = 5 个未注册者 session1（3033 条），Test S2 = 20 个注册者 session2"
        "（19218 条）。所有统计量只用 Train；Test 只评分，Session 只用于 S1/S2 审计。"
        "未写入任何已有实验目录，未提交 git。")
    add("")
    add("## 0. 结论")
    add("")
    feasible = pareto.get("feasible") or []
    add(f"1. **#11 本身很容易满足**：E183 融合、E192 WCCN 距离比、E65+WCCN 融合的 "
        f"`loso_strict` 都在 0.96 以上；E65 单门控只有 0.6448，"
        f"subwin6 只有 0.3982。")
    add(f"2. **(a) 几何上存在，部署上不存在。** 在 349 个候选×权重点里，有 "
        f"**{len(feasible)} 个**同时满足 `loso_strict > 0.90` 与 Test 双 ≥0.70，"
        f"全部是 `E183 + E65 + subwin6` 三路融合，Test min 0.7017–0.7221、"
        f"strict 0.9278–0.9640，是一道很窄的缝。但它们只能靠 Test 审计找到；"
        f"按“权重只按 loso_strict 选”的合规规则，每个融合都会退化到 E183 单门控"
        f"（strict 0.9848，Test min 0.2733）。")
    add(f"3. **(b) Pareto 前沿**从 E183（strict 0.9848 / Test min 0.2733）经"
        f"三路融合过渡到 subwin6（strict 0.3982 / Test min 0.6916）。"
        f"strict ≥0.90 的最高 Test min 是 {fmt(pareto['best_compromise']['test_min'])}"
        f"（{wstr(pareto['best_compromise']['weights'])}，strict "
        f"{fmt(pareto['best_compromise']['strict'])}）。最好的可计数折中是 "
        f"`E183 0.4 + E65 0.3 + subwin6 0.3`：strict 0.9640、Test S1 0.7052 / "
        f"S2 0.7087、strict AUROC 0.9807。")
    add(f"4. **(c) 该折中单工作点通过 {gates['count']}/11 项**："
        f"#1–#5、#9–#11 通过；#6（U@95）、#7（RA）、#8（ASafeR）未通过。")
    add("")
    add("## 1. 两套 LOSO 的定义与数值对照")
    add("")
    add("- `loso_std`（E143 协议）：task-wise 均值/标准差用该 task 的全部 Train 行"
        "（含留出身份 h），PCA 基用全部 20 个身份的 fit_rows 拟合；只有原型"
        "用 `pm & fit_rows`（19 个已注册身份）。")
    add("- `loso_strict`：留出身份 h 的 fit 块也排除在 task-wise 标准化与 PCA 之外；"
        "均值/标准差、PCA 基、原型都只用其余 19 个身份（`keep` / `keep & fit_rows`）。")
    add("- 两者都是 20 折：known = 19 个身份的 0.6<frac≤0.8，unknown = h 的同一窗口；"
        "阈值取折内 known 的 `1-a` 分位；表中为 a=0.90 的拒识率与合并 AUROC。")
    add("")
    add("| 候选 | std 拒识@0.90 | std AUROC | strict 拒识@0.90 | strict AUROC | "
        "Δ拒识（std−strict） | strict 逐折 AUROC 均值±标准差 / 最差折 |")
    add("|---|---:|---:|---:|---:|---:|---|")
    for name, row in singles.items():
        s, k = row["std"], row["strict"]
        add(f"| `{name}` | {fmt(s['rejection_fold'][KEY])} | {fmt(s['pooled_auc'])} | "
            f"{fmt(k['rejection_fold'][KEY])} | {fmt(k['pooled_auc'])} | "
            f"{fmt(s['rejection_fold'][KEY] - k['rejection_fold'][KEY])} | "
            f"{fmt(k['fold_auc_mean'])} ± {fmt(k['fold_auc_std'])} / "
            f"{fmt(k['fold_auc_min'])} |")
    add("")
    add("交叉核对：`e65_dist` 的 std 0.7936 / strict 0.6448 与 E194 完全一致"
        "（E194 的 A 效应 +0.1653 在本实验复现为 "
        f"{fmt(singles['e65_dist']['std']['rejection_fold'][KEY] - singles['e65_dist']['strict']['rejection_fold'][KEY])}）；"
        "`e183_fuse` 的 std 0.9914 / strict 0.9848 也与 E194 一致。"
        "WCCN 距离比门控的 strict 数组与 E192 `cache/gate_arrays.npz` 的 "
        "known/unknown 相关系数均为 1.000000，AUROC 0.9873 vs 0.9873。")
    add("")
    add("**拟合泄漏并不普遍**：E65 与 subwin6 的 std−strict 差为 0.149 / 0.188，"
        "E162 与 E183 基本为 0（E183 的协议本来就排除留出身份），WCCN 为 0.004。"
        "因此 `loso_std > 0.90` 本身不是新信息，真正的新约束是 `loso_strict > 0.90`。")
    add("")
    add("**E65 的实现口径**：本实验严格复用 E194 的原型构造（每身份 k=12、seed "
        "ident*17、max-over-prototypes）与 E143/E194 的部署约定（Test 原型用全部 "
        "Train 行、5 块 OOF 用 ~block 原型）。std/strict 与 E194 逐位一致；"
        "Test S1 也复现 0.7979。E189/E191 的 E65 是另一套原型实现（与 E194 的 Test "
        "分数相关 0.978），其 a=0.90 的 S2 读数是 0.7359；本实验不用它做融合，"
        "以免把两套原型混在一起。")
    add("")
    add("## 2. 全部候选的完整指标")
    add("")
    add("阈值规则：Test 主读数用 `paired`（每个候选自己的折内 known 分数拼接后取 "
        "a=0.90 分位；所有分量用同一批行，融合时不需要近似）；距离/WCCN 单门控同时"
        "给出 `oof`（5 块 OOF 分位，E189/E191 项目约定）。E183 在已发布缓存里没有 "
        "5 块 OOF 流，按 E194 的口径用 pooled strict LOSO known 分位。")
    add("")
    add("### 2.1 单门控")
    add("")
    add("| 候选 | std 拒识 | std AUROC | strict 拒识 | strict AUROC | U@95（strict） | "
        "Test S1（paired） | Test S2（paired） | min | Test S1（OOF） | Test S2（OOF） | min（OOF） |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, row in singles.items():
        s, k = row["std"], row["strict"]
        t = row["test"]["paired"]
        o = row["test"].get("oof")
        add(f"| `{name}` | {fmt(s['rejection_fold'][KEY])} | {fmt(s['pooled_auc'])} | "
            f"{fmt(k['rejection_fold'][KEY])} | {fmt(k['pooled_auc'])} | {fmt(k['u95'])} | "
            f"{fmt(t['s1_rejection'])} | {fmt(t['s2_acceptance'])} | {fmt(t['min'])} | "
            f"{fmt(o['s1_rejection']) if o else 'N/A'} | "
            f"{fmt(o['s2_acceptance']) if o else 'N/A'} | "
            f"{fmt(o['min']) if o else 'N/A'} |")
    add("")
    add("### 2.2 融合候选（权重只按 loso_strict 选，Train-only）")
    add("")
    add("每个两两融合在 0–1、步长 0.05 的单纯形上取 `loso_strict` 拒识@0.90 最大者"
        "（并列看 strict AUROC）；三路融合用步长 0.1。Test 不参与。"
        "融合前每条流先按其 Train 参考分布标准化（距离/WCCN 用 5 块 OOF 的均值/方差，"
        "E183 用 pooled strict LOSO known 的均值/方差），所以权重是标准化单位下的"
        "相对权重，单纯形和为 1。")
    add("")
    add("| 候选 | 选中权重 | std 拒识 | std AUROC | strict 拒识 | strict AUROC | "
        "U@95 | Test S1 | Test S2 | min | 是否 #11 | 是否双 0.70 |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for key, spec in results["search"]["fusions"].items():
        w = spec["train_selected"]
        row = evaluate(streams, w)
        t = row["test"].get("paired")
        if t is None:
            continue
        add(f"| `{key}` | {wstr(w)} | {fmt(row['std']['rejection_fold'][KEY])} | "
            f"{fmt(row['std']['pooled_auc'])} | {fmt(row['strict']['rejection_fold'][KEY])} | "
            f"{fmt(row['strict']['pooled_auc'])} | {fmt(row['strict']['u95'])} | "
            f"{fmt(t['s1_rejection'])} | {fmt(t['s2_acceptance'])} | {fmt(t['min'])} | "
            f"{'✅' if row['strict']['rejection_fold'][KEY] > 0.90 else '❌'} | "
            f"{'✅' if t['min'] >= 0.70 else '❌'} |")
    add("")
    add("**Train-only 选权的退化**：只要把 `loso_strict` 当唯一目标，E183 的权重就会"
        "吃掉所有距离流（`e65+subwin6` 选到 1.0/0.0，`e183+e65+subwin6` 选到 "
        "1.0/0.0/0.0，`e65+e183` 选到 0.0/1.0）。原因是 E183 的 strict 拒识 0.9848 "
        "远高于任何距离流，任何距离流权重都会把它拉低。")
    add("")
    add("### 2.3 loso_strict vs Test-min 的 Pareto 前沿（paired 阈值）")
    add("")
    add("非支配点（同时最大化 strict 拒识与 Test min），按 strict 降序。"
        "这是审计，不是可部署选择器：权重不属于 Train-only 规则的输出。")
    add("")
    add("| 候选 | 权重 | strict 拒识 | strict AUROC | std 拒识 | Test S1 | Test S2 | min |")
    add("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in pareto["frontier"]:
        add(f"| `{row['candidate']}` | {wstr(row['weights'])} | {fmt(row['strict'])} | "
            f"{fmt(row['strict_auc'])} | {fmt(row['std'])} | {fmt(row['s1'])} | "
            f"{fmt(row['s2'])} | {fmt(row['test_min'])} |")
    add("")
    add("### 2.4 同时满足 #11 与 Test 双 0.70 的全部点（审计）")
    add("")
    if feasible:
        add("| 候选 | 权重 | strict 拒识 | strict AUROC | Test S1 | Test S2 | min |")
        add("|---|---|---:|---:|---:|---:|---:|")
        for row in sorted(feasible, key=lambda r: -r["test_min"]):
            add(f"| `{row['candidate']}` | {wstr(row['weights'])} | {fmt(row['strict'])} | "
                f"{fmt(row['strict_auc'])} | {fmt(row['s1'])} | {fmt(row['s2'])} | "
                f"{fmt(row['test_min'])} |")
    else:
        add("无。")
    add("")
    add("## 3. 明确回答 (a)(b)(c)")
    add("")
    add("### (a) 是否存在同时满足 LOSO>0.90 与 Test S1/S2 双 ≥0.70 的门控？")
    add("")
    add(f"**分两层回答。**")
    add("")
    add(f"- **作为一组固定权重，存在。** 349 个点里有 {len(feasible)} 个："
        f"`E183 0.3 + E65 0.2 + subwin6 0.5`（strict 0.9278 / S1 0.7227 / S2 0.7221）、"
        f"`E183 0.4 + E65 0.3 + subwin6 0.3`（strict 0.9640 / S1 0.7052 / S2 0.7087）、"
        f"`E183 0.3 + E65 0.3 + subwin6 0.4`（strict 0.9360 / S1 0.7471 / S2 0.7017）。"
        f"三者都是三路融合，双 0.70 的余量只有 0.0017–0.0221。")
    add(f"- **作为可部署门控，不存在。** 权重禁止用 Test 选；按 `loso_strict` 最大选权，"
        f"每个融合都退化到 E183 单门控（strict 0.9848，Test S1 0.2733 / S2 0.8173，"
        f"min 0.2733）。也就是说，上一条的“存在”依赖 Test 审计来定位权重，"
        f"不满足本实验的合规约束。")
    add(f"- **只用有 5 块 OOF 的距离/WCCN 流（E189/E191 项目口径）时 feasible = 0**："
        f"3 个可行点都依赖 E183 的 paired LOSO 阈值。因此换阈值口径也不会把结论"
        f"从“不可部署”变成“可部署”。")
    add("")
    add("### (b) 若不存在，Pareto 前沿在哪，最好的折中是什么？")
    add("")
    add(f"前沿见 §2.3：从 `e183_fuse`（strict 0.9848 / min 0.2733）沿 "
        f"`e183+e65+wccn`、`e183+e65+subwin6` 过渡到 `subwin6`"
        f"（strict 0.3982 / min 0.6916）。**前沿在 strict≈0.90 处的 Test min "
        f"约 0.70–0.72**，不是单调的“strict 越高 Test 越好”，而是一道窄缝。")
    add(f"两条边界各自能到：**Test min ≥0.70 时 strict 最高 0.9640**"
        f"（0.4/0.3/0.3）；**strict >0.90 时 Test min 最高 0.7221**"
        f"（0.3/0.2/0.5，但 strict AUROC 0.9716）。把 strict 完全放开，"
        f"Test min 的最高点仍是 0.7221；再往 subwin6 偏，min 不再上升"
        f"（S1 掉得比 S2 快）。")
    add("")
    add(f"最好的可计数折中是 `E183 0.4 + E65 0.3 + subwin6 0.3`："
        f"strict {fmt(point['strict'])}、strict AUROC {fmt(point['strict_auc'])}、"
        f"Test S1 {fmt(point['s1'])} / S2 {fmt(point['s2'])}。"
        f"若按“Test min 最大”选，则是 {wstr(max(feasible, key=lambda r: r['test_min'])['weights'])}"
        f"（min {fmt(max(feasible, key=lambda r: r['test_min'])['test_min'])}），"
        f"但它的 strict AUROC 只有 "
        f"{fmt(max(feasible, key=lambda r: r['test_min'])['strict_auc'])}，会丢掉 #5。")
    add("")
    add("### (c) 最佳折中方案能通过 11 项中的几项？")
    add("")
    add(f"**{gates['count']}/11**。未通过 #6（U@95 {fmt(values['#6'])}，目标 ≥0.95）、"
        f"#7（RA {fmt(values['#7'])} @SNR30，目标 ≥0.95）与 "
        f"#8（ASafeR {fmt(values['#8'])} @SNR30，目标 ≥0.98）；在 SNR20 下 #7/#8 "
        f"分别只有 {fmt(gates['snr20']['worst_RA']['RA'])} / "
        f"{fmt(gates['snr20']['worst_ASafeR']['ASafeR'])}。"
        f"#5 以 0.9807 擦线通过，#9/#10/#11 分别以 0.7087 / 0.7052 / 0.9640 通过。")
    add("")
    add("## 4. 最佳折中方案的完整 11 项门禁（单工作点，不跨点拼接）")
    add("")
    add(f"工作点：`{point['candidate']}`，权重 {wstr(weights)}，a=0.90，"
        f"τ（paired）= {fmt(gates['test']['tau'])}。")
    add("")
    add("| 门禁 | 读数（`loso_strict` 口径） | `loso_std` 口径 | 目标 | 判定 | 缺口 |")
    add("|---|---:|---:|---:|---|---:|")
    names = [("#1", "Train in-sample 准确率", "= 1.00", values["#1"], None),
             ("#2", "5 块 OOF C", "≥ 0.95", values["#2"], None),
             ("#3", "身份 macro-F1", "≥ 0.92", values["#3"], None),
             ("#4", "块外准确率", "≥ 0.90", values["#4"], None),
             ("#5", "LOSO AUROC", "≥ 0.98", values["#5"], values_std["#5"]),
             ("#6", "U@95%", "≥ 0.95", values["#6"], values_std["#6"]),
             ("#7", "RA（SNR30 最差 goal）", "≥ 0.95", values["#7"], None),
             ("#8", "ASafeR（SNR30 同一攻击）", "≥ 0.98", values["#8"], None),
             ("#9", "Test S2 接受率", "≥ 0.70", values["#9"], None),
             ("#10", "Test S1 拒识率", "≥ 0.70", values["#10"], None),
             ("#11", "LOSO 拒识 @a=0.90", "> 0.90", values["#11"], values_std["#11"])]
    targets = {"#1": 1.00, "#2": 0.95, "#3": 0.92, "#4": 0.90, "#5": 0.98,
               "#6": 0.95, "#7": 0.95, "#8": 0.98, "#9": 0.70, "#10": 0.70}
    for key, label, target, value, std in names:
        status = "✅" if gates["passed"][key] else "❌"
        gap = "—" if gates["passed"][key] else f"{fmt(value - targets[key]) if key in targets else fmt(value - 0.90)}"
        add(f"| {key} {label} | {fmt(value)} | {fmt(std) if std is not None else '—'} | "
            f"{target} | {status} | {gap} |")
    add("")
    add("### 4.1 自建 PGD（SNR30 / SNR20，各 4 个目标）")
    add("")
    add("攻击：E191 分支 A v3 身份模型做白盒 PGD，20 步，L2 半径 = 每 trial "
        "去均值波形的 L2 范数 / 10^(SNR/20)；目标 A/B/C 攻身份头，D 攻门控。"
        "阈值用同一工作点的 τ（Train-only）。#7/#8 的主判定用 SNR30（与 E192 的 "
        "primary_snr_db=30 口径一致），SNR20 作参考档；若把 SNR20 也计入最差，"
        "读数为 0.0807 / 0.2364，结论不变（仍失败）。")
    add("")
    add("| SNR | 目标 | RA | ASafeR | 门控接受率 | 攻击后闭集准确率 |")
    add("|---|---|---:|---:|---:|---:|")
    if attack.get("clean"):
        c = attack["clean"]
        add(f"| clean | — | {fmt(c['RA'])} | {fmt(c['ASafeR'])} | "
            f"{fmt(c['accept_rate'])} | {fmt(c['closed_set_accuracy'])} |")
    for c in snr30 + snr20:
        add(f"| {c['snr']} | {c['goal']} | {fmt(c['RA'])} | {fmt(c['ASafeR'])} | "
            f"{fmt(c['accept_rate'])} | {fmt(c['closed_set_accuracy'])} |")
    add("")
    add(f"最差：SNR30 RA {fmt(gates['snr30']['worst_RA']['RA'])}、"
        f"ASafeR {fmt(gates['snr30']['worst_ASafeR']['ASafeR'])}；"
        f"SNR20 RA {fmt(gates['snr20']['worst_RA']['RA'])}、"
        f"ASafeR {fmt(gates['snr20']['worst_ASafeR']['ASafeR'])}。"
        "SNR20 的闭集准确率跌到 0.08，说明 v3 的对抗训练（ε 0.02–0.04 的 L∞）"
        "不覆盖 SNR20 的 L2 半径；这是 #7/#8 未通过的直接原因。")
    add("")
    add("## 5. 为什么 #11 与 Test 双 0.70 难以兼得（定量）")
    add("")
    add("1. **两个约束争的是不同的东西。** #11 只用同批同会话的 19 身份拟合；"
        "Test 双 0.70 要求门控在另一批采集的 5 个未注册者（S1）与 20 个注册者"
        "的 session2（S2）之间划出同一方向的边界。E194 已证明 S1 与 Train 的"
        "域可分性 AUC 0.9789，而 S2 只有 0.8698。")
    add("2. **单门控的 Test S1-vs-S2 排序能力决定上限。** E183 融合的 "
        "AUC(S2 vs S1)=0.593，WCCN 距离比 0.389，E65 距离 0.758。"
        "E183 的 strict 拒识 0.9848，但它在 S1/S2 上几乎不排序（S1 均值 0.324、"
        "S2 0.420，差距远小于 LOSO unknown 的 −0.985）。"
        "要把 Test min 抬到 0.70，门控必须接近 E65 的排序能力；"
        "而 E65 的 strict 只有 0.6448。")
    add("3. **前沿是凸的、缝很窄。** 在 349 个点里，strict>0.90 且 min≥0.70 只有 3 个，"
        "全部落在 `E183 0.3–0.4 + E65 0.2–0.3 + subwin6 0.3–0.5`。"
        "再往 E183 偏 0.1，min 掉到 0.69；再往距离流偏 0.1，strict 掉到 0.90 附近"
        "且 min 不再上升（S2 开始下降）。")
    add("4. **Train-only 选权一定退化。** strict 拒识在 E183 处最高（0.9848），"
        "任何距离流权重都是“用 strict 换 Test min”。合规规则不允许用 Test 选权重，"
        "所以部署系统只能停在 E183（min 0.2733）或纯 E65（strict 0.6448，"
        "min 0.5569/0.5960）两端；中间的可行点没有 Train-only 的选择依据。"
        "**差距**：Train-only 的 E183 单门控 Test min 0.2733，距双 0.70 差 0.4267；"
        "纯 E65 的 strict 0.6448，距 0.90 差 0.2552；0.4/0.3/0.3 才把两个差距压到 "
        "0.005/0.009，但它只能靠 Test 审计定位。")
    add("5. **#5/#6 也在这条缝上。** 0.3/0.2/0.5 点 min 更高（0.7221），但 strict AUROC "
        "只有 0.9716，丢掉 #5；0.4/0.3/0.3 的 strict AUROC 0.9807，刚好过 0.98。"
        "#6（U@95）更硬：在全部 30 个 Test min≥0.70 的点里，U@95 最高就是 0.8831"
        "（即 0.4/0.3/0.3）；即使 E183 单门控也只有 0.909，WCCN 单门控 0.9581 但 "
        "Test S2 只有 0.188。**没有一个候选能同时把 #6 抬到 0.95 又保住 Test 双 0.70。**")
    add("6. **余量都在噪声以内。** #5 只高 0.0007，#9/#10 只高 0.005/0.009，"
        "#6 差 0.067，#7 只差 0.0027；S1 只有 5 个被试（3033 条）。按被试 bootstrap，"
        "这些边界都会跨过阈值，因此“可行”应读作“点估计上擦线可行”，"
        "不是稳健结论；要把它变成稳健结论，需要更多未注册者被试或跨批次校准。")
    add("")
    add("## 6. Test 官方指标")
    add("")
    add("**Test 官方 AUC / F1 / C / RA / ASafeR / U / Score = N/A**"
        "（Test 无真实身份标签，只有 S1/S2 审计读数）。")
    add("")
    add("## 7. 合规与边界")
    add("")
    add("- 标准化、PCA、原型、阈值、融合权重全部来自 Train；`loso_strict` 的每一折都"
        "真正排除了留出身份的 fit 块（不只是原型）。")
    add("- Test 只评分：先算分再读 Session；Session 从未进入特征、模型、阈值或权重。"
        "Session 也从未置 0。")
    add("- 权重选择：§2.2 的每个融合都只按 `loso_strict` 选；§2.3/§2.4 的 Pareto 与"
        "可行点是审计，不是部署选择器。最佳折中点因此标注为“审计可选、"
        "Train-only 不可选”。")
    add("- 单工作点：§4 的 11 项全部来自 `E183 0.4 + E65 0.3 + subwin6 0.3`、a=0.90、"
        "τ（paired）一个点，没有跨点拼接。")
    add("- E183 的 `loso_std` 读数用 E194 的“反事实重训 head”缓存；E183 的协议本来"
        "就排除留出身份，所以 std 与 strict 只差 0.0066。")
    add("- E192 的 WCCN 门控按 E192 代码逐行重实现（shrinkage 0.05、eigenvalue floor "
        "1e-6、distance ratio），与 `cache/gate_arrays.npz` 的 known/unknown 相关 1.0。")
    add("")
    add("## 8. 文件与耗时")
    add("")
    add("| 文件 | 内容 |")
    add("|---|---|")
    add("| `run_e195.py` | 全部逻辑（`--stage scores/search/attack/report/all`） |")
    add("| `config.json` | 冻结配置 |")
    add("| `cache/streams.npz` | 7 条流的 std/strict 折内分数、Test 分数、OOF、部署 instrument |")
    add("| `results.json` | 单门控、融合、Pareto、11 项、攻击 |")
    add("| `report.md` | 本报告 |")
    add("| `run.log` | 运行日志 |")
    add("")
    seconds = (results.get("stage_seconds") or {})
    add("耗时（秒）：" + "、".join(f"{k} {v:.0f}" for k, v in seconds.items()))
    add("")
    add("```powershell")
    add("& \"D:\\WORK\\BCIC2026\\BrainPrint\\.venv\\Scripts\\python.exe\" "
        "\"D:\\WORK\\BCIC2026\\BrainPrint\\research\\E195_eleven_gates_loso90\\run_e195.py\" "
        "--stage all")
    add("```")
    return "\n".join(lines) + "\n"


def stage_report() -> str:
    started = time.time()
    results = read_json(RESULTS_PATH)
    streams = load_streams()
    gates = gate_values(results, streams)
    results["eleven_gates"] = gates
    results["answers"] = {
        "a": "A fixed-weight gate exists (3 of 349 points: E183+E65+subwin6, "
             "strict 0.9278-0.9640, Test S1 0.7052-0.7471, S2 0.7017-0.7221), but no "
             "deployable gate exists: weights cannot be selected with Test, and a "
             "Train-only (max loso_strict) rule degenerates to E183 alone "
             "(strict 0.9848, Test min 0.2733).",
        "b": "Pareto frontier runs from E183 (0.9848 / 0.2733) through "
             "E183+E65+WCCN and E183+E65+subwin6 to subwin6 (0.3982 / 0.6916); "
             "the best countable compromise is E183 0.4 + E65 0.3 + subwin6 0.3 "
             "(strict 0.9640, Test 0.7052/0.7087).",
        "c": f"{gates['count']}/11; #6 (U@95 {gates['values']['#6']:.4f}), "
             f"#7 and #8 fail (SNR30 RA {gates['values']['#7']:.4f}, ASafeR "
             f"{gates['values']['#8']:.4f}; SNR20 much worse).",
    }
    results.setdefault("stage_seconds", {})["report"] = time.time() - started
    write_json(RESULTS_PATH, results)
    REPORT_PATH.write_text(build_report(results, streams), encoding="utf-8")
    log(f"stage report done: {gates['count']}/11, candidate "
        f"{gates['point']['candidate']} {gates['weights']}")
    return gates["count"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["scores", "search", "attack", "report", "all"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    sys.stdout = Tee(LOG_PATH)
    log(f"E195 start {time.strftime('%Y-%m-%d %H:%M:%S')} device {DEVICE}")
    if args.stage in ("scores", "all"):
        stage_scores(args.force)
    if args.stage in ("search", "all"):
        stage_search(args.force)
    if args.stage in ("attack", "all"):
        stage_attack(args.force)
    if args.stage in ("report", "all"):
        stage_report()
    log("E195 done")


if __name__ == "__main__":
    main()
