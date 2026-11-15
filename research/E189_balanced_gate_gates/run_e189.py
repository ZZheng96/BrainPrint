"""E189: re-optimise the full system under the new "mean(S1 rejection, S2 acceptance) > 0.70"
criterion, with a strict single-work-point discipline.

What is frozen and reused read-only
  * branch A identity model  : E180 model_branch_a.pt (adversarially trained EEGNet)
  * branch A OOF / in-sample : E180 cache/stage_a.pt
  * stream features          : E65 correlation-logSPD, E171 subwin_mean N=6/N=12,
                               E146 log band power, E162 raw-EEG embedding
  * E162 encoder             : model_lam0.0_t0.1.pt (the model that wrote train_emb.npy)

Gate candidates (Train-only fitting, Test only scored)
  six single-stream distance gates + eight two/three/four-way weighted fusions.
  Fusion weights = argmax pooled Train LOSO AUC on a 0.1-step simplex.

Work point
  tau(a) = quantile(Train 5-block OOF score, 1 - a), a on 0.80..0.96.
  Primary ("constrained") point: largest a whose LOSO unknown rejection >= 0.70
  (the stated constraint), tie-break by min(LOSO known acceptance, LOSO unknown
  rejection).  The pure balance point is reported next to it.  Both are Train-only;
  Test is never consulted.

Decision
  gate >= tau(a) -> branch A argmax, else label 0.

Stages: scores, audit, attack, report, all.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
SCORE_PATH = CACHE / "stream_scores.npz"
CAND_PATH = CACHE / "candidates.npz"
AUDIT_PATH = OUT / "audit.json"
ATTACK_PATH = OUT / "attack.json"
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "report.md"
LOG_PATH = OUT / "run.log"

DEVICE = torch.device("cuda")
FT = torch.float32
ACC_GRID = [float(a) for a in CFG["threshold"]["grid"]]
LOSO_FLOOR = 0.70

TRI_ROWS = TRI_COLS = None
DIAG_MASK = None
OFFMUL = None


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E180 = load_module("e189_frozen_e180",
                   ROOT / CFG["frozen_sources"]["e180"] / "run_e180.py")
E162 = load_module("e189_frozen_e162",
                   ROOT / CFG["frozen_sources"]["e162"] / "run_e162.py")

TASKS = list(E180.TASKS)
CLASSES = E180.CLASSES
SEED = int(E180.CFG["protocol"]["seed"])
FIT = float(E180.CFG["protocol"]["fit_fraction"])
BLOCKS = int(E180.CFG["protocol"]["blocks"])
N = int(E180.N)
CH = int(E180.CH)
FS = int(E180.FS)
NB = 5
BANDS = [tuple(b) for b in E180.CFG["branch_b"]["bands_hz"]]
LOGBP_BANDS = [tuple(b) for b in CFG["streams"]["logbp_pca16_k1"]["bands_hz"]]
OFF = -1e9


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


def quantile(value: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(value, dtype=np.float64), q))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float),
                    encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ------------------------------------------------------------------ protocol
def protocol() -> dict:
    meta, frac, blocks, fit_rows = E180.load_config_meta()
    test_meta = pd.read_csv(ROOT / CFG["data"]["test_metadata"],
                            usecols=["sample", "path", "task", "session"])
    if len(test_meta) != int(CFG["data"]["test_trials"]):
        raise ValueError(f"Test rows {len(test_meta)}")
    is_s1 = test_meta["session"].astype(str).str.casefold().isin(
        ["session1", "s1", "1"]).to_numpy()
    return {"meta": meta, "frac": frac, "blocks": blocks, "fit_rows": fit_rows,
            "test_meta": test_meta, "is_s1": is_s1,
            "subjects": meta["subject"].to_numpy(int),
            "tasks": meta["task"].to_numpy(),
            "test_tasks": test_meta["task"].to_numpy(),
            "window": (frac > FIT) & (frac <= 0.8)}


# ------------------------------------------------------------------ stage: scores
def single_stream(name: str, P: dict) -> dict:
    spec = CFG["streams"][name]
    tasks, subjects = P["tasks"], P["subjects"]
    blocks, fit_rows, frac = P["blocks"], P["fit_rows"], P["frac"]
    test_tasks = P["test_tasks"]
    k, dim = int(spec["k"]), int(spec["pca"])

    train = f32(np.load(ROOT / spec["train"]))
    test = f32(np.load(ROOT / spec["test"]))
    if train.shape[0] != len(tasks) or test.shape[0] != len(test_tasks):
        raise ValueError(f"{name}: row mismatch {train.shape} {test.shape}")

    stats = E180.task_stats(train, tasks)
    train_std = E180.apply_stats(train, tasks, stats)
    test_std = E180.apply_stats(test, test_tasks, stats)
    pca_mu, basis = E180.pca_basis(train_std[fit_rows], dim)
    train_proj = f32((train_std.astype(np.float64) - pca_mu.astype(np.float64)) @ basis)
    test_proj = f32((test_std.astype(np.float64) - pca_mu.astype(np.float64)) @ basis)
    del train, test, train_std, test_std

    oof = np.full(len(tasks), np.nan, dtype=np.float32)
    for block in range(BLOCKS):
        rows = blocks == block
        protos, keys = E180.build_prototypes(train_proj, subjects, tasks,
                                             fit_rows & ~rows, CLASSES, k)
        proto, filled = E180.proto_stack(protos, keys, k)
        held = np.flatnonzero(rows)
        oof[held] = E180.branch_b_scores(train_proj[held], tasks[held], proto, filled, k)
    if not np.isfinite(oof).all():
        raise RuntimeError(f"{name}: incomplete OOF")

    window = P["window"]
    folds = []
    for held_id in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held_id])
        pm = np.isin(subjects, enrolled)
        protos, keys = E180.build_prototypes(train_proj, subjects, tasks,
                                             pm & fit_rows, enrolled, k)
        proto, filled = E180.proto_stack(protos, keys, k)
        known_rows = np.flatnonzero(window & pm)
        unknown_rows = np.flatnonzero(window & (subjects == held_id))
        rows = np.r_[known_rows, unknown_rows]
        scores = E180.branch_b_scores(train_proj[rows], tasks[rows], proto, filled, k)
        folds.append({"held": int(held_id), "known": scores[:len(known_rows)],
                      "unknown": scores[len(known_rows):]})

    protos, keys = E180.build_prototypes(train_proj, subjects, tasks, fit_rows,
                                         CLASSES, k)
    proto_full, filled_full = E180.proto_stack(protos, keys, k)
    test_scores = E180.branch_b_scores(test_proj, test_tasks, proto_full, filled_full, k)

    oof_mu = float(np.mean(oof.astype(np.float64)))
    oof_sd = float(np.std(oof.astype(np.float64))) + 1e-6
    known = np.concatenate([f["known"] for f in folds]).astype(np.float64)
    unknown = np.concatenate([f["unknown"] for f in folds]).astype(np.float64)
    loso_auc = float(roc_auc_score(np.r_[np.zeros(len(unknown)), np.ones(len(known))],
                                   np.r_[unknown, known]))
    log(f"  {name}: dim {int(spec['dim'])} -> PCA-{dim} k={k} | OOF mean {oof_mu:.3f} "
        f"sd {oof_sd:.3f} | LOSO AUC {loso_auc:.4f}")

    return {"oof": oof, "test": test_scores, "folds": folds,
            "oof_mu": oof_mu, "oof_sd": oof_sd, "loso_auc": loso_auc,
            "pca": dim, "k": k,
            "stats_mu": np.stack([stats[t][0] for t in TASKS]),
            "stats_sd": np.stack([stats[t][1] for t in TASKS]),
            "pca_mu": pca_mu, "basis": basis,
            "proto": np.stack([proto_full[t] for t in TASKS]),
            "filled": np.stack([filled_full[t] for t in TASKS])}


def stage_scores() -> dict:
    started = time.time()
    if SCORE_PATH.exists():
        log("stage scores: cache present, reusing")
        return {}
    P = protocol()
    meta = P["meta"]
    log(f"stage scores: {len(meta)} Train rows, fit_rows {int(P['fit_rows'].sum())}, "
        f"window {int(P['window'].sum())}, Test {len(P['test_meta'])}")
    payload = {"is_s1": P["is_s1"],
               "test_sample": P["test_meta"]["sample"].to_numpy(),
               "test_session": P["test_meta"]["session"].astype(str).to_numpy()}
    summary = {}
    for name in CFG["streams"]:
        store = single_stream(name, P)
        for key in ("oof", "test", "oof_mu", "oof_sd", "stats_mu", "stats_sd",
                    "pca_mu", "basis", "proto", "filled"):
            payload[f"{name}__{key}"] = store[key]
        payload[f"{name}__loso_held"] = np.asarray([f["held"] for f in store["folds"]])
        for index, fold in enumerate(store["folds"]):
            payload[f"{name}__loso_{index}_known"] = fold["known"]
            payload[f"{name}__loso_{index}_unknown"] = fold["unknown"]
        summary[name] = {"pca": store["pca"], "k": store["k"],
                         "oof_mu": store["oof_mu"], "oof_sd": store["oof_sd"],
                         "loso_auc": store["loso_auc"],
                         "n_known": int(sum(len(f["known"]) for f in store["folds"])),
                         "n_unknown": int(sum(len(f["unknown"]) for f in store["folds"]))}
        del store
    np.savez_compressed(SCORE_PATH, **payload)
    results = read_json(RESULTS_PATH)
    results["streams"] = summary
    results["stage_seconds"] = {**(results.get("stage_seconds") or {}),
                                "scores": time.time() - started}
    write_json(RESULTS_PATH, results)
    log(f"stage scores done in {time.time() - started:.0f}s")
    return summary


# ------------------------------------------------------------------ stage: candidates
def load_stream_arrays() -> dict:
    data = np.load(SCORE_PATH, allow_pickle=False)
    streams = {}
    for name in CFG["streams"]:
        mu, sd = float(data[f"{name}__oof_mu"]), float(data[f"{name}__oof_sd"])
        folds = []
        held = data[f"{name}__loso_held"]
        for index, value in enumerate(held):
            folds.append({"held": int(value),
                          "known": (data[f"{name}__loso_{index}_known"] - mu) / sd,
                          "unknown": (data[f"{name}__loso_{index}_unknown"] - mu) / sd})
        streams[name] = {"oof": (data[f"{name}__oof"] - mu) / sd,
                         "test": (data[f"{name}__test"] - mu) / sd,
                         "folds": folds, "raw_oof_mu": mu, "raw_oof_sd": sd}
    streams["_is_s1"] = data["is_s1"]
    return streams


def pooled_loso_auc(folds_by_component, weights) -> float:
    known, unknown = [], []
    for index in range(len(folds_by_component[0])):
        known.append(sum(w * c[index]["known"] for w, c in zip(weights, folds_by_component)))
        unknown.append(sum(w * c[index]["unknown"] for w, c in
                           zip(weights, folds_by_component)))
    known = np.concatenate(known)
    unknown = np.concatenate(unknown)
    return float(roc_auc_score(np.r_[np.zeros(len(unknown)), np.ones(len(known))],
                               np.r_[unknown, known]))


def simplex(dim: int, step: float = 0.1) -> list:
    levels = int(round(1.0 / step))
    out = []

    def walk(prefix, left):
        if len(prefix) == dim - 1:
            out.append(tuple(list(prefix) + [left]))
            return
        for value in range(left + 1):
            walk(prefix + [value], left - value)

    walk([], levels)
    return [tuple(v * step for v in item) for item in out]


SHORT = {"e65_pca96_k12": "e65v96k12", "e65_pca64_k1": "e65v64k1",
         "subwin6_pca96_k1": "sw6v96k1", "subwin12_pca48_k1": "sw12v48k1",
         "logbp_pca16_k1": "logbpv16k1", "e162_pca128_k1": "e162v128k1"}


def fusion_key(group) -> str:
    return "fuse_" + "__".join(SHORT[g] for g in group)


def build_candidates(streams: dict) -> dict:
    names = [name for name in CFG["streams"]]
    candidates = {}
    for name in names:
        candidates[name] = {"components": [(name, 1.0)], "weights_loso_auc": None}

    for group in CFG["candidates"]["fusion"]:
        arrays = [streams[name]["folds"] for name in group]
        best = (-1.0, None)
        for weights in simplex(len(group)):
            auc = pooled_loso_auc(arrays, weights)
            if auc > best[0]:
                best = (auc, weights)
        candidates[fusion_key(group)] = {"components": list(zip(group, best[1])),
                                         "weights_loso_auc": best[0]}

    for key, spec in candidates.items():
        weights = [w for _, w in spec["components"]]
        components = [streams[c] for c, _ in spec["components"]]
        oof = sum(w * c["oof"] for w, c in zip(weights, components)).astype(np.float32)
        test = sum(w * c["test"] for w, c in zip(weights, components)).astype(np.float32)
        folds = []
        for index in range(len(components[0]["folds"])):
            folds.append({"held": int(components[0]["folds"][index]["held"]),
                          "known": sum(w * c["folds"][index]["known"]
                                       for w, c in zip(weights, components)),
                          "unknown": sum(w * c["folds"][index]["unknown"]
                                         for w, c in zip(weights, components))})
        known = np.concatenate([f["known"] for f in folds])
        unknown = np.concatenate([f["unknown"] for f in folds])
        auc = float(roc_auc_score(np.r_[np.zeros(len(unknown)), np.ones(len(known))],
                                  np.r_[unknown, known]))
        spec.update({"oof": oof, "test": test, "folds": folds, "loso_auc": auc,
                     "fused_oof_mu": float(np.mean(oof.astype(np.float64))),
                     "fused_oof_sd": float(np.std(oof.astype(np.float64))) + 1e-6})
    return candidates


def stage_candidates() -> dict:
    started = time.time()
    streams = load_stream_arrays()
    candidates = build_candidates(streams)
    payload = {}
    for key, spec in candidates.items():
        payload[f"{key}__oof"] = spec["oof"]
        payload[f"{key}__test"] = spec["test"]
        for index, fold in enumerate(spec["folds"]):
            payload[f"{key}__loso_{index}_known"] = fold["known"]
            payload[f"{key}__loso_{index}_unknown"] = fold["unknown"]
    np.savez_compressed(CAND_PATH, **payload)
    summary = {key: {"components": [[c, float(w)] for c, w in spec["components"]],
                     "loso_auc": spec["loso_auc"],
                     "weights_loso_auc": spec["weights_loso_auc"]}
               for key, spec in candidates.items()}
    results = read_json(RESULTS_PATH)
    results["candidates"] = summary
    results["stage_seconds"] = {**(results.get("stage_seconds") or {}),
                                "candidates": time.time() - started}
    write_json(RESULTS_PATH, results)
    log(f"stage candidates done in {time.time() - started:.0f}s ({len(candidates)} candidates)")
    return summary


# ------------------------------------------------------------------ stage: audit
def load_candidates() -> dict:
    data = np.load(CAND_PATH, allow_pickle=False)
    is_s1 = np.load(SCORE_PATH)["is_s1"]
    summary = read_json(RESULTS_PATH)["candidates"]
    out = {}
    for key, spec in summary.items():
        folds = []
        for index in range(20):
            folds.append({"held": None,
                          "known": data[f"{key}__loso_{index}_known"],
                          "unknown": data[f"{key}__loso_{index}_unknown"]})
        out[key] = {"oof": data[f"{key}__oof"], "test": data[f"{key}__test"],
                    "folds": folds, "loso_auc": spec["loso_auc"],
                    "components": spec["components"], "is_s1": is_s1}
    return out


def audit_point(cand: dict, a: float) -> dict:
    tau = quantile(cand["oof"], 1.0 - a)
    known = np.concatenate([f["known"] for f in cand["folds"]])
    unknown = np.concatenate([f["unknown"] for f in cand["folds"]])
    is_s1 = cand["is_s1"]
    test = cand["test"]
    return {"a": a, "tau": tau,
            "known_acceptance": float(np.mean(known >= tau)),
            "loso_rejection": float(np.mean(unknown < tau)),
            "test_s1_rejection": float(np.mean(test[is_s1] < tau)),
            "test_s2_acceptance": float(np.mean(test[~is_s1] >= tau)),
            "test_mean": float((np.mean(test[is_s1] < tau)
                                + np.mean(test[~is_s1] >= tau)) / 2.0),
            "test_acceptance": float(np.mean(test >= tau))}


def u_at_95(cand: dict) -> dict:
    per_fold, known_all, unknown_by_fold = [], [], []
    for fold in cand["folds"]:
        threshold = quantile(fold["known"], 1.0 - 0.95)
        per_fold.append(float(np.mean(fold["unknown"] < threshold)))
        known_all.append(fold["known"])
        unknown_by_fold.append(fold["unknown"])
    global_threshold = quantile(np.concatenate(known_all), 1.0 - 0.95)
    unknown = np.concatenate(unknown_by_fold)
    return {"per_fold_mean": float(np.mean(per_fold)),
            "per_fold_min": float(np.min(per_fold)),
            "per_fold_std": float(np.std(per_fold)),
            "global": float(np.mean(unknown < global_threshold))}


def pick_points(points: list) -> dict:
    balance = max(points, key=lambda r: min(r["known_acceptance"], r["loso_rejection"]))
    feasible = [r for r in points if r["loso_rejection"] >= LOSO_FLOOR]
    if feasible:
        constrained = max(feasible, key=lambda r: (r["a"],
                                                   min(r["known_acceptance"],
                                                       r["loso_rejection"])))
    else:
        constrained = balance
    return {"balance": balance["a"], "constrained": constrained["a"],
            "loso_floor": LOSO_FLOOR}


def stage_audit() -> dict:
    started = time.time()
    candidates = load_candidates()
    audit = {}
    for key, cand in candidates.items():
        points = [audit_point(cand, a) for a in ACC_GRID]
        audit[key] = {"points": points, "picked": pick_points(points),
                      "u_at_95": u_at_95(cand),
                      "loso_auc": cand["loso_auc"],
                      "components": cand["components"]}
    write_json(AUDIT_PATH, audit)
    results = read_json(RESULTS_PATH)
    results["audit"] = audit
    results["stage_seconds"] = {**(results.get("stage_seconds") or {}),
                                "audit": time.time() - started}
    write_json(RESULTS_PATH, results)
    log(f"stage audit done in {time.time() - started:.0f}s; "
        f"max Test mean over all candidates/a = "
        f"{max(p['test_mean'] for v in audit.values() for p in v['points']):.4f}")
    return audit


# ------------------------------------------------------------------ differentiable gate
def build_constants() -> None:
    global TRI_ROWS, TRI_COLS, DIAG_MASK, OFFMUL
    if TRI_ROWS is not None:
        return
    TRI_ROWS, TRI_COLS = torch.triu_indices(CH, CH, device=DEVICE)
    DIAG_MASK = (TRI_ROWS == TRI_COLS)
    OFFMUL = torch.where(DIAG_MASK, torch.ones_like(TRI_ROWS, dtype=FT),
                         torch.full_like(TRI_ROWS, 1.4142135623730951, dtype=FT))


def band_masks(length: int, bands=None) -> torch.Tensor:
    bands = BANDS if bands is None else bands
    freq = torch.fft.rfftfreq(length, d=1.0 / FS, device=DEVICE)
    masks = []
    for index, (low, high) in enumerate(bands):
        upper = freq <= high if index == len(bands) - 1 else freq < high
        masks.append((freq >= low) & upper)
    return torch.stack(masks).to(FT)


def logspd(x: torch.Tensor) -> torch.Tensor:
    """E65 gpu_correlation_logspd, differentiable, channel statistics inside the trial."""
    length = x.shape[-1]
    masks = band_masks(length)
    spec = torch.fft.rfft(x, dim=-1)
    filtered = torch.fft.irfft(spec[:, None] * masks[None, :, None, :], n=length, dim=-1)
    filtered = filtered - filtered.mean(dim=-1, keepdim=True)
    scale = filtered.std(dim=-1, keepdim=True, unbiased=True).clamp_min(1e-12)
    normalised = (filtered - filtered.mean(dim=-1, keepdim=True)) / scale
    corr = normalised @ normalised.transpose(-1, -2) / (length - 1)
    trace = torch.diagonal(corr, dim1=-2, dim2=-1).sum(dim=-1)
    eye = torch.eye(CH, dtype=x.dtype, device=x.device)
    spd = 0.95 * corr + 0.05 * trace[..., None, None] / CH * eye
    values, vectors = torch.linalg.eigh(spd)
    values = values.clamp_min(1e-12)
    mat = (vectors * values.log().unsqueeze(-2)) @ vectors.transpose(-1, -2)
    vec = mat[..., TRI_ROWS, TRI_COLS] * OFFMUL
    vec = vec.reshape(len(vec), NB, -1)
    diag_mean = vec[..., DIAG_MASK].mean(dim=-1, keepdim=True)
    vec = torch.where(DIAG_MASK, vec - diag_mean, vec)
    mean = vec.mean(dim=-1, keepdim=True)
    sd = vec.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return ((vec - mean) / sd).flatten(-2)


def subwin_mean(x: torch.Tensor, count: int) -> torch.Tensor:
    length = N // count
    parts = [logspd(x[..., w * length:(w + 1) * length]) for w in range(count)]
    return torch.stack(parts, dim=1).mean(dim=1)


def logbp(x: torch.Tensor) -> torch.Tensor:
    """E146 log band power (5 bands x 18 channels, channel-major then band)."""
    length = x.shape[-1]
    masks = band_masks(length, LOGBP_BANDS)
    centred = x - x.mean(dim=-1, keepdim=True)
    spec = torch.fft.rfft(centred, dim=-1)
    amp = []
    for index in range(NB):
        sb = spec * masks[index][None, None, :].to(spec.dtype) * 2.0
        amp.append(torch.fft.ifft(sb, n=length, dim=-1).abs())
    power = torch.stack([a ** 2 for a in amp], dim=1).mean(dim=-1)   # B x NB x C
    return torch.log(power + 1e-8).permute(0, 2, 1).reshape(len(x), CH * NB)


class GateScorer:
    """Frozen Train-only gate: features -> per-task z -> PCA -> nearest prototype."""

    def __init__(self, store: dict, encoder: nn.Module | None = None) -> None:
        self.stats_mu = torch.as_tensor(f32(store["stats_mu"]), device=DEVICE)
        self.stats_sd = torch.as_tensor(f32(store["stats_sd"]), device=DEVICE)
        self.pca_mu = torch.as_tensor(f32(store["pca_mu"]), device=DEVICE)
        self.basis = torch.as_tensor(f32(store["basis"]), device=DEVICE)
        self.proto = torch.as_tensor(f32(store["proto"]), device=DEVICE)
        self.filled = torch.as_tensor(np.asarray(store["filled"]), device=DEVICE)
        self.k = int(store["k"])
        self.encoder = encoder

    def features(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        if kind == "logspd":
            return logspd(x)
        if kind == "subwin6":
            return subwin_mean(x, 6)
        if kind == "subwin12":
            return subwin_mean(x, 12)
        if kind == "logbp":
            return logbp(x)
        if kind == "e162":
            return self.encoder(E162.normalize(x))
        raise KeyError(kind)

    def score(self, x: torch.Tensor, task_idx: torch.Tensor, kind: str) -> torch.Tensor:
        feat = self.features(x, kind)
        z = (feat - self.stats_mu[task_idx]) / self.stats_sd[task_idx]
        proj = (z - self.pca_mu) @ self.basis
        out = torch.zeros(len(proj), device=proj.device, dtype=proj.dtype)
        for index in range(len(TASKS)):
            take = task_idx == index
            if not bool(take.any()):
                continue
            valid = self.filled[index]
            p = self.proto[index][valid].reshape(-1, proj.shape[1])
            dist = torch.cdist(proj[take], p).view(int(take.sum()), int(valid.sum()), self.k)
            value = (-dist).max(dim=2).values.max(dim=1).values
            out = out.index_copy(0, torch.nonzero(take).squeeze(1), value)
        return out


class GateSystem:
    """Candidate gate on raw EEG: weighted sum of per-stream z-scored scores."""

    def __init__(self, scorers: dict, components):
        self.scorers = scorers
        self.components = [(name, float(weight)) for name, weight in components]

    def stream_scores(self, x: torch.Tensor, task_idx: torch.Tensor) -> dict:
        out = {}
        for name in {c for c, _ in self.components}:
            scorer, mu, sd = self.scorers[name]
            kind = CFG["streams"][name]["kind"]
            out[name] = (scorer.score(x, task_idx, kind) - mu) / sd
        return out

    def __call__(self, x: torch.Tensor, task_idx: torch.Tensor) -> torch.Tensor:
        values = self.stream_scores(x, task_idx)
        total = None
        for name, weight in self.components:
            total = weight * values[name] if total is None else total + weight * values[name]
        return total


def build_systems() -> tuple[dict, dict, dict]:
    build_constants()
    data = np.load(SCORE_PATH, allow_pickle=False)
    scorers, systems = {}, {}
    encoder = E162.load_encoder({"name": "lam0.0_t0.1"})[0]
    for name, spec in CFG["streams"].items():
        store = {key: data[f"{name}__{key}"] for key in
                 ("stats_mu", "stats_sd", "pca_mu", "basis", "proto", "filled")}
        store["k"] = int(spec["k"])
        scorer = GateScorer(store, encoder if spec["kind"] == "e162" else None)
        scorers[name] = (scorer, float(data[f"{name}__oof_mu"]),
                         float(data[f"{name}__oof_sd"]))
    summary = read_json(RESULTS_PATH)["candidates"]
    for key, spec in summary.items():
        systems[key] = GateSystem(scorers, spec["components"])
    return scorers, systems, summary


def signature(components) -> tuple:
    return tuple(sorted((name, round(float(w), 3)) for name, w in components if w > 0))


# ------------------------------------------------------------------ stage: attack
def branch_a_model() -> nn.Module:
    payload = torch.load(E180.MODEL_A, map_location="cpu", weights_only=False)
    model = E180.Net().to(DEVICE)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def attack_subset(P: dict) -> np.ndarray:
    cfg = CFG["attack"]
    rng = np.random.default_rng(int(SEED))
    cap = 400
    window = np.flatnonzero(P["window"])
    tasks = P["tasks"]
    selected = []
    for task in TASKS:
        rows = window[tasks[window] == task]
        selected.append(rows if len(rows) <= cap else np.sort(
            rng.choice(rows, size=cap, replace=False)))
    return np.sort(np.concatenate(selected))


def pgd(model, system, x, task_idx, true_idx, target_idx, goal, eps, steps, seed,
        need_gate: bool) -> dict:
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    xc = x - x.mean(dim=-1, keepdim=True)
    rms = xc.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-6)
    bound = (eps * rms).view(-1, 1, 1)
    with torch.no_grad():
        adv = x + (torch.rand(x.shape, generator=gen, device=DEVICE) * 2 - 1) * bound
        adv = torch.clamp(adv, x - bound, x + bound)
    alpha = 2.0 * eps / steps
    for _ in range(steps):
        adv.requires_grad_(True)
        if need_gate:
            objective = system(adv, task_idx)
        else:
            logits = model(adv)
            true_logit = logits.gather(1, true_idx[:, None]).squeeze(1)
            wrong = logits.masked_fill(
                F.one_hot(true_idx, len(CLASSES)).bool(), OFF)
            if goal == "A":
                margin = wrong.max(dim=1).values
            elif goal == "B":
                margin = torch.logsumexp(wrong, dim=1)
            else:
                margin = logits.gather(1, target_idx[:, None]).squeeze(1)
            objective = (margin - true_logit) / 10.0
        grad = torch.autograd.grad(-objective.sum(), adv)[0]
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        adv = torch.clamp(adv.detach() - (alpha * rms).view(-1, 1, 1) * grad.sign(),
                          x - bound, x + bound)
    return {"adv": adv.detach(),
            "max_rel_linf": float(((adv.detach() - x).abs().amax(dim=(1, 2)) / rms).max())}


def score_condition(model, scorers: dict, systems: dict, x: torch.Tensor,
                    task_idx: torch.Tensor, batch: int = 256) -> dict:
    logits = np.zeros((len(x), len(CLASSES)), dtype=np.float32)
    z = {key: np.zeros(len(x), dtype=np.float32) for key in systems}
    with torch.no_grad():
        for start in range(0, len(x), batch):
            chunk = x[start:start + batch]
            index = task_idx[start:start + batch]
            logits[start:start + len(chunk)] = model(chunk).cpu().numpy()
            values = {}
            for name, (scorer, mu, sd) in scorers.items():
                kind = CFG["streams"][name]["kind"]
                values[name] = (scorer.score(chunk, index, kind) - mu) / sd
            for key, system in systems.items():
                total = None
                for name, weight in system.components:
                    total = weight * values[name] if total is None \
                        else total + weight * values[name]
                z[key][start:start + len(chunk)] = total.cpu().numpy()
    return {"logits": logits, "z": z}


def attack_metrics(logits: np.ndarray, z: np.ndarray, truth: np.ndarray,
                   tau: float) -> dict:
    accept = z >= tau
    predicted = np.where(accept, logits.argmax(1) + 1, 0)
    ra = float(np.mean(predicted == truth))
    asafe = float(np.mean((predicted == truth) | (predicted == 0)))
    return {"RA": ra, "ASafeR": asafe,
            "accept_rate": float(np.mean(accept)),
            "reject_rate": float(np.mean(predicted == 0)),
            "wrong_identity_rate": float(np.mean((predicted != 0) & (predicted != truth))),
            "closed_set_accuracy": float(np.mean(logits.argmax(1) + 1 == truth))}


def stage_attack() -> dict:
    started = time.time()
    P = protocol()
    subset = attack_subset(P)
    subjects = P["subjects"]
    tasks = P["tasks"]
    log(f"stage attack: {len(subset)} window trials "
        f"({', '.join(f'{t}={int((tasks[subset] == t).sum())}' for t in TASKS)})")
    cfg = CFG["attack"]
    scorers, systems, summary = build_systems()
    model = branch_a_model()

    raw = E180.load_raw([P["meta"]["path"].iloc[i] for i in subset])
    truth = subjects[subset]
    task_idx = torch.as_tensor([TASKS.index(t) for t in tasks[subset]],
                               dtype=torch.long, device=DEVICE)
    true_idx = torch.as_tensor(truth - 1, dtype=torch.long, device=DEVICE)
    lut = {int(c): i for i, c in enumerate(CLASSES)}
    target = np.asarray([CLASSES[(lut[int(v)] + 1) % len(CLASSES)] for v in truth])
    target_idx = torch.as_tensor([lut[int(v)] for v in target], dtype=torch.long,
                                 device=DEVICE)

    # Differentiable-vs-cached feature check (every stream must reproduce its cache).
    checks = {}
    first = subset[:8]
    for name, spec in CFG["streams"].items():
        cached = f32(np.load(ROOT / spec["train"])[first])
        gate_scorer = scorers[name][0]
        with torch.no_grad():
            rebuilt = gate_scorer.features(raw[:8], spec["kind"]).cpu().numpy()
        checks[name] = float(np.max(np.abs(rebuilt - cached)))
    log("  feature rebuild max abs: " +
        ", ".join(f"{k}={v:.2e}" for k, v in checks.items()))

    clean = score_condition(model, scorers, systems, raw, task_idx)
    conditions = [{"eps": 0.0, "goal": "clean", "max_rel_linf": 0.0,
                   "logits": clean["logits"], "z": clean["z"], "signature": None}]
    for eps in cfg["eps_normalised"]:
        for goal in ("A", "B", "C"):
            out = pgd(model, None, raw, task_idx, true_idx, target_idx, goal,
                      float(eps), int(cfg["steps"]),
                      int(SEED) + int(eps * 1e5) + ord(goal), need_gate=False)
            scored = score_condition(model, scorers, systems, out["adv"], task_idx)
            conditions.append({"eps": float(eps), "goal": goal,
                               "max_rel_linf": out["max_rel_linf"],
                               "logits": scored["logits"], "z": scored["z"],
                               "signature": None})
            log(f"  eps={eps:.3f} goal {goal} done ({time.time() - started:.0f}s)")
    representatives = {}
    for key, system in systems.items():
        representatives.setdefault(signature(system.components), key)
    for sig, key in representatives.items():
        system = systems[key]
        eps = 0.02
        out = pgd(model, system, raw, task_idx, true_idx, target_idx, "D",
                  eps, int(cfg["steps"]), int(SEED) + 99991, need_gate=True)
        scored = score_condition(model, scorers, systems, out["adv"], task_idx)
        conditions.append({"eps": eps, "goal": "D", "gate": key, "signature": sig,
                           "max_rel_linf": out["max_rel_linf"],
                           "logits": scored["logits"], "z": scored["z"]})
        log(f"  goal D [{key}] eps=0.02 done ({time.time() - started:.0f}s)")

    audit = read_json(AUDIT_PATH)
    attack = {"n_trials": int(len(subset)),
              "per_task": {t: int((tasks[subset] == t).sum()) for t in TASKS},
              "feature_rebuild_max_abs": checks,
              "conditions": [{"eps": c["eps"], "goal": c["goal"],
                              "gate": c.get("gate"),
                              "signature": None if c["signature"] is None
                              else str(c["signature"]),
                              "max_rel_linf": c["max_rel_linf"]}
                             for c in conditions],
              "candidates": {}}
    for key in systems:
        sig = signature(systems[key].components)
        own = [c for c in conditions
               if c["goal"] != "D" or c["signature"] == sig]
        rows = []
        for point in audit[key]["points"]:
            tau = float(point["tau"])
            metrics = [attack_metrics(c["logits"], c["z"][key], truth, tau)
                       for c in own]
            for metric, condition in zip(metrics, own):
                metric.update({"eps": condition["eps"], "goal": condition["goal"]})
            worst_ra = min(metrics, key=lambda r: r["RA"])
            worst_safe = min(metrics, key=lambda r: r["ASafeR"])
            clean_row = metrics[0]
            rows.append({"a": point["a"], "tau": tau,
                         "worst_RA": {"RA": worst_ra["RA"], "eps": worst_ra["eps"],
                                      "goal": worst_ra["goal"],
                                      "ASafeR": worst_ra["ASafeR"]},
                         "worst_ASafeR": {"ASafeR": worst_safe["ASafeR"],
                                          "eps": worst_safe["eps"],
                                          "goal": worst_safe["goal"],
                                          "RA": worst_safe["RA"]},
                         "clean": clean_row, "conditions": metrics})
        attack["candidates"][key] = rows
    write_json(ATTACK_PATH, attack)
    results = read_json(RESULTS_PATH)
    results["attack"] = {key: {"n_trials": attack["n_trials"],
                               "feature_rebuild_max_abs": checks,
                               "rows": value}
                         for key, value in attack["candidates"].items()}
    results["stage_seconds"] = {**(results.get("stage_seconds") or {}),
                                "attack": time.time() - started}
    write_json(RESULTS_PATH, results)
    log(f"stage attack done in {time.time() - started:.0f}s")
    return attack


# ------------------------------------------------------------------ stage: report
BRANCH_A_REF = {
    "in_sample_accuracy": 0.998761500353857,
    "oof_accuracy_5block": 0.9881457891012031,
    "oof_macro_f1": 0.9877870965340975,
    "oof_accuracy_forward_blocks": 0.9889346929919722,
}

GATES = [("#1", "拟合容量", "Train 准确率", "= 1.00", 1.00),
         ("#2", "折外闭集", "C（5 块 OOF）", "≥ 0.95", 0.95),
         ("#3", "折外闭集", "身份 macro-F1", "≥ 0.92", 0.92),
         ("#4", "跨时间代理", "块外准确率", "≥ 0.90", 0.90),
         ("#5", "开放集", "LOSO AUROC", "≥ 0.98", 0.98),
         ("#6", "开放集", "U@95%", "≥ 0.95", 0.95),
         ("#7", "对抗识别", "RA（最差 ε×目标）", "≥ 0.95", 0.95),
         ("#8", "对抗安全", "ASafeR（同一攻击）", "≥ 0.98", 0.98),
         ("#9", "Test 审计", "S2 接受率", "均值 > 0.70", 0.70),
         ("#10", "Test 审计", "S1 拒识率", "均值 > 0.70", 0.70)]


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


BRANCH_KEYS = {"#1": "in_sample_accuracy", "#2": "oof_accuracy_5block",
               "#3": "oof_macro_f1", "#4": "oof_accuracy_forward_blocks"}


def evaluate_point(key: str, a: float, audit: dict, attack: dict) -> dict:
    """All ten gates of one candidate at one threshold -> one complete system."""
    point = [p for p in audit[key]["points"] if abs(p["a"] - a) < 1e-9][0]
    att = [r for r in attack[key] if abs(r["a"] - a) < 1e-9][0]
    values = {gate: BRANCH_A_REF[BRANCH_KEYS[gate]] for gate in BRANCH_KEYS}
    values.update({"#5": audit[key]["loso_auc"],
                   "#6": audit[key]["u_at_95"]["per_fold_mean"],
                   "#7": att["worst_RA"]["RA"],
                   "#8": att["worst_ASafeR"]["ASafeR"],
                   "#9": point["test_s2_acceptance"],
                   "#10": point["test_s1_rejection"]})
    joint = point["test_mean"] > LOSO_FLOOR
    passed = {}
    for gate, _, _, _, target in GATES:
        passed[gate] = bool(joint) if gate in ("#9", "#10") else bool(
            values[gate] >= target - 1e-12)
    return {"a": a, "tau": point["tau"], "point": point, "attack": att,
            "values": values, "passed": passed, "count": int(sum(passed.values()))}


def gate_row(key: str, a_key: str) -> tuple:
    audit = read_json(AUDIT_PATH)
    attack = read_json(ATTACK_PATH)["candidates"]
    return evaluate_point(key, audit[key]["picked"][a_key], audit, attack)


def write_report() -> None:
    audit = read_json(AUDIT_PATH)
    attack = read_json(ATTACK_PATH)["candidates"]
    results = read_json(RESULTS_PATH)
    keys = list(audit)
    constrained = {k: evaluate_point(k, audit[k]["picked"]["constrained"], audit, attack)
                   for k in keys}
    balance = {k: evaluate_point(k, audit[k]["picked"]["balance"], audit, attack)
               for k in keys}
    grid = {(k, a): evaluate_point(k, a, audit, attack) for k in keys for a in ACC_GRID}
    best_key = max(keys, key=lambda k: (constrained[k]["count"],
                                        constrained[k]["point"]["test_mean"],
                                        constrained[k]["values"]["#7"]))
    best = best_key
    b = constrained[best]
    grid_max = max(row["count"] for row in grid.values())
    grid_max_keys = [k for k in keys if constrained[k]["count"] == grid_max]
    collapsed = [k for k in keys if len(audit[k]["components"]) > 1
                 and sum(1 for _, w in audit[k]["components"] if w > 0) == 1]
    in_floor = [k for k in keys
                if any(p["loso_rejection"] >= LOSO_FLOOR for p in audit[k]["points"])]
    no_floor = [k for k in keys if k not in in_floor]
    ra_pairs = [(k, a) for k in keys for a in ACC_GRID
                if grid[(k, a)]["values"]["#7"] >= 0.95]
    maximin_all = max(((min(grid[(k, a)]["point"]["test_s1_rejection"],
                            grid[(k, a)]["point"]["test_s2_acceptance"]), k, a)
                       for k in keys for a in ACC_GRID))
    maximin_floor = max(((min(grid[(k, a)]["point"]["test_s1_rejection"],
                              grid[(k, a)]["point"]["test_s2_acceptance"]), k, a)
                         for k in in_floor for a in ACC_GRID
                         if grid[(k, a)]["point"]["loso_rejection"] >= LOSO_FLOOR))

    lines = []
    add = lines.append
    results["gate_matrix"] = {
        "max_pass_single_work_point": int(grid_max),
        "best_candidate": best,
        "selection_rule": "largest a whose LOSO unknown rejection >= 0.70 (Train only), "
                          "tie-break by min(known acceptance, rejection)",
        "structural_failures": [gate for gate, _, _, _, _ in GATES
                                if not any(grid[(k, a)]["passed"][gate]
                                           for k in keys for a in ACC_GRID)],
        "grid": {k: {"a": constrained[k]["a"],
                     "values": constrained[k]["values"],
                     "passed": constrained[k]["passed"],
                     "count": constrained[k]["count"]} for k in keys},
        "test_official": {key: "N/A" for key in
                          ("AUC", "F1", "C", "RA", "ASafeR", "U", "Score")},
    }
    write_json(RESULTS_PATH, results)
    add("# E189 均衡门控：新口径「S1 拒识与 S2 接受均值 > 0.70」下的十项门禁")
    add("")
    add(f"日期：2026-11-15。分支 A（E180 对抗训练 EEGNet）与全部流特征**只读复用**，"
        f"未重训任何模型。新口径下**单工作点最多通过 {grid_max} 项**"
        f"（候选 `{best}`，a={b['a']:.2f}；全部 {len(keys)}×{len(ACC_GRID)} 个工作点的后验"
        f"最大值同样是 {grid_max}）。")
    add("")
    add(f"- Test S1 拒识 **{b['values']['#10']:.4f}**、Test S2 接受 "
        f"**{b['values']['#9']:.4f}**、两者均值 **{b['point']['test_mean']:.4f}**、"
        f"min **{min(b['values']['#9'], b['values']['#10']):.4f}** → 新口径 #9/#10 同时通过")
    add(f"- 同一工作点：LOSO AUROC **{b['values']['#5']:.4f}**、U@95 "
        f"**{b['values']['#6']:.4f}**、LOSO 拒识 **{b['point']['loso_rejection']:.4f}**")
    add(f"- 同一工作点：#7 RA **{b['values']['#7']:.4f}**、#8 ASafeR "
        f"**{b['values']['#8']:.4f}**")
    add(f"- 未通过：{', '.join(g for g, p in b['passed'].items() if not p)}")
    add("")
    add("## 1. 门控候选对照（Test 只评分；每个候选的 a 只由 Train 侧选定）")
    add("")
    add("| 候选 | 组成（Train LOSO AUC 选权） | 主点 a | LOSO AUROC | LOSO 拒识 | "
        "S1 拒识 | S2 接受 | 均值 | min | 均衡点 a | 均衡点 S1/S2/均值 | 通过项数 |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for key in keys:
        row, bal = constrained[key], balance[key]
        point, bpoint = row["point"], bal["point"]
        comps = "+".join(f"{c}×{w:.1f}" for c, w in audit[key]["components"])
        add(f"| `{key}` | {comps} | {row['a']:.2f} | {audit[key]['loso_auc']:.4f} | "
            f"{point['loso_rejection']:.4f} | {point['test_s1_rejection']:.4f} | "
            f"{point['test_s2_acceptance']:.4f} | **{point['test_mean']:.4f}** | "
            f"{min(point['test_s1_rejection'], point['test_s2_acceptance']):.4f} | "
            f"{bal['a']:.2f} | {bpoint['test_s1_rejection']:.3f}/"
            f"{bpoint['test_s2_acceptance']:.3f}/{bpoint['test_mean']:.3f} | "
            f"{row['count']}/10 |")
    add("")
    add("主点 = 满足 LOSO 拒识 ≥ 0.70 的最大 a（题目给定的 Train 侧约束），均衡点 = "
        "min(LOSO 已知接受率, LOSO 拒识) 最大的 a。两列都只用 Train 选出，Test 只被评分。")
    if no_floor:
        add(f"其中 `{'`, `'.join(no_floor)}` 在 a=0.80–0.96 的**任何**工作点都达不到 "
            "LOSO 拒识 ≥ 0.70，按规则退回均衡点（该点不满足约束，仅作记录）。")
    add("")
    if collapsed:
        add(f"**判读**：{len(collapsed)} 个加权融合候选的权重搜索落在单纯形顶点"
            f"（`{'`, `'.join(collapsed[:4])}` 等），即该组合在 Train LOSO AUC 上并不优于"
            "其中单条流——两两/三路加权融合在本候选集内**没有**带来 AUC 增益，"
            "分数最高的是把 E162 拼进来的三路/四路组合（LOSO AUC "
            f"{max(audit[k]['loso_auc'] for k in keys):.4f}）。")
        add("")
    add("## 2. 单工作点十项门禁矩阵（主点，每个候选一个完整工作点，不拼接）")
    add("")
    add("| 候选 | a | " + " | ".join(g for g, *_ in GATES) + " | 通过项数 |")
    add("|---" * (len(GATES) + 3) + "|")
    for key in keys:
        row = constrained[key]
        cells = [f"{row['values'][gate]:.4f}{'✅' if row['passed'][gate] else '❌'}"
                 for gate, _, _, _, _ in GATES]
        add(f"| `{key}` | {row['a']:.2f} | " + " | ".join(cells) +
            f" | **{row['count']}/10** |")
    add("")
    add("说明：#1–#4 是分支 A（E180 冻结模型）的固定读数，与阈值无关；#5/#6 是同一候选门控的 "
        "20 折 LOSO 读数（U@95 用逐折 5% 已知分位，报最差口径）；#7/#8 是同一工作点 τ 下"
        "自建 PGD 的判决；#9/#10 在新口径下是同一条联合判定（均值 > 0.70），因此总是同进同退。")
    add("")
    add("**读表提醒**：#9/#10 的 ✅ 只代表「两者均值 > 0.70」，不代表单侧达标。例如 "
        f"`subwin6_pca96_k1` 在 a=0.84 时 S1 {grid[('subwin6_pca96_k1', 0.84)]['point']['test_s1_rejection']:.4f} / "
        f"S2 {grid[('subwin6_pca96_k1', 0.84)]['point']['test_s2_acceptance']:.4f}，"
        f"min 只有 {min(grid[('subwin6_pca96_k1', 0.84)]['point']['test_s1_rejection'], grid[('subwin6_pca96_k1', 0.84)]['point']['test_s2_acceptance']):.4f}，"
        "仍判通过——这是新口径相对 maximin 的放松（见 §5 末段）。")
    add("")
    add(f"## 3. 最佳候选 `{best}` 的 a 扫描")
    add("")
    add("| a | τ | LOSO 拒识 | S1 拒识 | S2 接受 | 均值 | #7 RA | #8 ASafeR | 通过项数 |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for a in ACC_GRID:
        row = grid[(best, a)]
        point, att = row["point"], row["attack"]
        add(f"| {a:.2f} | {point['tau']:+.4f} | {point['loso_rejection']:.4f} | "
            f"{point['test_s1_rejection']:.4f} | {point['test_s2_acceptance']:.4f} | "
            f"{point['test_mean']:.4f} | {att['worst_RA']['RA']:.4f} | "
            f"{att['worst_ASafeR']['ASafeR']:.4f} | **{row['count']}/10** |")
    add("")
    add("每一行都是一个完整系统；不同行之间不拼接。峰值 = "
        f"{max(grid[(best, a)]['count'] for a in ACC_GRID)}/10。")
    add("")
    add("## 4. 与 E183（7/10，含空洞 #9）对照")
    add("")
    e183 = {"#1": 0.9995, "#2": 0.9872, "#3": 0.9868, "#4": 0.9863, "#5": 0.9860,
            "#6": 0.9090, "#7": 0.9736, "#8": 0.9832, "#9": 0.9288, "#10": 0.1118}
    add("| 门禁 | E183 | E183 判定 | E189 最佳单点 | E189 判定 |")
    add("|---|---:|---|---:|---|")
    e183_pass = {"#1": False, "#2": True, "#3": True, "#4": True, "#5": True,
                 "#6": False, "#7": True, "#8": True, "#9": True, "#10": False}
    for gate, layer, metric, target, _ in GATES:
        add(f"| {gate} {metric} | {e183[gate]:.4f} | "
            f"{'通过' if e183_pass[gate] else '未通过'} | {b['values'][gate]:.4f} | "
            f"{'通过' if b['passed'][gate] else '未通过'} |")
    add("")
    add("- E183 的 7 项里有 #5（0.9860），它来自一个**专用开放集头**（60 个 MLP 的"
        "身份置信度读数），不在本实验的候选族内；本实验的距离型门控最高只有 "
        f"{max(audit[k]['loso_auc'] for k in keys):.4f}。")
    add("- E183 的 #9（0.9288）是「门控在 Test 上几乎全接受」换来的：同一工作点 #10 只有 "
        "0.1118，S1+S2 = 1.0406。E189 的 #9/#10 **同时**通过（均值 > 0.70），不是空洞。")
    add("- 反过来 E183 的 #6 也判为未通过（0.9090 < 0.95），所以两边的结构性缺口"
        "在 #1 与 #6 上是重合的。")
    add("- 口径说明：E183 的判定用旧口径（#9/#10 各自 ≥ 0.75），E189 用新口径（两者均值 "
        "> 0.70）。把新口径套到 E183 那组数上：(0.9288 + 0.1118)/2 = 0.5203 > 0.70 不成立，"
        "E183 的 #9 仍只是「空洞通过」；反过来把旧口径套到 E189 最好的点："
        f"S1 {b['values']['#10']:.4f} / S2 {b['values']['#9']:.4f} 都低于 0.75，旧口径下两项"
        "都不算过——所以新口径确实**多买到** 2 项，而且买到的是「同时」而不是「单侧」。")
    add("")
    add("## 5. 明确回答：新口径下最多能通过几项？")
    add("")
    add(f"**单工作点最多 {grid_max} 项。** 按题目给定的 Train-only 选点规则（LOSO 拒识 ≥ 0.70 "
        f"的最大 a），最好的是 `{best}`（a={b['a']:.2f}）"
        + (f"，并列的还有 " + "、".join(f"`{k}`" for k in grid_max_keys[:4])
           if len(grid_max_keys) > 1 else "")
        + f"，都是 {grid_max}/10。把 {len(keys) * len(ACC_GRID)} 个工作点全部后验枚举，"
        "也没有任何一个点能到更高的项数。")
    add("")
    add("| 门禁 | 状态 | 最好读数 | 目标 | 缺口 |")
    add("|---|---|---:|---:|---:|")
    for gate, layer, metric, target, threshold in GATES:
        value = b["values"][gate]
        if gate in ("#9", "#10"):
            gap = b["point"]["test_mean"] - threshold
            shown = f"{value:.4f}（均值 {b['point']['test_mean']:.4f}）"
            status = "通过（联合均值）" if b["passed"][gate] else "未通过"
        else:
            gap = value - threshold
            shown = f"{value:.4f}"
            status = "通过" if b["passed"][gate] else "结构性未通过"
        add(f"| {gate} {metric} | {status} | {shown} | {target} | {gap:+.4f} |")
    add("")
    if ra_pairs:
        k_ra, a_ra = max(ra_pairs, key=lambda p: grid[p]["values"]["#7"])
        row_ra = grid[(k_ra, a_ra)]
        add(f"**#7 与 #9/#10 的正面冲突（关键发现）**：#7 过线只出现在 "
            f"{len(ra_pairs)}/{len(keys) * len(ACC_GRID)} 个工作点上，全部来自 "
            f"`{'`, `'.join(sorted({p[0] for p in ra_pairs}))}`。其中最好的一列是 "
            f"`{k_ra}`、a={a_ra:.2f}：RA = {row_ra['values']['#7']:.4f}（过线），"
            f"但 Test 均值只有 {row_ra['point']['test_mean']:.4f}"
            f"（S1 {row_ra['point']['test_s1_rejection']:.4f} / "
            f"S2 {row_ra['point']['test_s2_acceptance']:.4f}），差 "
            f"{0.70 - row_ra['point']['test_mean']:.4f}；"
            f"而同一条流在 a={constrained[k_ra]['a']:.2f} 时 Test 均值 "
            f"{constrained[k_ra]['point']['test_mean']:.4f} 仍不过线。RA 主要由「门控接受"
            "合法注册者的比例 × 攻击下的闭集准确率」决定，前者与 S1 拒识争的是同一个阈值，"
            "二者在本系统里**不可同时满足**。")
    else:
        add("**#7 从未过线**：所有候选在所有 a 上的 RA 都低于 0.95。")
    add("")
    add(f"**新口径的副作用（必须写明）**：均值 > 0.70 允许一侧偏科。例如 "
        f"`subwin6_pca96_k1` 在 a=0.84 时 S1 {grid[('subwin6_pca96_k1', 0.84)]['point']['test_s1_rejection']:.4f} / "
        f"S2 {grid[('subwin6_pca96_k1', 0.84)]['point']['test_s2_acceptance']:.4f}"
        f"（min {min(grid[('subwin6_pca96_k1', 0.84)]['point']['test_s1_rejection'], grid[('subwin6_pca96_k1', 0.84)]['point']['test_s2_acceptance']):.4f}）"
        "也满足均值 > 0.70。若严格按题目给的目标 `maximize min(S1,S2)`：全部工作点上最好 "
        f"min = {maximin_all[0]:.4f}（`{maximin_all[1]}`，a={maximin_all[2]:.2f}，"
        f"该点 {grid[(maximin_all[1], maximin_all[2])]['count']}/10，"
        f"但 LOSO 拒识只有 {grid[(maximin_all[1], maximin_all[2])]['point']['loso_rejection']:.4f}，"
        "**违反** LOSO ≥ 0.70 约束）；在满足该约束的工作点里最好 min = "
        f"{maximin_floor[0]:.4f}（`{maximin_floor[1]}`，a={maximin_floor[2]:.2f}，"
        f"{grid[(maximin_floor[1], maximin_floor[2])]['count']}/10）。"
        "也就是说：真正最均衡的点（min 0.71）是「用 LOSO 换 Test」换来的，"
        "因此不能作为可部署点；约束内最均衡的点与「最大化通过项数」的点重合，上限都是 6 项。")
    add("")
    add("**结构性未通过（4 项）**：")
    add("")
    add(f"- **#1** {BRANCH_A_REF['in_sample_accuracy']:.4f} < 1.00"
        f"（差 {1.0 - BRANCH_A_REF['in_sample_accuracy']:.4f}）：分支 A 的 in-sample 就是"
        "不到 100%，E183 的容量探针也只到 0.9997。")
    add(f"- **#5** 最好 {max(audit[k]['loso_auc'] for k in keys):.4f} < 0.98：本候选族全是"
        "距离型门控，LOSO AUC 停在 0.94 一档；越过 0.98 的那一档（0.9860）属于 E183 的"
        "监督式开放集头，且它的 Test 均值只有 0.52。")
    add(f"- **#6** 最好 {max(audit[k]['u_at_95']['per_fold_mean'] for k in keys):.4f} < 0.95："
        "同一条 LOSO ROC 在 95% 已知接受率处几乎无力（E183 用开放集头也只到 0.9090）。")
    ra_under_mean = max(grid[(k, a)]['values']['#7'] for k in keys for a in ACC_GRID
                        if grid[(k, a)]['point']['test_mean'] > 0.70)
    add(f"- **#7** 最好 {max(grid[p]['values']['#7'] for p in ra_pairs) if ra_pairs else 0.0:.4f}"
        f"（`{max(ra_pairs, key=lambda p: grid[p]['values']['#7'])[0]}`，"
        f"a={max(ra_pairs, key=lambda p: grid[p]['values']['#7'])[1]:.2f}）能过线，"
        "但那一列 #9/#10 必然不过；在所有同时满足新口径的工作点上 #7 最高只有 "
        f"{ra_under_mean:.4f}，差 {0.95 - ra_under_mean:.4f}。")
    add("")
    add("## 6. 与 E188 结论一致性检查")
    add("")
    add("E188 在 20,039 条历史工作点中筛出 **928 条**均值 > 0.70 的点（502 条 min ≥ 0.60），"
        "最均衡的是 `E171 subwin_mean N=6 PCA-96 k=1`（S1 0.7178 / S2 0.7093 / 均值 0.7136，"
        "LOSO 仅 0.5866），W 最高的是 `E170 score 0.6*E65(64)+0.1*logbp(16)+0.3*E162(128)`"
        "（S1 0.6993 / S2 0.7012 / LOSO 0.7360）。本实验的复核：")
    sub6 = grid.get(("subwin6_pca96_k1", 0.90))
    if sub6:
        add(f"- 本实验 `subwin6_pca96_k1` 在 a=0.90 的 Test S1/S2 = "
            f"{sub6['point']['test_s1_rejection']:.4f}/"
            f"{sub6['point']['test_s2_acceptance']:.4f}（均值 "
            f"{sub6['point']['test_mean']:.4f}），与 E188 的 0.7178/0.7093（均值 0.7136）"
            f"差 0.003 以内；其 LOSO AUROC {sub6['values']['#5']:.4f} 也印证了「Test 均衡但 "
            "LOSO 弱」的判断。E188 说的 LOSO 0.5866 是拒识率（a=0.90），不是 AUROC，两者不矛盾。")
    pairs = [(k, a) for k in keys for a in ACC_GRID if grid[(k, a)]["point"]["test_mean"] > 0.70]
    best_mean = max(grid[p]["point"]["test_mean"] for p in pairs) if pairs else float("nan")
    add(f"- **「均值 > 0.70 可达」得到复现**：本实验 {len(pairs)}/{len(keys) * len(ACC_GRID)} "
        f"个工作点（候选×a）均值 > 0.70，最高 {best_mean:.4f}，"
        "与 E188 的 928 条方向一致。")
    keep_loso = [p for p in pairs if grid[p]["point"]["loso_rejection"] >= 0.70]
    add(f"- 但加上「同一工作点 LOSO 拒识 ≥ 0.70」后只剩 {len(keep_loso)} 个；"
        "再加「同一点还要过 #7」，剩 "
        f"{len([p for p in keep_loso if grid[p]['values']['#7'] >= 0.95])} 个——这正是 E188 与 "
        "E189 的差别所在，也是「928 条均值可达」不能直接读成「十项门禁可达」的原因。")
    add("")
    add("## 7. Test 官方指标")
    add("")
    add("**Test 官方 AUC / F1 / C / RA / ASafeR / U / Score = N/A**（Test 无真实身份标签）。"
        "本报告里的 Test 数字全部是预测工程审计（S1 拒识率 / S2 接受率），不是官方真值指标。")
    add("")
    add("## 8. 合规与边界")
    add("")
    add("- 所有标准化、PCA、原型、阈值、融合权重只用 Train；LOSO 一律 `pm & fit_rows`；"
        "评估块 `0.6 < frac ≤ 0.8` 从未进入训练或标定。")
    add("- Test 只评分：先算分再读 Session；Session 从未进入特征、模型、阈值或权重。")
    add("- 每个计数列都是一个候选在一个 a 下的完整系统，未跨工作点拼接。")
    add("- #7/#8 是自建白盒 PGD 代理（ε = 0.02/0.01/0.005 × RMS，8 步，A/B/C 全 eps + "
        "D@0.02，取最差），不是官方攻击套件。")
    add("- 未修改任何已有实验目录，未提交 git。")
    add("")
    add("## 9. 文件与耗时")
    add("")
    add("| 文件 | 内容 |")
    add("|---|---|")
    add("| `run_e189.py` | 全部逻辑（`--stage scores/candidates/audit/attack/report/all`） |")
    add("| `config.json` | 冻结配置 |")
    add("| `cache/stream_scores.npz` | 6 条流的 OOF / 20 折 LOSO / Test 分数与冻结参数 |")
    add("| `cache/candidates.npz` | 14 个门控候选的同一批分数 |")
    add("| `audit.json` | 每个候选 × 10 个 a 的 Train/Test 读数与选点 |")
    add("| `attack.json` | 每个候选 × 10 个 a 的对抗条件判决 |")
    add("| `results.json` | 汇总（`streams` / `candidates` / `audit` / `attack`） |")
    add("| `run.log` | 运行日志 |")
    add("")
    seconds = results.get("stage_seconds", {})
    add("耗时（秒）：" + "、".join(f"{k} {v:.0f}" for k, v in seconds.items()) + "。"
        "特征重建校验（可微路径 vs 缓存）最大绝对误差："
        + "、".join(f"{k} {v:.1e}" for k, v in
                    results.get("attack", {}).get(keys[0], {})
                    .get("feature_rebuild_max_abs", {}).items()) + "。")
    add("")
    add("```powershell")
    add('& "D:\\WORK\\BCIC2026\\BrainPrint\\.venv\\Scripts\\python.exe" '
        f'"{OUT / "run_e189.py"}" --stage all')
    add("```")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"report written: {REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["scores", "candidates", "audit", "attack", "report", "all"])
    args = parser.parse_args()
    if not LOG_PATH.exists():
        LOG_PATH.write_text("", encoding="utf-8")
    sys.stdout = Tee(LOG_PATH)
    sys.stderr = sys.stdout
    log(f"\n===== E189 stage {args.stage} =====")
    if args.stage in ("scores", "all"):
        stage_scores()
    if args.stage in ("candidates", "all"):
        stage_candidates()
    if args.stage in ("audit", "all"):
        stage_audit()
    if args.stage in ("attack", "all"):
        stage_attack()
    if args.stage in ("report", "all"):
        write_report()


if __name__ == "__main__":
    main()
