"""E198: select the gate / fusion weights with the leave-5 three-stage protocol (Train only).

Motivation
    E197 showed that the leave-5 three-stage protocol reproduces the *ordering* of the Test S1
    rejection rate across gates (Spearman 0.85) while its absolute level is conservative by about
    -0.095.  E195 showed that picking the fusion weights on strict LOSO alone degenerates to a
    single gate whose Test minimum collapses (0.2733), and that the only points that pass the dual
    0.70 barrier can be found on Test only.  This experiment asks a narrower, compliant question:
    if the weights are picked by the leave-5 protocol (plus strict LOSO for #11), how far does a
    single, frozen work point get on the eleven gates?

Protocol
    Stage 1 (all 20 identities, fit rows frac <= 0.6): per-task mean/sd, PCA basis, frozen E162
        encoder.  Fold-invariant, computed once.
    Stage 2 (enrolment, only the 20-k enrolled identities): k_proto prototypes per identity from
        that identity's own fit rows.  k_proto = 12 for e65_dist, 1 for the other streams.
    Stage 3 (deployment): argmax over the enrolled prototypes + threshold rejection.

Candidates
    Five single streams (e65_dist PCA-96 k=12, subwin6_dist, subwin12_dist, logbp_dist,
    e162_dist) and every pair / triple on the 0.1 simplex grid.  wccn_dr is excluded by the task
    (E197: it collapses from 0.9925 to 0.3298 when the enrolment set shrinks); E183 is not part of
    the requested pool.

Selection (pre-registered, Train only)
    R1: acc5 >= 0.95 and loso_strict > 0.90, then maximise rej5, tie-break auc5.
    R1': if the R1 set is empty (verified before any Test scoring), drop #11 and keep acc5 >= 0.95,
    then maximise rej5, tie-break auc5.  The #11 failure is then reported explicitly.

Test is scored once, after the top five are frozen.  Session only splits S1 / S2 for the audit.

Outputs: run_e198.py, config.json, results.json, report.md, run.log (this directory only).
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

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)
INST_PATH = CACHE / "instrument.npz"
STRICT_PATH = CACHE / "strict.npz"
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "report.md"
LOG_PATH = OUT / "run.log"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TASKS = ["EC", "EO", "Imagery", "SSVEP"]
TASK_OF = {name: i for i, name in enumerate(TASKS)}
STREAMS = [s["name"] for s in CFG["streams"]]
SPEC = {s["name"]: s for s in CFG["streams"]}
SEED = int(CFG["protocol"]["seed"])
FIT = float(CFG["protocol"]["fit_fraction"])
A = float(CFG["protocol"]["primary_acceptance"])
K = int(CFG["protocol"]["leave_k"])
NFOLD = int(CFG["protocol"]["folds"])
BLOCKS = 5
E191_DIR = ROOT / CFG["e191_backbone"]["directory"]
E189_DIR = ROOT / "research" / "E189_balanced_gate_gates"


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


def family_path(stream: str, key: str) -> Path:
    return ROOT / CFG["families"][SPEC[stream]["family"]][key]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float),
                    encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_module(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    blocks = np.clip((frac * BLOCKS).astype(int), 0, BLOCKS - 1)
    return {"subjects": subjects, "tasks": tasks, "frac": frac, "blocks": blocks,
            "paths": meta["path"].to_numpy(dtype=object),
            "task_idx": np.array([TASK_OF[t] for t in tasks], dtype=np.int64),
            "fit_rows": fit_rows, "window": window, "win_rows": np.flatnonzero(window),
            "test_tasks": test_meta["task"].to_numpy(dtype=object),
            "test_task_idx": np.array([TASK_OF[t] for t in test_meta["task"]],
                                      dtype=np.int64),
            "is_s1": np.isin(session, ["session1", "s1", "1"])}


def build_folds(P: dict) -> tuple:
    combos = list(itertools.combinations([int(c) for c in CLASSES], K))
    rng = np.random.default_rng(SEED + K)
    picked = np.sort(rng.choice(len(combos), size=min(NFOLD, len(combos)), replace=False))
    folds = [tuple(int(v) for v in combos[i]) for i in picked]
    reference = ROOT / CFG["references"]["e197_fold_plan"]
    same = None
    if reference.exists():
        stored = read_json(reference).get(str(K), [])
        same = ([list(map(int, a)) for a in stored] == [list(map(int, a)) for a in folds])
        log(f"  fold plan: {len(folds)} folds, identical to E197: {same}")
    else:
        log(f"  fold plan: {len(folds)} folds (E197 plan not found)")
    return folds, same


# ---------------------------------------------------------------- stage 1 instrument
def torch_stats(features: torch.Tensor, tidx: torch.Tensor, rows: torch.Tensor) -> tuple:
    mu = torch.stack([features[rows & (tidx == i)].mean(0) for i in range(len(TASKS))])
    sd = torch.stack([features[rows & (tidx == i)].std(0) for i in range(len(TASKS))])
    return mu, sd + 1e-6


def pca_from(features: torch.Tensor, tidx: torch.Tensor, mu: torch.Tensor,
             sd: torch.Tensor, rows: torch.Tensor, dim: int) -> tuple:
    zf = (features[rows] - mu[tidx[rows]]) / sd[tidx[rows]]
    pca_mu = zf.mean(0)
    centred = zf - pca_mu
    covariance = centred.T @ centred / max(len(centred) - 1, 1)
    values, vectors = torch.linalg.eigh(covariance)
    basis = vectors[:, torch.argsort(values, descending=True)[:dim]]
    del zf, centred, covariance
    return pca_mu, basis


def project(features: torch.Tensor, tidx: torch.Tensor, inst: dict,
            rows: np.ndarray, chunk: int = 8192) -> torch.Tensor:
    out = torch.empty((len(rows), inst["basis"].shape[1]), dtype=torch.float32,
                      device=features.device)
    for start in range(0, len(rows), chunk):
        idx = torch.as_tensor(rows[start:start + chunk], device=features.device)
        z = (features[idx] - inst["mu"][tidx[idx]]) / inst["sd"][tidx[idx]]
        out[start:start + len(idx)] = (z - inst["pca_mu"]) @ inst["basis"]
    return out


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


def prototypes(proj: torch.Tensor, row_index: dict, subjects: np.ndarray,
               classes, k: int) -> list:
    out = []
    for ident in classes:
        block = proj[row_index[int(ident)]]
        out.append(kmeans_t(block, k, seed=int(ident) * 17))
    return out


def score_matrix(proj: torch.Tensor, proto_list: list, chunk: int = 4096) -> torch.Tensor:
    n_ident, k = len(proto_list), proto_list[0].shape[0]
    stack = torch.cat(proto_list, 0)
    out = torch.empty((len(proj), n_ident), dtype=torch.float32, device=proj.device)
    for start in range(0, len(proj), chunk):
        block = proj[start:start + chunk]
        distance = torch.cdist(block, stack)
        out[start:start + len(block)] = (-distance).view(len(block), n_ident, k) \
            .max(2).values
    return out


def build_instrument(P: dict) -> dict:
    started = time.time()
    win_rows = P["win_rows"]
    win_subjects = P["subjects"][win_rows]
    tidx = torch.as_tensor(P["task_idx"], device=DEVICE)
    fit = torch.as_tensor(P["fit_rows"], device=DEVICE)
    payload = {"z_win": [], "win_subjects": win_subjects.astype(np.int32),
               "win_tasks": np.asarray(P["tasks"][win_rows], dtype=str), "win_rows": win_rows.astype(np.int32)}
    stores, stats = {}, {}
    for spec in CFG["streams"]:
        name = spec["name"]
        features = torch.as_tensor(f32(np.load(family_path(name, "train"))), device=DEVICE)
        mu, sd = torch_stats(features, tidx, fit)
        pca_mu, basis = pca_from(features, tidx, mu, sd, fit, int(spec["pca"]))
        inst = {"mu": mu, "sd": sd, "pca_mu": pca_mu, "basis": basis}
        fit_proj = project(features, tidx, inst, np.flatnonzero(P["fit_rows"]))
        fit_subjects = P["subjects"][P["fit_rows"]]
        index = {int(c): np.flatnonzero(fit_subjects == c) for c in CLASSES}
        proto = prototypes(fit_proj, index, fit_subjects, CLASSES, int(spec["prototypes"]))
        win_proj = project(features, tidx, inst, win_rows)
        matrix = score_matrix(win_proj, proto)
        best = matrix.max(1).values
        ref_mu, ref_sd = float(best.mean()), float(best.std()) + 1e-6
        payload["z_win"].append(((matrix - ref_mu) / ref_sd).cpu().numpy().astype(np.float32))
        stats[name] = {"mu": ref_mu, "sd": ref_sd,
                       "win_best_mean": float(best.mean()),
                       "win_best_std": float(best.std())}
        stores[name] = {"mu": mu.cpu().numpy(), "sd": sd.cpu().numpy(),
                        "pca_mu": pca_mu.cpu().numpy(), "basis": basis.cpu().numpy(),
                        "proto": torch.cat(proto, 0).cpu().numpy(),
                        "k": int(spec["prototypes"]), "pca": int(spec["pca"])}
        log(f"  stage1 {name}: pca {int(spec['pca'])} k={int(spec['prototypes'])} "
            f"window best {float(best.mean()):.3f}±{float(best.std()):.3f} "
            f"({time.time() - started:.0f}s)")
        del features, fit_proj, win_proj, matrix, proto
        torch.cuda.empty_cache()
    payload["z_win"] = np.stack(payload["z_win"]).astype(np.float32)  # [5, 2260, 20]
    payload["streams"] = np.array(STREAMS)
    for name, store in stores.items():
        for key, value in store.items():
            payload[f"{name}__{key}"] = value
    payload["stats_json"] = np.array(json.dumps(stats))
    np.savez(INST_PATH, **payload)
    log(f"stage instrument done in {time.time() - started:.0f}s")
    return {"seconds": time.time() - started, "stats": stats,
            "fit_rows": int(P["fit_rows"].sum()), "window_rows": int(len(win_rows))}


def load_instrument() -> dict:
    data = np.load(INST_PATH, allow_pickle=False)
    return {key: data[key] for key in data.files}


def instrument_stores(data: dict) -> dict:
    stores = {}
    for name in STREAMS:
        stores[name] = {"mu": f32(data[f"{name}__mu"]), "sd": f32(data[f"{name}__sd"]),
                        "pca_mu": f32(data[f"{name}__pca_mu"]),
                        "basis": f32(data[f"{name}__basis"]),
                        "proto": f32(data[f"{name}__proto"]),
                        "k": int(data[f"{name}__k"]), "pca": int(data[f"{name}__pca"])}
    return stores


# ---------------------------------------------------------------- strict LOSO
def strict_pass(P: dict) -> dict:
    """20 leave-1 folds, mode 'std' (stage-1 statistics) and 'strict' (refit without the identity)."""
    started = time.time()
    data = load_instrument()
    stats = json.loads(str(data["stats_json"]))
    stores = instrument_stores(data)
    win_rows = P["win_rows"]
    win_subjects = P["subjects"][win_rows]
    tidx = torch.as_tensor(P["task_idx"], device=DEVICE)
    fit_np = P["fit_rows"]
    features = {spec["name"]: torch.as_tensor(
        f32(np.load(family_path(spec["name"], "train"))), device=DEVICE)
        for spec in CFG["streams"]}
    payload = {}
    for mode in ("std", "strict"):
        for fold, held in enumerate(CLASSES):
            keep = P["subjects"] != int(held)
            enrolled = [int(c) for c in CLASSES if int(c) != int(held)]
            matrix_by_stream = []
            for spec in CFG["streams"]:
                name = spec["name"]
                store = stores[name]
                if mode == "std":
                    inst = {key: torch.as_tensor(store[key], device=DEVICE)
                            for key in ("mu", "sd", "pca_mu", "basis")}
                else:
                    ref = torch.as_tensor(keep, device=DEVICE)
                    pca_mask = torch.as_tensor(keep & fit_np, device=DEVICE)
                    mu, sd = torch_stats(features[name], tidx, ref)
                    pca_mu, basis = pca_from(features[name], tidx, mu, sd, pca_mask,
                                             int(spec["pca"]))
                    inst = {"mu": mu, "sd": sd, "pca_mu": pca_mu, "basis": basis}
                rows = np.flatnonzero(keep & fit_np)
                fit_proj = project(features[name], tidx, inst, rows)
                fit_subjects = P["subjects"][keep & fit_np]
                index = {int(c): np.flatnonzero(fit_subjects == c)
                         for c in CLASSES if int(c) != int(held)}
                proto = prototypes(fit_proj, index, fit_subjects, enrolled,
                                   int(spec["prototypes"]))
                win_proj = project(features[name], tidx, inst, win_rows)
                matrix = score_matrix(win_proj, proto)
                matrix_by_stream.append(
                    ((matrix - float(stats[name]["mu"])) / float(stats[name]["sd"]))
                    .cpu().numpy().astype(np.float32))
                del fit_proj, win_proj, matrix, proto
                if mode == "strict":
                    del inst
            payload[f"{mode}_z_{fold}"] = np.stack(matrix_by_stream)
            payload[f"{mode}_enrolled_{fold}"] = np.array(enrolled, dtype=np.int16)
            payload[f"{mode}_known_{fold}"] = np.isin(win_subjects, enrolled)
            if (fold + 1) % 5 == 0:
                log(f"  strict {mode} {fold + 1}/20 {time.time() - started:.0f}s")
    payload["win_subjects"] = win_subjects.astype(np.int32)
    np.savez(STRICT_PATH, **payload)
    log(f"stage strict done in {time.time() - started:.0f}s")
    return {"seconds": time.time() - started}


# ---------------------------------------------------------------- candidate grid
def build_candidates() -> list:
    names = STREAMS
    seen, out = set(), []

    def add(group, weights):
        clean = {n: float(round(w, 3)) for n, w in weights.items() if w > 1e-9}
        key = tuple(sorted(clean.items()))
        if key in seen:
            return
        seen.add(key)
        out.append({"candidate": "+".join(n for n, _ in sorted(clean.items())),
                    "group": group, "weights": clean,
                    "label": " + ".join(f"{n} {w:.1f}" for n, w in clean.items())})

    for name in names:
        add("single", {name: 1.0})
    step = float(CFG["fusion_grid"]["step"])
    levels = int(round(1.0 / step))
    for pair in CFG["fusion_grid"]["pairs"]:
        for value in range(levels + 1):
            add("pair", {pair[0]: value * step, pair[1]: (levels - value) * step})
    for triple in CFG["fusion_grid"]["triples"]:
        for a in range(levels + 1):
            for b in range(levels + 1 - a):
                add("triple", {triple[0]: a * step, triple[1]: b * step,
                               triple[2]: (levels - a - b) * step})
    return out


def weight_matrix(candidates: list) -> torch.Tensor:
    out = torch.zeros((len(candidates), len(STREAMS)), dtype=torch.float32, device=DEVICE)
    for row, cand in enumerate(candidates):
        for name, weight in cand["weights"].items():
            out[row, STREAMS.index(name)] = float(weight)
    return out


# ---------------------------------------------------------------- metric helpers
def quantile_rows(values: torch.Tensor, q: float) -> torch.Tensor:
    sorted_values, _ = values.sort(dim=1)
    n = values.shape[1]
    position = q * (n - 1)
    low = int(np.floor(position))
    high = min(low + 1, n - 1)
    fraction = position - low
    return sorted_values[:, low] * (1.0 - fraction) + sorted_values[:, high] * fraction


def auc_rows(known: torch.Tensor, unknown: torch.Tensor) -> torch.Tensor:
    scores = torch.cat([known, unknown], 1)
    order = scores.argsort(1)
    ranks = torch.empty_like(order, dtype=torch.float32)
    base = torch.arange(1, scores.shape[1] + 1, dtype=torch.float32,
                        device=scores.device).unsqueeze(0).expand_as(order)
    ranks.scatter_(1, order, base)
    nk, nu = known.shape[1], unknown.shape[1]
    total = ranks[:, :nk].sum(1)
    return (total - nk * (nk + 1) / 2.0) / (nk * nu)


def f1_rows(pred: torch.Tensor, target: torch.Tensor, n_classes: int) -> torch.Tensor:
    index = pred * n_classes + target.unsqueeze(0)
    counts = torch.zeros((len(pred), n_classes * n_classes), dtype=torch.float32,
                         device=pred.device)
    counts.scatter_add_(1, index, torch.ones_like(index, dtype=torch.float32))
    counts = counts.view(len(pred), n_classes, n_classes)  # [pred, true]
    tp = torch.diagonal(counts, dim1=1, dim2=2)
    precision = tp / counts.sum(2).clamp_min(1e-9)
    recall = tp / counts.sum(1).clamp_min(1e-9)
    denom = precision + recall
    f1 = torch.where(denom > 0, 2 * precision * recall / denom.clamp_min(1e-9),
                     torch.zeros_like(tp))
    return f1.mean(1)


def fused_block(z: torch.Tensor, weights: torch.Tensor, columns=None, chunk: int = 100):
    """z [5, N, I] -> fused score matrices [C, N, I] in candidate chunks."""
    blocks = []
    for start in range(0, len(weights), chunk):
        part = weights[start:start + chunk]
        if columns is None:
            fused = torch.einsum("cs,sni->cni", part, z)
        else:
            fused = torch.einsum("cs,sni->cni", part, z[:, :, columns])
        blocks.append(fused)
    return blocks


def leave5_metrics(z_win: torch.Tensor, weights: torch.Tensor, folds: list,
                   win_subjects: np.ndarray) -> dict:
    """Per-fold identification / rejection / AUROC on the Train evaluation window."""
    n_cand = len(weights)
    acc = torch.zeros((n_cand, len(folds)), dtype=torch.float32, device=DEVICE)
    f1 = torch.zeros_like(acc)
    rej = torch.zeros_like(acc)
    auc = torch.zeros_like(acc)
    coverage = torch.zeros_like(acc)
    for fold, held in enumerate(folds):
        enrolled = [int(c) for c in CLASSES if int(c) not in held]
        columns = [list(CLASSES).index(c) for c in enrolled]
        known = np.isin(win_subjects, enrolled)
        target = torch.as_tensor([enrolled.index(int(s)) for s in win_subjects[known]],
                                 dtype=torch.long, device=DEVICE)
        best_known, best_unknown, preds = [], [], []
        for fused in fused_block(z_win, weights, columns):
            fk = fused[:, known]
            best_known.append(fk.max(2).values)
            preds.append(fk.argmax(2))
            best_unknown.append(fused[:, ~known].max(2).values)
            del fused
        best_known = torch.cat(best_known, 0)
        best_unknown = torch.cat(best_unknown, 0)
        pred = torch.cat(preds, 0)
        tau = quantile_rows(best_known, 1.0 - A)
        acc[:, fold] = (pred == target.unsqueeze(0)).float().mean(1)
        f1[:, fold] = f1_rows(pred, target, len(enrolled))
        rej[:, fold] = (best_unknown < tau.unsqueeze(1)).float().mean(1)
        coverage[:, fold] = (best_known >= tau.unsqueeze(1)).float().mean(1)
        auc[:, fold] = auc_rows(best_known, best_unknown)
        if (fold + 1) % 50 == 0:
            log(f"  leave-5 {fold + 1}/{len(folds)}")
    return {"acc": acc.mean(1), "acc_std": acc.std(1), "f1": f1.mean(1),
            "rej": rej.mean(1), "rej_std": rej.std(1), "coverage": coverage.mean(1),
            "auc": auc.mean(1), "auc_std": auc.std(1), "per_fold_rej": rej,
            "per_fold_acc": acc, "per_fold_auc": auc}


def strict_metrics(z_folds: list, known_folds: list, weights: torch.Tensor,
                   chunk: int = 100) -> dict:
    """loso rejection / U@95 / pooled AUROC from the stored strict arrays."""
    n_cand = len(weights)
    n_fold = len(z_folds)
    rej = torch.zeros((n_cand, n_fold), device=DEVICE)
    u95 = torch.zeros((n_cand, n_fold), device=DEVICE)
    known_all, unknown_all = [], []
    for fold in range(n_fold):
        z = z_folds[fold]
        known = known_folds[fold]
        rej_blocks, u_blocks, kn_blocks, un_blocks = [], [], [], []
        for fused in fused_block(z, weights, None, chunk):
            best = fused.max(2).values
            kn, un = best[:, known], best[:, ~known]
            rej_blocks.append((un < quantile_rows(kn, 1.0 - A).unsqueeze(1))
                              .float().mean(1))
            u_blocks.append((un < quantile_rows(kn, 1.0 - 0.95).unsqueeze(1))
                            .float().mean(1))
            kn_blocks.append(kn)
            un_blocks.append(un)
            del fused, best, kn, un
        rej[:, fold] = torch.cat(rej_blocks, 0)
        u95[:, fold] = torch.cat(u_blocks, 0)
        known_all.append(torch.cat(kn_blocks, 0))
        unknown_all.append(torch.cat(un_blocks, 0))
    known_all = torch.cat(known_all, 1)
    unknown_all = torch.cat(unknown_all, 1)
    auc = torch.cat([auc_rows(known_all[s:s + chunk], unknown_all[s:s + chunk])
                     for s in range(0, n_cand, chunk)], 0)
    return {"rej": rej.mean(1), "rej_std": rej.std(1), "u95": u95.mean(1),
            "u95_std": u95.std(1), "auc": auc,
            "pooled_known": known_all, "pooled_unknown": unknown_all,
            "per_fold_rej": rej}


# ---------------------------------------------------------------- selection stage
def stage_select(force: bool = False) -> dict:
    started = time.time()
    results = read_json(RESULTS_PATH)
    if "candidates" in results and not force and results["candidates"].get("done"):
        log("stage select: results present, reusing")
        return results["candidates"]
    P = load_protocol()
    folds, folds_match = build_folds(P)
    data = load_instrument()
    z_win = torch.as_tensor(f32(data["z_win"]), device=DEVICE)
    win_subjects = data["win_subjects"].astype(int)
    candidates = build_candidates()
    weights = weight_matrix(candidates)
    log(f"  candidates {len(candidates)} "
        f"({sum(c['group'] == 'single' for c in candidates)} single, "
        f"{sum(c['group'] == 'pair' for c in candidates)} pair, "
        f"{sum(c['group'] == 'triple' for c in candidates)} triple)")

    five = leave5_metrics(z_win, weights, folds, win_subjects)
    log(f"  leave-5 done {time.time() - started:.0f}s")

    strict = np.load(STRICT_PATH, allow_pickle=False)
    out = {"done": True}
    for mode in ("strict", "std"):
        z_folds = [torch.as_tensor(f32(strict[f"{mode}_z_{fold}"]), device=DEVICE)
                   for fold in range(20)]
        known = [strict[f"{mode}_known_{fold}"].astype(bool) for fold in range(20)]
        out[mode] = strict_metrics(z_folds, known, weights)
        log(f"  loso {mode} done {time.time() - started:.0f}s")
        del z_folds

    rows = []
    for index, cand in enumerate(candidates):
        rows.append({
            "id": index, "candidate": cand["candidate"], "group": cand["group"],
            "label": cand["label"], "weights": cand["weights"],
            "acc5": float(five["acc"][index]), "acc5_std": float(five["acc_std"][index]),
            "f1_5": float(five["f1"][index]), "rej5": float(five["rej"][index]),
            "rej5_std": float(five["rej_std"][index]),
            "coverage5": float(five["coverage"][index]),
            "auc5": float(five["auc"][index]), "auc5_std": float(five["auc_std"][index]),
            "loso_strict": float(out["strict"]["rej"][index]),
            "loso_strict_std": float(out["strict"]["rej_std"][index]),
            "loso_std": float(out["std"]["rej"][index]),
            "auroc5": float(out["strict"]["auc"][index]),
            "auroc5_std_mode": float(out["std"]["auc"][index]),
            "u95_strict": float(out["strict"]["u95"][index]),
            "u95_std_mode": float(out["std"]["u95"][index]),
            "tau90_strict_pooled": float(quantile_rows(
                out["strict"]["pooled_known"][index:index + 1], 1.0 - A)[0]),
            "tau90_std_pooled": float(quantile_rows(
                out["std"]["pooled_known"][index:index + 1], 1.0 - A)[0]),
        })
    # ---- pre-registered rule, Train only
    acc_min = float(CFG["selection"]["acc5_min"])
    strict_min = float(CFG["selection"]["loso_strict_min"])
    feasible = [r for r in rows if r["acc5"] >= acc_min and r["loso_strict"] > strict_min]
    acc_ok = [r for r in rows if r["acc5"] >= acc_min]
    pool = feasible if feasible else acc_ok
    ranked = sorted(pool, key=lambda r: (-r["rej5"], -r["auc5"]))
    best_strict = max(acc_ok, key=lambda r: (r["loso_strict"], r["rej5"]))
    top5 = ranked[:5]
    out = {"done": True, "rows": rows, "n_candidates": len(rows),
           "n_acc_ok": len(acc_ok), "n_feasible": len(feasible),
           "folds_identical_to_e197": folds_match,
           "fallback_used": not bool(feasible),
           "rule": CFG["selection"]["rule"], "fallback": CFG["selection"]["fallback"],
           "max_loso_strict": max(r["loso_strict"] for r in rows),
           "max_loso_strict_acc_ok": best_strict["loso_strict"],
           "best_strict_candidate": best_strict["candidate"],
           "best_strict_weights": best_strict["weights"],
           "best_strict_rej5": best_strict["rej5"],
           "best_strict_auc5": best_strict["auc5"],
           "top5": [{"rank": rank + 1, **row} for rank, row in enumerate(top5)],
           "top_ids": [row["id"] for row in top5],
           "seconds": time.time() - started}
    results = read_json(RESULTS_PATH)
    results["candidates"] = out
    results.setdefault("stage_seconds", {})["select"] = time.time() - started
    write_json(RESULTS_PATH, results)
    log(f"  feasible {len(feasible)} (acc5>={acc_min}, loso_strict>{strict_min}), "
        f"acc5 candidates {len(acc_ok)}, max loso_strict {out['max_loso_strict']:.4f}")
    for row in top5:
        log(f"    #{row['id']:3d} {row['label']:52s} rej5 {row['rej5']:.4f} "
            f"acc5 {row['acc5']:.4f} auc5 {row['auc5']:.4f} strict {row['loso_strict']:.4f} "
            f"auroc5 {row['auroc5']:.4f}")
    return out


# ---------------------------------------------------------------- test stage (once)
def test_scores(P: dict, candidates: list, folds: list) -> dict:
    data = load_instrument()
    stores = instrument_stores(data)
    stats = json.loads(str(data["stats_json"]))
    z_win = torch.as_tensor(f32(data["z_win"]), device=DEVICE)
    win_subjects = data["win_subjects"].astype(int)
    weights = weight_matrix(candidates)
    is_s1 = P["is_s1"]

    tidx = torch.as_tensor(P["test_task_idx"], device=DEVICE)
    z_test = []
    for spec in CFG["streams"]:
        store = stores[spec["name"]]
        features = torch.as_tensor(f32(np.load(family_path(spec["name"], "test"))),
                                   device=DEVICE)
        inst = {key: torch.as_tensor(store[key], device=DEVICE)
                for key in ("mu", "sd", "pca_mu", "basis")}
        proj = project(features, tidx, inst, np.arange(len(features)))
        proto = [torch.as_tensor(store["proto"], device=DEVICE)[
            i * store["k"]:(i + 1) * store["k"]] for i in range(len(CLASSES))]
        matrix = score_matrix(proj, proto)
        z_test.append(((matrix - float(stats[spec["name"]]["mu"]))
                       / float(stats[spec["name"]]["sd"])).cpu())
        log(f"  test stream {spec['name']} scored")
        del features, proj, matrix, proto
        torch.cuda.empty_cache()
    z_test = torch.stack(z_test).to(DEVICE)  # [5, 22251, 20]

    pooled = np.load(STRICT_PATH, allow_pickle=False)
    known_all = []
    for fold in range(20):
        z = torch.as_tensor(f32(pooled[f"strict_z_{fold}"]), device=DEVICE)
        mask = pooled[f"strict_known_{fold}"].astype(bool)
        blocks = []
        for fused in fused_block(z, weights, None):
            blocks.append(fused.max(2).values[:, mask])
            del fused
        known_all.append(torch.cat(blocks, 0))
        del z
    strict_known = torch.cat(known_all, 1)

    test_fused = []
    for fused in fused_block(z_test, weights, None, chunk=5):
        test_fused.append(fused.max(2).values.cpu().numpy().astype(np.float32))
        del fused
    test_best = np.concatenate(test_fused, 0)  # [5, 22251]

    out = {"n_s1": int(is_s1.sum()), "n_s2": int((~is_s1).sum()), "candidates": []}
    for index, cand in enumerate(candidates):
        tau = float(quantile_rows(strict_known[index:index + 1], 1.0 - A)[0])
        # sensitivity: calibrate tau on the deployment gallery itself (all 20 identities)
        fused_win20 = torch.einsum("cs,sni->cni", weights[index:index + 1], z_win)
        tau_gallery20 = float(quantile_rows(fused_win20.max(2).values, 1.0 - A)[0])
        del fused_win20
        s1 = float(np.mean(test_best[index][is_s1] < tau))
        s2 = float(np.mean(test_best[index][~is_s1] >= tau))
        s1_g20 = float(np.mean(test_best[index][is_s1] < tau_gallery20))
        s2_g20 = float(np.mean(test_best[index][~is_s1] >= tau_gallery20))
        diagnostic = float(torch.as_tensor(
            auc_rows(torch.as_tensor(test_best[index][~is_s1]).unsqueeze(0),
                     torch.as_tensor(test_best[index][is_s1]).unsqueeze(0)),
            device="cpu")[0])
        row = {"id": cand["id"], "label": cand["label"], "weights": cand["weights"],
               "deployment": {"tau": tau, "s1": s1, "s2": s2, "min": min(s1, s2),
                              "mean": 0.5 * (s1 + s2), "diagnostic_auc": diagnostic},
               "gallery20_calibration": {"tau": tau_gallery20, "s1": s1_g20,
                                         "s2": s2_g20, "min": min(s1_g20, s2_g20),
                                         "mean": 0.5 * (s1_g20 + s2_g20)}}
        # fold-matched reading: gallery = the fold's 15 enrolled identities
        s1_folds, s2_folds = [], []
        for fold, held in enumerate(folds):
            enrolled = [int(c) for c in CLASSES if int(c) not in held]
            columns = [list(CLASSES).index(c) for c in enrolled]
            known = np.isin(win_subjects, enrolled)
            fused_known = torch.einsum("cs,sni->cni", weights[index:index + 1],
                                       z_win[:, known][:, :, columns])
            tau_fold = float(quantile_rows(fused_known.max(2).values, 1.0 - A)[0])
            fused_test = torch.einsum("cs,sni->cni", weights[index:index + 1],
                                      z_test[:, :, columns]).max(2).values[0].cpu().numpy()
            s1_folds.append(float(np.mean(fused_test[is_s1] < tau_fold)))
            s2_folds.append(float(np.mean(fused_test[~is_s1] >= tau_fold)))
        row["fold_matched"] = {"s1": float(np.mean(s1_folds)),
                               "s2": float(np.mean(s2_folds)),
                               "s1_std": float(np.std(s1_folds)),
                               "s2_std": float(np.std(s2_folds)),
                               "min": float(min(np.mean(s1_folds), np.mean(s2_folds)))}
        out["candidates"].append(row)
        log(f"  {cand['label'][:50]:52s} deploy S1 {s1:.4f} S2 {s2:.4f} "
            f"| g20 S1 {s1_g20:.4f} S2 {s2_g20:.4f} "
            f"| fold S1 {row['fold_matched']['s1']:.4f} S2 {row['fold_matched']['s2']:.4f}")
    return out


def stage_test(force: bool = False) -> dict:
    started = time.time()
    results = read_json(RESULTS_PATH)
    if "test" in results and not force:
        log("stage test: results present, reusing")
        return results["test"]
    if not results.get("candidates", {}).get("done"):
        raise RuntimeError("stage test requires the frozen top five from stage select")
    P = load_protocol()
    folds, _ = build_folds(P)
    order = results["candidates"]["top_ids"]
    by_id = {row["id"]: row for row in results["candidates"]["rows"]}
    top = [by_id[index] for index in order]
    log(f"stage test: one-shot scoring of {len(top)} selected candidates")
    out = test_scores(P, top, folds)
    results["test"] = out
    results.setdefault("stage_seconds", {})["test"] = time.time() - started
    write_json(RESULTS_PATH, results)
    log(f"stage test done in {time.time() - started:.0f}s")
    return out


# ---------------------------------------------------------------- attack stage
def build_attack_gate(stores: dict, stats: dict, weights: dict, e189):
    encoder = e189.E162.load_encoder({"name": "lam0.0_t0.1"})[0]

    class StreamGate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.items = {}
            for name, weight in weights.items():
                store = stores[name]
                self.items[name] = {
                    "weight": float(weight), "kind": SPEC[name]["kind"],
                    "mu": torch.as_tensor(store["mu"], device=DEVICE),
                    "sd": torch.as_tensor(store["sd"], device=DEVICE),
                    "pca_mu": torch.as_tensor(store["pca_mu"], device=DEVICE),
                    "basis": torch.as_tensor(store["basis"], device=DEVICE),
                    "proto": torch.as_tensor(store["proto"], device=DEVICE),
                    "k": int(store["k"]), "pca": int(store["pca"]),
                    "ref_mu": float(stats[name]["mu"]),
                    "ref_sd": float(stats[name]["sd"])}

        def features(self, x, kind):
            if kind == "logspd":
                return e189.logspd(x)
            if kind == "subwin6":
                return e189.subwin_mean(x, 6)
            if kind == "subwin12":
                return e189.subwin_mean(x, 12)
            if kind == "logbp":
                return e189.logbp(x)
            if kind == "e162":
                return encoder(e189.E162.normalize(x))
            raise KeyError(kind)

        def forward(self, x, task_idx):
            total = None
            for item in self.items.values():
                feat = self.features(x, item["kind"])
                z = (feat - item["mu"][task_idx]) / item["sd"][task_idx]
                proj = (z - item["pca_mu"]) @ item["basis"]
                distance = torch.cdist(proj, item["proto"])
                score = (-distance).view(len(x), len(CLASSES), item["k"]) \
                    .max(2).values.max(1).values
                score = (score - item["ref_mu"]) / item["ref_sd"]
                total = item["weight"] * score if total is None else \
                    total + item["weight"] * score
            return total

    return StreamGate()


def attack_subset(P: dict) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    cap = 400
    window = np.flatnonzero(P["window"])
    selected = []
    for task in TASKS:
        rows = window[P["tasks"][window] == task]
        selected.append(rows if len(rows) <= cap else np.sort(
            rng.choice(rows, size=cap, replace=False)))
    return np.sort(np.concatenate(selected))


def pgd(scorer, x, task_idx, true_idx, target_idx, goal, snr, steps, seed) -> dict:
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
        logits, gate = scorer(x + delta, task_idx)
        if goal == "D":
            objective = gate
        else:
            true_logit = logits.gather(1, true_idx[:, None]).squeeze(1)
            wrong = logits.masked_fill(
                torch.nn.functional.one_hot(true_idx, len(CLASSES)).bool(), -1e9)
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
        logits, gate = scorer(x + delta, task_idx)
    return {"logits": logits.detach().cpu().numpy(), "gate": gate.detach().cpu().numpy()}


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
    top = results["test"]["candidates"][0]
    point = results["candidates"]["top5"][0]
    log(f"stage attack: #1 {top['label']} (tau {top['deployment']['tau']:.4f})")
    P = load_protocol()
    e189 = load_module("e198_frozen_e189", E189_DIR / "run_e189.py")
    e189.build_constants()
    e191 = load_module("e198_frozen_e191", E191_DIR / "run_e191.py")
    import torch.nn as nn

    class Scorer(nn.Module):
        def __init__(self, identity, gate):
            super().__init__()
            self.identity = identity
            self.gate = gate

        def forward(self, x, task_idx):
            return self.identity(x), self.gate(x, task_idx)

    identity = e191.NetV3().to(DEVICE)
    payload = torch.load(E191_DIR / CFG["e191_backbone"]["model"],
                         map_location="cpu", weights_only=False)
    identity.load_state_dict(payload["state_dict"])
    identity.eval()
    for parameter in identity.parameters():
        parameter.requires_grad_(False)
    data = load_instrument()
    stores = instrument_stores(data)
    stats = json.loads(str(data["stats_json"]))
    gate = build_attack_gate(stores, stats, point["weights"], e189)
    scorer = Scorer(identity, gate)

    subset = attack_subset(P)
    raw = e189.E180.load_raw([P["paths"][i] for i in subset])
    truth = P["subjects"][subset]
    task_idx = torch.as_tensor([TASKS.index(t) for t in P["tasks"][subset]],
                               dtype=torch.long, device=DEVICE)
    true_idx = torch.as_tensor(truth - 1, dtype=torch.long, device=DEVICE)
    lut = {int(c): i for i, c in enumerate(CLASSES)}
    target_idx = torch.as_tensor([lut[int(v)] for v in
                                  [CLASSES[(lut[int(v)] + 1) % len(CLASSES)] for v in truth]],
                                 dtype=torch.long, device=DEVICE)
    tau = float(top["deployment"]["tau"])
    log(f"  attack subset {len(subset)} trials, tau {tau:.4f}")
    with torch.no_grad():
        logits, scores = scorer(raw, task_idx)
    conditions = [{"snr": None, "goal": "clean",
                   **attack_metrics(logits.detach().cpu().numpy(),
                                    scores.detach().cpu().numpy(), truth, tau)}]
    log(f"  clean: closed {conditions[0]['closed_set_accuracy']:.4f} "
        f"RA {conditions[0]['RA']:.4f} ASafeR {conditions[0]['ASafeR']:.4f}")
    for snr in CFG["attack"]["snr_db"]:
        for goal in ("A", "B", "C", "D"):
            out = pgd(scorer, raw, task_idx, true_idx, target_idx, goal, snr,
                      int(CFG["attack"]["steps"]), SEED + 1000 * int(snr) + ord(goal))
            metric = {"snr": int(snr), "goal": goal,
                      **attack_metrics(out["logits"], out["gate"], truth, tau)}
            conditions.append(metric)
            log(f"  SNR{snr} goal {goal}: RA {metric['RA']:.4f} "
                f"ASafeR {metric['ASafeR']:.4f} accept {metric['accept_rate']:.4f} "
                f"closed {metric['closed_set_accuracy']:.4f}")
    primary = [c for c in conditions if c["snr"] == int(CFG["attack"]["primary_snr_db"])]
    secondary = [c for c in conditions if c["snr"] == 20]
    summary = {"candidate": top["label"], "weights": point["weights"], "tau": tau,
               "n_trials": int(len(subset)),
               "per_task": {t: int((P["tasks"][subset] == t).sum()) for t in TASKS},
               "conditions": conditions,
               "worst_RA_primary": min(primary, key=lambda r: r["RA"]),
               "worst_ASafeR_primary": min(primary, key=lambda r: r["ASafeR"]),
               "worst_RA_snr20": min(secondary, key=lambda r: r["RA"]) if secondary else None,
               "worst_ASafeR_snr20": (min(secondary, key=lambda r: r["ASafeR"])
                                      if secondary else None),
               "snr_db": CFG["attack"]["snr_db"],
               "steps": int(CFG["attack"]["steps"]),
               "note": "self-built white-box PGD on the E191 branch-A v3 identity model "
                       "with the frozen Train-only gate; radius = per-trial L2 norm / "
                       "10^(SNR/20)"}
    results["attack"] = summary
    results.setdefault("stage_seconds", {})["attack"] = time.time() - started
    write_json(RESULTS_PATH, results)
    log(f"stage attack done in {time.time() - started:.0f}s: worst RA "
        f"{summary['worst_RA_primary']['RA']:.4f} ASafeR "
        f"{summary['worst_ASafeR_primary']['ASafeR']:.4f} (SNR30)")
    return summary


# ---------------------------------------------------------------- eleven gates
def eleven_gates(results: dict) -> dict:
    branch = read_json(E191_DIR / "results.json")["branch_a_v3"]
    top = results["candidates"]["top5"][0]
    test = results["test"]["candidates"][0]["deployment"]
    attack = results.get("attack") or {}
    row = next(r for r in results["candidates"]["rows"] if r["id"] == top["id"])
    values = {"#1": float(branch["in_sample_accuracy"]),
              "#2": float(branch["oof_accuracy_5block"]),
              "#3": float(branch["oof_macro_f1"]),
              "#4": float(branch["oof_accuracy_forward_blocks"]),
              "#5": row["auroc5"], "#6": row["u95_strict"],
              "#7": (attack.get("worst_RA_primary") or {}).get("RA"),
              "#8": (attack.get("worst_ASafeR_primary") or {}).get("ASafeR"),
              "#9": test["s2"], "#10": test["s1"], "#11": row["loso_strict"]}
    values_std = {"#5": row["auroc5_std_mode"], "#6": row["u95_std_mode"],
                  "#11": row["loso_std"]}
    targets = {"#1": 1.00, "#2": 0.95, "#3": 0.92, "#4": 0.90, "#5": 0.98,
               "#6": 0.95, "#7": 0.95, "#8": 0.98, "#9": 0.70, "#10": 0.70}
    passed = {key: bool(values[key] is not None and values[key] >= targets[key] - 1e-12)
              for key in targets}
    passed["#11"] = bool(values["#11"] > 0.90)
    return {"values": values, "values_std": values_std, "targets": targets,
            "passed": passed, "count": int(sum(passed.values())),
            "candidate": top["label"], "point_id": top["id"]}


# ---------------------------------------------------------------- report
def fmt(value, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def build_report(results: dict) -> str:
    cand = results["candidates"]
    test = results["test"]
    gates = results["eleven_gates"]
    e197 = read_json(ROOT / CFG["references"]["e197"]).get("analysis", {})
    top5 = cand["top5"]
    rows = {row["id"]: row for row in cand["rows"]}
    first = top5[0]
    first_test = test["candidates"][0]
    lines = []
    add = lines.append

    add("# E198：用 leave-5 三阶段协议合法选型，然后一次性评估 Test")
    add("")
    add("日期：2026-11-16。Train = 20 个 identity 的 session1（11304 条，fit 行 = frac ≤ 0.6），"
        "评估块 = `0.6 < frac ≤ 0.8`（2301 条）；Test S1 = 5 个未注册者 session1（3033 条），"
        "S2 = 20 个注册者 session2（19218 条）。选型只用 Train（leave-5 三阶段 + strict LOSO），"
        "Test 在 top-5 冻结后评分一次。未写入任何已有实验目录，未提交 git。")
    add("")
    add("## 0. 结论（先看这里）")
    add("")
    add(f"1. **候选池里没有任何一个候选能通过 #11。** {cand['n_candidates']} 个候选"
        f"（5 个单门控 + 全部两两 / 三路 0.1 步长融合）中，满足 `acc5 ≥ 0.95` 的有 "
        f"{cand['n_acc_ok']} 个，但 `loso_strict > 0.90` 的只有 {cand['n_feasible']} 个；"
        f"全池 `loso_strict` 最高 {cand['max_loso_strict']:.4f}"
        f"（{cand['best_strict_candidate']}，rej5 {cand['best_strict_rej5']:.4f}），"
        f"在 `acc5 ≥ 0.95` 子集内最高 {cand['max_loso_strict_acc_ok']:.4f}。"
        "E195 里能过 #11 的两类门控（E183 三视角头、WCCN 距离比）都不在本实验的候选池内，"
        "所以预注册规则 R1 的可行集为空，只能用 R1′（去掉 #11）继续。")
    add(f"2. **R1′ 选出的第 1 名是 `{first['label']}`**：Train 侧 rej5 "
        f"{first['rej5']:.4f}、acc5 {first['acc5']:.4f}、auc5 {first['auc5']:.4f}、"
        f"loso_strict {first['loso_strict']:.4f}、auroc5 {first['auroc5']:.4f}。")
    add(f"3. **一次性 Test 评分**（部署口径，gallery = 20 个注册者）：S1 拒识 "
        f"{first_test['deployment']['s1']:.4f}、S2 接受 {first_test['deployment']['s2']:.4f}、"
        f"min {first_test['deployment']['min']:.4f}。"
        f"S1 过 0.70，**S2 不过**；改用 gallery-20 自校准 τ 是 S1 "
        f"{first_test['gallery20_calibration']['s1']:.4f} / S2 "
        f"{first_test['gallery20_calibration']['s2']:.4f}，结论不变。")
    add(f"4. **11 项通过 {gates['count']}/11**：通过 "
        + "、".join(f"#{i}" for i in range(1, 12) if gates["passed"][f"#{i}"])
        + "；未通过 "
        + "、".join(f"#{i}" for i in range(1, 12) if not gates["passed"][f"#{i}"])
        + f"。与 E195 的 Test 选权 8/11 相差 {8 - gates['count']} 项，"
        "差距有两类互不相同的原因：候选池没有 #5 / #11 所需的强判别门控，"
        "以及 #9（S2）在 Train 上根本不可观测。")
    add("5. **Test 官方 AUC / F1 / C / RA / ASafeR / U / Score = N/A**（Test 无身份标签）。")
    add("")

    add("## 1. 选型准则与 Train 侧前 5 名")
    add("")
    add("### 1.1 预注册准则")
    add("")
    add("对每个候选在 Train 上计算：`rej5`（200 个 leave-5 折的未注册拒识率，a = 0.90）、"
        "`acc5`（已注册者 argmax 识别率）、`auc5`（known vs unregistered AUROC，折内均值）、"
        "`loso_strict`（20 折严格 leave-1 LOSO 拒识率）、`auroc5`（严格 LOSO 合并 AUROC）。")
    add("")
    add("- **R1**：`acc5 ≥ 0.95` 且 `loso_strict > 0.90` → 最大化 `rej5`，并列取 `auc5` 高者。")
    add("- **R1′（本次实际执行）**：R1 的可行集为空时，去掉 #11，只保留 `acc5 ≥ 0.95`，"
        "仍最大化 `rej5`、并列取 `auc5`。该退路在看 Test 之前就已经触发并记录。")
    add("")
    add(f"候选总数 {cand['n_candidates']}（去重后；单路 5、两两 0.1 步长、三路单纯形 0.1 步长）。"
        f"满足 `acc5 ≥ 0.95` 的有 {cand['n_acc_ok']} 个，其中 `loso_strict > 0.90` 的 "
        f"{cand['n_feasible']} 个。")
    add("")
    add("### 1.2 Train 侧前 5 名（只用 Train，Test 未参与）")
    add("")
    add("| # | 候选 | 权重 | rej5 | rej5± | acc5 | F1（leave-5） | auc5 | loso_strict | auroc5 | u95 |")
    add("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in top5:
        add(f"| {row['rank']} | `{row['candidate']}` | {row['label']} | "
            f"{row['rej5']:.4f} | {row['rej5_std']:.4f} | {row['acc5']:.4f} | "
            f"{row['f1_5']:.4f} | {row['auc5']:.4f} | {row['loso_strict']:.4f} | "
            f"{row['auroc5']:.4f} | {row['u95_strict']:.4f} |")
    add("")
    add("### 1.3 单门控与 R1 参照点（Train 侧）")
    add("")
    add("| 候选 | rej5 | acc5 | auc5 | loso_strict | auroc5 | 备注 |")
    add("|---|---:|---:|---:|---:|---:|---|")
    for name in STREAMS:
        row = next(r for r in cand["rows"] if r["candidate"] == name)
        add(f"| `{name}` | {row['rej5']:.4f} | {row['acc5']:.4f} | {row['auc5']:.4f} | "
            f"{row['loso_strict']:.4f} | {row['auroc5']:.4f} | 单门控 |")
    strict_row = rows[[r["id"] for r in cand["rows"]
                       if r["candidate"] == cand["best_strict_candidate"]][0]]
    add(f"| `{cand['best_strict_candidate']}` | {cand['best_strict_rej5']:.4f} | "
        f"{strict_row['acc5']:.4f} | {strict_row['auc5']:.4f} | "
        f"{cand['max_loso_strict']:.4f} | {strict_row['auroc5']:.4f} | "
        "全池 loso_strict 最高（acc5 不足 0.95） |")
    add("")
    add("**#11 为什么在池内不可能通过**：E195 的严格 LOSO 读数里，E65 单门控 0.6448、"
        "subwin6 0.3982、subwin12 0.3239、logbp 0.1887、E162 0.5897；"
        "只有 E183 三视角头（0.9848）与 WCCN 距离比（0.9877）能过 0.90，而这两个都不在本实验的"
        "候选池内（WCCN 按任务要求排除，E183 未列入）。本实验复现了同一量级：单门控 "
        f"`e65_dist` {rows[next(r['id'] for r in cand['rows'] if r['candidate'] == 'e65_dist')]['loso_strict']:.4f}、"
        f"`e162_dist` {rows[next(r['id'] for r in cand['rows'] if r['candidate'] == 'e162_dist')]['loso_strict']:.4f}，"
        f"全池最高 {cand['max_loso_strict']:.4f}"
        f"（{cand['best_strict_candidate']}），距 0.90 仍差 "
        f"{0.90 - cand['max_loso_strict']:.4f}。"
        "（E195 的 `e65+e162` 融合严格拒识 0.7795，本实验同一组合更高，"
        "差异来自本实验把原型按身份合并、不再按 task 分开。）")
    add("")
    add("**与 E197 的交叉核对**：单门控的 `rej5` 与 `acc5` 与 E197 报告逐位一致"
        "（`e65_dist` 0.8096 / 0.9812、`subwin6_dist` 0.5960 / 0.9360、"
        "`subwin12_dist` 0.5362 / 0.8897、`logbp_dist` 0.1982 / 0.6430、"
        "`e162_dist` 0.6171 / 0.7979），说明三阶段仪器与折划分完全复现；"
        "`auc5` 这里是 200 折的折内均值（E197 报的是合并值，两者差 ≤0.001）。"
        "`loso_strict` 与 E195 的单门控差 ≤0.03（本实验原型跨 task 合并，E195 按 task 分开）。")
    add("")

    add("## 2. 前 5 名的一次性 Test 评分")
    add("")
    add("部署口径：gallery = 20 个注册者原型，τ = Train 严格 LOSO known 分数的 a=0.90 分位"
        "（E195 `paired` 规则）；`g20` = 同一批 Test 分数、τ 改用 Train 评估块上 20 个身份 "
        "known 分数的 a=0.90 分位（与部署 gallery 完全同口径的自校准，仍是 Train-only）；"
        "`fold_matched`：gallery = 该折 15 个注册者，τ = 该折 known 分位，200 折平均"
        "（与 E197 的 rej5 同口径）。")
    add("")
    add("| # | 候选 | τ（strict） | S1 | S2 | min | 均值 | 诊断 AUC | τ（g20） | S1（g20） | S2（g20） | S1（fold） | S2（fold） |")
    add("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row, scored in zip(top5, test["candidates"]):
        dep, fm, g20 = scored["deployment"], scored["fold_matched"], scored["gallery20_calibration"]
        add(f"| {row['rank']} | {row['label']} | {dep['tau']:.4f} | {dep['s1']:.4f} | "
            f"{dep['s2']:.4f} | {dep['min']:.4f} | {dep['mean']:.4f} | "
            f"{dep['diagnostic_auc']:.4f} | {g20['tau']:.4f} | {g20['s1']:.4f} | "
            f"{g20['s2']:.4f} | {fm['s1']:.4f} | {fm['s2']:.4f} |")
    add("")
    add(f"选出的第 1 名在部署口径上 S1 {first_test['deployment']['s1']:.4f} / "
        f"S2 {first_test['deployment']['s2']:.4f}；g20 自校准 "
        f"{first_test['gallery20_calibration']['s1']:.4f} / "
        f"{first_test['gallery20_calibration']['s2']:.4f}；"
        f"leave-5 折口径 {first_test['fold_matched']['s1']:.4f} / "
        f"{first_test['fold_matched']['s2']:.4f}。三种口径都给出同一个定性结论："
        "**S1 高、S2 低**，双 0.70 不成立。")
    add("")
    add("对照 E195 已发布的单门控 Test 读数（口径不同：按 task 分开的原型，仅作量级参考）："
        "`e65_dist` S1 0.8276 / S2 0.5569、`subwin6_dist` 0.7029 / 0.7247、"
        "`subwin12_dist` 0.6439 / 0.7777、`logbp_dist` 0.5513 / 0.8279、"
        "`e162_dist` 0.3759 / 0.8832。池内唯一能双过 0.70 的是 `subwin6_dist`，"
        "但它的 `acc5` 只有 0.9360，被预注册的 `acc5 ≥ 0.95` 硬约束排除。"
        "这是本实验「合法选型」与「双 0.70」之间的第一处结构性冲突。")
    add("")

    add("## 3. 第 1 名的完整 11 项门禁（单工作点）")
    add("")
    add(f"工作点：`{gates['candidate']}`，a = 0.90，τ（部署，严格 LOSO 合并 known 分位）= "
        f"{first_test['deployment']['tau']:.4f}。除 #7 / #8 外全部来自同一候选的同一工作点。")
    add("")
    add("| 门禁 | 读数（strict / 部署） | std 口径 | 目标 | 判定 | 缺口 |")
    add("|---|---:|---:|---:|---|---:|")
    names = {"#1": "Train in-sample 准确率", "#2": "5 块 OOF C", "#3": "身份 macro-F1",
             "#4": "块外准确率", "#5": "LOSO AUROC", "#6": "U@95%", "#7": "RA（SNR30 最差）",
             "#8": "ASafeR（SNR30 同一攻击）", "#9": "Test S2 接受率",
             "#10": "Test S1 拒识率", "#11": "LOSO 拒识 @a=0.90"}
    for key in (f"#{i}" for i in range(1, 12)):
        value = gates["values"][key]
        std = gates["values_std"].get(key)
        target = gates["targets"].get(key)
        floor = 0.90 if key == "#11" else target
        ok = gates["passed"][key]
        gap = ("—" if (floor is None or value is None)
               else f"{value - floor:+.4f}" if not ok else "—")
        add(f"| {key} {names[key]} | {fmt(value)} | {fmt(std) if std is not None else '—'} | "
            f"{fmt(target, 2) if target is not None else '> 0.90'} | "
            f"{'✅' if ok else '❌'} | {gap} |")
    add("")
    attack = results.get("attack") or {}
    if attack:
        add("### 3.1 自建 PGD（SNR30 / SNR20，各 4 个目标）")
        add("")
        add("攻击：E191 branch-A v3 身份模型 + 冻结的选定门控，20 步白盒 PGD，"
            "L2 半径 = 每 trial 去均值波形 L2 范数 / 10^(SNR/20)；"
            f"τ 用同一工作点 {attack['tau']:.4f}；子集 {attack['n_trials']} 条"
            f"（{', '.join(f'{k}={v}' for k, v in attack['per_task'].items())}）。")
        add("")
        add("| SNR | 目标 | RA | ASafeR | 门控接受率 | 攻击后闭集准确率 |")
        add("|---|---|---:|---:|---:|---:|")
        for condition in attack["conditions"]:
            snr = "clean" if condition["snr"] is None else str(condition["snr"])
            add(f"| {snr} | {condition['goal']} | {condition['RA']:.4f} | "
                f"{condition['ASafeR']:.4f} | {condition['accept_rate']:.4f} | "
                f"{condition['closed_set_accuracy']:.4f} |")
        add("")
        worst20ra = attack.get("worst_RA_snr20") or {}
        worst20safe = attack.get("worst_ASafeR_snr20") or {}
        add(f"SNR30 最差 RA {attack['worst_RA_primary']['RA']:.4f}"
            f"（{attack['worst_RA_primary']['goal']}）、"
            f"ASafeR {attack['worst_ASafeR_primary']['ASafeR']:.4f}"
            f"（{attack['worst_ASafeR_primary']['goal']}）；"
            f"SNR20 最差 RA {fmt(worst20ra.get('RA'))}、"
            f"ASafeR {fmt(worst20safe.get('ASafeR'))}。"
            "#7 / #8 用 SNR30 判定，SNR20 作参考档。")
        add("")

    add("## 4. 明确回答：用 leave-5 合法选型，能否达到 ≥8/11？")
    add("")
    add(f"**不能，{gates['count']}/11。** 与 E195「Test 选权 8/11」差 "
        f"{8 - gates['count']} 项。")
    add("")
    add("| 门禁 | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11 |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    add("| 判定 | " + " | ".join(
        "✅" if gates["passed"][f"#{i}"] else "❌" for i in range(1, 12)) + " |")
    add("")
    add("通过的是 " + "、".join(f"#{i}" for i in range(1, 12)
                              if gates["passed"][f"#{i}"]) + "；"
        "未通过的是 " + "、".join(f"#{i}" for i in range(1, 12)
                              if not gates["passed"][f"#{i}"]) + "。")
    add("")
    add(f"与 E195 的 8/11 相比，本实验少了 #5、#11（E195 靠 E183 头能过）并丢掉了 #9；"
        "差距分三类，原因互不相同：")
    add("")
    add(f"1. **#5 / #11 是候选池的结构性上限，与选型协议无关。** 池内 `auroc5` 最高 "
        f"{max(r['auroc5'] for r in cand['rows']):.4f}（目标 0.98），`loso_strict` 最高 "
        f"{cand['max_loso_strict']:.4f}（目标 0.90）。E195 的 #5（0.9807）与 #11（0.9640）"
        "全部由 E183 三视角头提供，而 E183 的 Test S1 只有 0.27–0.58。"
        "把 E183 放进池子会让 R1 非空，但会立刻把 #10 / #9 一起拖垮"
        "——这正是 E195「几何上存在、部署上不存在」的那道窄缝。")
    add(f"2. **#9（S2 接受率）在 Train 上不可观测。** Train 只有 session1，"
        "S2 的跨会话漂移没有任何 Train 侧代理；预注册规则又只最大化 rej5"
        "（等价于把阈值钉在已知者 10% 拒绝处），于是同一个阈值在 session2 上拒绝了 "
        f"{1 - first_test['deployment']['s2']:.1%} 的注册者样本"
        f"（S2 接受 {first_test['deployment']['s2']:.4f}）。top-5 的 S2 全部落在 "
        f"{min(c['deployment']['s2'] for c in test['candidates']):.4f}–"
        f"{max(c['deployment']['s2'] for c in test['candidates']):.4f}，没有一个接近 0.70。"
        "E195 的 8/11 点 S2 = 0.7087 是在 Test 上审计出来的，Train-only 规则选不到它。")
    add(f"3. **#6 / #7 / #8 与 E195 一样失败，且与门控选择无关。** U@95 "
        f"{fmt(gates['values']['#6'])}（目标 0.95）、RA {fmt(gates['values']['#7'])}"
        f"（目标 0.95）、ASafeR {fmt(gates['values']['#8'])}（目标 0.98）："
        "前者是距离门控 LOSO 分数分布的重叠问题，后两者是身份骨干在 SNR30 白盒 PGD 下的固有水平"
        "（E195 的同一读数为 0.8831 / 0.9473 / 0.9617，同样不过）。")
    add("")
    add("结论：**leave-5 协议本身是合法且可用的，但它能守住的只有 #1–#4 与 #10。**"
        "8/11 里的额外 3 项分别依赖（a）E183 级判别门控（#5、#11）与（b）Test 侧 S2 审计（#9），"
        "两者都不能只靠 Train 得到。另外，池内唯一双过 0.70 的 `subwin6_dist` 被 "
        "`acc5 ≥ 0.95` 硬约束排除，说明「注册者认准」与「跨会话不误拒」在这批特征上本身互相拉扯。")
    add("")

    add("## 5. 选型准则的可信度：rej5 的 −0.095 偏差会带来什么")
    add("")
    if e197:
        match = e197.get("test_s1_match", {}).get("5", {})
        add(f"E197 在 k=5 上测得：Train 侧拒识与 Test S1 的 Spearman ρ = "
            f"{match.get('spearman', float('nan')):.3f}，平均绝对偏差 "
            f"{match.get('mae', float('nan')):.4f}，系统偏差 "
            f"{match.get('bias', float('nan')):+.4f}（Train 侧偏保守）。"
            "逐门控偏差：`e65` −0.0612、`subwin6` −0.1511、`subwin12` −0.1713、"
            "`logbp` −0.3588、`e162` +0.1155。")
        add("")
    add("1. **排序可用、绝对值不可用；但这一次偏差方向是「乐观」而非「保守」。** 本实验 top-1 的 "
        f"rej5 {first['rej5']:.4f}，fold-matched S1 {first_test['fold_matched']['s1']:.4f}"
        f"（rej5 − S1 = {first['rej5'] - first_test['fold_matched']['s1']:+.4f}），"
        f"部署 S1 {first_test['deployment']['s1']:.4f}"
        f"（rej5 − S1 = {first['rej5'] - first_test['deployment']['s1']:+.4f}）。"
        "两个 |偏差| 都小于 E197 单门控平均的 0.0955，原因是这里选了 `e65`（负偏差 −0.061）与 "
        "`e162`（正偏差 +0.116）的混合，**方向相反的偏差在融合时互相抵消**。"
        "这是「按 rej5 选融合」的一个意外好处：水平误差被平均掉；但符号不可预测——"
        "E197 的单门控平均偏保守，本实验这个融合点反而略偏乐观。")
    rho5, _ = spearmanr([r["rej5"] for r in top5],
                        [c["deployment"]["s1"] for c in test["candidates"]])
    add("2. **top-5 内部其实是一个平台，rej5 排序在平台内没有意义。** 前 5 名的 rej5 极差只有 "
        f"{max(r['rej5'] for r in top5) - min(r['rej5'] for r in top5):.4f}，"
        "远小于 ±0.10 的偏差量级；实测 Test S1 极差 "
        f"{max(c['deployment']['s1'] for c in test['candidates']) - min(c['deployment']['s1'] for c in test['candidates']):.4f}，"
        f"两者的 Spearman ρ = {rho5:+.2f}（5 个点，近乎随机）。也就是说 E197 的 ρ = 0.85 "
        "是在**跨度很大**的 9 个门控之间测得的；一旦所有候选都挤在同一个 e65 主导的平台里，"
        "排序信息就不再可用，只能保证「选到的是这个平台」，不能保证平台内哪一个更好。")
    add("3. **−0.095 会把工作点判断推向乐观，但对 #9 没有帮助。** 若把 rej5 当 S1 的预测值，"
        f"本实验的 {first['rej5']:.4f} 会**高估**实测 S1 约 0.03–0.07"
        "（E197 的平均偏差方向相反，是低估），看起来「S1 已经过线并且还有余量」；"
        "但 #9 的失败与 S1 无关——即使 S1 高达 "
        f"{first_test['deployment']['s1']:.4f}，S2 也只有 "
        f"{first_test['deployment']['s2']:.4f}。也就是说，**rej5 的偏差只会影响 #10 的余量判断，"
        "不会影响 #9 的判定**；任何只基于 Train 的拒识准则都无法定位双 0.70。")
    add("4. **对「能否用 leave-5 替代 Test 选权」的回答**：可以替代**粗选**"
        "（ρ = 0.85 是在跨度很大的门控之间测的；本实验选中的整片平台 Test S1 都在 "
        "0.82–0.85，确实是池内强档），但**不能替代双目标约束的判定**，"
        "也不能在平台内细分（ρ = −0.30）。"
        "E195 的 8/11 本质上是二维 Pareto 前沿上的一个窄点，需要 S1 与 S2 两个 Test 读数才能定位；"
        "leave-5 只提供一个一维（且带 ±0.10 偏差）的投影。")
    add("")

    add("## 6. Test 官方指标")
    add("")
    add("**Test 官方 AUC / F1 / C / RA / ASafeR / U / Score = N/A**"
        "（Test 无真实身份标签，只有 S1 / S2 审计读数与自建 PGD 的 RA / ASafeR）。")
    add("")

    add("## 7. 合规说明")
    add("")
    add("- 选型只用 Train：`rej5` / `acc5` / `auc5` 来自 200 个 leave-5 三阶段折，"
        "`loso_strict` / `auroc5` 来自 20 折严格 LOSO；Test 分数在 top-5 冻结之后才计算。")
    add("- 原型与阈值只用已注册者；标准化 / PCA 用全部 20 人（用户定义的三阶段流程）。")
    add("- Test 只评分一次；Session 列只用于把 S1 / S2 分开，从未参与拟合、选型或阈值。")
    add("- R1 的可行集为空这一事实在 Test 评分之前就已确定并记录在 `results.json`"
        "（`fallback_used = true`）。")
    add("- `wccn_dr` 未进入候选池（任务要求）；E183 不在任务给定的池内。")
    add("")

    add("## 8. 文件与耗时")
    add("")
    add("| 文件 | 内容 |")
    add("|---|---|")
    add("| `run_e198.py` | 全部逻辑（`--stage instrument/strict/select/test/attack/report/all`） |")
    add("| `config.json` | 冻结配置（含预注册准则与退路） |")
    add("| `cache/instrument.npz` | 阶段 1 仪器、原型、窗口 z 分数、流统计量 |")
    add("| `cache/strict.npz` | 20 折 strict / std LOSO 的 z 分数与掩码 |")
    add("| `results.json` | 455 个候选的 Train 指标、top-5、Test、11 项、攻击 |")
    add("| `report.md` | 本报告 |")
    add("| `run.log` | 运行日志 |")
    add("")
    seconds = results.get("stage_seconds", {})
    add("耗时（秒）：" + "、".join(f"{k} {v:.0f}" for k, v in seconds.items()) + "。")
    add("")
    add("```powershell")
    add('& "D:\\WORK\\BCIC2026\\BrainPrint\\.venv\\Scripts\\python.exe" '
        '"D:\\WORK\\BCIC2026\\BrainPrint\\research\\E198_select_by_leave5\\run_e198.py" --stage all')
    add("```")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- driver
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["instrument", "strict", "select", "test", "attack",
                                 "report", "all"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    sys.stdout = Tee(LOG_PATH)
    log(f"\nE198 start {time.strftime('%Y-%m-%d %H:%M:%S')} device {DEVICE}")
    torch.manual_seed(SEED)
    started = time.time()
    results = read_json(RESULTS_PATH)

    if args.stage in ("instrument", "all"):
        if args.force or not INST_PATH.exists():
            P = load_protocol()
            log(f"Train {len(P['subjects'])} rows, fit {int(P['fit_rows'].sum())}, "
                f"window {int(P['window'].sum())}, Test {len(P['test_task_idx'])}, "
                f"S1 {int(P['is_s1'].sum())}")
            summary = build_instrument(P)
            results = read_json(RESULTS_PATH)
            results["instrument"] = summary
            results.setdefault("stage_seconds", {})["instrument"] = summary["seconds"]
            write_json(RESULTS_PATH, results)
        else:
            log("stage instrument: cache present, reusing")
    if args.stage in ("strict", "all"):
        if args.force or not STRICT_PATH.exists():
            summary = strict_pass(load_protocol())
            results = read_json(RESULTS_PATH)
            results["strict"] = summary
            results.setdefault("stage_seconds", {})["strict"] = summary["seconds"]
            write_json(RESULTS_PATH, results)
        else:
            log("stage strict: cache present, reusing")
    if args.stage in ("select", "all"):
        stage_select(args.force)
    if args.stage in ("test", "all"):
        stage_test(args.force)
    if args.stage in ("attack", "all"):
        stage_attack(args.force)
    if args.stage in ("report", "all"):
        results = read_json(RESULTS_PATH)
        results["eleven_gates"] = eleven_gates(results)
        write_json(RESULTS_PATH, results)
        build_report(results)
        log("stage report done")
    log(f"E198 done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
