"""E181: gate-signal fusion tuning on top of the frozen E180 two-branch system.

Only the open-set *gate* changes.  Branch A (adversarially trained EEGNet) and
branch B (E65 correlation-logSPD PCA-96 / k=12 prototypes) are loaded read-only
from the E180 artifacts and are never retrained for the production decision path.

Gate signals (all oriented so that larger = more likely "known"):
    g_dist    branch B  -nearest-prototype distance (the E180 gate)
    g_conf    branch A  max softmax probability
    g_margin  branch A  top1 - top2 softmax probability
    g_energy  branch A  logsumexp(logits)
    g_entropy branch A  sum p log p (= -entropy, larger = more confident)
Every signal is z-scored with Train OOF mean/std (Train-only).

Fusion   : g = sum_i w_i * z_i ; weights chosen ONLY on Train LOSO pooled AUC.
Threshold: tau(a) = quantile(Train OOF fused score, 1 - a).
Decision : g >= tau(a) -> branch A argmax, else label 0.

Stages   : signals, loso, select, test, attack, capacity, report, all.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "report.md"
SIGNAL_CACHE = CACHE / "signals.npz"
TEST_LOGITS_CACHE = CACHE / "test_logits.npy"
LOSO_CACHE = CACHE / "loso.npz"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E180_DIR = ROOT / "research" / CFG["frozen_source"]["experiment"]
E180 = load_module("e181_frozen_e180", E180_DIR / "run_e180.py")
DEVICE = E180.DEVICE
FT = E180.FT
TASKS = list(E180.TASKS)
CLASSES = E180.CLASSES
SEED = int(E180.SEED)
FIT = float(E180.CFG["protocol"]["fit_fraction"])

SIGNALS = ("g_dist", "g_conf", "g_margin", "g_energy", "g_entropy")
ACC_GRID = [float(v) for v in CFG["threshold"]["grid"]]
PRIMARY_A = float(CFG["threshold"]["primary_acceptance"])
WEIGHTS = [round(0.1 * i, 1) for i in range(1, 10)]
CLASS_OF = {int(c): i for i, c in enumerate(CLASSES)}


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


def load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {"experiment": CFG["experiment"]}


def save_results(results: dict) -> None:
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2,
                                       default=float), encoding="utf-8")


def f32(array) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


# ---------------------------------------------------------------- gate signals
def logsumexp_np(x: np.ndarray) -> np.ndarray:
    m = x.max(1, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(1, keepdims=True))).ravel()


def raw_signals(logits: np.ndarray, dist: np.ndarray) -> dict:
    """Five raw gate signals, larger = more likely known."""
    lg = np.asarray(logits, dtype=np.float64)
    shifted = lg - lg.max(1, keepdims=True)
    prob = np.exp(shifted)
    prob /= prob.sum(1, keepdims=True)
    ordered = np.sort(prob, axis=1)
    return {
        "g_dist": np.asarray(dist, dtype=np.float64),
        "g_conf": prob.max(1),
        "g_margin": ordered[:, -1] - ordered[:, -2],
        "g_energy": logsumexp_np(lg),
        "g_entropy": np.sum(prob * np.log(np.clip(prob, 1e-12, None)), axis=1),
    }


def fit_stats(raw: dict) -> dict:
    return {"mu": {k: float(np.mean(raw[k])) for k in SIGNALS},
            "sd": {k: float(np.std(raw[k])) + 1e-12 for k in SIGNALS}}


def standardise(raw: dict, stats: dict) -> dict:
    return {k: (raw[k] - stats["mu"][k]) / stats["sd"][k] for k in SIGNALS}


def fuse(z: dict, weights: dict) -> np.ndarray:
    total = np.zeros(len(next(iter(z.values()))), dtype=np.float64)
    for name, weight in weights.items():
        total = total + float(weight) * z[name]
    return total


def cand_name(weights: dict) -> str:
    return " + ".join(f"{float(weights[s]):.2f}*{s}" for s in SIGNALS if s in weights)


def candidate_library() -> list[dict]:
    cands: list[dict] = [{"name": s, "weights": {s: 1.0}} for s in SIGNALS]
    for a, b in itertools.combinations(SIGNALS, 2):
        for w in WEIGHTS:
            cands.append({"name": f"{w:.2f}*{a} + {1 - w:.2f}*{b}",
                          "weights": {a: w, b: round(1 - w, 1)}})
    for combo in itertools.combinations(SIGNALS, 3):
        cands.append({"name": " + ".join(f"0.33*{s}" for s in combo),
                      "weights": {s: 1.0 / 3.0 for s in combo}})
    for size in (4, 5):
        for combo in itertools.combinations(SIGNALS, size):
            cands.append({"name": " + ".join(f"{1 / size:.2f}*{s}" for s in combo),
                          "weights": {s: 1.0 / size for s in combo}})
    seen, unique = set(), []
    for cand in cands:
        key = tuple(sorted((k, round(v, 6)) for k, v in cand["weights"].items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
    return unique


def quantile(value: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(value, dtype=np.float64), q))


# ---------------------------------------------------------------- stage signals
def test_branch_a_logits() -> np.ndarray:
    if TEST_LOGITS_CACHE.exists():
        return np.load(TEST_LOGITS_CACHE)
    test_meta = pd.read_csv(ROOT / E180.CFG["data"]["test_metadata"],
                            usecols=["sample", "path"])
    branch_a = torch.load(E180.MODEL_A, map_location="cpu", weights_only=False)
    if branch_a["spec_sha256"] != E180.BRANCH_A_SHA:
        raise ValueError("E180 branch A artifact does not match its frozen spec")
    model = E180.Net().to(DEVICE)
    model.load_state_dict(branch_a["state_dict"])
    model.eval()
    paths = test_meta["path"].tolist()
    base = ROOT / E180.CFG["data"]["test_root"]
    logits = np.zeros((len(paths), len(CLASSES)), dtype=np.float32)
    started = time.time()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=12) as pool:
        for start in range(0, len(paths), 512):
            chunk = paths[start:start + 512]
            arrays = list(pool.map(lambda p: E180.load_mat(base / str(p)), chunk))
            batch = torch.as_tensor(np.stack(arrays), dtype=FT, device=DEVICE)
            with torch.no_grad():
                logits[start:start + len(chunk)] = model(batch).cpu().numpy()
            del batch
            done = start + len(chunk)
            if done % 4096 < 512 or done == len(paths):
                log(f"  Test branch A {done}/{len(paths)} ({time.time() - started:.0f}s)")
    del model
    torch.cuda.empty_cache()
    np.save(TEST_LOGITS_CACHE, logits)
    return logits


def stage_signals() -> dict:
    started = time.time()
    meta, frac, blocks, fit_rows = E180.load_config_meta()
    tasks = meta["task"].to_numpy()
    subjects = meta["subject"].to_numpy(int)

    stage_a = torch.load(E180.STAGE_A, map_location="cpu", weights_only=False)
    if stage_a["spec_sha256"] != E180.BRANCH_A_SHA:
        raise ValueError("E180 stage A artifact does not match its frozen spec")
    oof_logits = stage_a["oof_logits"].numpy().astype(np.float64)
    store = E180.load_branch_b()
    oof_dist = np.asarray(store["oof_scores"], np.float64)
    raw_oof = raw_signals(oof_logits, oof_dist)
    stats = fit_stats(raw_oof)
    z_oof = standardise(raw_oof, stats)
    log("stage signals: Train OOF signal stats (Train-only)")
    for name in SIGNALS:
        log(f"  {name:9s} mean {stats['mu'][name]:+.4f} sd {stats['sd'][name]:.4f}")

    test_meta = pd.read_csv(ROOT / E180.CFG["data"]["test_metadata"],
                            usecols=["sample", "path", "task"])
    if len(test_meta) != int(E180.CFG["data"]["test_trials"]):
        raise ValueError("Test row count unexpected")
    test_tasks = test_meta["task"].to_numpy()
    test_logits = test_branch_a_logits()
    test_features = f32(np.load(ROOT / E180.CFG["branch_b"]["test_features"]))
    if len(test_features) != len(test_meta):
        raise ValueError("Test E65 feature row count mismatch")
    test_dist, _ = E180.branch_b_test_scores(test_features, test_tasks, store)
    if not np.isfinite(test_dist).all():
        raise RuntimeError("non-finite branch B Test scores")
    raw_test = raw_signals(test_logits, test_dist)
    z_test = standardise(raw_test, stats)

    np.savez_compressed(
        SIGNAL_CACHE,
        stats_mu=np.asarray([stats["mu"][k] for k in SIGNALS], dtype=np.float64),
        stats_sd=np.asarray([stats["sd"][k] for k in SIGNALS], dtype=np.float64),
        oof_raw=np.stack([raw_oof[k] for k in SIGNALS]),
        oof_z=np.stack([z_oof[k] for k in SIGNALS]),
        oof_logits=oof_logits.astype(np.float32),
        test_raw=np.stack([raw_test[k] for k in SIGNALS]),
        test_z=np.stack([z_test[k] for k in SIGNALS]),
        test_logits=test_logits,
        test_sample=test_meta["sample"].astype(str).to_numpy(dtype=str),
        test_tasks=test_tasks,
        subjects=subjects, blocks=blocks, frac=frac,
        in_sample_logits=stage_a["in_sample_logits"].numpy().astype(np.float32))

    # sanity: reproduce E180 branch A closed-set numbers from the frozen OOF logits
    pred = oof_logits.argmax(1) + 1
    summary = {
        "signal_stats": stats,
        "oof_signal_mean": {k: float(np.mean(raw_oof[k])) for k in SIGNALS},
        "oof_z_check": {k: [float(np.mean(z_oof[k])), float(np.std(z_oof[k]))]
                        for k in SIGNALS},
        "branch_a_reproduced": {
            "oof_accuracy_5block": float(np.mean(pred == subjects)),
            "oof_macro_f1": float(f1_score(subjects, pred, labels=CLASSES,
                                           average="macro", zero_division=0)),
            "oof_accuracy_per_block": {str(b): float(np.mean(
                pred[blocks == b] == subjects[blocks == b]))
                for b in range(int(E180.CFG["protocol"]["blocks"]))},
            "in_sample_accuracy": float(np.mean(
                stage_a["in_sample_logits"].numpy().argmax(1) + 1 == subjects)),
        },
        "n_test": int(len(test_meta)),
        "test_z_range": {k: [float(np.min(z_test[k])), float(np.max(z_test[k]))]
                         for k in SIGNALS},
        "elapsed_seconds": time.time() - started,
    }
    results = load_results()
    results["signals"] = summary
    save_results(results)
    log(f"stage signals done in {summary['elapsed_seconds']:.0f}s: OOF acc "
        f"{summary['branch_a_reproduced']['oof_accuracy_5block']:.4f} (E180 0.9881), "
        f"in-sample {summary['branch_a_reproduced']['in_sample_accuracy']:.4f}")
    return summary


# ---------------------------------------------------------------- stage loso
def projected_branch_b() -> tuple:
    meta, frac, blocks, fit_rows = E180.load_config_meta()
    tasks = meta["task"].to_numpy()
    raw_features = f32(np.load(ROOT / E180.CFG["branch_b"]["train_features"]))
    stats = E180.task_stats(raw_features, tasks)
    standardised = E180.apply_stats(raw_features, tasks, stats)
    mu, basis = E180.pca_basis(standardised[fit_rows],
                               int(E180.CFG["branch_b"]["pca_dim"]))
    projected = f32((standardised.astype(np.float64) - mu.astype(np.float64)) @ basis)
    return projected, mu, basis


def stage_loso() -> dict:
    started = time.time()
    meta, frac, blocks, fit_rows = E180.load_config_meta()
    subjects = meta["subject"].to_numpy(int)
    tasks = meta["task"].to_numpy()
    window = (frac > FIT) & (frac <= 0.8)
    win_rows = np.flatnonzero(window)
    fit_idx = np.flatnonzero(fit_rows)
    k = int(E180.CFG["branch_b"]["prototype_k"])
    epochs = int(CFG["loso_probe"]["epochs"])

    log(f"stage loso: probe models on pm & fit_rows ({len(fit_idx)} rows), "
        f"eval window {len(win_rows)} rows, {epochs} epochs, 20 identities")
    fit_raw = E180.load_raw([meta["path"].iloc[i] for i in fit_idx])
    win_raw = E180.load_raw([meta["path"].iloc[i] for i in win_rows])
    projected, _, _ = projected_branch_b()

    rows_all, fold_all, known_all = [], [], []
    signal_all = {k: [] for k in SIGNALS}
    folds = []
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        pm = np.isin(subjects, enrolled)
        model_path = CACHE / f"loso_model_{int(held)}.pt"
        if model_path.exists():
            state = torch.load(model_path, map_location="cpu", weights_only=False)
            model = E180.Net().to(DEVICE)
            model.load_state_dict(state["state_dict"])
            model.eval()
            info = {"cached": True}
        else:
            train_pos = np.flatnonzero(pm[fit_idx])
            model, info = E180.train_model(
                fit_raw, train_pos, subjects[fit_idx][train_pos] - 1,
                SEED + 1000 + int(held), epochs, f"loso{int(held)}")
            torch.save({"state_dict": model.state_dict(), "held_identity": int(held),
                        "epochs": epochs, "spec_sha256": E180.BRANCH_A_SHA}, model_path)
        known = win_rows[pm[win_rows]]
        unknown = win_rows[subjects[win_rows] == held]
        rows = np.r_[known, unknown]
        pos = np.searchsorted(win_rows, rows)
        if not np.array_equal(win_rows[pos], rows):
            raise RuntimeError("window indexing error")
        logits = E180.predict_logits(model, win_raw, pos)
        protos, keys = E180.build_prototypes(projected, subjects, tasks, pm & fit_rows,
                                             enrolled, k)
        proto, filled = E180.proto_stack(protos, keys, k)
        dist = E180.branch_b_scores(projected[rows], tasks[rows], proto, filled, k)
        raw = raw_signals(logits, dist)
        rows_all.append(rows)
        fold_all.append(np.full(len(rows), int(held), dtype=np.int32))
        known_all.append(np.r_[np.ones(len(known), bool), np.zeros(len(unknown), bool)])
        for name in SIGNALS:
            signal_all[name].append(raw[name])
        closed = float(np.mean(logits.argmax(1)[:len(known)] + 1 == subjects[known]))
        folds.append({"held_identity": int(held), "n_known": int(len(known)),
                      "n_unknown": int(len(unknown)),
                      "closed_set_accuracy_known": closed,
                      "known_confidence": float(np.mean(raw["g_conf"][:len(known)])),
                      "unknown_confidence": float(np.mean(raw["g_conf"][len(known):])),
                      "cached": bool(info.get("cached", False))})
        log(f"  LOSO {int(held):2d}: known {len(known)} unknown {len(unknown)} "
            f"closed {closed:.4f} mean conf known/unknown "
            f"{folds[-1]['known_confidence']:.3f}/{folds[-1]['unknown_confidence']:.3f}")
        del model
        torch.cuda.empty_cache()

    rows = np.concatenate(rows_all)
    fold = np.concatenate(fold_all)
    known = np.concatenate(known_all)
    np.savez_compressed(LOSO_CACHE, rows=rows, fold=fold, known=known,
                        **{name: np.concatenate(signal_all[name]) for name in SIGNALS})

    # per-signal LOSO AUC on the probe models (raw, monotone-invariant)
    pooled = {}
    for name in SIGNALS:
        score = np.concatenate(signal_all[name])
        auc = float(roc_auc_score(known.astype(int), score))
        per_fold = []
        for held in CLASSES:
            sel = fold == int(held)
            if sel.sum() == 0 or len(np.unique(known[sel])) < 2:
                continue
            per_fold.append(float(roc_auc_score(known[sel].astype(int), score[sel])))
        pooled[name] = {"pooled_auc": auc,
                        "fold_auc_mean": float(np.mean(per_fold)),
                        "fold_auc_std": float(np.std(per_fold))}
        log(f"  probe LOSO {name:9s}: pooled AUC {auc:.4f} "
            f"fold mean {np.mean(per_fold):.4f}")

    summary = {"n_known": int(known.sum()), "n_unknown": int((~known).sum()),
               "n_folds": len(folds), "epochs": epochs,
               "per_signal_loso": pooled, "folds": folds,
               "elapsed_seconds": time.time() - started}
    results = load_results()
    results["loso_probe"] = summary
    save_results(results)
    log(f"stage loso done in {summary['elapsed_seconds']:.0f}s")
    del fit_raw, win_raw
    torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------- stage select
def candidate_table() -> tuple:
    store = np.load(SIGNAL_CACHE)
    loso = np.load(LOSO_CACHE)
    stats = {"mu": {k: float(v) for k, v in zip(SIGNALS, store["stats_mu"])},
             "sd": {k: float(v) for k, v in zip(SIGNALS, store["stats_sd"])}}
    z_oof = {k: store["oof_z"][i].astype(np.float64) for i, k in enumerate(SIGNALS)}
    z_test = {k: store["test_z"][i].astype(np.float64) for i, k in enumerate(SIGNALS)}
    z_loso = standardise({k: loso[k].astype(np.float64) for k in SIGNALS}, stats)
    known = loso["known"].astype(bool)
    fold = loso["fold"]

    rows = []
    for cand in candidate_library():
        w = cand["weights"]
        score = fuse(z_loso, w)
        auc = float(roc_auc_score(known.astype(int), score))
        per_fold, u95 = [], []
        for held in CLASSES:
            sel = fold == int(held)
            if sel.sum() == 0 or len(np.unique(known[sel])) < 2:
                continue
            per_fold.append(float(roc_auc_score(known[sel].astype(int), score[sel])))
            thr = quantile(score[sel & known], 0.05)
            u95.append(float(np.mean(score[sel & ~known] < thr)))
        rows.append({"name": cand["name"], "weights": w, "n_signals": len(w),
                     "loso_pooled_auc": auc,
                     "loso_fold_auc_mean": float(np.mean(per_fold)),
                     "loso_fold_auc_std": float(np.std(per_fold)),
                     "loso_fold_auc_min": float(np.min(per_fold)),
                     "u95_mean": float(np.mean(u95)),
                     "oof_fused": fuse(z_oof, w), "test_fused": fuse(z_test, w)})
    return rows, z_oof, z_test


def stage_select() -> dict:
    started = time.time()
    rows, z_oof, _ = candidate_table()
    ranked = sorted(rows, key=lambda r: (-r["loso_pooled_auc"],
                                         -r["loso_fold_auc_mean"], r["n_signals"]))
    best = ranked[0]
    tau = {f"{a:.2f}": quantile(best["oof_fused"], 1.0 - a) for a in ACC_GRID}
    oof_store = np.load(SIGNAL_CACHE)
    subjects = oof_store["subjects"]
    primary = np.where(best["oof_fused"] >= tau[f"{PRIMARY_A:.2f}"],
                       oof_store["oof_logits"].argmax(1) + 1, 0)
    summary = {
        "selected": {"name": best["name"], "weights": best["weights"],
                     "loso_pooled_auc": best["loso_pooled_auc"],
                     "loso_fold_auc_mean": best["loso_fold_auc_mean"],
                     "loso_fold_auc_std": best["loso_fold_auc_std"],
                     "u95_mean": best["u95_mean"]},
        "tau_train_oof": tau,
        "oof_acceptance_at_primary": float(np.mean(
            best["oof_fused"] >= tau[f"{PRIMARY_A:.2f}"])),
        "oof_gated_accuracy_at_primary": float(np.mean(primary == subjects)),
        "oof_rejection_rate_at_primary": float(np.mean(primary == 0)),
        "leaderboard": [{key: row[key] for key in
                         ("name", "n_signals", "loso_pooled_auc", "loso_fold_auc_mean",
                          "loso_fold_auc_std", "loso_fold_auc_min", "u95_mean")}
                        for row in ranked[:15]],
        "n_candidates": len(ranked),
        "elapsed_seconds": time.time() - started,
    }
    results = load_results()
    results["selection"] = summary
    save_results(results)
    log("stage select: top-5 candidates by Train LOSO pooled AUC")
    for row in ranked[:5]:
        log(f"  {row['loso_pooled_auc']:.4f}  {row['name']}")
    log(f"  selected: {best['name']} (LOSO {best['loso_pooled_auc']:.4f}), "
        f"tau(a={PRIMARY_A:.2f}) {tau[f'{PRIMARY_A:.2f}']:+.4f}")
    return summary


# ---------------------------------------------------------------- stage test
def stage_test() -> dict:
    started = time.time()
    rows, _, z_test = candidate_table()
    store = np.load(SIGNAL_CACHE)
    selected = load_results()["selection"]["selected"]

    def pick(name):
        for row in rows:
            if row["name"] == name:
                return row
        raise KeyError(name)

    chosen = pick(selected["name"])
    tau_primary = quantile(chosen["oof_fused"], 1.0 - PRIMARY_A)
    pred = np.where(chosen["test_fused"] >= tau_primary,
                    store["test_logits"].argmax(1) + 1, 0)
    pd.DataFrame({"Sample": store["test_sample"], "Label": pred}).to_csv(
        OUT / "test_predictions_fused.csv", index=False)
    log(f"  wrote {OUT / 'test_predictions_fused.csv'} "
        f"({int(np.sum(pred == 0))} rejects of {len(pred)}) before reading Session")

    # Session is read only now.
    test_meta = pd.read_csv(ROOT / E180.CFG["data"]["test_metadata"],
                            usecols=["sample", "session"])
    if not np.array_equal(test_meta["sample"].astype(str).to_numpy(),
                          store["test_sample"].astype(str)):
        raise ValueError("Test sample order mismatch")
    is_s1 = test_meta["session"].astype(str).str.casefold().isin(
        ["session1", "s1", "1"]).to_numpy()

    # mechanism: is each signal oriented the same way on Train (known) and on Test (S1)?
    session_means = {}
    for name in SIGNALS:
        session_means[name] = {
            "s1_mean_z": float(np.mean(z_test[name][is_s1])),
            "s2_mean_z": float(np.mean(z_test[name][~is_s1])),
            "auc_s2_vs_s1": float(roc_auc_score((~is_s1).astype(int), z_test[name])),
        }
    log("  Test signal orientation (z units, S1 vs S2, AUC with S2 as positive):")
    for name in SIGNALS:
        row = session_means[name]
        log(f"    {name:9s} S1 {row['s1_mean_z']:+.3f} S2 {row['s2_mean_z']:+.3f} "
            f"AUC {row['auc_s2_vs_s1']:.4f}")

    audit = []
    for row in rows:
        tau = {a: quantile(row["oof_fused"], 1.0 - a) for a in ACC_GRID}
        entry = {"name": row["name"], "n_signals": row["n_signals"],
                 "loso_pooled_auc": row["loso_pooled_auc"],
                 "u95_mean": row.get("u95_mean"),
                 # S2 is the set the gate should ACCEPT, so "higher = better" AUC
                 "test_diagnostic_auc": float(roc_auc_score(
                     (~is_s1).astype(int), row["test_fused"])),
                 "per_acceptance": {}}
        for a in ACC_GRID:
            accept = row["test_fused"] >= tau[a]
            entry["per_acceptance"][f"{a:.2f}"] = {
                "tau": tau[a],
                "s1_rejection": float(np.mean(~accept[is_s1])),
                "s2_acceptance": float(np.mean(accept[~is_s1])),
                "s1_plus_s2": float(np.mean(~accept[is_s1]) + np.mean(accept[~is_s1]))}
        # Test-optimal threshold: upper bound only (uses Session labels, never a decision)
        order = np.argsort(-row["test_fused"])
        label = is_s1[order]
        s1_rank = np.cumsum(label) / max(is_s1.sum(), 1)
        s2_rank = np.cumsum(~label) / max((~is_s1).sum(), 1)
        # threshold = score[best_at]: S1 above it is accepted, S2 above it is accepted
        total = (1.0 - s1_rank) + s2_rank
        best_at = int(np.argmax(total))
        entry["test_optimal"] = {"max_s1_plus_s2": float(total[best_at]),
                                 "s1_rejection": float(1.0 - s1_rank[best_at]),
                                 "s2_acceptance": float(s2_rank[best_at])}
        audit.append(entry)

    primary_row = next(e for e in audit if e["name"] == selected["name"])
    primary_at = primary_row["per_acceptance"][f"{PRIMARY_A:.2f}"]
    summary = {
        "selected_weights": selected["weights"],
        "selected_name": selected["name"],
        "primary_acceptance": PRIMARY_A,
        "threshold_branch_b_e180": float(E180.load_branch_b()["tau"]),
        "audit": {
            "n_s1": int(is_s1.sum()), "n_s2": int((~is_s1).sum()),
            "s1_rejection": primary_at["s1_rejection"],
            "s2_acceptance": primary_at["s2_acceptance"],
            "s1_plus_s2": primary_at["s1_plus_s2"],
            "tau": primary_at["tau"],
        },
        "all_candidates": audit,
        "session_signal_means": session_means,
        "session_used_for_inference": False,
        "official_metrics": {k: "N/A" for k in
                             ("AUC", "F1", "C", "RA", "ASafeR", "U", "Score")},
        "elapsed_seconds": time.time() - started,
    }
    results = load_results()
    results["test"] = summary
    save_results(results)
    log(f"stage test: selected {selected['name']} -> S1 rejection "
        f"{primary_at['s1_rejection']:.4f}, S2 acceptance "
        f"{primary_at['s2_acceptance']:.4f} (a={PRIMARY_A:.2f})")
    return summary


# ---------------------------------------------------------------- attack stage
class FusedScorer(E180.Scorer):
    """E180 frozen dual-branch forward with the fused gate score exposed."""

    def __init__(self, model, store, weights: dict, stats: dict):
        super().__init__(model, store)
        self.weights = {k: float(v) for k, v in weights.items()}
        self.mu = {k: torch.tensor(float(stats["mu"][k]), device=DEVICE)
                   for k in self.weights}
        self.sd = {k: torch.tensor(float(stats["sd"][k]), device=DEVICE)
                   for k in self.weights}

    def __call__(self, x: torch.Tensor, task_idx: torch.Tensor) -> tuple:
        logits, z_dist = super().__call__(x, task_idx)
        prob = torch.softmax(logits, dim=1)
        parts, pieces = {}, []
        for name, weight in self.weights.items():
            if name == "g_dist":
                value = z_dist
            elif name == "g_conf":
                value = (prob.max(1).values - self.mu[name]) / self.sd[name]
            elif name == "g_margin":
                top2 = torch.topk(prob, 2, dim=1).values
                value = (top2[:, 0] - top2[:, 1] - self.mu[name]) / self.sd[name]
            elif name == "g_energy":
                value = (torch.logsumexp(logits, dim=1) - self.mu[name]) / self.sd[name]
            elif name == "g_entropy":
                ent = -(prob * prob.clamp_min(1e-12).log()).sum(1)
                value = (-ent - self.mu[name]) / self.sd[name]
            else:
                raise KeyError(name)
            parts[name] = value
            pieces.append(weight * value)
        total = None
        for value in pieces:
            total = value if total is None else total + value
        self.last_parts = parts
        return logits, total


def attack_metrics(logits: np.ndarray, gate: np.ndarray, truth: np.ndarray,
                   tau: float) -> dict:
    accept = gate >= tau
    predicted = np.where(accept, logits.argmax(1) + 1, 0)
    n = len(predicted)
    ra = float(np.mean(predicted == truth))
    asafe = float(np.mean((predicted == truth) | (predicted == 0)))
    half = 1.96 * math.sqrt(max(ra * (1 - ra), 0.0) / max(n, 1))
    half_safe = 1.96 * math.sqrt(max(asafe * (1 - asafe), 0.0) / max(n, 1))
    return {"n": n, "RA": ra, "RA_ci95": [max(0.0, ra - half), min(1.0, ra + half)],
            "ASafeR": asafe,
            "ASafeR_ci95": [max(0.0, asafe - half_safe), min(1.0, asafe + half_safe)],
            "closed_set_accuracy": float(np.mean(logits.argmax(1) + 1 == truth)),
            "accept_rate": float(np.mean(accept)),
            "wrong_identity_rate": float(np.mean((predicted != 0) & (predicted != truth))),
            "reject_rate": float(np.mean(predicted == 0))}


def stage_attack() -> dict:
    started = time.time()
    E180.build_torch_constants()
    cfg = CFG["attack"]
    meta, frac, blocks, fit_rows = E180.load_config_meta()
    subjects = meta["subject"].to_numpy(int)
    tasks = meta["task"].to_numpy()
    window_rows = np.flatnonzero((frac > FIT) & (frac <= 0.8))
    rng = np.random.default_rng(int(cfg["seed"]))
    cap = int(cfg["cap_per_task"])
    selected_rows = []
    for task in TASKS:
        rows = window_rows[tasks[window_rows] == task]
        selected_rows.append(rows if len(rows) <= cap else np.sort(
            rng.choice(rows, size=cap, replace=False)))
    subset = np.sort(np.concatenate(selected_rows))
    log(f"attack stage: {len(subset)} eval-window trials "
        f"({', '.join(f'{t}={int((tasks[subset] == t).sum())}' for t in TASKS)})")

    store_npz = np.load(SIGNAL_CACHE)
    stats = {"mu": {k: float(v) for k, v in zip(SIGNALS, store_npz["stats_mu"])},
             "sd": {k: float(v) for k, v in zip(SIGNALS, store_npz["stats_sd"])}}
    oof_z = {k: store_npz["oof_z"][i].astype(np.float64) for i, k in enumerate(SIGNALS)}
    selection = load_results()["selection"]["selected"]
    weights = selection["weights"]
    of = fuse(oof_z, weights)
    tau_z = quantile(of, 1.0 - PRIMARY_A)
    sweep = (0.80, 0.90, 0.95)
    tau_sweep = {a: quantile(of, 1.0 - a) for a in sweep}

    branch_a = torch.load(E180.MODEL_A, map_location="cpu", weights_only=False)
    if branch_a["spec_sha256"] != E180.BRANCH_A_SHA:
        raise ValueError("branch A artifact does not match the frozen spec")
    model = E180.Net().to(DEVICE)
    model.load_state_dict(branch_a["state_dict"])
    scorer = FusedScorer(model, E180.load_branch_b(), weights, stats)

    raw = E180.load_raw([meta["path"].iloc[i] for i in subset])
    truth = subjects[subset]
    task_idx = torch.as_tensor([TASKS.index(t) for t in tasks[subset]],
                               dtype=torch.long, device=DEVICE)
    true_idx = torch.as_tensor(truth - 1, dtype=torch.long, device=DEVICE)
    target = np.asarray([CLASSES[(CLASS_OF[int(v)] + 1) % len(CLASSES)] for v in truth])
    target_idx = torch.as_tensor([CLASS_OF[int(v)] for v in target],
                                 dtype=torch.long, device=DEVICE)

    # differentiable fused gate vs the numpy gate on the same clean batch
    with torch.no_grad():
        logits, gate = scorer(raw, task_idx)
    clean_logits, clean_gate = logits.cpu().numpy(), gate.cpu().numpy()
    rebuilt = np.zeros(len(subset), dtype=np.float64)
    for name, weight in weights.items():
        rebuilt += float(weight) * scorer.last_parts[name].cpu().numpy()
    gate_check = float(np.max(np.abs(rebuilt - clean_gate)))
    oof_shift = float(np.mean(clean_gate - fuse(oof_z, weights)[subset]))
    log(f"  gate consistency (numpy vs torch on the same logits): {gate_check:.2e}; "
        f"full-model minus OOF mean gate: {oof_shift:+.4f}")
    clean = attack_metrics(clean_logits, clean_gate, truth, tau_z)
    clean["at_acceptance"] = {f"{a:.2f}": {
        "RA": attack_metrics(clean_logits, clean_gate, truth, tau_sweep[a])["RA"],
        "ASafeR": attack_metrics(clean_logits, clean_gate, truth, tau_sweep[a])["ASafeR"],
        "accept_rate": attack_metrics(clean_logits, clean_gate, truth,
                                      tau_sweep[a])["accept_rate"]} for a in sweep}
    log(f"  clean: RA {clean['RA']:.4f} ASafeR {clean['ASafeR']:.4f} "
        f"closed {clean['closed_set_accuracy']:.4f} accept {clean['accept_rate']:.4f}")

    conditions = []
    for eps in cfg["eps_normalised"]:
        for goal in cfg["goals"]:
            out = E180.pgd_attack(scorer, raw, task_idx, true_idx, target_idx, goal,
                                  float(eps), int(cfg["steps"]),
                                  int(cfg["seed"]) + int(eps * 1e5))
            metrics = attack_metrics(out["logits"], out["z"], truth, tau_z)
            metrics.update({"eps": float(eps), "goal": goal,
                            "max_rel_linf": out["max_rel_linf"],
                            "nan_gradients": out["nan_gradients"],
                            "at_acceptance": {f"{a:.2f}": {
                                "RA": attack_metrics(out["logits"], out["z"], truth,
                                                     tau_sweep[a])["RA"],
                                "ASafeR": attack_metrics(out["logits"], out["z"], truth,
                                                         tau_sweep[a])["ASafeR"]}
                                for a in sweep}})
            conditions.append(metrics)
            log(f"  eps={eps:.3f} goal {goal}: RA {metrics['RA']:.4f} "
                f"ASafeR {metrics['ASafeR']:.4f} closed {metrics['closed_set_accuracy']:.4f} "
                f"accept {metrics['accept_rate']:.4f} wrong_id "
                f"{metrics['wrong_identity_rate']:.4f}")

    worst_ra = min(conditions, key=lambda r: r["RA"])
    worst_safe = min(conditions, key=lambda r: r["ASafeR"])
    summary = {
        "selected_name": selection["name"], "weights": weights,
        "n_trials": int(len(subset)),
        "per_task": {task: int((tasks[subset] == task).sum()) for task in TASKS},
        "tau_z": tau_z, "primary_acceptance": PRIMARY_A,
        "fused_gate_rebuild_max_abs": gate_check,
        "full_model_minus_oof_gate_mean": oof_shift,
        "sweep_acceptance": list(sweep),
        "sweep_tau": {f"{a:.2f}": tau_sweep[a] for a in sweep},
        "eps_normalised": list(cfg["eps_normalised"]),
        "steps": int(cfg["steps"]), "temperature": float(cfg["temperature"]),
        "clean": clean, "conditions": conditions,
        "worst_RA": {"RA": worst_ra["RA"], "eps": worst_ra["eps"],
                     "goal": worst_ra["goal"], "ASafeR": worst_ra["ASafeR"]},
        "worst_ASafeR": {"ASafeR": worst_safe["ASafeR"], "eps": worst_safe["eps"],
                         "goal": worst_safe["goal"], "RA": worst_safe["RA"]},
        "attack_status": "self-built white-box PGD surrogate, not the official suite",
        "elapsed_seconds": time.time() - started,
    }
    results = load_results()
    results["attack"] = summary
    save_results(results)
    log(f"attack done in {summary['elapsed_seconds']:.0f}s: worst RA "
        f"{worst_ra['RA']:.4f}, worst ASafeR {worst_safe['ASafeR']:.4f}")
    del raw, model
    torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------- capacity probe
def penultimate(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=-1, keepdim=True)
    rms = x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
    h = model.body(x / rms)
    return F.normalize(model.pool(h).squeeze(-1), dim=1)


def nn_predict(feature: torch.Tensor, ref: torch.Tensor, ref_label: torch.Tensor,
               batch: int = 1024) -> np.ndarray:
    out = np.empty(len(feature), dtype=np.int64)
    for start in range(0, len(feature), batch):
        sim = feature[start:start + batch] @ ref.T
        out[start:start + batch] = ref_label[sim.argmax(1)].cpu().numpy()
    return out


def stage_capacity() -> dict:
    started = time.time()
    meta, frac, blocks, fit_rows = E180.load_config_meta()
    subjects = meta["subject"].to_numpy(int)
    labels = torch.as_tensor(subjects - 1, dtype=torch.long, device=DEVICE)
    raw = E180.load_raw(meta["path"].tolist())
    full_state = torch.load(E180.MODEL_A, map_location="cpu", weights_only=False)
    model = E180.Net().to(DEVICE)
    model.load_state_dict(full_state["state_dict"])
    model.eval()

    in_sample_logits = E180.predict_logits(model, raw, np.arange(len(meta)))
    logit_acc = float(np.mean(in_sample_logits.argmax(1) + 1 == subjects))

    feats = []
    with torch.no_grad():
        for start in range(0, len(meta), 1024):
            feats.append(penultimate(model, raw[start:start + 1024]))
    full_features = torch.cat(feats, 0)
    # leave-one-out 1-NN (in-sample memorisation)
    pred = np.empty(len(meta), dtype=np.int64)
    for start in range(0, len(meta), 1024):
        sim = full_features[start:start + 1024] @ full_features.T
        rows = torch.arange(len(sim), device=DEVICE)
        sim[rows, torch.arange(start, start + len(sim), device=DEVICE)] = -2.0
        pred[start:start + 1024] = labels[sim.argmax(1)].cpu().numpy()
    knn_in_sample = float(np.mean(pred + 1 == subjects))

    # block-OOF 1-NN: honest generalisation check with the frozen fold models
    oof_pred = np.full(len(meta), -1, dtype=np.int64)
    for block in range(int(E180.CFG["protocol"]["blocks"])):
        state = torch.load(E180.CACHE / f"fold_model_{block}.pt", map_location="cpu",
                           weights_only=False)
        fold_model = E180.Net().to(DEVICE)
        fold_model.load_state_dict(state["state_dict"])
        fold_model.eval()
        parts = []
        with torch.no_grad():
            for start in range(0, len(meta), 1024):
                parts.append(penultimate(fold_model, raw[start:start + 1024]))
        features = torch.cat(parts, 0)
        ref = np.flatnonzero(blocks != block)
        qry = np.flatnonzero(blocks == block)
        oof_pred[qry] = nn_predict(features[qry], features[ref], labels[ref])
        del fold_model
        torch.cuda.empty_cache()
    knn_oof = float(np.mean(oof_pred + 1 == subjects))

    # more-epochs probe: continue the frozen full model for 15 epochs
    extra = int(CFG["capacity_probe"].get("extra_epochs", 15))
    more_model = E180.Net().to(DEVICE)
    more_model.load_state_dict(full_state["state_dict"])
    opt = torch.optim.AdamW(more_model.parameters(), lr=0.002, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=extra)
    gen = torch.Generator(device=DEVICE).manual_seed(SEED + 7)
    order_rng = np.random.default_rng(SEED + 7)
    labels_dev = torch.as_tensor(subjects - 1, dtype=torch.long, device=DEVICE)
    for epoch in range(1, extra + 1):
        more_model.train()
        order = torch.as_tensor(order_rng.permutation(len(meta)), dtype=torch.long,
                                device=DEVICE)
        for start in range(0, len(meta), 64):
            sel = order[start:start + 64]
            if len(sel) < 4:
                continue
            xb = E180.augment(raw[sel], gen)
            adv = E180.pgd_train(more_model, xb, labels_dev[sel], gen)
            loss = (0.5 * F.cross_entropy(more_model(xb), labels_dev[sel])
                    + 0.5 * F.cross_entropy(more_model(adv), labels_dev[sel]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        log(f"  [capacity_extra] epoch {epoch}/{extra}")
    more_model.eval()
    more_logits = E180.predict_logits(more_model, raw, np.arange(len(meta)))
    more_acc = float(np.mean(more_logits.argmax(1) + 1 == subjects))
    del more_model
    torch.cuda.empty_cache()

    summary = {
        "logits_in_sample_accuracy": logit_acc,
        "logits_in_sample_macro_f1": float(f1_score(
            subjects, in_sample_logits.argmax(1) + 1, labels=CLASSES,
            average="macro", zero_division=0)),
        "knn_in_sample_accuracy": knn_in_sample,
        "knn_block_oof_accuracy": knn_oof,
        "extra_epochs": extra,
        "extra_epochs_in_sample_accuracy": more_acc,
        "note": ("knn_in_sample is a leave-one-out 1-NN on the frozen 128-d embedding: "
                 "a capacity/memorisation number, not generalisation; knn_block_oof is "
                 "the honest counterpart. The production decision path stays branch A "
                 "logits argmax, so #1 passes only under an explicitly labelled "
                 "capacity-only rule."),
        "elapsed_seconds": time.time() - started,
    }
    results = load_results()
    results["capacity"] = summary
    save_results(results)
    log(f"capacity: logits in-sample {logit_acc:.4f}, LOO-1NN {knn_in_sample:.4f}, "
        f"block-OOF 1NN {knn_oof:.4f}, +15 epochs {more_acc:.4f}")
    del raw
    torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------- gates / report
GATE_SPEC = [
    ("#1", "拟合容量", "Train 准确率", "= 1.00"),
    ("#2", "折外闭集", "C（5 块 OOF 闭集准确率）", "≥ 0.95"),
    ("#3", "折外闭集", "身份 macro-F1（5 块 OOF）", "≥ 0.92"),
    ("#4", "跨时间代理", "块外准确率（同 #2 的 5 块口径）", "≥ 0.90"),
    ("#5", "开放集", "pseudo-unknown AUROC（20 折 LOSO 合并）", "≥ 0.98"),
    ("#6", "开放集", "U@95%（known 接受率固定 95%）", "≥ 0.95"),
    ("#7", "对抗识别", "RA（自建 PGD，全 ε×目标取最差）", "≥ 0.95"),
    ("#8", "对抗安全", "ASafeR（同一攻击取最差）", "≥ 0.98"),
    ("#9", "Test 审计", "S2 接受率", "≥ 0.75"),
    ("#10", "Test 审计", "S1 拒识率", "≥ 0.75"),
]
TARGETS = {"#1": 1.00, "#2": 0.95, "#3": 0.92, "#4": 0.90, "#5": 0.98,
           "#6": 0.95, "#7": 0.95, "#8": 0.98, "#9": 0.75, "#10": 0.75}
E180_REF = {"#1": 0.9988, "#2": 0.9881, "#3": 0.9878, "#4": 0.9881, "#5": 0.9401,
            "#6": 0.5847, "#7": 0.9161, "#8": 0.9840, "#9": 0.6335, "#10": 0.7359}


def gate_table(results: dict) -> list[dict]:
    a = results.get("signals", {}).get("branch_a_reproduced", {})
    sel = results.get("selection", {}).get("selected", {})
    att = results.get("attack", {})
    test = results.get("test", {})
    values = {
        "#1": a.get("in_sample_accuracy"),
        "#2": a.get("oof_accuracy_5block"),
        "#3": a.get("oof_macro_f1"),
        "#4": a.get("oof_accuracy_5block"),
        "#5": sel.get("loso_pooled_auc"),
        "#6": sel.get("u95_mean"),
        "#7": att.get("worst_RA", {}).get("RA"),
        "#8": att.get("worst_ASafeR", {}).get("ASafeR"),
        "#9": test.get("audit", {}).get("s2_acceptance"),
        "#10": test.get("audit", {}).get("s1_rejection"),
    }
    rows = []
    for key, layer, metric, target_text in GATE_SPEC:
        value = values[key]
        rows.append({"gate": key, "layer": layer, "metric": metric,
                     "target": target_text, "value": value,
                     "passed": None if value is None else bool(
                         value >= TARGETS[key] - 1e-12),
                     "e180": E180_REF[key]})
    return rows


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def write_report(results: dict) -> None:
    sig = results.get("signals", {})
    a = sig.get("branch_a_reproduced", {})
    loso = results.get("loso_probe", {})
    sel = results.get("selection", {})
    test = results.get("test", {})
    att = results.get("attack", {})
    cap = results.get("capacity", {})
    rows = gate_table(results)
    n_pass = sum(1 for r in rows if r["passed"])
    selected = sel.get("selected", {})
    audit = test.get("audit", {})
    by_name = {row["name"]: row for row in test.get("all_candidates", [])}
    singles = [row for row in test.get("all_candidates", [])
               if row["n_signals"] == 1 and "*" not in row["name"]]

    def test_cell(name, a_key="0.90"):
        row = by_name.get(name)
        if row is None:
            return "—", "—"
        cell = row["per_acceptance"].get(a_key)
        if cell is None:
            return "—", "—"
        return fmt(cell["s1_rejection"]), fmt(cell["s2_acceptance"])

    lines = [
        "# E181 门控信号融合（冻结 E180 双分支，只改门控）",
        "",
        "日期：2026-11-15。分支 A（对抗训练 EEGNet）与分支 B（E65 correlation-logSPD "
        "PCA-96 / k=12 原型）直接读取 E180 冻结产物，**身份判别部分完全未改动、未重训**。",
        "唯一变化：开放集门控由单信号 `g_dist` 换成五个信号的加权融合；权重只用 "
        "Train LOSO 门控 AUC 选择，阈值只用 Train OOF 分位。",
        "",
        f"**十项门禁通过 {n_pass}/10。**",
        "",
        f"选定门控：`{selected.get('name', 'N/A')}`，Train LOSO 门控 AUC "
        f"**{fmt(selected.get('loso_pooled_auc'))}**；Test S1 拒识 "
        f"**{fmt(audit.get('s1_rejection'))}**、Test S2 接受 "
        f"**{fmt(audit.get('s2_acceptance'))}**（a={fmt(PRIMARY_A, 2)}）。",
        "",
        "关键数值：OOF C " + fmt(a.get("oof_accuracy_5block")) + " / macro-F1 "
        + fmt(a.get("oof_macro_f1")) + " / in-sample " + fmt(a.get("in_sample_accuracy"))
        + " / RA " + fmt(att.get("worst_RA", {}).get("RA")) + " / ASafeR "
        + fmt(att.get("worst_ASafeR", {}).get("ASafeR")) + "。",
        "",
        "## 1. 门控信号定义",
        "",
        "| 信号 | 来源 | 定义（越大越像注册者） | Train OOF 均值 / 标准差 |",
        "|---|---|---|---:|",
    ]
    definition = {
        "g_dist": ("分支 B", "−最近原型欧氏距离（E180 原门控）"),
        "g_conf": ("分支 A", "softmax 最大值"),
        "g_margin": ("分支 A", "softmax top1 − top2"),
        "g_energy": ("分支 A", "logsumexp(logits)"),
        "g_entropy": ("分支 A", "Σ p log p（= −熵）"),
    }
    for name in SIGNALS:
        source, text = definition[name]
        lines.append(f"| `{name}` | {source} | {text} | "
                     f"{sig.get('signal_stats', {}).get('mu', {}).get(name, float('nan')):+.4f} / "
                     f"{sig.get('signal_stats', {}).get('sd', {}).get(name, float('nan')):.4f} |")
    lines += [
        "",
        "五个信号都以 Train OOF 的均值/标准差做 z 标准化（Train-only，population std）。",
        "",
        "## 2. 单信号与融合：Train LOSO 门控 AUC vs Test 审计",
        "",
        "**两个来源严格不同**：Train LOSO 门控 AUC 由留身份探针模型（19 身份、"
        "`pm & fit_rows` 训练、`0.6<frac≤0.8` 窗口评估）在 Train 上算出，是唯一的选权依据；"
        "Test S1 拒识 / S2 接受是冻结后的工程审计，**不参与任何选择**。",
        "",
        "### 2.1 单信号",
        "",
        "| 信号 | Train LOSO 门控 AUC | LOSO 逐折均值±std | LOSO U@95 | Test 诊断 AUC* | "
        "Test S1 拒识 | Test S2 接受 | S1+S2 | Test 最优 S1+S2† |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in SIGNALS:
        probe = loso.get("per_signal_loso", {}).get(name, {})
        s1, s2 = test_cell(name)
        trow = by_name.get(name, {})
        lines.append(
            f"| `{name}` | {fmt(probe.get('pooled_auc'))} | "
            f"{fmt(probe.get('fold_auc_mean'))} ± {fmt(probe.get('fold_auc_std'))} | "
            f"{fmt(trow.get('u95_mean'))} | "
            f"{fmt(trow.get('test_diagnostic_auc'))} | {s1} | {s2} | "
            f"{fmt(trow.get('per_acceptance', {}).get('0.90', {}).get('s1_plus_s2'))} | "
            f"{fmt(trow.get('test_optimal', {}).get('max_s1_plus_s2'))} |")
    lines += [
        "",
        "\\* Test 诊断 AUC = 用 Session 列把 S2 当正类（应接受）、S1 当负类（应拒识）算出的 AUC，"
        "只作诊断，不参与选权/选阈值。",
        "† Test 最优 S1+S2 是**犯规上界**：阈值直接在 Test ROC 上扫，仅用于回答「还差多少」。",
        "",
        "### 2.1b 方向冲突（机制读数）",
        "",
        "Train 上五个信号都被定义为「越大越像注册者」。看它们在 Test 上是否还保持同一方向：",
        "",
        "| 信号 | S1 均值 (z) | S2 均值 (z) | AUC（S2 为正） | Test 上与 Train 同向？ |",
        "|---|---:|---:|---:|---|",
    ]
    for name in SIGNALS:
        row = test.get("session_signal_means", {}).get(name, {})
        auc = row.get("auc_s2_vs_s1")
        lines.append(
            f"| `{name}` | {row.get('s1_mean_z', float('nan')):+.3f} | "
            f"{row.get('s2_mean_z', float('nan')):+.3f} | {fmt(auc)} | "
            + ("同向（可用）" if (auc or 0.0) > 0.5 else "反向（有害）") + " |")
    lines += [
        "",
        "门禁要做的是「接受 S2、拒识 S1」，所以理想信号应当在 S2 上取高值。"
        "`g_dist` 在 Test 上方向正确（S2 更靠近原型，AUC 0.756），"
        "而分支 A 的四个置信度类信号在 Test 上**方向相反**（S1 反而更自信，AUC 0.40–0.41）："
        "分支 A 是只在 session1 上训练的判别器，S1 与训练同会话、自然更像「见过」的样本，"
        "S2 跨会话则整体置信度被压低。也就是说，分支 A 的置信度测的是**会话相似度**，"
        "不是「身份是否注册」，在这道门禁上它的符号正好是反的。",
        "",
        "这正是「Train LOSO 上融合有效（0.9401→0.9694）、Test 上融合反而变差"
        "（S1+S2 1.369→1.315）」的原因：LOSO 全程同会话，分支 A 的置信度在那里是有效信息；"
        "Test 跨会话，它变成反向噪声。**任何非负权融合都无法把 Test ROC 抬过 `g_dist` 单独的上限。**",
        "",
        "### 2.2 融合候选（按 Train LOSO 门控 AUC 排序，前 12）",
        "",
        "| 融合 | Train LOSO 门控 AUC | 逐折均值±std | Test 诊断 AUC* | Test S1 拒识 | "
        "Test S2 接受 | S1+S2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sel.get("leaderboard", [])[:12]:
        test_row = by_name.get(row["name"], {})
        cell = test_row.get("per_acceptance", {}).get("0.90", {})
        lines.append(f"| {row['name']} | {fmt(row['loso_pooled_auc'])} | "
                     f"{fmt(row['loso_fold_auc_mean'])} ± {fmt(row['loso_fold_auc_std'])} | "
                     f"{fmt(test_row.get('test_diagnostic_auc'))} | "
                     f"{fmt(cell.get('s1_rejection'))} | {fmt(cell.get('s2_acceptance'))} | "
                     f"{fmt(cell.get('s1_plus_s2'))} |")
    lines += [
        "",
        f"候选总数 {sel.get('n_candidates')}（5 单信号 + 10 组两两 × 9 个 w + 10 组三路等权 "
        "+ 4/5 路等权）。选定规则：Train LOSO 合并 AUC 最大，平手时按逐折均值、再按信号数少者。",
        "",
        "## 3. 阈值与工作点（Test 审计，多 a）",
        "",
        f"τ(a) = Train OOF 融合分数的 `quantile(1-a)`。主报告工作点 a={fmt(PRIMARY_A, 2)}"
        "（与 E180 相同）。下表按 a 扫描，**全部是审计值**。",
        "",
        "| a | τ | S1 拒识 | S2 接受 | S1+S2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, cell in by_name.get(selected.get("name"), {}).get("per_acceptance",
                                                               {}).items():
        lines.append(f"| {key} | {cell['tau']:+.4f} | {fmt(cell['s1_rejection'])} | "
                     f"{fmt(cell['s2_acceptance'])} | {fmt(cell['s1_plus_s2'])} |")
    lines += [
        "",
        "## 4. 完整十项门禁表",
        "",
        "| # | 层面 | 指标 | 目标 | 实测 | 判定 | 证据来源 |",
        "|---|---|---|---|---:|---|---|",
    ]
    evidence = {
        "#1": "分支 A 冻结全量模型 in-sample argmax（§7 另有容量探针）",
        "#2": "分支 A 冻结 5 块 OOF 闭集准确率",
        "#3": "同一批 5 块 OOF 预测的身份 macro-F1",
        "#4": "5 块 OOF 同口径",
        "#5": "融合门控留身份探针 LOSO 合并 AUROC（§2）",
        "#6": "同一 LOSO 的 U@95 均值（§2）",
        "#7": "融合系统自建 PGD（ε×目标 A/B/C/D 取最差，D 直接攻融合门控）",
        "#8": "同一攻击条件的 ASafeR 最差值",
        "#9": "冻结后 Test 一次性评分（a=0.90）",
        "#10": "同 #9",
    }
    for row in rows:
        mark = "通过" if row["passed"] else ("未通过" if row["passed"] is not None
                                             else "N/A")
        lines.append(f"| {row['gate']} | {row['layer']} | {row['metric']} | "
                     f"{row['target']} | {fmt(row['value'])} | {mark} | "
                     f"{evidence[row['gate']]} |")
    lines += [
        "",
        "## 5. 与 E180 逐项对照",
        "",
        "| # | 门禁 | 目标 | E180（单门控） | E181（融合门控） | 变化 | 判定 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        value = row["value"]
        delta = "N/A" if value is None else f"{value - row['e180']:+.4f}"
        mark = "通过" if row["passed"] else ("未通过" if row["passed"] is not None
                                             else "N/A")
        lines.append(f"| {row['gate']} | {row['metric']} | {row['target']} | "
                     f"{fmt(row['e180'])} | {fmt(value)} | {delta} | {mark} |")
    e180_pass = 4
    unchanged = [row for row in rows if row["gate"] in ("#2", "#3", "#4")]
    lines += [
        "",
        f"通过项数：E180 **{e180_pass}/10** → E181 **{n_pass}/10**。"
        "#2/#3/#4 完全不变，因为身份判别部分没有被改动"
        "（同一份 OOF logits / 同一份全量模型）。",
        "",
        "## 6. 回答：门控融合能否让 S1 与 S2 同时 ≥ 0.75？",
        "",
        f"**不能。** 选定融合门控在 a={fmt(PRIMARY_A, 2)} 工作点上 Test S1 拒识 "
        f"{fmt(audit.get('s1_rejection'))}"
        f"（{(audit.get('s1_rejection') or 0.0) - 0.75:+.4f} 相对 0.75）、"
        f"S2 接受 {fmt(audit.get('s2_acceptance'))}"
        f"（{(audit.get('s2_acceptance') or 0.0) - 0.75:+.4f} 相对 0.75），"
        f"两者之和 {fmt(audit.get('s1_plus_s2'))}，距离同时达标的 1.50 还差 "
        f"{fmt(max(0.0, 1.50 - (audit.get('s1_plus_s2') or 0.0)))}。",
        "",
    ]
    opt = by_name.get(selected.get("name"), {}).get("test_optimal", {})
    diag = by_name.get(selected.get("name"), {}).get("test_diagnostic_auc")
    best_opt = max([(row.get("test_optimal", {}).get("max_s1_plus_s2", 0.0), row["name"])
                    for row in test.get("all_candidates", [])] or [(0.0, "N/A")])
    best_auc = max([row.get("test_diagnostic_auc") or 0.0
                    for row in test.get("all_candidates", [])] or [0.0])
    lines += [
        f"即使**犯规**地扫描 Test 自身最优阈值（仅作上界），选定门控的最大 "
        f"S1+S2 = {fmt(opt.get('max_s1_plus_s2'))}"
        f"（S1 {fmt(opt.get('s1_rejection'))} / S2 {fmt(opt.get('s2_acceptance'))}），"
        f"对应 Test 诊断 AUC {fmt(diag)}。"
        f"全部 {sel.get('n_candidates')} 个候选里最高的犯规上界是 "
        f"{fmt(best_opt[0])}（`{best_opt[1]}`），仍然到不了 1.50。",
        "",
        "算术：若门控在 S1/S2 上是等方差高斯，`max(S1拒识 + S2接受) = 2·AUC`，"
        "两门同时 ≥0.75 需要 AUC ≥ 0.75 且阈值恰好落在交叠点；但阈值被限制为由 Train "
        "OOF 分位给出，存在**阈值迁移损失**（实测 `2·AUC − max(S1+S2)` 中位数 "
        + fmt(transfer_loss(results)) + "）。"
        "按本实验各候选的（诊断 AUC, 犯规上界）拟合，"
        "要把 S1+S2 搬到 1.50，Test 诊断 AUC 需要达到约 "
        + fmt(required_auc(results)) + "（当前最好的候选只有 "
        + fmt(best_auc) + "）。",
        "",
        "**「通过 5/10」的正确读法**：#10 是靠把工作点整体往「多拒识」方向搬才拿到 0.9651，"
        "代价是 #9 从 E180 的 0.6335 崩到 0.3494；两者是同一个 ROC 的两端。"
        "#5/#6 的提升（0.9401→0.9694、0.5847→0.8145）都发生在**同会话 LOSO** 口径上，"
        "没有迁移到 Test。所以相对 E180 的「+1 项」不是能力提升，而是工作点移动 + "
        "同会话指标的虚高。真正决定 #9/#10 的 Test 诊断 AUC，全实验最高只有 "
        + fmt(best_auc) + "（`g_dist` 系候选），与 E180 的 0.756 相同。",
        "",
        "## 7. #1 拟合容量的尝试（如实标注）",
        "",
        "| 变体 | in-sample 准确率 | 说明 |",
        "|---|---:|---|",
        f"| 冻结全量模型 logits argmax（主口径） | {fmt(cap.get('logits_in_sample_accuracy'))} | "
        "与 E180 同一模型、同一判决，0.9988 |",
        f"| 同一模型 128-d 嵌入 leave-one-out 1-NN | "
        f"{fmt(cap.get('knn_in_sample_accuracy'))} | **容量/记忆指标**，不是泛化指标 |",
        f"| 同一模型的块外 1-NN（诚实对照） | {fmt(cap.get('knn_block_oof_accuracy'))} | "
        "5 块 OOF 口径的 1-NN，说明记忆规则在折外并不更好 |",
        f"| 继续训练 {cap.get('extra_epochs')} epoch | "
        f"{fmt(cap.get('extra_epochs_in_sample_accuracy'))} | 更多 epoch 的容量探针 |",
        "",
        "结论：两个便宜的容量探针都**没有**把 #1 抬到 1.00——128-d 嵌入的 leave-one-out 1-NN "
        f"只有 {fmt(cap.get('knn_in_sample_accuracy'))}（比 logits 还低，说明 128 维池化特征里"
        "同身份样本并非彼此最近），继续训练 15 epoch 只到 "
        f"{fmt(cap.get('extra_epochs_in_sample_accuracy'))}（仍差 "
        f"{fmt(1.0 - (cap.get('extra_epochs_in_sample_accuracy') or 0.0))}）。"
        "要过 #1 需要更大 width / 更长训练，但那本质是**拟合容量**（24.8 万参数对 1.13 万样本），"
        f"块外 1-NN {fmt(cap.get('knn_block_oof_accuracy'))} 与 5 块 OOF 闭集 "
        f"{fmt(a.get('oof_accuracy_5block'))} 才是泛化读数。"
        "主门禁表按冻结模型的 logits 判决记为未通过，避免把记忆当能力。",
        "",
        "## 8. 对抗（#7/#8）",
        "",
        f"攻击子集与 E180 完全相同（同 seed、同窗口、每任务上限 400，共 "
        f"{att.get('n_trials')} 条）。目标 D 改为**直接最大化融合门控分数**（五个信号全部可微）。",
        "",
        "| 条件 | ε | 目标 | RA | RA CI95 | ASafeR | ASafeR CI95 | 闭集准确率 | 接受率 | 误认率 |",
        "|---|---:|---|---:|---|---:|---|---:|---:|---:|",
    ]

    def row_line(entry, label, eps_text, goal_text):
        return (f"| {label} | {eps_text} | {goal_text} | "
                f"{fmt(entry['RA'])} | [{fmt(entry['RA_ci95'][0], 3)}, "
                f"{fmt(entry['RA_ci95'][1], 3)}] | {fmt(entry['ASafeR'])} | "
                f"[{fmt(entry['ASafeR_ci95'][0], 3)}, {fmt(entry['ASafeR_ci95'][1], 3)}] | "
                f"{fmt(entry['closed_set_accuracy'])} | {fmt(entry['accept_rate'])} | "
                f"{fmt(entry['wrong_identity_rate'])} |")

    lines.append(row_line(att.get("clean", {"RA": 0, "RA_ci95": [0, 0], "ASafeR": 0,
                                             "ASafeR_ci95": [0, 0],
                                             "closed_set_accuracy": 0, "accept_rate": 0,
                                             "wrong_identity_rate": 0}), "clean", "—", "—"))
    for entry in att.get("conditions", []):
        lines.append(row_line(entry, "adv", f"{entry['eps']:.3f}", entry["goal"]))
    lines += [
        "",
        f"判定：#7 RA 最差 **{fmt(att.get('worst_RA', {}).get('RA'))}**"
        f"（ε={att.get('worst_RA', {}).get('eps')}, 目标 "
        f"{att.get('worst_RA', {}).get('goal')}）；#8 ASafeR 最差 "
        f"**{fmt(att.get('worst_ASafeR', {}).get('ASafeR'))}**"
        f"（ε={att.get('worst_ASafeR', {}).get('eps')}, 目标 "
        f"{att.get('worst_ASafeR', {}).get('goal')}）。",
        "",
        "## 9. 合规与边界",
        "",
        "- 分支 A / 分支 B 全部来自 E180 冻结产物；E181 未重训身份模型、未改任何已有目录。",
        "- 融合权重只用 **Train LOSO 门控 AUC** 选择；阈值只用 **Train OOF** 分位；"
        "PCA / 标准化 / 原型统计量全部 Train-only。",
        "- LOSO 探针严格 `pm & fit_rows`（19 个注册身份的拟合块），评估块 "
        "`0.6<frac≤0.8` 从未进入探针训练或原型。",
        "- Test 只评分：先写 `test_predictions_fused.csv`，之后才读取 Session 做 #9/#10 审计；"
        "Session 从未参与特征/模型/阈值/权重。",
        "- 表格中所有 Test 诊断 AUC 与 Test 最优阈值上界都是**审计/上界**，不是决策依据。",
        "- #7/#8 是自建白盒 PGD 代理，不是官方攻击套件。",
        "- 未提交 git。",
        "",
        "## 10. Test 官方指标",
        "",
        "**Test 官方 AUC / F1 / C / RA / ASafeR / U / Score = N/A**（Test 无真实标签）。",
        "",
        "## 11. 文件",
        "",
        "| 文件 | 内容 |",
        "|---|---|",
        "| `run_e181.py` | 全部逻辑（`--stage signals/loso/select/test/attack/capacity/all/report`） |",
        "| `config.json` | 配置与合规约束 |",
        "| `results.json` | 全部数值（含所有候选 × 所有 a 的审计） |",
        "| `run.log` | 运行日志 |",
        "| `cache/signals.npz` | Train OOF / Test 五个信号（原始与 z） |",
        "| `cache/loso.npz` | 20 折留身份探针的逐行信号 |",
        "| `cache/loso_model_*.pt` | 20 个留身份探针模型 |",
        "| `cache/test_logits.npy` | 冻结分支 A 的 Test logits |",
        "| `test_predictions_fused.csv` | 选定门控的 Test 预测（工程审计用） |",
        "",
        "耗时（秒）：" + "、".join(
            f"{name}={value:.0f}" for name, value in
            (results.get("stage_elapsed_seconds") or {}).items() if value is not None)
        + f"；合计 {results.get('runtime_seconds_total', float('nan')):.0f} s。",
        "",
        "```powershell",
        f"& \"{ROOT / '.venv' / 'Scripts' / 'python.exe'}\" \"{OUT / 'run_e181.py'}\" --stage all",
        "```",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def required_auc(results: dict) -> float:
    """Diagnostic Test AUC needed for S1+S2 = 1.50, from the observed ROC losses."""
    rows = [row for row in results.get("test", {}).get("all_candidates", [])
            if (row.get("test_diagnostic_auc") or 0.0) > 0.55]
    if not rows:
        return float("nan")
    auc = np.asarray([row["test_diagnostic_auc"] for row in rows], dtype=float)
    top = np.asarray([row["test_optimal"]["max_s1_plus_s2"] for row in rows], dtype=float)
    if top.max() >= 1.50:
        return float(np.min(auc[top >= 1.50]))
    slope, intercept = np.polyfit(auc, top, 1)
    if slope <= 1e-6:
        return float("nan")
    return float((1.50 - intercept) / slope)


def transfer_loss(results: dict) -> float:
    """Median gap between the ideal 2*AUC bound and the realised Test-optimal sum."""
    gaps = []
    for row in results.get("test", {}).get("all_candidates", []):
        auc = row.get("test_diagnostic_auc") or 0.0
        top = row.get("test_optimal", {}).get("max_s1_plus_s2")
        if auc > 0.55 and top is not None:
            gaps.append(2.0 * auc - top)
    return float(np.median(gaps)) if gaps else float("nan")


def refresh_meta(results: dict) -> dict:
    results["gates"] = gate_table(results)
    stage_elapsed = {name: results.get(name, {}).get("elapsed_seconds")
                     for name in ("signals", "loso_probe", "selection", "test",
                                  "attack", "capacity")}
    results["stage_elapsed_seconds"] = stage_elapsed
    results["runtime_seconds_total"] = float(
        sum(v for v in stage_elapsed.values() if v is not None))
    results["environment"] = {"python": platform.python_version(),
                              "torch": torch.__version__, "numpy": np.__version__,
                              "gpu": torch.cuda.get_device_name(0)}
    return results


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all",
                        choices=("signals", "loso", "select", "test", "attack",
                                 "capacity", "all", "report"))
    args = parser.parse_args()
    if args.stage == "report":
        results = refresh_meta(load_results())
        save_results(results)
        write_report(results)
        print("report.md rebuilt from results.json")
        return
    if not torch.cuda.is_available():
        parser.error("E181 requires CUDA")
    sys.stdout = Tee(OUT / "run.log")
    log(f"\n===== E181 stage={args.stage} {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    started = time.time()
    if args.stage in ("signals", "all"):
        stage_signals()
    if args.stage in ("loso", "all"):
        stage_loso()
    if args.stage in ("select", "all"):
        stage_select()
    if args.stage in ("test", "all"):
        stage_test()
    if args.stage in ("attack", "all"):
        stage_attack()
    if args.stage in ("capacity", "all"):
        stage_capacity()
    results = refresh_meta(load_results())
    save_results(results)
    write_report(results)
    log(f"done in {time.time() - started:.0f}s -> {RESULTS_PATH} / {REPORT_PATH}")


if __name__ == "__main__":
    main()
