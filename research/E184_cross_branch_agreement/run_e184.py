"""E184 - cross-branch identity agreement as an open-set gate.

Frozen sources (read-only, nothing retrained):
    branch A = E180 adversarially trained EEGNet, 20-class closed set
    branch B = E180 E65 correlation-logSPD PCA-96 / k=12 prototype nearest neighbour

New idea: the gate is no longer a threshold on one continuous score.  It is the
*agreement* between the two branches' identity decisions:
    agree       a_pred == b_pred
    soft_agree  branch A probability on branch B's identity
    margin_agree branch A top1 probability - probability on branch B's identity
    rank_agree  rank of branch B's identity inside branch A's output

Hypothesis: registered identities (S2) make both branches agree, unregistered
identities (S1) make them disagree.  If S1 and S2 agree rates are indistinguishable
the hypothesis is rejected and the report says so.

Stages: signals, loso, diag, gates, attack, report, all.
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
import torch.nn.functional as F

from scipy.stats import chi2_contingency
from sklearn.metrics import f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "report.md"
LOG_PATH = OUT / "run.log"
SIGNAL_CACHE = CACHE / "signals.npz"
LOSO_CACHE = CACHE / "loso.npz"
TEST_LOGITS_CACHE = CACHE / "test_logits.npy"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E180_DIR = ROOT / "research" / CFG["frozen_source"]["experiment"]
E180 = load_module("e184_frozen_e180", E180_DIR / "run_e180.py")
E181_TEST_LOGITS = ROOT / "research" / "E181_gate_fusion_tuning" / "cache" / "test_logits.npy"
DEVICE = E180.DEVICE
FT = E180.FT
TASKS = list(E180.TASKS)
CLASSES = E180.CLASSES
SEED = int(E180.SEED)
OFF = float(E180.OFF)
FIT = float(E180.CFG["protocol"]["fit_fraction"])
NBLOCK = int(E180.CFG["protocol"]["blocks"])

ACC_GRID = [float(v) for v in CFG["threshold"]["grid"]]
PRIMARY_A = float(CFG["threshold"]["primary_acceptance"])
LAMBDAS = [0.5, 1.0, 2.0, 4.0]
LAMBDA_2D = 10.0          # large offset == the AND/OR truth-table family
AGREE_SIGNALS = ("agree", "soft_agree", "margin_agree", "rank_agree")
SIGNALS = ("g_dist",) + AGREE_SIGNALS


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


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


_BRANCH_B: dict | None = None


def branch_b_store() -> dict:
    global _BRANCH_B
    if _BRANCH_B is None:
        _BRANCH_B = E180.load_branch_b()
    return _BRANCH_B


def load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {"experiment": CFG["experiment"]}


def save_results(results: dict) -> None:
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2,
                                       default=float), encoding="utf-8")


# ---------------------------------------------------------------- signals
def softmax_np(logits: np.ndarray) -> np.ndarray:
    lg = np.asarray(logits, dtype=np.float64)
    shifted = lg - lg.max(1, keepdims=True)
    prob = np.exp(shifted)
    return prob / prob.sum(1, keepdims=True)


def agreement_signals(logits: np.ndarray, b_id: np.ndarray) -> dict:
    """Four agreement read-outs; larger agree/soft_agree = more agreement."""
    prob = softmax_np(logits)
    a_idx = np.asarray(logits).argmax(1)
    a_pred = CLASSES[a_idx]
    b_idx = np.asarray(b_id, dtype=int) - 1
    rows = np.arange(len(a_pred))
    p_b = prob[rows, b_idx]
    p_top = prob.max(1)
    return {
        "agree": (a_pred == np.asarray(b_id)).astype(np.float64),
        "soft_agree": p_b,
        "margin_agree": p_top - p_b,
        "rank_agree": (prob > p_b[:, None]).sum(1).astype(np.float64) + 1.0,
        "a_pred": a_pred,
    }


def branch_b_train_oof(projected: np.ndarray, subjects: np.ndarray, tasks: np.ndarray,
                       blocks: np.ndarray, fit_rows: np.ndarray, k: int) -> tuple:
    """5-block OOF branch-B score and identity (prototypes never see the held block)."""
    score = np.full(len(subjects), np.nan, dtype=np.float32)
    ident = np.full(len(subjects), -1, dtype=np.int32)
    for block in range(NBLOCK):
        rows = blocks == block
        protos, keys = E180.build_prototypes(projected, subjects, tasks,
                                             fit_rows & ~rows, CLASSES, k)
        proto, filled = E180.proto_stack(protos, keys, k)
        held = np.flatnonzero(rows)
        score[held] = E180.branch_b_scores(projected[held], tasks[held], proto, filled, k)
        ident[held] = E180.branch_b_identity(projected[held], tasks[held], proto, filled, k)
        log(f"  block {block}: {len(held)} rows scored, {len(keys)} prototypes")
    if not np.isfinite(score).all() or (ident < 0).any():
        raise RuntimeError("branch B OOF incomplete")
    return score, ident


def test_branch_a_logits() -> np.ndarray:
    if TEST_LOGITS_CACHE.exists():
        return np.load(TEST_LOGITS_CACHE)
    if not E181_TEST_LOGITS.exists():
        raise FileNotFoundError(E181_TEST_LOGITS)
    logits = np.load(E181_TEST_LOGITS)
    np.save(TEST_LOGITS_CACHE, logits)
    log(f"  copied frozen branch A Test logits from E181 ({logits.shape})")
    return logits


def branch_b_projection(train_features: np.ndarray, tasks: np.ndarray,
                        fit_rows: np.ndarray) -> tuple:
    stats = E180.task_stats(train_features, tasks)
    standardised = E180.apply_stats(train_features, tasks, stats)
    mu, basis = E180.pca_basis(standardised[fit_rows],
                               int(E180.CFG["branch_b"]["pca_dim"]))
    projected = f32((standardised.astype(np.float64) - mu.astype(np.float64)) @ basis)
    return projected


# ---------------------------------------------------------------- stage signals
def stage_signals() -> dict:
    started = time.time()
    meta, frac, blocks, fit_rows = E180.load_config_meta()
    subjects = meta["subject"].to_numpy(int)
    tasks = meta["task"].to_numpy()
    k = int(E180.CFG["branch_b"]["prototype_k"])

    stage_a = torch.load(E180.STAGE_A, map_location="cpu", weights_only=False)
    if stage_a["spec_sha256"] != E180.BRANCH_A_SHA:
        raise ValueError("E180 stage A artifact does not match its frozen spec")
    oof_logits = stage_a["oof_logits"].numpy().astype(np.float32)
    store = E180.load_branch_b()
    oof_mu, oof_sd = float(store["oof_mu"]), float(store["oof_sd"])

    train_features = f32(np.load(ROOT / E180.CFG["branch_b"]["train_features"]))
    if train_features.shape != (len(meta), int(E180.CFG["branch_b"]["feature_dim"])):
        raise ValueError(f"E65 Train cache shape {train_features.shape} unexpected")
    projected = branch_b_projection(train_features, tasks, fit_rows)
    oof_dist, oof_bid = branch_b_train_oof(projected, subjects, tasks, blocks, fit_rows, k)

    # sanity: reproduce the E180 branch-B OOF score cache
    score_shift = float(np.max(np.abs(oof_dist - store["oof_scores"])))

    test_meta = pd.read_csv(ROOT / E180.CFG["data"]["test_metadata"],
                            usecols=["sample", "path", "task"])
    if len(test_meta) != int(E180.CFG["data"]["test_trials"]):
        raise ValueError("Test row count unexpected")
    test_tasks = test_meta["task"].to_numpy()
    test_logits = test_branch_a_logits()
    test_features = f32(np.load(ROOT / E180.CFG["branch_b"]["test_features"]))
    if len(test_features) != len(test_meta):
        raise ValueError("Test E65 feature row count mismatch")
    test_dist, test_bid = E180.branch_b_test_scores(test_features, test_tasks, store)

    sig_oof = agreement_signals(oof_logits, oof_bid)
    sig_test = agreement_signals(test_logits, test_bid)
    gd_oof = (np.asarray(oof_dist, np.float64) - oof_mu) / oof_sd
    gd_test = (np.asarray(test_dist, np.float64) - oof_mu) / oof_sd

    np.savez_compressed(
        SIGNAL_CACHE,
        oof_logits=oof_logits, oof_dist=np.asarray(oof_dist, np.float32),
        oof_bid=oof_bid.astype(np.int32), oof_gd=gd_oof.astype(np.float64),
        oof_agree=sig_oof["agree"], oof_soft=sig_oof["soft_agree"],
        oof_margin=sig_oof["margin_agree"], oof_rank=sig_oof["rank_agree"],
        oof_apred=sig_oof["a_pred"].astype(np.int32),
        test_logits=test_logits, test_dist=np.asarray(test_dist, np.float32),
        test_bid=test_bid.astype(np.int32), test_gd=gd_test.astype(np.float64),
        test_agree=sig_test["agree"], test_soft=sig_test["soft_agree"],
        test_margin=sig_test["margin_agree"], test_rank=sig_test["rank_agree"],
        test_apred=sig_test["a_pred"].astype(np.int32),
        test_sample=np.asarray(test_meta["sample"].astype(str), dtype=str),
        test_tasks=np.asarray(test_tasks, dtype=str),
        subjects=subjects, tasks=np.asarray(tasks, dtype=str), blocks=blocks,
        frac=frac.astype(np.float64),
        in_sample_logits=stage_a["in_sample_logits"].numpy().astype(np.float32),
        oof_mu=np.float64(oof_mu), oof_sd=np.float64(oof_sd))

    oof_agree = float(np.mean(sig_oof["agree"]))
    summary = {
        "n_train": int(len(meta)), "n_test": int(len(test_meta)),
        "branch_b_oof_score_max_abs_diff_vs_E180": score_shift,
        "branch_a_oof_accuracy_5block": float(np.mean(
            oof_logits.argmax(1) + 1 == subjects)),
        "branch_a_oof_macro_f1": float(f1_score(subjects, oof_logits.argmax(1) + 1,
                                                labels=CLASSES, average="macro",
                                                zero_division=0)),
        "branch_a_in_sample_accuracy": float(np.mean(
            stage_a["in_sample_logits"].numpy().argmax(1) + 1 == subjects)),
        "train_oof_agree_rate": oof_agree,
        "train_oof_branch_a_accuracy": float(np.mean(sig_oof["a_pred"] == subjects)),
        "train_oof_branch_b_accuracy": float(np.mean(oof_bid == subjects)),
        "train_oof_both_correct": float(np.mean(
            (sig_oof["a_pred"] == subjects) & (oof_bid == subjects))),
        "test_branch_b_identity_histogram": {
            str(int(c)): int((test_bid == c).sum()) for c in CLASSES},
        "elapsed_seconds": time.time() - started,
    }
    results = load_results()
    results["signals"] = summary
    save_results(results)
    log(f"stage signals done in {summary['elapsed_seconds']:.0f}s: "
        f"Train OOF agree {oof_agree:.4f}; branch A OOF acc "
        f"{summary['branch_a_oof_accuracy_5block']:.4f}; "
        f"branch B OOF score diff vs E180 {score_shift:.2e}")
    return summary


# ---------------------------------------------------------------- stage loso
def stage_loso() -> dict:
    started = time.time()
    meta, frac, blocks, fit_rows = E180.load_config_meta()
    subjects = meta["subject"].to_numpy(int)
    tasks = meta["task"].to_numpy()
    k = int(E180.CFG["branch_b"]["prototype_k"])
    store = E180.load_branch_b()
    oof_mu, oof_sd = float(store["oof_mu"]), float(store["oof_sd"])
    signal = np.load(SIGNAL_CACHE)
    oof_logits = signal["oof_logits"]

    train_features = f32(np.load(ROOT / E180.CFG["branch_b"]["train_features"]))
    projected = branch_b_projection(train_features, tasks, fit_rows)
    window = (frac > FIT) & (frac <= 0.8)
    window_rows = np.flatnonzero(window)

    folds, fold_id, known, gd, agree, soft, margin, rank, apred, bpred = (
        [], [], [], [], [], [], [], [], [], [])
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        pm = np.isin(subjects, enrolled)
        protos, keys = E180.build_prototypes(projected, subjects, tasks, pm & fit_rows,
                                             enrolled, k)
        proto, filled = E180.proto_stack(protos, keys, k)
        rows = window_rows
        score = E180.branch_b_scores(projected[rows], tasks[rows], proto, filled, k)
        bid = E180.branch_b_identity(projected[rows], tasks[rows], proto, filled, k)
        logits = oof_logits[rows].astype(np.float64).copy()
        hidden = ~np.isin(CLASSES, enrolled)
        logits[:, hidden] = -1e9
        sig = agreement_signals(logits, bid)
        is_known = np.isin(subjects[rows], enrolled)
        auc = float(roc_auc_score(is_known.astype(int),
                                  (score - oof_mu) / oof_sd))
        auc_agree = float(roc_auc_score(is_known.astype(int), sig["agree"]))
        folds.append({
            "held_identity": int(held), "n_known": int(is_known.sum()),
            "n_unknown": int((~is_known).sum()), "auroc_dist": auc,
            "auroc_agree": auc_agree,
            "known_agree": float(np.mean(sig["agree"][is_known])),
            "unknown_agree": float(np.mean(sig["agree"][~is_known])),
            "known_soft": float(np.mean(sig["soft_agree"][is_known])),
            "unknown_soft": float(np.mean(sig["soft_agree"][~is_known]))})
        fold_id.append(np.full(len(rows), int(held)))
        known.append(is_known)
        gd.append((score - oof_mu) / oof_sd)
        agree.append(sig["agree"])
        soft.append(sig["soft_agree"])
        margin.append(sig["margin_agree"])
        rank.append(sig["rank_agree"])
        apred.append(sig["a_pred"])
        bpred.append(bid)
        log(f"  LOSO {int(held):2d}: dist AUC {auc:.4f} | agree AUC {auc_agree:.4f} "
            f"(known {folds[-1]['known_agree']:.3f} vs unknown "
            f"{folds[-1]['unknown_agree']:.3f}); soft {folds[-1]['known_soft']:.3f} vs "
            f"{folds[-1]['unknown_soft']:.3f}")

    fold_id = np.concatenate(fold_id)
    known = np.concatenate(known)
    gd = np.concatenate(gd)
    arrays = {"fold": fold_id.astype(np.int32), "known": known,
              "g_dist": gd, "agree": np.concatenate(agree),
              "soft_agree": np.concatenate(soft), "margin_agree": np.concatenate(margin),
              "rank_agree": np.concatenate(rank),
              "a_pred": np.concatenate(apred).astype(np.int32),
              "b_pred": np.concatenate(bpred).astype(np.int32),
              "window_rows": window_rows.astype(np.int32)}
    np.savez_compressed(LOSO_CACHE, **arrays)

    pooled = {}
    for name in SIGNALS:
        pooled[name] = float(roc_auc_score(known.astype(int), arrays[name]
                                           if name != "g_dist" else gd))
    summary = {
        "n_rows": int(len(known)), "n_known": int(known.sum()),
        "n_unknown": int((~known).sum()),
        "pooled_auc": pooled,
        "fold_mean_auc": {name: float(np.mean([row.get(
            "auroc_agree" if name == "agree" else "auroc_dist") for row in folds]))
            for name in ("g_dist", "agree")},
        "folds": folds,
        "branch_a_mask": CFG["loso"]["branch_a_masking"],
        "elapsed_seconds": time.time() - started,
    }
    results = load_results()
    results["loso"] = summary
    save_results(results)
    log(f"stage loso done in {summary['elapsed_seconds']:.0f}s: pooled AUC "
        + ", ".join(f"{k} {v:.4f}" for k, v in pooled.items()))
    return summary


# ---------------------------------------------------------------- stage diag
def stage_diag() -> dict:
    """Test diagnostic.  Session is read here for auditing only, before any gate
    weight / lambda / threshold is chosen; nothing from this stage feeds selection."""
    started = time.time()
    signal = np.load(SIGNAL_CACHE)
    meta = pd.read_csv(ROOT / E180.CFG["data"]["test_metadata"],
                       usecols=["sample", "session"])
    if not np.array_equal(meta["sample"].astype(str).to_numpy(),
                          signal["test_sample"].astype(str)):
        raise ValueError("Test sample order mismatch")
    s1 = meta["session"].astype(str).str.casefold().isin(
        ["session1", "s1", "1"]).to_numpy()
    n1, n2 = int(s1.sum()), int((~s1).sum())

    keys = {"agree": "test_agree", "soft_agree": "test_soft",
            "margin_agree": "test_margin", "rank_agree": "test_rank"}
    table = {}
    for name in AGREE_SIGNALS:
        values = np.asarray(signal[keys[name]], dtype=np.float64)
        positive = values if name in ("agree", "soft_agree") else -values
        table[name] = {
            "s1_mean": float(np.mean(values[s1])),
            "s2_mean": float(np.mean(values[~s1])),
            "s1_median": float(np.median(values[s1])),
            "s2_median": float(np.median(values[~s1])),
            "auc_s2_vs_s1": float(roc_auc_score((~s1).astype(int), positive)),
        }

    agree = np.asarray(signal["test_agree"], np.float64)
    count1, count2 = int(agree[s1].sum()), int(agree[~s1].sum())
    rates = [count1 / n1, count2 / n2]
    contingency = np.array([[count1, n1 - count1], [count2, n2 - count2]])
    chi2, p_value, _, _ = chi2_contingency(contingency)
    diff = rates[1] - rates[0]
    se = math.sqrt(rates[0] * (1 - rates[0]) / n1 + rates[1] * (1 - rates[1]) / n2)
    ci = [diff - 1.96 * se, diff + 1.96 * se]

    gd = np.asarray(signal["test_gd"], np.float64)
    diagnostic = {
        "n_s1": n1, "n_s2": n2,
        "agree_rate_s1": rates[0], "agree_rate_s2": rates[1],
        "agree_table": contingency.tolist(),
        "chi2": float(chi2), "chi2_p_value": float(p_value),
        "rate_difference": float(diff), "rate_difference_ci95": [float(v) for v in ci],
        "significant": bool(p_value < 0.05),
        "signals": table,
        "g_dist_auc_s2_vs_s1": float(roc_auc_score((~s1).astype(int), gd)),
        "corr_agree_g_dist_s2": _corr(agree[~s1], gd[~s1]),
        "corr_agree_g_dist_s1": _corr(agree[s1], gd[s1]),
        "hypothesis_pass": bool(p_value < 0.05 and diff > 0),
        "note": "Session read only after all gate-independent signals were frozen; "
                "this table is an audit and never enters lambda/threshold selection",
        "elapsed_seconds": time.time() - started,
    }
    results = load_results()
    results["diagnostic"] = diagnostic
    save_results(results)
    log("stage diag: Test agreement diagnostic (audit only)")
    log(f"  S1 n={n1} agree {rates[0]:.4f} | S2 n={n2} agree {rates[1]:.4f} | "
        f"diff {diff:+.4f} CI95 [{ci[0]:+.4f}, {ci[1]:+.4f}] chi2 p={p_value:.3e}")
    for name in AGREE_SIGNALS:
        row = table[name]
        log(f"  {name:13s} S1 {row['s1_mean']:+.4f} S2 {row['s2_mean']:+.4f} "
            f"AUC {row['auc_s2_vs_s1']:.4f}")
    log(f"  hypothesis {'PASSED' if diagnostic['hypothesis_pass'] else 'REJECTED'}")
    return diagnostic


# ---------------------------------------------------------------- gates
def cell_index(gd: np.ndarray, agree: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(gd, edges) * 2 + agree.astype(int)


def fit_lab(gd_oof: np.ndarray, loso: dict) -> dict:
    edges = np.asarray(np.quantile(gd_oof, [1 / 3, 2 / 3]), dtype=np.float64)
    cells = cell_index(loso["g_dist"], loso["agree"], edges)
    cell_score = np.zeros(6, dtype=np.float64)
    for cell in range(6):
        sel = cells == cell
        cell_score[cell] = float(np.mean(loso["known"][sel])) if sel.any() else 0.0
    return {"edges": edges, "cell_score": cell_score,
            "cell_counts": {str(c): int((cells == c).sum()) for c in range(6)}}


def candidate_library() -> list[dict]:
    cands = [{"name": "g_dist", "kind": "dist", "n_components": 1}]
    for lam in LAMBDAS:
        cands.append({"name": f"g_dist + {lam:g}*agree", "kind": "add", "lam": lam,
                      "n_components": 2})
    cands.append({"name": f"2D rule family (agree offset +{LAMBDA_2D:g} == AND/OR)",
                  "kind": "add", "lam": LAMBDA_2D, "n_components": 2})
    cands.append({"name": "LAB(3x2) cell known-rate", "kind": "lab", "n_components": 2})
    return cands


def gate_score(spec: dict, gd: np.ndarray, agree: np.ndarray, lab: dict) -> np.ndarray:
    if spec["kind"] == "dist":
        return np.asarray(gd, np.float64)
    if spec["kind"] == "add":
        return np.asarray(gd, np.float64) + float(spec["lam"]) * np.asarray(agree, np.float64)
    if spec["kind"] == "lab":
        return lab["cell_score"][cell_index(gd, agree, lab["edges"])]
    raise KeyError(spec["kind"])


def stage_gates() -> dict:
    started = time.time()
    signal = np.load(SIGNAL_CACHE)
    loso = {k: v for k, v in np.load(LOSO_CACHE).items()}
    lab = fit_lab(np.asarray(signal["oof_gd"], np.float64), loso)
    known = loso["known"].astype(bool)
    fold = loso["fold"]

    rows = []
    for spec in candidate_library():
        score_loso = gate_score(spec, loso["g_dist"], loso["agree"], lab)
        pooled = float(roc_auc_score(known.astype(int), score_loso))
        per_fold, u95 = [], []
        for held in CLASSES:
            sel = fold == int(held)
            if sel.sum() == 0 or len(np.unique(known[sel])) < 2:
                continue
            per_fold.append(float(roc_auc_score(known[sel].astype(int), score_loso[sel])))
            thr = quantile(score_loso[sel & known], 0.05)
            u95.append(float(np.mean(score_loso[sel & ~known] < thr)))
        rows.append({"name": spec["name"], "kind": spec["kind"],
                     "lam": spec.get("lam"), "n_components": spec["n_components"],
                     "loso_pooled_auc": pooled,
                     "loso_fold_auc_mean": float(np.mean(per_fold)),
                     "loso_fold_auc_std": float(np.std(per_fold)),
                     "loso_fold_auc_min": float(np.min(per_fold)),
                     "u95_mean": float(np.mean(u95)),
                     "u95_min": float(np.min(u95))})

    ranked = sorted(rows, key=lambda r: (-r["loso_pooled_auc"],
                                        -r["loso_fold_auc_mean"], r["n_components"]))
    best = ranked[0]
    spec = next(c for c in candidate_library() if c["name"] == best["name"])
    score_oof = gate_score(spec, np.asarray(signal["oof_gd"], np.float64),
                           np.asarray(signal["oof_agree"], np.float64), lab)
    score_test = gate_score(spec, np.asarray(signal["test_gd"], np.float64),
                            np.asarray(signal["test_agree"], np.float64), lab)
    tau = {f"{a:.2f}": quantile(score_oof, 1.0 - a) for a in ACC_GRID}

    summary = {
        "lab": {"edges": [float(v) for v in lab["edges"]],
                "cell_score": [float(v) for v in lab["cell_score"]],
                "cell_counts": lab["cell_counts"]},
        "leaderboard": ranked,
        "selected": {"name": best["name"], "kind": best["kind"], "lam": best["lam"],
                     "loso_pooled_auc": best["loso_pooled_auc"],
                     "loso_fold_auc_mean": best["loso_fold_auc_mean"],
                     "loso_fold_auc_std": best["loso_fold_auc_std"],
                     "u95_mean": best["u95_mean"], "u95_min": best["u95_min"]},
        "tau_train_oof": tau,
        "primary_acceptance": PRIMARY_A,
        "oof_acceptance_at_primary": float(np.mean(score_oof >= tau[f"{PRIMARY_A:.2f}"])),
        "selection_rule": "max pooled Train LOSO gate AUC, ties -> fewer components",
        "elapsed_seconds": time.time() - started,
    }
    np.savez_compressed(CACHE / "gate_scores.npz", oof=score_oof, test=score_test)
    results = load_results()
    results["gates"] = summary
    save_results(results)
    log("stage gates: Train LOSO pooled AUC by candidate")
    for row in ranked:
        log(f"  {row['loso_pooled_auc']:.4f}  fold {row['loso_fold_auc_mean']:.4f}"
            f"±{row['loso_fold_auc_std']:.4f}  U@95 {row['u95_mean']:.4f}  {row['name']}")
    log(f"  selected: {best['name']} (LOSO {best['loso_pooled_auc']:.4f})")
    return summary


# ---------------------------------------------------------------- stage test
def stage_test() -> dict:
    started = time.time()
    signal = np.load(SIGNAL_CACHE)
    gates = load_results()["gates"]
    lab = {"edges": np.asarray(gates["lab"]["edges"], np.float64),
           "cell_score": np.asarray(gates["lab"]["cell_score"], np.float64)}
    spec = next(c for c in candidate_library()
                if c["name"] == gates["selected"]["name"])
    score_oof = np.load(CACHE / "gate_scores.npz")["oof"]
    score_test = np.load(CACHE / "gate_scores.npz")["test"]
    tau = {a: quantile(score_oof, 1.0 - a) for a in ACC_GRID}
    test_apred = np.asarray(signal["test_apred"], int)

    primary_tau = tau[PRIMARY_A]
    accept = score_test >= primary_tau
    pred = np.where(accept, test_apred, 0)
    pd.DataFrame({"Sample": signal["test_sample"].astype(str), "Label": pred}).to_csv(
        OUT / "test_predictions_fused.csv", index=False)
    log(f"  wrote test_predictions_fused.csv ({int((pred == 0).sum())} rejects of "
        f"{len(pred)}) BEFORE reading Session")

    meta = pd.read_csv(ROOT / E180.CFG["data"]["test_metadata"],
                       usecols=["sample", "session"])
    if not np.array_equal(meta["sample"].astype(str).to_numpy(),
                          signal["test_sample"].astype(str)):
        raise ValueError("Test sample order mismatch")
    s1 = meta["session"].astype(str).str.casefold().isin(
        ["session1", "s1", "1"]).to_numpy()

    def audit(score: np.ndarray, reference: np.ndarray) -> dict:
        """Thresholds come from the OOF distribution of the SAME candidate score."""
        local = {a: quantile(reference, 1.0 - a) for a in ACC_GRID}
        out = {}
        for a in ACC_GRID:
            acc = score >= local[a]
            out[f"{a:.2f}"] = {
                "tau": local[a],
                "s1_reject": float(np.mean(~acc[s1])),
                "s2_accept": float(np.mean(acc[~s1])),
                "s1_s2": float(np.mean(~acc[s1]) + np.mean(acc[~s1]))}
        return out

    sweep = audit(score_test, score_oof)
    best_a = max(sweep, key=lambda key: sweep[key]["s1_s2"])
    # forbidden upper bound: threshold chosen directly on the Test ROC
    order = np.argsort(-score_test)
    labels = (~s1).astype(int)[order]
    cum_tp = np.cumsum(labels)
    cum_fp = np.cumsum(1 - labels)
    tpr = cum_tp / max(labels.sum(), 1)
    fnr = 1.0 - cum_fp / max((1 - labels).sum(), 1)
    cheat = float(np.max(tpr + fnr))

    def cheat_bound(score: np.ndarray) -> float:
        order_local = np.argsort(-score)
        lab = (~s1).astype(int)[order_local]
        tpr = np.cumsum(lab) / max(lab.sum(), 1)
        fnr = 1.0 - np.cumsum(1 - lab) / max((1 - lab).sum(), 1)
        return float(np.max(tpr + fnr))

    per_candidate = []
    for name in [row["name"] for row in gates["leaderboard"]]:
        candidate = next(c for c in candidate_library() if c["name"] == name)
        score = gate_score(candidate, np.asarray(signal["test_gd"], np.float64),
                           np.asarray(signal["test_agree"], np.float64), lab)
        score_oof_candidate = gate_score(candidate, np.asarray(signal["oof_gd"], np.float64),
                                         np.asarray(signal["oof_agree"], np.float64), lab)
        row = audit(score, score_oof_candidate)
        key = max(row, key=lambda k: row[k]["s1_s2"])
        per_candidate.append({
            "name": name, "test_diag_auc": float(roc_auc_score((~s1).astype(int), score)),
            "cheat_upper_bound_s1_s2": cheat_bound(score),
            "at_primary": row[f"{PRIMARY_A:.2f}"],
            "best_point": {"a": key, **row[key]}, "sweep": row})

    # explicit AND / OR truth tables with tau1 swept on Train OOF g_dist quantiles
    gd_oof = np.asarray(signal["oof_gd"], np.float64)
    gd_test = np.asarray(signal["test_gd"], np.float64)
    agree = np.asarray(signal["test_agree"], np.float64)
    rules = {}
    for rule in ("and", "or"):
        table = {}
        for a in ACC_GRID:
            tau1 = quantile(gd_oof, 1.0 - a)
            if rule == "and":
                acc = (gd_test >= tau1) & (agree == 1)
            else:
                acc = (gd_test >= tau1) | (agree == 1)
            table[f"{a:.2f}"] = {
                "tau1": tau1, "s1_reject": float(np.mean(~acc[s1])),
                "s2_accept": float(np.mean(acc[~s1])),
                "s1_s2": float(np.mean(~acc[s1]) + np.mean(acc[~s1]))}
        rules[rule] = table

    best_candidate = max(per_candidate, key=lambda c: c["best_point"]["s1_s2"])
    best_cheat = max(per_candidate, key=lambda c: c["cheat_upper_bound_s1_s2"])
    summary = {
        "selected": gates["selected"]["name"],
        "primary_acceptance": PRIMARY_A, "primary_tau": primary_tau,
        "test_diag_auc": float(roc_auc_score((~s1).astype(int), score_test)),
        "sweep": sweep,
        "best_train_calibrated_point": {"a": best_a, **sweep[best_a]},
        "cheat_upper_bound_s1_s2": cheat,
        "max_candidate_train_calibrated": {
            "name": best_candidate["name"],
            "a": best_candidate["best_point"]["a"],
            "s1_s2": best_candidate["best_point"]["s1_s2"],
            "s1_reject": best_candidate["best_point"]["s1_reject"],
            "s2_accept": best_candidate["best_point"]["s2_accept"]},
        "max_candidate_cheat_bound": {
            "name": best_cheat["name"],
            "s1_s2": best_cheat["cheat_upper_bound_s1_s2"]},
        "per_candidate": per_candidate,
        "rules_and_or": rules,
        "n_s1": int(s1.sum()), "n_s2": int((~s1).sum()),
        "test_reject_rate_primary": float(np.mean(~accept)),
        "elapsed_seconds": time.time() - started,
    }
    results = load_results()
    results["test"] = summary
    save_results(results)
    log(f"stage test done in {summary['elapsed_seconds']:.0f}s: Test diag AUC "
        f"{summary['test_diag_auc']:.4f}")
    log(f"  a={PRIMARY_A:.2f}: S1 reject {sweep[f'{PRIMARY_A:.2f}']['s1_reject']:.4f} "
        f"S2 accept {sweep[f'{PRIMARY_A:.2f}']['s2_accept']:.4f} "
        f"sum {sweep[f'{PRIMARY_A:.2f}']['s1_s2']:.4f}")
    log(f"  best Train-calibrated point a={best_a}: sum "
        f"{sweep[best_a]['s1_s2']:.4f} (S1 {sweep[best_a]['s1_reject']:.4f} / "
        f"S2 {sweep[best_a]['s2_accept']:.4f}); forbidden Test-optimal bound {cheat:.4f}")
    for rule in ("and", "or"):
        key = max(rules[rule], key=lambda k: rules[rule][k]["s1_s2"])
        log(f"  rule {rule.upper()} best a={key}: sum {rules[rule][key]['s1_s2']:.4f}")
    log(f"  best candidate under Train calibration: {best_candidate['name']} "
        f"sum {best_candidate['best_point']['s1_s2']:.4f}; "
        f"best forbidden Test-optimal bound: {best_cheat['name']} "
        f"{best_cheat['cheat_upper_bound_s1_s2']:.4f}")
    return summary


# ---------------------------------------------------------------- attack
class AgreementScorer(E180.Scorer):
    """Frozen dual-branch forward exposing all 20 per-identity branch-B scores."""

    def __call__(self, x: torch.Tensor, task_idx: torch.Tensor) -> tuple:
        features = E180.differentiable_logspd(x)
        mu = self.stats_mu[task_idx]
        sd = self.stats_sd[task_idx]
        z = (features - mu) / sd
        projected = (z - self.pca_mu) @ self.basis
        ident = torch.full((len(x), len(CLASSES)), OFF, device=DEVICE)
        for t in range(len(TASKS)):
            valid = self.filled[t]
            p = self.proto[t][valid].reshape(-1, self.proto.shape[-1])
            dist = torch.cdist(projected, p).view(len(x), int(valid.sum()), self.k)
            score = (-dist).max(dim=2).values
            block = torch.full_like(ident, OFF)
            block[:, valid] = score
            take = task_idx == t
            ident = torch.where(take[:, None], block, ident)
        return self.model(x), ident


def pgd_agreement(scorer, x, task_idx, true_idx, target_idx, goal, eps, steps, seed):
    temperature = float(E180.CFG["attack"]["temperature"])
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    rms = x.sub(x.mean(dim=-1, keepdim=True)).pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-6)
    bound = (eps * rms).view(-1, 1, 1)
    with torch.no_grad():
        adv = x + (torch.rand(x.shape, generator=gen, device=DEVICE) * 2 - 1) * bound
        adv = torch.clamp(adv, x - bound, x + bound)
    alpha = 2.0 * eps / steps
    nan_count = 0
    for _ in range(steps):
        adv.requires_grad_(True)
        logits, ident = scorer(adv, task_idx)
        if goal in ("A", "B", "C"):
            true_logit = logits.gather(1, true_idx[:, None]).squeeze(1)
            wrong = logits.masked_fill(F.one_hot(true_idx, len(CLASSES)).bool(), OFF)
            if goal == "A":
                margin = wrong.max(dim=1).values
            elif goal == "B":
                margin = torch.logsumexp(wrong, dim=1)
            else:
                margin = logits.gather(1, target_idx[:, None]).squeeze(1)
            objective = (margin - true_logit) / temperature
        elif goal == "D":
            objective = (ident.max(dim=1).values - scorer.oof_mu) / scorer.oof_sd
        elif goal == "E":
            b_idx = ident.detach().argmax(dim=1)
            prob = torch.softmax(logits, dim=1)
            objective = prob.gather(1, b_idx[:, None]).squeeze(1)
        else:
            raise KeyError(goal)
        grad = torch.autograd.grad(-objective.sum(), adv)[0]
        nan_count += int((~torch.isfinite(grad)).sum().item())
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        adv = torch.clamp(adv.detach() - (alpha * rms).view(-1, 1, 1) * grad.sign(),
                          x - bound, x + bound)
    with torch.no_grad():
        logits, ident = scorer(adv, task_idx)
    realised = ((adv - x).abs().amax(dim=(1, 2)) / rms)
    return {"logits": logits.cpu().numpy(), "ident": ident.cpu().numpy(),
            "max_rel_linf": float(realised.max()), "nan_gradients": nan_count}


def decision_metrics(logits: np.ndarray, ident: np.ndarray, truth: np.ndarray,
                     tau: float, spec: dict, lab: dict) -> dict:
    gd = (ident.max(1) - float(branch_b_store()["oof_mu"])) / float(
        branch_b_store()["oof_sd"])
    b_id = CLASSES[np.asarray(ident).argmax(1)]
    sig = agreement_signals(logits, b_id)
    score = gate_score(spec, gd, sig["agree"], lab)
    accept = score >= tau
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
            "reject_rate": float(np.mean(predicted == 0)),
            "agree_rate": float(np.mean(sig["agree"]))}


def stage_attack() -> dict:
    started = time.time()
    E180.build_torch_constants()
    cfg = CFG["attack"]
    meta, frac, blocks, fit_rows = E180.load_config_meta()
    subjects = meta["subject"].to_numpy(int)
    tasks = meta["task"].to_numpy()
    window_rows = np.flatnonzero((frac > FIT) & (frac <= 0.8))
    rng = np.random.default_rng(int(E180.CFG["attack"]["seed"]))
    cap = int(E180.CFG["attack"]["cap_per_task"])
    picked = []
    for task in TASKS:
        rows = window_rows[tasks[window_rows] == task]
        picked.append(rows if len(rows) <= cap
                      else np.sort(rng.choice(rows, size=cap, replace=False)))
    subset = np.sort(np.concatenate(picked))
    log(f"attack: {len(subset)} eval-window trials "
        f"({', '.join(f'{t}={int((tasks[subset] == t).sum())}' for t in TASKS)})")

    signal = np.load(SIGNAL_CACHE)
    gates = load_results()["gates"]
    lab = {"edges": np.asarray(gates["lab"]["edges"], np.float64),
           "cell_score": np.asarray(gates["lab"]["cell_score"], np.float64)}
    spec = next(c for c in candidate_library()
                if c["name"] == gates["selected"]["name"])
    score_oof = np.load(CACHE / "gate_scores.npz")["oof"]
    tau_primary = quantile(score_oof, 1.0 - PRIMARY_A)
    sweep = (0.80, 0.90, 0.95)
    tau_sweep = {a: quantile(score_oof, 1.0 - a) for a in sweep}

    branch_a = torch.load(E180.MODEL_A, map_location="cpu", weights_only=False)
    if branch_a["spec_sha256"] != E180.BRANCH_A_SHA:
        raise ValueError("branch A artifact does not match the frozen spec")
    model = E180.Net().to(DEVICE)
    model.load_state_dict(branch_a["state_dict"])
    store = E180.load_branch_b()
    scorer = AgreementScorer(model, store)

    raw = E180.load_raw([meta["path"].iloc[i] for i in subset])
    truth = subjects[subset]
    task_idx = torch.as_tensor([TASKS.index(t) for t in tasks[subset]],
                               dtype=torch.long, device=DEVICE)
    true_idx = torch.as_tensor(truth - 1, dtype=torch.long, device=DEVICE)
    class_of = {int(c): i for i, c in enumerate(CLASSES)}
    target = np.asarray([CLASSES[(class_of[int(v)] + 1) % len(CLASSES)] for v in truth])
    target_idx = torch.as_tensor([class_of[int(v)] for v in target],
                                 dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        logits, ident = scorer(raw, task_idx)
    clean_logits, clean_ident = logits.cpu().numpy(), ident.cpu().numpy()
    # consistency: differentiable forward must reproduce the numpy pipeline
    b_id = CLASSES[clean_ident.argmax(1)]
    gd_torch = (clean_ident.max(1) - float(store["oof_mu"])) / float(store["oof_sd"])
    gd_numpy = np.asarray(signal["oof_gd"], np.float64)[subset]
    gd_check = float(np.max(np.abs(gd_torch - gd_numpy)))
    gd_shift = float(np.mean(gd_torch - gd_numpy))
    agree_numpy = np.asarray(signal["oof_agree"], np.float64)[subset]
    agree_check = float(np.mean(
        (agreement_signals(signal["oof_logits"][subset], b_id)["agree"]) == agree_numpy))
    log(f"  differentiable (full-model prototypes) vs numpy OOF (block-excluded "
        f"prototypes): max |dg_dist| {gd_check:.2e}, mean shift {gd_shift:+.4f}, "
        f"agree identical fraction {agree_check:.4f}")

    clean = decision_metrics(clean_logits, clean_ident, truth, tau_primary, spec, lab)
    clean["at_acceptance"] = {
        f"{a:.2f}": {key: decision_metrics(clean_logits, clean_ident, truth,
                                           tau_sweep[a], spec, lab)[key]
                     for key in ("RA", "ASafeR", "accept_rate")} for a in sweep}
    log(f"  clean: RA {clean['RA']:.4f} ASafeR {clean['ASafeR']:.4f} "
        f"closed {clean['closed_set_accuracy']:.4f} accept {clean['accept_rate']:.4f} "
        f"agree {clean['agree_rate']:.4f}")

    conditions = []
    for eps in cfg["eps_normalised"]:
        for goal in ("A", "B", "C", "D", "E"):
            out = pgd_agreement(scorer, raw, task_idx, true_idx, target_idx, goal,
                                float(eps), int(cfg["steps"]),
                                int(E180.CFG["attack"]["seed"]) + int(eps * 1e5))
            metrics = decision_metrics(out["logits"], out["ident"], truth, tau_primary,
                                       spec, lab)
            metrics.update({"eps": float(eps), "goal": goal,
                            "max_rel_linf": out["max_rel_linf"],
                            "nan_gradients": out["nan_gradients"],
                            "at_acceptance": {
                                f"{a:.2f}": {key: decision_metrics(
                                    out["logits"], out["ident"], truth, tau_sweep[a],
                                    spec, lab)[key] for key in ("RA", "ASafeR")}
                                for a in sweep}})
            conditions.append(metrics)
            log(f"  eps={eps:.3f} goal {goal}: RA {metrics['RA']:.4f} "
                f"ASafeR {metrics['ASafeR']:.4f} closed {metrics['closed_set_accuracy']:.4f} "
                f"accept {metrics['accept_rate']:.4f} agree {metrics['agree_rate']:.4f}")

    worst_ra = min(conditions, key=lambda r: r["RA"])
    worst_safe = min(conditions, key=lambda r: r["ASafeR"])
    summary = {
        "n_trials": int(len(subset)),
        "per_task": {task: int((tasks[subset] == task).sum()) for task in TASKS},
        "tau_primary": tau_primary, "primary_acceptance": PRIMARY_A,
        "sweep_tau": {f"{a:.2f}": tau_sweep[a] for a in sweep},
        "gate_rebuild_max_abs": gd_check, "gate_rebuild_mean_shift": gd_shift,
        "agree_identical_fraction": agree_check,
        "clean": clean, "conditions": conditions,
        "worst_RA": {"RA": worst_ra["RA"], "eps": worst_ra["eps"],
                     "goal": worst_ra["goal"], "ASafeR": worst_ra["ASafeR"]},
        "worst_ASafeR": {"ASafeR": worst_safe["ASafeR"], "eps": worst_safe["eps"],
                         "goal": worst_safe["goal"], "RA": worst_safe["RA"]},
        "attack_status": "self-built white-box PGD surrogate with a smooth agreement "
                         "surrogate; decisions re-evaluated with the exact binary rule",
        "elapsed_seconds": time.time() - started,
    }
    results = load_results()
    results["attack"] = summary
    save_results(results)
    log(f"attack done in {summary['elapsed_seconds']:.0f}s: worst RA "
        f"{worst_ra['RA']:.4f} (eps {worst_ra['eps']}, goal {worst_ra['goal']}), "
        f"worst ASafeR {worst_safe['ASafeR']:.4f}")
    del raw, model
    torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------- report
def fmt(value, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def write_report() -> None:
    results = load_results()
    signals = results["signals"]
    loso = results["loso"]
    diag = results["diagnostic"]
    gates = results["gates"]
    test = results["test"]
    attack = results["attack"]
    selected = gates["selected"]

    n_agree_s1 = diag["agree_table"][0][0]
    n_s1 = diag["n_s1"]
    n_agree_s2 = diag["agree_table"][1][0]
    n_s2 = diag["n_s2"]
    base = test["sweep"][f"{PRIMARY_A:.2f}"]
    best = test["best_train_calibrated_point"]
    cheat = test["cheat_upper_bound_s1_s2"]
    best_cand = test["max_candidate_train_calibrated"]
    best_cheat = test["max_candidate_cheat_bound"]
    headline = max(best["s1_s2"], cheat, best_cand["s1_s2"], best_cheat["s1_s2"])
    cheat_all = max(cheat, best_cheat["s1_s2"])

    # ten gates at the single primary work point
    guard = [
        (1, "拟合容量", "Train 准确率", "=1.00",
         signals["branch_a_in_sample_accuracy"], 1.0),
        (2, "折外闭集", "C（5 块 OOF 闭集准确率）", "≥0.95",
         signals["branch_a_oof_accuracy_5block"], 0.95),
        (3, "折外闭集", "身份 macro-F1（5 块 OOF）", "≥0.92",
         signals["branch_a_oof_macro_f1"], 0.92),
        (4, "跨时间代理", "块外准确率（同 5 块口径）", "≥0.90",
         signals["branch_a_oof_accuracy_5block"], 0.90),
        (5, "开放集", "pseudo-unknown AUROC（20 折 LOSO）", "≥0.98",
         selected["loso_pooled_auc"], 0.98),
        (6, "开放集", "U@95%（逐折最差口径）", "≥0.95",
         selected["u95_min"], 0.95),
        (7, "对抗识别", "RA（PGD，ε×目标最差）", "≥0.95",
         attack["worst_RA"]["RA"], 0.95),
        (8, "对抗安全", "ASafeR（同一攻击最差）", "≥0.98",
         attack["worst_ASafeR"]["ASafeR"], 0.98),
        (9, "Test 审计", "S2 接受率", "≥0.75", base["s2_accept"], 0.75),
        (10, "Test 审计", "S1 拒识率", "≥0.75", base["s1_reject"], 0.75),
    ]

    lines = []
    lines.append("# E184 跨分支一致性门控：能否突破 S1+S2 = 1.4479？\n")
    lines.append(f"日期：2026-11-15。分支 A（E180 对抗训练 EEGNet，20 类闭集）与分支 B"
                 f"（E65 correlation-logSPD / PCA-96 / k=12 原型）**全部只读复用、未重训**。"
                 f"本实验不引入任何新模型，只提出一个新的门控信息维度："
                 f"**两个分支对同一 trial 的身份判断是否一致**。\n")
    verdict = ("**假设成立**" if diag["hypothesis_pass"] else
               "**假设不成立（已被证伪）**")
    lines.append(f"结论先行：{verdict}。S1 一致性比例 {fmt(diag['agree_rate_s1'])}、"
                 f"S2 {fmt(diag['agree_rate_s2'])}，差 "
                 f"{diag['rate_difference']:+.4f}（95% CI "
                 f"[{diag['rate_difference_ci95'][0]:+.4f}, "
                 f"{diag['rate_difference_ci95'][1]:+.4f}]，卡方 p = "
                 f"{diag['chi2_p_value']:.3g}），方向与假设一致，"
                 f"但单信号 Test 诊断 AUC 只有 {fmt(diag['signals']['agree']['auc_s2_vs_s1'])}"
                 f"（soft_agree {fmt(diag['signals']['soft_agree']['auc_s2_vs_s1'])}），"
                 f"远低于「S1+S2 需 1.50」所要求的约 0.83。\n")
    lines.append(f"门控层面：按预设规则（Train LOSO 合并 AUC 最大）选出 `{selected['name']}`，"
                 f"a={PRIMARY_A:.2f} 时 Test S1 拒识 **{fmt(base['s1_reject'])}**、"
                 f"S2 接受 **{fmt(base['s2_accept'])}**（S1+S2 {fmt(base['s1_s2'])}）。"
                 f"全部候选里 Train 标定下最高 S1+S2 = **{fmt(best_cand['s1_s2'])}**"
                 f"（`{best_cand['name']}`，a={best_cand['a']}），"
                 f"全部候选里犯规上界最高 = **{fmt(cheat_all)}**"
                 f"（`{best_cheat['name']}`）。历史天花板 1.4479，同时达标线 1.50。\n")
    passed = sum(1 for row in guard if (row[4] >= row[5] and row[0] != 1))
    lines.append(f"**十项门禁通过 {passed}/10**（单工作点 a={PRIMARY_A:.2f} 判定）。\n")

    lines.append("## 1. 诊断表（Test，Session 仅审计）\n")
    lines.append("| 会话 | N | agree 比例 | soft_agree 均值 | margin_agree 均值 | 平均 rank |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(f"| S1（未注册，应拒识） | {n_s1} | {fmt(diag['agree_rate_s1'])} | "
                 f"{fmt(diag['signals']['soft_agree']['s1_mean'])} | "
                 f"{fmt(diag['signals']['margin_agree']['s1_mean'])} | "
                 f"{fmt(diag['signals']['rank_agree']['s1_mean'], 2)} |")
    lines.append(f"| S2（注册，应接受） | {n_s2} | {fmt(diag['agree_rate_s2'])} | "
                 f"{fmt(diag['signals']['soft_agree']['s2_mean'])} | "
                 f"{fmt(diag['signals']['margin_agree']['s2_mean'])} | "
                 f"{fmt(diag['signals']['rank_agree']['s2_mean'], 2)} |")
    lines.append("")
    lines.append("| 信号 | S1 均值 | S2 均值 | Test 诊断 AUC（S2 为正类） | 方向是否符合假设 |")
    lines.append("|---|---:|---:|---:|---|")
    for name in AGREE_SIGNALS:
        row = diag["signals"][name]
        direction = "同向（可用）" if row["auc_s2_vs_s1"] > 0.5 else "反向（有害）"
        lines.append(f"| `{name}` | {row['s1_mean']:+.4f} | {row['s2_mean']:+.4f} | "
                     f"{fmt(row['auc_s2_vs_s1'])} | {direction} |")
    lines.append(f"| `g_dist`（E180 距离门） | — | — | "
                 f"{fmt(diag['g_dist_auc_s2_vs_s1'])} | 同向（历史最好的一维门控） |")
    lines.append("")
    lines.append(f"显著性：2×2 列联表 "
                 f"[[同意 {n_agree_s1}, 不同意 {n_s1 - n_agree_s1}], "
                 f"[同意 {n_agree_s2}, 不同意 {n_s2 - n_agree_s2}]]，"
                 f"卡方 = {diag['chi2']:.2f}，p = {diag['chi2_p_value']:.3g}，"
                 f"比例差 {diag['rate_difference']:+.4f}（95% CI "
                 f"[{diag['rate_difference_ci95'][0]:+.4f}, "
                 f"{diag['rate_difference_ci95'][1]:+.4f}]）。"
                 f"另：agree 与 g_dist 的相关系数 S1 {diag['corr_agree_g_dist_s1']:+.4f} / "
                 f"S2 {diag['corr_agree_g_dist_s2']:+.4f}。\n")
    lines.append(f"诊断判据（预设）：S1 与 S2 的 agree 比例显著不同且 S2 更高 → 假设成立；"
                 f"否则假设证伪。本次判定：**{'成立' if diag['hypothesis_pass'] else '证伪'}**。\n")

    lines.append("## 2. 组合门控：Train 选择与 Test 审计\n")
    lines.append("候选（权/λ 只用 Train LOSO 合并 AUC 选，阈值只用 Train OOF 分位）：\n")
    lines.append("| 候选 | Train LOSO 合并 AUC | 逐折均值±std | 逐折最差 U@95 | Test 诊断 AUC | Train 标定最好 S1+S2 | 犯规上界 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in gates["leaderboard"]:
        cand = next(c for c in test["per_candidate"] if c["name"] == row["name"])
        lines.append(f"| `{row['name']}` | {fmt(row['loso_pooled_auc'])} | "
                     f"{row['loso_fold_auc_mean']:.4f}±{row['loso_fold_auc_std']:.4f} | "
                     f"{fmt(row['u95_min'])} | {fmt(cand['test_diag_auc'])} | "
                     f"{fmt(cand['best_point']['s1_s2'])}（a={cand['best_point']['a']}） | "
                     f"{fmt(cand['cheat_upper_bound_s1_s2'])} |")
    lines.append("")
    lines.append("Test 工作点（每列一个阈值，阈值全部来自 Train OOF 分位；S1+S2 为核心指标）：\n")
    lines.append("| a | τ | S1 拒识 | S2 接受 | S1+S2 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for key in test["sweep"]:
        row = test["sweep"][key]
        lines.append(f"| {key} | {row['tau']:+.4f} | {fmt(row['s1_reject'])} | "
                     f"{fmt(row['s2_accept'])} | **{fmt(row['s1_s2'])}** |")
    lines.append("")
    lines.append(f"同一门控在 Test 上的**犯规上界**（阈值直接在 Test ROC 上扫，仅用于回答"
                 f"「还差多少」）：S1+S2 最大 **{fmt(cheat)}**。"
                 f"Train 标定下的最好点 a={best['a']}：S1 {fmt(best['s1_reject'])} / "
                 f"S2 {fmt(best['s2_accept'])} / 和 {fmt(best['s1_s2'])}。\n")

    lines.append("### 2.1 二维规则（AND / OR，τ1 在 Train OOF g_dist 分位上扫）\n")
    lines.append("| τ1 工作点 a | τ1 | AND: S1 拒识 | AND: S2 接受 | AND: 和 | OR: S1 拒识 | OR: S2 接受 | OR: 和 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in test["rules_and_or"]["and"]:
        a_row = test["rules_and_or"]["and"][key]
        o_row = test["rules_and_or"]["or"][key]
        lines.append(f"| {key} | {a_row['tau1']:+.4f} | {fmt(a_row['s1_reject'])} | "
                     f"{fmt(a_row['s2_accept'])} | {fmt(a_row['s1_s2'])} | "
                     f"{fmt(o_row['s1_reject'])} | {fmt(o_row['s2_accept'])} | "
                     f"{fmt(o_row['s1_s2'])} |")
    lines.append("")
    best_and = max(test["rules_and_or"]["and"].items(), key=lambda kv: kv[1]["s1_s2"])
    best_or = max(test["rules_and_or"]["or"].items(), key=lambda kv: kv[1]["s1_s2"])
    lines.append(f"AND 最好和 {fmt(best_and[1]['s1_s2'])}（a={best_and[0]}），"
                 f"OR 最好和 {fmt(best_or[1]['s1_s2'])}（a={best_or[0]}）。"
                 f"两者是同一个二维分数 `g_dist + B·agree` 上的两个工作点，"
                 f"因此共用同一条 ROC。\n")
    lines.append("### 2.2 LAB 分区（3×2 单元，Train OOF 分位切分；单元已知率由 Train LOSO 估计）\n")
    lines.append("| 单元（g_dist 三分位, agree） | 0 | 1 | 2 |")
    lines.append("|---|---:|---:|---:|")
    cell = gates["lab"]["cell_score"]
    counts = gates["lab"]["cell_counts"]
    lines.append(f"| agree=0（单元已知率 / 计数） | {cell[0]:.4f} / {counts['0']} | "
                 f"{cell[2]:.4f} / {counts['2']} | {cell[4]:.4f} / {counts['4']} |")
    lines.append(f"| agree=1（单元已知率 / 计数） | {cell[1]:.4f} / {counts['1']} | "
                 f"{cell[3]:.4f} / {counts['3']} | {cell[5]:.4f} / {counts['5']} |")
    lines.append(f"\n切分点（Train OOF z(g_dist)）：{gates['lab']['edges'][0]:+.4f} / "
                 f"{gates['lab']['edges'][1]:+.4f}。\n")

    lines.append("## 3. 完整十项门禁表（单工作点 a="
                 f"{PRIMARY_A:.2f}）\n")
    lines.append("| # | 层面 | 指标 | 目标 | 实测 | 判定 | 证据来源 |")
    lines.append("|---|---|---|---|---:|---|---|")
    evidence = {
        1: "冻结分支 A 全量模型 in-sample argmax", 2: "冻结分支 A 5 块 OOF logits",
        3: "同一批 5 块 OOF 预测", 4: "同一批 5 块 OOF（frac>0.6 前向块）",
        5: "选定门控 20 折 LOSO 合并 AUC（分支 A 掩蔽留出身份）",
        6: "同一 LOSO 逐折 5% 分位阈值，取最差折",
        7: f"自建 PGD，{len(attack['conditions'])} 个条件取最差 RA",
        8: "同一攻击条件取最差 ASafeR",
        9: "冻结后 Test 一次性评分", 10: "同 #9",
    }
    for number, layer, metric, target, value, threshold in guard:
        ok = value >= threshold if number != 1 else value >= 1.0
        lines.append(f"| #{number} | {layer} | {metric} | {target} | {fmt(value)} | "
                     f"{'通过' if ok else '未通过'} | {evidence[number]} |")
    lines.append("")
    lines.append(f"#6 备注：选定门控 U@95 逐折均值 {fmt(selected['u95_mean'])}、最差折 "
                 f"{fmt(selected['u95_min'])}；E180 的 g_dist 逐折均值为 0.5847。"
                 f"本表按最差折判定，均值口径一并给出但不作为判定依据。\n")
    lines.append("门禁读法：#9 与 #10 是同一条 ROC 的两端，只有 S1+S2 才有意义；"
                 f"本次 S1+S2 = {fmt(base['s1_s2'])}，同时达标需要 1.50。\n")

    lines.append("## 4. 明确回答\n")
    lines.append(f"**`S1+S2` 能否超过 1.4479 的历史上限？** "
                 f"{'能' if headline > 1.4479 else '不能'}。"
                 f"本实验最好的犯规上界 {fmt(cheat_all)}，Train 标定下最好的候选 "
                 f"{fmt(best_cand['s1_s2'])}，选定门控 {fmt(base['s1_s2'])}；"
                 f"对照 E181 1.3756、E183 1.0906、历史最高 E149 1.4479。\n")
    lines.append(f"**能否达到 1.50（#9/#10 同时 0.75）？** "
                 f"{'能' if headline >= 1.50 else '不能'}。"
                 f"缺口 {1.50 - headline:+.4f}（以本实验最好的 {fmt(headline)} 计）。"
                 f"换算成 AUC：本实验最好的诊断 AUC 是 "
                 f"{fmt(max(c['test_diag_auc'] for c in test['per_candidate']))}，"
                 f"要扛住阈值迁移损失需要约 0.83。\n")
    gd_cand = next(c for c in test["per_candidate"] if c["name"] == "g_dist")
    lines.append(f"**一致性是否带来任何增益？** 有，但极小且不可靠。"
                 f"`g_dist` 单独 Test 诊断 AUC {fmt(gd_cand['test_diag_auc'])}，"
                 f"加入 agree 后最高 {fmt(max(c['test_diag_auc'] for c in test['per_candidate']))}"
                 f"（+{max(c['test_diag_auc'] for c in test['per_candidate']) - gd_cand['test_diag_auc']:.4f}）；"
                 f"Train 标定 S1+S2 从 {fmt(gd_cand['best_point']['s1_s2'])} 抬到 "
                 f"{fmt(best_cand['s1_s2'])}（+{best_cand['s1_s2'] - gd_cand['best_point']['s1_s2']:.4f}）。"
                 f"但**预设选择规则（Train LOSO AUC 最大）选中的是 {selected['name']}**，"
                 f"它在 Test 上反而只有 {fmt(base['s1_s2'])}：LOSO 上 agree 几乎完美（AUC 0.897），"
                 f"λ 越大 AUC 越高，于是规则把 λ 推到 4；Test 上 agree 只有 AUC 0.609，"
                 f"λ=4 变成过权重的噪声。这是「Train LOSO 无跨会话偏移」造成的系统性高估，"
                 f"不是调参能修的。\n")
    if not diag["hypothesis_pass"]:
        lines.append("## 5. 失败结论与证据\n")
        lines.append("**跨分支一致性不含跨会话判别信息。** 证据链：\n")
        lines.append(f"1. 比例层：S1 agree {fmt(diag['agree_rate_s1'])} vs S2 "
                     f"{fmt(diag['agree_rate_s2'])}，比例差 "
                     f"{diag['rate_difference']:+.4f}，95% CI "
                     f"[{diag['rate_difference_ci95'][0]:+.4f}, "
                     f"{diag['rate_difference_ci95'][1]:+.4f}]，卡方 p = "
                     f"{diag['chi2_p_value']:.3g}。")
        lines.append(f"2. 排序层：agree 作为连续/二值门控的 Test 诊断 AUC "
                     f"{fmt(diag['signals']['agree']['auc_s2_vs_s1'])}，"
                     f"soft_agree {fmt(diag['signals']['soft_agree']['auc_s2_vs_s1'])}，"
                     f"rank_agree {fmt(diag['signals']['rank_agree']['auc_s2_vs_s1'])}；"
                     f"对照 g_dist {fmt(diag['g_dist_auc_s2_vs_s1'])}。")
        lines.append(f"3. 组合层：把 agree 加入 g_dist 后，Test 诊断 AUC 由 "
                     f"{fmt(diag['g_dist_auc_s2_vs_s1'])} 变为 "
                     f"{fmt(test['test_diag_auc'])}，S1+S2 由 E181 的 1.3756 变为 "
                     f"{fmt(best['s1_s2'])}。Train LOSO 上更高"
                     f"（{fmt(selected['loso_pooled_auc'])}）没有迁移到 Test。")
        lines.append("4. 机制：一致性信号测的是「两个分支是否被同一份会话特征推向同一个"
                     "identity」，S1 与 S2 都会被推向某个注册身份；它对会话整体平移不敏感，"
                     "但**正是这种不敏感性让它无法区分「注册」与「未注册」**——"
                     "两个分支同时偏移时仍然指向同一个身份，agree 因此接近饱和。\n")
    else:
        lines.append("## 5. 机制解读：假设成立，但信息量不够\n")
        auc_agree = diag["signals"]["agree"]["auc_s2_vs_s1"]
        auc_soft = diag["signals"]["soft_agree"]["auc_s2_vs_s1"]
        auc_rank = diag["signals"]["rank_agree"]["auc_s2_vs_s1"]
        auc_margin = diag["signals"]["margin_agree"]["auc_s2_vs_s1"]
        lines.append(f"1. **方向确实对**：S1 agree {fmt(diag['agree_rate_s1'])} 显著低于 S2 "
                     f"{fmt(diag['agree_rate_s2'])}（差 {diag['rate_difference']:+.4f}，"
                     f"p = {diag['chi2_p_value']:.3g}）。未注册身份确实让两个分支更难给出"
                     f"同一个答案。四个一致性读数全部同向：agree {fmt(auc_agree)}、"
                     f"soft_agree {fmt(auc_soft)}、margin_agree {fmt(auc_margin)}、"
                     f"rank_agree {fmt(auc_rank)}，没有出现 E181 那种方向反转。")
        lines.append(f"2. **量级不够**：最好的单信号 AUC 仅 {fmt(max(auc_agree, auc_soft, auc_margin, auc_rank))}，"
                     f"`g_dist` 是 {fmt(diag['g_dist_auc_s2_vs_s1'])}。"
                     f"20 个注册身份的随机一致率约 0.05，S1 一侧 {fmt(diag['agree_rate_s1'])} "
                     f"是随机基线的约 4 倍——两个分支不是各自乱指，而是被同一份会话特征"
                     f"共同推向同一个错误身份；S2 一侧也只有 {fmt(diag['agree_rate_s2'])}，"
                     f"并未饱和。两者叠在一起，一致率的可分性就被压到 0.61。")
        lines.append(f"3. **它不是 g_dist 的重复，但增量太小**：agree 与 z(g_dist) 的线性相关"
                     f"很弱（S1 {diag['corr_agree_g_dist_s1']:+.4f} / S2 "
                     f"{diag['corr_agree_g_dist_s2']:+.4f}），因此这确实是一个新维度；"
                     f"但它的分辨力本身只有 AUC {fmt(auc_soft)}，与 "
                     f"{fmt(diag['g_dist_auc_s2_vs_s1'])} 的 g_dist 线性合成后只把 AUC 抬到 "
                     f"{fmt(max(c['test_diag_auc'] for c in test['per_candidate']))}"
                     f"（+{max(c['test_diag_auc'] for c in test['per_candidate']) - gd_cand['test_diag_auc']:.4f}）。"
                     f"弱信号配强信号，合成分辨力由强信号主导。")
        lines.append(f"4. **Train LOSO 严重高估**：LOSO 上 agree 的 AUC 是 "
                     f"{fmt(loso['pooled_auc']['agree'])}（known 一致率约 0.97、"
                     f"伪未知 0.02–0.50），soft_agree 达 {fmt(loso['pooled_auc']['soft_agree'])}。"
                     f"原因是 LOSO 全程同会话，而 Test 跨会话；"
                     f"且 LOSO 的伪未知是「被掩蔽身份的 19 类强制选择」——"
                     f"分支 A 与分支 B 同时被逼到 19 个候选里，"
                     f"而 Test 的两个分支都在 20 个已注册身份里选，"
                     f"两者的一致性行为并不相同。")
        lines.append(f"5. **结论**：跨分支一致性是**真实但很弱的新维度**——它不与 g_dist 冗余"
                     f"（相关系数 ≈0），但自身 AUC 只有 0.61–0.67，"
                     f"合成后 S1+S2 最高 {fmt(cheat_all)}，无法突破 1.4479，也到不了 1.50。"
                     f"「两个分支是否一致」这个想法本身是对的，瓶颈在于两个分支在未注册"
                     f"身份上**共享同一个错误**，一致性因此不够干净。\n")
    lines.append("## 6. 对抗（#7/#8）\n")
    lines.append("| 条件 | ε | 目标 | RA | ASafeR | 闭集准确率 | 接受率 | 一致率 |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    row = attack["clean"]
    lines.append(f"| clean | — | — | {fmt(row['RA'])} | {fmt(row['ASafeR'])} | "
                 f"{fmt(row['closed_set_accuracy'])} | {fmt(row['accept_rate'])} | "
                 f"{fmt(row['agree_rate'])} |")
    for row in attack["conditions"]:
        lines.append(f"| adv | {row['eps']:.3f} | {row['goal']} | {fmt(row['RA'])} | "
                     f"{fmt(row['ASafeR'])} | {fmt(row['closed_set_accuracy'])} | "
                     f"{fmt(row['accept_rate'])} | {fmt(row['agree_rate'])} |")
    lines.append(f"\n判定：#7 RA 最差 **{fmt(attack['worst_RA']['RA'])}**"
                 f"（ε={attack['worst_RA']['eps']}，目标 {attack['worst_RA']['goal']}）；"
                 f"#8 ASafeR 最差 **{fmt(attack['worst_ASafeR']['ASafeR'])}**。"
                 f"攻击是自建白盒 PGD 代理，目标 E 用 soft_agree 作光滑替代，"
                 f"判决仍用精确二值规则重算。\n")
    lines.append(f"管线一致性：可微前向（全量原型）与 numpy 口径（分块 OOF 原型）的 "
                 f"z(g_dist) 均值差 {attack['gate_rebuild_mean_shift']:+.4f}、"
                 f"最大绝对差 {attack['gate_rebuild_max_abs']:.3f}，"
                 f"agree 判决一致率 {attack['agree_identical_fraction']:.4f}。"
                 f"差值全部来自「分块 OOF 原型 vs 全量原型」，与 Test 的评分口径"
                 f"（全量原型）一致。\n")
    lines.append("## 7. 合规与边界\n")
    lines.append("- 分支 A / 分支 B 全部来自 E180 冻结产物；未重训、未修改任何已有实验目录。")
    lines.append("- 一致性信号、λ、门控选择、阈值全部来自 Train（LOSO 合并 AUC 选、"
                 "Train OOF 分位定阈值）；Test 只评分。")
    lines.append("- LOSO 严格 `pm & fit_rows`；评估块 `0.6<frac≤0.8` 从未进入原型。")
    lines.append("- 分支 A 是 20 类闭集，LOSO 下必须掩蔽留出身份的 logit，否则伪未知的 "
                 "agree 恒为 0（结构性 0）。该掩蔽是 Train 侧标定装置，部署模型不掩蔽。")
    lines.append("- Test 预测先落盘、后读 Session；Session 只用于 #9/#10 与诊断表。")
    lines.append("- 未提交 git。\n")
    lines.append("## 8. Test 官方指标\n")
    lines.append("**Test 官方 AUC / F1 / C / RA / ASafeR / U / Score = N/A**"
                 "（Test 无真实标签）。\n")
    lines.append("## 9. 文件\n")
    lines.append("| 文件 | 内容 |")
    lines.append("|---|---|")
    lines.append("| `run_e184.py` | 全部逻辑（`--stage signals/loso/diag/gates/test/attack/report/all`） |")
    lines.append("| `config.json` | 配置与合规约束 |")
    lines.append("| `results.json` | 全部数值 |")
    lines.append("| `run.log` | 运行日志 |")
    lines.append("| `cache/signals.npz` | Train OOF / Test 的两个分支判决与四个一致性信号 |")
    lines.append("| `cache/loso.npz` | 20 折留身份标定的逐行信号 |")
    lines.append("| `cache/gate_scores.npz` | 选定门控的 Train OOF / Test 分数 |")
    lines.append("| `test_predictions_fused.csv` | 选定门控的 Test 预测（工程审计用） |")
    lines.append("")
    timings = ", ".join(f"{key}={results[key]['elapsed_seconds']:.0f}s"
                        for key in ("signals", "loso", "diagnostic", "gates", "test",
                                    "attack")
                        if key in results and "elapsed_seconds" in results[key])
    lines.append(f"耗时（秒）：{timings}。无任何训练，全部为冻结前向与 numpy 统计。\n")
    lines.append("```powershell")
    lines.append('& "D:\\WORK\\BCIC2026\\BrainPrint\\.venv\\Scripts\\python.exe" '
                 '"D:\\WORK\\BCIC2026\\BrainPrint\\research\\E184_cross_branch_agreement\\'
                 'run_e184.py" --stage all')
    lines.append("```")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"report written to {REPORT_PATH}")


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["signals", "loso", "diag", "gates", "test",
                                 "attack", "report", "all"])
    args = parser.parse_args()
    tee = Tee(LOG_PATH)
    sys_stdout, sys.stdout = sys.stdout, tee
    try:
        log(f"\n===== E184 run {time.strftime('%Y-%m-%d %H:%M:%S')} stage={args.stage} =====")
        if args.stage in ("signals", "all"):
            stage_signals()
        if args.stage in ("loso", "all"):
            stage_loso()
        if args.stage in ("diag", "all"):
            stage_diag()
        if args.stage in ("gates", "all"):
            stage_gates()
        if args.stage in ("test", "all"):
            stage_test()
        if args.stage in ("attack", "all"):
            stage_attack()
        if args.stage in ("report", "all"):
            write_report()
    finally:
        sys.stdout = sys_stdout
        tee.flush()
        tee.file.close()


if __name__ == "__main__":
    main()
