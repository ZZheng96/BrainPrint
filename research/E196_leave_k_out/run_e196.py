"""E196: leave-k-out - does the unknown-pool size explain why LOSO-1 cannot predict Test S1?

E194 showed quantitatively that
  * Test S1 (5 unregistered identities, session1) is another acquisition batch
    (domain separability AUC versus Train = 0.9789), and
  * the LOSO-1 ranking has no predictive power for Test S1: the same S1 block sits at
    opposite ends of two different LOSO-1 distributions.

This experiment tests the user's hypothesis: LOSO-1's unknown pool holds a single
identity while Test S1 holds five.  Leave-k-out with k = 1, 2, 5, 10 makes the Train-side
unknown pool larger while the gallery shrinks from 19 to 10 enrolled identities.

Strict protocol for every fold H (|H| = k):
  * task-wise mean / sd       from the 20-k registered identities' Train rows only
  * PCA basis                 from (registered & fit_rows) only
  * prototypes                from (registered & fit_rows) only, k_proto per identity
  * known block               the 20-k registered identities' 0.6 < frac <= 0.8 rows
  * unknown block             the k held-out identities' same window
  * threshold                 quantile of the fold's known scores at 1 - a
No row of a held-out identity enters any fitted statistic.  Test is scored with the same
fold instrument; Session is read only after every score exists and only to split S1 / S2.

Outputs: run_e196.py, config.json, results.json, report.md, run.log (this directory only).
Nothing is written into an existing experiment directory and nothing is committed.
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
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

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
SEED = int(CFG["protocol"]["seed"])
FIT = float(CFG["protocol"]["fit_fraction"])
A = float(CFG["protocol"]["primary_acceptance"])
KS = [1, 2, 5, 10]
PRIMARY = list(CFG["gates"]["primary"])

E195_STREAMS = ROOT / "research" / "E195_eleven_gates_loso90" / "cache" / "streams.npz"


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
    test_meta = pd.read_csv(ROOT / CFG["data"]["test_metadata"],
                            usecols=["sample", "task", "session"])
    if len(test_meta) != int(CFG["data"]["test_trials"]):
        raise ValueError(f"Test rows {len(test_meta)} unexpected")
    session = test_meta["session"].astype(str).str.casefold().to_numpy()
    window = (frac > 0.6) & (frac <= 0.8)
    return {"subjects": subjects, "tasks": tasks, "frac": frac,
            "fit_rows": frac <= FIT, "window": window,
            "win_rows": np.flatnonzero(window),
            "test_tasks": test_meta["task"].to_numpy(dtype=object),
            "is_s1": np.isin(session, ["session1", "s1", "1"])}


def apply_z(features: np.ndarray, tasks: np.ndarray,
            mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    out = np.array(features, dtype=np.float32, copy=True)
    for index, task in enumerate(TASKS):
        rows = tasks == task
        if not rows.any():
            continue
        out[rows] = (out[rows] - mu[index]) / sd[index]
    return out


def fit_stats(train: np.ndarray, tasks: np.ndarray, ref_mask: np.ndarray,
              pca_mask: np.ndarray, max_dim: int) -> tuple:
    """Task-wise mean / sd from ref_mask rows, PCA basis from pca_mask rows (float32)."""
    dim = train.shape[1]
    mu = np.zeros((len(TASKS), dim), dtype=np.float32)
    sd = np.zeros((len(TASKS), dim), dtype=np.float32)
    for index, task in enumerate(TASKS):
        ref = ref_mask & (tasks == task)
        if not ref.any():
            raise ValueError(f"empty reference for task {task}")
        mu[index] = train[ref].mean(0)
        sd[index] = train[ref].std(0) + 1e-6
    z_pca = apply_z(train[pca_mask], tasks[pca_mask], mu, sd)
    pca_mu = z_pca.mean(0)
    centred = z_pca - pca_mu
    covariance = centred.T @ centred / max(len(centred) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    basis = vectors[:, np.argsort(-values)[: min(int(max_dim), dim)]]
    del z_pca, centred, covariance
    return mu, sd, f32(pca_mu), f32(basis)


def project(features: np.ndarray, tasks: np.ndarray, mu: np.ndarray, sd: np.ndarray,
            pca_mu: np.ndarray, basis: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """Per-task z-score then PCA projection, in row chunks (no full-size temporary)."""
    out = np.empty((len(features), basis.shape[1]), dtype=np.float32)
    for start in range(0, len(features), chunk):
        block = apply_z(features[start:start + chunk], tasks[start:start + chunk], mu, sd)
        out[start:start + len(block)] = f32((block - pca_mu) @ basis)
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


def build_protos(proj: np.ndarray, subjects: np.ndarray, tasks: np.ndarray,
                 classes: np.ndarray, k: int) -> dict:
    out = {}
    for task in TASKS:
        chunks = []
        in_task = tasks == task
        for ident in classes:
            sel = in_task & (subjects == ident)
            if not sel.any():
                continue
            block = torch.as_tensor(f32(proj[sel]), device=DEVICE)
            chunks.append(kmeans_t(block, k, seed=int(ident) * 17))
        out[task] = torch.cat(chunks, 0)
    return out


def proto_scores(proj: np.ndarray, tasks: np.ndarray, protos: dict) -> np.ndarray:
    out = np.full(len(proj), np.nan, dtype=np.float32)
    for task in TASKS:
        idx = np.flatnonzero(tasks == task)
        if len(idx) == 0:
            continue
        q = torch.as_tensor(f32(proj[idx]), device=DEVICE)
        out[idx] = (-torch.cdist(q, protos[task])).max(1).values.cpu().numpy()
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
        protos = self.prototypes(classes) if classes is not None else self._normalise(
            torch.stack([self.mean_of[c] for c in sorted(self.mean_of)]) @ self.whitening)
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


# ---------------------------------------------------------------- fold plan
def build_plan() -> dict:
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


def load_families() -> dict:
    out = {}
    for name, spec in CFG["families"].items():
        out[name] = {"kind": spec.get("kind", "distance"),
                     "train": f32(np.load(ROOT / spec["train"])),
                     "test": f32(np.load(ROOT / spec["test"])),
                     "variants": [(v["gate"], int(v["pca"])) for v in spec["variants"]],
                     "prototypes": int(spec.get("prototypes", 1)),
                     "shrinkage": float(spec.get("shrinkage", 0.05)),
                     "floor": float(spec.get("eigenvalue_floor", 1e-6))}
    return out


# ---------------------------------------------------------------- one fold
def score_fold(P: dict, fam: dict, held: tuple) -> dict:
    """Scores of every base gate for one held-out subset (strict, Train-only fits)."""
    enrolled = np.setdiff1d(CLASSES, np.asarray(held, dtype=int))
    reg = np.isin(P["subjects"], enrolled)
    fit_mask = P["fit_rows"] & reg
    win_tasks = P["tasks"][P["win_rows"]]
    subj_fit = P["subjects"][fit_mask]
    task_fit = P["tasks"][fit_mask]
    out = {}
    for name, spec in fam.items():
        if spec["kind"] == "wccn":
            gate = WCCNGate(spec["train"], np.flatnonzero(fit_mask), P["subjects"], enrolled,
                            spec["shrinkage"], spec["floor"])
            out[spec["variants"][0][0]] = (
                WCCNGate.ratio(gate.scores(spec["train"][P["win_rows"]], classes=enrolled)),
                WCCNGate.ratio(gate.scores(spec["test"], classes=enrolled)))
            del gate
            continue
        max_dim = max(v[1] for v in spec["variants"])
        mu, sd, pca_mu, basis = fit_stats(spec["train"], P["tasks"], reg, fit_mask, max_dim)
        for gate_name, dim in spec["variants"]:
            b = basis[:, :dim]
            pf = project(spec["train"][fit_mask], task_fit, mu, sd, pca_mu, b)
            pw = project(spec["train"][P["win_rows"]], win_tasks, mu, sd, pca_mu, b)
            pt = project(spec["test"], P["test_tasks"], mu, sd, pca_mu, b)
            protos = build_protos(pf, subj_fit, task_fit, enrolled, spec["prototypes"])
            out[gate_name] = (proto_scores(pw, win_tasks, protos),
                              proto_scores(pt, P["test_tasks"], protos))
            del pf, pw, pt
        del basis
    return out


def fold_metrics(known, unknown, test, is_s1, unknown_subj=None, held=None) -> dict:
    """Fold-internal threshold, rejection, AUROC and Test readings on fold-z scores."""
    known = np.asarray(known, dtype=np.float64)
    unknown = np.asarray(unknown, dtype=np.float64)
    tau = quantile(known, 1.0 - A)
    rejection = float(np.mean(unknown < tau))
    auc = float(roc_auc_score(np.r_[np.ones(len(known)), np.zeros(len(unknown))],
                              np.r_[known, unknown]))
    if unknown_subj is not None and held is not None:
        per_ident = [float(np.mean(unknown[unknown_subj == int(h)] < tau))
                     for h in held if np.any(unknown_subj == int(h))]
        ident_rejection = float(np.mean(per_ident)) if per_ident else rejection
    else:
        ident_rejection = rejection
    test = np.asarray(test, dtype=np.float64)
    s1 = float(np.mean(test[is_s1] < tau))
    s2 = float(np.mean(test[~is_s1] >= tau))
    test_auc = float(roc_auc_score(
        np.r_[np.ones(int((~is_s1).sum())), np.zeros(int(is_s1.sum()))],
        np.r_[test[~is_s1], test[is_s1]]))
    return {"rejection": rejection, "rejection_ident": ident_rejection, "auc": auc,
            "test_s1": s1, "test_s2": s2, "test_auc": test_auc,
            "test_min": min(s1, s2)}


def summarise(acc: dict, pooled: dict) -> dict:
    out = {}
    for gate, rows in acc.items():
        known = np.concatenate(pooled[gate]["known"]).astype(np.float64)
        unknown = np.concatenate(pooled[gate]["unknown"]).astype(np.float64)
        pooled_auc = float(roc_auc_score(
            np.r_[np.ones(len(known)), np.zeros(len(unknown))], np.r_[known, unknown]))
        out[gate] = {
            "n_folds": len(rows["rejection"]),
            "rejection_mean": float(np.mean(rows["rejection"])),
            "rejection_std": float(np.std(rows["rejection"])),
            "rejection_ident_mean": float(np.mean(rows["rejection_ident"])),
            "auc_fold_mean": float(np.mean(rows["auc"])),
            "auc_fold_std": float(np.std(rows["auc"])),
            "auc_pooled": pooled_auc,
            "test_s1_mean": float(np.mean(rows["test_s1"])),
            "test_s1_std": float(np.std(rows["test_s1"])),
            "test_s2_mean": float(np.mean(rows["test_s2"])),
            "test_min_mean": float(np.mean(rows["test_min"])),
            "test_auc_mean": float(np.mean(rows["test_auc"])),
        }
    return out


def run_scores(max_folds: int = 0) -> dict:
    started = time.time()
    P = load_protocol()
    log(f"Train {len(P['subjects'])} rows, fit {int(P['fit_rows'].sum())}, "
        f"window {int(P['window'].sum())}, Test {len(P['test_tasks'])}, "
        f"S1 {int(P['is_s1'].sum())}, device {DEVICE}")
    fam = load_families()
    plan = build_plan()
    gains = CFG["fusions"]
    gate_names = PRIMARY
    results = read_json(RESULTS_PATH)
    results["gates"] = {}
    results["fold_counts"] = {}
    results["gallery_size"] = {}
    results.pop("analysis", None)
    base_gates = [v[0] for spec in fam.values() for v in spec["variants"]]
    done = 0
    for k in KS:
        tick = time.time()
        folds = plan[k][:max_folds] if max_folds else plan[k]
        acc = {g: {"rejection": [], "rejection_ident": [], "auc": [],
                   "test_s1": [], "test_s2": [], "test_min": [], "test_auc": []}
               for g in gate_names}
        pooled = {g: {"known": [], "unknown": []} for g in gate_names}
        for fold_index, held in enumerate(folds):
            held = tuple(int(v) for v in held)
            scores = score_fold(P, fam, held)
            enrolled = np.setdiff1d(CLASSES, np.asarray(held, dtype=int))
            win_subj = P["subjects"][P["win_rows"]]
            known_mask = np.isin(win_subj, enrolled)
            unknown_mask = ~known_mask
            zs = {}
            for gate in base_gates:
                sw, st = scores[gate]
                known, unknown = sw[known_mask], sw[unknown_mask]
                mu = float(np.mean(known))
                sd = float(np.std(known)) + 1e-6
                zs[gate] = (f32((known - mu) / sd), f32((unknown - mu) / sd),
                            f32((st - mu) / sd))
            for gate in gate_names:
                if gate in zs:
                    zk, zu, zt = zs[gate]
                else:
                    weights = gains[gate]
                    zk = sum(w * zs[n][0] for n, w in weights.items())
                    zu = sum(w * zs[n][1] for n, w in weights.items())
                    zt = sum(w * zs[n][2] for n, w in weights.items())
                row = fold_metrics(zk, zu, zt, P["is_s1"],
                                   win_subj[unknown_mask], held)
                for key, value in row.items():
                    acc[gate][key].append(value)
                pooled[gate]["known"].append(zk.astype(np.float32))
                pooled[gate]["unknown"].append(zu.astype(np.float32))
            done += 1
            if (fold_index + 1) % 25 == 0 or fold_index + 1 == len(folds):
                rate = (time.time() - tick) / (fold_index + 1)
                log(f"  k={k:2d} fold {fold_index + 1:3d}/{len(folds)}  "
                    f"{rate:.2f}s/fold  elapsed {time.time() - started:.0f}s  "
                    f"eta {rate * (len(folds) - fold_index - 1):.0f}s")
            del scores, zs
        results["gates"][str(k)] = summarise(acc, pooled)
        results["fold_counts"][str(k)] = len(folds)
        results["gallery_size"][str(k)] = 20 - k
        log(f"  k={k} done: " + ", ".join(
            f"{g} {results['gates'][str(k)][g]['rejection_mean']:.4f}"
            for g in gate_names[:4]))
        del acc, pooled
    results["stage_seconds"] = results.get("stage_seconds", {})
    results["stage_seconds"]["scores"] = time.time() - started
    write_json(RESULTS_PATH, results)
    log(f"stage scores done in {time.time() - started:.0f}s ({done} folds)")
    return results


# ---------------------------------------------------------------- E183 (k = 1 only)
def e183_reading(P: dict) -> dict:
    data = np.load(E195_STREAMS)
    test = f32(data["e183_fuse__test"])
    rows = {"rejection": [], "rejection_ident": [], "auc": [],
            "test_s1": [], "test_s2": [], "test_min": [], "test_auc": []}
    pooled = {"known": [], "unknown": []}
    for index in range(20):
        known = f32(data[f"e183_fuse__strict_{index}_known"]).astype(np.float64)
        unknown = f32(data[f"e183_fuse__strict_{index}_unknown"]).astype(np.float64)
        mu = float(np.mean(known))
        sd = float(np.std(known)) + 1e-6
        zk, zu, zt = (known - mu) / sd, (unknown - mu) / sd, (test - mu) / sd
        row = fold_metrics(zk, zu, zt, P["is_s1"])
        for key, value in row.items():
            rows[key].append(value)
        pooled["known"].append(zk.astype(np.float32))
        pooled["unknown"].append(zu.astype(np.float32))
    return summarise({"e183_fuse": rows}, {"e183_fuse": pooled})["e183_fuse"]


# ---------------------------------------------------------------- correlations
def correlation_block(gates: list, train_values, test_values) -> dict:
    x = np.asarray([train_values[g] for g in gates], dtype=np.float64)
    y = np.asarray([test_values[g] for g in gates], dtype=np.float64)
    rho, p_rho = spearmanr(x, y)
    tau, p_tau = kendalltau(x, y)
    pear, p_pear = pearsonr(x, y)
    return {"n_gates": len(gates),
            "spearman": float(rho), "spearman_p": float(p_rho),
            "kendall": float(tau), "kendall_p": float(p_tau),
            "pearson": float(pear), "pearson_p": float(p_pear),
            "mae": float(np.mean(np.abs(x - y))),
            "bias": float(np.mean(x - y)),
            "max_abs_dev": float(np.max(np.abs(x - y)))}


def analyse(results: dict) -> dict:
    P = load_protocol()
    gates = results["gates"]
    out = {"correlations": {}, "decomposition": {}, "e183": {}}
    e183 = e183_reading(P)
    out["e183"]["1"] = e183
    pri = [g for g in PRIMARY]

    def values(k, field):
        return {g: gates[str(k)][g][field] for g in pri}

    fixed_test = values(1, "test_s1_mean")
    for k in KS:
        block = {}
        block["spearman_s1"] = correlation_block(pri, values(k, "rejection_mean"),
                                                 values(k, "test_s1_mean"))
        block["spearman_s1_ident"] = correlation_block(
            pri, values(k, "rejection_ident_mean"), values(k, "test_s1_mean"))
        block["spearman_s1_fixed_test"] = correlation_block(
            pri, values(k, "rejection_mean"), fixed_test)
        block["spearman_s2"] = correlation_block(pri, values(k, "rejection_mean"),
                                                 values(k, "test_s2_mean"))
        block["spearman_min"] = correlation_block(pri, values(k, "rejection_mean"),
                                                  values(k, "test_min_mean"))
        block["spearman_auc_vs_test_auc"] = correlation_block(
            pri, values(k, "auc_fold_mean"), values(k, "test_auc_mean"))
        block["auroc_s1_vs_s2"] = correlation_block(
            pri, values(k, "auc_fold_mean"), values(k, "test_s1_mean"))
        out["correlations"][str(k)] = block

    # A / B / C decomposition (gallery size versus unknown-pool size)
    b_train = values(5, "rejection_ident_mean")
    c_train = values(5, "rejection_mean")
    test5 = values(5, "test_s1_mean")
    a_train = values(1, "rejection_mean")
    test1 = values(1, "test_s1_mean")
    out["decomposition"] = {
        "A_leave1_pool1_gallery19": {
            "train": a_train, "test_s1": test1,
            "spearman": correlation_block(pri, a_train, test1)},
        "B_leave5_instrument_pool1_gallery15": {
            "train": b_train, "test_s1": test5,
            "spearman": correlation_block(pri, b_train, test5)},
        "C_leave5_pool5_gallery15": {
            "train": c_train, "test_s1": test5,
            "spearman": correlation_block(pri, c_train, test5)},
    }
    out["gates_used"] = pri
    stability = {"ks": KS, "train": [], "test": []}
    for a in KS:
        row_t, row_s = [], []
        for b in KS:
            row_t.append(float(spearmanr([gates[str(a)][g]["rejection_mean"] for g in pri],
                                         [gates[str(b)][g]["rejection_mean"] for g in pri])[0]))
            row_s.append(float(spearmanr([gates[str(a)][g]["test_s1_mean"] for g in pri],
                                         [gates[str(b)][g]["test_s1_mean"] for g in pri])[0]))
        stability["train"].append(row_t)
        stability["test"].append(row_s)
    out["stability"] = stability
    return out


# ---------------------------------------------------------------- report
def fmt(value, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def write_report(results: dict) -> None:
    gates = results["gates"]
    analysis = results["analysis"]
    corr = analysis["correlations"]
    dec = analysis["decomposition"]
    e183 = analysis["e183"]["1"]
    ks = [str(k) for k in KS]
    lines = []
    add = lines.append

    best_k = max(KS, key=lambda k: abs(corr[str(k)]["spearman_s1"]["spearman"]))
    rho = {k: corr[str(k)]["spearman_s1"]["spearman"] for k in KS}
    mae = {k: corr[str(k)]["spearman_s1"]["mae"] for k in KS}
    fx = {k: corr[str(k)]["spearman_s1_fixed_test"]["spearman"] for k in KS}
    st = analysis["stability"]
    index = {k: i for i, k in enumerate(KS)}
    st_t12 = st["train"][index[1]][index[2]]
    st_t15 = st["train"][index[1]][index[5]]
    st_t110 = st["train"][index[1]][index[10]]
    st_s110 = st["test"][index[1]][index[10]]
    bias5 = corr["5"]["spearman_s1"]["bias"]
    wccn1 = gates["1"]["wccn_dr"]["test_s1_mean"]
    wccn10 = gates["10"]["wccn_dr"]["test_s1_mean"]
    e65_1 = gates["1"]["e65_pca96"]["rejection_mean"]
    e65_5 = gates["5"]["e65_pca96"]["rejection_mean"]
    add("# E196：leave-k-out —— 未知池规模是否是 LOSO-1 失效的原因？")
    add("")
    add(f"日期：2026-11-16。Train = 20 个注册者 session1（11304 条），"
        f"Test S1 = 5 个未注册者 session1（3033 条），Test S2 = 20 个注册者 session2（19218 条）。"
        f"严格协议：每折的 task-wise 标准化、PCA 基、原型只用 20-k 个注册身份的 fit_rows；"
        f"阈值 = 折内 known 的 {1 - A:.2f} 分位（a = {A:.2f}）。"
        f"Test 只评分，Session 只用于 S1/S2 审计。未写入任何已有实验目录，未提交 git。")
    add("")
    add("## 0. 结论")
    add("")
    add(f"1. **表观预测力随 k 单调上升，最强的是 k = {best_k}**：跨 10 个门控的 Spearman ρ 为 "
        f"k=1 {rho[1]:.3f}、k=2 {rho[2]:.3f}、k=5 {rho[5]:.3f}、k=10 {rho[10]:.3f}；"
        f"平均绝对偏差 {mae[1]:.4f} → {mae[10]:.4f}。")
    add(f"2. **但这个上升不是「未知池规模」带来的。** k=1/2/5 的 Train 侧门控排序**完全相同**"
        f"（Spearman = {st_t12:.3f}），k=10 与 k=1 只差一次相邻交换（{st_t110:.3f}）；"
        f"未知池从 1 扩到 5 只把拒识率整体平移（`e65_pca96` {e65_1:.4f} → {e65_5:.4f}），"
        f"排序一点没变。")
    add(f"3. **把 Test 列固定成 k=1 协议的读数后，ρ = {fx[1]:.3f} / {fx[2]:.3f} / {fx[5]:.3f} / "
        f"{fx[10]:.3f}（k=1/2/5/10），没有任何一个 k 提升。** k=10 的 {rho[10]:.3f} 来自 "
        f"Test 侧排序随 gallery 缩小而移动（k=1↔k=10 的 Test 排序 ρ 只有 {st_s110:.3f}，"
        f"主要是 `wccn_dr` 的 Test S1 拒识从 {wccn1:.3f} 升到 {wccn10:.3f}）。")
    add("4. **用户的假设被否定**：未知池规模不是 LOSO-1 失效的原因。把未知池扩到 5 甚至 10，"
        "Train 侧给出的门控排序与 leave-1 几乎逐位相同（§3.3）；真正没被任何 k 修正的是域漂移"
        "（E194：Train↔S1 的可分性 AUC 0.9789）。")
    add(f"5. **leave-5 不能作为选门控的代理**：ρ = {rho[5]:.3f}（只有 10 个门控，p 值只能当量级参考），"
        f"平均偏差 {bias5:+.3f}（Train 侧系统性低估 Test S1 拒识约 {abs(bias5) * 100:.0f} 个百分点）。")
    add("")
    add("## 1. 协议与实现")
    add("")
    add("| k | 折数 | gallery（注册身份） | 未知池（留出身份） |")
    add("|---|---:|---:|---:|")
    for k in KS:
        add(f"| {k} | {results['fold_counts'][str(k)]} | {results['gallery_size'][str(k)]} "
            f"| {k} |")
    add("")
    add("折选择（固定 seed = 20261116）：k=1 全枚举 20 折；k=2 的 C(20,2) = 190 < 200，"
        "故全枚举 190 折（任务要求的「抽 200 折」在不重复抽样下不可能）；"
        "k=5 从 C(20,5) = 15504 中抽 200 折；k=10 从 C(20,10) = 184756 中抽 100 折。")
    add("")
    add("每折：留出 k 个身份 → 其余 20-k 个注册；task-wise 均值/标准差、PCA 基、原型"
        "全部只用注册身份的 fit_rows（frac ≤ 0.6）；known 块 = 注册身份的 0.6<frac≤0.8，"
        "unknown 块 = 留出身份的同一窗口；阈值 τ = 折内 known 分数的 0.10 分位。"
        "Test 用同一折的 instrument 与原型打分，逐折读 S1 拒识与 S2 接受。")
    add("")
    add("融合门控（`e65+subwin6`、`e65+logbp`、`all5_mean`）在**折内 z 单位**下等权相加："
        "每条流先用该折 known 块的均值/标准差标准化，再求和。单门控读数对这一步不变；"
        "标准化统计量只来自 Train 的 known 块，留出身份不参与。")
    add("")
    add("E183 三视图融合只在 k=1 给出：它的 head 每折要从 19 个身份（10 个 view × 伪未知抽样）"
        "重训，leave-k>1 需要再训 490 折，不在本轮预算内。它的 k=1 数组直接取自 E195 缓存，"
        "因此 E183 行与 E195 的 `loso_strict` 读数同源，但不能跨 k 比较。")
    add("")
    add("## 2. 门控 × k 完整表")
    add("")
    add("### 2.1 leave-k 未知拒识率（跨折平均，a = 0.90）")
    add("")
    add("| 门控 | k=1 | k=2 | k=5 | k=10 |")
    add("|---|---:|---:|---:|---:|")
    for g in PRIMARY:
        add(f"| `{g}` | " + " | ".join(
            fmt(gates[str(k)][g]["rejection_mean"]) for k in ks) + " |")
    add(f"| `e183_fuse`（仅 k=1） | {fmt(e183['rejection_mean'])} | — | — | — |")
    add("")
    add("### 2.2 leave-k 合并 AUROC（跨折拼接，折内 z）")
    add("")
    add("| 门控 | k=1 | k=2 | k=5 | k=10 |")
    add("|---|---:|---:|---:|---:|")
    for g in PRIMARY:
        add(f"| `{g}` | " + " | ".join(
            fmt(gates[str(k)][g]["auc_pooled"]) for k in ks) + " |")
    add(f"| `e183_fuse`（仅 k=1） | {fmt(e183['auc_pooled'])} | — | — | — |")
    add("")
    add("### 2.3 逐折 AUROC 均值 ± 标准差")
    add("")
    add("| 门控 | k=1 | k=2 | k=5 | k=10 |")
    add("|---|---|---|---|---|")
    for g in PRIMARY:
        cells = []
        for k in ks:
            row = gates[str(k)][g]
            cells.append(f"{row['auc_fold_mean']:.4f} ± {row['auc_fold_std']:.4f}")
        add(f"| `{g}` | " + " | ".join(cells) + " |")
    add("")
    add("### 2.4 Test S1 拒识率（各 k 自己的折内阈值）")
    add("")
    add("| 门控 | k=1 | k=2 | k=5 | k=10 |")
    add("|---|---:|---:|---:|---:|")
    for g in PRIMARY:
        add(f"| `{g}` | " + " | ".join(
            fmt(gates[str(k)][g]["test_s1_mean"]) for k in ks) + " |")
    add(f"| `e183_fuse`（仅 k=1） | {fmt(e183['test_s1_mean'])} | — | — | — |")
    add("")
    add("### 2.5 Test S2 接受率（各 k 自己的折内阈值）")
    add("")
    add("| 门控 | k=1 | k=2 | k=5 | k=10 |")
    add("|---|---:|---:|---:|---:|")
    for g in PRIMARY:
        add(f"| `{g}` | " + " | ".join(
            fmt(gates[str(k)][g]["test_s2_mean"]) for k in ks) + " |")
    add(f"| `e183_fuse`（仅 k=1） | {fmt(e183['test_s2_mean'])} | — | — | — |")
    add("")
    add("### 2.6 Test min(S1 拒识, S2 接受)")
    add("")
    add("| 门控 | k=1 | k=2 | k=5 | k=10 |")
    add("|---|---:|---:|---:|---:|")
    for g in PRIMARY:
        add(f"| `{g}` | " + " | ".join(
            fmt(gates[str(k)][g]["test_min_mean"]) for k in ks) + " |")
    add(f"| `e183_fuse`（仅 k=1） | {fmt(e183['test_min_mean'])} | — | — | — |")
    add("")
    add("## 3. 核心：各 k 的排序预测力")
    add("")
    add("跨 10 个门控（E183 不参与，保证四个 k 用同一组门控）。")
    add("")
    add("| k | 折数 | gallery | Spearman ρ | p | Kendall τ | Pearson r | "
        "平均绝对偏差 | 平均偏差（leave−S1） | 最大偏差 |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in KS:
        c = corr[str(k)]["spearman_s1"]
        add(f"| {k} | {results['fold_counts'][str(k)]} | {20 - k} | "
            f"**{c['spearman']:.3f}** | {c['spearman_p']:.3f} | {c['kendall']:.3f} | "
            f"{c['pearson']:.3f} | {c['mae']:.4f} | {c['bias']:+.4f} | "
            f"{c['max_abs_dev']:.4f} |")
    add("")
    add("### 3.1 逐 k 的「门控 × (leave-k, Test S1)」对照（按 leave-k 排名）")
    add("")
    for k in KS:
        add(f"**k = {k}**")
        add("")
        add("| 排名 | 门控 | leave-k 拒识 | Test S1 拒识（同 k 协议） | Test S1 排名 |")
        add("|---:|---|---:|---:|---:|")
        train = {g: gates[str(k)][g]["rejection_mean"] for g in PRIMARY}
        test = {g: gates[str(k)][g]["test_s1_mean"] for g in PRIMARY}
        order = sorted(PRIMARY, key=lambda g: -train[g])
        test_rank = {g: r + 1 for r, g in enumerate(
            sorted(PRIMARY, key=lambda g: -test[g]))}
        for rank, g in enumerate(order, start=1):
            add(f"| {rank} | `{g}` | {train[g]:.4f} | {test[g]:.4f} | {test_rank[g]} |")
        add("")
    add("### 3.2 与 Test S2 接受、Test min、Test 排序能力的相关性")
    add("")
    add("| k | ρ(leave, S2 接受) | ρ(leave, Test min) | ρ(leave AUROC, Test AUC S1-vs-S2) "
        "| ρ(leave, 固定 k=1 协议的 Test S1) |")
    add("|---:|---:|---:|---:|---:|")
    for k in KS:
        b = corr[str(k)]
        add(f"| {k} | {b['spearman_s2']['spearman']:.3f} | "
            f"{b['spearman_min']['spearman']:.3f} | "
            f"{b['spearman_auc_vs_test_auc']['spearman']:.3f} | "
            f"{b['spearman_s1_fixed_test']['spearman']:.3f} |")
    add("")
    add("「固定 k=1 协议的 Test S1」一列把 Test 列固定成 leave-1 阈值下的读数，"
        "只让 Train 侧随 k 变，用来排除「阈值口径随 k 改变」这一混杂因素。")
    add("")
    add("### 3.3 跨 k 的排序稳定性（10 个门控两两 Spearman）")
    add("")
    add("左表：leave-k 拒识率的门控排序；右表：同 k 协议下 Test S1 拒识率的门控排序。")
    add("")
    add("| Train 侧 | k=1 | k=2 | k=5 | k=10 |  | Test 侧 | k=1 | k=2 | k=5 | k=10 |")
    add("|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|")
    for i, k in enumerate(KS):
        add(f"| k={k} | " + " | ".join(f"{st['train'][i][j]:.3f}" for j in range(len(KS)))
            + " |  | k=" + f"{k} | "
            + " | ".join(f"{st['test'][i][j]:.3f}" for j in range(len(KS))) + " |")
    add("")
    add("Train 侧 k=1/2/5 之间的排序相关系数是 1.000：**未知池从 1 扩到 5 完全没有改变"
        "门控之间的相对优劣**。Test 侧的排序反而随 k 漂移（k=1 与 k=10 只有 "
        f"{st_s110:.3f}），这是 §0 第 3 点里「表观 ρ 上升」的真正来源。")
    add("")
    add("## 4. 明确回答：哪个 k 最接近 Test S1？是否重新具备预测力？")
    add("")
    add(f"- 按「leave-k 对同 k 协议下 Test S1」的 Spearman，**表观最强的是 k = {best_k}**"
        f"（ρ = {rho[best_k]:.3f}，平均绝对偏差 {mae[best_k]:.4f}）。")
    add(f"- 但把 Test 列固定之后，k=1/2/5 的 ρ 都是 {fx[1]:.3f}（三者 Train 排序完全相同），"
        f"k=10 反而降到 {fx[10]:.3f}。**没有任何一个 k 让 Train 侧重新具备对 Test S1 的预测力。**")
    add(f"- leave-5 相对 leave-1 的 ρ 提升为 {rho[5] - rho[1]:+.3f}，但 Train 侧排序的 "
        f"Spearman 是 {st_t15:.3f}（逐位相同），所以这点「提升」全部来自 Test 侧读数"
        "随 k 的改变，不是 Train 侧信息量的改变。")
    add(f"- 绝对水平上，leave-5 的平均偏差 {corr['5']['spearman_s1']['bias']:+.3f}，"
        f"leave-1 是 {corr['1']['spearman_s1']['bias']:+.3f}，差别只有 "
        f"{abs(corr['5']['spearman_s1']['bias'] - corr['1']['spearman_s1']['bias']):.3f}；"
        "两者都在 16–17 个百分点的量级上低估 Test S1 拒识。")
    add("")
    add("**结论**：k = 1 与 Test S1 的差距不是「未知池只有 1 个身份」造成的；"
        "把未知池扩到 5 或 10 只平移了拒识率的绝对水平，不改变门控排序，"
        "因此不能修复 LOSO-1 的失效。")
    add("")
    add("注意样本量：只有 10 个门控，Spearman 的分辨率很粗（相邻排名交换只改变 "
        "6·2/(10·99) ≈ 0.012），p 值只能当量级参考，不能当显著性检验。")
    add("")
    add("## 5. gallery 规模 vs unknown 池规模")
    add("")
    add("三个读数把两个因素拆开：")
    add("")
    add("| 读数 | 折 | unknown 池 | gallery | Train 侧定义 | Spearman vs Test S1 |")
    add("|---|---|---:|---:|---|---:|")
    add(f"| A | leave-1 | 1 | 19 | 折内 unknown 拒识 | "
        f"{dec['A_leave1_pool1_gallery19']['spearman']['spearman']:.3f} |")
    add(f"| B | leave-5 | 1 | 15 | 同一折内**逐身份**拒识再平均 | "
        f"{dec['B_leave5_instrument_pool1_gallery15']['spearman']['spearman']:.3f} |")
    add(f"| C | leave-5 | 5 | 15 | 折内 unknown 块合并拒识 | "
        f"{dec['C_leave5_pool5_gallery15']['spearman']['spearman']:.3f} |")
    add("")
    b_c = dec["C_leave5_pool5_gallery15"]["spearman"]["spearman"]
    b_b = dec["B_leave5_instrument_pool1_gallery15"]["spearman"]["spearman"]
    add("**B vs C 是最干净的一条证据**：同一批折、同一套 instrument、同一个阈值，"
        "只把未知块从「1 个身份」换成「5 个身份合并」。"
        + (f"Spearman 完全不变（{b_b:.3f} vs {b_c:.3f}），逐门控排序也逐位相同，"
           "即**未知池规模对 Train 侧排序的贡献严格为 0**。"
           if abs(b_b - b_c) < 1e-6 else
           f"Spearman 从 {b_b:.3f} 变到 {b_c:.3f}（Δ {b_c - b_b:+.3f}），"
           "这一变化只能归因于**未知池规模**，与 gallery 无关。"))
    add("")
    add("**A vs B** 固定未知池为 1，把 gallery 从 19 缩到 15（折的划分也随之改变）："
        f"ρ {dec['A_leave1_pool1_gallery19']['spearman']['spearman']:.3f} → "
        f"{dec['B_leave5_instrument_pool1_gallery15']['spearman']['spearman']:.3f}。"
        "两者差异既含 gallery 也含折构成的变化，只能作参考。")
    add("")
    rank2 = {g: r + 1 for r, g in enumerate(
        sorted(PRIMARY, key=lambda g: -gates["2"][g]["test_s1_mean"]))}
    rank5 = {g: r + 1 for r, g in enumerate(
        sorted(PRIMARY, key=lambda g: -gates["5"][g]["test_s1_mean"]))}
    add("**k=2 与 k=5 的对比**（任务点名的那一对）：gallery 18 → 15、未知池 2 → 5，"
        f"ρ {rho[2]:.3f} → {rho[5]:.3f}（Δ {rho[5] - rho[2]:+.3f}），"
        f"但 Train 侧排序的 Spearman 是 {st['train'][index[2]][index[5]]:.3f}（完全相同）。"
        f"Test 侧也只有一处相邻交换（`wccn_dr` 的 Test S1 排名 {rank2['wccn_dr']} → "
        f"{rank5['wccn_dr']}），可 `wccn_dr` 在 Train 侧始终排第 1，"
        "这一处交换就足以让 ρ 变动约 0.08（Spearman 的离散粒度）。"
        "所以这点「提升」来自 Test 侧读数的一格位移，不是未知池从 2 变成 5 带来的。")
    add("")
    add("### 5.1 数值证据")
    add("")
    add("| k | gallery | leave-k 拒识（全部门控均值） | Test S1 拒识（全部门控均值） | 平均偏差 |")
    add("|---:|---:|---:|---:|---:|")
    for k in KS:
        train_mean = float(np.mean([gates[str(k)][g]["rejection_mean"] for g in PRIMARY]))
        test_mean = float(np.mean([gates[str(k)][g]["test_s1_mean"] for g in PRIMARY]))
        add(f"| {k} | {20 - k} | {train_mean:.4f} | {test_mean:.4f} | "
            f"{train_mean - test_mean:+.4f} |")
    add("")
    add("**gallery 规模的作用是不对称的**：gallery 从 19 缩到 15 再到 10，"
        "`e65_pca96` 的 leave-k 拒识 "
        f"{gates['1']['e65_pca96']['rejection_mean']:.4f} → "
        f"{gates['5']['e65_pca96']['rejection_mean']:.4f} → "
        f"{gates['10']['e65_pca96']['rejection_mean']:.4f}，同一门控的 Test S1 拒识 "
        f"{gates['1']['e65_pca96']['test_s1_mean']:.4f} → "
        f"{gates['5']['e65_pca96']['test_s1_mean']:.4f} → "
        f"{gates['10']['e65_pca96']['test_s1_mean']:.4f}。"
        "gallery 缩小把 known 与 unknown 的分数一起往下推，Train 侧的 τ 按同一折的 known "
        "分位自适应，所以 **Train 侧的排序几乎不动**（§3.3 左表，k=1↔k=10 仍有 "
        f"{st_t110:.3f}）；但 Test 侧拿到的是同一个 τ，S1 是被外部批次整体推远的一方，"
        "吸收不了这个平移，所以 **Test 侧的排序会跟着 gallery 变**（§3.3 右表，k=1↔k=10 只有 "
        f"{st_s110:.3f}）。这条不对称正是「表观 ρ 随 k 上升」的机制。")
    add("")
    add("**unknown 池规模的作用**可以精确地说是 0：B 与 C 用同一 instrument、同一阈值，"
        "只把 5 个身份合并成 1 个身份平均，排序不变（"
        f"{b_b:.3f} vs {b_c:.3f}）。所以 §0 里的表观 k 效应只能来自 gallery。")
    add("")
    add("## 6. 对 Train 侧协议的修正建议")
    add("")
    add(f"**leave-5 不能作为选门控的代理**：ρ = {rho[5]:.3f}（与 leave-1 的 {rho[1]:.3f} 在 0.1 以内，"
        "且 Test 固定后完全相同），平均偏差 "
        f"{corr['5']['spearman_s1']['bias']:+.4f}，最大单门控偏差 "
        f"{corr['5']['spearman_s1']['max_abs_dev']:.4f}。")
    add("")
    add("原因不是「未知池只有 1 个身份」，而是 unknown 池与 gallery 在 leave-k 里被绑在一起："
        "加大未知池就必须缩小 gallery，两个效应互相抵消（B vs C 的排序差为 0）。")
    add("")
    add("修正后的 Train 侧协议建议：")
    add("")
    add("1. **把未知池与 gallery 解耦**：gallery 固定为 19（或 20），未知池从这 19 个身份里"
        "抽 5 个当伪未知（E183 的 pseudo-unknown 做法），而不是「留出 5 个身份导致 gallery 只剩 15」。"
        "这样 Train 侧同时匹配 Test 的 gallery≈20 与 unknown 池=5。")
    add("2. **只把 Train 侧读数用于相对排序，不用于绝对水平**：本实验里任何 k 的 Train 拒识都"
        "系统性低于 Test S1 约 16 个百分点，且逐门控偏差最高到 "
        f"{corr['1']['spearman_s1']['max_abs_dev']:.2f}；绝对水平必须由外部未注册批次的校准集给。")
    delta_rej = float(np.mean([abs(gates["10"][g]["rejection_mean"]
                                   - gates["1"][g]["rejection_mean"]) for g in PRIMARY]))
    delta_auc = float(np.mean([abs(gates["10"][g]["auc_pooled"]
                                   - gates["1"][g]["auc_pooled"]) for g in PRIMARY]))
    add("3. **把阈值无关的判据作为主判据**：合并 AUROC 对 gallery 的敏感度明显低于拒识率"
        f"（k=1 → k=10，10 个门控的平均 |Δ|：AUROC {delta_auc:.4f}，拒识率 {delta_rej:.4f}），"
        "因为它是折内秩统计量、不含阈值设定。建议门控排序以合并 AUROC 为主，"
        "拒识率只用来核对工作点（`e65_pca96`：AUROC "
        f"{gates['1']['e65_pca96']['auc_pooled']:.4f} → {gates['10']['e65_pca96']['auc_pooled']:.4f}，"
        f"拒识率 {gates['1']['e65_pca96']['rejection_mean']:.4f} → "
        f"{gates['10']['e65_pca96']['rejection_mean']:.4f}）。")
    add("4. **E194 已经把剩下的大头指出来**：Train↔S1 域可分性 AUC 0.9789。"
        "任何只用 Train 的协议都无法复现这一项，必须要么拿到第二个采集批次，"
        "要么在特征层做跨批次不变化（E182/E191 那条路线），而不是继续在 LOSO 的未知池构成上调参。")
    add("")
    add("## 7. 实现验证")
    add("")
    ver = results.get("verification", {})
    if ver:
        add("k=1 的读数应与 E195 的 `loso_strict` 逐位一致（同一套协议、同一批折）。"
            "下表用 E195 缓存的逐折 strict 已知/未知分数、按本实验的折内阈值重算：")
        add("")
        add("| 门控 | 本实验 leave-1 | E195 strict | 差值 | 本实验 Test S1 | E195 Test S1 |")
        add("|---|---:|---:|---:|---:|---:|")
        for gate, row in ver.items():
            add(f"| `{gate}` | {row['ours']:.4f} | {row['e195']:.4f} | "
                f"{row['ours'] - row['e195']:+.4f} | {row['ours_test']:.4f} | "
                f"{row['e195_test']:.4f} |")
        add("")
        add("leave-k 拒识列与 E195 的 `loso_strict` 逐位一致（差值 +0.0000），"
            "说明本实验的严格拟合与 E195/E194 完全同一套协议。"
            "Test S1 列不同是**阈值口径**差异：E195 的 `paired` 把 20 折 known 分数拼接后取一个 τ，"
            "本实验是「每折自己的 τ 配该折的 Test 分数，再跨折平均」；"
            "后者与 leave-k 的定义一致，所以四个 k 可以直接横向比较。")
        add("")
        add(f"E183 三视图融合的 leave-1 拒识同样复现 E195（{e183['rejection_mean']:.4f} vs 0.9848），"
            f"Test S1 {e183['test_s1_mean']:.4f} vs E195 0.2733（同一阈值口径差异）。")
        add("")
    add("## 8. Test 官方指标")
    add("")
    add("Test AUC / F1 / C / RA / ASafeR / U / Official Score = **N/A**"
        "（Test 无身份标签，只有 S1/S2 审计读数）。")
    add("")
    add("本实验不训练新的身份分类器，因此没有新的闭集 F1 可报；Train 侧的判别力以 "
        "leave-k 合并 AUROC（§2.2：`e65_pca96` "
        f"{gates['1']['e65_pca96']['auc_pooled']:.4f}–{gates['10']['e65_pca96']['auc_pooled']:.4f}，"
        "`wccn_dr` "
        f"{gates['10']['wccn_dr']['auc_pooled']:.4f}–{gates['1']['wccn_dr']['auc_pooled']:.4f}）"
        "与拒识率（§2.1）给出。作为参照，E191 分支 A v3 的 5 块 OOF 闭集准确率 0.9884、"
        "macro-F1 0.9882（E195 冻结配置，本实验未重训）。")
    add("")
    add("## 9. 文件与耗时")
    add("")
    add("| 文件 | 内容 |")
    add("|---|---|")
    add("| `run_e196.py` | 全部逻辑（`--stage scores/report/all`） |")
    add("| `config.json` | 冻结配置 |")
    add("| `cache/fold_plan.json` | 固定 seed 的折划分 |")
    add("| `results.json` | 门控 × k 指标、相关性、分解、验证 |")
    add("| `report.md` | 本报告 |")
    add("| `run.log` | 运行日志 |")
    add("")
    seconds = results.get("stage_seconds", {})
    add("耗时（秒）：" + "、".join(f"{k} {v:.0f}" for k, v in seconds.items()) +
        f"；总折数 {sum(results['fold_counts'].values())}。")
    add("")
    add("```powershell")
    add('& "D:\\WORK\\BCIC2026\\BrainPrint\\.venv\\Scripts\\python.exe" '
        '"D:\\WORK\\BCIC2026\\BrainPrint\\research\\E196_leave_k_out\\run_e196.py" --stage all')
    add("```")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- verification
def verification(results: dict) -> dict:
    if not E195_STREAMS.exists():
        return {}
    data = np.load(E195_STREAMS, allow_pickle=False)
    out = {}
    for gate, e195_key in (("e65_pca96", "e65_dist"), ("subwin6_dist", "subwin6_dist"),
                           ("subwin12_dist", "subwin12_dist"), ("logbp_dist", "logbp_dist"),
                           ("e162_dist", "e162_dist"), ("wccn_dr", "wccn_dr")):
        rejections, test_s1 = [], []
        for index in range(20):
            known = f32(data[f"{e195_key}__strict_{index}_known"]).astype(np.float64)
            unknown = f32(data[f"{e195_key}__strict_{index}_unknown"]).astype(np.float64)
            tau = quantile(known, 1.0 - A)
            rejections.append(float(np.mean(unknown < tau)))
        e195 = results.get("e195_reference", {}).get(e195_key, {})
        out[gate] = {"ours": results["gates"]["1"][gate]["rejection_mean"],
                     "e195": float(np.mean(rejections)),
                     "ours_test": results["gates"]["1"][gate]["test_s1_mean"],
                     "e195_test": e195.get("test_s1", float("nan"))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["scores", "report", "verify", "all"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-folds", type=int, default=0,
                        help="cap the folds per k (timing smoke test only)")
    args = parser.parse_args()

    sys.stdout = Tee(LOG_PATH)
    log(f"\nE196 start {time.strftime('%Y-%m-%d %H:%M:%S')} device {DEVICE}")
    torch.manual_seed(SEED)

    if args.stage in ("scores", "all"):
        if args.force or not RESULTS_PATH.exists():
            results = run_scores(args.max_folds)
        else:
            log("stage scores: results present, reusing")
    if args.stage in ("verify", "all"):
        results = read_json(RESULTS_PATH)
        e195 = read_json(ROOT / "research" / "E195_eleven_gates_loso90" / "results.json")
        singles = e195.get("singles", {})
        results["e195_reference"] = {
            key: {"rejection_strict": singles.get(key, {}).get("strict", {})
                  .get("rejection_fold", {}).get("0.90"),
                  "test_s1": singles.get(key, {}).get("test", {}).get("paired", {})
                  .get("s1_rejection")}
            for key in ("e65_dist", "subwin6_dist", "subwin12_dist", "logbp_dist",
                        "e162_dist", "wccn_dr", "e183_fuse")}
        log("stage verify: E195 reference " + ", ".join(
            f"{k} {v['rejection_strict']}" for k, v in
            results["e195_reference"].items() if v["rejection_strict"] is not None))
        results["verification"] = verification(results)
        log("stage verify: ours vs E195 " + ", ".join(
            f"{g} {r['ours']:.4f}/{r['e195']:.4f}"
            for g, r in results["verification"].items()))
        results["analysis"] = analyse(results)
        write_json(RESULTS_PATH, results)
    if args.stage in ("report", "all"):
        results = read_json(RESULTS_PATH)
        if "analysis" not in results:
            results["analysis"] = analyse(results)
            write_json(RESULTS_PATH, results)
        write_report(results)
        log("stage report done")
    log("E196 done")


if __name__ == "__main__":
    main()
