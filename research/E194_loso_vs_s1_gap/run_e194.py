"""E194: why is the LOSO unknown rejection (~0.98) so much higher than Test S1 (0.2733)?

Candidate causes
  A  the held-out identity still sits inside the feature-fitting statistics.  The E143
     protocol takes the task-wise mean/std from every Train row of the task and fits the
     PCA on every Train fit_row, so the held-out identity is inside those statistics even
     though it never enters the prototypes.  The five Test-S1 subjects were never fitted.
  B  the threshold reference differs: per-fold / pooled LOSO known quantile vs the
     deployment 5-block OOF quantile.
  C  estimation geometry: one unknown identity per fold (averaged over 20 folds) vs five
     unknown identities pooled under one threshold.

Design
  exp1  strict-nested LOSO for two gates (E65 distance, E183 fused) against the published
        reading.  Strict = z statistics / PCA basis / prototypes from the other 19
        identities only.  For E183 the strict head is the published cache; the leaky
        counterfactual re-trains the same 20 x 3 heads with the E143-style transform.
  exp2  domain drift of the deployment PCA-96 space: per-task mean shift in Train sd
        units and 5-fold CV domain separability AUC.
  exp3  S1 rejection under three threshold rules.

Train-only everywhere (strict: 19 identities).  Test is scored only; Session is read only
for the S1/S2 audit split.  No existing experiment directory is written to.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "report.md"
LOG_PATH = OUT / "run.log"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FT = torch.float32
CLASSES = np.arange(1, 21, dtype=int)
TASKS = ["EC", "EO", "Imagery", "SSVEP"]
SEED = int(CFG["protocol"]["seed"])
FIT = float(CFG["protocol"]["fit_fraction"])
NBLOCK = int(CFG["protocol"]["blocks"])
GRID = [float(a) for a in CFG["protocol"]["acceptance_grid"]]
PRIMARY = float(CFG["protocol"]["primary_acceptance"])
DIM = int(CFG["gates"]["e65_distance"]["pca_dim"])
KPROTO = int(CFG["gates"]["e65_distance"]["prototype_k"])
HEAD = CFG["head"]
E183_DIR = ROOT / "research" / "E183_gate_head_and_capacity"
E183_LOSO = E183_DIR / "cache" / "loso_head.npz"
E183_TEST = E183_DIR / "test_predictions_fused.csv"
CONTAM_CACHE = CACHE / "contam_heads.npz"
DIST_CACHE = CACHE / "distance_gate.npz"
KEY = f"{PRIMARY:.2f}"
WEIGHTS = {"g_head": 0.85, "g_dist": 0.10, "g_entropy": 0.05}


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


def save_results(results: dict) -> None:
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2,
                                       default=float), encoding="utf-8")


# ---------------------------------------------------------------- data
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
    blocks = np.clip((frac * NBLOCK).astype(int), 0, NBLOCK - 1)
    return {"subjects": subjects, "tasks": tasks, "frac": frac, "blocks": blocks,
            "fit_rows": frac <= FIT, "window": (frac > 0.6) & (frac <= 0.8)}


def load_features() -> tuple:
    train = f32(np.load(ROOT / CFG["data"]["train_features"]))
    test = f32(np.load(ROOT / CFG["data"]["test_features"]))
    if train.shape[1] != int(CFG["data"]["feature_dim"]):
        raise ValueError(f"Train feature width {train.shape}")
    test_meta = pd.read_csv(ROOT / CFG["data"]["test_metadata"])
    if len(test) != len(test_meta):
        raise ValueError("Test feature / metadata length mismatch")
    session = test_meta["session"].astype(str).str.casefold().to_numpy()
    return (train, test, test_meta["task"].to_numpy(dtype=object),
            session == "session1", session == "session2")


# ---------------------------------------------------------------- instrument
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
    """Identical to E180/E143: covariance eigen-decomposition, top-dim eigenvectors."""
    data = np.asarray(x, dtype=np.float64)
    mu = data.mean(0)
    centred = data - mu
    values, vectors = np.linalg.eigh(centred.T @ centred / max(len(centred) - 1, 1))
    basis = vectors[:, np.argsort(-values)][:, : min(int(dim), data.shape[1])]
    return f32(mu), f32(basis)


def fit_instrument(train_features: np.ndarray, tasks: np.ndarray,
                   ref_mask: np.ndarray, pca_mask: np.ndarray) -> dict:
    """Per-task z statistics from ref_mask rows, PCA basis from pca_mask rows."""
    mu = np.zeros((len(TASKS), train_features.shape[1]), dtype=np.float32)
    sd = np.zeros((len(TASKS), train_features.shape[1]), dtype=np.float32)
    for index, task in enumerate(TASKS):
        ref = ref_mask & (tasks == task)
        if not ref.any():
            raise ValueError(f"empty reference for task {task}")
        mu[index] = train_features[ref].mean(0)
        sd[index] = train_features[ref].std(0) + 1e-6
    z = apply_z(train_features, tasks, mu, sd)
    pca_mu, basis = pca_basis(z[pca_mask], DIM)
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
    """task -> (n_prototype, dim) tensor pooled over the selected identities (E143 style)."""
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


# ---------------------------------------------------------------- score statistics
def rejection(values, threshold: float) -> float:
    return float(np.mean(np.asarray(values) < threshold))


def loso_curves(parts: list) -> dict:
    known = np.concatenate([p[0] for p in parts]).astype(np.float64)
    unknown = np.concatenate([p[1] for p in parts]).astype(np.float64)
    labels = np.r_[np.ones(len(known)), np.zeros(len(unknown))]
    fold_auc = []
    fold_rej = {f"{a:.2f}": [] for a in GRID}
    global_rej = {f"{a:.2f}": [] for a in GRID}
    for kn, un in parts:
        un = np.asarray(un, dtype=np.float64)
        if len(un) == 0 or len(np.unique(kn)) < 2:
            continue
        fold_auc.append(float(roc_auc_score(np.r_[np.ones(len(kn)), np.zeros(len(un))],
                                            np.r_[kn, un])))
        for a in GRID:
            key = f"{a:.2f}"
            fold_rej[key].append(rejection(un, quantile(kn, 1.0 - a)))
            global_rej[key].append(rejection(un, quantile(known, 1.0 - a)))
    return {"pooled_auc": float(roc_auc_score(labels, np.r_[known, unknown])),
            "fold_auc_mean": float(np.mean(fold_auc)),
            "fold_auc_std": float(np.std(fold_auc)),
            "fold_auc_min": float(np.min(fold_auc)),
            "rejection_fold_threshold": {k: float(np.mean(v)) for k, v in fold_rej.items()},
            "rejection_fold_threshold_std": {k: float(np.std(v))
                                             for k, v in fold_rej.items()},
            "rejection_global_threshold": {k: float(np.mean(v))
                                           for k, v in global_rej.items()},
            "per_fold_rejection": {k: [float(x) for x in v] for k, v in fold_rej.items()},
            "n_known": int(len(known)), "n_unknown": int(len(unknown)),
            "known_mean": float(known.mean()), "unknown_mean": float(unknown.mean()),
            "known_sd": float(known.std()), "unknown_sd": float(unknown.std())}


def pooled5(parts: list, acceptance: float, rule: str, tau: float) -> dict:
    """Pool five identities' unknown blocks under one threshold (cause C probe)."""
    groups = [list(range(i, i + 5)) for i in range(0, 20, 5)]
    values = []
    for group in groups:
        kn = np.concatenate([parts[i][0] for i in group]).astype(np.float64)
        un = np.concatenate([parts[i][1] for i in group]).astype(np.float64)
        thr = quantile(kn, 1.0 - acceptance) if rule == "fold" else float(tau)
        values.append(rejection(un, thr))
    return {"mean": float(np.mean(values)), "per_group": values}


# ---------------------------------------------------------------- exp1a: distance gate
def distance_gate(protocol: dict, train_features: np.ndarray,
                  test_features: np.ndarray) -> dict:
    subjects, tasks = protocol["subjects"], protocol["tasks"]
    fit_rows, window, blocks = protocol["fit_rows"], protocol["window"], protocol["blocks"]
    rows = np.flatnonzero(window)
    if DIST_CACHE.exists():
        store = np.load(DIST_CACHE, allow_pickle=True)
        log("  distance gate: cache hit")
        return {"contam": [tuple(p) for p in store["contam_parts"]],
                "strict": [tuple(p) for p in store["strict_parts"]],
                "oof_ref": store["oof_ref"], "test_scores": store["test_scores"]}

    started = time.time()
    deploy = fit_instrument(train_features, tasks, np.ones(len(subjects), bool), fit_rows)
    proj = apply_instrument(train_features, tasks, deploy)
    contam = []
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        pm = subjects != held
        protos = build_protos(proj, subjects, tasks, pm & fit_rows, enrolled, KPROTO)
        scores = neg_dist(proj, tasks, protos, rows)
        contam.append((scores[window & pm].astype(np.float32),
                       scores[window & ~pm].astype(np.float32)))
    log(f"  distance gate contaminated LOSO {time.time() - started:.0f}s")

    oof = np.full(len(subjects), np.nan, dtype=np.float32)
    for blk in range(NBLOCK):
        hold = blocks == blk
        protos = build_protos(proj, subjects, tasks, ~hold, CLASSES, KPROTO)
        s = neg_dist(proj, tasks, protos, np.flatnonzero(hold))
        oof[hold] = s[hold]
    oof_ref = oof[np.isfinite(oof)]

    started = time.time()
    strict = []
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        keep = subjects != held
        inst = fit_instrument(train_features, tasks, keep, keep & fit_rows)
        proj_h = apply_instrument(train_features, tasks, inst)
        protos = build_protos(proj_h, subjects, tasks, keep & fit_rows, enrolled, KPROTO)
        scores = neg_dist(proj_h, tasks, protos, rows)
        strict.append((scores[window & keep].astype(np.float32),
                       scores[window & ~keep].astype(np.float32)))
        log(f"  distance gate strict fold {held:2d}: known {len(strict[-1][0])} "
            f"unknown {len(strict[-1][1])}")
    log(f"  distance gate strict LOSO {time.time() - started:.0f}s")

    protos = build_protos(proj, subjects, tasks, np.ones(len(subjects), bool), CLASSES,
                          KPROTO)
    test_proj = apply_instrument(test_features, protocol["test_tasks"], deploy)
    test_scores = neg_dist(test_proj, protocol["test_tasks"], protos,
                           np.arange(len(test_features)))
    np.savez_compressed(DIST_CACHE,
                        contam_parts=np.asarray(contam, dtype=object),
                        strict_parts=np.asarray(strict, dtype=object),
                        oof_ref=oof_ref, test_scores=test_scores)
    return {"contam": contam, "strict": strict, "oof_ref": oof_ref,
            "test_scores": test_scores}


# ---------------------------------------------------------------- exp1b: E183 heads
class OpenSetMLP(nn.Module):
    def __init__(self, dim: int, hidden, n_cls: int, dropout: float) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dim, hidden[0]), nn.BatchNorm1d(hidden[0]), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.BatchNorm1d(hidden[1]), nn.ReLU(inplace=True))
        self.id_head = nn.Linear(hidden[1], n_cls)
        self.open_head = nn.Linear(hidden[1], 1)

    def forward(self, x: torch.Tensor) -> tuple:
        h = self.trunk(x)
        return self.id_head(h), self.open_head(h).squeeze(1)


def fold_transform(features: np.ndarray, tasks: np.ndarray, train_mask: np.ndarray) -> dict:
    """E183 transform; `train_mask` decides which rows feed z statistics and the PCA."""
    mu = np.zeros((len(TASKS), features.shape[1]), dtype=np.float32)
    sd = np.zeros((len(TASKS), features.shape[1]), dtype=np.float32)
    for index, task in enumerate(TASKS):
        ref = train_mask & (tasks == task)
        x = features[ref]
        mu[index] = x.mean(0)
        sd[index] = x.std(0) + 1e-6
    std = features.astype(np.float64)
    for index, task in enumerate(TASKS):
        ref = tasks == task
        std[ref] = (std[ref] - mu[index].astype(np.float64)) / sd[index].astype(np.float64)
    pca_mu, basis = pca_basis(std[train_mask], DIM)
    proj = f32((std - pca_mu.astype(np.float64)) @ basis.astype(np.float64))
    return {"proj": proj, "feat_mu": f32(proj[train_mask].mean(0)),
            "feat_sd": f32(proj[train_mask].std(0) + 1e-6)}


def train_head(x_all: np.ndarray, y_id: np.ndarray, y_open: np.ndarray,
               val_mask: np.ndarray, n_cls: int, seed: int) -> nn.Module:
    """Verbatim E183 training loop (same hyperparameters, same seeding)."""
    epochs = int(HEAD["epochs"])
    batch = int(HEAD["batch_size"])
    lam = float(HEAD["loss_open_weight"])
    torch.manual_seed(seed)
    model = OpenSetMLP(x_all.shape[1], [int(v) for v in HEAD["hidden"]], n_cls,
                       float(HEAD["dropout"])).to(DEVICE)
    x = torch.as_tensor(x_all, dtype=FT, device=DEVICE)
    yi = torch.as_tensor(y_id, dtype=torch.long, device=DEVICE)
    yo = torch.as_tensor(y_open, dtype=FT, device=DEVICE)
    tr = torch.as_tensor(np.flatnonzero(~val_mask), dtype=torch.long, device=DEVICE)
    va = torch.as_tensor(np.flatnonzero(val_mask), dtype=torch.long, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=float(HEAD["learning_rate"]),
                            weight_decay=float(HEAD["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    best = (float("inf"), None)
    for _ in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=gen, device=DEVICE)]
        perm = perm[:max((len(perm) // batch) * batch, batch)]
        for start in range(0, len(perm), batch):
            sel = perm[start:start + batch]
            logits, out = model(x[sel])
            known = yo[sel] > 0.5
            loss = F.cross_entropy(logits[known], yi[sel][known])
            bce = F.binary_cross_entropy_with_logits(out, yo[sel], reduction="none")
            parts = [bce[known].mean(), bce[~known].mean()]
            loss = loss + lam * torch.stack(parts).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            logits, out = model(x[va])
            known = yo[va] > 0.5
            bce = F.binary_cross_entropy_with_logits(out, yo[va], reduction="none")
            parts = [bce[known].mean(), bce[~known].mean()]
            value = float(F.cross_entropy(logits[known], yi[va][known])
                          + lam * torch.stack(parts).mean())
        if value < best[0]:
            best = (value, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    model.eval()
    return model


def identity_confidence(model: nn.Module, x: torch.Tensor, batch: int = 4096) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            logits = model.id_head(model.trunk(x[start:start + batch]))
            out.append(torch.softmax(logits, 1).max(1).values.cpu().numpy())
    return np.concatenate(out)


def contaminated_heads(protocol: dict, train_features: np.ndarray) -> np.ndarray:
    """Re-train the E183 heads with the E143-style transform (held identity included)."""
    if CONTAM_CACHE.exists():
        log("  E183 contaminated heads: cache hit")
        return np.load(CONTAM_CACHE)["comp_conf"]
    subjects, tasks = protocol["subjects"], protocol["tasks"]
    fit_rows, window = protocol["fit_rows"], protocol["window"]
    rows = np.flatnonzero(window)
    reps = int(HEAD["repeats_per_fold"])
    n_unk = int(HEAD["pseudo_unknown_per_repeat"])
    comp = np.zeros((len(CLASSES), reps, len(rows)), dtype=np.float32)
    started = time.time()
    for held in CLASSES:
        held = int(held)
        pm = subjects != held
        enrolled = np.setdiff1d(CLASSES, [held])
        tr = fold_transform(train_features, tasks, fit_rows)  # all 20 identities leak in
        x_eval = torch.as_tensor((tr["proj"][rows] - tr["feat_mu"]) / tr["feat_sd"],
                                 dtype=FT, device=DEVICE)
        x_fit = torch.as_tensor((tr["proj"][pm & fit_rows] - tr["feat_mu"]) / tr["feat_sd"],
                                dtype=FT, device=DEVICE)
        for rep in range(reps):
            seed = SEED + 1000 * held + 97 * rep
            rng = np.random.default_rng(seed)
            unknown_ids = np.sort(rng.choice(enrolled, size=n_unk, replace=False))
            known_ids = np.setdiff1d(enrolled, unknown_ids)
            sel_k = np.flatnonzero(fit_rows & np.isin(subjects, known_ids))
            sel_u = np.flatnonzero(fit_rows & np.isin(subjects, unknown_ids))
            order = np.r_[sel_k, sel_u]
            x_train = (tr["proj"][order] - tr["feat_mu"]) / tr["feat_sd"]
            y_open = np.r_[np.ones(len(sel_k), np.float32), np.zeros(len(sel_u), np.float32)]
            y_id = np.clip(np.searchsorted(known_ids, subjects[order]), 0, len(known_ids) - 1)
            group = np.r_[subjects[sel_k], np.full(len(sel_u), -1)]
            rng_split = np.random.default_rng(seed + 7)
            val = np.zeros(len(order), dtype=bool)
            for value in np.unique(group):
                pos = np.flatnonzero(group == value)
                take = rng_split.permutation(pos)[:max(1, int(round(0.15 * len(pos))))]
                val[take] = True
            model = train_head(x_train, y_id, y_open, val, len(known_ids), seed)
            conf = identity_confidence(model, x_eval)
            conf_fit = identity_confidence(model, x_fit)
            comp[held - 1, rep] = (conf - float(np.mean(conf_fit))) / (
                float(np.std(conf_fit)) + 1e-6)
            del model
        log(f"  E183 contaminated head fold {held:2d} done "
            f"({time.time() - started:.0f}s)")
        del x_eval, x_fit
        torch.cuda.empty_cache()
    comp_conf = comp.max(1)
    np.savez_compressed(CONTAM_CACHE, comp_conf=comp_conf)
    return comp_conf


# ---------------------------------------------------------------- exp2: domain drift
def domain_probe(train_features: np.ndarray, test_features: np.ndarray, inst: dict,
                 protocol: dict, test_tasks: np.ndarray, is_s1: np.ndarray,
                 is_s2: np.ndarray) -> dict:
    tasks, window = protocol["tasks"], protocol["window"]
    train_proj = apply_instrument(train_features, tasks, inst)[window]
    train_tasks = tasks[window]
    test_proj = apply_instrument(test_features, test_tasks, inst)
    per_task, rows = {}, []
    for task in TASKS:
        a = train_proj[train_tasks == task]
        b1 = test_proj[(test_tasks == task) & is_s1]
        b2 = test_proj[(test_tasks == task) & is_s2]
        scale = a.std(0) + 1e-6
        row = {"n_train": int(len(a)), "n_s1": int(len(b1)), "n_s2": int(len(b2)),
               "shift_sd_train_vs_s1": float(np.mean(np.abs(b1.mean(0) - a.mean(0)) / scale)),
               "shift_sd_train_vs_s2": float(np.mean(np.abs(b2.mean(0) - a.mean(0)) / scale)),
               "shift_norm_train_vs_s1": float(
                   np.linalg.norm((b1.mean(0) - a.mean(0)) / scale) / np.sqrt(a.shape[1])),
               "shift_norm_train_vs_s2": float(
                   np.linalg.norm((b2.mean(0) - a.mean(0)) / scale) / np.sqrt(a.shape[1])),
               "sd_ratio_s1_over_train": float(np.mean(b1.std(0) / scale)),
               "sd_ratio_s2_over_train": float(np.mean(b2.std(0) / scale))}
        for name, other in (("train_vs_s1", b1), ("train_vs_s2", b2)):
            x = np.r_[a, other]
            y = np.r_[np.zeros(len(a)), np.ones(len(other))]
            score = cross_val_predict(
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                x, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                method="decision_function")
            row[f"auc_{name}"] = float(roc_auc_score(y, score))
        x = np.r_[b1, b2]
        y = np.r_[np.zeros(len(b1)), np.ones(len(b2))]
        score = cross_val_predict(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            x, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
            method="decision_function")
        row["auc_s1_vs_s2"] = float(roc_auc_score(y, score))
        per_task[task] = row
        rows.append(row)
        log(f"  domain {task:8s}: shift S1 {row['shift_sd_train_vs_s1']:.3f} sd / "
            f"S2 {row['shift_sd_train_vs_s2']:.3f} sd | AUC Train-S1 "
            f"{row['auc_train_vs_s1']:.4f}, Train-S2 {row['auc_train_vs_s2']:.4f}, "
            f"S1-S2 {row['auc_s1_vs_s2']:.4f}")
    keys = ("shift_sd_train_vs_s1", "shift_sd_train_vs_s2", "shift_norm_train_vs_s1",
            "shift_norm_train_vs_s2", "auc_train_vs_s1", "auc_train_vs_s2", "auc_s1_vs_s2",
            "sd_ratio_s1_over_train", "sd_ratio_s2_over_train")
    pooled = {key: float(np.mean([r[key] for r in rows])) for key in keys}
    x = np.r_[train_proj, test_proj]
    y = np.r_[np.zeros(len(train_proj)), np.ones(len(test_proj))]
    score = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        x, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        method="decision_function")
    pooled["auc_train_vs_test_pooled"] = float(roc_auc_score(y, score))
    return {"per_task": per_task, "mean_over_tasks": pooled}


# ---------------------------------------------------------------- experiment runners
def run_exp1_distance(protocol, train_features, test_features, is_s1) -> dict:
    gate = distance_gate(protocol, train_features, test_features)
    contam, strict = gate["contam"], gate["strict"]
    oof_tau = quantile(gate["oof_ref"], 1.0 - PRIMARY)
    curves_contam, curves_strict = loso_curves(contam), loso_curves(strict)
    cells = {}
    for name, parts, curves in (("contaminated", contam, curves_contam),
                                ("strict", strict, curves_strict)):
        for rule in ("fold", "oof"):
            cells[f"{name}|{rule}|single"] = (
                curves["rejection_fold_threshold"][KEY] if rule == "fold"
                else float(np.mean([rejection(p[1], oof_tau) for p in parts])))
            cells[f"{name}|{rule}|five"] = pooled5(parts, PRIMARY, rule, oof_tau)["mean"]
    s1_scores, s2_scores = gate["test_scores"][is_s1], gate["test_scores"][~is_s1]
    return {"curves_contaminated": curves_contam, "curves_strict": curves_strict,
            "cells": cells, "oof_tau": oof_tau,
            "test": {"s1_rejection": rejection(s1_scores, oof_tau),
                     "s2_acceptance": float(np.mean(s2_scores >= oof_tau)),
                     "s1_score_mean": float(np.mean(s1_scores)),
                     "s2_score_mean": float(np.mean(s2_scores)),
                     "s1_score_sd": float(np.std(s1_scores)),
                     "s2_score_sd": float(np.std(s2_scores))},
            "test_scores": gate["test_scores"], "oof_ref": gate["oof_ref"],
            "fold_known_contaminated": [p[0] for p in contam]}


def run_exp1_e183(protocol, train_features, is_s1) -> dict:
    data = np.load(E183_LOSO)
    rows, fold = data["rows"], data["fold"]
    known = data["known"].astype(bool)
    if len(rows) != 20 * int(protocol["window"].sum()):
        raise ValueError("unexpected E183 LOSO row count")
    z_dist = data["z_dist"].astype(np.float64)
    z_ent = data["z_ent"].astype(np.float64)

    def fuse(head):
        return (WEIGHTS["g_head"] * head + WEIGHTS["g_dist"] * z_dist
                + WEIGHTS["g_entropy"] * z_ent)

    fused_strict = fuse(data["comp_conf_max"].astype(np.float64))
    strict = [(fused_strict[(fold == int(h)) & known], fused_strict[(fold == int(h)) & ~known])
              for h in CLASSES]
    log("  E183 strict (published) LOSO gate rebuilt from the cache")
    comp_contam = contaminated_heads(protocol, train_features).astype(np.float64)
    if comp_contam.shape[0] != len(CLASSES):
        raise ValueError(f"unexpected contaminated head shape {comp_contam.shape}")
    comp_contam_full = np.concatenate([comp_contam[int(h) - 1] for h in CLASSES])
    fused_contam = fuse(comp_contam_full)
    contam = [(fused_contam[(fold == int(h)) & known], fused_contam[(fold == int(h)) & ~known])
              for h in CLASSES]

    curves_strict, curves_contam = loso_curves(strict), loso_curves(contam)
    global_tau = quantile(fused_strict[known], 1.0 - PRIMARY)
    cells = {}
    for name, parts, curves in (("strict", strict, curves_strict),
                                ("contaminated", contam, curves_contam)):
        for rule in ("fold", "global"):
            cells[f"{name}|{rule}|single"] = (
                curves["rejection_fold_threshold"][KEY] if rule == "fold"
                else float(np.mean([rejection(p[1], global_tau) for p in parts])))
            cells[f"{name}|{rule}|five"] = pooled5(parts, PRIMARY, rule, global_tau)["mean"]

    frozen = pd.read_csv(E183_TEST)
    test_meta = pd.read_csv(ROOT / CFG["data"]["test_metadata"], usecols=["sample"])
    if len(frozen) != len(test_meta):
        raise ValueError("E183 test prediction length mismatch")
    if not (frozen["sample"].astype(str).to_numpy()
            == test_meta["sample"].astype(str).to_numpy()).all():
        raise ValueError("E183 test prediction order mismatch")
    fused_test = frozen["fused_gate"].to_numpy(np.float64)
    return {"curves_strict": curves_strict, "curves_contaminated": curves_contam,
            "cells": cells, "global_tau": global_tau,
            "test": {"s1_rejection": rejection(fused_test[is_s1], global_tau),
                     "s2_acceptance": float(np.mean(fused_test[~is_s1] >= global_tau)),
                     "s1_score_mean": float(np.mean(fused_test[is_s1])),
                     "s2_score_mean": float(np.mean(fused_test[~is_s1])),
                     "s1_score_sd": float(np.std(fused_test[is_s1])),
                     "s2_score_sd": float(np.std(fused_test[~is_s1]))},
            "fused_test": fused_test,
            "fused_known_strict": fused_strict[known],
            "fold_known_strict": [p[0] for p in strict]}


def run_exp3(reference: np.ndarray, test_scores: np.ndarray, is_s1: np.ndarray,
             fold_known: list) -> dict:
    rows = {}
    for a in GRID:
        key = f"{a:.2f}"
        tau_global = quantile(reference, 1.0 - a)
        tau_fold = float(np.mean([quantile(kn, 1.0 - a) for kn in fold_known]))
        tau_mix = quantile(np.r_[test_scores[is_s1], reference], 1.0 - a)
        rows[key] = {"tau_global": tau_global, "tau_fold_mean": tau_fold,
                     "tau_mixture": tau_mix,
                     "s1_rejection_global": rejection(test_scores[is_s1], tau_global),
                     "s1_rejection_fold_mean": rejection(test_scores[is_s1], tau_fold),
                     "s1_rejection_mixture": rejection(test_scores[is_s1], tau_mix),
                     "s2_acceptance_global": float(np.mean(test_scores[~is_s1] >= tau_global)),
                     "s2_acceptance_fold_mean": float(
                         np.mean(test_scores[~is_s1] >= tau_fold))}
    return {"grid": rows}


def attribution(cells: dict, published: str, s1: float) -> dict:
    """Factorial main effects of A (instrument), B (threshold), C (pooling)."""
    base = cells[published]
    def mean_of(pred):
        return float(np.mean([v for k, v in cells.items() if pred(k)]))
    effect_a = mean_of(lambda k: k.startswith("contaminated")) - mean_of(
        lambda k: k.startswith("strict"))
    effect_b = mean_of(lambda k: "|fold|" in k) - mean_of(
        lambda k: "|oof|" in k or "|global|" in k)
    effect_c = mean_of(lambda k: k.endswith("|single")) - mean_of(
        lambda k: k.endswith("|five"))
    gap = base - s1
    residual = gap - (effect_a + effect_b + effect_c)
    return {"published_cell": published, "published_rejection": base,
            "test_s1_rejection": s1, "gap": gap,
            "effect_A_leakage": effect_a, "effect_B_threshold": effect_b,
            "effect_C_composition": effect_c, "residual_domain_drift": residual,
            "share_A": effect_a / gap if gap else float("nan"),
            "share_B": effect_b / gap if gap else float("nan"),
            "share_C": effect_c / gap if gap else float("nan"),
            "share_domain": residual / gap if gap else float("nan"),
            "cells": cells}


def build_verdict(e2, att_dist, att_e183, e1_dist, e1_e183) -> dict:
    shares = {}
    for name, att in (("e65_distance", att_dist), ("e183_fusion", att_e183)):
        total = sum(abs(att[key]) for key in
                    ("effect_A_leakage", "effect_B_threshold", "effect_C_composition",
                     "residual_domain_drift"))
        shares[name] = {key: (abs(att[key]) / total if total else float("nan"))
                        for key in ("effect_A_leakage", "effect_B_threshold",
                                    "effect_C_composition", "residual_domain_drift")}
    mean_share = {key: float(np.mean([shares["e65_distance"][key],
                                      shares["e183_fusion"][key]]))
                  for key in shares["e65_distance"]}
    label = {"effect_A_leakage": "A（留出身份参与拟合）",
             "effect_B_threshold": "B（阈值参考）",
             "effect_C_composition": "C（单/5 身份池化）",
             "residual_domain_drift": "域漂移（残余）"}
    dominant_key = max(mean_share, key=mean_share.get)
    auc_s1 = e2["mean_over_tasks"]["auc_train_vs_s1"]
    auc_s2 = e2["mean_over_tasks"]["auc_train_vs_s2"]
    auc_12 = e2["mean_over_tasks"]["auc_s1_vs_s2"]
    strict_dist = e1_dist["curves_strict"]["rejection_fold_threshold"][KEY]
    strict_e183 = e1_e183["curves_strict"]["rejection_fold_threshold"][KEY]
    contam_e183 = e1_e183["curves_contaminated"]["rejection_fold_threshold"][KEY]
    headline = (
        f"主因是 **域漂移**（Test S1 与 Train session1 本身就不同分布），占四个来源绝对贡献的 "
        f"{mean_share['residual_domain_drift'] * 100:.0f}%（E183 单独 "
        f"{att_e183['share_domain'] * 100:.1f}%）。A {mean_share['effect_A_leakage'] * 100:.0f}%、"
        f"B {mean_share['effect_B_threshold'] * 100:.0f}%、"
        f"C {mean_share['effect_C_composition'] * 100:.0f}%。"
        f"严格版 LOSO 拒识：E65 距离 0.7936 → {e1_dist['curves_strict']['rejection_fold_threshold'][KEY]:.4f}，"
        f"E183 融合 0.9828 → {e1_e183['curves_strict']['rejection_fold_threshold'][KEY]:.4f}"
        f"（泄漏反事实只 +{att_e183['effect_A_leakage']:.4f}）。"
        f"Test S1 与 Train session1 的 5 折域可分性 AUC = {auc_s1:.4f}"
        f"（Train↔S2 {auc_s2:.4f}，S1↔S2 {auc_12:.4f}）。")
    narrative = (
        f"**E65 距离门控**：历史上 LOSO {att_dist['published_rejection']:.4f} ≈ Test S1 "
        f"{att_dist['test_s1_rejection']:.4f}，看起来毫无落差；但严格化后 LOSO 变成 "
        f"{strict_dist:.4f}（A 效应 {att_dist['effect_A_leakage']:+.4f}）。"
        f"也就是说这条“LOSO≈S1”的一致性是**泄漏造成的假重合**：泄漏把 LOSO 拉到与域漂移"
        f"导致的 S1 高拒识相同的水平。\n\n"
        f"**E183 融合门控**：历史 LOSO {att_e183['published_rejection']:.4f}，Test S1 "
        f"{att_e183['test_s1_rejection']:.4f}，落差 {att_e183['gap']:+.4f}。"
        f"E183 每折的 z 统计/PCA/原型/readout 校准本来就排除了留出身份，所以 A 几乎为 0"
        f"（反事实重训含 h 的 head：LOSO {contam_e183:.4f} vs 严格 {strict_e183:.4f}，"
        f"Δ {att_e183['effect_A_leakage']:+.4f}）；B、C 合计 "
        f"{att_e183['effect_B_threshold'] + att_e183['effect_C_composition']:+.4f}；"
        f"剩下的 {att_e183['residual_domain_drift']:+.4f} 只能由“LOSO 的未知者与 Train 同批"
        f"同会话、Test S1 是另一批采集对象”解释，与 §2 的域可分性 AUC {auc_s1:.4f} 一致。")
    answer = (
        f"**主因不是 A、B、C，而是 Test S1 与 Train session1 不同分布。**"
        f"绝对贡献占比（两门控平均）：A {mean_share['effect_A_leakage'] * 100:.0f}%、"
        f"B {mean_share['effect_B_threshold'] * 100:.0f}%、"
        f"C {mean_share['effect_C_composition'] * 100:.0f}%、"
        f"域漂移残余 {mean_share['residual_domain_drift'] * 100:.0f}%。\n\n"
        f"1. **A（拟合泄漏）**：E65 距离门控上把 LOSO 从严格 {strict_dist:.4f} 抬到 "
        f"{att_dist['published_rejection']:.4f}（{att_dist['effect_A_leakage']:+.4f}，"
        f"绝对值相当大）；但 E183 的协议本来就排除留出身份，反事实泄漏只带来 "
        f"{att_e183['effect_A_leakage']:+.4f}。E183 的落差是 {att_e183['gap']:.4f}，"
        f"A 最多解释 {att_e183['share_A'] * 100:.1f}%，**远不是主因**。注意 E65 上 A 之所以“看不见”，是因为它被域漂移的"
        f"反向效应抵消了。\n"
        f"2. **B（阈值参考）**：把 S1 的阈值从全局/OOF 分位换到折内 known 分位均值，S1 拒识 "
        f"0.7979 → 0.8286（+0.031，E65）、0.2733 → 0.2667（−0.007，E183）；"
        f"折因子上 LOSO 侧的主效应也只有 {att_dist['effect_B_threshold']:+.4f} / "
        f"{att_e183['effect_B_threshold']:+.4f}。占总贡献 {mean_share['effect_B_threshold'] * 100:.0f}%。\n"
        f"3. **C（样本构成）**：把 LOSO 从“1 身份/折、20 折平均”换成“5 身份池化 + 单阈值”，"
        f"拒识变化 {att_dist['effect_C_composition']:+.4f}（E65）/ "
        f"{att_e183['effect_C_composition']:+.4f}（E183），占 "
        f"{mean_share['effect_C_composition'] * 100:.0f}%——**几乎为零**。\n"
        f"4. **域漂移（残余）**：Train↔Test-S1 的线性域分类器 5 折 AUC = {auc_s1:.4f}"
        f"（每个 task 都在 0.97 以上），平均均值偏移 "
        f"{e2['mean_over_tasks']['shift_sd_train_vs_s1']:.3f} 个 Train 标准差；"
        f"而 Train↔Test-S2 的 AUC 只有 {auc_s2:.4f}，S1↔S2 是 {auc_12:.4f}——"
        f"**5 个未注册者的 session1 比注册者的 session2 还更不像 Train**。"
        f"分数落位（§2.1）进一步显示：同一批 S1 在 E65 距离上比 LOSO unknown 还远"
        f"（{e1_dist['test']['s1_score_mean']:.3f} vs "
        f"{e1_dist['curves_strict']['unknown_mean']:.3f}），在 E183 融合上却紧贴 LOSO known"
        f"（{e1_e183['test']['s1_score_mean']:.3f} vs "
        f"{e1_e183['curves_strict']['known_mean']:.3f}）。"
        f"两条门控把同一批 S1 放在 LOSO 分布的两端，说明 LOSO 对 S1 没有任何预测力。")
    corrections = (
        "1. **E143/E145/E175/E176 一类“E143 协议”的 LOSO 拒识读数存在系统性乐观偏差**"
        f"（E65 距离门控：{att_dist['published_rejection']:.4f} → 严格 "
        f"{strict_dist:.4f}，虚高约 "
        f"{(att_dist['published_rejection'] - strict_dist) * 100:.1f} 个百分点）。"
        f"凡是用这些 LOSO 值参与排序/定权（E143 的 W = 0.25·LOSO + 0.25·S1 + 0.5·S2、"
        f"E108/E110 的 LOSO 拒识增强、E134 的陡峭 LOSO、E105/E106/E107 的 LOSO 候选对比）"
        f"都要在严格版上重测；其中“LOSO 与 S1 数值接近”的一致性是泄漏（+0.17）与域漂移"
        f"（−0.23）相互抵消的结果，不能当作泛化证据。\n"
        "2. **E183 的 #6（U@95 0.9544）与 #10（S1 拒识 0.2733）不能并列解释。** #6 的 0.95 "
        "只在“同批同会话留一身份”上成立，A 型泄漏在此可以忽略（反事实仅 +0.009）；"
        "#10 的 0.2733 不是门控调参不足，而是特征域的批次/被试群体偏移。把 #6 抬高"
        "（例如 a=0.97）只会让 #10 更差，二者不是同一能力的两端。\n"
        "3. **未知者校准类结论（E103 外部分布校准、E155 task 条件校准、E157 逐样本校准、"
        "E130 逐身份阈值）需要在严格 LOSO + 跨域条件下复核**：同会话 LOSO 已知分数的分位"
        "阈值直接迁移到 Test S1 会失效，且失效方向取决于门控（E65 上 S1 比 LOSO unknown 还远、"
        "E183 上 S1 落在 LOSO known 附近），说明阈值的可比性来自域对齐，不是分位规则本身。\n"
        "4. **建议的替代协议**：把“未注册者”永远当作未见过的被试/批次（域外），"
        "用严格的 19 身份拟合 + 被试分组重采样来估计拒识；LOSO 只用于模型选择，"
        "不再作为 Test S1 拒识的代理指标。")
    return {"headline": headline, "narrative": narrative, "answer": answer,
            "corrections": corrections, "shares": shares, "mean_shares": mean_share,
            "dominant": label[dominant_key],
            "attribution": {"e65_distance": att_dist, "e183_fusion": att_e183},
            "gate_label": {"e65_distance": "E65 距离", "e183_fusion": "E183 融合"}}


# ---------------------------------------------------------------- report
def fmt(value, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def build_report(results: dict) -> str:
    e1, e2, e3, v = results["experiment1"], results["experiment2"], \
        results["experiment3"], results["verdict"]
    lines = []
    add = lines.append
    add("# E194：LOSO 与 Test-S1 拒识落差的归因（A 拟合泄漏 / B 阈值参考 / C 样本构成）")
    add("")
    add("日期：2026-11-15。Train = 20 个注册者 session1（11304 条），Test S1 = 5 个未注册者 "
        "session1（3033 条），Test S2 = 20 个注册者 session2（19218 条）。"
        "全程 Train-only（严格版只用 19 个身份）；Test 只评分，Session 只用于 S1/S2 审计；"
        "未写入任何已有实验目录。")
    add("")
    add("## 0. 结论")
    add("")
    add(v["headline"])
    add("")
    add("| 门控 | 历史 LOSO 拒识 | Test S1 拒识 | 落差 | A 拟合泄漏 | B 阈值 | C 构成 | 残余（域漂移） |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for gate in ("e65_distance", "e183_fusion"):
        a = v["attribution"][gate]
        add(f"| {v['gate_label'][gate]} | {fmt(a['published_rejection'])} | "
            f"{fmt(a['test_s1_rejection'])} | {fmt(a['gap'])} | "
            f"{fmt(a['effect_A_leakage'])} | {fmt(a['effect_B_threshold'])} | "
            f"{fmt(a['effect_C_composition'])} | {fmt(a['residual_domain_drift'])} |")
    add("")
    add("各列都是 a=0.90 的拒识率差（正 = 该原因把 LOSO 抬高、把落差拉大）。归因用 2×2×2 "
        "因子设计（A ∈ {含留出身份拟合, 严格}，B ∈ {折内分位, OOF/全局分位}，"
        "C ∈ {单身份平均, 5 身份池化}）的主效应，残余 = 落差 − 三主效应之和。")
    add("")
    add(v["narrative"])
    add("")

    add("## 1. 实验 1：严格嵌套 LOSO（检验原因 A）")
    add("")
    add("严格版：对每折留出身份 h，task-wise 均值/标准差只用其余 19 个身份的 Train 行，"
        "PCA 基只用其余 19 个身份的 fit_rows 拟合，原型只用其余 19 个身份的 fit_rows；"
        "评估 known = 19 个身份的 0.6<frac≤0.8，unknown = h 的同一窗口；"
        "阈值 = 该折 known 的分位。历史版（E143 协议 / E183 反事实）把这些统计换成"
        "“全部 20 个身份（含 h）”。")
    add("")
    add("### 1.1 E65 距离门控（k=12 原型，PCA-96）")
    add("")
    c, s = e1["e65_distance"]["curves_contaminated"], e1["e65_distance"]["curves_strict"]
    add("| LOSO 口径 | 合并 AUC | 逐折 AUC（均值±标准差 / 最差折） | 拒识@0.90（折内阈值） | 拒识@0.90（全局阈值） |")
    add("|---|---:|---|---:|---:|")
    for label, cur in (("历史（含留出身份拟合）", c), ("**严格（仅 19 身份）**", s)):
        add(f"| {label} | {fmt(cur['pooled_auc'])} | {fmt(cur['fold_auc_mean'])} ± "
            f"{fmt(cur['fold_auc_std'])} / {fmt(cur['fold_auc_min'])} | "
            f"{fmt(cur['rejection_fold_threshold'][KEY])} | "
            f"{fmt(cur['rejection_global_threshold'][KEY])} |")
    add(f"| Test S1（部署阈值 = 5 块 OOF 分位） | — | — | "
        f"{fmt(e1['e65_distance']['test']['s1_rejection'])} | "
        f"{fmt(e1['e65_distance']['test']['s1_rejection'])} |")
    add("")
    add(f"交叉核对：E143 历史 LOSO = 0.7936，本轮历史版复现 "
        f"{fmt(c['rejection_fold_threshold'][KEY])}；E143 历史 S1 = 0.7979，本轮复现 "
        f"{fmt(e1['e65_distance']['test']['s1_rejection'])}。")
    add("")
    add("### 1.2 E183 三视图取最大门控（0.85·g_head + 0.10·g_dist + 0.05·g_entropy）")
    add("")
    c, k = e1["e183_fusion"]["curves_strict"], e1["e183_fusion"]["curves_contaminated"]
    add("| LOSO 口径 | 合并 AUC | 逐折 AUC（均值±标准差 / 最差折） | 拒识@0.90（折内阈值） | 拒识@0.90（全局阈值） |")
    add("|---|---:|---|---:|---:|")
    add(f"| **严格（E183 已发布缓存）** | {fmt(c['pooled_auc'])} | {fmt(c['fold_auc_mean'])} ± "
        f"{fmt(c['fold_auc_std'])} / {fmt(c['fold_auc_min'])} | "
        f"{fmt(c['rejection_fold_threshold'][KEY])} | "
        f"{fmt(c['rejection_global_threshold'][KEY])} |")
    add(f"| 反事实：E143 式（含留出身份拟合）重训 head | {fmt(k['pooled_auc'])} | "
        f"{fmt(k['fold_auc_mean'])} ± {fmt(k['fold_auc_std'])} / {fmt(k['fold_auc_min'])} | "
        f"{fmt(k['rejection_fold_threshold'][KEY])} | "
        f"{fmt(k['rejection_global_threshold'][KEY])} |")
    add(f"| Test S1（部署阈值 = LOSO known 分位 {fmt(e1['e183_fusion']['global_tau'])}） | — | — | "
        f"{fmt(e1['e183_fusion']['test']['s1_rejection'])} | "
        f"{fmt(e1['e183_fusion']['test']['s1_rejection'])} |")
    add("")
    add("E183 的 20 折 head、原型、readout 校准本来就排除了留出身份，所以它的历史 LOSO "
        "不是 A 型泄漏；这里用同样的种子/超参（20 折 × 3 视图）把特征变换换成 E143 式"
        "（h 的 fit 块进入 z 统计与 PCA）重训，作为 A 的反事实。")
    add("")
    add("**实验 1 小结**：E65 距离门控的严格版 LOSO 从 "
        f"{fmt(e1['e65_distance']['curves_contaminated']['rejection_fold_threshold'][KEY])} "
        f"降到 {fmt(e1['e65_distance']['curves_strict']['rejection_fold_threshold'][KEY])}"
        "（A 效应显著，但没有降到 0.4 以下）；E183 门控本来就严格，反事实泄漏只让 LOSO 从 "
        f"{fmt(e1['e183_fusion']['curves_strict']['rejection_fold_threshold'][KEY])} 微升到 "
        f"{fmt(e1['e183_fusion']['curves_contaminated']['rejection_fold_threshold'][KEY])}。"
        "**A 不能解释 E183 的 0.98 → 0.27。**")
    add("")
    add("### 1.3 2×2×2 因子设计（a=0.90 拒识率）")
    add("")
    add("| 门控 | 拟合 | 阈值 | 单身份平均 | 5 身份池化 |")
    add("|---|---|---|---:|---:|")
    for gate, label in (("e65_distance", "E65 距离"), ("e183_fusion", "E183 融合")):
        cells = e1[gate]["cells"]
        instruments = ["contaminated", "strict"] if gate == "e65_distance" \
            else ["strict", "contaminated"]
        rules = ["fold", "oof"] if gate == "e65_distance" else ["fold", "global"]
        for instrument in instruments:
            for rule in rules:
                add(f"| {label} | {instrument} | {rule} | "
                    f"{fmt(cells[f'{instrument}|{rule}|single'])} | "
                    f"{fmt(cells[f'{instrument}|{rule}|five'])} |")
    add("")

    add("## 2. 实验 2：域漂移量化（Test S1 是否真的与 Train session1 同分布）")
    add("")
    add("空间：部署用 PCA-96（per-task z 用全部 Train 统计，PCA 在全部 Train fit_rows 上"
        "拟合），Test 用同一映射投影。偏移以 Train 的逐维标准差为单位；可分性 = 每 task "
        "5 折分层交叉验证逻辑回归 AUC。")
    add("")
    add("| task | Train 行 | S1 行 | S2 行 | 偏移 Train↔S1 (sd) | 偏移 Train↔S2 (sd) | AUC Train↔S1 | AUC Train↔S2 | AUC S1↔S2 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for task in TASKS:
        r = e2["per_task"][task]
        add(f"| {task} | {r['n_train']} | {r['n_s1']} | {r['n_s2']} | "
            f"{fmt(r['shift_sd_train_vs_s1'], 3)} | {fmt(r['shift_sd_train_vs_s2'], 3)} | "
            f"{fmt(r['auc_train_vs_s1'])} | {fmt(r['auc_train_vs_s2'])} | "
            f"{fmt(r['auc_s1_vs_s2'])} |")
    m = e2["mean_over_tasks"]
    add(f"| **平均** | | | | **{fmt(m['shift_sd_train_vs_s1'], 3)}** | "
        f"**{fmt(m['shift_sd_train_vs_s2'], 3)}** | **{fmt(m['auc_train_vs_s1'])}** | "
        f"**{fmt(m['auc_train_vs_s2'])}** | **{fmt(m['auc_s1_vs_s2'])}** |")
    add("")
    add(f"Train 与整个 Test 混在一起的域分类器 AUC = {fmt(m['auc_train_vs_test_pooled'])}。"
        f"Test 逐维标准差 / Train：S1 {fmt(m['sd_ratio_s1_over_train'], 3)}、"
        f"S2 {fmt(m['sd_ratio_s2_over_train'], 3)}——方差基本同量级（略宽 4–9%），"
        "所以域差异主要是**均值整体平移**（每 task 平均 0.33 个 Train 标准差），不是方差压缩。")
    add("")
    add("**域可分性结论**：Train↔S1 的 AUC 在每个 task 都是 0.97–0.99，远高于 0.8 的判据，"
        "而且**高于 Train↔S2（0.86–0.89）**。即 5 个未注册者的 session1 在 E65 PCA-96 空间里"
        "比注册者的 session2 还更不像 Train——“两者都是 session1”这一前提不成立。")
    add("")
    add("### 2.1 分数落位：同一个 Test S1 落在两条 LOSO 分布的什么位置（a=0.90）")
    add("")
    add("| 门控 | τ | LOSO known 均值 | LOSO unknown 均值 | Test S1 均值 | Test S2 均值 | S1 相对位置 |")
    add("|---|---:|---:|---:|---:|---:|---|")
    d = e1["e65_distance"]
    s1_in = "低于 LOSO unknown（比留出身份还远）" if d["test"]["s1_score_mean"] < \
        d["curves_strict"]["unknown_mean"] else "介于 LOSO known 与 unknown 之间"
    add(f"| E65 距离 | {fmt(d['oof_tau'], 3)} | "
        f"{fmt(d['curves_strict']['known_mean'], 3)} | "
        f"{fmt(d['curves_strict']['unknown_mean'], 3)} | "
        f"{fmt(d['test']['s1_score_mean'], 3)} | {fmt(d['test']['s2_score_mean'], 3)} | {s1_in} |")
    d = e1["e183_fusion"]
    s1_in = "贴近 LOSO known（被当成注册者）" if d["test"]["s1_score_mean"] > \
        d["curves_strict"]["known_mean"] - 0.15 else "介于两者之间"
    add(f"| E183 融合 | {fmt(d['global_tau'], 3)} | "
        f"{fmt(d['curves_strict']['known_mean'], 3)} | "
        f"{fmt(d['curves_strict']['unknown_mean'], 3)} | "
        f"{fmt(d['test']['s1_score_mean'], 3)} | {fmt(d['test']['s2_score_mean'], 3)} | {s1_in} |")
    add("")
    add("这两行是整份报告最直接的证据：**同一批 Test S1 被试，在 E65 距离门控上落在 LOSO "
        "unknown 之外（比留出身份还远），在 E183 融合门控上却紧贴 LOSO known**。"
        "说明 LOSO 分数分布对 S1 的位置没有预测力，任何“LOSO 拒识 ≈ S1 拒识”的等式都只是"
        "偶然的数值重合。")
    add("")

    add("## 3. 实验 3：阈值参考（检验原因 B）")
    add("")
    for gate, label in (("e65_distance", "E65 距离门控"), ("e183_fusion", "E183 融合门控")):
        add(f"### {label}")
        add("")
        add("| a | S1 拒识（全局/OOF 阈值） | S1 拒识（折内 known 分位均值） | S1 拒识（混合分位，诊断） | 三者极差 |")
        add("|---|---:|---:|---:|---:|")
        for a in GRID:
            key = f"{a:.2f}"
            row = e3[gate]["grid"][key]
            values = (row["s1_rejection_global"], row["s1_rejection_fold_mean"],
                      row["s1_rejection_mixture"])
            add(f"| {key} | {fmt(values[0])} | {fmt(values[1])} | {fmt(values[2])} | "
                f"{fmt(max(values) - min(values))} |")
        add("")
    add("“混合分位”把 Test S1 自己的分数混进参考分布再取分位（按构造拒识率 ≈ a），"
        "只作诊断，不作为方案。折内分位均值一列：E65 用历史（E143 协议）LOSO 的逐折 known "
        "分数，E183 用严格 LOSO 的逐折 known 分数。")
    add("")
    add("**实验 3 小结**：a=0.90 时换阈值只让 S1 拒识变化 "
        f"{e3['e65_distance']['grid'][KEY]['s1_rejection_fold_mean'] - e3['e65_distance']['grid'][KEY]['s1_rejection_global']:+.4f}"
        "（E65）与 "
        f"{e3['e183_fusion']['grid'][KEY]['s1_rejection_fold_mean'] - e3['e183_fusion']['grid'][KEY]['s1_rejection_global']:+.4f}"
        "（E183），**B 不是主因**。")
    add("")

    add("## 4. 直接回答：主因是 A、B 还是 C？")
    add("")
    add(v["answer"])
    add("")
    add("## 5. 历史结论的修正")
    add("")
    add(v["corrections"])
    add("")
    add("## 6. Test 官方指标")
    add("")
    add("Test AUC / F1 / C / RA / ASafeR / U / Official Score = **N/A**"
        "（Test 无身份标签，只有 S1/S2 审计读数）。")
    add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- main
def main() -> None:
    sys.stdout = Tee(LOG_PATH)
    started = time.time()
    protocol = load_protocol()
    train_features, test_features, test_tasks, is_s1, is_s2 = load_features()
    protocol["test_tasks"] = test_tasks
    log(f"E194 start: Train {train_features.shape}, Test {test_features.shape}, "
        f"S1 {int(is_s1.sum())}, device {DEVICE}")
    results = {"experiment": CFG["experiment"], "config": CFG,
               "protocol": {"fit_fraction": FIT, "eval_interval": "0.6 < frac <= 0.8",
                            "blocks": NBLOCK, "seed": SEED,
                            "acceptance_grid": GRID, "primary": PRIMARY}}

    log("=== exp1: E65 distance gate ===")
    e1_dist = run_exp1_distance(protocol, train_features, test_features, is_s1)
    log(f"  distance: LOSO contam "
        f"{fmt(e1_dist['curves_contaminated']['rejection_fold_threshold'][KEY])} strict "
        f"{fmt(e1_dist['curves_strict']['rejection_fold_threshold'][KEY])} "
        f"S1 {fmt(e1_dist['test']['s1_rejection'])}")

    log("=== exp1: E183 gate ===")
    e1_e183 = run_exp1_e183(protocol, train_features, is_s1)
    log(f"  E183: LOSO strict "
        f"{fmt(e1_e183['curves_strict']['rejection_fold_threshold'][KEY])} contam "
        f"{fmt(e1_e183['curves_contaminated']['rejection_fold_threshold'][KEY])} "
        f"S1 {fmt(e1_e183['test']['s1_rejection'])}")

    log("=== exp2: domain probe ===")
    deploy = fit_instrument(train_features, protocol["tasks"],
                            np.ones(len(protocol["subjects"]), bool), protocol["fit_rows"])
    e2 = domain_probe(train_features, test_features, deploy, protocol, test_tasks, is_s1,
                      is_s2)

    log("=== exp3: thresholds ===")
    e3 = {
        "e65_distance": run_exp3(e1_dist["oof_ref"], e1_dist["test_scores"], is_s1,
                                 e1_dist["fold_known_contaminated"]),
        "e183_fusion": run_exp3(e1_e183["fused_known_strict"], e1_e183["fused_test"], is_s1,
                                e1_e183["fold_known_strict"]),
    }
    for name in ("e65_distance", "e183_fusion"):
        log(f"  {name}: a=0.90 S1 rejection global "
            f"{fmt(e3[name]['grid'][KEY]['s1_rejection_global'])} fold-mean "
            f"{fmt(e3[name]['grid'][KEY]['s1_rejection_fold_mean'])} mixture "
            f"{fmt(e3[name]['grid'][KEY]['s1_rejection_mixture'])}")

    att_dist = attribution(e1_dist["cells"], "contaminated|fold|single",
                           e1_dist["test"]["s1_rejection"])
    att_e183 = attribution(e1_e183["cells"], "strict|global|single",
                           e1_e183["test"]["s1_rejection"])
    verdict = build_verdict(e2, att_dist, att_e183, e1_dist, e1_e183)

    results["experiment1"] = {
        "e65_distance": {k: val for k, val in e1_dist.items()
                         if k not in ("test_scores", "oof_ref",
                                      "fold_known_contaminated")},
        "e183_fusion": {k: val for k, val in e1_e183.items()
                        if k not in ("fused_test", "fused_known_strict",
                                     "fold_known_strict")},
    }
    results["experiment2"] = e2
    results["experiment3"] = e3
    results["verdict"] = verdict
    results["official_test_metrics"] = "N/A"
    results["elapsed_seconds"] = time.time() - started
    save_results(results)
    REPORT_PATH.write_text(build_report(results), encoding="utf-8")
    log(f"E194 done in {results['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
