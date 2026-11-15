"""E183: dedicated open-set gate head on E65 features + gate fusion + capacity probe.

Only gates #1/#5/#6/#7 are targeted.  Nothing in any existing experiment directory is
modified; branch A (E182 v2) and branch B (E180) are loaded read-only.

Head  : 2-layer MLP (96 -> 128 -> 64) with two heads (identity softmax over the
        identities a fold trained as registered, plus a registered/unregistered
        logistic head) on E65 correlation-logSPD -> per-task z -> PCA-96.
Nesting: outer fold holds out one identity (never in any head / PCA / statistic /
        threshold); inside the 19 enrolled identities 4 are drawn as pseudo-unknown
        (3 independent draws per fold) and are excluded from the identity head.
Readouts: the head exposes three novelty read-outs per inner view --
        open-set logit, identity-head confidence, trunk nearest-centroid distance.
        One read-out is chosen on the pooled Train LOSO AUC (Train-only).
Gate  : g = w_head*z(g_head) + w_dist*z(g_dist) + w_ent*z(g_entropy), weights chosen
        only on the pooled Train LOSO AUC.  tau(a) = quantile of the pooled Train
        LOSO known score at 1 - a.

Stages: head, fusion, capacity, test, attack, report, all.
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
FOLD_DIR = CACHE / "head_folds"
FOLD_DIR.mkdir(exist_ok=True)
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "report.md"
LOG_PATH = OUT / "run.log"
LOSO_CACHE = CACHE / "loso_head.npz"
TEST_PRED = OUT / "test_predictions_fused.csv"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E180 = load_module("e183_frozen_e180",
                   ROOT / "research" / CFG["frozen_sources"]["e180"] / "run_e180.py")
E182_DIR = ROOT / "research" / CFG["frozen_sources"]["e182"]

DEVICE = E180.DEVICE
FT = E180.FT
TASKS = list(E180.TASKS)
CLASSES = E180.CLASSES
SEED = int(E180.CFG["protocol"]["seed"])
FIT = float(E180.CFG["protocol"]["fit_fraction"])
REPS = int(CFG["head"]["repeats_per_fold"])
COMPONENTS = ("open", "trunk", "conf")
SIGNALS = ("g_head", "g_dist", "g_entropy")
ACC_GRID = [float(a) for a in CFG["threshold"]["grid"]]
PRIMARY6 = float(CFG["threshold"]["gate6_working_point"])
PRIMARY7 = float(CFG["threshold"]["gate7_working_point"])
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


def f32(array) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {"experiment": CFG["experiment"]}


def save_results(results: dict) -> None:
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2,
                                       default=float), encoding="utf-8")


def quantile(value: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(value, dtype=np.float64), q))


def fold_path(held: int) -> Path:
    return FOLD_DIR / f"head_{int(held)}.pt"


# ---------------------------------------------------------------- head model
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


def open_logits(model: nn.Module, x: torch.Tensor, batch: int = 4096) -> np.ndarray:
    model.eval()
    out = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch):
            chunk = x[start:start + batch]
            out[start:start + len(chunk)] = model.open_head(
                model.trunk(chunk)).squeeze(1).cpu().numpy()
    return out


def trunk_embedding(model: nn.Module, x: torch.Tensor, batch: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            out.append(model.trunk(x[start:start + batch]).cpu().numpy())
    return np.concatenate(out)


def identity_confidence(model: nn.Module, x: torch.Tensor, batch: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            logits = model.id_head(model.trunk(x[start:start + batch]))
            out.append(torch.softmax(logits, 1).max(1).values.cpu().numpy())
    return np.concatenate(out)


def new_model(n_cls: int, dim: int) -> nn.Module:
    return OpenSetMLP(dim, [int(v) for v in CFG["head"]["hidden"]], n_cls,
                      float(CFG["head"]["dropout"])).to(DEVICE)


def fold_transform(features: np.ndarray, tasks: np.ndarray, train_mask: np.ndarray,
                   dim: int) -> dict:
    """Per-task z-score + PCA-96 + PCA-score standardisation, all from train_mask rows."""
    n_task, dim_in = len(TASKS), features.shape[1]
    mu = np.zeros((n_task, dim_in), dtype=np.float32)
    sd = np.zeros((n_task, dim_in), dtype=np.float32)
    for index, task in enumerate(TASKS):
        ref = train_mask & (tasks == task)
        x = features[ref]
        mu[index] = x.mean(0)
        sd[index] = x.std(0) + 1e-6
    std = features.astype(np.float64)
    for index, task in enumerate(TASKS):
        ref = tasks == task
        std[ref] = (std[ref] - mu[index].astype(np.float64)) / sd[index].astype(np.float64)
    pca_mu, basis = E180.pca_basis(std[train_mask], dim)
    proj = f32((std - pca_mu.astype(np.float64)) @ basis.astype(np.float64))
    feat_mu = f32(proj[train_mask].mean(0))
    feat_sd = f32(proj[train_mask].std(0) + 1e-6)
    return {"stats_mu": mu, "stats_sd": sd, "pca_mu": pca_mu, "pca_basis": basis,
            "proj": proj, "feat_mu": feat_mu, "feat_sd": feat_sd}


def train_head(x_all: np.ndarray, y_id: np.ndarray, y_open: np.ndarray,
               val_mask: np.ndarray, n_cls: int, seed: int) -> tuple:
    cfg = CFG["head"]
    epochs = int(cfg["epochs"])
    batch = int(cfg["batch_size"])
    lam_open = float(cfg["loss_open_weight"])
    torch.manual_seed(seed)
    model = new_model(n_cls, x_all.shape[1])
    x = torch.as_tensor(x_all, dtype=FT, device=DEVICE)
    yi = torch.as_tensor(y_id, dtype=torch.long, device=DEVICE)
    yo = torch.as_tensor(y_open, dtype=FT, device=DEVICE)
    tr = torch.as_tensor(np.flatnonzero(~val_mask), dtype=torch.long, device=DEVICE)
    va = torch.as_tensor(np.flatnonzero(val_mask), dtype=torch.long, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                            weight_decay=float(cfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    best = (float("inf"), -1, None)
    curve = []
    for epoch in range(epochs):
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
            loss = loss + lam_open * torch.stack(parts).mean()
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
                          + lam_open * torch.stack(parts).mean())
        curve.append(value)
        if value < best[0]:
            best = (value, epoch, {k: v.detach().clone()
                                   for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    model.eval()
    return model, {"best_val_loss": best[0], "best_epoch": best[1], "val_curve": curve}


def head_components(model: nn.Module, x_eval: torch.Tensor, x_fit: torch.Tensor,
                    known_ids: np.ndarray, fit_ids: np.ndarray) -> tuple:
    """Three novelty read-outs of one head, z-scored with the head's own training rows."""
    out = open_logits(model, x_eval)
    out_fit = open_logits(model, x_fit)
    mu_open, sd_open = float(np.mean(out_fit)), float(np.std(out_fit)) + 1e-6
    emb = trunk_embedding(model, x_eval)
    emb_fit = trunk_embedding(model, x_fit)
    centroids = np.stack([emb_fit[fit_ids == ident].mean(0) for ident in known_ids])
    d = torch.cdist(torch.as_tensor(emb, device=DEVICE),
                    torch.as_tensor(centroids.astype(np.float32), device=DEVICE))
    nn = (-d.min(1).values).cpu().numpy()
    d_fit = torch.cdist(torch.as_tensor(emb_fit, device=DEVICE),
                        torch.as_tensor(centroids.astype(np.float32), device=DEVICE))
    nn_fit = (-d_fit.min(1).values).cpu().numpy()
    conf = identity_confidence(model, x_eval)
    conf_fit = identity_confidence(model, x_fit)
    values = {"open": (out - mu_open) / sd_open,
              "trunk": (nn - float(np.mean(nn_fit))) / (float(np.std(nn_fit)) + 1e-6),
              "conf": (conf - float(np.mean(conf_fit))) / (float(np.std(conf_fit)) + 1e-6)}
    calibration = {"mu_open": mu_open, "sd_open": sd_open,
                   "mu_trunk": float(np.mean(nn_fit)),
                   "sd_trunk": float(np.std(nn_fit)) + 1e-6,
                   "mu_conf": float(np.mean(conf_fit)),
                   "sd_conf": float(np.std(conf_fit)) + 1e-6}
    return values, calibration, centroids


# ---------------------------------------------------------------- stage head
def gate_metrics(score: np.ndarray, known: np.ndarray, fold: np.ndarray,
                 subset: np.ndarray | None = None) -> dict:
    score = np.asarray(score, dtype=np.float64)
    if subset is not None:
        score, known, fold = score[subset], known[subset], fold[subset]
    pooled = float(roc_auc_score(known.astype(int), score))
    auc_fold, u95, u95_pooled = [], [], []
    for held in CLASSES:
        sel = fold == int(held)
        if sel.sum() == 0 or len(np.unique(known[sel])) < 2:
            continue
        auc_fold.append(float(roc_auc_score(known[sel].astype(int), score[sel])))
        thr = quantile(score[sel & known], 0.05)
        u95.append(float(np.mean(score[sel & ~known] < thr)))
    pooled_thr = quantile(score[known], 0.05)
    u95_pooled.append(float(np.mean(score[~known] < pooled_thr)))
    return {"pooled_auc": pooled, "fold_auc_mean": float(np.mean(auc_fold)),
            "fold_auc_std": float(np.std(auc_fold)), "fold_auc_min": float(np.min(auc_fold)),
            "u95_mean": float(np.mean(u95)), "u95_std": float(np.std(u95)),
            "u95_min": float(np.min(u95)), "u95_pooled_threshold": float(np.mean(u95_pooled))}


def stage_head() -> dict:
    started = time.time()
    meta, frac, blocks, fit_rows = E180.load_config_meta()
    subjects = meta["subject"].to_numpy(int)
    tasks = meta["task"].to_numpy()
    window = (frac > FIT) & (frac <= 0.8)
    win_rows = np.flatnonzero(window)
    cfg = CFG["head"]
    dim = int(cfg["pca_dim"])
    k = int(E180.CFG["branch_b"]["prototype_k"])
    n_unk = int(cfg["pseudo_unknown_per_repeat"])
    features = f32(np.load(ROOT / E180.CFG["branch_b"]["train_features"]))
    if features.shape != (len(meta), int(cfg["feature_dim"])):
        raise ValueError(f"E65 feature cache shape {features.shape} unexpected")
    loso82 = np.load(E182_DIR / "cache" / "loso_v2.npz")
    signals82 = np.load(E182_DIR / "cache" / "signals_v2.npz")
    ent_mu, ent_sd = float(signals82["stats_mu"][4]), float(signals82["stats_sd"][4])
    store = E180.load_branch_b()
    oof_mu, oof_sd = float(store["oof_mu"]), float(store["oof_sd"])
    log(f"stage head: {len(meta)} Train rows, window {len(win_rows)}, PCA-{dim}, "
        f"{REPS} views x {n_unk} pseudo-unknown identities per fold")

    rows_all, fold_all, known_all, known_k_all = [], [], [], []
    comp_all = {name: [] for name in COMPONENTS}
    z_dist_all, z_ent_all = [], []
    centroids_all = np.zeros((len(CLASSES), REPS, len(CLASSES), 64), dtype=np.float32)
    k_mask = np.zeros((len(CLASSES), REPS, len(CLASSES)), dtype=bool)
    folds = []
    for held in CLASSES:
        held = int(held)
        pm = subjects != held
        enrolled = np.setdiff1d(CLASSES, [held])
        train_mask = pm & fit_rows
        tr = fold_transform(features, tasks, train_mask, dim)
        proj = tr["proj"]
        rows = win_rows
        x_eval = torch.as_tensor((proj[rows] - tr["feat_mu"]) / tr["feat_sd"],
                                 dtype=FT, device=DEVICE)
        x_fit = torch.as_tensor((proj[train_mask] - tr["feat_mu"]) / tr["feat_sd"],
                                dtype=FT, device=DEVICE)
        fit_ids = subjects[train_mask]
        protos, keys = E180.build_prototypes(proj, subjects, tasks, train_mask,
                                             enrolled, k)
        proto, filled = E180.proto_stack(protos, keys, k)
        dist = E180.branch_b_scores(proj[rows], tasks[rows], proto, filled, k)
        z_dist = (dist - oof_mu) / oof_sd
        ent_vec = np.full(len(meta), np.nan, dtype=np.float64)
        sel = loso82["fold"] == held
        ent_vec[loso82["rows"][sel]] = loso82["g_entropy"][sel]
        z_ent = (ent_vec[rows] - ent_mu) / ent_sd
        if not np.isfinite(z_ent).all():
            raise RuntimeError(f"fold {held}: entropy lookup incomplete")

        cache_fold = fold_path(held)
        if cache_fold.exists():
            art = torch.load(cache_fold, map_location="cpu", weights_only=False)
            if art.get("pca_dim") != dim or art.get("reps") != REPS:
                art = None
        else:
            art = None
        if art is None:
            states, us, ks, mu_reg, sd_reg, inner = [], [], [], [], [], []
            for rep in range(REPS):
                seed = SEED + 1000 * held + 97 * rep
                rng = np.random.default_rng(seed)
                unknown_ids = np.sort(rng.choice(enrolled, size=n_unk, replace=False))
                known_ids = np.setdiff1d(enrolled, unknown_ids)
                sel_k = np.flatnonzero(fit_rows & np.isin(subjects, known_ids))
                sel_u = np.flatnonzero(fit_rows & np.isin(subjects, unknown_ids))
                order = np.r_[sel_k, sel_u]
                x_train = (proj[order] - tr["feat_mu"]) / tr["feat_sd"]
                y_open = np.r_[np.ones(len(sel_k), np.float32),
                               np.zeros(len(sel_u), np.float32)]
                y_id = np.clip(np.searchsorted(known_ids, subjects[order]), 0,
                               len(known_ids) - 1)
                group = np.r_[subjects[sel_k], np.full(len(sel_u), -1)]
                rng_split = np.random.default_rng(seed + 7)
                val = np.zeros(len(order), dtype=bool)
                for value in np.unique(group):
                    pos = np.flatnonzero(group == value)
                    take = rng_split.permutation(pos)[:max(1, int(round(0.15 * len(pos))))]
                    val[take] = True
                model, info = train_head(x_train, y_id, y_open, val, len(known_ids), seed)
                tr_out = open_logits(model, torch.as_tensor(x_train, dtype=FT,
                                                            device=DEVICE))
                states.append({key: value.detach().cpu()
                               for key, value in model.state_dict().items()})
                us.append(unknown_ids.tolist())
                ks.append(known_ids.tolist())
                mu_reg.append(float(tr_out.mean()))
                sd_reg.append(float(tr_out.std()) + 1e-6)
                inner.append(info["best_epoch"])
                log(f"  fold {held:2d} view {rep}: |K| {len(known_ids)} "
                    f"|U| {len(unknown_ids)} best epoch {info['best_epoch']} "
                    f"val {info['best_val_loss']:.4f}")
            art = {"held": held, "pca_dim": dim, "reps": REPS, "U": us, "K": ks,
                   "state_dicts": states, "mu_reg": mu_reg, "sd_reg": sd_reg,
                   "inner_best_epoch": inner,
                   "stats_mu": tr["stats_mu"], "stats_sd": tr["stats_sd"],
                   "pca_mu": tr["pca_mu"], "pca_basis": tr["pca_basis"],
                   "feat_mu": tr["feat_mu"], "feat_sd": tr["feat_sd"]}
            torch.save(art, cache_fold)

        comp = {name: [] for name in COMPONENTS}
        cal = {name: [] for name in ("mu_open", "sd_open", "mu_trunk", "sd_trunk",
                                     "mu_conf", "sd_conf")}
        for rep in range(REPS):
            model = new_model(len(art["K"][rep]), dim)
            model.load_state_dict(art["state_dicts"][rep])
            model.eval()
            values, calibration, centroids = head_components(
                model, x_eval, x_fit, np.asarray(art["K"][rep]), fit_ids)
            for name in COMPONENTS:
                comp[name].append(values[name])
            for name in cal:
                cal[name].append(calibration[name])
            for slot, ident in enumerate(art["K"][rep]):
                centroids_all[held - 1, rep, int(ident) - 1] = centroids[slot]
                k_mask[held - 1, rep, int(ident) - 1] = True
            del model
        art.update(dict(cal))
        torch.save(art, cache_fold)
        rows_all.append(rows)
        fold_all.append(np.full(len(rows), held, dtype=np.int32))
        known_all.append(pm[rows])
        known_k_all.append(np.isin(subjects[rows], art["K"][0]))
        for name in COMPONENTS:
            comp_all[name].append(np.stack(comp[name]))
        z_dist_all.append(z_dist)
        z_ent_all.append(z_ent)
        folds.append({"held_identity": held, "n_known": int(pm[rows].sum()),
                      "n_unknown": int((~pm[rows]).sum()),
                      "inner_best_epoch": art["inner_best_epoch"]})
        log(f"  fold {held:2d}: rows {len(rows)}, known {int(pm[rows].sum())}, "
            f"unknown {int((~pm[rows]).sum())}")
        del tr, proj, x_eval, x_fit
        torch.cuda.empty_cache()

    rows = np.concatenate(rows_all)
    fold = np.concatenate(fold_all)
    known = np.concatenate(known_all)
    known_k = np.concatenate(known_k_all)
    components = {name: np.concatenate(comp_all[name], axis=1) for name in COMPONENTS}
    z_dist = np.concatenate(z_dist_all)
    z_ent = np.concatenate(z_ent_all)
    payload = {"rows": rows, "fold": fold, "known": known, "known_k": known_k,
               "z_dist": z_dist, "z_ent": z_ent,
               "centroids": centroids_all, "k_mask": k_mask}
    for name in COMPONENTS:
        payload[f"comp_{name}"] = components[name]
        payload[f"comp_{name}_max"] = components[name].max(0)
        payload[f"comp_{name}_mean"] = components[name].mean(0)
    np.savez_compressed(LOSO_CACHE, **payload)

    strict = known_k | ~known
    per_readout = {}
    for name in COMPONENTS:
        for agg, score in (("max_over_views", components[name].max(0)),
                           ("mean_over_views", components[name].mean(0))):
            per_readout[f"{name}_{agg}"] = gate_metrics(score, known, fold)
            per_readout[f"{name}_{agg}"]["strict_known"] = gate_metrics(
                score, known_k, fold, strict)
            log(f"  LOSO {name:5s} {agg:16s}: pooled AUC "
                f"{per_readout[f'{name}_{agg}']['pooled_auc']:.4f} U@95 "
                f"{per_readout[f'{name}_{agg}']['u95_mean']:.4f}")
    for name, score in (("g_dist", z_dist), ("g_entropy", z_ent)):
        per_readout[name] = gate_metrics(score, known, fold)
        per_readout[name]["strict_known"] = gate_metrics(score, known_k, fold, strict)
    summary = {"n_rows": int(len(rows)), "n_known": int(known.sum()),
               "n_unknown": int((~known).sum()), "views": REPS,
               "pseudo_unknown": n_unk, "pca_dim": dim,
               "readouts": per_readout, "folds": folds,
               "elapsed_seconds": time.time() - started}
    results = load_results()
    results["head"] = summary
    save_results(results)
    log(f"stage head done in {summary['elapsed_seconds']:.0f}s")
    return summary


# ---------------------------------------------------------------- stage fusion
def fusion_candidates() -> list:
    grid = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    cands, seen = [], set()
    for w_head in grid:
        for w_dist in grid:
            w_ent = round(1.0 - w_head - w_dist, 2)
            if w_ent < -1e-9:
                continue
            key = (round(w_head, 2), round(w_dist, 2), round(max(w_ent, 0.0), 2))
            if key in seen:
                continue
            seen.add(key)
            cands.append({"g_head": key[0], "g_dist": key[1], "g_entropy": key[2]})
    return cands


def candidate_name(weights: dict) -> str:
    parts = [f"{weights[s]:.2f}*{s}" for s in SIGNALS if weights[s] > 0]
    return " + ".join(parts) if parts else "empty"


def stage_fusion() -> dict:
    started = time.time()
    data = np.load(LOSO_CACHE)
    fold, known, known_k = data["fold"], data["known"], data["known_k"]
    strict = known_k | ~known
    z = {"g_dist": data["z_dist"].astype(np.float64),
         "g_entropy": data["z_ent"].astype(np.float64)}

    readouts = {}
    for name in COMPONENTS:
        for agg, suffix in (("max_over_views", "max"), ("mean_over_views", "mean")):
            score = data[f"comp_{name}_{suffix}"].astype(np.float64)
            readouts[f"{name}_{agg}"] = {
                "score": score, "pooled_auc": float(roc_auc_score(known.astype(int), score)),
                "u95": gate_metrics(score, known, fold)["u95_mean"]}
    best_readout = max(readouts, key=lambda key: readouts[key]["pooled_auc"])
    log(f"stage fusion: read-out selected on Train LOSO AUC: {best_readout} "
        f"({readouts[best_readout]['pooled_auc']:.4f})")
    z["g_head"] = readouts[best_readout]["score"]

    rows = []
    for weights in fusion_candidates():
        score = sum(weights[s] * z[s] for s in SIGNALS)
        metrics = gate_metrics(score, known, fold)
        rows.append({"weights": weights, "name": candidate_name(weights),
                     "n_signals": int(sum(1 for s in SIGNALS if weights[s] > 0)),
                     **metrics, "score": score,
                     "strict": gate_metrics(score, known_k, fold, strict)})
    ranked = sorted(rows, key=lambda r: (-r["pooled_auc"], -r["fold_auc_mean"],
                                         r["n_signals"]))
    best = ranked[0]
    tau = {f"{a:.2f}": quantile(best["score"][known], 1.0 - a) for a in ACC_GRID}
    per_op = {}
    for a in ACC_GRID:
        thr = tau[f"{a:.2f}"]
        reject = [float(np.mean(best["score"][(fold == int(h)) & ~known] < thr))
                  for h in CLASSES]
        per_op[f"{a:.2f}"] = {"tau": thr,
                              "unknown_rejection": float(np.mean(reject)),
                              "known_acceptance": float(np.mean(best["score"][known] >= thr))}
    summary = {
        "readout_selection": {"selected": best_readout,
                              "candidates": {key: {"pooled_auc": value["pooled_auc"],
                                                   "u95_mean": value["u95"]}
                                             for key, value in readouts.items()}},
        "selected": {"name": best["name"], "weights": best["weights"],
                     "pooled_auc": best["pooled_auc"],
                     "fold_auc_mean": best["fold_auc_mean"],
                     "fold_auc_std": best["fold_auc_std"],
                     "u95_mean": best["u95_mean"], "u95_std": best["u95_std"],
                     "u95_min": best["u95_min"],
                     "u95_pooled_threshold": best["u95_pooled_threshold"],
                     "strict_pooled_auc": best["strict"]["pooled_auc"],
                     "strict_u95_mean": best["strict"]["u95_mean"]},
        "tau_train_loso_known": tau, "operating_points": per_op,
        "leaderboard": [{key: row[key] for key in
                         ("name", "weights", "n_signals", "pooled_auc", "fold_auc_mean",
                          "fold_auc_std", "u95_mean", "u95_std", "u95_min")}
                        for row in ranked[:15]],
        "n_candidates": len(ranked), "elapsed_seconds": time.time() - started}
    results = load_results()
    results["fusion"] = summary
    save_results(results)
    for row in ranked[:6]:
        log(f"  {row['pooled_auc']:.4f}  {row['name']}  U@95 {row['u95_mean']:.4f}")
    log(f"  selected {best['name']}: LOSO AUC {best['pooled_auc']:.4f}, "
        f"U@95 {best['u95_mean']:.4f} (pooled-threshold {best['u95_pooled_threshold']:.4f})")
    return summary


# ---------------------------------------------------------------- stage capacity
def identity_distance(proj: np.ndarray, tasks: np.ndarray, proto: dict,
                      k: int) -> np.ndarray:
    out = np.full((len(proj), len(CLASSES)), np.inf, dtype=np.float32)
    for task in TASKS:
        rows = np.flatnonzero(tasks == task)
        if len(rows) == 0:
            continue
        p = torch.as_tensor(f32(proto[task]), device=DEVICE)
        for start in range(0, len(rows), 4096):
            chunk = rows[start:start + 4096]
            q = torch.as_tensor(f32(proj[chunk]), device=DEVICE)
            d = torch.cdist(q, p.reshape(-1, p.shape[-1])).view(len(chunk), len(CLASSES), k)
            out[chunk] = d.min(dim=2).values.cpu().numpy()
    return out


def stage_capacity() -> dict:
    started = time.time()
    meta, frac, blocks, fit_rows = E180.load_config_meta()
    subjects = meta["subject"].to_numpy(int)
    tasks = meta["task"].to_numpy()
    k = int(E180.CFG["branch_b"]["prototype_k"])
    features = f32(np.load(ROOT / E180.CFG["branch_b"]["train_features"]))
    store = E180.load_branch_b()
    std = features.astype(np.float64)
    for index, task in enumerate(TASKS):
        ref = tasks == task
        std[ref] = ((std[ref] - store["stats_mu"][index].astype(np.float64))
                    / store["stats_sd"][index].astype(np.float64))
    proj = f32((std - store["pca_mu"].astype(np.float64)) @ store["pca_basis"])
    stage_a = torch.load(E182_DIR / "cache" / "stage_a_v2.pt", map_location="cpu",
                         weights_only=False)
    oof_logits = stage_a["oof_logits"].numpy()
    in_logits = stage_a["in_sample_logits"].numpy()
    in_sample = float(np.mean(in_logits.argmax(1) + 1 == subjects))
    log(f"stage capacity: frozen branch A v2 in-sample argmax {in_sample:.4f}")

    d_oof = np.zeros((len(meta), len(CLASSES)), np.float32)
    for block in range(int(E180.CFG["protocol"]["blocks"])):
        rows = blocks == block
        protos, keys = E180.build_prototypes(proj, subjects, tasks,
                                             fit_rows & ~rows, CLASSES, k)
        proto, filled = E180.proto_stack(protos, keys, k)
        d_oof[rows] = identity_distance(proj[rows], tasks[rows], proto, k)
    protos, keys = E180.build_prototypes(proj, subjects, tasks, fit_rows, CLASSES, k)
    proto, filled = E180.proto_stack(protos, keys, k)
    d_in = identity_distance(proj, tasks, proto, k)

    def accuracy(logits: np.ndarray, d: np.ndarray, alpha: float) -> float:
        return float(np.mean((logits.astype(np.float32) - alpha * d).argmax(1) + 1
                             == subjects))

    alphas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
    oof_sweep = [{"alpha": alpha, "oof_accuracy": accuracy(oof_logits, d_oof, alpha),
                  "in_sample_accuracy": accuracy(in_logits, d_in, alpha)}
                 for alpha in alphas]
    best = max(oof_sweep, key=lambda r: (r["oof_accuracy"], -r["alpha"]))
    alpha = best["alpha"]
    summary = {
        "in_sample_logits_argmax": in_sample,
        "oof_accuracy_C": accuracy(oof_logits, d_oof, 0.0),
        "oof_accuracy_forward_blocks": float(np.mean(
            (oof_logits.argmax(1) + 1 == subjects)[frac > 0.6])),
        "oof_macro_f1": float(f1_score(subjects, oof_logits.argmax(1) + 1,
                                       labels=CLASSES, average="macro", zero_division=0)),
        "alpha_sweep": oof_sweep, "alpha_selected_oof": alpha,
        "in_sample_prototype_assisted": accuracy(in_logits, d_in, alpha),
        "in_sample_prototype_only": float(np.mean(d_in.argmin(1) + 1 == subjects)),
        "oof_accuracy_prototype_assisted": best["oof_accuracy"],
        "in_sample_best_technical": float(max(row["in_sample_accuracy"]
                                              for row in oof_sweep)),
        "in_sample_best_technical_alpha": float(max(oof_sweep,
                                                    key=lambda r: r["in_sample_accuracy"]
                                                    )["alpha"]),
        "caveat": ("prototype-assisted in-sample accuracy uses prototypes that contain "
                   "the evaluated rows themselves: capacity / memorisation probe only. "
                   "The OOF column is the honest control."),
        "elapsed_seconds": time.time() - started}
    results = load_results()
    results["capacity"] = summary
    save_results(results)
    log(f"  in-sample logits {summary['in_sample_logits_argmax']:.4f}; alpha {alpha} -> "
        f"in-sample {summary['in_sample_prototype_assisted']:.4f}, OOF "
        f"{best['oof_accuracy']:.4f}; prototype-only {summary['in_sample_prototype_only']:.4f}")
    return summary


# ---------------------------------------------------------------- frozen heads
def fold_inputs(features: np.ndarray, tasks: np.ndarray,
                art: dict) -> torch.Tensor:
    x = torch.as_tensor(features, dtype=FT, device=DEVICE)
    out = torch.empty((len(x), art["pca_basis"].shape[1]), dtype=FT, device=DEVICE)
    pca_mu = torch.as_tensor(art["pca_mu"], dtype=FT, device=DEVICE)
    basis = torch.as_tensor(art["pca_basis"], dtype=FT, device=DEVICE)
    for index, task in enumerate(TASKS):
        ref = tasks == task
        if not bool(ref.any()):
            continue
        mu = torch.as_tensor(art["stats_mu"][index], dtype=FT, device=DEVICE)
        sd = torch.as_tensor(art["stats_sd"][index], dtype=FT, device=DEVICE)
        out[ref] = (((x[ref] - mu) / sd) - pca_mu) @ basis
    fmu = torch.as_tensor(art["feat_mu"], dtype=FT, device=DEVICE)
    fsd = torch.as_tensor(art["feat_sd"], dtype=FT, device=DEVICE)
    return (out - fmu) / fsd


def load_fold_models(held: int) -> tuple:
    art = torch.load(fold_path(int(held)), map_location="cpu", weights_only=False)
    models = []
    for rep in range(len(art["state_dicts"])):
        model = new_model(len(art["K"][rep]), int(CFG["head"]["pca_dim"]))
        model.load_state_dict(art["state_dicts"][rep])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models.append(model)
    return art, models


def stage_test() -> dict:
    started = time.time()
    dim = int(CFG["head"]["pca_dim"])
    test_meta = pd.read_csv(ROOT / E180.CFG["data"]["test_metadata"],
                            usecols=["sample", "path", "task"])
    test_features = f32(np.load(ROOT / E180.CFG["branch_b"]["test_features"]))
    tasks = test_meta["task"].to_numpy()
    inputs = np.load(E182_DIR / "cache" / "test_inputs.npz", allow_pickle=True)
    if not np.array_equal(test_meta["sample"].astype(str).to_numpy(), inputs["sample"]):
        raise ValueError("Test sample order mismatch with the frozen E182 cache")
    logits = inputs["logits_v2"]
    store = E180.load_branch_b()
    z_dist = ((inputs["dist"].astype(np.float64) - float(store["oof_mu"]))
              / float(store["oof_sd"]))
    signals82 = np.load(E182_DIR / "cache" / "signals_v2.npz")
    z_ent = signals82["test_z"][4].astype(np.float64)

    data = np.load(LOSO_CACHE)
    centroids, k_mask = data["centroids"], data["k_mask"]
    selected = load_results()["fusion"]["readout_selection"]["selected"]
    component = selected.split("_")[0]
    per_fold = np.zeros((len(CLASSES), len(test_features)), dtype=np.float32)
    for held in CLASSES:
        art, models = load_fold_models(int(held))
        x = fold_inputs(test_features, tasks, art)
        views = []
        for rep, model in enumerate(models):
            if component == "open":
                value = open_logits(model, x)
            elif component == "trunk":
                emb = trunk_embedding(model, x)
                cent = torch.as_tensor(centroids[held - 1, rep][k_mask[held - 1, rep]],
                                       device=DEVICE)
                d = torch.cdist(torch.as_tensor(emb, device=DEVICE), cent)
                value = (-d.min(1).values).cpu().numpy()
            else:
                value = identity_confidence(model, x)
            views.append(value)
        views = np.stack(views)
        per_fold[held - 1] = views.max(0)
        del x, models
        torch.cuda.empty_cache()
    z_head = per_fold.max(0).astype(np.float64)
    variants = {"max_over_folds": z_head,
                "median_over_folds": np.median(per_fold, axis=0).astype(np.float64),
                "mean_over_folds": per_fold.mean(0).astype(np.float64)}
    # the LOSO gate uses one fold's 3 views per trial: the median over folds keeps the
    # same view count, the max/mean over 20 folds change the score scale
    z_head = variants["median_over_folds"]

    weights = load_results()["fusion"]["selected"]["weights"]
    fused = (float(weights["g_head"]) * z_head + float(weights["g_dist"]) * z_dist
             + float(weights["g_entropy"]) * z_ent)
    tau = load_results()["fusion"]["tau_train_loso_known"]
    primary = f"{PRIMARY6:.2f}"
    accept = fused >= tau[primary]
    predicted = np.where(accept, logits.argmax(1) + 1, 0)
    pd.DataFrame({"sample": test_meta["sample"].astype(str), "task": tasks,
                  "prediction": predicted, "accepted": accept.astype(int),
                  "fused_gate": fused, "g_head": z_head, "g_dist": z_dist,
                  "g_entropy": z_ent}).to_csv(TEST_PRED, index=False)
    log(f"stage test: predictions written to {TEST_PRED.name} "
        f"({int(accept.sum())}/{len(accept)} accepted at a={primary})")

    # Session is read only after every prediction is on disk
    session = pd.read_csv(ROOT / E180.CFG["data"]["test_metadata"],
                          usecols=["session"])["session"].to_numpy()
    is_s1 = session == "session1"
    is_s2 = session == "session2"
    audit = {}
    for name, sel in (("S1", is_s1), ("S2", is_s2)):
        audit[name] = {"n": int(sel.sum()),
                       "acceptance": float(np.mean(accept[sel])),
                       "rejection": float(np.mean(~accept[sel])),
                       "g_head_mean_z": float(np.mean(z_head[sel])),
                       "g_dist_mean_z": float(np.mean(z_dist[sel])),
                       "auc_s2_vs_s1_head": float(roc_auc_score(
                           is_s2.astype(int), z_head)),
                       "auc_s2_vs_s1_fused": float(roc_auc_score(
                           is_s2.astype(int), fused))}
    fused_mean_folds = None
    alt = {}
    for name, value in variants.items():
        fused_v = (float(weights["g_head"]) * value + float(weights["g_dist"]) * z_dist
                   + float(weights["g_entropy"]) * z_ent)
        alt[name] = {"S1_rejection": float(np.mean(~(fused_v[is_s1] >= tau[primary]))),
                     "S2_acceptance": float(np.mean(fused_v[is_s2] >= tau[primary])),
                     "auc_s2_vs_s1": float(roc_auc_score(is_s2.astype(int), fused_v))}
    grid = {f"{a:.2f}": {
        "S1_rejection": float(np.mean(~(fused[is_s1] >= tau[f"{a:.2f}"]))),
        "S2_acceptance": float(np.mean(fused[is_s2] >= tau[f"{a:.2f}"]))}
        for a in ACC_GRID}
    summary = {"selected_weights": weights, "readout": selected,
               "primary_acceptance": PRIMARY6, "tau": tau,
               "audit": audit, "audit_mean_over_folds": alt, "grid": grid,
               "session_used_for_inference": False,
               "official_metrics": "N/A (Test has no identity labels)",
               "elapsed_seconds": time.time() - started}
    results = load_results()
    results["test"] = summary
    save_results(results)
    log(f"  audit a={primary}: S1 rejection {audit['S1']['rejection']:.4f} "
        f"S2 acceptance {audit['S2']['acceptance']:.4f}")
    return summary


# ---------------------------------------------------------------- stage attack
class FusedHeadScorer:
    """Frozen branch A v2 + branch B gate + differentiable selected open-set read-out."""

    def __init__(self, model_a: nn.Module, store: dict, folds: list, weights: dict,
                 ent_stats: tuple, component: str) -> None:
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

    def __call__(self, x: torch.Tensor, task_idx: torch.Tensor, groups: list) -> tuple:
        logits = self.model_a(x)
        prob = torch.softmax(logits, dim=1)
        ent = (prob * prob.clamp_min(1e-12).log()).sum(1)
        g_ent = (-ent - self.ent_mu) / self.ent_sd
        features = E180.differentiable_logspd(x)
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
            art = self.folds[fold_index]
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
                    value = ((-d.min(1).values) - art["mu_trunk"][rep]) \
                        / art["sd_trunk"][rep]
                views.append(value)
            head = head.index_copy(0, index, torch.stack(views).max(0).values)
        g = (self.weights["g_head"] * head + self.weights["g_dist"] * g_dist
             + self.weights["g_entropy"] * g_ent)
        return logits, g


def pgd_attack(scorer, x: torch.Tensor, task_idx: torch.Tensor, groups: list,
               true_idx: torch.Tensor, target_idx: torch.Tensor, goal: str,
               eps: float, steps: int, seed: int) -> dict:
    cfg = CFG["attack"]
    temperature = float(cfg["temperature"])
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    xc = x - x.mean(dim=-1, keepdim=True)
    rms = xc.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-6)
    bound = (eps * rms).view(-1, 1, 1)
    with torch.no_grad():
        adv = x + (torch.rand(x.shape, generator=gen, device=DEVICE) * 2 - 1) * bound
        adv = torch.clamp(adv, x - bound, x + bound)
    alpha = 2.0 * eps / steps
    nan_count = 0
    for _ in range(steps):
        adv.requires_grad_(True)
        logits, gate = scorer(adv, task_idx, groups)
        if goal == "D":
            objective = gate
        else:
            true_logit = logits.gather(1, true_idx[:, None]).squeeze(1)
            wrong = logits.masked_fill(F.one_hot(true_idx, len(CLASSES)).bool(), -1e9)
            if goal == "A":
                margin = wrong.max(dim=1).values
            elif goal == "B":
                margin = torch.logsumexp(wrong, dim=1)
            else:
                margin = logits.gather(1, target_idx[:, None]).squeeze(1)
            objective = (margin - true_logit) / temperature
        grad = torch.autograd.grad(-objective.sum(), adv)[0]
        nan_count += int((~torch.isfinite(grad)).sum().item())
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        adv = torch.clamp(adv.detach() - (alpha * rms).view(-1, 1, 1) * grad.sign(),
                          x - bound, x + bound)
    with torch.no_grad():
        logits, gate = scorer(adv, task_idx, groups)
    realised = ((adv - x).abs().amax(dim=(1, 2)) / rms)
    return {"logits": logits.detach().cpu().numpy(), "gate": gate.detach().cpu().numpy(),
            "max_rel_linf": float(realised.max()), "nan_gradients": nan_count}


def attack_metrics(logits: np.ndarray, gate: np.ndarray, truth: np.ndarray,
                   tau: float) -> dict:
    accept = gate >= tau
    predicted = np.where(accept, logits.argmax(1) + 1, 0)
    ra = float(np.mean(predicted == truth))
    asafe = float(np.mean((predicted == truth) | (predicted == 0)))
    return {"n": len(predicted), "RA": ra, "ASafeR": asafe,
            "closed_set_accuracy": float(np.mean(logits.argmax(1) + 1 == truth)),
            "accept_rate": float(np.mean(accept)),
            "wrong_identity_rate": float(np.mean((predicted != 0) & (predicted != truth)))}


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

    data = np.load(LOSO_CACHE)
    known, k_mask = data["known"], data["k_mask"]
    weights = load_results()["fusion"]["selected"]["weights"]
    readout = load_results()["fusion"]["readout_selection"]["selected"]
    component = readout.split("_")[0]
    selected_name = load_results()["fusion"]["selected"]["name"]
    fused_known = sum(float(weights[s]) * data[key].astype(np.float64) for s, key in
                      (("g_head", f"comp_{component}_max"), ("g_dist", "z_dist"),
                       ("g_entropy", "z_ent")))[known]
    tau = {f"{a:.2f}": quantile(fused_known, 1.0 - a) for a in ACC_GRID}

    head_choice = {}
    for value in CLASSES:
        for held in CLASSES:
            if int(value) != int(held) and k_mask[int(held) - 1, :, int(value) - 1].all():
                head_choice[int(value)] = int(held)
                break
        if int(value) not in head_choice:
            raise RuntimeError(f"no fold treats identity {value} as registered")

    branch_a = torch.load(E182_DIR / "model_branch_a_v2.pt", map_location="cpu",
                          weights_only=False)
    expected = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))[
        "frozen_hashes"]["branch_a_v2"]
    if branch_a["spec_sha256"] != expected:
        raise ValueError("branch A v2 artifact does not match the frozen E182 spec")
    model_a = E180.Net().to(DEVICE)
    model_a.load_state_dict(branch_a["state_dict"])
    model_a.eval()
    store = E180.load_branch_b()
    signals82 = np.load(E182_DIR / "cache" / "signals_v2.npz")
    ent_stats = (float(signals82["stats_mu"][4]), float(signals82["stats_sd"][4]))

    centroids = data["centroids"]
    folds = []
    for held in CLASSES:
        art, models = load_fold_models(int(held))
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
        entry["centroids"] = [torch.as_tensor(centroids[int(held) - 1, rep][
            k_mask[int(held) - 1, rep]], device=DEVICE) for rep in range(REPS)]
        entry["k_mask"] = k_mask[int(held) - 1]
        folds.append(entry)
    scorer = FusedHeadScorer(model_a, store, folds, weights, ent_stats, component)

    raw = E180.load_raw([meta["path"].iloc[i] for i in subset])
    truth = subjects[subset]
    task_idx = torch.as_tensor([TASKS.index(t) for t in tasks[subset]],
                               dtype=torch.long, device=DEVICE)
    true_idx = torch.as_tensor(truth - 1, dtype=torch.long, device=DEVICE)
    target = np.asarray([CLASSES[(CLASS_OF[int(v)] + 1) % len(CLASSES)] for v in truth])
    target_idx = torch.as_tensor([CLASS_OF[int(v)] for v in target],
                                 dtype=torch.long, device=DEVICE)
    assign = np.asarray([head_choice[int(v)] for v in truth])
    groups = []
    for held in CLASSES:
        index = np.flatnonzero(assign == int(held))
        if len(index):
            groups.append((int(held), torch.as_tensor(index, dtype=torch.long,
                                                      device=DEVICE)))

    def sweep(logits: np.ndarray, gate: np.ndarray) -> dict:
        out = {}
        for a in ACC_GRID:
            row = attack_metrics(logits, gate, truth, tau[f"{a:.2f}"])
            out[f"{a:.2f}"] = {"RA": row["RA"], "ASafeR": row["ASafeR"],
                               "accept_rate": row["accept_rate"]}
        return out

    with torch.no_grad():
        logits, gate = scorer(raw, task_idx, groups)
    clean_logits, clean_gate = logits.detach().cpu().numpy(), gate.detach().cpu().numpy()
    clean = attack_metrics(clean_logits, clean_gate, truth, tau[f"{PRIMARY7:.2f}"])
    clean["at_acceptance"] = sweep(clean_logits, clean_gate)
    log(f"  clean: closed-set {clean['closed_set_accuracy']:.4f}, accept(a="
        f"{PRIMARY7:.2f}) {clean['accept_rate']:.4f}, RA {clean['RA']:.4f}")

    conditions = []
    for eps in cfg["eps_normalised"]:
        for goal in cfg["goals"]:
            out = pgd_attack(scorer, raw, task_idx, groups, true_idx, target_idx, goal,
                             float(eps), int(cfg["steps"]),
                             int(cfg["seed"]) + int(float(eps) * 1e5))
            metrics = attack_metrics(out["logits"], out["gate"], truth, tau[f"{PRIMARY7:.2f}"])
            metrics.update({"eps": float(eps), "goal": goal,
                            "max_rel_linf": out["max_rel_linf"],
                            "nan_gradients": out["nan_gradients"],
                            "at_acceptance": sweep(out["logits"], out["gate"])})
            conditions.append(metrics)
            log(f"  eps={metrics['eps']:.3f} goal {goal}: RA {metrics['RA']:.4f} "
                f"ASafeR {metrics['ASafeR']:.4f} closed "
                f"{metrics['closed_set_accuracy']:.4f} accept {metrics['accept_rate']:.4f}")

    worst = {}
    for a in ACC_GRID:
        key = f"{a:.2f}"
        ra_row = min(conditions, key=lambda r: r["at_acceptance"][key]["RA"])
        safe_row = min(conditions, key=lambda r: r["at_acceptance"][key]["ASafeR"])
        worst[key] = {"RA": ra_row["at_acceptance"][key]["RA"],
                      "RA_eps": ra_row["eps"], "RA_goal": ra_row["goal"],
                      "ASafeR": safe_row["at_acceptance"][key]["ASafeR"],
                      "ASafeR_eps": safe_row["eps"], "ASafeR_goal": safe_row["goal"],
                      "clean_RA": clean["at_acceptance"][key]["RA"],
                      "clean_accept": clean["at_acceptance"][key]["accept_rate"]}
    summary = {"selected_weights": weights, "readout": readout,
               "selected_name": selected_name, "n_trials": int(len(subset)),
               "per_task": {task: int((tasks[subset] == task).sum()) for task in TASKS},
               "tau_loso": tau, "sweep_acceptance": ACC_GRID,
               "eps_normalised": list(cfg["eps_normalised"]),
               "steps": int(cfg["steps"]), "clean": clean, "conditions": conditions,
               "worst": worst,
               "head_assignment": {str(k): int(v) for k, v in head_choice.items()},
               "attack_status": "self-built white-box PGD surrogate, not the official suite",
               "elapsed_seconds": time.time() - started}
    results = load_results()
    results["attack"] = summary
    save_results(results)
    log(f"attack done in {summary['elapsed_seconds']:.0f}s: worst RA "
        f"{worst[f'{PRIMARY7:.2f}']['RA']:.4f} at a={PRIMARY7:.2f}, worst ASafeR "
        f"{worst[f'{PRIMARY7:.2f}']['ASafeR']:.4f}")
    del raw, model_a
    torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------- main
STAGES = {"head": stage_head, "fusion": stage_fusion, "capacity": stage_capacity,
          "test": stage_test, "attack": stage_attack}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    args = parser.parse_args()
    sys.stdout = Tee(LOG_PATH)
    t0 = time.time()
    order = ["head", "fusion", "capacity", "test", "attack"] \
        if args.stage == "all" else [args.stage]
    results = load_results()
    results["experiment"] = CFG["experiment"]
    results["frozen_hashes"] = {
        "branch_a_v2": json.loads((E182_DIR / "results.json").read_text(
            encoding="utf-8"))["branch_a_v2_spec_sha256"]}
    results["config"] = CFG
    save_results(results)
    for name in order:
        log(f"=== stage {name} ===")
        STAGES[name]()
    log(f"total {time.time() - t0:.0f}s")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
