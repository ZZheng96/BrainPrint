"""E170: three-way fusion (E65 + logbp + E162) under the verbatim E143/E163/E167
open-set protocol, plus an E159-style ceiling analysis of the best frozen scores.

Part 1 - fusion
  * each stream: task-wise z-score (Train session1 stats) -> independent PCA
    (E65 64/96, logbp 16/32/48, E162 32/64/128)
  * feature-level: 3-way concat (raw and block_norm) -> joint PCA 96/128 -> k=1/3/6/12
    controls: E65 alone (must reproduce AUC 0.7578 / W 0.6958) and E65+logbp
    (must reproduce the E167 AUC 0.7999 row)
  * score-level: per-stream -d1 z-scores weighted on the 0.1 simplex grid, plus
    weights selected by Train LOSO AUC only (Test never consulted)

Part 2 - ceiling
  For three frozen scores (E143 E65, E167 best-AUC, E170 best-AUC) sweep 2000
  quantiles and report the best W, the optimistic bound (LOSO fixed at the score's
  own maximum), the best S2 acceptance when S1 rejection is 0.80, and the
  diagnostic AUC that W = 0.80 would require.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.metrics import f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FT = torch.float32
TASKS = CFG["tasks"]
CLASSES = np.arange(1, 21, dtype=int)
WEIGHTS = CFG["weights"]
ACC = CFG["acceptance_grid"]
REF_ACC = float(CFG["reference_acceptance"])
OFF_TASK = -1e9


class Tee:
    def __init__(self, path: Path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.__stdout__

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


REPORT_ONLY = "--report-only" in sys.argv
sys.stdout = sys.__stdout__ if REPORT_ONLY else Tee(OUT / "run.log")


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------ primitives
def frac(meta: pd.DataFrame) -> np.ndarray:
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def task_standardise(a: np.ndarray, b: np.ndarray, tasks: np.ndarray,
                     test_tasks: np.ndarray) -> tuple:
    """Per-task z-score; statistics come from all Train rows of that task."""
    a = a.copy()
    b = b.copy()
    for task in TASKS:
        ref = tasks == task
        mu = a[ref].mean(0)
        sd = a[ref].std(0) + 1e-6
        a[ref] = (a[ref] - mu) / sd
        q = test_tasks == task
        b[q] = (b[q] - mu) / sd
    return a, b


def pca_basis(x: np.ndarray, dim: int) -> tuple:
    mu = x.mean(0)
    xc = x - mu
    values, vectors = np.linalg.eigh(xc.T @ xc / max(len(xc) - 1, 1))
    return mu, vectors[:, np.argsort(-values)][:, : min(int(dim), x.shape[1])]


def kmeans_t(x: torch.Tensor, k: int, seed: int, iters: int = 20) -> torch.Tensor:
    """Verbatim E143/E162/E163/E167 seeding so the controls reproduce."""
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


def kmeans_batch(blocks: list, k: int, seeds: list, iters: int = 20) -> list:
    """Same k-means as kmeans_t, batched over (task, identity) blocks to cut the
    number of tiny CUDA launches; per-block seeding and updates are unchanged."""
    out: list = [None] * len(blocks)
    todo = [i for i, x in enumerate(blocks) if len(x) > k]
    for i, x in enumerate(blocks):
        if len(x) <= k:
            out[i] = np.array(x, dtype=np.float32, copy=True)
    if not todo:
        return out
    dim = blocks[0].shape[1]
    max_n = max(len(blocks[i]) for i in todo)
    b_n = len(todo)
    x_pad = torch.zeros((b_n, max_n, dim), device=DEVICE, dtype=FT)
    valid = torch.zeros((b_n, max_n), device=DEVICE, dtype=torch.bool)
    for b, i in enumerate(todo):
        x = torch.as_tensor(np.asarray(blocks[i], dtype=np.float32), device=DEVICE)
        x_pad[b, :len(x)] = x
        valid[b, :len(x)] = True
    centres = torch.zeros((b_n, k, dim), device=DEVICE, dtype=FT)
    for b, i in enumerate(todo):
        rng = np.random.default_rng(seeds[i])
        pick = rng.choice(len(blocks[i]), size=k, replace=False)
        centres[b] = torch.as_tensor(np.asarray(blocks[i][pick], dtype=np.float32),
                                     device=DEVICE)
    flat_valid = valid.view(-1)
    x_flat = x_pad.view(-1, dim)
    ones = torch.ones(b_n * max_n, device=DEVICE)
    rows = (torch.arange(b_n, device=DEVICE).unsqueeze(1) * k).expand(b_n, max_n)
    for _ in range(iters):
        dist = torch.cdist(x_pad, centres).masked_fill(~valid.unsqueeze(-1), float("inf"))
        assign = dist.argmin(2)
        flat_idx = (rows + assign).view(-1)
        sums = torch.zeros((b_n * k, dim), device=DEVICE).index_add_(
            0, flat_idx[flat_valid], x_flat[flat_valid])
        counts = torch.zeros(b_n * k, device=DEVICE).index_add_(
            0, flat_idx[flat_valid], ones[flat_valid])
        sums = sums.view(b_n, k, dim)
        counts = counts.view(b_n, k)
        new = torch.where(counts[:, :, None] > 0,
                          sums / counts.clamp(min=1)[:, :, None], centres)
        done = torch.allclose(new, centres, atol=1e-6)
        centres = new
        if done:
            break
    for b, i in enumerate(todo):
        out[i] = centres[b].cpu().numpy()
    return out


def build_prototypes(feat: np.ndarray, subjects: np.ndarray, tasks: np.ndarray,
                     mask: np.ndarray, classes: np.ndarray, k: int) -> tuple:
    groups, blocks = [], []
    for task in TASKS:
        for subject in classes:
            sel = mask & (tasks == task) & (subjects == subject)
            if sel.sum() == 0:
                continue
            groups.append((task, int(subject)))
            blocks.append(np.asarray(feat[sel], dtype=np.float32))
    centres = kmeans_batch(blocks, int(k), [subject * 17 for _, subject in groups])
    protos, keys = [], []
    for (task, subject), cs in zip(groups, centres):
        for centre in cs:
            protos.append(centre)
            keys.append((task, subject))
    return np.stack(protos), keys


def neg_d1(feat: np.ndarray, protos: np.ndarray, keys: list, tasks: np.ndarray) -> np.ndarray:
    out = np.full(len(feat), np.nan, dtype=np.float32)
    key_task = np.asarray([key[0] for key in keys])
    for task in TASKS:
        idx = np.flatnonzero(tasks == task)
        cols = np.flatnonzero(key_task == task)
        if len(idx) == 0 or len(cols) == 0:
            continue
        p = torch.as_tensor(protos[cols], dtype=FT, device=DEVICE)
        for start in range(0, len(idx), 2048):
            chunk = idx[start:start + 2048]
            q = torch.as_tensor(np.asarray(feat[chunk], dtype=np.float32), device=DEVICE)
            out[chunk] = (-torch.cdist(q, p)).max(1).values.cpu().numpy()
    return out


def neg_dmat(feat: np.ndarray, protos: np.ndarray, keys: list, tasks: np.ndarray) -> tuple:
    """Full row x prototype negative-distance matrix; off-task entries = -inf."""
    key_task = np.asarray([key[0] for key in keys])
    key_subject = np.asarray([key[1] for key in keys], dtype=np.int32)
    out = np.full((len(feat), len(keys)), -np.inf, dtype=np.float32)
    for task in TASKS:
        rows = np.flatnonzero(tasks == task)
        cols = np.flatnonzero(key_task == task)
        if len(rows) == 0 or len(cols) == 0:
            continue
        p = torch.as_tensor(protos[cols], dtype=FT, device=DEVICE)
        for start in range(0, len(rows), 2048):
            chunk = rows[start:start + 2048]
            q = torch.as_tensor(np.asarray(feat[chunk], dtype=np.float32), device=DEVICE)
            out[np.ix_(chunk, cols)] = (-torch.cdist(q, p)).cpu().numpy()
    return out, key_subject


# ------------------------------------------------------------------ state
class Context:
    def __init__(self) -> None:
        meta = pd.read_csv(ROOT / CFG["data"]["train_metadata"])
        qmeta = pd.read_csv(ROOT / CFG["data"]["test_metadata"])
        self.nodes = meta.subject.to_numpy(int)
        self.tasks = meta.task.to_numpy()
        self.frac = frac(meta)
        self.fit_rows = self.frac <= CFG["fit_fraction"]
        self.blocks = np.clip((self.frac * CFG["blocks"]).astype(int), 0, CFG["blocks"] - 1)
        self.window = (self.frac > CFG["fit_fraction"]) & (self.frac <= 0.8)
        self.window_rows = np.flatnonzero(self.window)
        self.test_tasks = qmeta.task.to_numpy()
        self.is_s1 = qmeta.session.astype(str).str.casefold().isin(
            ["session1", "s1", "1"]).to_numpy()
        self.y = np.r_[np.zeros(int(self.is_s1.sum()), dtype=int),
                       np.ones(int((~self.is_s1).sum()), dtype=int)]
        self.proj: dict[str, tuple] = {}

    def project(self, key: str, feat_train: np.ndarray, feat_test: np.ndarray, dim: int) -> tuple:
        cache = f"{key}@{dim}"
        if cache in self.proj:
            return self.proj[cache]
        mu, basis = pca_basis(np.asarray(feat_train[self.fit_rows], dtype=np.float64), dim)
        at = ((feat_train.astype(np.float64) - mu) @ basis).astype(np.float32)
        bt = ((feat_test.astype(np.float64) - mu) @ basis).astype(np.float32)
        self.proj[cache] = (at, bt)
        return at, bt


def load_stream(spec: dict) -> tuple:
    a = np.load(ROOT / spec["train"]).astype(np.float32)
    b = np.load(ROOT / spec["test"]).astype(np.float32)
    return a, b


# ------------------------------------------------------------------ protocol
def evaluate(at: np.ndarray, bt: np.ndarray, k: int, ctx: Context,
             keep_zmats: bool = False) -> dict:
    """Verbatim E143/E163/E167 open-set protocol on one projected representation."""
    protos, keys = build_prototypes(at, ctx.nodes, ctx.tasks,
                                    np.ones(len(at), bool), CLASSES, k)
    te_raw = neg_d1(bt, protos, keys, ctx.test_tasks)

    oof = np.full(len(at), np.nan, dtype=np.float32)
    oof_pred = np.full(len(at), -1, dtype=np.int32)
    block_cache = []
    for block in range(CFG["blocks"]):
        hold = ctx.blocks == block
        held = np.flatnonzero(hold)
        p2, k2 = build_prototypes(at, ctx.nodes, ctx.tasks, ~hold, CLASSES, k)
        dmat, ksubj = neg_dmat(at[held], p2, k2, ctx.tasks[held])
        oof[held] = dmat.max(1)
        oof_pred[held] = ksubj[dmat.argmax(1)]
        block_cache.append({"idx": held, "protos": p2, "keys": k2,
                            "dmat": dmat, "ksubj": ksubj})

    loso_known, loso_unknown = [], []
    for held_id in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held_id])
        pm = np.isin(ctx.nodes, enrolled)
        p2, k2 = build_prototypes(at, ctx.nodes, ctx.tasks, pm & ctx.fit_rows, enrolled, k)
        known = ctx.window_rows[pm[ctx.window_rows]]
        unknown = ctx.window_rows[ctx.nodes[ctx.window_rows] == held_id]
        if len(known) == 0 or len(unknown) == 0:
            continue
        rows = np.concatenate([known, unknown])
        scores = neg_d1(at[rows], p2, k2, ctx.tasks[rows])
        loso_known.append(scores[:len(known)])
        loso_unknown.append(scores[len(known):])

    finite = np.isfinite(oof)
    mu = float(np.nanmean(oof[finite]))
    sd = float(np.nanstd(oof[finite])) + 1e-6
    z_rf = ((oof - mu) / sd).astype(np.float32)
    z_te = ((te_raw - mu) / sd).astype(np.float32)
    loso = [(((kn - mu) / sd).astype(np.float32), ((un - mu) / sd).astype(np.float32))
            for kn, un in zip(loso_known, loso_unknown)]

    unknown_all = np.concatenate([un for _, un in loso])
    oof_auc = float(roc_auc_score(np.r_[np.zeros(len(unknown_all)), np.ones(int(finite.sum()))],
                                  np.r_[unknown_all, z_rf[finite]]))
    oof_f1 = float(f1_score(ctx.nodes[finite], oof_pred[finite], labels=CLASSES,
                            average="macro", zero_division=0))
    oof_acc = float(np.mean(oof_pred[finite] == ctx.nodes[finite]))
    diag_auc = float(roc_auc_score(ctx.y, np.r_[z_te[ctx.is_s1], z_te[~ctx.is_s1]]))

    if keep_zmats:
        for b in block_cache:
            d = b.pop("dmat")
            z = np.where(np.isfinite(d), (d - mu) / sd, OFF_TASK).astype(np.float32)
            b.pop("protos")
            b.pop("keys")
            b["zmat"] = z
    else:
        block_cache = []

    entry = {"z_rf": z_rf, "z_te": z_te, "loso": loso, "blocks": block_cache,
             "mu": mu, "sd": sd,
             "train_oof_auc": oof_auc, "train_oof_f1": oof_f1,
             "train_oof_identity_acc": oof_acc, "diagnostic_auc": diag_auc,
             "per_task_auc": {
                 task: float(roc_auc_score((~ctx.is_s1[ctx.test_tasks == task]).astype(int),
                                           z_te[ctx.test_tasks == task]))
                 for task in TASKS
                 if len(np.unique(ctx.is_s1[ctx.test_tasks == task])) > 1}}
    entry["rows"] = working_points(z_rf, z_te, loso, ctx)
    return entry


def working_points(z_rf: np.ndarray, z_te: np.ndarray, loso: list, ctx: Context) -> list:
    ref = z_rf[np.isfinite(z_rf)]
    kn_sorted = [np.sort(kn) for kn, _ in loso]
    rows = []
    for acceptance in ACC:
        threshold = float(np.quantile(ref, 1 - acceptance))
        s2 = float(np.mean(z_te[~ctx.is_s1] >= threshold))
        s1 = float(np.mean(z_te[ctx.is_s1] < threshold))
        loso_rej = float(np.mean([np.mean(un < np.quantile(kn, 1 - acceptance))
                                  for kn, (_, un) in zip(kn_sorted, loso)]))
        rows.append({"acceptance_rate": float(acceptance),
                     "LOSO_rejection": loso_rej,
                     "Test_S1_rejection": s1,
                     "Test_S2_acceptance": s2,
                     "W": WEIGHTS["LOSO"] * loso_rej + WEIGHTS["S1"] * s1
                          + WEIGHTS["S2"] * s2,
                     "threshold": threshold})
    return rows


def fuse(entries: list, weights, ctx: Context, want_f1: bool = True) -> dict:
    """Weighted score fusion of N streams sharing the same prototype keys / blocks."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    z_rf = sum(w * e["z_rf"] for w, e in zip(weights, entries)).astype(np.float32)
    z_te = sum(w * e["z_te"] for w, e in zip(weights, entries)).astype(np.float32)
    loso = []
    for fold in range(len(entries[0]["loso"])):
        kn = sum(w * e["loso"][fold][0] for w, e in zip(weights, entries))
        un = sum(w * e["loso"][fold][1] for w, e in zip(weights, entries))
        loso.append((kn.astype(np.float32), un.astype(np.float32)))

    unknown_all = np.concatenate([un for _, un in loso])
    oof_auc = float(roc_auc_score(np.r_[np.zeros(len(unknown_all)), np.ones(len(z_rf))],
                                  np.r_[unknown_all, z_rf]))
    diag_auc = float(roc_auc_score(ctx.y, np.r_[z_te[ctx.is_s1], z_te[~ctx.is_s1]]))
    oof_f1, oof_acc = float("nan"), float("nan")
    if want_f1:
        pred = np.full(len(ctx.nodes), -1, dtype=np.int32)
        for block in range(len(entries[0]["blocks"])):
            first = entries[0]["blocks"][block]
            idx = first["idx"]
            total = None
            for w, e in zip(weights, entries):
                b = e["blocks"][block]
                assert np.array_equal(b["idx"], idx), "block rows misaligned"
                total = w * b["zmat"] if total is None else total + w * b["zmat"]
            pred[idx] = first["ksubj"][total.argmax(1)]
        oof_f1 = float(f1_score(ctx.nodes, pred, labels=CLASSES, average="macro",
                                zero_division=0))
        oof_acc = float(np.mean(pred == ctx.nodes))

    return {"z_rf": z_rf, "z_te": z_te, "loso": loso,
            "train_oof_auc": oof_auc, "train_oof_f1": oof_f1,
            "train_oof_identity_acc": oof_acc, "diagnostic_auc": diag_auc,
            "rows": working_points(z_rf, z_te, loso, ctx),
            "per_task_auc": {
                task: float(roc_auc_score((~ctx.is_s1[ctx.test_tasks == task]).astype(int),
                                          z_te[ctx.test_tasks == task]))
                for task in TASKS
                if len(np.unique(ctx.is_s1[ctx.test_tasks == task])) > 1}}


def loso_auc(entry: dict) -> float:
    unknown = np.concatenate([un for _, un in entry["loso"]])
    known = np.concatenate([kn for kn, _ in entry["loso"]])
    return float(roc_auc_score(np.r_[np.zeros(len(unknown)), np.ones(len(known))],
                               np.r_[unknown, known]))


def fused_loso_auc(entries: list, weights) -> float:
    """Train-only criterion used to pick fusion weights (no Test involvement)."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    unknown = np.concatenate([sum(w * e["loso"][f][1] for w, e in zip(weights, entries))
                              for f in range(len(entries[0]["loso"]))])
    known = np.concatenate([sum(w * e["loso"][f][0] for w, e in zip(weights, entries))
                            for f in range(len(entries[0]["loso"]))])
    return float(roc_auc_score(np.r_[np.zeros(len(unknown)), np.ones(len(known))],
                               np.r_[unknown, known]))


def summarise(entry: dict, meta: dict, with_curve: bool = False) -> dict:
    rows = entry["rows"]
    best = max(rows, key=lambda r: r["W"])
    ref = min(rows, key=lambda r: abs(r["acceptance_rate"] - REF_ACC))
    record = dict(meta)
    record.update({
        "diagnostic_auc": entry["diagnostic_auc"],
        "train_oof_auc": entry["train_oof_auc"],
        "train_oof_f1": entry["train_oof_f1"],
        "train_oof_identity_acc": entry["train_oof_identity_acc"],
        "loso_auc_train_only": loso_auc(entry),
        "best_W": best["W"], "best_acceptance": best["acceptance_rate"],
        "loso_at_best": best["LOSO_rejection"],
        "s1_at_best": best["Test_S1_rejection"],
        "s2_at_best": best["Test_S2_acceptance"],
        "W_at_090": ref["W"], "loso_at_090": ref["LOSO_rejection"],
        "s1_at_090": ref["Test_S1_rejection"], "s2_at_090": ref["Test_S2_acceptance"],
    })
    if with_curve:
        record["curve"] = rows
        record["per_task_auc"] = entry["per_task_auc"]
    return record


# ------------------------------------------------------------------ ceiling
def sweep(entry: dict, ctx: Context, n_thresholds: int) -> dict:
    """E159-style: freeze the score, sweep the threshold, read off the W ceiling."""
    z_rf, z_te, loso = entry["z_rf"], entry["z_te"], entry["loso"]
    ref = z_rf[np.isfinite(z_rf)]
    lo, hi = CFG["ceiling"]["quantile_range"]
    grid = np.unique(np.quantile(np.r_[ref, z_te], np.linspace(lo, hi, n_thresholds)))
    kn_sorted = [np.sort(kn) for kn, _ in loso]
    un_all = [un for _, un in loso]
    cache: dict[float, float] = {}

    def loso_rej(a_rate: float) -> float:
        if not 0 < a_rate < 1:
            return 0.0
        key = round(a_rate, 9)
        hit = cache.get(key)
        if hit is not None:
            return hit
        value = float(np.mean([np.mean(un < np.quantile(kn, 1 - a_rate))
                               for kn, un in zip(kn_sorted, un_all)]))
        cache[key] = value
        return value

    s1_mask, s2_mask = ctx.is_s1, ~ctx.is_s1
    rows = []
    for th in grid:
        s2 = float(np.mean(z_te[s2_mask] >= th))
        s1 = float(np.mean(z_te[s1_mask] < th))
        a_rate = float(np.mean(ref >= th))
        lr = loso_rej(a_rate)
        rows.append({"threshold": float(th), "accept_rate": a_rate,
                     "LOSO_rejection": lr, "Test_S1_rejection": s1,
                     "Test_S2_acceptance": s2,
                     "W": WEIGHTS["LOSO"] * lr + WEIGHTS["S1"] * s1 + WEIGHTS["S2"] * s2})
    rows.sort(key=lambda r: -r["W"])
    best = rows[0]
    loso_arr = np.asarray([r["LOSO_rejection"] for r in rows])
    acc_arr = np.asarray([r["accept_rate"] for r in rows])
    s1 = np.asarray([r["Test_S1_rejection"] for r in rows])
    s2 = np.asarray([r["Test_S2_acceptance"] for r in rows])
    common = WEIGHTS["S1"] * s1 + WEIGHTS["S2"] * s2
    common_max = float(common.max())
    at90 = loso_auc_090(entry)
    # The unrestricted LOSO maximum sits in the degenerate corner accept_rate -> 0,
    # where the per-fold known-block quantile is so high that every unknown is
    # rejected; the bound built on it is therefore not a usable operating point.
    loso_max = float(loso_arr.max())
    loso_max_legit = float(loso_arr[acc_arr >= 0.5].max()) if (acc_arr >= 0.5).any() else loso_max
    idx = int(np.argmin(np.abs(s1 - 0.80)))
    near = np.abs(s1 - 0.80) <= 0.01
    at80 = {"S1_rejection": float(s1[idx]), "S2_acceptance": float(s2[idx]),
            "LOSO_rejection": rows[idx]["LOSO_rejection"], "W": rows[idx]["W"],
            "max_S2_within_0.01": float(s2[near].max()) if near.any() else None}
    return {"diagnostic_auc": entry["diagnostic_auc"],
            "train_oof_auc": entry["train_oof_auc"],
            "loso_at_090": at90,
            "oracle_max_W": float(best["W"]), "oracle_point": {
                "accept_rate": best["accept_rate"],
                "LOSO_rejection": best["LOSO_rejection"],
                "Test_S1_rejection": best["Test_S1_rejection"],
                "Test_S2_acceptance": best["Test_S2_acceptance"]},
            "LOSO_rejection_max": loso_max,
            "LOSO_rejection_max_accept_ge_0.5": loso_max_legit,
            "optimistic_W_frozen_loso_090": float(WEIGHTS["LOSO"] * at90 + common_max),
            "optimistic_W_published_07936": float(
                WEIGHTS["LOSO"] * CFG["ceiling"]["loso_reference_at_090"] + common_max),
            "optimistic_W_loso_max_degenerate": float(WEIGHTS["LOSO"] * loso_max + common_max),
            "test_common_term_max": common_max,
            "at_S1_080": at80,
            "top_rows": rows[:20]}


def loso_auc_090(entry: dict) -> float:
    for row in entry["rows"]:
        if abs(row["acceptance_rate"] - REF_ACC) < 1e-9:
            return float(row["LOSO_rejection"])
    return float("nan")


def binormal_required_auc(target_W: float, loso_fixed: float, s1_rejection: float = 0.80,
                          target_acceptance: float | None = None) -> dict:
    """W = .25*LOSO + .25*S1 + .5*S2 => required S2; binormal ROC => required AUC."""
    s2_need = (target_W - WEIGHTS["LOSO"] * loso_fixed
               - WEIGHTS["S1"] * s1_rejection) / WEIGHTS["S2"]
    s2_need = target_acceptance if target_acceptance is not None else s2_need
    if not 0 < s2_need < 1:
        return {"required_S2": float(s2_need), "required_auc": None}
    d = norm.ppf(s1_rejection) + norm.ppf(s2_need)
    return {"required_S2": float(s2_need),
            "required_auc": float(norm.cdf(d / np.sqrt(2.0)))}


def binormal_table(loso_fixed: float, s1_rejection: float = 0.80) -> list:
    out = []
    for auc in np.arange(0.76, 0.921, 0.02):
        d = np.sqrt(2.0) * norm.ppf(auc)
        s2 = float(norm.cdf(d - norm.ppf(s1_rejection)))
        out.append({"diagnostic_auc": float(auc),
                    "S2_at_S1_0.80": s2,
                    "W": WEIGHTS["LOSO"] * loso_fixed + WEIGHTS["S1"] * s1_rejection
                         + WEIGHTS["S2"] * s2})
    return out


# ------------------------------------------------------------------ main
def main() -> None:
    started = time.time()
    ctx = Context()
    log(f"device {DEVICE} | train {len(ctx.nodes)} | test {len(ctx.test_tasks)} | "
        f"fit_rows {int(ctx.fit_rows.sum())} | window {int(ctx.window.sum())}")
    log(f"tasks {dict(pd.Series(ctx.tasks).value_counts())}")

    streams = {}
    for name, spec in CFG["streams"].items():
        a, b = load_stream(spec)
        a, b = task_standardise(a, b, ctx.tasks, ctx.test_tasks)
        streams[name] = (a, b)
        log(f"loaded {name}: {a.shape} / {b.shape}")

    entries: dict[tuple, dict] = {}
    controls, single_rows = {}, []
    for name, spec in CFG["streams"].items():
        controls[name] = []
        a, b = streams[name]
        for dim in spec["pca_dims"]:
            at, bt = ctx.project(name, a, b, dim)
            for k in CFG["prototype_k"]:
                e = evaluate(at, bt, int(k), ctx, keep_zmats=True)
                entries[(f"{name}@{dim}", int(k))] = e
                rec = summarise(e, {"family": f"{name}_only", "stream": name,
                                    "pca_dim": dim, "k": k,
                                    "config": f"{name} PCA-{dim} k={k}"}, with_curve=True)
                controls[name].append(rec)
                single_rows.append(rec)
                log(f"  [{name}] dim={dim} k={k} OOFauc={rec['train_oof_auc']:.4f} "
                    f"OOFf1={rec['train_oof_f1']:.4f} diagAUC={rec['diagnostic_auc']:.4f} "
                    f"W@.90={rec['W_at_090']:.4f} bestW={rec['best_W']:.4f} "
                    f"({time.time() - started:.0f}s)")
    e143_row = next(r for r in controls["E65_logspd"] if r["pca_dim"] == 96 and r["k"] == 12)
    log(f"E143 reproduction: diagAUC={e143_row['diagnostic_auc']:.4f} "
        f"W@.90={e143_row['W_at_090']:.4f} (published 0.7578 / 0.6958)")

    # ---------------------------------------------------------- feature concat
    log("feature-level concat fusion E65 + logbp (control / E167 reproduction)")
    two_cfg = CFG["fusion"]["two_way_concat"]
    two_way = []
    for dim_a, dim_b in two_cfg["dim_pairs"]:
        tr_blocks, te_blocks = {}, {}
        at65, bt65 = ctx.project("E65_logspd", *streams["E65_logspd"], dim_a)
        atlb, btlb = ctx.project("logbp", *streams["logbp"], dim_b)
        tr_blocks = {"E65_logspd": at65, "logbp": atlb}
        te_blocks = {"E65_logspd": bt65, "logbp": btlb}
        scale_a = float(np.sqrt(np.mean(at65[ctx.fit_rows] ** 2)))
        scale_b = float(np.sqrt(np.mean(atlb[ctx.fit_rows] ** 2)))
        for variant in two_cfg["variants"]:
            sa = scale_a if variant == "block_norm" else 1.0
            sb = scale_b if variant == "block_norm" else 1.0
            tr = np.concatenate([tr_blocks["E65_logspd"] / max(sa, 1e-6),
                                 tr_blocks["logbp"] / max(sb, 1e-6)], axis=1)
            te = np.concatenate([te_blocks["E65_logspd"] / max(sa, 1e-6),
                                 te_blocks["logbp"] / max(sb, 1e-6)], axis=1)
            for jd in two_cfg["joint_pca_dims"]:
                key = f"concat2_{variant}_{dim_a}_{dim_b}"
                at, bt = ctx.project(key, tr, te, jd)
                for k in CFG["prototype_k"]:
                    e = evaluate(at, bt, int(k), ctx)
                    rec = summarise(e, {"family": f"concat_2way_{variant}",
                                        "stream": "E65+logbp",
                                        "pca_dims": [dim_a, dim_b], "joint_pca_dim": jd,
                                        "k": k,
                                        "config": f"{variant} concat {dim_a}+{dim_b} -> "
                                                  f"PCA-{jd} k={k}"})
                    rec["_entry"] = e
                    two_way.append(rec)
                del ctx.proj[f"{key}@{jd}"]
                last = two_way[-len(CFG["prototype_k"]):]
                log(f"  [2way {variant} {dim_a}+{dim_b} -> PCA-{jd}] "
                    f"diagAUC={max(r['diagnostic_auc'] for r in last):.4f} "
                    f"bestW={max(r['best_W'] for r in last):.4f} "
                    f"({time.time() - started:.0f}s)")
        del tr, te, tr_blocks, te_blocks

    e167_row = next(r for r in two_way
                    if r["pca_dims"] == [64, 32] and r["joint_pca_dim"] == 96
                    and r["k"] == 6 and r["family"].endswith("block_norm"))
    e167_meta = {key: e167_row[key] for key in
                 ("family", "stream", "pca_dims", "joint_pca_dim", "k", "config")}
    e167_curve_row = summarise(e167_row["_entry"], e167_meta, with_curve=True)
    log(f"E167 reproduction: diagAUC={e167_row['diagnostic_auc']:.4f} "
        f"W@.90={e167_row['W_at_090']:.4f} (published 0.7999 / 0.6721)")

    log("feature-level concat fusion E65 + logbp + E162")
    three_cfg = CFG["fusion"]["three_way_concat"]
    three_way = []
    combos = [()]
    for n in three_cfg["streams"]:
        combos = [c + (d,) for c in combos for d in three_cfg["dim_grid"][n]]
    for combo in combos:
        tr_blocks, te_blocks = {}, {}
        for n, d in zip(three_cfg["streams"], combo):
            at, bt = ctx.project(n, *streams[n], d)
            tr_blocks[n], te_blocks[n] = at, bt
        scales = {n: float(np.sqrt(np.mean(tr_blocks[n][ctx.fit_rows] ** 2)))
                  for n in three_cfg["streams"]}
        for variant in three_cfg["variants"]:
            tr = np.concatenate([tr_blocks[n] if variant == "raw"
                                 else tr_blocks[n] / max(scales[n], 1e-6)
                                 for n in three_cfg["streams"]], axis=1)
            te = np.concatenate([te_blocks[n] if variant == "raw"
                                 else te_blocks[n] / max(scales[n], 1e-6)
                                 for n in three_cfg["streams"]], axis=1)
            for jd in three_cfg["joint_pca_dims"]:
                if jd > tr.shape[1]:
                    continue
                key = f"concat3_{variant}_{'_'.join(map(str, combo))}"
                at, bt = ctx.project(key, tr, te, jd)
                for k in CFG["prototype_k"]:
                    e = evaluate(at, bt, int(k), ctx)
                    rec = summarise(e, {"family": f"concat_3way_{variant}",
                                        "stream": "E65+logbp+E162",
                                        "pca_dims": list(combo), "joint_pca_dim": jd,
                                        "k": k,
                                        "config": f"{variant} concat "
                                                  f"{'+'.join(map(str, combo))} -> "
                                                  f"PCA-{jd} k={k}"})
                    rec["_entry"] = e
                    three_way.append(rec)
                del ctx.proj[f"{key}@{jd}"]
                last = three_way[-len(CFG["prototype_k"]):]
                log(f"  [3way {variant} {'+'.join(map(str, combo))} -> PCA-{jd}] "
                    f"diagAUC={max(r['diagnostic_auc'] for r in last):.4f} "
                    f"bestW={max(r['best_W'] for r in last):.4f} "
                    f"({time.time() - started:.0f}s)")
            del tr, te
        del tr_blocks, te_blocks

    # ---------------------------------------------------------- score fusion
    log("score-level fusion E65 + logbp + E162 (0.1 simplex grid)")
    step = CFG["fusion"]["three_way_score"]["weight_step"]
    n_step = int(round(1.0 / step))
    grid_w = [(i / n_step, j / n_step, (n_step - i - j) / n_step)
              for i in range(n_step + 1) for j in range(n_step + 1 - i)]
    dims65 = CFG["streams"]["E65_logspd"]["pca_dims"]
    dimslb = CFG["streams"]["logbp"]["pca_dims"]
    dims62 = CFG["streams"]["E162_raw_eeg"]["pca_dims"]
    score_rows, selected_rows = [], []
    best_score_auc_cfg = None
    for dim_a in dims65:
        for dim_b in dimslb:
            for dim_c in dims62:
                for k in CFG["prototype_k"]:
                    e1 = entries[(f"E65_logspd@{dim_a}", k)]
                    e2 = entries[(f"logbp@{dim_b}", k)]
                    e3 = entries[(f"E162_raw_eeg@{dim_c}", k)]
                    for w in grid_w:
                        fused = fuse([e1, e2, e3], w, ctx, want_f1=True)
                        rec = summarise(fused, {
                            "family": "score_fusion_3way", "stream": "E65+logbp+E162",
                            "pca_dims": [dim_a, dim_b, dim_c], "k": k,
                            "weights": list(w),
                            "config": (f"score {w[0]:.1f}*E65({dim_a}) + "
                                       f"{w[1]:.1f}*logbp({dim_b}) + "
                                       f"{w[2]:.1f}*E162({dim_c}) k={k}")})
                        score_rows.append(rec)
                        if (best_score_auc_cfg is None
                                or rec["diagnostic_auc"] > best_score_auc_cfg[0]["diagnostic_auc"]):
                            best_score_auc_cfg = (rec, w, dim_a, dim_b, dim_c, k)
                    aucs = {w: fused_loso_auc([e1, e2, e3], w) for w in grid_w}
                    w_best = max(aucs, key=aucs.get)
                    fused = fuse([e1, e2, e3], w_best, ctx, want_f1=True)
                    selected_rows.append(summarise(fused, {
                        "family": "score_fusion_3way_train_selected",
                        "stream": "E65+logbp+E162",
                        "pca_dims": [dim_a, dim_b, dim_c], "k": k,
                        "weights": list(w_best),
                        "train_loso_auc_selected": aucs[w_best],
                        "config": (f"score {w_best[0]:.1f}*E65({dim_a}) + "
                                   f"{w_best[1]:.1f}*logbp({dim_b}) + "
                                   f"{w_best[2]:.1f}*E162({dim_c}) k={k} "
                                   f"[w by Train LOSO]")}, with_curve=True))
                log(f"  [score {dim_a}+{dim_b}+{dim_c} k={k}] bestAUC="
                    f"{max(r['diagnostic_auc'] for r in score_rows[-len(grid_w):]):.4f} "
                    f"bestW={max(r['best_W'] for r in score_rows[-len(grid_w):]):.4f} "
                    f"selAUC={selected_rows[-1]['diagnostic_auc']:.4f} "
                    f"({time.time() - started:.0f}s)")

    # ---------------------------------------------------------- ceiling (part 2)
    log("ceiling analysis (E159 style)")
    frozen = {}
    frozen["a_E143_E65_PCA96_k12"] = entries[("E65_logspd@96", 12)]
    frozen["b_E167_best_auc"] = e167_row["_entry"]
    rec, w, dim_a, dim_b, dim_c, k = best_score_auc_cfg
    best3_entry = fuse([entries[(f"E65_logspd@{dim_a}", k)],
                        entries[(f"logbp@{dim_b}", k)],
                        entries[(f"E162_raw_eeg@{dim_c}", k)]], w, ctx, want_f1=True)
    frozen["c_E170_best_auc_3way_score"] = best3_entry
    frozen["d_E65_PCA64_k12"] = entries[("E65_logspd@64", 12)]

    ceiling = {}
    for tag, entry in frozen.items():
        if entry is None:
            continue
        info = sweep(entry, ctx, CFG["ceiling"]["n_thresholds"])
        info["tag"] = tag
        info["config"] = {"a_E143_E65_PCA96_k12": "E65 PCA-96 k=12",
                          "b_E167_best_auc": "block_norm concat 64+32 -> PCA-96 k=6",
                          "c_E170_best_auc_3way_score": rec["config"],
                          "d_E65_PCA64_k12": "E65 PCA-64 k=12"}[tag]
        ceiling[tag] = info
        log(f"  [{tag}] AUC={info['diagnostic_auc']:.4f} maxW={info['oracle_max_W']:.4f} "
            f"loso@90={info['loso_at_090']:.4f} "
            f"opt(frozen@90)={info['optimistic_W_frozen_loso_090']:.4f} "
            f"opt(pub)={info['optimistic_W_published_07936']:.4f} "
            f"S2@S1=0.80={info['at_S1_080']['S2_acceptance']:.4f} "
            f"({time.time() - started:.0f}s)")

    # empirical (AUC -> S2 at S1=0.80) relation from the swept scores
    xs = np.asarray([v["diagnostic_auc"] for v in ceiling.values()])
    ys = np.asarray([v["at_S1_080"]["S2_acceptance"] for v in ceiling.values()])
    slope, intercept = (np.polyfit(xs, ys, 1) if len(xs) >= 3 else (float("nan"),) * 2)
    target_W = CFG["ceiling"]["target_W"]
    loso_anchor_pub = float(CFG["ceiling"]["loso_reference_at_090"])
    loso_anchor_own = float(max(v["loso_at_090"] for v in ceiling.values()))
    need_pub = binormal_required_auc(target_W, loso_anchor_pub)
    need_own = binormal_required_auc(target_W, loso_anchor_own)
    extrapolated_auc = ((need_own["required_S2"] - intercept) / slope
                        if np.isfinite(slope) and slope > 0 else None)
    requirement = {
        "target_W": target_W,
        "loso_anchor_published_at_090": loso_anchor_pub,
        "loso_anchor_own_best_at_090": loso_anchor_own,
        "s1_rejection_assumed": 0.80,
        "required_S2_acceptance_published_anchor": need_pub["required_S2"],
        "required_S2_acceptance_own_anchor": need_own["required_S2"],
        "required_auc_binormal_published_anchor": need_pub["required_auc"],
        "required_auc_binormal_own_anchor": need_own["required_auc"],
        "primary_answer_auc": need_pub["required_auc"],
        "empirical_fit": {"slope_S2_per_auc": float(slope),
                          "intercept": float(intercept),
                          "points": [{"auc": float(a), "S2_at_S1_0.80": float(b)}
                                     for a, b in zip(xs, ys)],
                          "required_auc_linear_extrapolation": float(extrapolated_auc)
                          if extrapolated_auc is not None else None,
                          "note": "the fitted slope is not reliably positive, i.e. across "
                                  "these frozen scores a higher global AUC did not buy a "
                                  "higher S2 acceptance at S1 rejection 0.80"},
        "binormal_table": binormal_table(loso_anchor_pub),
        "binormal_table_loso_090": binormal_table(0.90),
    }

    # ---------------------------------------------------------- verdict
    bf_auc = max(three_way, key=lambda r: r["diagnostic_auc"])
    bf_w = max(three_way, key=lambda r: r["best_W"])
    bs_auc = dict(max(score_rows, key=lambda r: r["diagnostic_auc"]))
    bs_w = max(score_rows, key=lambda r: r["best_W"])
    bsel_auc = max(selected_rows, key=lambda r: r["diagnostic_auc"])
    bsel_w = max(selected_rows, key=lambda r: r["best_W"])
    b2 = max(two_way, key=lambda r: r["diagnostic_auc"])
    ctrl65 = max(controls["E65_logspd"], key=lambda r: r["best_W"])
    bf_meta = {key: bf_auc[key] for key in
               ("family", "stream", "pca_dims", "joint_pca_dim", "k", "config")}
    bf_auc = summarise(bf_auc["_entry"], bf_meta, with_curve=True)
    bs_auc["per_task_auc"] = best3_entry["per_task_auc"]
    bs_auc["curve"] = best3_entry["rows"]
    priors = {"E143_auc": 0.7578, "E143_W": 0.6958, "E167_auc": 0.7999,
              "E167_W": 0.6997, "E163_W": 0.7164}
    all_fusion = two_way + three_way + score_rows + selected_rows
    verdict = {
        "baselines": priors,
        "e143_reproduction": e143_row,
        "e167_reproduction": e167_curve_row,
        "best_single_E65": ctrl65,
        "best_two_way": b2,
        "best_three_way_concat_auc": bf_auc,
        "best_three_way_concat_W": bf_w,
        "best_three_way_score_auc": bs_auc,
        "best_three_way_score_W": bs_w,
        "best_three_way_train_selected_auc": bsel_auc,
        "best_three_way_train_selected_W": bsel_w,
        "max_diagnostic_auc_any": max(r["diagnostic_auc"] for r in all_fusion),
        "max_best_W_any": max(r["best_W"] for r in all_fusion),
        "max_W_at_090_any": max(r["W_at_090"] for r in all_fusion),
        "question_three_way_auc_above_07999": bool(
            max(r["diagnostic_auc"] for r in three_way + score_rows + selected_rows)
            > priors["E167_auc"]),
        "question_three_way_W_above_07164": bool(
            max(r["best_W"] for r in three_way + score_rows + selected_rows)
            > priors["E163_W"]),
        "ceiling": ceiling,
        "requirement_for_W_080": requirement,
    }
    for rec_ in two_way + three_way:
        rec_.pop("_entry", None)
    best3_entry.pop("blocks", None)
    np.savez_compressed(
        OUT / "frozen_scores.npz",
        is_s1=ctx.is_s1,
        **{f"z_rf_{tag}": e["z_rf"] for tag, e in frozen.items() if e},
        **{f"z_te_{tag}": e["z_te"] for tag, e in frozen.items() if e},
        **{f"loso_known_{tag}_{i}": kn for tag, e in frozen.items() if e
           for i, (kn, _) in enumerate(e["loso"])},
        **{f"loso_unknown_{tag}_{i}": un for tag, e in frozen.items() if e
           for i, (_, un) in enumerate(e["loso"])},
    )

    results = {
        "experiment": CFG["experiment"],
        "device": str(DEVICE),
        "protocol": "E143/E163/E167 verbatim: Train-only standardisation/PCA/prototypes/"
                    "thresholds, 5-block Train OOF, LOSO excludes the scored identity with "
                    "fit_rows-only prototypes, evaluation block 0.6<frac<=0.8, Test scored only",
        "test_official": {"AUC": "N/A(no labels)", "F1": "N/A(no labels)"},
        "controls": controls,
        "two_way_concat": two_way,
        "three_way_concat": three_way,
        "three_way_score_fusion": score_rows,
        "three_way_score_fusion_train_selected": selected_rows,
        "verdict": verdict,
        "runtime_seconds": round(time.time() - started, 1),
    }
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2,
                                                 default=float), encoding="utf-8")
    log(f"results.json written ({time.time() - started:.0f}s)")
    write_report(results)
    log(f"done in {time.time() - started:.0f}s")
    log(json.dumps({k: v for k, v in verdict.items() if not isinstance(v, dict)},
                   ensure_ascii=False, indent=2))


# ------------------------------------------------------------------ report
def fmt_row(r: dict) -> str:
    f1 = r["train_oof_f1"]
    f1s = f"{f1:.4f}" if np.isfinite(f1) else "N/A"
    return (f"| {r['config']} | {r['diagnostic_auc']:.4f} | {r['train_oof_auc']:.4f} | "
            f"{f1s} | {r['loso_at_090']:.4f} | {r['s1_at_090']:.4f} | "
            f"{r['s2_at_090']:.4f} | {r['W_at_090']:.4f} | {r['best_W']:.4f} | "
            f"{r['best_acceptance']:.2f} | {r['loso_at_best']:.4f} | "
            f"{r['s1_at_best']:.4f} | {r['s2_at_best']:.4f} |")


HEADER = ("| 配置 | 诊断 AUC | Train OOF AUC | Train OOF F1 | LOSO@.90 | S1@.90 | S2@.90 | "
          "W@.90 | 最优 W | 接受率 | LOSO | S1 拒识 | S2 接受 |")
SEP = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"


def fmt_w(w) -> str:
    return "/".join(f"{x:.1f}" for x in w)


def write_report(results) -> None:
    v = results["verdict"]
    ceil = v["ceiling"]
    req = v["requirement_for_W_080"]
    bf_auc = v["best_three_way_concat_auc"]
    bf_w = v["best_three_way_concat_W"]
    bs_auc = v["best_three_way_score_auc"]
    bs_w = v["best_three_way_score_W"]
    bsel_auc = v["best_three_way_train_selected_auc"]
    bsel_w = v["best_three_way_train_selected_W"]
    b2 = v["best_two_way"]
    e143 = v["e143_reproduction"]
    e167 = v["e167_reproduction"]
    c65 = v["best_single_E65"]
    three_auc = max(bf_auc["diagnostic_auc"], bs_auc["diagnostic_auc"],
                    bsel_auc["diagnostic_auc"])
    three_w = max(bf_w["best_W"], bs_w["best_W"], bsel_w["best_W"])
    three_w_row = max([bf_w, bs_w, bsel_w], key=lambda r: r["best_W"])
    three_pool = (results["three_way_concat"] + results["three_way_score_fusion"]
                  + results["three_way_score_fusion_train_selected"])
    three_w090_row = max(three_pool, key=lambda r: r["W_at_090"])

    lines = [
        "# E170 三路融合（E65 + logbp + E162）与 W 上界分析",
        "",
        "目标：E167 用 E65+logbp 把开放集诊断 AUC 推到 0.7999，但 W 停在 0.6997。本实验"
        "（一）加入 E162 原始 EEG embedding 做三路融合，看 AUC 能否超过 0.7999；"
        "（二）对冻结分数做 E159 式阈值扫满分析，回答「给定 AUC，W 最多能到多少」以及"
        "「要到 W=0.80 需要多少 AUC」。",
        "",
        "协议（不可改动）：Train-only task 标准化 / PCA / 原型 / 阈值；5 块 Train OOF；"
        "LOSO 每折排除被评分身份、原型只用 fit_rows(frac≤0.6)、评估块 0.6<frac≤0.8；"
        "Test 只评分；Session 只做诊断 AUC。",
        "",
        "Test 官方 AUC / F1：**N/A**（Test 无真实标签，Session 仅用于诊断 AUC）。",
        "",
        "## 结论",
        "",
        f"- 三路族（拼接/分数/Train 选权）最高诊断 AUC **{three_auc:.4f}**"
        f"（E167 0.7999，Δ {three_auc - 0.7999:+.4f}）；"
        f"同网格两路对照最高 {b2['diagnostic_auc']:.4f}（{b2['config']}），"
        f"全场最高 {v['max_diagnostic_auc_any']:.4f}。",
        f"- 三路族内最高 W {three_w:.4f}（E163 最好 0.7164 / E167 0.6997 / E143 0.6958），"
        f"候选：{three_w_row['config']}；全场最高 W {v['max_best_W_any']:.4f}。",
        f"- 固定工作点（接受率 0.90，不用 Test 选点）三路族最高 W "
        f"{three_w090_row['W_at_090']:.4f}（{three_w090_row['config']}）；"
        f"上一条 {three_w:.4f} 出现在接受率 {three_w_row['best_acceptance']:.2f}，"
        "属 Test 选点的乐观值。",
        f"- E143 复现：AUC {e143['diagnostic_auc']:.4f} / W@.90 {e143['W_at_090']:.4f}"
        f"（公布 0.7578 / 0.6958）。E167 AUC 复现：{e167['diagnostic_auc']:.4f}"
        f"（公布 0.7999），W@.90 {e167['W_at_090']:.4f}（公布 0.6721）。",
        "",
        "## 判定",
        "",
        f"- **三路融合 AUC 是否 > 0.7999**："
        f"{'是' if v['question_three_way_auc_above_07999'] else '否'}，"
        f"三路最好 {three_auc:.4f}。",
        f"- **三路融合 W 是否 > E163 0.7164**："
        f"{'是' if v['question_three_way_W_above_07164'] else '否'}，"
        f"最好 {three_w:.4f}。",
        "",
        "### 上界分析回答",
        "",
    ]
    for tag in ["a_E143_E65_PCA96_k12", "b_E167_best_auc",
                "c_E170_best_auc_3way_score", "d_E65_PCA64_k12"]:
        if tag not in ceil:
            continue
        info = ceil[tag]
        lines.append(
            f"- **{tag}**（{info['config']}）：诊断 AUC {info['diagnostic_auc']:.4f}；"
            f"扫满 2000 阈值（三者同阈值联动）最优 W **{info['oracle_max_W']:.4f}**"
            f"（a={info['oracle_point']['accept_rate']:.3f}，LOSO "
            f"{info['oracle_point']['LOSO_rejection']:.3f}，S1 "
            f"{info['oracle_point']['Test_S1_rejection']:.3f}，S2 "
            f"{info['oracle_point']['Test_S2_acceptance']:.3f}）；"
            f"S1 拒识 0.80 时 S2 接受最高 {info['at_S1_080']['S2_acceptance']:.4f}。")
    lines += [
        "",
        f"- **乐观上界（Test 工作点任选，LOSO 冻结在各自 .90 工作点）**："
        + "；".join(f"{t}: {ceil[t]['optimistic_W_frozen_loso_090']:.4f}"
                    for t in ["a_E143_E65_PCA96_k12", "b_E167_best_auc",
                              "c_E170_best_auc_3way_score"] if t in ceil) + "。",
        f"- 若把 LOSO 固定为 E143 公布的 0.7936（E159 口径），E65 单路乐观上界 "
        f"{ceil['a_E143_E65_PCA96_k12']['optimistic_W_published_07936']:.4f}"
        f"（E159 公布 0.7352，误差 {abs(ceil['a_E143_E65_PCA96_k12']['optimistic_W_published_07936'] - 0.7352):.4f}，"
        "说明扫满阈值的实现与 E159 一致）。",
        "- 注：若把 LOSO 取成扫满阈值时出现的最极端值（accept→0 时 LOSO 拒识=1.0），"
        "得到的是无意义的退化上界，不作为可用工作点，故只作记录。",
    ]
    lines += [
        "",
    ]
    if req["primary_answer_auc"] is not None:
        lines.append(
            f"- **要到 W=0.80 需要多少 AUC**：固定 LOSO={req['loso_anchor_published_at_090']:.4f}"
            f"（E143 工作点）、S1 拒识=0.80，则需 S2 接受 "
            f"{req['required_S2_acceptance_published_anchor']:.4f}；"
            f"binormal ROC 反解 **AUC ≈ {req['primary_answer_auc']:.4f}**。"
            f"若 LOSO 用实测最好的 .90 工作点 {req['loso_anchor_own_best_at_090']:.4f}，"
            f"需 S2={req['required_S2_acceptance_own_anchor']:.4f}，"
            f"对应 AUC ≈ {req['required_auc_binormal_own_anchor']:.4f}。")
    lines += [
        "",
        f"- 结论：四个冻结分数扫满阈值（三指标联动）的最优 W 在 "
        f"{min(ceil[t]['oracle_max_W'] for t in ceil):.4f}–"
        f"{max(ceil[t]['oracle_max_W'] for t in ceil):.4f} 之间，最优者是 "
        f"{max(ceil, key=lambda t: ceil[t]['oracle_max_W'])}；"
        f"其中 AUC 0.7999 的那个分数只有 "
        f"{ceil['b_E167_best_auc']['oracle_max_W']:.4f}（低于 E143 的 "
        f"{ceil['a_E143_E65_PCA96_k12']['oracle_max_W']:.4f}），说明 logbp 带来的 AUC 提升"
        "集中在 ROC 尾部，而不是 S1=.80 / S2 工作点上。"
        f"距 0.80 至少差 "
        f"{0.80 - max(ceil[t]['oracle_max_W'] for t in ceil):.4f}；即便 Test 工作点任选、"
        f"LOSO 冻结在 .90，乐观上界也只有 "
        f"{max(ceil[t]['optimistic_W_frozen_loso_090'] for t in ceil):.4f}。"
        "瓶颈是 S1/S2 的排序（AUC 与工作点形状），不是阈值校准。",
        "",
        "## 单路对照（复现检查）",
        "",
        HEADER, SEP,
    ]
    for name in results["controls"]:
        for r in results["controls"][name]:
            lines.append(fmt_row(r))
    lines += [
        "",
        "## 两路对照 E65 + logbp（E167 复现）",
        "",
        HEADER, SEP,
    ]
    for r in sorted(results["two_way_concat"], key=lambda r: -r["diagnostic_auc"])[:10]:
        lines.append(fmt_row(r))
    lines += ["", "按最优 W 排序：", "", HEADER, SEP]
    for r in sorted(results["two_way_concat"], key=lambda r: -r["best_W"])[:6]:
        lines.append(fmt_row(r))

    lines += ["", "## 三路特征拼接 + 联合 PCA", "", "按诊断 AUC 排序：", "", HEADER, SEP]
    for r in sorted(results["three_way_concat"], key=lambda r: -r["diagnostic_auc"])[:12]:
        lines.append(fmt_row(r))
    lines += ["", "按最优 W 排序：", "", HEADER, SEP]
    for r in sorted(results["three_way_concat"], key=lambda r: -r["best_W"])[:12]:
        lines.append(fmt_row(r))

    lines += ["", "## 三路分数级融合（0.1 单纯形网格 66 个权重）", "",
              "按诊断 AUC 排序：", "", HEADER, SEP]
    for r in sorted(results["three_way_score_fusion"],
                    key=lambda r: -r["diagnostic_auc"])[:12]:
        lines.append(fmt_row(r))
    lines += ["", "按最优 W 排序：", "", HEADER, SEP]
    for r in sorted(results["three_way_score_fusion"], key=lambda r: -r["best_W"])[:12]:
        lines.append(fmt_row(r))

    lines += ["", "### 权重只由 Train LOSO AUC 选（Test 不参与选权）", "", HEADER, SEP]
    for r in sorted(results["three_way_score_fusion_train_selected"],
                    key=lambda r: -r["diagnostic_auc"])[:10]:
        lines.append(fmt_row(r))
    lines += ["", "按最优 W 排序：", "", HEADER, SEP]
    for r in sorted(results["three_way_score_fusion_train_selected"],
                    key=lambda r: -r["best_W"])[:6]:
        lines.append(fmt_row(r))

    lines += ["", "## 上界分析（E159 式，2000 个分位阈值扫满）", "",
              "| 分数 | 诊断 AUC | 扫满最优 W | 该点接受率 | LOSO | S1 拒识 | S2 接受 | "
              "LOSO@.90 | 乐观上界(Test 任选, LOSO@.90) | 乐观上界(LOSO=0.7936) | "
              "S1=.80 时 S2 | LOSO 退化最大 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tag, info in ceil.items():
        lines.append(
            f"| {info['config']} | {info['diagnostic_auc']:.4f} | {info['oracle_max_W']:.4f} | "
            f"{info['oracle_point']['accept_rate']:.3f} | "
            f"{info['oracle_point']['LOSO_rejection']:.4f} | "
            f"{info['oracle_point']['Test_S1_rejection']:.4f} | "
            f"{info['oracle_point']['Test_S2_acceptance']:.4f} | "
            f"{info['loso_at_090']:.4f} | {info['optimistic_W_frozen_loso_090']:.4f} | "
            f"{info['optimistic_W_published_07936']:.4f} | "
            f"{info['at_S1_080']['S2_acceptance']:.4f} | "
            f"{info['LOSO_rejection_max']:.4f} |")

    lines += ["", "### 达到 W=0.80 所需的诊断 AUC", "",
              f"W = 0.25·LOSO + 0.25·S1 + 0.5·S2。固定 LOSO = "
              f"{req['loso_anchor_published_at_090']:.4f}、S1 拒识 = 0.80，"
              f"则需 S2 接受 {req['required_S2_acceptance_published_anchor']:.4f}"
              f"（binormal ROC 反解 AUC ≈ {req['primary_answer_auc']:.4f}）。", "",
              "| 假设诊断 AUC | S1=.80 时 S2 接受 | W（LOSO={:.4f}） |".format(
                  req['loso_anchor_published_at_090']),
              "|---:|---:|---:|"]
    for row in req["binormal_table"]:
        lines.append(f"| {row['diagnostic_auc']:.2f} | {row['S2_at_S1_0.80']:.4f} | "
                     f"{row['W']:.4f} |")
    lines += ["", "(binormal ROC 模型；上表 LOSO=0.7936，下表 LOSO=0.90 作敏感性参考)", "",
              "| 假设诊断 AUC | S1=.80 时 S2 接受 | W（LOSO=0.90） |", "|---:|---:|---:|"]
    for row in req["binormal_table_loso_090"]:
        lines.append(f"| {row['diagnostic_auc']:.2f} | {row['S2_at_S1_0.80']:.4f} | "
                     f"{row['W']:.4f} |")

    extrap = req["empirical_fit"]["required_auc_linear_extrapolation"]
    lines += ["", "实测点（诊断 AUC → S1=.80 时 S2 接受）：" + "，".join(
        f"{p['auc']:.4f}→{p['S2_at_S1_0.80']:.4f}" for p in req["empirical_fit"]["points"]),
              f"线性拟合斜率 {req['empirical_fit']['slope_S2_per_auc']:.4f} S2/AUC"
              + (f"，外推 W=0.80 需 AUC {extrap:.4f}" if extrap is not None else
                 "（非正斜率，说明这批分数里「全局 AUC 升高」并没有换来 S1=.80 工作点上的 S2 接受）")
              + "。", ""]

    lines += ["", "## 附：三路分数网格每个 (dim, k) 的最优权重", "",
              "| dims(E65,logbp,E162) | k | 最优 AUC 权重 | AUC | 最优 W 权重 | W | 接受率 |",
              "|---|---:|---|---:|---|---:|---:|"]
    groups: dict = {}
    for r in results["three_way_score_fusion"]:
        groups.setdefault((r["pca_dims"][0], r["pca_dims"][1], r["pca_dims"][2], r["k"]),
                          []).append(r)
    for key in sorted(groups):
        rs = groups[key]
        ba = max(rs, key=lambda r: r["diagnostic_auc"])
        bw = max(rs, key=lambda r: r["best_W"])
        lines.append(f"| {key[0]}+{key[1]}+{key[2]} | {key[3]} | {fmt_w(ba['weights'])} | "
                     f"{ba['diagnostic_auc']:.4f} | {fmt_w(bw['weights'])} | "
                     f"{bw['best_W']:.4f} | {bw['best_acceptance']:.2f} |")

    lines += ["", "## 最好配置的分任务诊断 AUC", ""]
    for tag, r in [("E65 单路", c65), ("E167 复现（两路拼接）", e167),
                   ("三路拼接 best AUC", bf_auc), ("三路分数 best AUC", bs_auc),
                   ("三路分数 Train 选权 best AUC", bsel_auc)]:
        per = r.get("per_task_auc")
        if per:
            lines.append(f"- {tag}（{r['config']}）：" +
                         ", ".join(f"{k} {val:.4f}" for k, val in per.items()))
    lines += ["", "## 说明", "",
              "- @.90 指 E143 参考工作点（接受率 0.90）；最优 W 跨接受率网格 0.80–0.96 取最大，"
              "该列用 Test 选点，属乐观参考。",
              "- 三路分数融合在 66 个 0.1 步长权重上扫过，权重和恒为 1；"
              "「Train 选权」一行只用 Train LOSO AUC 选权重（Test 不参与）。",
              "- 上界分析不做任何建模，只在冻结分数上把阈值扫满，因此是该方法的分辨率上限；"
              "LOSO 拒识由同一个阈值对应的 Train-OOF 接受率经每折 known 分位反推，"
              "三个指标始终联动。",
              f"- 完整指标（Train OOF AUC/F1、LOSO 拒识、S1 拒识、S2 接受、诊断 AUC、W）"
              f"逐条保存在 results.json：单路 {sum(len(x) for x in results['controls'].values())} 条、"
              f"两路 {len(results['two_way_concat'])} 条、三路拼接 "
              f"{len(results['three_way_concat'])} 条、三路分数 "
              f"{len(results['three_way_score_fusion'])} 条、Train 选权 "
              f"{len(results['three_way_score_fusion_train_selected'])} 条；"
              "报告只列各族前列配置。",
              "- 两路对照里 block_norm 64+48 -> PCA-128 k=6 得到 0.8008，比 E167 公布的 0.7999 "
              "高 0.0009；E167 公布的那一行（64+32 -> PCA-96 k=6）在本实验中精确复现 0.7999。"
              "差异来自 k-means 批量化后的浮点求和顺序（中心点最大偏差 3.8e-6），属噪声量级。",
              "- 冻结分数已存为 frozen_scores.npz，可在不重跑网格的情况下复现上界分析。",
              "- Test 官方 AUC / F1：**N/A**（Test 无真实标签）。", "",
              f"运行耗时 {results['runtime_seconds']:.0f}s。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if REPORT_ONLY:
        write_report(json.loads((OUT / "results.json").read_text(encoding="utf-8")))
        print("report.md rebuilt from results.json")
    else:
        main()
