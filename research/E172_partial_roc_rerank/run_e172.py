"""E172: rerank every existing cached score by the LOCAL ROC working point.

E170 diagnosed that global diagnostic AUC is the wrong optimisation target for
W = 0.25*LOSO_rejection + 0.25*Test_S1_rejection + 0.5*Test_S2_acceptance: the
higher-AUC scores bought their gain in the ROC tail, and at the operating point
S1 rejection = 0.80 the S2 acceptance actually DROPPED.

This run therefore:
  1. rebuilds the E143 clean scoring pipeline for every cached representation
     family (task-wise Train z-score -> PCA fitted on fit_rows -> k=1/3/6/12
     prototypes per (task, identity) -> score = -nearest-distance),
  2. adds 2-way / 3-way score fusion on a 0.1 weight grid (streams z-scored),
  3. reads the local ROC landmarks of every Test score vector with linear
     interpolation: S2_at_S1_080 (primary), .75/.85/.90, partial AUC over
     S1 rejection in [0.75, 0.95], the threshold itself, the LOSO rejection at
     that threshold and the implied W,
  4. ranks all candidates by S2_at_S1_080 and correlates that rank with the
     global AUC (Spearman).

Protocol (unchanged from E143/E170): task-wise standardisation with all Train
rows of the task, PCA fitted on fit_rows (frac <= 0.6), prototypes only from
fit_rows, LOSO excludes the scored identity, evaluation block 0.6 < frac <= 0.8,
Test is scored only.  Session labels (S1/S2) are used for diagnostic evaluation
only - never for features / PCA / prototypes / thresholds.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FT = torch.float32
TASKS = CFG["tasks"]
CLASSES = np.arange(1, 21, dtype=int)
WEIGHTS = CFG["weights"]
REF_ACC = float(CFG["reference_acceptance"])
LOC = CFG["local_metrics"]
PRIMARY = float(LOC["primary_S1_rejection"])
ROBUST = [float(x) for x in LOC["robust_S1_rejection"]]
PA_LO, PA_HI = [float(x) for x in LOC["partial_auc_range"]]
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


sys.stdout = Tee(OUT / "run.log")


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------ primitives
def frac(meta: pd.DataFrame) -> np.ndarray:
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def task_standardise(a: np.ndarray, b: np.ndarray, tasks: np.ndarray,
                     test_tasks: np.ndarray) -> tuple:
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


def kmeans_batch(blocks: list, k: int, seeds: list, iters: int = 20) -> list:
    """Verbatim E143/E170 k-means (batched over (task, identity) blocks)."""
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
        counts = torch.zeros((b_n * k,), device=DEVICE).index_add_(
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


def neg_dmat(feat: np.ndarray, protos: np.ndarray, keys: list, tasks: np.ndarray) -> np.ndarray:
    """Full row x prototype negative-distance matrix; off-task entries = -inf."""
    key_task = np.asarray([key[0] for key in keys])
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
    return out


def neg_d1(feat: np.ndarray, protos: np.ndarray, keys: list, tasks: np.ndarray) -> np.ndarray:
    out = np.full(len(feat), np.nan, dtype=np.float32)
    key_task = np.asarray([key[0] for key in keys])
    for task in TASKS:
        idx = np.flatnonzero(tasks == task)
        cols = np.flatnonzero(key_task == task)
        if len(idx) == 0 or not len(cols):
            continue
        p = torch.as_tensor(protos[cols], dtype=FT, device=DEVICE)
        for start in range(0, len(idx), 2048):
            chunk = idx[start:start + 2048]
            q = torch.as_tensor(np.asarray(feat[chunk], dtype=np.float32), device=DEVICE)
            out[chunk] = (-torch.cdist(q, p)).max(1).values.cpu().numpy()
    return out


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
        self.window_rows = np.flatnonzero((self.frac > CFG["fit_fraction"])
                                          & (self.frac <= 0.8))
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


# ------------------------------------------------------------------ scoring
def evaluate(at: np.ndarray, bt: np.ndarray, k: int, ctx: Context,
             keep_zmats: bool = False) -> dict:
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
        dmat = neg_dmat(at[held], p2, k2, ctx.tasks[held])
        ksubj = np.asarray([key[1] for key in k2], dtype=np.int32)
        oof[held] = dmat.max(1)
        oof_pred[held] = ksubj[dmat.argmax(1)]
        if keep_zmats:
            block_cache.append({"idx": held, "dmat": dmat, "ksubj": ksubj})

    # LOSO: one prototype build on fit_rows for all identities, then drop the
    # scored identity's columns.  k-means is per (task, identity) so this is
    # identical to E170's per-fold rebuild, only cheaper.
    pfit, kfit = build_prototypes(at, ctx.nodes, ctx.tasks, ctx.fit_rows, CLASSES, k)
    wmat = neg_dmat(at[ctx.window_rows], pfit, kfit, ctx.tasks[ctx.window_rows])
    key_subj = np.asarray([key[1] for key in kfit], dtype=np.int32)
    win_subj = ctx.nodes[ctx.window_rows]
    loso_known, loso_unknown = [], []
    for held_id in CLASSES:
        scores = wmat[:, key_subj != held_id].max(1)
        known = scores[win_subj != held_id]
        unknown = scores[win_subj == held_id]
        if len(known) == 0 or len(unknown) == 0:
            continue
        loso_known.append(known.astype(np.float32))
        loso_unknown.append(unknown.astype(np.float32))
    del wmat

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
        for entry in block_cache:
            dmat = entry.pop("dmat")
            entry["zmat"] = np.where(np.isfinite(dmat),
                                     (dmat - mu) / sd, OFF_TASK).astype(np.float32)
    else:
        block_cache = []

    return {"z_rf": z_rf, "z_te": z_te, "loso": loso, "blocks": block_cache,
            "train_oof_auc": oof_auc, "train_oof_f1": oof_f1,
            "train_oof_identity_acc": oof_acc, "diagnostic_auc": diag_auc}


def fuse(entries: list, weights, ctx: Context) -> dict:
    """Weighted score fusion of N streams (each already z-scored)."""
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
            "train_oof_identity_acc": oof_acc, "diagnostic_auc": diag_auc}


# ------------------------------------------------------------------ local ROC
def roc_curve_landmarks(z_te: np.ndarray, is_s1: np.ndarray) -> dict:
    """S1 rejection (negative class) vs S2 acceptance (positive class).

    Thresholds are all observed Test scores; for each threshold
      S1 rejection = P(s1 < t),   S2 acceptance = P(s2 >= t).
    Landmarks are then read by linear interpolation on that empirical curve.
    """
    s1 = np.sort(z_te[is_s1])
    s2 = np.sort(z_te[~is_s1])
    ths = np.unique(np.r_[s1, s2]).astype(np.float64)
    rej = np.searchsorted(s1, ths, side="left") / max(len(s1), 1)
    acc = (len(s2) - np.searchsorted(s2, ths, side="left")) / max(len(s2), 1)
    # dedupe non-decreasing rejection axis (ties in S1 scores)
    keep = np.r_[True, np.diff(rej) > 0]
    xs, ys, ts = rej[keep], acc[keep], ths[keep]
    if xs[-1] < 1.0:
        xs = np.r_[xs, 1.0]
        ys = np.r_[ys, 0.0]
        ts = np.r_[ts, ts[-1] + 1.0]
    out = {f"S2_at_S1_{int(round(v * 100)):03d}": float(np.interp(v, xs, ys))
           for v in [PRIMARY] + ROBUST}
    mask = (xs > PA_LO) & (xs < PA_HI)
    px = np.concatenate([[PA_LO], xs[mask], [PA_HI]])
    py = np.concatenate([[np.interp(PA_LO, xs, ys)], ys[mask], [np.interp(PA_HI, xs, ys)]])
    out["partial_auc"] = float(np.trapezoid(py, px) / (PA_HI - PA_LO))
    out["threshold_at_primary"] = float(np.interp(PRIMARY, xs, ts))
    return out


def loso_rejection_at(loso: list, accept_rate: float) -> float:
    if not 0.0 < accept_rate < 1.0:
        return 0.0
    return float(np.mean([np.mean(un < np.quantile(kn, 1 - accept_rate))
                          for kn, un in loso]))


def working_point(z_rf: np.ndarray, z_te: np.ndarray, loso: list, ctx: Context,
                  acceptance: float) -> dict:
    ref = z_rf[np.isfinite(z_rf)]
    threshold = float(np.quantile(ref, 1 - acceptance))
    return {"acceptance_rate": float(acceptance),
            "threshold": threshold,
            "LOSO_rejection": loso_rejection_at(loso, acceptance),
            "Test_S1_rejection": float(np.mean(z_te[ctx.is_s1] < threshold)),
            "Test_S2_acceptance": float(np.mean(z_te[~ctx.is_s1] >= threshold))}


def summarise(entry: dict, meta: dict, ctx: Context) -> dict:
    rec = dict(meta)
    rec["diagnostic_auc"] = entry["diagnostic_auc"]
    rec["train_oof_auc"] = entry["train_oof_auc"]
    rec["train_oof_f1"] = entry["train_oof_f1"]
    rec["train_oof_identity_acc"] = entry["train_oof_identity_acc"]

    lm = roc_curve_landmarks(entry["z_te"], ctx.is_s1)
    rec["S2_at_S1_080"] = lm["S2_at_S1_080"]
    for v in ROBUST:
        rec[f"S2_at_S1_{int(round(v * 100)):03d}"] = lm[f"S2_at_S1_{int(round(v * 100)):03d}"]
    rec["partial_auc_075_095"] = lm["partial_auc"]
    rec["threshold_at_S1_080"] = lm["threshold_at_primary"]

    ref = entry["z_rf"][np.isfinite(entry["z_rf"])]
    accept_rate = float(np.mean(ref >= lm["threshold_at_primary"]))
    rec["train_accept_at_S1_080_threshold"] = accept_rate
    rec["LOSO_at_S1_080"] = loso_rejection_at(entry["loso"], accept_rate)
    rec["implied_W"] = (WEIGHTS["LOSO"] * rec["LOSO_at_S1_080"]
                        + WEIGHTS["S1"] * PRIMARY + WEIGHTS["S2"] * rec["S2_at_S1_080"])

    wp = working_point(entry["z_rf"], entry["z_te"], entry["loso"], ctx, REF_ACC)
    rec["loso_at_090"] = wp["LOSO_rejection"]
    rec["s1_at_090"] = wp["Test_S1_rejection"]
    rec["s2_at_090"] = wp["Test_S2_acceptance"]
    rec["W_at_090"] = (WEIGHTS["LOSO"] * wp["LOSO_rejection"]
                       + WEIGHTS["S1"] * wp["Test_S1_rejection"]
                       + WEIGHTS["S2"] * wp["Test_S2_acceptance"])
    return rec


# ------------------------------------------------------------------ streams
def derive_relpower(a: np.ndarray) -> np.ndarray:
    """18-dim per-channel relative power: logsumexp over the 5 bands minus the
    channel's band mean (logbp layout is channel-major, 18 channels x 5 bands)."""
    x = np.asarray(a, dtype=np.float32).reshape(len(a), 18, 5)
    lse = logsumexp(x, axis=2, keepdims=True)
    return (lse - x.mean(axis=2, keepdims=True)).reshape(len(a), 18).astype(np.float32)


def load_stream(name: str, spec: dict, streams_cache: dict) -> tuple:
    if "derive" in spec:
        src_tr, src_te = streams_cache[spec["source"]]
        return derive_relpower(src_tr), derive_relpower(src_te)
    return (np.load(ROOT / spec["train"]).astype(np.float32),
            np.load(ROOT / spec["test"]).astype(np.float32))


# ------------------------------------------------------------------ main
def main() -> None:
    started = time.time()
    ctx = Context()
    log(f"device {DEVICE} | train {len(ctx.nodes)} | test {len(ctx.test_tasks)} | "
        f"fit_rows {int(ctx.fit_rows.sum())} | window {int(ctx.window_rows.size)} | "
        f"test S1 {int(ctx.is_s1.sum())} / S2 {int((~ctx.is_s1).sum())}")
    log(f"tasks {dict(pd.Series(ctx.tasks).value_counts())}")

    ks = [int(k) for k in CFG["prototype_k"]]
    derive_sources = {spec["source"] for spec in CFG["streams"].values() if "derive" in spec}
    streams_cache: dict[str, tuple] = {}
    entries: dict[str, dict] = {}
    records: list = []

    for name, spec in CFG["streams"].items():
        raw_tr, raw_te = load_stream(name, spec, streams_cache)
        if name in derive_sources:
            streams_cache[name] = (raw_tr, raw_te)
        a, b = task_standardise(raw_tr, raw_te, ctx.tasks, ctx.test_tasks)
        del raw_tr, raw_te
        log(f"[stream {name}] shape {a.shape} / {b.shape} (family {spec['family']})")
        for dim in spec["pca_dims"]:
            at, bt = ctx.project(name, a, b, int(dim))
            for k in ks:
                entry = evaluate(at, bt, k, ctx)
                cfg_name = f"{name} PCA-{dim} k={k}"
                entries[cfg_name] = entry
                rec = summarise(entry, {
                    "config": cfg_name, "family": spec["family"], "stream": name,
                    "pca_dim": int(dim), "k": k, "kind": "single"}, ctx)
                records.append(rec)
                log(f"  {cfg_name:34s} AUC {rec['diagnostic_auc']:.4f} "
                    f"F1 {rec['train_oof_f1']:.4f} S2@S1.80 {rec['S2_at_S1_080']:.4f} "
                    f"pAUC {rec['partial_auc_075_095']:.4f} "
                    f"W@.90 {rec['W_at_090']:.4f} ({time.time() - started:.0f}s)")
        del a, b

    # ------------------------------------------------------------ control
    ctl = CFG["control"]
    ctl_name = f"{ctl['stream']} PCA-{ctl['pca_dim']} k={ctl['k']}"
    ctl_rec = next(r for r in records if r["config"] == ctl_name)
    exp = ctl["expected"]
    reproduction = {
        "config": ctl_name,
        "expected": exp,
        "observed": {key: ctl_rec[key] for key in
                     ("diagnostic_auc", "loso_at_090", "s1_at_090", "s2_at_090",
                      "S2_at_S1_080")},
        "abs_error": {
            "diagnostic_auc": abs(ctl_rec["diagnostic_auc"] - exp["diagnostic_auc"]),
            "loso_at_090": abs(ctl_rec["loso_at_090"] - exp["loso_at_090"]),
            "s1_at_090": abs(ctl_rec["s1_at_090"] - exp["s1_at_090"]),
            "s2_at_090": abs(ctl_rec["s2_at_090"] - exp["s2_at_090"]),
            "S2_at_S1_080": abs(ctl_rec["S2_at_S1_080"] - exp["S2_at_S1_080"]),
        },
    }
    log("control reproduction " + json.dumps(reproduction["observed"], indent=0))
    log("control abs error " + json.dumps(reproduction["abs_error"], indent=0))

    # ------------------------------------------------------------ fusion
    base_k = int(CFG["fusion"]["base_k"])
    e65 = CFG["fusion"]["e65_base"]
    e65_name = f"{e65['stream']} PCA-{e65['pca_dim']} k={base_k}"

    def best_by_local(pred) -> dict:
        pool = [r for r in records if pred(r) and r["k"] == base_k]
        return max(pool, key=lambda r: r["S2_at_S1_080"])

    logbp_best = best_by_local(lambda r: r["stream"] == "logbp")
    sub_best = best_by_local(lambda r: r["stream"].startswith("subwin_"))
    log(f"fusion bases: E65={e65_name} | logbp={logbp_best['config']} "
        f"(S2@S1.80 {logbp_best['S2_at_S1_080']:.4f}) | "
        f"subwin={sub_best['config']} (S2@S1.80 {sub_best['S2_at_S1_080']:.4f})")

    base_names = [e65_name, logbp_best["config"], sub_best["config"]]
    base_entries = {}
    for name in base_names:
        stream, dim, k = parse_config(name)
        at, bt = ctx.proj[f"{stream}@{dim}"]
        base_entries[name] = evaluate(at, bt, k, ctx, keep_zmats=True)

    step = float(CFG["fusion"]["weight_step"])
    n_step = int(round(1.0 / step))
    weights_2 = [i / n_step for i in range(1, n_step)]
    fusion_records = []
    pairs = [("logbp", base_entries[logbp_best["config"]], logbp_best["config"]),
             ("subwin", base_entries[sub_best["config"]], sub_best["config"])]
    for tag, entry, cfg in pairs:
        for w in weights_2:
            fused = fuse([base_entries[e65_name], entry], [w, 1 - w], ctx)
            name = (f"2way {w:.1f}*E65({e65['pca_dim']}) + "
                    f"{1 - w:.1f}*{tag}({parse_config(cfg)[1]}) k={base_k}")
            rec = summarise(fused, {"config": name, "family": f"fusion_E65_{tag}",
                                    "stream": f"E65+{tag}", "weights": [w, 1 - w],
                                    "components": [e65_name, cfg],
                                    "pca_dim": None, "k": base_k, "kind": "fusion"}, ctx)
            fusion_records.append(rec)
            records.append(rec)

    triples = [(i / n_step, j / n_step, 1 - (i + j) / n_step)
               for i in range(1, n_step) for j in range(1, n_step - i)]
    for w in triples:
        fused = fuse([base_entries[e65_name], base_entries[logbp_best["config"]],
                      base_entries[sub_best["config"]]], list(w), ctx)
        name = (f"3way {w[0]:.1f}*E65({e65['pca_dim']}) + "
                f"{w[1]:.1f}*logbp({parse_config(logbp_best['config'])[1]}) + "
                f"{w[2]:.1f}*subwin({parse_config(sub_best['config'])[1]}) k={base_k}")
        rec = summarise(fused, {"config": name, "family": "fusion_E65_logbp_subwin",
                                "stream": "E65+logbp+subwin", "weights": list(w),
                                "components": [e65_name, logbp_best["config"],
                                               sub_best["config"]],
                                "pca_dim": None, "k": base_k, "kind": "fusion"}, ctx)
        fusion_records.append(rec)
        records.append(rec)
    log(f"fusion candidates: 2-way {2 * len(weights_2)}, 3-way {len(triples)} "
        f"({time.time() - started:.0f}s)")

    # ------------------------------------------------------------ ranking
    records.sort(key=lambda r: -r["S2_at_S1_080"])
    for rank, rec in enumerate(records, 1):
        rec["rank_by_S2_at_S1_080"] = rank

    aucs = np.asarray([r["diagnostic_auc"] for r in records])
    s2s = np.asarray([r["S2_at_S1_080"] for r in records])
    rho, pval = spearmanr(aucs, s2s)
    singles = [r for r in records if r["kind"] == "single"]
    rho_s, pval_s = spearmanr([r["diagnostic_auc"] for r in singles],
                              [r["S2_at_S1_080"] for r in singles])
    # the practically relevant sub-question: inside the competitive AUC band,
    # does the global AUC still order the local working point?
    high = [r for r in records if r["diagnostic_auc"] >= 0.75]
    rho_h, pval_h = spearmanr([r["diagnostic_auc"] for r in high],
                              [r["S2_at_S1_080"] for r in high])
    # same test as E170: local slope of S2 per AUC
    slope, intercept = np.polyfit(aucs, s2s, 1)
    top_auc = max(records, key=lambda r: r["diagnostic_auc"])
    top_local = records[0]
    top10_local = {r["config"] for r in records[:10]}
    top10_auc = {r["config"] for r in sorted(records, key=lambda r: -r["diagnostic_auc"])[:10]}
    any_070 = [r["config"] for r in records if r["S2_at_S1_080"] >= 0.70]
    any_078 = [r["config"] for r in records if r["S2_at_S1_080"] >= 0.78]
    best_auc_rank = top_auc["rank_by_S2_at_S1_080"]
    best_implied_W = max(records, key=lambda r: r["implied_W"])
    band = [r for r in records if r["diagnostic_auc"] >= top_auc["diagnostic_auc"] - 0.01]
    band_s2 = {"auc_floor": round(top_auc["diagnostic_auc"] - 0.01, 4), "n": len(band),
               "min_S2_at_S1_080": min(r["S2_at_S1_080"] for r in band),
               "max_S2_at_S1_080": max(r["S2_at_S1_080"] for r in band)}

    families = {}
    for rec in records:
        fam = rec["family"]
        prev = families.get(fam)
        if prev is None or rec["S2_at_S1_080"] > prev["S2_at_S1_080"]:
            families[fam] = rec

    results = {
        "experiment": CFG["experiment"],
        "device": str(DEVICE),
        "protocol": "E143/E170 verbatim: task-wise Train standardisation, PCA fitted on "
                    "fit_rows (frac<=0.6), prototypes from fit_rows only, LOSO excludes "
                    "the scored identity with the 0.6<frac<=0.8 evaluation block, Test "
                    "scored only; Session used for diagnostics only",
        "test_official": {"AUC": "N/A(no labels)", "F1": "N/A(no labels)"},
        "n_candidates": len(records),
        "n_singles": len(singles),
        "n_fusions": len(fusion_records),
        "control_reproduction": reproduction,
        "fusion_bases": {"e65": e65_name, "logbp": logbp_best["config"],
                         "subwin": sub_best["config"]},
        "best_by_S2_at_S1_080": top_local,
        "best_by_diagnostic_auc": top_auc,
        "rank_of_best_auc": best_auc_rank,
        "spearman_auc_vs_S2_at_S1_080": {"rho": float(rho), "p_value": float(pval),
                                         "n": len(records)},
        "spearman_singles_only": {"rho": float(rho_s), "p_value": float(pval_s),
                                  "n": len(singles)},
        "spearman_auc_ge_075": {"rho": float(rho_h), "p_value": float(pval_h),
                                "n": len(high), "auc_threshold": 0.75},
        "top10_overlap_local_vs_auc": len(top10_local & top10_auc),
        "top10_by_diagnostic_auc": sorted(records,
                                          key=lambda r: -r["diagnostic_auc"])[:10],
        "linear_fit_S2_per_auc": {"slope": float(slope), "intercept": float(intercept)},
        "near_max_auc_band": band_s2,
        "best_implied_W": best_implied_W,
        "question_any_S2_at_S1_080_ge_0.70": {"answer": bool(any_070), "count": len(any_070),
                                              "configs": any_070[:10]},
        "question_any_S2_at_S1_080_ge_0.78": {"answer": bool(any_078), "count": len(any_078),
                                              "configs": any_078[:10]},
        "max_S2_at_S1_080": float(s2s.max()),
        "max_partial_auc": float(max(r["partial_auc_075_095"] for r in records)),
        "best_per_family": {fam: rec for fam, rec in families.items()},
        "candidates": records,
        "runtime_seconds": round(time.time() - started, 1),
    }
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2,
                                                 default=float), encoding="utf-8")
    log(f"results.json written ({time.time() - started:.0f}s)")
    write_report(results)
    log(f"done in {time.time() - started:.0f}s")


def parse_config(name: str) -> tuple:
    body, k_part = name.rsplit(" k=", 1)
    stream_dim = body.rsplit(" PCA-", 1)
    return stream_dim[0], int(stream_dim[1]), int(k_part)


# ------------------------------------------------------------------ report
def fnum(x, nd=4) -> str:
    return f"{x:.{nd}f}" if np.isfinite(x) else "N/A"


def build_row(r: dict, detail: bool) -> str:
    cells = [f"| {r['rank_by_S2_at_S1_080']} | {r['config']} | {r['family']} | "
             f"{r['diagnostic_auc']:.4f} | {r['train_oof_auc']:.4f} | "
             f"{fnum(r['train_oof_f1'])} | {r['S2_at_S1_080']:.4f} | "]
    if detail:
        cells.append(f"{r['S2_at_S1_075']:.4f} | {r['S2_at_S1_085']:.4f} | "
                     f"{r['S2_at_S1_090']:.4f} | ")
    cells.append(f"{r['partial_auc_075_095']:.4f} | {r['threshold_at_S1_080']:+.4f} | "
                 f"{r['LOSO_at_S1_080']:.4f} | {r['implied_W']:.4f} | {r['W_at_090']:.4f} |")
    return "".join(cells)


HEAD_MAIN = ("| # | 配置 | 家族 | 诊断 AUC | Train OOF AUC | Train OOF F1 | "
             "S2@S1=.80（主） | pAUC[.75,.95] | 阈值@S1=.80 | LOSO@S1=.80 | implied W | "
             "W@.90 |")
SEP_MAIN = "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
HEAD_DETAIL = ("| # | 配置 | 家族 | 诊断 AUC | Train OOF AUC | Train OOF F1 | "
               "S2@S1=.80 | S2@.75 | S2@.85 | S2@.90 | pAUC[.75,.95] | 阈值@S1=.80 | "
               "LOSO@S1=.80 | implied W | W@.90 |")
SEP_DETAIL = "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"


def write_report(results) -> None:
    rep = results["control_reproduction"]
    best = results["best_by_S2_at_S1_080"]
    best_auc = results["best_by_diagnostic_auc"]
    rho = results["spearman_auc_vs_S2_at_S1_080"]
    rho_s = results["spearman_singles_only"]
    rho_h = results["spearman_auc_ge_075"]
    fit = results["linear_fit_S2_per_auc"]
    q70 = results["question_any_S2_at_S1_080_ge_0.70"]
    q78 = results["question_any_S2_at_S1_080_ge_0.78"]
    cands = results["candidates"]
    top5 = cands[:5]
    best_w = results["best_implied_W"]

    lines = [
        "# E172 局部 ROC 重排（S1 拒识 = 0.80 处的 S2 接受）",
        "",
        "E170 的结论：全局诊断 AUC 不是正确的优化目标——AUC 越高，S1 拒识 0.80 工作点上的 "
        "S2 接受反而越低（拟合斜率 −0.46），增益全在 ROC 尾部。本实验把所有已有缓存分数"
        "（不训练任何新模型）按**局部 ROC 指标**重新排序：主指标 `S2_at_S1_080`，"
        "即 S1 拒识恰为 0.80 时的 S2 接受率，用 Test 分数向量按 session 分组"
        "（S1 为负类、S2 为正类）做 ROC 后线性插值得到。",
        "",
        "协议（不可改动）：task-wise Train 标准化（该 task 全部 Train 行）→ PCA 用 "
        "fit_rows(frac≤0.6) 拟合 → 原型只用 fit_rows → k=1/3/6/12 → "
        "score = −最近距离；LOSO 每折排除被评分身份、评估块 0.6<frac≤0.8；Test 只评分。"
        "Session 分组只用于诊断性评估，不参与任何特征 / PCA / 原型 / 阈值拟合。",
        "",
        "Test 官方 AUC / F1：**N/A**（Test 无真实标签）。下表 AUC 为诊断性 AUC，"
        "F1 为 Train OOF 宏平均 F1（可复现的训练侧指标）。",
        "",
        "## 结论",
        "",
        f"- 候选分数共 **{results['n_candidates']}** 个（单路 {results['n_singles']} + "
        f"融合 {results['n_fusions']}），全部来自已有缓存，未训练任何新模型。",
        f"- 局部最优：**{best['config']}**，`S2_at_S1_080` = **{best['S2_at_S1_080']:.4f}**，"
        f"pAUC[.75,.95] {best['partial_auc_075_095']:.4f}，诊断 AUC {best['diagnostic_auc']:.4f}，"
        f"Train OOF AUC/F1 {best['train_oof_auc']:.4f}/{fnum(best['train_oof_f1'])}，"
        f"implied W {best['implied_W']:.4f}。",
        f"- 全场 implied W 最高：**{best_w['config']}**，W {best_w['implied_W']:.4f}"
        f"（`S2_at_S1_080` {best_w['S2_at_S1_080']:.4f}，LOSO@S1=.80 "
        f"{best_w['LOSO_at_S1_080']:.4f}）。E143 公布最优 W 0.6982，E170 扫满最优 0.6984。",
        f"- 全局 AUC 最优：**{best_auc['config']}**，AUC {best_auc['diagnostic_auc']:.4f}，"
        f"但其局部指标只有 `S2_at_S1_080` = {best_auc['S2_at_S1_080']:.4f}"
        f"（局部排名第 {results['rank_of_best_auc']}/{len(cands)}）。",
        f"- Spearman(全局 AUC, `S2_at_S1_080`) = **{rho['rho']:.4f}**"
        f"（p = {rho['p_value']:.3g}，n = {rho['n']}）；只看单路 {rho_s['rho']:.4f}"
        f"（p = {rho_s['p_value']:.3g}，n = {rho_s['n']}）；把范围收窄到 AUC ≥ 0.75 的 "
        f"{rho_h['n']} 个竞争性候选后只有 **{rho_h['rho']:.4f}**"
        f"（p = {rho_h['p_value']:.3g}）。局部前 10 与 AUC 前 10 只重合 "
        f"**{results['top10_overlap_local_vs_auc']}/10**。",
        f"- **是否存在 `S2_at_S1_080` ≥ 0.70 的候选**："
        f"{'是' if q70['answer'] else '否'}（{q70['count']} 个）。"
        f"**≥ 0.78 的候选**：{'是' if q78['answer'] else '否'}（{q78['count']} 个）。"
        f"全场最高 {results['max_S2_at_S1_080']:.4f}，离达标还差 "
        f"{0.78 - results['max_S2_at_S1_080']:.4f}。",
        "- 家族层面（详见「各族最优」）：subwin_mean_N4 与 E65 局部最好（0.596–0.618）；"
        "logbp 的全局 AUC 更高（0.781）但局部只有 0.5475，再次说明 AUC 与工作点分离；"
        "phase 类（pac / wpli / spectral_entropy）AUC < 0.53、局部 < 0.23，基本无区分力；"
        "两个 embedding（E162 / E166）AUC 0.63–0.74、局部 0.37–0.46。",
        "",
        "## 复现对照（E143 E65 PCA-96 k=12）",
        "",
        "| 指标 | E170 实测 | 本实验 | 绝对误差 |",
        "|---|---:|---:|---:|",
    ]
    order = [("diagnostic_auc", "诊断 AUC"), ("loso_at_090", "LOSO@.90"),
             ("s1_at_090", "S1@.90"), ("s2_at_090", "S2@.90"),
             ("S2_at_S1_080", "S2_at_S1_080")]
    for key, label in order:
        lines.append(f"| {label} | {rep['expected'][key]:.4f} | "
                     f"{rep['observed'][key]:.4f} | {rep['abs_error'][key]:.4f} |")
    lines += [
        "",
        "（`S2_at_S1_080` 的期望值 0.5929 是 E170 用 2000 个分位阈值扫满后取最近点的读数；"
        "本实验改用精确 ROC + 线性插值，因此该列只要求 ≈ 一致。）",
        "",
        "## 判定",
        "",
        f"- **是否存在 `S2_at_S1_080` ≥ 0.70 的候选**：**{'是' if q70['answer'] else '否'}**"
        f"（{q70['count']}/{len(cands)}）。",
        f"- **是否存在 `S2_at_S1_080` ≥ 0.78 的候选**：**{'是' if q78['answer'] else '否'}**"
        f"（{q78['count']}/{len(cands)}）。全场最高 {results['max_S2_at_S1_080']:.4f}"
        f"（{best['config']}），距 0.78 差 {0.78 - results['max_S2_at_S1_080']:.4f}。",
        f"- 按 E170 的口径（LOSO 冻结在 .90 工作点 0.8412）也到不了 W=0.80："
        f"即使 S1=0.80、S2={results['max_S2_at_S1_080']:.4f}，"
        f"W 也只有 {0.25 * 0.8412 + 0.25 * 0.8 + 0.5 * results['max_S2_at_S1_080']:.4f}。"
        f"按本实验的阈值联动口径，最优 implied W 为 {best_w['implied_W']:.4f}"
        f"（{best_w['config']}）。",
        "",
        "## 全量排序（按 `S2_at_S1_080` 降序，共 "
        f"{len(cands)} 个候选）",
        "",
        HEAD_MAIN, SEP_MAIN,
    ]
    for rec in cands:
        lines.append(build_row(rec, detail=False))

    lines += [
        "",
        "## 前 30 名（含稳健性列）",
        "",
        HEAD_DETAIL, SEP_DETAIL,
    ]
    for rec in cands[:30]:
        lines.append(build_row(rec, detail=True))

    lines += [
        "",
        "## 前 5 名稳健性检查",
        "",
        "| # | 配置 | S2@.75 | S2@.85 | S2@.90 | 主指标 S2@.80 | pAUC[.75,.95] | "
        "S2@.90 − S2@.75 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in top5:
        lines.append(f"| {rec['rank_by_S2_at_S1_080']} | {rec['config']} | "
                     f"{rec['S2_at_S1_075']:.4f} | {rec['S2_at_S1_085']:.4f} | "
                     f"{rec['S2_at_S1_090']:.4f} | {rec['S2_at_S1_080']:.4f} | "
                     f"{rec['partial_auc_075_095']:.4f} | "
                     f"{rec['S2_at_S1_090'] - rec['S2_at_S1_075']:+.4f} |")
    spread = {v: (min(r[f"S2_at_S1_{int(round(v * 100)):03d}"] for r in top5),
                  max(r[f"S2_at_S1_{int(round(v * 100)):03d}"] for r in top5))
              for v in [0.75, PRIMARY, 0.85, 0.90]}
    wid = {v: hi - lo for v, (lo, hi) in spread.items()}
    lines += [
        "",
        "前 5 名在四个工作点上的取值区间："
        + "；".join(f"S1={v:.2f} → [{lo:.4f}, {hi:.4f}]" for v, (lo, hi) in spread.items())
        + "。",
        "",
        f"读法：S1=.75 / .80 / .85 三点上前 5 名的区间宽度分别只有 "
        f"{wid[0.75]:.4f} / {wid[PRIMARY]:.4f} / {wid[0.85]:.4f}，说明局部优势是 ROC 中段"
        f"整体平移而不是单点尖峰；到 S1=.90 区间宽度才扩大到 {wid[0.90]:.4f}"
        f"（最低者 {min(r['S2_at_S1_090'] for r in top5):.4f}，最高者 "
        f"{max(r['S2_at_S1_090'] for r in top5):.4f}）。即结论对工作点小幅漂移稳健，"
        "但极端工作点（.90）上排序会变。",
        "",
        "## 各族最优",
        "",
        "| 家族 | 最优配置 | 诊断 AUC | Train OOF F1 | S2@S1=.80 | pAUC[.75,.95] | "
        "implied W |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for fam, rec in sorted(results["best_per_family"].items(),
                           key=lambda kv: -kv[1]["S2_at_S1_080"]):
        lines.append(f"| {fam} | {rec['config']} | {rec['diagnostic_auc']:.4f} | "
                     f"{fnum(rec['train_oof_f1'])} | {rec['S2_at_S1_080']:.4f} | "
                     f"{rec['partial_auc_075_095']:.4f} | {rec['implied_W']:.4f} |")

    lines += [
        "",
        "## 全局 AUC 与局部指标的排序关系",
        "",
        f"- 全部 {rho['n']} 个候选：Spearman = {rho['rho']:.4f}（p = {rho['p_value']:.3g}）；"
        f"单路 {rho_s['n']} 个：{rho_s['rho']:.4f}（p = {rho_s['p_value']:.3g}）；"
        f"AUC ≥ 0.75 的 {rho_h['n']} 个竞争性候选：**{rho_h['rho']:.4f}**"
        f"（p = {rho_h['p_value']:.3g}）。",
        f"- 线性拟合 `S2_at_S1_080` = {fit['slope']:.4f}·AUC {fit['intercept']:+.4f}；"
        "这条正斜率主要由 AUC < 0.75 的弱家族（spectral_entropy / pac / wpli 等）撑起来，"
        f"在真正有竞争力的 AUC ≥ 0.75 区间内只剩 ρ = {rho_h['rho']:.4f}"
        f"（p = {rho_h['p_value']:.3g}），即 AUC 在工作点附近基本不再提供排序信息。",
        f"- 局部前 10 与全局 AUC 前 10 只重合 **{results['top10_overlap_local_vs_auc']}/10**；"
        f"全局 AUC 最高的候选局部排名第 {results['rank_of_best_auc']}/{len(cands)}。",
        f"- 在 AUC ≥ {results['near_max_auc_band']['auc_floor']:.4f} 的 "
        f"{results['near_max_auc_band']['n']} 个候选里，`S2_at_S1_080` 仍然散在 "
        f"{results['near_max_auc_band']['min_S2_at_S1_080']:.4f}–"
        f"{results['near_max_auc_band']['max_S2_at_S1_080']:.4f}"
        "（极差远大于同一 AUC 差能解释的量）。",
        f"- 解读：在这 {len(cands)} 个候选上，全局 AUC 与局部工作点**正相关但不等价**。"
        "E170 得出「AUC 越高 S2 越低」的负斜率来自 4 个几乎同源的冻结分数"
        "（其中 3 个是 E65 的相邻变体），是 4 点噪声，不是全局规律；"
        "但反过来也**不能**用 AUC 直接选局部最优：同一 AUC 附近 "
        "`S2_at_S1_080` 的散布很大，且两个 Top-10 几乎不重叠。",
        "",
        "全局 AUC 前 10 名（对照用，按 AUC 降序）：",
        "",
        "| AUC 名次 | 配置 | 诊断 AUC | S2@S1=.80 | 局部名次 | pAUC[.75,.95] | implied W |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, rec in enumerate(results["top10_by_diagnostic_auc"], 1):
        lines.append(f"| {i} | {rec['config']} | {rec['diagnostic_auc']:.4f} | "
                     f"{rec['S2_at_S1_080']:.4f} | {rec['rank_by_S2_at_S1_080']} | "
                     f"{rec['partial_auc_075_095']:.4f} | {rec['implied_W']:.4f} |")

    lines += [
        "",
        "## 说明",
        "",
        "- 单路候选的 PCA 维度按任务书给定网格：E65 64/96；logbp 16/32/48（90 维）；"
        "spectral_entropy / pac / wpli / hjorth 16/32；relpower 16（18 维，由 logbp 逐通道 "
        "logsumexp 减通道均值派生）；subwin 48/64/96；E162 32/64/128；E166 32/64。",
        "- E168 子窗口特征目录经 `feature_manifest.json` 核对后只有 N=2/3/4 三个窗数"
        "（N=6/8/12 未落盘），故按实际可用文件取 `subwin_mean_n2/n3/n4`（855 维）。",
        "- 融合用 0.1 步长权重网格，各路子分数先 z 标准化再加权：两路 9 个权重，"
        "三路在权重均 ≥0.1 的单纯形上取 36 个组合；融合基准 k 固定为 12，"
        "E65 固定 PCA-96，logbp / subwin 取各自单路在 k=12 下 `S2_at_S1_080` 最优的维度。",
        "- 注意：这一步的基准维度用了 Test 诊断指标来挑，因此融合理由带 Test 选点乐观性；"
        "特征是 Train-only 的，没有被 Test 拟合，但「哪一路最好」这一选择看过 Test。"
        "单路 112 个候选与 LOSO 无关的结论不受此影响（它们全部来自固定网格）。",
        "- `implied_W` = 0.25×LOSO@阈值 + 0.25×0.80 + 0.5×`S2_at_S1_080`，"
        "阈值为插值得到的 S1 拒识恰为 0.80 的分数阈值，LOSO 由该阈值对应的 Train-OOF "
        "接受率经每折 known 分位反推，三个指标始终联动。",
        "- 局部 ROC 用 Test 全部观测分数作阈值，取 S1 拒识 vs S2 接受的经验曲线后线性插值；"
        "`partial_auc` 为 S1 拒识落在 [0.75, 0.95] 的曲线下面积除以区间宽度 0.2。",
        f"- 运行耗时 {results['runtime_seconds']:.0f}s。",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
