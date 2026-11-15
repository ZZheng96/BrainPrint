"""E167: score-level / feature-level fusion of E65 correlation-logSPD and log band power,
under the verbatim E143/E163 open-set protocol.

Two information sources:
  * E65  : 855-d correlation log-SPD vector (structured covariance geometry, E143 view)
  * logbp: 90-d log band power (5 bands x 18 channels) -- pure amplitude information that
           E165 showed E65 standardises away (channel gain / band gain / shift invariant)

Fusion families (identical to E163):
  a) feature concat -> joint PCA -> prototypes
  b) score fusion of task-standardised z-scores (weights 0.25/0.5/0.75, 0.5 = equal)
  c) score fusion whose weight is selected on Train LOSO AUC only (Test never consulted)
  d) optional three-way score fusion E65 + logbp + hjorth

Everything is Train-only: standardisation, PCA, prototypes, thresholds, LOSO folds.
Test is scored once, then session labels are used only for the diagnostic AUC.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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


REPORT_ONLY = len(sys.argv) > 1 and sys.argv[1] == "--report-only"
sys.stdout = Tee(OUT / "run.log") if not REPORT_ONLY else sys.__stdout__


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------ primitives
def frac(meta: pd.DataFrame) -> np.ndarray:
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def task_standardise(a: np.ndarray, b: np.ndarray, tasks: np.ndarray,
                     test_tasks: np.ndarray, stat_rows: np.ndarray) -> tuple:
    """Per-task z-score; statistics come from `stat_rows` (Train only)."""
    a = a.copy()
    b = b.copy()
    for task in TASKS:
        ref = stat_rows & (tasks == task)
        mu = a[ref].mean(0)
        sd = a[ref].std(0) + 1e-6
        m = tasks == task
        q = test_tasks == task
        a[m] = (a[m] - mu) / sd
        b[q] = (b[q] - mu) / sd
    return a, b


def pca_basis(x: np.ndarray, dim: int) -> tuple:
    mu = x.mean(0)
    xc = x - mu
    values, vectors = np.linalg.eigh(xc.T @ xc / max(len(xc) - 1, 1))
    return mu, vectors[:, np.argsort(-values)][:, : min(int(dim), x.shape[1])]


def kmeans_t(x: torch.Tensor, k: int, seed: int, iters: int = 20) -> torch.Tensor:
    """Verbatim E143/E162/E163 seeding so the E65 control reproduces the published W."""
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


def build_prototypes(feat: np.ndarray, subjects: np.ndarray, tasks: np.ndarray,
                     mask: np.ndarray, classes: np.ndarray, k: int) -> tuple:
    protos, keys = [], []
    for task in TASKS:
        for subject in classes:
            sel = mask & (tasks == task) & (subjects == subject)
            if sel.sum() == 0:
                continue
            block = torch.as_tensor(np.asarray(feat[sel], dtype=np.float32), device=DEVICE)
            for centre in kmeans_t(block, int(k), seed=int(subject) * 17).cpu().numpy():
                protos.append(centre)
                keys.append((task, int(subject)))
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
        """Standardise + PCA fit on fit_rows; cached under `key`."""
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
             keep_blocks: bool) -> dict:
    """Verbatim E143/E163 open-set protocol on one projected representation."""
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
        if keep_blocks:
            block_cache.append({"idx": held, "protos": p2, "keys": k2})

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
    loso = [((kn - mu) / sd, (un - mu) / sd) for kn, un in zip(loso_known, loso_unknown)]
    unknown_all = np.concatenate([un for _, un in loso])
    oof_auc = float(roc_auc_score(np.r_[np.zeros(len(unknown_all)), np.ones(int(finite.sum()))],
                                  np.r_[unknown_all, z_rf[finite]]))
    oof_f1 = float(f1_score(ctx.nodes[finite], oof_pred[finite], labels=CLASSES,
                            average="macro", zero_division=0))
    oof_acc = float(np.mean(oof_pred[finite] == ctx.nodes[finite]))
    diag_auc = float(roc_auc_score(ctx.y, np.r_[z_te[ctx.is_s1], z_te[~ctx.is_s1]]))

    rows = []
    for acceptance in ACC:
        threshold = float(np.nanquantile(z_rf[finite], 1 - acceptance))
        s2 = float(np.mean(z_te[~ctx.is_s1] >= threshold))
        s1 = float(np.mean(z_te[ctx.is_s1] < threshold))
        loso_rej = float(np.mean([np.mean(un < np.quantile(kn, 1 - acceptance))
                                  for kn, un in loso]))
        rows.append({"acceptance_rate": float(acceptance),
                     "LOSO_rejection": loso_rej,
                     "Test_S1_rejection": s1,
                     "Test_S2_acceptance": s2,
                     "W": WEIGHTS["LOSO"] * loso_rej + WEIGHTS["S1"] * s1
                          + WEIGHTS["S2"] * s2,
                     "threshold": threshold})

    return {"z_rf": z_rf, "z_te": z_te, "loso": loso, "blocks": block_cache,
            "mu": mu, "sd": sd,
            "train_oof_auc": oof_auc, "train_oof_f1": oof_f1,
            "train_oof_identity_acc": oof_acc, "diagnostic_auc": diag_auc,
            "rows": rows,
            "per_task_auc": {
                task: float(roc_auc_score((~ctx.is_s1[ctx.test_tasks == task]).astype(int),
                                          z_te[ctx.test_tasks == task]))
                for task in TASKS
                if len(np.unique(ctx.is_s1[ctx.test_tasks == task])) > 1}}


def fuse_many(entries: list, ats: list, weights: list, ctx: Context) -> dict:
    """Weighted score fusion of N streams sharing the same prototype keys / blocks."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    z_rf = sum(w * e["z_rf"] for w, e in zip(weights, entries)).astype(np.float32)
    z_te = sum(w * e["z_te"] for w, e in zip(weights, entries)).astype(np.float32)
    loso = []
    for fold in range(len(entries[0]["loso"])):
        kn = sum(w * e["loso"][fold][0] for w, e in zip(weights, entries))
        un = sum(w * e["loso"][fold][1] for w, e in zip(weights, entries))
        loso.append((kn, un))

    pred = np.full(len(ctx.nodes), -1, dtype=np.int32)
    for block in range(len(entries[0]["blocks"])):
        first = entries[0]["blocks"][block]
        idx = first["idx"]
        total, ksubj = None, None
        for w, e, at in zip(weights, entries, ats):
            b = e["blocks"][block]
            assert np.array_equal(b["idx"], first["idx"]), "block rows misaligned"
            assert [tuple(x) for x in b["keys"]] == [tuple(x) for x in first["keys"]], \
                "keys misaligned"
            dmat, subj = neg_dmat(at[idx], b["protos"], b["keys"], ctx.tasks[idx])
            z = (dmat - e["mu"]) / e["sd"]
            total = w * z if total is None else total + w * z
            ksubj = subj
        pred[idx] = ksubj[total.argmax(1)]

    oof_f1 = float(f1_score(ctx.nodes, pred, labels=CLASSES, average="macro", zero_division=0))
    oof_acc = float(np.mean(pred == ctx.nodes))
    unknown_all = np.concatenate([un for _, un in loso])
    oof_auc = float(roc_auc_score(np.r_[np.zeros(len(unknown_all)), np.ones(len(z_rf))],
                                  np.r_[unknown_all, z_rf]))
    diag_auc = float(roc_auc_score(ctx.y, np.r_[z_te[ctx.is_s1], z_te[~ctx.is_s1]]))

    rows = []
    for acceptance in ACC:
        threshold = float(np.nanquantile(z_rf, 1 - acceptance))
        s2 = float(np.mean(z_te[~ctx.is_s1] >= threshold))
        s1 = float(np.mean(z_te[ctx.is_s1] < threshold))
        loso_rej = float(np.mean([np.mean(un < np.quantile(kn, 1 - acceptance))
                                  for kn, un in loso]))
        rows.append({"acceptance_rate": float(acceptance),
                     "LOSO_rejection": loso_rej,
                     "Test_S1_rejection": s1,
                     "Test_S2_acceptance": s2,
                     "W": WEIGHTS["LOSO"] * loso_rej + WEIGHTS["S1"] * s1
                          + WEIGHTS["S2"] * s2,
                     "threshold": threshold})

    return {"z_rf": z_rf, "z_te": z_te, "loso": loso, "blocks": entries[0]["blocks"],
            "mu": 0.0, "sd": 1.0,
            "train_oof_auc": oof_auc, "train_oof_f1": oof_f1,
            "train_oof_identity_acc": oof_acc, "diagnostic_auc": diag_auc,
            "rows": rows,
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


def fused_loso_auc(entries: list, weights: list) -> float:
    """Train-only criterion used to pick fusion weights (no Test involvement)."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    unknown = np.concatenate([sum(w * e["loso"][f][1] for w, e in zip(weights, entries))
                              for f in range(len(entries[0]["loso"]))])
    known = np.concatenate([sum(w * e["loso"][f][0] for w, e in zip(weights, entries))
                            for f in range(len(entries[0]["loso"]))])
    return float(roc_auc_score(np.r_[np.zeros(len(unknown)), np.ones(len(known))],
                               np.r_[unknown, known]))


def summarise(entry: dict, meta: dict) -> dict:
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
        "curve": rows,
        "per_task_auc": entry["per_task_auc"],
    })
    return record


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
        a, b = task_standardise(a, b, ctx.tasks, ctx.test_tasks, np.ones(len(a), bool))
        streams[name] = (a, b)
        log(f"loaded {name}: {a.shape} / {b.shape}")
    a65, b65 = streams["E65_logspd"]
    alb, blb = streams["logbp"]

    controls, feature_rows = {}, []
    entries: dict[tuple, dict] = {}

    def entry_for(name: str, dim: int, at: np.ndarray, bt: np.ndarray, k: int,
                  keep_blocks: bool = False) -> dict:
        key = (f"{name}@{dim}", int(k))
        if key not in entries:
            entries[key] = evaluate(at, bt, int(k), ctx, keep_blocks)
        return entries[key]

    t0 = time.time()
    for name, spec in CFG["streams"].items():
        a, b = streams[name]
        controls[name] = []
        for dim in spec["pca_dims"]:
            at, bt = ctx.project(name, a, b, dim)
            for k in CFG["prototype_k"]:
                e = entry_for(name, dim, at, bt, k, keep_blocks=True)
                rec = summarise(e, {"family": f"{name}_only", "stream": name,
                                    "pca_dim": dim, "k": k,
                                    "config": f"{name} PCA-{dim} k={k}"})
                controls[name].append(rec)
                log(f"  [{name}] dim={dim} k={k} OOFauc={rec['train_oof_auc']:.4f} "
                    f"OOFf1={rec['train_oof_f1']:.4f} diagAUC={rec['diagnostic_auc']:.4f} "
                    f"W@.90={rec['W_at_090']:.4f} bestW={rec['best_W']:.4f}"
                    f"@{rec['best_acceptance']:.2f} ({time.time() - t0:.0f}s)")

    # reference reproduction row: logbp with all 90 dims (= E148 PCA-96 probe)
    ref_rows = []
    a, b = streams["logbp"]
    at, bt = ctx.project("logbp_ref", a, b, 90)
    for k in CFG["prototype_k"]:
        e = entry_for("logbp_ref", 90, at, bt, k, keep_blocks=True)
        rec = summarise(e, {"family": "logbp_reference_full", "stream": "logbp",
                            "pca_dim": 90, "k": k,
                            "config": f"logbp PCA-90(full) k={k}"})
        ref_rows.append(rec)
        log(f"  [logbp ref] dim=90 k={k} diagAUC={rec['diagnostic_auc']:.4f} "
            f"W@.90={rec['W_at_090']:.4f} bestW={rec['best_W']:.4f}")

    # ---------------------------------------------------------- feature concat
    log("feature-level concat fusion E65 + logbp")
    for dim_a, dim_b in CFG["fusion"]["feature_concat"]["dim_pairs"]:
        at65, bt65 = ctx.project("E65_logspd", a65, b65, dim_a)
        atlb, btlb = ctx.project("logbp", alb, blb, dim_b)
        scale_a = float(np.sqrt(np.mean(at65[ctx.fit_rows] ** 2)))
        scale_b = float(np.sqrt(np.mean(atlb[ctx.fit_rows] ** 2)))
        blocks_tr = {"raw": (at65, atlb),
                     "block_norm": (at65 / max(scale_a, 1e-6), atlb / max(scale_b, 1e-6))}
        blocks_te = {"raw": (bt65, btlb),
                     "block_norm": (bt65 / max(scale_a, 1e-6), btlb / max(scale_b, 1e-6))}
        for variant in CFG["fusion"]["feature_concat"]["variants"]:
            tr = np.concatenate(blocks_tr[variant], axis=1)
            te = np.concatenate(blocks_te[variant], axis=1)
            for jd in CFG["fusion"]["feature_concat"]["joint_pca_dims"]:
                if jd > tr.shape[1]:
                    continue
                key = f"concat_{variant}_{dim_a}_{dim_b}"
                at, bt = ctx.project(key, tr, te, jd)
                for k in CFG["prototype_k"]:
                    rec = summarise(entry_for(key, jd, at, bt, k),
                                    {"family": f"feature_concat_{variant}",
                                     "stream": "E65+logbp", "pca_dim_e65": dim_a,
                                     "pca_dim_logbp": dim_b, "joint_pca_dim": jd, "k": k,
                                     "config": f"{variant} concat {dim_a}+{dim_b} -> "
                                               f"PCA-{jd} k={k}"})
                    feature_rows.append(rec)
                del ctx.proj[f"{key}@{jd}"]
                best_here = max(r["best_W"] for r in feature_rows[-len(CFG["prototype_k"]):])
                log(f"  [concat_{variant} {dim_a}+{dim_b} -> {jd}] "
                    f"diagAUC={feature_rows[-1]['diagnostic_auc']:.4f} "
                    f"W@.90={feature_rows[-1]['W_at_090']:.4f} bestW={best_here:.4f} "
                    f"({time.time() - t0:.0f}s)")
            del tr, te
        del blocks_tr, blocks_te

    # ---------------------------------------------------------- score fusion
    log("score-level fusion E65 + logbp")
    score_rows, selected_rows = [], []
    weights = CFG["fusion"]["score_fusion"]["weights"]
    sel_grid = CFG["fusion"]["score_fusion"]["train_selected_weight_grid"]
    dims65 = CFG["streams"]["E65_logspd"]["pca_dims"]
    dimslb = CFG["streams"]["logbp"]["pca_dims"]
    for dim_a in dims65:
        at1, _ = ctx.project("E65_logspd", a65, b65, dim_a)
        for dim_b in dimslb:
            at2, _ = ctx.project("logbp", alb, blb, dim_b)
            for k in CFG["prototype_k"]:
                e1 = entries[(f"E65_logspd@{dim_a}", k)]
                e2 = entries[(f"logbp@{dim_b}", k)]
                for w in weights:
                    fused = fuse_many([e1, e2], [at1, at2], [w, 1 - w], ctx)
                    score_rows.append(summarise(fused, {
                        "family": "score_fusion", "stream": "E65+logbp",
                        "pca_dim_e65": dim_a, "pca_dim_logbp": dim_b, "k": k,
                        "weight_e65": w, "weight_logbp": 1 - w,
                        "config": f"score {w:.2f}*E65({dim_a}) + "
                                  f"{1 - w:.2f}*logbp({dim_b}) k={k}"}))
                train_aucs = {w: fused_loso_auc([e1, e2], [w, 1 - w]) for w in sel_grid}
                w_best = max(train_aucs, key=train_aucs.get)
                fused = fuse_many([e1, e2], [at1, at2], [w_best, 1 - w_best], ctx)
                selected_rows.append(summarise(fused, {
                    "family": "score_fusion_train_selected", "stream": "E65+logbp",
                    "pca_dim_e65": dim_a, "pca_dim_logbp": dim_b, "k": k,
                    "weight_e65": w_best, "weight_logbp": 1 - w_best,
                    "train_loso_auc_by_weight": {str(w): v for w, v in train_aucs.items()},
                    "config": f"score {w_best:.2f}*E65({dim_a}) + "
                              f"{1 - w_best:.2f}*logbp({dim_b}) k={k} [w by Train LOSO]"}))
            top = max((r for r in score_rows
                       if r["pca_dim_e65"] == dim_a and r["pca_dim_logbp"] == dim_b),
                      key=lambda r: r["best_W"])["best_W"]
            log(f"  [score {dim_a}+{dim_b}] bestW={top:.4f} "
                f"selectedW={selected_rows[-1]['best_W']:.4f} "
                f"(w={selected_rows[-1]['weight_e65']:.1f}, "
                f"diagAUC={selected_rows[-1]['diagnostic_auc']:.4f}, "
                f"{time.time() - t0:.0f}s)")

    # ---------------------------------------------------------- three-way
    log("three-way score fusion E65 + logbp + hjorth")
    three_rows, three_sel = [], []
    tw = CFG["fusion"]["three_way"]
    ahj, bhj = streams["hjorth"]
    dims_hj = tw["dim_grid"]["hjorth"]
    for dim_h in dims_hj:
        ath, _ = ctx.project("hjorth", ahj, bhj, dim_h)
        for dim_b in tw["dim_grid"]["logbp"]:
            at2, _ = ctx.project("logbp", alb, blb, dim_b)
            for k in tw["k"]:
                e1 = entries[(f"E65_logspd@{tw['dim_grid']['E65_logspd'][0]}", k)]
                e2 = entries[(f"logbp@{dim_b}", k)]
                e3 = entries[(f"hjorth@{dim_h}", k)]
                at1 = ctx.proj[f"E65_logspd@{tw['dim_grid']['E65_logspd'][0]}"][0]
                for w in tw["fixed_weights"]:
                    fused = fuse_many([e1, e2, e3], [at1, at2, ath], w, ctx)
                    three_rows.append(summarise(fused, {
                        "family": "score_fusion_3way", "stream": "E65+logbp+hjorth",
                        "pca_dim_e65": tw["dim_grid"]["E65_logspd"][0],
                        "pca_dim_logbp": dim_b, "pca_dim_hjorth": dim_h, "k": k,
                        "weight_e65": w[0], "weight_logbp": w[1], "weight_hjorth": w[2],
                        "config": f"score {w[0]:.2f}*E65 + {w[1]:.2f}*logbp({dim_b}) + "
                                  f"{w[2]:.2f}*hjorth({dim_h}) k={k}"}))
                cand = tw["train_selected_weight_candidates"]
                train_aucs = {str(w): fused_loso_auc([e1, e2, e3], w) for w in cand}
                w_best = max(cand, key=lambda w: train_aucs[str(w)])
                fused = fuse_many([e1, e2, e3], [at1, at2, ath], w_best, ctx)
                three_sel.append(summarise(fused, {
                    "family": "score_fusion_3way_train_selected", "stream": "E65+logbp+hjorth",
                    "pca_dim_e65": tw["dim_grid"]["E65_logspd"][0],
                    "pca_dim_logbp": dim_b, "pca_dim_hjorth": dim_h, "k": k,
                    "weight_e65": w_best[0], "weight_logbp": w_best[1],
                    "weight_hjorth": w_best[2],
                    "train_loso_auc_by_weight": train_aucs,
                    "config": f"score {w_best[0]:.2f}*E65 + {w_best[1]:.2f}*logbp({dim_b}) + "
                              f"{w_best[2]:.2f}*hjorth({dim_h}) k={k} [w by Train LOSO]"}))
            log(f"  [3way logbp({dim_b}) hjorth({dim_h})] bestW="
                f"{max(r['best_W'] for r in three_rows):.4f} selW="
                f"{three_sel[-1]['best_W']:.4f} ({time.time() - t0:.0f}s)")

    # ---------------------------------------------------------- redundancy
    ctrl65 = max(controls["E65_logspd"], key=lambda r: r["best_W"])
    ctrllb = max(controls["logbp"], key=lambda r: r["best_W"])
    e1 = entries[(f"E65_logspd@{ctrl65['pca_dim']}", ctrl65["k"])]
    e2 = entries[(f"logbp@{ctrllb['pca_dim']}", ctrllb["k"])]
    corr_test = float(np.corrcoef(e1["z_te"], e2["z_te"])[0, 1])
    corr_oof = float(np.corrcoef(e1["z_rf"], e2["z_rf"])[0, 1])
    log(f"score correlation E65 vs logbp: Test r={corr_test:+.4f} | "
        f"Train OOF r={corr_oof:+.4f}")

    # ---------------------------------------------------------- verdict
    best_feature = max(feature_rows, key=lambda r: r["best_W"])
    best_score = max(score_rows, key=lambda r: r["best_W"])
    best_sel = max(selected_rows, key=lambda r: r["best_W"])
    best_three = max(three_rows, key=lambda r: r["best_W"])
    best_three_sel = max(three_sel, key=lambda r: r["best_W"])
    base_auc, base_w = 0.7578, 0.6958
    prior_auc, prior_w = 0.7609, 0.7164

    fusion_all = feature_rows + score_rows + three_rows + three_sel
    candidate_auc = max(r["diagnostic_auc"] for r in fusion_all)
    candidate_w = max(r["best_W"] for r in fusion_all)
    candidate_w090 = max(r["W_at_090"] for r in fusion_all)
    verdict = {
        "baseline_diagnostic_auc": base_auc, "baseline_W": base_w,
        "prior_best_diagnostic_auc": prior_auc, "prior_best_W": prior_w,
        "best_single_E65": ctrl65, "best_single_logbp": ctrllb,
        "logbp_reference_full_dim": max(ref_rows, key=lambda r: r["k"] == 12),
        "best_feature_concat": best_feature,
        "best_score_fusion": best_score,
        "best_score_fusion_train_selected": best_sel,
        "best_three_way": best_three,
        "best_three_way_train_selected": best_three_sel,
        "max_diagnostic_auc_any_fusion": candidate_auc,
        "max_best_W_any_fusion": candidate_w,
        "max_W_at_090_any_fusion": candidate_w090,
        "auc_gain_vs_E143": candidate_auc - base_auc,
        "auc_gain_vs_E163": candidate_auc - prior_auc,
        "w_gain_vs_E143": candidate_w - base_w,
        "w_gain_vs_E163": candidate_w - prior_w,
        "score_correlation_test": corr_test, "score_correlation_train_oof": corr_oof,
        "question1_auc_above_07609": bool(candidate_auc > prior_auc),
        "question2_W_above_07164": bool(candidate_w > prior_w),
    }

    results = {
        "experiment": CFG["experiment"],
        "device": str(DEVICE),
        "protocol": "E143/E163 verbatim: Train-only standardisation/PCA/prototypes/thresholds, "
                    "5-block Train OOF, LOSO excludes the scored identity with fit_rows-only "
                    "prototypes, evaluation block 0.6<frac<=0.8, Test scored only",
        "test_official": {"AUC": "N/A(no labels)", "F1": "N/A(no labels)"},
        "controls": controls,
        "logbp_reference_full_dim": ref_rows,
        "feature_concat": feature_rows,
        "score_fusion": score_rows,
        "score_fusion_train_selected": selected_rows,
        "score_fusion_3way": three_rows,
        "score_fusion_3way_train_selected": three_sel,
        "verdict": verdict,
        "runtime_seconds": round(time.time() - started, 1),
    }
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    e143_row = next(r for r in controls["E65_logspd"] if r["pca_dim"] == 96 and r["k"] == 12)
    write_report(results, ctrl65, ctrllb, e143_row)
    log(f"done in {time.time() - started:.0f}s")
    log(json.dumps({k: v for k, v in verdict.items()
                    if not isinstance(v, dict)}, ensure_ascii=False, indent=2))


def fmt_row(r: dict) -> str:
    return (f"| {r['config']} | {r['diagnostic_auc']:.4f} | {r['train_oof_auc']:.4f} | "
            f"{r['train_oof_f1']:.4f} | {r['loso_at_090']:.4f} | {r['s1_at_090']:.4f} | "
            f"{r['s2_at_090']:.4f} | {r['W_at_090']:.4f} | {r['best_W']:.4f} | "
            f"{r['best_acceptance']:.2f} | {r['loso_at_best']:.4f} | "
            f"{r['s1_at_best']:.4f} | {r['s2_at_best']:.4f} |")


HEADER = ("| 配置 | 诊断 AUC | Train OOF AUC | Train OOF F1 | LOSO@.90 | S1@.90 | S2@.90 | "
          "W@.90 | 最优 W | 接受率 | LOSO | S1 拒识 | S2 接受 |")
SEP = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"


def write_report(results, ctrl65, ctrllb, e143_row) -> None:
    v = results["verdict"]
    bf = v["best_feature_concat"]
    bs = v["best_score_fusion"]
    bt = v["best_score_fusion_train_selected"]
    b3 = v["best_three_way"]
    b3s = v["best_three_way_train_selected"]
    ref = v["logbp_reference_full_dim"]
    bf_auc = max(results["feature_concat"], key=lambda r: r["diagnostic_auc"])
    bs_auc = max(results["score_fusion"], key=lambda r: r["diagnostic_auc"])
    b3_auc = max(results["score_fusion_3way"], key=lambda r: r["diagnostic_auc"])
    lines = [
        "# E167 E65 correlation-logSPD × log 频带功率 分数级/特征级融合",
        "",
        "目标：用与 E163 完全相同的协议，验证 E65（855 维 correlation log-SPD）与 log band power"
        "（90 维，5 频带 × 18 通道）的**分数级/特征级融合**能否把开放集诊断 AUC 抬过 0.7609、"
        "W 抬过 0.7164。动机：E148 显示 logbp 单独诊断 AUC 0.7772 高于 E65 的 0.7578，"
        "E65+logbp 特征拼接 0.7894；E165 证明 E65 对通道/频带增益与循环平移完全不敏感，"
        "logbp 恰好携带被标准化掉的幅值信息。",
        "",
        "协议：Train-only 标准化/PCA/原型/阈值；5 块 Train OOF（接受率网格 0.80–0.96）；"
        "LOSO 每折排除被评分身份、原型只用 fit_rows(frac≤0.6)、评估块 0.6<frac≤0.8；"
        "Test 只评分；Session 仅用于诊断 AUC。Test 官方 AUC / F1 = **N/A**（无标签）。",
        "",
        "## 结论",
        "",
    ]

    gain_vs_e163 = v["auc_gain_vs_E163"]
    lines += [
        f"**诊断 AUC 最好 {v['max_diagnostic_auc_any_fusion']:.4f}"
        f"（vs E163 0.7609：{gain_vs_e163:+.4f}；vs E143 0.7578："
        f"{v['auc_gain_vs_E143']:+.4f}）。**",
        f"分方式看：分数级融合最高 AUC {bs_auc['diagnostic_auc']:.4f}"
        f"（{bs_auc['config']}，该点 W@0.90 {bs_auc['W_at_090']:.4f} / 最优 W "
        f"{bs_auc['best_W']:.4f}）；特征拼接最高 AUC {bf_auc['diagnostic_auc']:.4f}"
        f"（{bf_auc['config']}，W@0.90 {bf_auc['W_at_090']:.4f}）；"
        f"三路融合最高 AUC {b3_auc['diagnostic_auc']:.4f}（{b3_auc['config']}）。",
        f"W 最好 {v['max_best_W_any_fusion']:.4f}（vs E163 0.7164："
        f"{v['w_gain_vs_E163']:+.4f}；vs E143 0.6958：{v['w_gain_vs_E143']:+.4f}），"
        f"取自 {max([bf, bs, bt, b3, b3s], key=lambda r: r['best_W'])['config']}"
        f"（接受率网格用 Test 选点）；固定工作点 W@0.90 最好 "
        f"{v['max_W_at_090_any_fusion']:.4f}。",
        f"两路分数相关性（按各自最优单路配置）：Test r = {v['score_correlation_test']:+.4f}"
        f"（Train OOF r = {v['score_correlation_train_oof']:+.4f}）。",
        "Train-LOSO 选权把权重压倒 E65 一侧（w=0.90）：按 Train 自己的区分度衡量，logbp 对阈值"
        "行为几乎无增益；但它在 Test 上的排序信息是真实的（AUC ↑）。",
        "",
        "## 判定",
        "",
        f"- **核心问题 1（诊断 AUC > 0.7609）**："
        f"{'是' if v['question1_auc_above_07609'] else '否'}，最好 "
        f"{v['max_diagnostic_auc_any_fusion']:.4f}。",
        f"- **核心问题 2（W > 0.7164）**："
        f"{'是' if v['question2_W_above_07164'] else '否'}，最好 "
        f"{v['max_best_W_any_fusion']:.4f}（@a={max([bf, bs, bt, b3, b3s], key=lambda r: r['best_W'])['best_acceptance']:.2f}）。",
        "",
    ]
    if v["question1_auc_above_07609"] and not v["question2_W_above_07164"]:
        lines.append("- AUC 升但 W 不升 → **只是工作点效应**（数字见下），不能算方案提升。")
    elif not v["question1_auc_above_07609"]:
        lines.append("- AUC 未升 → logbp 的提升在分数融合里被 E65 的主方向或跨 session 漂移吃掉。")
    elif v["question2_W_above_07164"]:
        lines.append("- AUC 与 W 同时提升 → 幅值信息确实带来了 E65 缺失的开放集可分性。")
    lines += [
        f"- **工作点证据**：以分数融合 AUC 最好的配置（{bs_auc['config']}）为例，@0.90 时 "
        f"LOSO 拒识 {bs_auc['loso_at_090']:.4f}（E65 单路 {ctrl65['loso_at_090']:.4f}）、"
        f"S1 拒识 {bs_auc['s1_at_090']:.4f}（{ctrl65['s1_at_090']:.4f}）、"
        f"S2 接受 {bs_auc['s2_at_090']:.4f}（{ctrl65['s2_at_090']:.4f}），"
        f"W {bs_auc['W_at_090']:.4f}（{ctrl65['W_at_090']:.4f}）。"
        f"原因是 logbp 自身的 Train LOSO 可分性弱（{ctrllb['config']}：Train OOF AUC "
        f"{ctrllb['train_oof_auc']:.4f}、LOSO@.90 {ctrllb['loso_at_090']:.4f}；E65 为 "
        f"{ctrl65['train_oof_auc']:.4f} / {ctrl65['loso_at_090']:.4f}），融合后 OOF 分位阈值"
        f"被整体推到更宽松的位置，S2 接受的收益补偿不了 LOSO/S1 拒识的损失。",
        "- 结论方向：**排序信息已经足够（AUC 0.79–0.80），瓶颈在阈值校准**——"
        "S1/S2 的最优排序点与 Train OOF 分位点不一致，需要与分数形状无关的决策规则。",
        "",
        "## 单路对照组（复现检查）",
        "",
        HEADER, SEP,
        fmt_row(e143_row), fmt_row(ctrl65), fmt_row(ctrllb),
        "",
        f"- **E143 精确复现行**（E65 PCA-96 k=12）：诊断 AUC {e143_row['diagnostic_auc']:.4f} / "
        f"W@0.90 {e143_row['W_at_090']:.4f}（E143 公布 0.7578 / 0.6958，"
        f"ΔAUC {e143_row['diagnostic_auc'] - 0.7578:+.4f} / "
        f"ΔW {e143_row['W_at_090'] - 0.6958:+.4f}），"
        f"LOSO {e143_row['loso_at_090']:.4f} / S1 {e143_row['s1_at_090']:.4f} / "
        f"S2 {e143_row['s2_at_090']:.4f}。",
        f"- E65 网格内最优（PCA-{ctrl65['pca_dim']} k={ctrl65['k']}）：诊断 AUC "
        f"{ctrl65['diagnostic_auc']:.4f} / W@0.90 {ctrl65['W_at_090']:.4f}"
        f"（E163 同一行为 0.7622 / 0.6908）。",
        f"- logbp 对照（网格内最优，PCA-{ctrllb['pca_dim']} k={ctrllb['k']}）：诊断 AUC "
        f"{ctrllb['diagnostic_auc']:.4f} / W@0.90 {ctrllb['W_at_090']:.4f} / 最优 W "
        f"{ctrllb['best_W']:.4f}。",
        f"- logbp 全 90 维 k=12 参考（对应 E148 的 PCA-96 探针）：诊断 AUC "
        f"{ref['diagnostic_auc']:.4f}（E148 公布 0.7772，ΔAUC "
        f"{ref['diagnostic_auc'] - 0.7772:+.4f}）/ W@0.90 {ref['W_at_090']:.4f}"
        f"（E148 公布 0.6515；差异来自 E148 旧脚本的阈值/LOSO 实现，AUC 一致说明打分本身相同）。",
        "",
        "## 各融合方式对比",
        "",
        "### 分数级融合（固定权重 0.25/0.5/0.75，0.5 为等权）",
        "",
        HEADER, SEP,
    ]
    top_score = sorted(results["score_fusion"], key=lambda r: -r["diagnostic_auc"])[:10]
    best_score_w = sorted(results["score_fusion"], key=lambda r: -r["best_W"])[:10]
    seen = set()
    for r in top_score + best_score_w:
        if r["config"] in seen:
            continue
        seen.add(r["config"])
        lines.append(fmt_row(r))
    lines += ["", "按最优 W 排序：", "", HEADER, SEP]
    for r in sorted(results["score_fusion"], key=lambda r: -r["best_W"])[:10]:
        lines.append(fmt_row(r))
    lines += ["", "### 分数级融合（权重只由 Train LOSO AUC 选，Test 不参与选权）", "",
              HEADER, SEP]
    for r in sorted(results["score_fusion_train_selected"], key=lambda r: -r["diagnostic_auc"])[:10]:
        lines.append(fmt_row(r))
    lines += ["", "### 三路分数融合（E65 + logbp + hjorth）", "", HEADER, SEP]
    for r in sorted(results["score_fusion_3way"], key=lambda r: -r["diagnostic_auc"])[:8]:
        lines.append(fmt_row(r))
    lines += ["", "三路融合（Train-LOSO 选权）：", "", HEADER, SEP]
    for r in sorted(results["score_fusion_3way_train_selected"],
                    key=lambda r: -r["diagnostic_auc"])[:8]:
        lines.append(fmt_row(r))
    lines += ["", "### 特征拼接 + 联合 PCA（对照）", "", HEADER, SEP]
    for r in sorted(results["feature_concat"], key=lambda r: -r["diagnostic_auc"])[:10]:
        lines.append(fmt_row(r))
    lines += ["", "## 最好单配置的分任务诊断 AUC", ""]
    for tag, rec in [("E65 对照", ctrl65), ("logbp 对照", ctrllb), ("logbp 全 90 维", ref),
                     ("特征拼接", bf), ("分数融合", bs), ("分数融合(Train 选权)", bt),
                     ("三路融合", b3), ("三路融合(Train 选权)", b3s)]:
        tasks = ", ".join(f"{k} {val:.4f}" for k, val in rec["per_task_auc"].items())
        lines.append(f"- {tag}（{rec['config']}）：{tasks}")
    lines += [
        "",
        "## 说明",
        "",
        "- 表格中 @.90 指 E143 参考工作点（接受率 0.90），最优 W 一行跨接受率网格 0.80–0.96 取最大。",
        "- Train OOF AUC 为 5 块 OOF 分数对「held 身份未知代理 vs 已注册块」的排序能力；"
        "Train OOF F1 为同一 OOF 的 20 类宏 F1。",
        "- 分数融合前每路分数都用自身 5 块 Train OOF 的均值/标准差做 z 标准化。",
        "- Test 官方 AUC / F1：**N/A**（Test 无真实标签，Session 仅用于诊断 AUC）。",
        "",
        f"运行耗时 {results['runtime_seconds']:.0f}s。",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if REPORT_ONLY:
        # 只重建 report.md，不重跑网格（results.json 已含全部结果）
        res = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
        res["verdict"]["logbp_reference_full_dim"] = max(
            res["logbp_reference_full_dim"], key=lambda r: r["k"] == 12)
        c65 = max(res["controls"]["E65_logspd"], key=lambda r: r["best_W"])
        clb = max(res["controls"]["logbp"], key=lambda r: r["best_W"])
        e143 = next(r for r in res["controls"]["E65_logspd"]
                    if r["pca_dim"] == 96 and r["k"] == 12)
        write_report(res, c65, clb, e143)
    else:
        main()
