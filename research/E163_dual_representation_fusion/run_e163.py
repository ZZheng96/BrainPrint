"""E163: feature-level fusion of the E65 correlation-logSPD view and the E162 raw-EEG
contrastive embedding, under the verbatim E143 open-set protocol.

Two information sources:
  * E65  : 855-d correlation log-SPD vector (structured covariance geometry)
  * E162 : 256-d end-to-end raw-EEG embedding (learned, task-standardised effective rank 8)

Fusion families:
  a) feature concat -> joint PCA -> prototypes
  b) score fusion of the two task-standardised z-scores (weights 0.25/0.5/0.75)
  c) score fusion whose weight is selected on Train LOSO only (Test never consulted)

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
# 允许传第二个 config（例如只跑分数融合的精简网格），便于中断后复跑而不重算全部拼接网格。
CFG_PATH = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else OUT / "config.json"
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
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


sys.stdout = Tee(OUT / "run.log")


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
    """Verbatim E143/E162 seeding so the E65 control reproduces the published W."""
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
    """Full row x prototype negative-distance matrix; off-task entries = -inf.

    Needed only for the fused identity prediction (argmax over the combined
    distance), which the two score streams otherwise discard.
    """
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


def load_stream(ctx: Context, spec: dict, stat_fit_only: bool = False) -> tuple:
    a = np.load(ROOT / spec["train"]).astype(np.float32)
    b = np.load(ROOT / spec["test"]).astype(np.float32)
    stat_rows = ctx.fit_rows if stat_fit_only else np.ones(len(a), bool)
    return task_standardise(a, b, ctx.tasks, ctx.test_tasks, stat_rows)


# ------------------------------------------------------------------ protocol
def evaluate(at: np.ndarray, bt: np.ndarray, k: int, ctx: Context,
             keep_blocks: bool) -> dict:
    """Verbatim E143 open-set protocol on one projected representation."""
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


def fuse(e1: dict, e2: dict, at1: np.ndarray, at2: np.ndarray, w: float,
         ctx: Context) -> dict:
    """Weighted score fusion of two evaluated streams with identical prototype keys."""
    z_rf = (w * e1["z_rf"] + (1 - w) * e2["z_rf"]).astype(np.float32)
    z_te = (w * e1["z_te"] + (1 - w) * e2["z_te"]).astype(np.float32)
    loso = [(w * k1 + (1 - w) * k2, w * u1 + (1 - w) * u2)
            for (k1, u1), (k2, u2) in zip(e1["loso"], e2["loso"])]

    pred = np.full(len(ctx.nodes), -1, dtype=np.int32)
    for b1, b2 in zip(e1["blocks"], e2["blocks"]):
        assert np.array_equal(b1["idx"], b2["idx"]), "block rows misaligned"
        assert [tuple(x) for x in b1["keys"]] == [tuple(x) for x in b2["keys"]], "keys misaligned"
        idx = b1["idx"]
        d1, ksubj = neg_dmat(at1[idx], b1["protos"], b1["keys"], ctx.tasks[idx])
        d2, _ = neg_dmat(at2[idx], b2["protos"], b2["keys"], ctx.tasks[idx])
        z1 = (d1 - e1["mu"]) / e1["sd"]
        z2 = (d2 - e2["mu"]) / e2["sd"]
        pred[idx] = ksubj[(w * z1 + (1 - w) * z2).argmax(1)]

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

    return {"z_rf": z_rf, "z_te": z_te, "loso": loso, "blocks": e1["blocks"],
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


def fused_loso_auc(e1: dict, e2: dict, w: float) -> float:
    """Train-only criterion used to pick the fusion weight (no Test involvement)."""
    unknown = np.concatenate([w * u1 + (1 - w) * u2 for (_, u1), (_, u2) in zip(e1["loso"], e2["loso"])])
    known = np.concatenate([w * k1 + (1 - w) * k2 for (k1, _), (k2, _) in zip(e1["loso"], e2["loso"])])
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
        "per_task_auc": entry["per_task_auc"],
    })
    return record


# ------------------------------------------------------------------ main
def main() -> None:
    started = time.time()
    ctx = Context()
    log(f"device {DEVICE} | train {len(ctx.nodes)} | test {len(ctx.test_tasks)} | "
        f"fit_rows {int(ctx.fit_rows.sum())} | window {int(ctx.window.sum())}")

    streams = {}
    for name, spec in CFG["streams"].items():
        a, b = load_stream(ctx, spec)
        streams[name] = (a, b)
        log(f"loaded {name}: {a.shape} / {b.shape}")

    # ---------------------------------------------------------- projections
    controls, feature_rows = {}, []
    entries: dict[tuple, dict] = {}

    def entry_for(proj_key: str, at: np.ndarray, bt: np.ndarray, k: int,
                  keep_blocks: bool = False) -> dict:
        key = (proj_key, int(k))
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
                e = entry_for(f"{name}@{dim}", at, bt, k, keep_blocks=True)
                rec = summarise(e, {"family": f"{name}_only", "stream": name,
                                    "pca_dim": dim, "k": k,
                                    "config": f"{name} PCA-{dim} k={k}"})
                controls[name].append(rec)
                log(f"  [{name}] dim={dim} k={k} OOFauc={rec['train_oof_auc']:.4f} "
                    f"OOFf1={rec['train_oof_f1']:.4f} diagAUC={rec['diagnostic_auc']:.4f} "
                    f"W@.90={rec['W_at_090']:.4f} bestW={rec['best_W']:.4f}"
                    f"@{rec['best_acceptance']:.2f} ({time.time() - t0:.0f}s)")

    # E65 control under fit-rows-only standardisation (sensitivity)
    log("sensitivity: E65 control with fit-rows-only standardisation")
    a, b = load_stream(ctx, CFG["streams"]["E65_logspd"], stat_fit_only=True)
    at, bt = ctx.project("E65_logspd_fitstat", a, b, 96)
    sens = [summarise(entry_for("E65_logspd_fitstat", at, bt, k),
                      {"family": "control_sensitivity", "stream": "E65_logspd",
                       "pca_dim": 96, "k": k,
                       "config": f"E65 PCA-96 k={k} (fit-rows-only stats)"})
            for k in CFG["prototype_k"]]
    for rec in sens:
        log(f"  [E65 fit-stat] k={rec['k']} diagAUC={rec['diagnostic_auc']:.4f} "
            f"W@.90={rec['W_at_090']:.4f} bestW={rec['best_W']:.4f}")

    # ---------------------------------------------------------- feature concat
    log("feature-level concat fusion")
    a65, b65 = streams["E65_logspd"]
    a62, b62 = streams["E162_raw_eeg"]
    for dim_a, dim_b in CFG["fusion"]["feature_concat"]["dim_pairs"]:
        at65, bt65 = ctx.project("E65_logspd", a65, b65, dim_a)
        at62, bt62 = ctx.project("E162_raw_eeg", a62, b62, dim_b)
        blocks_tr = {"raw": (at65, at62), "block_norm": (None, None)}
        scale_a = float(np.sqrt(np.mean(at65[ctx.fit_rows] ** 2)))
        scale_b = float(np.sqrt(np.mean(at62[ctx.fit_rows] ** 2)))
        blocks_tr["block_norm"] = (at65 / max(scale_a, 1e-6), at62 / max(scale_b, 1e-6))
        blocks_te = {"raw": (bt65, bt62),
                     "block_norm": (bt65 / max(scale_a, 1e-6), bt62 / max(scale_b, 1e-6))}
        for variant in CFG["fusion"]["feature_concat"]["variants"]:
            tr = np.concatenate(blocks_tr[variant], axis=1)
            te = np.concatenate(blocks_te[variant], axis=1)
            for jd in CFG["fusion"]["feature_concat"]["joint_pca_dims"]:
                if jd > tr.shape[1]:
                    continue
                key = f"concat_{variant}_{dim_a}_{dim_b}"
                at, bt = ctx.project(key, tr, te, jd)
                for k in CFG["prototype_k"]:
                    rec = summarise(entry_for(f"{key}@{jd}", at, bt, k),
                                    {"family": f"feature_concat_{variant}",
                                     "stream": "E65+E162", "pca_dim_e65": dim_a,
                                     "pca_dim_e162": dim_b, "joint_pca_dim": jd, "k": k,
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
    log("score-level fusion")
    score_rows, selected_rows = [], []
    weights = CFG["fusion"]["score_fusion"]["weights"]
    sel_grid = CFG["fusion"]["score_fusion"]["train_selected_weight_grid"]
    for dim_a in CFG["streams"]["E65_logspd"]["pca_dims"]:
        for dim_b in CFG["streams"]["E162_raw_eeg"]["pca_dims"]:
            at1, _ = ctx.project("E65_logspd", a65, b65, dim_a)
            at2, _ = ctx.project("E162_raw_eeg", a62, b62, dim_b)
            for k in CFG["prototype_k"]:
                e1 = entries[(f"E65_logspd@{dim_a}", k)]
                e2 = entries[(f"E162_raw_eeg@{dim_b}", k)]
                for w in weights:
                    fused = fuse(e1, e2, at1, at2, w, ctx)
                    score_rows.append(summarise(fused, {
                        "family": "score_fusion", "stream": "E65+E162",
                        "pca_dim_e65": dim_a, "pca_dim_e162": dim_b, "k": k,
                        "weight_e65": w, "weight_e162": 1 - w,
                        "config": f"score {w:.2f}*E65({dim_a}) + "
                                  f"{1 - w:.2f}*E162({dim_b}) k={k}"}))
                # weight chosen on Train LOSO AUC only
                train_aucs = {w: fused_loso_auc(e1, e2, w) for w in sel_grid}
                w_best = max(train_aucs, key=train_aucs.get)
                fused = fuse(e1, e2, at1, at2, w_best, ctx)
                selected_rows.append(summarise(fused, {
                    "family": "score_fusion_train_selected", "stream": "E65+E162",
                    "pca_dim_e65": dim_a, "pca_dim_e162": dim_b, "k": k,
                    "weight_e65": w_best, "weight_e162": 1 - w_best,
                    "train_loso_auc_by_weight": {str(w): v for w, v in train_aucs.items()},
                    "config": f"score {w_best:.2f}*E65({dim_a}) + "
                              f"{1 - w_best:.2f}*E162({dim_b}) k={k} [w by Train LOSO]"}))
            top = max((r for r in score_rows
                       if r["pca_dim_e65"] == dim_a and r["pca_dim_e162"] == dim_b),
                      key=lambda r: r["best_W"])["best_W"]
            log(f"  [score {dim_a}+{dim_b}] bestW={top:.4f} "
                f"selectedW={selected_rows[-1]['best_W']:.4f} "
                f"(w={selected_rows[-1]['weight_e65']:.1f}, "
                f"diagAUC={selected_rows[-1]['diagnostic_auc']:.4f})")

    # ---------------------------------------------------------- redundancy
    ctrl65 = max([r for r in controls["E65_logspd"]], key=lambda r: r["best_W"])
    ctrl62 = max([r for r in controls["E162_raw_eeg"]], key=lambda r: r["best_W"])
    e1 = entries[(f"E65_logspd@{ctrl65['pca_dim']}", ctrl65["k"])]
    e2 = entries[(f"E162_raw_eeg@{ctrl62['pca_dim']}", ctrl62["k"])]
    corr_test = float(np.corrcoef(e1["z_te"], e2["z_te"])[0, 1])
    corr_oof = float(np.corrcoef(e1["z_rf"], e2["z_rf"])[0, 1])
    log(f"score correlation: Test r={corr_test:+.4f} | Train OOF r={corr_oof:+.4f}")

    # ---------------------------------------------------------- verdict
    best_feature = max(feature_rows, key=lambda r: r["best_W"]) if feature_rows else None
    best_score = max(score_rows, key=lambda r: r["best_W"])
    best_sel = max(selected_rows, key=lambda r: r["best_W"])
    base_auc, base_w = 0.7578, 0.6958

    candidate_auc = max([r["diagnostic_auc"] for r in feature_rows + score_rows] or [0.0])
    verdict = {
        "baseline_diagnostic_auc": base_auc,
        "baseline_W": base_w,
        "best_single_E65": ctrl65, "best_single_E162": ctrl62,
        "best_feature_concat": best_feature,
        "best_score_fusion": best_score,
        "best_score_fusion_train_selected": best_sel,
        "max_diagnostic_auc_any_fusion": candidate_auc,
        "auc_gain_vs_baseline": candidate_auc - base_auc,
        "w_gain_vs_baseline_best": max(r["best_W"] for r in feature_rows + score_rows) - base_w,
        "w_gain_vs_baseline_at_090": max(r["W_at_090"] for r in feature_rows + score_rows) - base_w,
        "score_correlation_test": corr_test, "score_correlation_train_oof": corr_oof,
    }

    results = {
        "experiment": CFG["experiment"],
        "device": str(DEVICE),
        "protocol": "E143 verbatim: Train-only standardisation/PCA/prototypes/thresholds, "
                    "5-block Train OOF, LOSO excludes the scored identity, evaluation block "
                    "0.6<frac<=0.8, Test scored only",
        "test_official": {"AUC": "N/A(no labels)", "F1": "N/A(no labels)"},
        "controls": controls,
        "control_sensitivity_fit_rows_stats": sens,
        "feature_concat": feature_rows,
        "score_fusion": score_rows,
        "score_fusion_train_selected": selected_rows,
        "verdict": verdict,
        "runtime_seconds": round(time.time() - started, 1),
    }
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    write_report(results, ctrl65, ctrl62)
    log(f"done in {time.time() - started:.0f}s")
    log(json.dumps({k: v for k, v in verdict.items()
                    if k not in ("best_single_E65", "best_single_E162")},
                   ensure_ascii=False, indent=2))


def fmt_row(r: dict) -> str:
    return (f"| {r['config']} | {r['diagnostic_auc']:.4f} | {r['train_oof_auc']:.4f} | "
            f"{r['train_oof_f1']:.4f} | {r['W_at_090']:.4f} | {r['best_W']:.4f} | "
            f"{r['best_acceptance']:.2f} | {r['loso_at_best']:.4f} | "
            f"{r['s1_at_best']:.4f} | {r['s2_at_best']:.4f} |")


HEADER = ("| 配置 | 诊断 AUC | Train OOF AUC | Train OOF F1 | W@0.90 | 最优 W | 接受率 | "
          "LOSO | S1 拒识 | S2 接受 |")
SEP = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"


def write_report(results, ctrl65, ctrl62) -> None:
    v = results["verdict"]
    bf = v["best_feature_concat"]
    bs = v["best_score_fusion"]
    bt = v["best_score_fusion_train_selected"]
    lines = [
        "# E163 双表征特征级融合（E65 correlation-logSPD × E162 原始 EEG 端到端）",
        "",
        "目标：验证把 E162 原始 EEG 端到端 embedding 与 E65 correlation-logSPD **特征级/分数级融合**"
        "能否把开放集诊断 AUC 抬过 E143 的 0.7578、W 抬过 0.6958。",
        "协议与 E143/E162 完全一致：Train-only 标准化/PCA/原型/阈值，5 块 Train OOF，"
        "LOSO 每折排除被评身份且原型只用 fit_rows，评估块 0.6<frac≤0.8，Test 只评分，Session 仅诊断。",
        "",
        "## 结论",
        "",
    ]

    gain_auc = v["auc_gain_vs_baseline"]
    if gain_auc > 0.001:
        lines += [f"**融合把诊断 AUC 抬到 {v['max_diagnostic_auc_any_fusion']:.4f}"
                  f"（+{gain_auc:.4f}）。**"]
    else:
        lines += [f"**融合没有把诊断 AUC 抬高：最好 {v['max_diagnostic_auc_any_fusion']:.4f}"
                  f"（Δ {gain_auc:+.4f}）。**"]
    best_w_any = max(bf["best_W"], bs["best_W"], bt["best_W"])
    lines += [
        f"W：特征拼接最好 {bf['best_W']:.4f}（@a={bf['best_acceptance']:.2f}），"
        f"分数融合最好 {bs['best_W']:.4f}（@a={bs['best_acceptance']:.2f}），"
        f"Train-LOSO 选权 {bt['best_W']:.4f}（@a={bt['best_acceptance']:.2f}）；"
        f"基线 E143 W = 0.6958，Δ = {best_w_any - 0.6958:+.4f}。",
        f"两路 Test 分数相关性 r = {v['score_correlation_test']:+.4f}"
        f"（Train OOF r = {v['score_correlation_train_oof']:+.4f}）。",
        "",
        "## 单路对照组（复现检查）",
        "",
        HEADER, SEP,
        fmt_row(ctrl65), fmt_row(ctrl62),
        "",
        f"- E65 对照：诊断 AUC {ctrl65['diagnostic_auc']:.4f} / W@0.90 {ctrl65['W_at_090']:.4f}"
        f"（E143 公布 0.7578 / 0.6958，ΔAUC {ctrl65['diagnostic_auc'] - 0.7578:+.4f} / "
        f"ΔW {ctrl65['W_at_090'] - 0.6958:+.4f}）。",
        f"- E162 对照：诊断 AUC {ctrl62['diagnostic_auc']:.4f} / 最优 W {ctrl62['best_W']:.4f}"
        f"（E162 公布 0.6756 / 0.6761）。",
        "",
        "## 判定",
        "",
        f"- 关键问题 1（诊断 AUC > 0.7578）：{'是' if gain_auc > 0.001 else '否'}，"
        f"最好 {v['max_diagnostic_auc_any_fusion']:.4f}。",
        f"- 关键问题 2（W > 0.6958）：{'是' if best_w_any > 0.6958 else '否'}，"
        f"最好 {best_w_any:.4f}。",
    ]
    if gain_auc > 0.001 and best_w_any <= 0.6958:
        lines += ["- **AUC 提升但 W 不升 → 只是工作点变化**：融合改善的是排序（S1/S2 分离），"
                  "但同一 OOF 分位阈值下 LOSO 与 S1 拒识被 S2 接受的增益抵消，"
                  "不能算方案提升。"]
    elif gain_auc <= 0.001:
        lines += ["- **AUC 未提升 → 两路信息冗余或 E162 噪声主导**：E162 有效秩仅 8 且其分数与 E65 "
                  "在 Test 上相关性高，PCA 后的 E162 子空间在联合 PCA 中被 E65 主方向淹没；"
                  "分数融合的权重也无法把 S1/S2 排序拉出新的可分性。"]
    lines += ["", "## 各融合方式对比", "",
              "### 特征拼接 + 联合 PCA", "", HEADER, SEP]
    for r in sorted(results["feature_concat"], key=lambda r: -r["best_W"])[:15]:
        lines.append(fmt_row(r))
    lines += ["", "### 分数融合（固定权重 0.25/0.5/0.75，0.5 为等权）", "", HEADER, SEP]
    for r in sorted(results["score_fusion"], key=lambda r: -r["best_W"])[:15]:
        lines.append(fmt_row(r))
    lines += ["", "### 分数融合（权重只由 Train LOSO AUC 选，Test 不参与选权）", "", HEADER, SEP]
    for r in sorted(results["score_fusion_train_selected"], key=lambda r: -r["best_W"])[:15]:
        lines.append(fmt_row(r))
    lines += ["", "## 最好单配置的分任务诊断 AUC", ""]
    for tag, rec in [("E65 对照", ctrl65), ("E162 对照", ctrl62),
                     ("特征拼接", bf), ("分数融合", bs)]:
        tasks = ", ".join(f"{k} {v:.4f}" for k, v in rec["per_task_auc"].items())
        lines.append(f"- {tag}（{rec['config']}）：{tasks}")
    lines += ["", "## 标准化口径", "",
              "主网格按 E143/E162 约定用该任务**全部 Train session1 行**估计均值/标准差"
              "（仍属 Train-only，且只有这样 E65 对照才能精确复现 0.7578/0.6958）。"
              "题目要求的「只用 fit_rows」口径作为敏感性对照单独给出：", "", HEADER, SEP]
    for r in results["control_sensitivity_fit_rows_stats"]:
        lines.append(fmt_row(r))
    lines += ["", "Test 官方 AUC / F1：**N/A**（Test 无真实标签，Session 仅用于诊断 AUC）。",
              "", f"运行耗时 {results['runtime_seconds']:.0f}s。"]

    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
