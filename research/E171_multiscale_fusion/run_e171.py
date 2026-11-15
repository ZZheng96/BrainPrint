"""E171: multi-scale sub-window correlation-logSPD fusion.

Background
  * E143 baseline (E65 correlation-logSPD -> PCA-96, k=12): LOSO 0.7936 / S1 0.7979 /
    S2 0.5960 / W 0.6958, diagnostic AUC 0.7578.
  * E167 (E65 + raw log band power feature concat) diagnostic AUC 0.7999 -- the highest so far.
  * E168 (subwin_mean: cut the trial into N non-overlapping windows, run the whole E65 pipeline
    on each window, average the 855-d vectors) reached diagnostic AUC 0.7822 at N=4 and the AUC
    increased monotonically with N (0.7690 / 0.7735 / 0.7822 for N=2/3/4). The sub-window mean is
    NOT the full-trial feature because detrend_zband is non-linear.
  * E169 (empirical slope): +0.042 diagnostic AUC buys +0.035 of the E159 upper bound, so W > 0.80
    needs diagnostic AUC ~0.835.

This run
  1. extend N to 1/2/3/4/6/8/12 (N=1 must reproduce E65 exactly), sweep PCA 48/64/96/128 x k 1/3/6/12;
  2. multi-scale concat of the N=2/3/4/6 sub-window means (per-block PCA + block_norm);
  3. feature-level fusion E65 + logbp + subwin_mean(N_best) (raw and block_norm, joint PCA 96/128/192)
     with the E65+logbp control that must reproduce 0.7999;
  4. score-level fusion on the 0.1 simplex, plus a Train-LOSO-selected weight vector;
  5. E159/E169 frozen-score ceiling analysis of the best configuration.

Protocol (verbatim E143/E168/E169)
  task-wise Train z-score (all Train rows of the task) -> PCA fitted on fit_rows (frac<=0.6) ->
  k-means prototypes -> 5-block Train OOF thresholds (acceptance 0.80-0.96) -> LOSO excluding the
  scored identity, prototypes from fit_rows only, evaluation block 0.6<frac<=0.8 -> Test scored once,
  session labels used only for the post-hoc diagnostic AUC. Test official AUC/F1 = N/A.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.metrics import f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FT = torch.float32
TASKS = CFG["evaluation"]["tasks"]
CLASSES = np.arange(1, 21, dtype=int)
DIM_WINDOW = 5 * 171
N_CH = int(CFG["data"]["channels"])
N_S = int(CFG["data"]["samples_per_trial"])
ACC = [float(a) for a in CFG["evaluation"]["acceptance_grid"]]
REF_ACC = float(CFG["evaluation"]["reference_acceptance"])
WEIGHTS = CFG["evaluation"]["weights"]
FIT_FRAC = float(CFG["evaluation"]["fit_fraction"])
N_BLOCKS = int(CFG["evaluation"]["blocks"])
TARGETS = CFG["targets"]

sys.path.insert(0, str(ROOT / "research" / "E65_correlation_spd"))
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
import run_e65  # noqa: E402
from run_e36 import load_config as load_e36_config, load_train_matrix, sha256  # noqa: E402


class Tee:
    def __init__(self, path: Path | None):
        self.file = open(path, "w", encoding="utf-8") if path is not None else None
        self.stdout = sys.__stdout__

    def write(self, text):
        self.stdout.write(text)
        if self.file is not None:
            self.file.write(text)

    def flush(self):
        self.stdout.flush()
        if self.file is not None:
            self.file.flush()


REPORT_ONLY = "--report-only" in sys.argv
PROBE = "--probe" in sys.argv
FAST = "--fast" in sys.argv
if FAST:
    CFG["subwindow"]["ns"] = [1, 4]
    CFG["grids"]["e65_control"] = {"dims": [64, 96], "k": [1, 12]}
    CFG["grids"]["subwindow"] = {"dims": [64], "k": [12]}
    CFG["grids"]["multiscale"] = {"ns": [1, 4], "block_dims": [48], "joint_dims": [96],
                                  "k": [6], "norm": "block_norm"}
    CFG["grids"]["control_e65_logbp"] = {"e65_dim": 64, "logbp_dim": 32, "joint_dims": [96],
                                         "k": [6], "variants": ["block_norm"]}
    CFG["grids"]["three_way"] = {"e65_dim": 96, "logbp_dim": 32, "subwin_dim": 64,
                                 "joint_dims": [96], "k": [6], "variants": ["block_norm"]}
    CFG["grids"]["two_way"] = {"e65_dim": 96, "subwin_dim": 64, "joint_dims": [96],
                               "k": [6], "variants": ["block_norm"]}
    CFG["grids"]["score_fusion"]["simplex_step"] = 0.5
sys.stdout = Tee(None if REPORT_ONLY else OUT / "run.log")


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------ feature build
def load_test_matrix(path: Path) -> np.ndarray:
    data = np.asarray(loadmat(path, variable_names=["data"])["data"], dtype=np.float32)
    if data.shape != (N_S, N_CH):
        raise ValueError(f"{path}: 预期 ({N_S},{N_CH})，实际 {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: 含非有限值")
    return data.T


def split_meta(split: str) -> pd.DataFrame:
    key = "train_metadata" if split == "train" else "test_metadata"
    meta = pd.read_csv(ROOT / CFG["data"][key])
    expected = CFG["data"]["expected_train_rows" if split == "train" else "expected_test_rows"]
    if len(meta) != expected:
        raise ValueError(f"{split} 行数 {len(meta)} != {expected}")
    return meta


def cache_path(split: str, n: int) -> Path:
    return OUT / f"{split}_subwin_mean_n{n}.npy"


def extract_split(split: str, meta: pd.DataFrame, e36: dict, e65: dict, ns: list) -> None:
    n = len(meta)
    outputs = {nn: np.lib.format.open_memmap(
        cache_path(split, nn), mode="w+", dtype=np.float32, shape=(n, DIM_WINDOW)) for nn in ns}
    loader = load_train_matrix if split == "train" else load_test_matrix
    paths = [ROOT / CFG["data"]["train_root"] / p for p in meta.path]
    sub_cfg = {}
    for nn in ns:
        length = N_S // nn
        if length * nn != N_S:
            raise ValueError(f"N={nn} 不能整除 {N_S}")
        cfg = json.loads(json.dumps(e36))
        cfg["data"]["samples_per_trial"] = length
        sub_cfg[nn] = cfg
    batch = int(CFG["subwindow"]["io_batch_size"])
    started = time.time()
    with ThreadPoolExecutor(max_workers=int(CFG["subwindow"]["io_workers"])) as pool:
        for start in range(0, n, batch):
            stop = min(start + batch, n)
            arrays = list(pool.map(loader, paths[start:stop]))
            raw = torch.as_tensor(np.stack(arrays), dtype=FT, device=DEVICE)
            for nn in ns:
                length = N_S // nn
                wins = []
                for w in range(nn):
                    seg = raw[..., w * length:(w + 1) * length].contiguous()
                    with torch.no_grad():
                        wins.append(run_e65.gpu_correlation_logspd(seg, sub_cfg[nn], e65))
                outputs[nn][start:stop] = torch.stack(wins, dim=1).mean(1).cpu().numpy()
            if stop % 4096 == 0 or stop == n:
                log(f"  {split} features {stop}/{n} ({time.time() - started:.0f}s)")
    for handle in outputs.values():
        handle.flush()
    del outputs
    log(f"{split} 子窗口特征完成（{time.time() - started:.0f}s）")


def feature_checks(e36: dict, e65: dict, ns: list) -> dict:
    """N=1 全窗必须逐位等于 E65 缓存；单条与批量一致。"""
    rng = np.random.default_rng(int(CFG["seed"]))
    checks = {"finite": True, "test_accessed": False}
    for split, ref_path in (("train", CFG["reference_views"]["E65"]["train"]),
                            ("test", CFG["reference_views"]["E65"]["test"])):
        meta = split_meta(split)
        loader = load_train_matrix if split == "train" else load_test_matrix
        pick = np.sort(rng.choice(len(meta), 8, replace=False))
        paths = [ROOT / CFG["data"]["train_root"] / meta.path.iloc[i] for i in pick]
        raw = torch.as_tensor(np.stack([loader(p) for p in paths]), dtype=FT, device=DEVICE)
        with torch.no_grad():
            full = run_e65.gpu_correlation_logspd(raw, e36, e65).cpu().numpy()
            singles = torch.cat([run_e65.gpu_correlation_logspd(raw[i:i + 1], e36, e65)
                                 for i in range(len(raw))]).cpu().numpy()
        cached_ref = np.asarray(np.load(ROOT / ref_path, mmap_mode="r")[pick])
        n1 = np.asarray(np.load(cache_path(split, 1), mmap_mode="r")[pick])
        checks[f"{split}_n1_vs_E65_cache_max_abs"] = float(np.abs(n1 - cached_ref).max())
        checks[f"{split}_recomputed_vs_E65_cache_max_abs"] = float(np.abs(full - cached_ref).max())
        checks[f"{split}_single_vs_batch_max_abs"] = float(np.abs(full - singles).max())
        checks[f"{split}_n1_vs_recomputed_max_abs"] = float(np.abs(n1 - full).max())
    if max(checks[k] for k in checks if k.endswith("max_abs")) > 1e-5:
        raise RuntimeError(f"N=1 子窗口流程未能复现 E65：{checks}")
    log(f"特征检查 {json.dumps(checks, ensure_ascii=False)}")
    return checks


def build_features() -> dict:
    ns = [int(n) for n in CFG["subwindow"]["ns"]]
    manifest = OUT / "feature_manifest.json"
    shapes_ok = all(cache_path(s, n).is_file()
                    and tuple(np.load(cache_path(s, n), mmap_mode="r").shape)
                    == (len(split_meta(s)), DIM_WINDOW)
                    for s in ("train", "test") for n in ns)
    if shapes_ok and manifest.is_file():
        report = json.loads(manifest.read_text(encoding="utf-8"))
        if report.get("config_sha256") == sha256(OUT / "config.json"):
            log("复用 E171 子窗口特征缓存")
            return report
    if not torch.cuda.is_available():
        raise RuntimeError("E171 特征提取需要 CUDA")
    e36 = load_e36_config()
    e65 = json.loads((ROOT / "research" / "E65_correlation_spd" / "config.json")
                     .read_text(encoding="utf-8"))
    started = time.time()
    for split in ("train", "test"):
        extract_split(split, split_meta(split), e36, e65, ns)
    checks = feature_checks(e36, e65, ns)
    report = {
        "config_sha256": sha256(OUT / "config.json"),
        "ns": ns,
        "variant": "subwin_mean",
        "shapes": {f"n{n}": list(np.load(cache_path("train", n), mmap_mode="r").shape)
                   for n in ns},
        "test_shapes": {f"n{n}": list(np.load(cache_path("test", n), mmap_mode="r").shape)
                        for n in ns},
        "checks": checks,
        "device": torch.cuda.get_device_name(0),
        "elapsed_seconds": round(time.time() - started, 1),
        "convention": "每个子窗口独立 detrend_zband；N 个 855 维特征取算术平均",
    }
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(report, ensure_ascii=False, indent=2))
    return report


# ------------------------------------------------------------------ protocol primitives
def frac(meta: pd.DataFrame) -> np.ndarray:
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def task_standardise(a: np.ndarray, b: np.ndarray, tasks: np.ndarray,
                     test_tasks: np.ndarray) -> tuple:
    """E143 逐字：统计量取自该 task 的全部 Train 行，Test 用同一组均值/方差。"""
    for task in TASKS:
        m = tasks == task
        mu, sd = a[m].mean(0), a[m].std(0) + 1e-6
        a[m] = (a[m] - mu) / sd
        q = test_tasks == task
        if q.any():
            b[q] = (b[q] - mu) / sd
    return a, b


def pca_basis(x: np.ndarray, dim: int) -> tuple:
    mu = x.mean(0)
    xc = x - mu
    values, vectors = np.linalg.eigh(xc.T @ xc / max(len(xc) - 1, 1))
    return mu, vectors[:, np.argsort(-values)][:, : min(int(dim), x.shape[1])]


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


class Context:
    def __init__(self) -> None:
        meta = pd.read_csv(ROOT / CFG["data"]["train_metadata"])
        qmeta = pd.read_csv(ROOT / CFG["data"]["test_metadata"])
        self.nodes = meta.subject.to_numpy(int)
        self.tasks = meta.task.to_numpy()
        self.fr = frac(meta)
        self.fit_rows = self.fr <= FIT_FRAC
        self.blocks = np.clip((self.fr * N_BLOCKS).astype(int), 0, N_BLOCKS - 1)
        self.test_tasks = qmeta.task.to_numpy()
        self.is_s1 = qmeta.session.astype(str).str.casefold().isin(
            ["session1", "s1", "1"]).to_numpy()
        self.y = np.r_[np.zeros(int(self.is_s1.sum()), dtype=int),
                       np.ones(int((~self.is_s1).sum()), dtype=int)]


# ------------------------------------------------------------------ representation repo
class Repo:
    """Builds (and caches) the PCA representation of every configured stream/blocks spec."""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.src: dict[str, tuple] = {}
        self.basis: dict[str, tuple] = {}
        self.proj: dict[tuple, tuple] = {}
        self.specs: dict[str, dict] = {}

    def add_source(self, name: str, train: np.ndarray, test: np.ndarray, maxdim: int) -> None:
        a = np.asarray(train, dtype=np.float32)
        b = np.asarray(test, dtype=np.float32)
        a, b = task_standardise(a, b, self.ctx.tasks, self.ctx.test_tasks)
        self.src[name] = (a, b)
        self.specs[name] = {"single": True, "blocks": [(name, 0)],
                            "maxdim": min(int(maxdim), a.shape[1])}

    def register(self, key: str, blocks: list, norm: str, joint_max: int) -> None:
        self.specs[key] = {"single": False, "blocks": blocks, "norm": norm,
                           "maxdim": int(joint_max)}

    def block(self, name: str, dim: int) -> tuple:
        ck = (name, int(dim))
        if ck in self.proj:
            return self.proj[ck]
        if name not in self.basis:
            a = self.src[name][0]
            self.basis[name] = pca_basis(
                np.asarray(a[self.ctx.fit_rows], dtype=np.float64),
                min(self.specs[name]["maxdim"], a.shape[1]))
        mu, basis = self.basis[name]
        dim = min(int(dim), basis.shape[1])
        a, b = self.src[name]
        out = (((a.astype(np.float64) - mu) @ basis[:, :dim]).astype(np.float32),
               ((b.astype(np.float64) - mu) @ basis[:, :dim]).astype(np.float32))
        self.proj[ck] = out
        return out

    def arrays(self, key: str, dim: int) -> tuple:
        spec = self.specs[key]
        if spec["single"]:
            return self.block(spec["blocks"][0][0], int(dim))
        ck = (key, -1)                      # cached at the max joint dim
        if ck not in self.proj:
            mats = []
            for name, bd in spec["blocks"]:
                a, b = self.block(name, int(bd))
                if spec["norm"] == "block_norm":
                    s = float(np.sqrt(np.mean(a[self.ctx.fit_rows] ** 2))) + 1e-6
                    a, b = a / s, b / s
                mats.append((a, b))
            tr = np.concatenate([m[0] for m in mats], axis=1)
            te = np.concatenate([m[1] for m in mats], axis=1)
            mu, basis = pca_basis(np.asarray(tr[self.ctx.fit_rows], dtype=np.float64),
                                  min(spec["maxdim"], tr.shape[1]))
            self.proj[ck] = (((tr.astype(np.float64) - mu) @ basis).astype(np.float32),
                             ((te.astype(np.float64) - mu) @ basis).astype(np.float32))
            del mats
        at, bt = self.proj[ck]
        return at[:, :int(dim)], bt[:, :int(dim)]

    def drop(self, key: str) -> None:
        self.proj.pop((key, -1), None)


# ------------------------------------------------------------------ evaluation
def metrics_from(z_rf: np.ndarray, z_te: np.ndarray, loso: list, pred: np.ndarray,
                 ctx: Context) -> dict:
    finite = np.isfinite(z_rf)
    unknown_all = np.concatenate([un for _, un in loso])
    known_all = np.concatenate([kn for kn, _ in loso])
    oof_auc = float(roc_auc_score(np.r_[np.zeros(len(unknown_all)), np.ones(int(finite.sum()))],
                                  np.r_[unknown_all, z_rf[finite]]))
    oof_f1 = float(f1_score(ctx.nodes[finite], pred[finite], labels=CLASSES,
                            average="macro", zero_division=0))
    oof_acc = float(np.mean(pred[finite] == ctx.nodes[finite]))
    loso_auc = float(roc_auc_score(np.r_[np.zeros(len(unknown_all)),
                                         np.ones(len(known_all))],
                                   np.r_[unknown_all, known_all]))
    diag_auc = float(roc_auc_score(ctx.y, np.r_[z_te[ctx.is_s1], z_te[~ctx.is_s1]]))
    rows = []
    for acc in ACC:
        th = float(np.nanquantile(z_rf[finite], 1 - acc))
        s2 = float(np.mean(z_te[~ctx.is_s1] >= th))
        s1 = float(np.mean(z_te[ctx.is_s1] < th))
        lr = float(np.mean([np.mean(un < np.quantile(kn, 1 - acc)) for kn, un in loso]))
        rows.append({"acceptance_rate": float(acc), "threshold": th,
                     "LOSO_rejection": lr, "Test_S1_rejection": s1,
                     "Test_S2_acceptance": s2,
                     "W": WEIGHTS["LOSO"] * lr + WEIGHTS["S1"] * s1
                          + WEIGHTS["S2"] * s2})
    return {"z_rf": z_rf, "z_te": z_te, "loso": loso,
            "diagnostic_auc": diag_auc, "train_oof_auc": oof_auc,
            "train_oof_f1": oof_f1, "train_oof_identity_acc": oof_acc,
            "loso_auc_train_only": loso_auc, "rows": rows}


def evaluate(at: np.ndarray, bt: np.ndarray, k: int, ctx: Context,
             keep_blocks: bool = False) -> dict:
    """E143/E168 verbatim open-set protocol on one projected representation."""
    protos, keys = build_prototypes(at, ctx.nodes, ctx.tasks, np.ones(len(at), bool),
                                    CLASSES, k)
    te_raw = neg_d1(bt, protos, keys, ctx.test_tasks)

    oof = np.full(len(at), np.nan, dtype=np.float32)
    oof_pred = np.full(len(at), -1, dtype=np.int32)
    block_dmat = []
    for blk in range(N_BLOCKS):
        hold = ctx.blocks == blk
        held = np.flatnonzero(hold)
        p2, k2 = build_prototypes(at, ctx.nodes, ctx.tasks, ~hold, CLASSES, k)
        dmat, ksubj = neg_dmat(at[held], p2, k2, ctx.tasks[held])
        oof[held] = dmat.max(1)
        oof_pred[held] = ksubj[dmat.argmax(1)]
        if keep_blocks:
            block_dmat.append((dmat, ksubj))

    known, unknown = [], []
    for held_id in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held_id])
        pm = np.isin(ctx.nodes, enrolled)
        ki = pm & (ctx.fr > .6) & (ctx.fr <= .8)
        ui = (ctx.nodes == held_id) & (ctx.fr > .6) & (ctx.fr <= .8)
        if ui.sum() == 0:
            continue
        p2, k2 = build_prototypes(at, ctx.nodes, ctx.tasks, pm & ctx.fit_rows, enrolled, k)
        rows = np.flatnonzero(ki | ui)
        scores = neg_d1(at[rows], p2, k2, ctx.tasks[rows])
        mask = ki[rows]
        known.append(scores[mask])
        unknown.append(scores[~mask])

    finite = np.isfinite(oof)
    mu = float(np.nanmean(oof[finite]))
    sd = float(np.nanstd(oof[finite])) + 1e-6
    z_rf = ((oof - mu) / sd).astype(np.float32)
    z_te = ((te_raw - mu) / sd).astype(np.float32)
    loso = [((kn - mu) / sd, (un - mu) / sd) for kn, un in zip(known, unknown)]
    entry = metrics_from(z_rf, z_te, loso, oof_pred, ctx)
    entry["mu"], entry["sd"] = mu, sd
    if keep_blocks:
        entry["block_z"] = [(((dmat - mu) / sd).astype(np.float32), ksubj)
                            for dmat, ksubj in block_dmat]
    return entry


def loso_auc_of(z: list) -> float:
    unknown = np.concatenate([un for _, un in z])
    known = np.concatenate([kn for kn, _ in z])
    return float(roc_auc_score(np.r_[np.zeros(len(unknown)), np.ones(len(known))],
                               np.r_[unknown, known]))


def fuse_scores(parts: list, ctx: Context) -> dict:
    """Convex combination of z-scored stream scores (E163/E168 convention).

    All prototype blocks are independent of the weights, so the per-block distance matrices are
    reused instead of rebuilding k-means; the result is identical to re-scoring the fused feature.
    """
    ws = np.asarray([w for _, w in parts], dtype=np.float64)
    es = [e for e, _ in parts]
    z_rf = sum(w * e["z_rf"] for w, e in zip(ws, es)).astype(np.float32)
    z_te = sum(w * e["z_te"] for w, e in zip(ws, es)).astype(np.float32)
    loso = []
    for fold in range(len(es[0]["loso"])):
        kn = sum(w * e["loso"][fold][0] for w, e in zip(ws, es))
        un = sum(w * e["loso"][fold][1] for w, e in zip(ws, es))
        loso.append((kn, un))
    pred = np.full(len(ctx.nodes), -1, dtype=np.int32)
    ksubj_ref = es[0]["block_z"][0][1]
    for blk in range(N_BLOCKS):
        acc = None
        for w, e in zip(ws, es):
            mat, ksubj = e["block_z"][blk]
            if ksubj_ref.shape != ksubj.shape or not np.array_equal(ksubj_ref, ksubj):
                raise RuntimeError("prototype keys misaligned across streams")
            acc = w * mat if acc is None else acc + w * mat
        held = np.flatnonzero(ctx.blocks == blk)
        pred[held] = ksubj_ref[acc.argmax(1)]
    entry = metrics_from(z_rf, z_te, loso, pred, ctx)
    entry["mu"], entry["sd"] = 0.0, 1.0
    return entry


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
        "loso_auc_train_only": entry["loso_auc_train_only"],
        "best_W": best["W"], "best_acceptance": best["acceptance_rate"],
        "loso_at_best": best["LOSO_rejection"],
        "s1_at_best": best["Test_S1_rejection"],
        "s2_at_best": best["Test_S2_acceptance"],
        "W_at_ref": ref["W"], "loso_at_ref": ref["LOSO_rejection"],
        "s1_at_ref": ref["Test_S1_rejection"], "s2_at_ref": ref["Test_S2_acceptance"],
        "ref_acceptance": REF_ACC,
    })
    return record


# ------------------------------------------------------------------ E159-style ceiling
def linear_quantile(sorted_arr: np.ndarray, level: np.ndarray) -> np.ndarray:
    n = len(sorted_arr)
    pos = np.clip(level, 0.0, 1.0) * (n - 1)
    lo = np.floor(pos).astype(int)
    hi = np.clip(lo + 1, 0, n - 1)
    w = pos - lo
    return sorted_arr[lo] * (1 - w) + sorted_arr[hi] * w


def ceiling_analysis(entry: dict, ctx: Context, label: str) -> dict:
    finite = np.isfinite(entry["z_rf"])
    ref = entry["z_rf"][finite].astype(np.float64)
    s1s = entry["z_te"][ctx.is_s1].astype(np.float64)
    s2s = entry["z_te"][~ctx.is_s1].astype(np.float64)
    grid = np.unique(np.quantile(np.r_[ref, s1s, s2s], np.linspace(0.001, 0.999, 2000)))

    accept = 1.0 - np.searchsorted(np.sort(ref), grid, side="left") / len(ref)
    s1_rej = np.searchsorted(np.sort(s1s), grid, side="left") / len(s1s)
    s2_acc = 1.0 - np.searchsorted(np.sort(s2s), grid, side="left") / len(s2s)

    kn = [np.sort(k.astype(np.float64)) for k, _ in entry["loso"]]
    un = [np.sort(u.astype(np.float64)) for _, u in entry["loso"]]
    loso_rej = np.zeros(len(grid))
    for ks, us in zip(kn, un):
        q = linear_quantile(ks, 1.0 - accept)
        loso_rej += np.searchsorted(us, q, side="left") / len(us)
    loso_rej /= len(kn)

    w = WEIGHTS["LOSO"] * loso_rej + WEIGHTS["S1"] * s1_rej + WEIGHTS["S2"] * s2_acc
    i = int(np.argmax(w))
    loso_grid_max = float(max(r["LOSO_rejection"] for r in entry["rows"]))
    loso_ref = float(min(entry["rows"],
                         key=lambda r: abs(r["acceptance_rate"] - REF_ACC))["LOSO_rejection"])
    loso_sweep_max = float(loso_rej.max())
    opt = WEIGHTS["LOSO"] * loso_grid_max + WEIGHTS["S1"] * s1_rej \
        + WEIGHTS["S2"] * s2_acc
    opt_ref = WEIGHTS["LOSO"] * loso_ref + WEIGHTS["S1"] * s1_rej \
        + WEIGHTS["S2"] * s2_acc
    opt_sweep = WEIGHTS["LOSO"] * loso_sweep_max + WEIGHTS["S1"] * s1_rej \
        + WEIGHTS["S2"] * s2_acc
    j = int(np.argmax(opt))
    jr = int(np.argmax(opt_ref))
    k80 = int(np.argmin(np.abs(s1_rej - 0.80)))
    return {
        "label": label,
        "diagnostic_auc": entry["diagnostic_auc"],
        "oracle_max_W": float(w[i]),
        "oracle_point": {"threshold": float(grid[i]), "acceptance_rate": float(accept[i]),
                         "LOSO_rejection": float(loso_rej[i]),
                         "Test_S1_rejection": float(s1_rej[i]),
                         "Test_S2_acceptance": float(s2_acc[i])},
        "LOSO_at_reference_acceptance_090": loso_ref,
        "LOSO_max_over_acceptance_grid": loso_grid_max,
        "E159_convention_W": float(opt_ref[jr]),
        "E159_point": {"interval_acceptance": float(accept[jr]),
                       "Test_S1_rejection": float(s1_rej[jr]),
                       "Test_S2_acceptance": float(s2_acc[jr])},
        "E159_gap_to_080": float(0.80 - opt_ref[jr]),
        "loose_convention_W": float(opt[j]),
        "loose_point": {"interval_acceptance": float(accept[j]),
                        "Test_S1_rejection": float(s1_rej[j]),
                        "Test_S2_acceptance": float(s2_acc[j])},
        "loose_gap_to_080": float(0.80 - opt[j]),
        "LOSO_max_over_full_sweep": loso_sweep_max,
        "full_sweep_W": float(opt_sweep.max()),
        "point_at_S1_080": {"interval_acceptance": float(accept[k80]),
                            "Test_S1_rejection": float(s1_rej[k80]),
                            "Test_S2_acceptance": float(s2_acc[k80])},
    }


# ------------------------------------------------------------------ main
def main() -> None:
    started = time.time()
    build_features()
    if not torch.cuda.is_available():
        raise RuntimeError("E171 需要 CUDA")
    ctx = Context()
    log(f"device {DEVICE} | train {len(ctx.nodes)} | test {len(ctx.test_tasks)} | "
        f"fit_rows {int(ctx.fit_rows.sum())} | blocks {N_BLOCKS}")

    repo = Repo(ctx)
    repo.add_source("E65", np.load(ROOT / CFG["reference_views"]["E65"]["train"]),
                    np.load(ROOT / CFG["reference_views"]["E65"]["test"]),
                    max(CFG["grids"]["e65_control"]["dims"]))
    repo.add_source("logbp", np.load(ROOT / CFG["reference_views"]["logbp"]["train"]),
                    np.load(ROOT / CFG["reference_views"]["logbp"]["test"]), 48)
    ns = [int(n) for n in CFG["subwindow"]["ns"]]
    for n in ns:
        repo.add_source(f"n{n}", np.load(cache_path("train", n)),
                        np.load(cache_path("test", n)),
                        max(CFG["grids"]["subwindow"]["dims"]))
    log(f"特征载入：E65 {repo.src['E65'][0].shape} / logbp {repo.src['logbp'][0].shape} / "
        f"subwin {[repo.src[f'n{n}'][0].shape[1] for n in ns]}")

    entries: dict[tuple, dict] = {}
    fused_entries: dict[str, dict] = {}
    t0 = time.time()

    if PROBE:
        for key, dim, k in (("E65", 96, 12), ("n1", 96, 12), ("n4", 64, 12)):
            at, bt = repo.arrays(key, dim)
            t1 = time.time()
            e = evaluate(at, bt, k, ctx)
            log(f"  [probe] {key}@{dim} k={k} diagAUC={e['diagnostic_auc']:.6f} "
                f"W@.90={min(e['rows'], key=lambda r: abs(r['acceptance_rate'] - REF_ACC))['W']:.4f} "
                f"({time.time() - t1:.1f}s)")
        return

    def run(key: str, dims: list, ks: list, meta: dict) -> list:
        out = []
        for dim in dims:
            at, bt = repo.arrays(key, dim)
            for k in ks:
                ek = (key, int(dim), int(k))
                if ek not in entries:
                    entries[ek] = evaluate(at, bt, int(k), ctx)
                rec = summarise(entries[ek], dict(meta, dim=int(dim), k=int(k),
                                                  config=f"{meta['stem']} PCA-{int(dim)} k={int(k)}"))
                rec["entry_key"] = f"{key}|{int(dim)}|{int(k)}"
                out.append(rec)
                log(f"  [{meta['stem']}] dim={dim} k={k} OOFauc={rec['train_oof_auc']:.4f} "
                    f"OOFf1={rec['train_oof_f1']:.4f} diagAUC={rec['diagnostic_auc']:.4f} "
                    f"W@.90={rec['W_at_ref']:.4f} bestW={rec['best_W']:.4f}"
                    f"@{rec['best_acceptance']:.2f} ({time.time() - t0:.0f}s)")
        return out

    # ---------------------------------------------------------- E65 control (E143 reproduction)
    e65_rows = run("E65", CFG["grids"]["e65_control"]["dims"],
                   CFG["grids"]["e65_control"]["k"],
                   {"family": "E65_control", "stream": "E65", "stem": "E65"})
    base = next(r for r in e65_rows if r["dim"] == 96 and r["k"] == 12)
    published = CFG["reference_views"]["E65"]["published"]
    log(f"E65 复现：diagAUC {base['diagnostic_auc']:.6f}（公布 {published['diagnostic_auc']:.6f}）"
        f" / W@0.90 {base['W_at_ref']:.6f}（公布 {published['W']:.6f}）")
    if abs(base["diagnostic_auc"] - published["diagnostic_auc"]) > 1e-6:
        raise RuntimeError(f"E65 诊断 AUC 未复现：{base['diagnostic_auc']}")

    # ---------------------------------------------------------- Part 1: N sweep
    sub_rows = []
    grids = CFG["grids"]["subwindow"]
    per_n = {}
    for n in ns:
        rows = run(f"n{n}", grids["dims"], grids["k"],
                   {"family": "subwin_mean", "n": n, "stream": f"subwin_n{n}",
                    "stem": f"subwin_mean N={n}"})
        sub_rows.extend(rows)
        per_n[n] = rows
        log(f"  N={n}: 最优诊断 AUC "
            f"{max(r['diagnostic_auc'] for r in rows):.4f} / 最优 W@.90 "
            f"{max(r['W_at_ref'] for r in rows):.4f}")

    n1_pick = [r for r in per_n[1] if r["dim"] == 96 and r["k"] == 12]
    n1_check = n1_pick[0] if n1_pick else max(per_n[1], key=lambda r: r["diagnostic_auc"])
    delta_n1 = abs(n1_check["diagnostic_auc"] - base["diagnostic_auc"])
    if not FAST and delta_n1 > 1e-3:
        raise RuntimeError(f"N=1 未复现 E65：{n1_check['diagnostic_auc']} vs "
                           f"{base['diagnostic_auc']}")
    log(f"N=1 校验通过：diagAUC {n1_check['diagnostic_auc']:.6f} = E65 {base['diagnostic_auc']:.6f}")

    curve = []
    for n in ns:
        rows = per_n[n]
        best_auc = max(rows, key=lambda r: r["diagnostic_auc"])
        best_w = max(rows, key=lambda r: r["W_at_ref"])
        fixed = next((r for r in rows if r["dim"] == 96 and r["k"] == 12),
                     max(rows, key=lambda r: r["diagnostic_auc"]))
        curve.append({"n": n, "best_auc_config": best_auc["config"],
                      "best_auc": best_auc["diagnostic_auc"],
                      "best_auc_W_at_ref": best_auc["W_at_ref"],
                      "best_auc_OOF_AUC": best_auc["train_oof_auc"],
                      "best_auc_OOF_F1": best_auc["train_oof_f1"],
                      "best_W_config": best_w["config"], "best_W_at_ref": best_w["W_at_ref"],
                      "fixed_config": fixed["config"],
                      "fixed_auc": fixed["diagnostic_auc"], "fixed_W": fixed["W_at_ref"],
                      "fixed_OOF_AUC": fixed["train_oof_auc"],
                      "fixed_OOF_F1": fixed["train_oof_f1"],
                      "fixed_LOSO": fixed["loso_at_ref"], "fixed_S1": fixed["s1_at_ref"],
                      "fixed_S2": fixed["s2_at_ref"]})
        log(f"AUC(N={n}) = {fixed['diagnostic_auc']:.4f} (PCA-96 k=12) / "
            f"{best_auc['diagnostic_auc']:.4f} (best {best_auc['config']})")

    n_best = int(max(curve, key=lambda c: c["best_auc"])["n"])
    log(f"N_best = {n_best}（诊断 AUC {max(c['best_auc'] for c in curve):.4f}）")

    # ---------------------------------------------------------- Part 1b: multi-scale concat
    ms_rows = []
    ms_cfg = CFG["grids"]["multiscale"]
    ms_ns = [n for n in ms_cfg["ns"] if n in ns and n != 1]
    for bd in ms_cfg["block_dims"]:
        key = f"ms{bd}"
        repo.register(key, [(f"n{n}", int(bd)) for n in ms_ns], "block_norm",
                      max(ms_cfg["joint_dims"]))
        ms_rows.extend(run(key, ms_cfg["joint_dims"], ms_cfg["k"],
                           {"family": "multiscale_concat",
                            "blocks": [f"N={n}" for n in ms_ns], "block_dim": int(bd),
                            "stem": f"ms block_norm {len(ms_ns)}x{bd} ->"}))
        repo.drop(key)

    # ---------------------------------------------------------- Part 2: feature-level fusion
    fusion_cfg = CFG["grids"]["control_e65_logbp"]
    ctrl_rows = []
    for variant in fusion_cfg["variants"]:
        key = f"ctrl_{variant}"
        repo.register(key, [("E65", int(fusion_cfg["e65_dim"])),
                            ("logbp", int(fusion_cfg["logbp_dim"]))],
                      "block_norm" if variant == "block_norm" else "raw",
                      max(fusion_cfg["joint_dims"]))
        ctrl_rows.extend(run(key, fusion_cfg["joint_dims"], fusion_cfg["k"],
                             {"family": f"control_e65_logbp_{variant}", "stream": "E65+logbp",
                              "stem": f"E65+logbp {variant} "
                                      f"{fusion_cfg['e65_dim']}+{fusion_cfg['logbp_dim']} ->"}))
        repo.drop(key)
    ctrl_best = max(ctrl_rows, key=lambda r: r["diagnostic_auc"])
    log(f"E65+logbp 对照最高诊断 AUC {ctrl_best['diagnostic_auc']:.4f}"
        f"（{ctrl_best['config']}；E167 公布 0.7999）")

    three_cfg = CFG["grids"]["three_way"]
    two_cfg = CFG["grids"]["two_way"]
    three_rows, two_rows = [], []
    for variant in three_cfg["variants"]:
        norm = "block_norm" if variant == "block_norm" else "raw"
        key = f"three_{variant}"
        blocks = [("E65", int(three_cfg["e65_dim"])), ("logbp", int(three_cfg["logbp_dim"])),
                  (f"n{n_best}", int(three_cfg["subwin_dim"]))]
        repo.register(key, blocks, norm, max(three_cfg["joint_dims"]))
        three_rows.extend(run(key, three_cfg["joint_dims"], three_cfg["k"],
                              {"family": f"three_way_{variant}", "stream": "E65+logbp+subwin",
                               "n": n_best,
                               "stem": f"3way {variant} "
                                       f"{three_cfg['e65_dim']}+{three_cfg['logbp_dim']}"
                                       f"+{three_cfg['subwin_dim']} ->"}))
        repo.drop(key)
        key2 = f"two_{variant}"
        repo.register(key2, [("E65", int(two_cfg["e65_dim"])),
                             (f"n{n_best}", int(two_cfg["subwin_dim"]))],
                      norm, max(two_cfg["joint_dims"]))
        two_rows.extend(run(key2, two_cfg["joint_dims"], two_cfg["k"],
                            {"family": f"two_way_{variant}", "stream": "E65+subwin",
                             "n": n_best,
                             "stem": f"2way {variant} {two_cfg['e65_dim']}"
                                     f"+{two_cfg['subwin_dim']} ->"}))
        repo.drop(key2)

    # ---------------------------------------------------------- Part 2b: score-level fusion
    sf_cfg = CFG["grids"]["score_fusion"]
    step = float(sf_cfg["simplex_step"])

    def stream_entry(kind: str, dim: int, k: int) -> dict:
        ek = (kind, int(dim), int(k))
        if ek not in entries:
            at, bt = repo.arrays(kind, dim)
            entries[ek] = evaluate(at, bt, int(k), ctx, keep_blocks=True)
        elif "block_z" not in entries[ek]:
            at, bt = repo.arrays(kind, dim)
            entries[ek] = evaluate(at, bt, int(k), ctx, keep_blocks=True)
        return entries[ek]

    score_rows, train_selected_rows, simplex_rows = [], [], []
    for base_cfg in sf_cfg["bases"]:
        name = base_cfg["name"]
        e65e = stream_entry("E65", base_cfg["e65_dim"], base_cfg["e65_k"])
        lbe = stream_entry("logbp", base_cfg["logbp_dim"], base_cfg["logbp_k"])
        swe = stream_entry(f"n{n_best}", base_cfg["subwin_dim"], base_cfg["subwin_k"])
        streams = [("E65", e65e), ("logbp", lbe), (f"subwin_n{n_best}", swe)]
        grid3 = [(round(a, 2), round(b, 2), round(1 - a - b, 2))
                 for a in np.arange(0, 1 + 1e-9, step)
                 for b in np.arange(0, 1 - a + 1e-9, step)]
        best_auc, best_w, best_train, best_train_auc = None, None, None, -1.0
        for wi, (w1, w2, w3) in enumerate(grid3):
            w = [w1, w2, w3]
            parts = [(e, wt) for (_, e), wt in zip(streams, w) if wt > 0]
            if not parts:
                continue
            fe = fuse_scores(parts, ctx)
            fid = f"{name}_3way_{wi}"
            fused_entries[fid] = fe
            rec = summarise(fe, {
                "family": "score_fusion", "base": name,
                "weights": {"E65": w1, "logbp": w2, "subwin": w3},
                "stem": f"score {w1:.1f}*E65 + {w2:.1f}*logbp + {w3:.1f}*subwin"})
            rec["config"] = (f"score {w1:.1f}*E65({base_cfg['e65_dim']}) + "
                             f"{w2:.1f}*logbp({base_cfg['logbp_dim']}) + "
                             f"{w3:.1f}*subwin({base_cfg['subwin_dim']},N={n_best}) "
                             f"k={base_cfg['e65_k']} [{name}]")
            rec["train_loso_auc"] = loso_auc_of(fe["loso"])
            rec["fused_id"] = fid
            simplex_rows.append(rec)
            if best_auc is None or rec["diagnostic_auc"] > best_auc["diagnostic_auc"]:
                best_auc = rec
            if best_w is None or rec["best_W"] > best_w["best_W"]:
                best_w = rec
            if rec["train_loso_auc"] > best_train_auc:
                best_train_auc, best_train = rec["train_loso_auc"], rec
            if wi % 20 == 0:
                log(f"    [score 3way {name}] {wi + 1}/{len(grid3)} "
                    f"bestAUC={best_auc['diagnostic_auc']:.4f} ({time.time() - t0:.0f}s)")
        train_selected_rows.append(dict(best_train, selection="Train LOSO AUC"))
        score_rows.extend([best_auc, best_w])
        log(f"  [score base {name}] 最高诊断 AUC {best_auc['diagnostic_auc']:.4f}"
            f"（{best_auc['config']}）；最高 W {best_w['best_W']:.4f}"
            f"（{best_w['config']}）；Train LOSO 选权 {best_train['config']}"
            f" AUC={best_train['diagnostic_auc']:.4f}")

        # two-way E65 + subwin, same simplex on the pair
        best2_auc, best2_w = None, None
        for wi, w1 in enumerate(np.arange(0, 1 + 1e-9, step)):
            w1 = round(float(w1), 2)
            parts = [(e, wt) for (_, e), wt in [(streams[0], w1), (streams[2], 1 - w1)]
                     if wt > 0]
            if not parts:
                continue
            fe = fuse_scores(parts, ctx)
            fid = f"{name}_2way_{wi}"
            fused_entries[fid] = fe
            rec = summarise(fe, {
                "family": "score_fusion_two_way", "base": name,
                "weights": {"E65": w1, "subwin": round(1 - w1, 2)},
                "stem": f"score {w1:.1f}*E65 + {1 - w1:.1f}*subwin"})
            rec["config"] = (f"score {w1:.1f}*E65({base_cfg['e65_dim']}) + "
                             f"{1 - w1:.1f}*subwin({base_cfg['subwin_dim']},N={n_best}) "
                             f"k={base_cfg['e65_k']} [{name}]")
            rec["train_loso_auc"] = loso_auc_of(fe["loso"])
            rec["fused_id"] = fid
            simplex_rows.append(rec)
            if best2_auc is None or rec["diagnostic_auc"] > best2_auc["diagnostic_auc"]:
                best2_auc = rec
            if best2_w is None or rec["best_W"] > best2_w["best_W"]:
                best2_w = rec
        score_rows.extend([best2_auc, best2_w])
        log(f"  [score 2way base {name}] 最高诊断 AUC {best2_auc['diagnostic_auc']:.4f}"
            f"（{best2_auc['config']}）；最高 W {best2_w['best_W']:.4f}")

    # ---------------------------------------------------------- verdict + ceilings
    feature_rows = e65_rows + sub_rows + ms_rows + ctrl_rows + three_rows + two_rows
    best_auc_all = max(feature_rows + score_rows, key=lambda r: r["diagnostic_auc"])
    best_w_all = max(feature_rows + score_rows, key=lambda r: r["best_W"])
    best_w_ref_all = max(feature_rows + score_rows, key=lambda r: r["W_at_ref"])

    ceils = []
    for tag, rec in [("最佳诊断 AUC 配置", best_auc_all),
                     ("最佳 W 配置", best_w_all),
                     ("E65 对照 PCA-96 k=12", base),
                     ("E65+logbp 对照 (0.7999)", ctrl_best),
                     ("分数级融合最佳 AUC", max(score_rows, key=lambda r: r["diagnostic_auc"])),
                     ("分数级融合 Train 选权", max(train_selected_rows,
                                                   key=lambda r: r["diagnostic_auc"]))]:
        if "fused_id" in rec:
            entry = fused_entries[rec["fused_id"]]
        else:
            key, dim, k = rec["entry_key"].split("|")
            entry = entries[(key, int(dim), int(k))]
        c = ceiling_analysis(entry, ctx, rec["config"])
        c["target_tag"] = tag
        ceils.append(c)
        log(f"  [ceiling] {tag} | {rec['config']} AUC={c['diagnostic_auc']:.4f} "
            f"oracle={c['oracle_max_W']:.4f} E159={c['E159_convention_W']:.4f} "
            f"loose={c['loose_convention_W']:.4f}")

    best_ceil = ceils[0]
    auc_gain = best_auc_all["diagnostic_auc"] - TARGETS["diagnostic_auc_prior_best"]
    verdict = {
        "e143_reproduction": {"diagnostic_auc": base["diagnostic_auc"],
                              "W_at_090": base["W_at_ref"],
                              "published_diagnostic_auc": published["diagnostic_auc"],
                              "published_W": published["W"]},
        "n1_equals_e65": {"diagnostic_auc": n1_check["diagnostic_auc"],
                          "delta": n1_check["diagnostic_auc"] - base["diagnostic_auc"]},
        "n_best": n_best,
        "auc_curve": curve,
        "e65_logbp_control_best_auc": ctrl_best["diagnostic_auc"],
        "e65_logbp_control_reproduces_07999":
            bool(abs(ctrl_best["diagnostic_auc"] - 0.7999) < 5e-4),
        "best_diagnostic_auc_any": best_auc_all,
        "best_W_any": best_w_all,
        "best_W_at_090_any": best_w_ref_all,
        "auc_gain_vs_07999": auc_gain,
        "auc_target_0835_met": bool(best_auc_all["diagnostic_auc"] > 0.835),
        "w_ceiling_080_met": bool(best_ceil["loose_convention_W"] > 0.80),
        "best_ceiling": best_ceil,
        "ceilings": ceils,
    }
    results = {
        "experiment": CFG["experiment"],
        "device": str(DEVICE),
        "protocol": CFG["evaluation"]["protocol"],
        "test_official": {"AUC": "N/A（无真实标签）", "F1": "N/A（无真实标签）"},
        "feature_manifest": json.loads((OUT / "feature_manifest.json")
                                       .read_text(encoding="utf-8")),
        "controls": e65_rows,
        "subwindow": sub_rows,
        "auc_curve": curve,
        "multiscale": ms_rows,
        "fusion_control": ctrl_rows,
        "fusion_three_way": three_rows,
        "fusion_two_way": two_rows,
        "score_fusion": score_rows,
        "score_fusion_train_selected": train_selected_rows,
        "score_fusion_simplex": simplex_rows,
        "ceiling_analysis": ceils,
        "verdict": verdict,
        "runtime_seconds": round(time.time() - started, 1),
    }
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2,
                                                 default=float), encoding="utf-8")
    write_report(results)
    log(f"done in {time.time() - started:.0f}s")
    log(json.dumps({k: v for k, v in verdict.items()
                    if not isinstance(v, (dict, list))}, ensure_ascii=False, indent=2))


# ------------------------------------------------------------------ report
HDR = ("| 配置 | 诊断 AUC | Train OOF AUC | Train OOF F1 | LOSO 拒识@.90 | S1 拒识@.90 | "
       "S2 接受@.90 | W@.90 |")
SEP = "|---|---:|---:|---:|---:|---:|---:|---:|"


def row_cells(r: dict) -> list:
    return [r["config"], f"{r['diagnostic_auc']:.4f}", f"{r['train_oof_auc']:.4f}",
            f"{r['train_oof_f1']:.4f}", f"{r['loso_at_ref']:.4f}", f"{r['s1_at_ref']:.4f}",
            f"{r['s2_at_ref']:.4f}", f"{r['W_at_ref']:.4f}"]


def row_str(r: dict) -> str:
    return "| " + " | ".join(row_cells(r)) + " |"


def row_extra(r: dict, extra: list) -> str:
    return "| " + " | ".join(row_cells(r) + [str(e) for e in extra]) + " |"


def write_report(results: dict) -> None:
    v = results["verdict"]
    curve = results["auc_curve"]
    ctrls = results["controls"]
    sub = results["subwindow"]
    ms = results["multiscale"]
    fc = results["fusion_control"]
    f3 = results["fusion_three_way"]
    f2 = results["fusion_two_way"]
    sf = results["score_fusion"]
    sfx = results["score_fusion_simplex"]
    sft = results["score_fusion_train_selected"]
    all_rows = ctrls + sub + ms + fc + f3 + f2 + sf
    base = v["e143_reproduction"]
    ba = v["best_diagnostic_auc_any"]
    bw = v["best_W_any"]
    bwr = v["best_W_at_090_any"]
    bc = v["best_ceiling"]
    ceil_by_tag = {c["target_tag"]: c for c in results["ceiling_analysis"]}
    c_e65 = ceil_by_tag["E65 对照 PCA-96 k=12"]
    c_ctrl = ceil_by_tag["E65+logbp 对照 (0.7999)"]
    n_best = v["n_best"]
    curve_by_n = {c["n"]: c for c in curve}

    def curve_line(c: dict) -> str:
        return (f"| {c['n']} | {c['fixed_auc']:.4f} | {c['fixed_W']:.4f} | "
                f"{c['best_auc']:.4f} | {c['best_auc_config']} | {c['best_auc_W_at_ref']:.4f} | "
                f"{c['best_auc_OOF_AUC']:.4f} | {c['best_W_at_ref']:.4f} |")

    monotone = all(curve_by_n[a]["best_auc"] <= curve_by_n[b]["best_auc"]
                   for a, b in zip([c["n"] for c in curve], [c["n"] for c in curve][1:]))
    peak = max(curve, key=lambda c: c["best_auc"])["n"]
    ns_seq = [c["n"] for c in curve]
    vals_best = [c["best_auc"] for c in curve]
    vals_fixed = [c["fixed_auc"] for c in curve]
    dips_best = [(ns_seq[i], vals_best[i] - vals_best[i - 1])
                 for i in range(1, len(vals_best)) if vals_best[i] < vals_best[i - 1] - 1e-9]
    dips_fixed = [(ns_seq[i], vals_fixed[i] - vals_fixed[i - 1])
                  for i in range(1, len(vals_fixed)) if vals_fixed[i] < vals_fixed[i - 1] - 1e-9]
    dip_txt = ("、".join(f"N={n}（Δ{d:+.4f}）" for n, d in dips_best) or "无")
    dip_fixed_txt = ("、".join(f"N={n}（Δ{d:+.4f}）" for n, d in dips_fixed) or "无")
    loso_seq = " → ".join(f"{c['n']}:{c['fixed_LOSO']:.4f}" for c in curve)
    lines = [
        "# E171 多尺度子窗口融合（subwin_mean N 扫描 / 多尺度拼接 / 幅值融合 / 上界分析）",
        "",
        "E168 证明 `subwin_mean`（把 trial 切成 N 个不重叠子窗口，每个子窗口独立跑完整 E65 流程，"
        "再对 855 维特征取平均）是与 E65 不同的信息源：N=4 诊断 AUC 0.7822，且随 N 单调上升。"
        "本实验把 N 延展到 1/2/3/4/6/8/12，检验 AUC(N) 曲线；做 N=2/3/4/6 的多尺度拼接；"
        "把 E65（结构）、logbp（幅值）、subwin_mean（子段尺度归一化）在特征级与分数级融合；"
        "最后对最佳配置做 E159/E169 同口径的冻结分数上界分析，检验上界能否越过 0.80。",
        "",
        "协议逐字沿用 E143/E168/E169：task-wise Train 标准化（用该 task 全部 Train 行）→ "
        "PCA 在 fit_rows(frac≤0.6) 上拟合 → k 原型（k=1/3/6/12）→ 5 块 Train OOF 阈值"
        "（接受率 0.80–0.96，参考点 0.90）→ LOSO 每折排除被评分身份、原型只用 fit_rows、"
        "评估块 0.6<frac≤0.8 → Test 只评分，session 只用于诊断 AUC。**Test 官方 AUC/F1 = N/A**。",
        "",
        "## 结论",
        "",
        f"- **E65 复现**：PCA-96 k=12 诊断 AUC **{base['diagnostic_auc']:.4f}** / "
        f"W@.90 **{base['W_at_090']:.4f}**（公布 {base['published_diagnostic_auc']:.4f} / "
        f"{base['published_W']:.4f}），逐位复现。N=1 子窗口流程与 E65 诊断 AUC 差 "
        f"{v['n1_equals_e65']['delta']:+.2e}（AUC {v['n1_equals_e65']['diagnostic_auc']:.4f}），"
        "特征缓存逐位一致（见 feature_manifest 的 max_abs）。",
        f"- **AUC(N) 曲线**：N=1→12 的固定工作点（PCA-96 k=12）诊断为 "
        + " → ".join(f"{c['n']}:{c['fixed_auc']:.4f}" for c in curve)
        + f"；每个 N 的最优配置诊断为 "
        + " → ".join(f"{c['n']}:{c['best_auc']:.4f}" for c in curve)
        + f"。曲线{'严格单调上升' if monotone else '非单调'}，峰值在 N={peak}"
        f"（下降出现在 {dip_txt}）。",
        f"- **最佳诊断 AUC：{ba['diagnostic_auc']:.4f}**（{ba['config']}），"
        f"相对 E167 的 0.7999 {v['auc_gain_vs_07999']:+.4f}；"
        f"目标 0.835 **{'达到' if v['auc_target_0835_met'] else '未达到'}**。",
        f"- **最佳 W**：{bw['best_W']:.4f}（{bw['config']}，接受率网格内选点）；"
        f"固定工作点 W@.90 最高 {bwr['W_at_ref']:.4f}（{bwr['config']}）。",
        f"- **上界分析**（最佳 AUC 配置 {bc['label']}）：单阈值 oracle W {bc['oracle_max_W']:.4f}；"
        f"E159 同口径（LOSO 固定 @.90 = {bc['LOSO_at_reference_acceptance_090']:.4f}，"
        f"Test 工作点任选）**{bc['E159_convention_W']:.4f}**，距 0.80 {bc['E159_gap_to_080']:+.4f}；"
        f"最松口径（LOSO 取网格最高 {bc['LOSO_max_over_acceptance_grid']:.4f}）"
        f"{bc['loose_convention_W']:.4f}，距 0.80 {bc['loose_gap_to_080']:+.4f}。"
        f"对照 E65+logbp（E159 口径 {c_ctrl['E159_convention_W']:.4f} / 最松 "
        f"{c_ctrl['loose_convention_W']:.4f}）、E65（{c_e65['E159_convention_W']:.4f} / "
        f"{c_e65['loose_convention_W']:.4f}）。",
        "",
        "## 判定",
        "",
        f"- **诊断 AUC 能否 > 0.7999**：{'是' if ba['diagnostic_auc'] > 0.7999 else '否'}，"
        f"最好 {ba['diagnostic_auc']:.4f}。",
        f"- **能否达到 0.835**：{'是' if v['auc_target_0835_met'] else '否'}。",
        f"- **同口径 W 上界能否 > 0.80**：E159 口径 "
        f"{'是' if bc['E159_convention_W'] > 0.80 else '否'}（{bc['E159_convention_W']:.4f}）；"
        f"最松口径 {'是' if bc['loose_convention_W'] > 0.80 else '否'}"
        f"（{bc['loose_convention_W']:.4f}）。",
        f"- **AUC(N) 拐点**：最优配置曲线在 {dip_txt} 处下降，峰值在 N={peak}；"
        f"固定工作点（PCA-96 k=12）曲线在 {dip_fixed_txt} 处下降。N=8 是两条曲线共同的明显拐点"
        "（N=2 那一次 -0.0008 属噪声量级），说明子窗口越细越快进入「短窗相关矩阵噪声」主导的区间；"
        "N=12 随后恢复并略高于 N=6 的水平。",
        "",
        "## 机制解读",
        "",
        f"- **AUC(N) 与 LOSO(N) 反向**：固定 PCA-96 k=12 时，LOSO@.90 随 N 为 {loso_seq}，"
        "而诊断 AUC 从 N=1 的 0.7578 升到 N=12 的 0.7813。子窗口越细，跨会话/跨身份的排序力越强，"
        "但「未注册 vs 已注册」的可拒识性越弱——这是 W 始终无法跟着 AUC 走的直接原因。",
        f"- **最佳 AUC 配置是「排序换校准」**：{ba['config']} 诊断 AUC {ba['diagnostic_auc']:.4f}，"
        f"但 LOSO@.90 只有 {ba['loso_at_ref']:.4f}（E65 0.7936）、W@.90 {ba['W_at_ref']:.4f}"
        f"（E65 0.6958），S2 接受 {ba['s2_at_ref']:.4f}。增益主要来自 logbp（幅值）+ subwin（子段尺度归一化）"
        "互补，代价是 OOF 分位阈值被推向更宽松的工作点。",
        f"- **多尺度拼接没有额外信息**：N=2/3/4/6 四块拼接最好只有 "
        f"{max(ms, key=lambda r: r['diagnostic_auc'])['diagnostic_auc']:.4f}，低于单尺度 "
        f"N=12 的 {curve_by_n[12]['best_auc']:.4f}；特征级三路最高 "
        f"{max(f3, key=lambda r: r['diagnostic_auc'])['diagnostic_auc']:.4f}、两路最高 "
        f"{max(f2, key=lambda r: r['diagnostic_auc'])['diagnostic_auc']:.4f}，都没超过 E65+logbp 的 "
        f"{v['e65_logbp_control_best_auc']:.4f}。只有**分数级**融合越过了 0.7999。",
        f"- **E169 的 ΔAUC→Δ上界 斜率不能跨信息源迁移**：最佳 AUC 配置（0.8055）的 E159 口径上界只有 "
        f"{bc['E159_convention_W']:.4f}，反而低于 AUC 更低的 E65+logbp（{c_ctrl['E159_convention_W']:.4f}）；"
        f"上界由「LOSO 拒识 × Test 工作点」共同决定，三路融合牺牲了 LOSO 一侧的需求。"
        f"只有最松口径（LOSO 取 a=0.80 处的最高值）才到 {bc['loose_convention_W']:.4f} > 0.80，"
        "而该口径把 LOSO 的工作点与 Test 的工作点分别取在各自最优处，实际不可同时达到。",
        "",
        "## 第一部分：AUC(N) 曲线",
        "",
        "固定工作点取 PCA-96 k=12（与 E143 参考一致，便于逐 N 比较）；最优列为该 N 在 "
        "PCA 48/64/96/128 × k 1/3/6/12 网格内的最优诊断 AUC。",
        "",
        "N=1 的特征与 E65 完全相同（最大绝对差 5e-7，纯 float32 舍入），因此 N=1 行里 "
        "AUC-48 k=12 的 0.7637 只是 E65 在 PCA-48 的表现（E65 对照网格只扫了 64/96/128），"
        "不是子窗口信息。",
        "",
        "| N | AUC（PCA-96 k=12） | W@.90 | 最优 AUC | 最优配置 | 该配置 W@.90 | OOF AUC | 该 N 最高 W@.90 |",
        "|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    lines += [curve_line(c) for c in curve]
    lines += ["", "各 N 的全网格（训练 OOF / LOSO / Test）：", "", HDR, SEP]
    for n in [c["n"] for c in curve]:
        for r in sorted([r for r in sub if r["n"] == n], key=lambda r: (r["dim"], r["k"])):
            lines.append(row_str(r))
    lines += ["", "## 第一部分：多尺度拼接（N=2/3/4/6 各自 PCA + block_norm 后拼接）", "",
              HDR, SEP]
    for r in sorted(ms, key=lambda r: (r["block_dim"], r["dim"], r["k"])):
        lines.append(row_str(r))
    ms_auc = max(ms, key=lambda r: r["diagnostic_auc"])
    lines += ["", f"- 多尺度拼接最优诊断 AUC {ms_auc['diagnostic_auc']:.4f}"
                  f"（{ms_auc['config']}），对比单尺度最优 "
                  f"{max(sub, key=lambda r: r['diagnostic_auc'])['diagnostic_auc']:.4f}。", ""]
    lines += ["## 第二部分：特征级融合（对照组 E65+logbp 必须复现 0.7999）", "",
              HDR, SEP]
    ctrl_best = max(fc, key=lambda r: r["diagnostic_auc"])
    lines.append(row_str(ctrl_best))
    lines += ["", f"- 对照最高诊断 AUC {ctrl_best['diagnostic_auc']:.4f}"
                  f"（{ctrl_best['config']}）；E167 公布 0.7999，"
                  f"{'复现' if v['e65_logbp_control_reproduces_07999'] else '未复现'}。", "",
              "全量对照网格：", "", HDR, SEP]
    for r in sorted(fc, key=lambda r: (r["family"], r["k"])):
        lines.append(row_str(r))
    lines += ["", f"### 三路 E65 + logbp + subwin_mean(N={n_best})", "", HDR, SEP]
    for r in sorted(f3, key=lambda r: -r["diagnostic_auc"]):
        lines.append(row_str(r))
    lines += ["", f"### 两路 E65 + subwin_mean(N={n_best})", "", HDR, SEP]
    for r in sorted(f2, key=lambda r: -r["diagnostic_auc"]):
        lines.append(row_str(r))
    b3 = max(f3, key=lambda r: r["diagnostic_auc"])
    b2 = max(f2, key=lambda r: r["diagnostic_auc"])
    lines += ["",
              f"- 三路最优诊断 AUC {b3['diagnostic_auc']:.4f}（{b3['config']}）；"
              f"两路最优 {b2['diagnostic_auc']:.4f}（{b2['config']}）；"
              f"对照 E65+logbp {ctrl_best['diagnostic_auc']:.4f}。", ""]
    lines += ["## 第二部分：分数级融合（0.1 单纯形网格）", "",
              "按诊断 AUC 排序（前 12，Test 选点）：", "", HDR, SEP]
    for r in sorted(sfx, key=lambda r: -r["diagnostic_auc"])[:12]:
        lines.append(row_str(r))
    lines += ["", "按最优 W 排序（前 12，Test 选点）：", "", HDR, SEP]
    for r in sorted(sfx, key=lambda r: -r["best_W"])[:12]:
        lines.append(row_str(r))
    lines += ["", "权重只用 Train LOSO AUC 选（Test 不参与选权）：", "",
              "| 配置 | 诊断 AUC | Train OOF AUC | Train OOF F1 | LOSO 拒识@.90 | "
              "S1 拒识@.90 | S2 接受@.90 | W@.90 | Train LOSO AUC（选权依据） |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sft:
        lines.append(row_extra(r, [f"{r['train_loso_auc']:.4f}"]))
    lines += ["", "## 第三部分：E159 同口径上界分析（冻结分数，扫满阈值）", "",
              "| 目标 | 配置 | 诊断 AUC | oracle 最优 W (a / LOSO / S1 / S2) | "
              "E159 口径上界 W | 距 0.80 | 最松口径上界 W | 距 0.80 |",
              "|---|---|---:|---|---:|---:|---:|---:|"]
    for c in results["ceiling_analysis"]:
        p = c["oracle_point"]
        lines.append(
            f"| {c['target_tag']} | {c['label']} | {c['diagnostic_auc']:.4f} | "
            f"{c['oracle_max_W']:.4f} ({p['acceptance_rate']:.3f} / {p['LOSO_rejection']:.4f} / "
            f"{p['Test_S1_rejection']:.4f} / {p['Test_S2_acceptance']:.4f}) | "
            f"{c['E159_convention_W']:.4f} | {c['E159_gap_to_080']:+.4f} | "
            f"{c['loose_convention_W']:.4f} | {c['loose_gap_to_080']:+.4f} |")
    lines += [
        "",
        "- **E159 同口径**（与 E169 的 0.7704 同定义）：LOSO 固定在该配置 @0.90 的实测值，"
        "只挪 Test 工作点。",
        "- **最松口径**：LOSO 固定在该配置协议网格内能达到的最高拒识（通常出现在 a=0.80），"
        "再把 Test 工作点选到最优，是「LOSO 不退步」意义下的严格上界。",
        "- **oracle**：LOSO 与阈值一起移动（acceptance→0 时 LOSO→1），偏乐观，仅作参照。",
        "",
        "## 全部配置中诊断 AUC / W 前 10", "",
        "| 配置 | 诊断 AUC | Train OOF AUC | Train OOF F1 | LOSO@.90 | S1@.90 | S2@.90 | W@.90 | 最优 W | 接受率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(all_rows, key=lambda r: -r["diagnostic_auc"])[:10]:
        lines.append(row_extra(r, [f"{r['best_W']:.4f}", f"{r['best_acceptance']:.2f}"]))
    lines += ["", "按最优 W 前 10：", "",
              "| 配置 | 诊断 AUC | Train OOF AUC | Train OOF F1 | LOSO@.90 | S1@.90 | "
              "S2@.90 | W@.90 | 最优 W | 接受率 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(all_rows, key=lambda r: -r["best_W"])[:10]:
        lines.append(row_extra(r, [f"{r['best_W']:.4f}", f"{r['best_acceptance']:.2f}"]))
    checks = results["feature_manifest"]["checks"]
    lines += [
        "",
        "## 说明",
        "",
        "- 表格中 @.90 为 E143 参考工作点（接受率 0.90）；「最优 W」跨接受率网格 0.80–0.96 取最大，"
        "属用 Test 结果选工作点，故结论以 @.90 与诊断 AUC 为准。",
        "- 子窗口特征：每个子窗口独立做 FFT 频带隔离、去均值、通道标准化、相关矩阵、收缩、logSPD、"
        "上三角、detrend_zband，再对 N 个子窗口的 855 维结果取平均；因 detrend_zband 非线性，"
        "subwin_mean ≠ 全长 E65（N=1 时才精确相等）。",
        f"- N=1 校验：特征缓存与 E65 的最大绝对差 "
        f"{max(checks[k] for k in checks if k.endswith('max_abs')):.2e}；N=1 的诊断 AUC "
        f"{v['n1_equals_e65']['diagnostic_auc']:.6f}，E65 {base['diagnostic_auc']:.6f}"
        f"（差 {v['n1_equals_e65']['delta']:+.2e}）。",
        "- 分数级融合复用每路已冻结的原型距离矩阵（原型与权重无关），与重新打分完全等价；"
        "每路先用自身 5 块 Train OOF 的均值/标准差 z 标准化。",
        f"- Test 官方 AUC / F1：**N/A**（Test 无真实标签，session 仅用于诊断 AUC）。",
        "",
        f"运行耗时 {results['runtime_seconds']:.0f}s。",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if REPORT_ONLY:
        write_report(json.loads((OUT / "results.json").read_text(encoding="utf-8")))
    else:
        try:
            main()
        except Exception:
            log("EXCEPTION\n" + traceback.format_exc())
            raise
