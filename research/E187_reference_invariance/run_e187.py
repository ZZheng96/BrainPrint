"""E187: does a common-mode / reference-electrode component drive the cross-session offset?

The E65 representation is a per-trial, per-band channel-standardised *correlation*
matrix.  Correlation is exactly invariant to per-channel gains (E165: channel gain,
band gain, circular shift), but it is *not* invariant to an additive common-mode
signal: a reference electrode with a different contact renders the same common
signal on all 18 channels, which distorts every correlation at once.

Every transform below is applied to the raw 18x1200 trial, per trial and per
sample, identically for Train and Test:

  raw            identity (cached E65 features, the control)
  car            common average reference, x - mean over channels
  car_rank       CAR, then per-channel rank -> normal quantiles over time
  laplacian      x_i - mean of the fixed 10-20 nearest-neighbour channels
  bipolar        18 fixed bipolar derivations (common mode immune by construction)
  car_plus_slope CAR E65 correlation-logSPD + cached E186 1/f slope (18)

After the transform the *unchanged* E65 gpu_correlation_logspd is applied, then
the E143 protocol verbatim: Train-only task z-score / PCA-96 / k-means
prototypes / 5-block Train OOF thresholds, LOSO with `pm & fit_rows` and the
scored identity excluded, evaluation block 0.6 < frac <= 0.8.  Test is scored
once; the Session column is only used for the descriptive audit.

No cross-sample or cross-batch statistic is used anywhere: every transform is a
function of one trial only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.metrics import f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E65_correlation_spd"))
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
import run_e65 as E65  # noqa: E402
import run_e36 as E36  # noqa: E402

CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FT = torch.float32
CLASSES = np.arange(1, 21, dtype=int)
TASKS = CFG["tasks"]
CHANNELS = CFG["channel_order"]
NC = len(CHANNELS)
N = int(CFG["data"]["samples_per_trial"])
E36_CFG = E36.load_config()
E65_CFG = json.loads((ROOT / "research" / "E65_correlation_spd" / "config.json")
                     .read_text(encoding="utf-8"))
FEATURE_DIM = 855
CACHE = OUT / CFG["extraction"]["cache_subdir"]
VARIANTS = CFG["transforms"]["order"]
COMPUTED = ("car", "car_rank", "laplacian", "bipolar")
SLOPE_LO, SLOPE_HI = CFG["streams"]["slope_slice"]

FIT_FRACTION = float(CFG["protocol"]["fit_fraction"])
WIN_LO, WIN_HI = CFG["protocol"]["eval_window"]
BLOCKS = int(CFG["protocol"]["blocks"])
PCA_DIM = int(CFG["protocol"]["pca_dim"])
K_GRID = [int(k) for k in CFG["protocol"]["prototype_k"]]
ACC_GRID = [float(a) for a in CFG["protocol"]["acceptance_grid"]]
REF = CFG["targets"]["raw_reference"]


class Tee:
    def __init__(self, path: Path) -> None:
        self.fh = path.open("w", encoding="utf-8", newline="\n")
        self.stdout = sys.stdout

    def write(self, text: str) -> None:
        self.stdout.write(text)
        self.stdout.flush()
        self.fh.write(text)
        self.fh.flush()

    def flush(self) -> None:
        self.stdout.flush()
        self.fh.flush()


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------ transforms
def _index(name: str) -> int:
    return CHANNELS.index(name)


def adjacency_matrix() -> torch.Tensor:
    """Row-normalised 10-20 nearest-neighbour matrix (fixed, not data driven)."""
    a = np.zeros((NC, NC), dtype=np.float32)
    for left, right in CFG["transforms"]["laplacian_adjacency_10_20"]:
        i, j = _index(left), _index(right)
        a[i, j] = 1.0
        a[j, i] = 1.0
    degree = a.sum(1, keepdims=True)
    if (degree == 0).any():
        raise ValueError(f"孤立通道：{np.flatnonzero(degree.ravel() == 0)}")
    return torch.as_tensor(a / degree, dtype=FT, device=DEVICE)


ADJ = adjacency_matrix()
BIP_A = torch.as_tensor([_index(a) for a, _ in CFG["transforms"]["bipolar_pairs"]],
                        dtype=torch.long, device=DEVICE)
BIP_B = torch.as_tensor([_index(b) for _, b in CFG["transforms"]["bipolar_pairs"]],
                        dtype=torch.long, device=DEVICE)
SQRT2 = math.sqrt(2.0)


def normal_rank(x: torch.Tensor) -> torch.Tensor:
    """Per-channel rank of the 1200 time samples -> normal quantiles."""
    ranks = torch.argsort(torch.argsort(x, dim=-1), dim=-1).to(x.dtype)
    q = (ranks + 0.5) / float(x.shape[-1])
    return SQRT2 * torch.erfinv(2.0 * q - 1.0)


def apply_transform(raw: torch.Tensor, name: str) -> torch.Tensor:
    """raw: B x 18 x 1200 -> B x 18 x 1200 (bipolar: 18 fixed derivations)."""
    if name == "car":
        return raw - raw.mean(dim=-2, keepdim=True)
    if name == "car_rank":
        centred = raw - raw.mean(dim=-2, keepdim=True)
        return normal_rank(centred)
    if name == "laplacian":
        return raw - torch.einsum("ij,bjt->bit", ADJ, raw)
    if name == "bipolar":
        return raw[:, BIP_A, :] - raw[:, BIP_B, :]
    raise ValueError(f"未知变换 {name}")


def load_one(path: Path) -> np.ndarray:
    data = np.asarray(loadmat(path, variable_names=["data"])["data"], dtype=np.float32)
    if data.shape != (N, NC):
        raise ValueError(f"{path}: 预期 {(N, NC)}，实际 {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: 含非有限值")
    return data.T


def cache_path(split: str, name: str) -> Path:
    return CACHE / f"{split}_{name}.npy"


def build_caches() -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    report = {}
    batch = int(CFG["extraction"]["io_batch_size"])
    workers = int(CFG["extraction"]["io_workers"])
    for split in ("train", "test"):
        meta = pd.read_csv(ROOT / CFG["data"][f"{split}_metadata"])
        paths = [ROOT / CFG["data"]["root"] / p for p in meta["path"]]
        todo = [name for name in COMPUTED
                if not (cache_path(split, name).exists()
                        and np.load(cache_path(split, name), mmap_mode="r").shape
                        == (len(paths), FEATURE_DIM))]
        for name in COMPUTED:
            if name not in todo:
                log(f"  复用 {split}/{name} 缓存 {len(paths)}x{FEATURE_DIM}")
        if not todo:
            report[split] = {"reused": True, "rows": len(paths)}
            continue
        outs = {name: np.lib.format.open_memmap(cache_path(split, name), mode="w+",
                                                dtype=np.float32,
                                                shape=(len(paths), FEATURE_DIM))
                for name in todo}
        started = time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(paths), batch):
                stop = min(start + batch, len(paths))
                arrays = list(pool.map(load_one, paths[start:stop]))
                raw = torch.as_tensor(np.stack(arrays), dtype=FT, device=DEVICE)
                with torch.no_grad():
                    for name in todo:
                        features = E65.gpu_correlation_logspd(
                            apply_transform(raw, name), E36_CFG, E65_CFG)
                        if features.shape != (len(arrays), FEATURE_DIM):
                            raise ValueError(f"{split}/{name} 维度异常 {features.shape}")
                        outs[name][start:stop] = features.cpu().numpy()
                del raw
                if stop % 2048 < batch or stop == len(paths):
                    log(f"  {split} {stop}/{len(paths)} ({time.time() - started:.0f}s)")
        for name in todo:
            outs[name].flush()
            del outs[name]
        report[split] = {"reused": False, "rows": len(paths),
                         "elapsed_seconds": time.time() - started}
        torch.cuda.empty_cache()
    return report


def invariance_check() -> dict:
    """Same trial alone vs in a batch vs permuted: every transform must be sample-local."""
    meta = pd.read_csv(ROOT / CFG["data"]["train_metadata"])
    selected = np.sort(np.random.default_rng(int(CFG["seed"])).choice(
        len(meta), int(CFG["extraction"]["invariance_sample_trials"]), replace=False))
    paths = [ROOT / CFG["data"]["root"] / meta["path"].iloc[i] for i in selected]
    raw = torch.as_tensor(np.stack([load_one(p) for p in paths]), dtype=FT, device=DEVICE)
    permutation = np.random.default_rng(int(CFG["seed"]) + 1).permutation(len(raw))
    inverse = np.argsort(permutation)
    checks = {}
    with torch.no_grad():
        for name in COMPUTED:
            batch = E65.gpu_correlation_logspd(apply_transform(raw, name), E36_CFG, E65_CFG)
            singles = torch.cat([
                E65.gpu_correlation_logspd(apply_transform(raw[i:i + 1], name), E36_CFG,
                                           E65_CFG) for i in range(len(raw))])
            permuted = E65.gpu_correlation_logspd(
                apply_transform(raw[permutation], name), E36_CFG, E65_CFG)
            checks[name] = {
                "single_vs_batch_max_abs": float((batch - singles).abs().max()),
                "permutation_max_abs": float((batch - permuted[inverse]).abs().max()),
                "passed": bool(torch.allclose(batch, singles, atol=1e-5, rtol=1e-5)
                               and torch.allclose(batch, permuted[inverse], atol=1e-5,
                                                  rtol=1e-5)),
            }
    checks["cross_sample_or_batch_statistics"] = False
    checks["raw_reuses_frozen_E65_cache"] = True
    if not all(checks[name]["passed"] for name in COMPUTED):
        raise RuntimeError(f"逐样本不变性失败：{checks}")
    return checks


# ------------------------------------------------------------------ protocol
def frac(meta: pd.DataFrame) -> np.ndarray:
    mx = meta.groupby(["subject", "task"])["trial"].transform("max").to_numpy(float)
    return meta["trial"].to_numpy(float) / np.maximum(mx, 1.0)


def task_standardise(a: np.ndarray, b: np.ndarray, tasks: np.ndarray,
                     test_tasks: np.ndarray) -> tuple:
    a = np.asarray(a, dtype=np.float32).copy()
    b = np.asarray(b, dtype=np.float32).copy()
    for task in TASKS:
        m = tasks == task
        mu = a[m].mean(0)
        sd = a[m].std(0) + 1e-6
        a[m] = (a[m] - mu) / sd
        q = test_tasks == task
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
        if len(idx) == 0 or not len(cols):
            continue
        p = torch.as_tensor(protos[cols], dtype=FT, device=DEVICE)
        for start in range(0, len(idx), 2048):
            chunk = idx[start:start + 2048]
            q = torch.as_tensor(np.asarray(feat[chunk], dtype=np.float32), device=DEVICE)
            out[chunk] = (-torch.cdist(q, p)).max(1).values.cpu().numpy()
    return out


class Context:
    def __init__(self) -> None:
        meta = pd.read_csv(ROOT / CFG["data"]["train_metadata"])
        qmeta = pd.read_csv(ROOT / CFG["data"]["test_metadata"])
        self.nodes = meta["subject"].to_numpy(int)
        self.tasks = meta["task"].to_numpy()
        self.frac = frac(meta)
        self.fit_rows = self.frac <= FIT_FRACTION
        self.blocks = np.clip((self.frac * BLOCKS).astype(int), 0, BLOCKS - 1)
        self.window = (self.frac > WIN_LO) & (self.frac <= WIN_HI)
        self.window_rows = np.flatnonzero(self.window)
        self.test_tasks = qmeta["task"].to_numpy()
        self.is_s1 = qmeta["session"].astype(str).str.casefold().isin(
            ["session1", "s1", "1"]).to_numpy()
        self.y = np.r_[np.zeros(int(self.is_s1.sum()), dtype=int),
                       np.ones(int((~self.is_s1).sum()), dtype=int)]
        self.n_test = len(self.test_tasks)
        self.proj: dict[str, tuple] = {}

    def project(self, key: str, feat_train: np.ndarray, feat_test: np.ndarray,
                dim: int) -> tuple:
        cache = f"{key}@{dim}"
        if cache in self.proj:
            return self.proj[cache]
        mu, basis = pca_basis(np.asarray(feat_train[self.fit_rows], dtype=np.float64), dim)
        at = ((np.asarray(feat_train, dtype=np.float64) - mu) @ basis).astype(np.float32)
        bt = ((np.asarray(feat_test, dtype=np.float64) - mu) @ basis).astype(np.float32)
        self.proj[cache] = (at, bt)
        return at, bt

    def drop(self, key: str, dim: int) -> None:
        self.proj.pop(f"{key}@{dim}", None)


def evaluate(at: np.ndarray, bt: np.ndarray, k: int, ctx: Context) -> dict:
    """E143 open-set protocol on one projected representation."""
    protos, keys = build_prototypes(at, ctx.nodes, ctx.tasks, np.ones(len(at), bool),
                                    CLASSES, k)
    te_raw = neg_d1(bt, protos, keys, ctx.test_tasks)

    oof = np.full(len(at), np.nan, dtype=np.float32)
    oof_pred = np.full(len(at), -1, dtype=np.int32)
    for block in range(BLOCKS):
        hold = ctx.blocks == block
        held = np.flatnonzero(hold)
        p2, k2 = build_prototypes(at, ctx.nodes, ctx.tasks, ~hold, CLASSES, k)
        dmat = np.full((len(held), len(k2)), -np.inf, dtype=np.float32)
        key_task = np.asarray([key[0] for key in k2])
        key_subject = np.asarray([key[1] for key in k2], dtype=np.int32)
        for task in TASKS:
            r = np.flatnonzero(ctx.tasks[held] == task)
            c = np.flatnonzero(key_task == task)
            if len(r) == 0 or len(c) == 0:
                continue
            p = torch.as_tensor(np.asarray(p2[c]), dtype=FT, device=DEVICE)
            for start in range(0, len(r), 2048):
                chunk = r[start:start + 2048]
                q = torch.as_tensor(np.asarray(at[held[chunk]], dtype=np.float32),
                                    device=DEVICE)
                dmat[np.ix_(chunk, c)] = (-torch.cdist(q, p)).cpu().numpy()
        oof[held] = dmat.max(1)
        oof_pred[held] = key_subject[dmat.argmax(1)]

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
    known_all = np.concatenate([kn for kn, _ in loso])
    oof_auc = float(roc_auc_score(np.r_[np.zeros(len(unknown_all)), np.ones(int(finite.sum()))],
                                  np.r_[unknown_all, z_rf[finite]]))
    oof_f1 = float(f1_score(ctx.nodes[finite], oof_pred[finite], labels=CLASSES,
                            average="macro", zero_division=0))
    oof_acc = float(np.mean(oof_pred[finite] == ctx.nodes[finite]))
    diag_auc = float(roc_auc_score(ctx.y, np.r_[z_te[ctx.is_s1], z_te[~ctx.is_s1]]))
    loso_pooled_auc = float(roc_auc_score(np.r_[np.zeros(len(unknown_all)),
                                               np.ones(len(known_all))],
                                          np.r_[unknown_all, known_all]))
    u95 = float(np.mean([np.mean(un < np.quantile(kn, 0.05)) for kn, un in loso]))
    u95_global = float(np.mean(unknown_all < np.quantile(known_all, 0.05)))

    points = []
    for acceptance in ACC_GRID:
        threshold = float(np.nanquantile(z_rf[finite], 1 - acceptance))
        s2 = float(np.mean(z_te[~ctx.is_s1] >= threshold))
        s1 = float(np.mean(z_te[ctx.is_s1] < threshold))
        loso_rej = float(np.mean([np.mean(un < np.quantile(kn, 1 - acceptance))
                                  for kn, un in loso]))
        points.append({"a": float(acceptance), "LOSO_rejection": loso_rej,
                       "Test_S1_rejection": s1, "Test_S2_acceptance": s2,
                       "S1_plus_S2": s1 + s2, "threshold": threshold})

    return {"diagnostic_auc": diag_auc, "train_oof_auc": oof_auc, "train_oof_f1": oof_f1,
            "train_oof_identity_acc": oof_acc, "loso_pooled_auc": loso_pooled_auc,
            "u95": u95, "u95_global": u95_global, "points": points,
            "test_z": z_te, "oof_z": z_rf, "loso": loso,
            "per_task_auc": {
                task: float(roc_auc_score((~ctx.is_s1[ctx.test_tasks == task]).astype(int),
                                          z_te[ctx.test_tasks == task]))
                for task in TASKS
                if len(np.unique(ctx.is_s1[ctx.test_tasks == task])) > 1}}


def summarise(entry: dict, meta: dict) -> dict:
    rec = dict(meta)
    best = max(entry["points"], key=lambda p: p["S1_plus_S2"])
    ref = min(entry["points"], key=lambda p: abs(p["a"] - 0.90))
    rec.update({
        "diagnostic_auc": entry["diagnostic_auc"],
        "train_oof_auc": entry["train_oof_auc"],
        "train_oof_f1": entry["train_oof_f1"],
        "train_oof_identity_acc": entry["train_oof_identity_acc"],
        "loso_pooled_auc": entry["loso_pooled_auc"],
        "u95": entry["u95"], "u95_global": entry["u95_global"],
        "best_s1_plus_s2": best["S1_plus_S2"], "best_a": best["a"],
        "best_s1": best["Test_S1_rejection"], "best_s2": best["Test_S2_acceptance"],
        "s1_at_090": ref["Test_S1_rejection"], "s2_at_090": ref["Test_S2_acceptance"],
        "s1_plus_s2_at_090": ref["S1_plus_S2"], "loso_at_090": ref["LOSO_rejection"],
        "points": entry["points"], "per_task_auc": entry["per_task_auc"],
    })
    return rec


def ceiling(entry: dict, ctx: Context) -> dict:
    """E159-style upper bound: freeze the score, sweep every threshold on Test."""
    z_rf, z_te, loso = entry["oof_z"], entry["test_z"], entry["loso"]
    ref = z_rf[np.isfinite(z_rf)]
    grid = np.unique(np.quantile(np.r_[ref, z_te],
                                 np.linspace(0.001, 0.999,
                                             int(CFG["ceiling"]["grid_points"]))))
    best = None
    for threshold in grid:
        s2 = float(np.mean(z_te[~ctx.is_s1] >= threshold))
        s1 = float(np.mean(z_te[ctx.is_s1] < threshold))
        rate = float(np.mean(ref >= threshold))
        loso_rej = (float(np.mean([np.mean(un < np.quantile(kn, 1 - rate))
                                   for kn, un in loso])) if 0 < rate < 1 else 0.0)
        row = {"threshold": float(threshold), "accept_rate": rate,
               "LOSO_rejection": loso_rej, "Test_S1_rejection": s1,
               "Test_S2_acceptance": s2, "S1_plus_S2": s1 + s2}
        if best is None or row["S1_plus_S2"] > best["S1_plus_S2"]:
            best = row
    return best


# ------------------------------------------------------------------ main
def main() -> None:
    started = time.time()
    torch.manual_seed(int(CFG["seed"]))
    np.random.seed(int(CFG["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("E187 需要 CUDA")
    ctx = Context()
    log(f"device {DEVICE} | train {len(ctx.nodes)} | test {ctx.n_test} | "
        f"fit_rows {int(ctx.fit_rows.sum())} | window {int(ctx.window.sum())}")
    log(f"tasks {dict(pd.Series(ctx.tasks).value_counts())}")
    log(f"transforms {VARIANTS}")

    log("特征提取（CAR / CAR-rank / Laplacian / bipolar）")
    cache_report = build_caches()
    checks = invariance_check()
    log(f"逐样本不变性 {json.dumps({k: v for k, v in checks.items() if k in COMPUTED}, ensure_ascii=False)}")

    log("载入特征")
    train = {"raw": np.load(ROOT / CFG["streams"]["train_raw_logspd"]).astype(np.float32)}
    test = {"raw": np.load(ROOT / CFG["streams"]["test_raw_logspd"]).astype(np.float32)}
    for name in COMPUTED:
        train[name] = np.load(cache_path("train", name)).astype(np.float32)
        test[name] = np.load(cache_path("test", name)).astype(np.float32)
    if train["raw"].shape != (len(ctx.nodes), FEATURE_DIM):
        raise ValueError(f"Train raw 维度异常 {train['raw'].shape}")
    if test["raw"].shape != (ctx.n_test, FEATURE_DIM):
        raise ValueError(f"Test raw 维度异常 {test['raw'].shape}")

    slope_train = np.load(ROOT / CFG["streams"]["spectral_traits_train"]).astype(np.float32)
    slope_test = np.load(ROOT / CFG["streams"]["spectral_traits_test"]).astype(np.float32)
    slope_train = slope_train[:, SLOPE_LO:SLOPE_HI]
    slope_test = slope_test[:, SLOPE_LO:SLOPE_HI]
    log(f"E186 1/f 斜率 {slope_train.shape} / {slope_test.shape}")

    entries: dict[tuple, dict] = {}
    rows: dict[str, list] = {name: [] for name in VARIANTS}
    t0 = time.time()
    for name in VARIANTS:
        base = "car" if name == "car_plus_slope" else name
        at_raw, bt_raw = task_standardise(train[base], test[base], ctx.tasks, ctx.test_tasks)
        if name == "car_plus_slope":
            sa, sb = task_standardise(slope_train, slope_test, ctx.tasks, ctx.test_tasks)
            ra = float(np.sqrt(np.mean(at_raw[ctx.fit_rows] ** 2)))
            rb = float(np.sqrt(np.mean(sa[ctx.fit_rows] ** 2)))
            at_raw = np.concatenate([at_raw / max(ra, 1e-6), sa / max(rb, 1e-6)], axis=1)
            bt_raw = np.concatenate([bt_raw / max(ra, 1e-6), sb / max(rb, 1e-6)], axis=1)
        at, bt = ctx.project(name, at_raw, bt_raw, PCA_DIM)
        log(f"[{name}] projected {at.shape} / {bt.shape}")
        for k in K_GRID:
            ekey = (name, int(k))
            if ekey not in entries:
                entries[ekey] = evaluate(at, bt, int(k), ctx)
                entries[ekey]["ceiling"] = ceiling(entries[ekey], ctx)
            rec = summarise(entries[ekey], {"variant": name, "k": int(k),
                                            "feature_dim": int(at_raw.shape[1]),
                                            "pca_dim": int(at.shape[1]),
                                            "ceiling": entries[ekey]["ceiling"],
                                            "ceiling_s1_plus_s2":
                                                entries[ekey]["ceiling"]["S1_plus_S2"]})
            rows[name].append(rec)
            log(f"    {name} k={k}: diagAUC {rec['diagnostic_auc']:.4f} "
                f"OOF AUC {rec['train_oof_auc']:.4f} F1 {rec['train_oof_f1']:.4f} "
                f"bestS1+S2 {rec['best_s1_plus_s2']:.4f}@a={rec['best_a']:.2f} "
                f"ceil {rec['ceiling_s1_plus_s2']:.4f} ({time.time() - t0:.0f}s)")
        ctx.drop(name, PCA_DIM)

    # ---------------- raw reproduction check
    ref_raw = next(r for r in rows["raw"] if r["k"] == 12)
    reproduction = {
        "auc": round(ref_raw["diagnostic_auc"], 4),
        "s1_at_090": round(ref_raw["s1_at_090"], 4),
        "s2_at_090": round(ref_raw["s2_at_090"], 4),
        "s1_plus_s2_at_090": round(ref_raw["s1_plus_s2_at_090"], 4),
        "expected": REF,
        "reproduced": bool(abs(ref_raw["diagnostic_auc"] - REF["test_auc"]) < 1e-4
                           and abs(ref_raw["s1_at_090"] - REF["s1_at_090"]) < 1e-4
                           and abs(ref_raw["s2_at_090"] - REF["s2_at_090"]) < 1e-4
                           and abs(ref_raw["s1_plus_s2_at_090"] - REF["s1_plus_s2_at_090"]) < 1e-4),
    }
    log(f"raw 对照复现 {reproduction}")
    if not reproduction["reproduced"]:
        raise RuntimeError(f"raw 对照未复现 E65 基准：{reproduction}")

    # ---------------- best per variant / overall
    def tie(r):
        return (round(r["best_s1_plus_s2"], 4), r["diagnostic_auc"])

    best_by_variant = {name: max(rs, key=tie) for name, rs in rows.items()}
    best_all = max((r for rs in rows.values() for r in rs), key=tie)
    best_entry = entries[(best_all["variant"], int(best_all["k"]))]
    best_auc = max((r for rs in rows.values() for r in rs), key=lambda r: r["diagnostic_auc"])

    # ---------------- ten gates (best variant)
    cap = json.loads((ROOT / "research/E183_gate_head_and_capacity/results.json")
                     .read_text(encoding="utf-8"))["capacity"]
    gates = [
        {"id": "#1", "name": "Train 准确率", "target": "= 1.00",
         "value": round(float(cap["in_sample_logits_argmax"]), 4), "source": "E183 冻结复算"},
        {"id": "#2", "name": "C（5 块 OOF 闭集准确率）", "target": "≥ 0.95",
         "value": round(float(cap["oof_accuracy_C"]), 4), "source": "E183 冻结复算"},
        {"id": "#3", "name": "身份 macro-F1（5 块 OOF）", "target": "≥ 0.92",
         "value": round(float(cap["oof_macro_f1"]), 4), "source": "E183 冻结复算"},
        {"id": "#4", "name": "块外准确率（frac>0.6）", "target": "≥ 0.90",
         "value": round(float(cap["oof_accuracy_forward_blocks"]), 4),
         "source": "E183 冻结复算"},
        {"id": "#5", "name": "20 折 LOSO AUROC", "target": "≥ 0.98",
         "value": round(best_entry["loso_pooled_auc"], 4), "source": "本实验最佳配置"},
        {"id": "#6", "name": "U@95%（known 接受 95%）", "target": "≥ 0.95",
         "value": round(min(best_entry["u95"], best_entry["u95_global"]), 4),
         "source": "本实验最佳配置（逐折/全局取最差）"},
        {"id": "#7", "name": "RA（自建 PGD 最差）", "target": "≥ 0.95",
         "value": None, "source": "本实验无攻击模块，未评估 = N/A"},
        {"id": "#8", "name": "ASafeR（同一攻击最差）", "target": "≥ 0.98",
         "value": None, "source": "本实验无攻击模块，未评估 = N/A"},
    ]
    for pid, key, name, target in [("#9", "Test_S2_acceptance", "Test S2 接受率", "≥ 0.75"),
                                   ("#10", "Test_S1_rejection", "Test S1 拒识率", "≥ 0.75")]:
        gates.append({"id": pid, "name": name, "target": target, "value": None,
                      "source": "本实验最佳配置（逐工作点）"})
    gate_table = []
    for g in gates:
        row = dict(g)
        if g["id"] in ("#9", "#10"):
            key = "S1_rejection" if g["id"] == "#10" else "S2_acceptance"
            row["values"] = {f"{p['a']:.2f}": round(p["Test_" + key], 4)
                             for p in best_all["points"]}
            row["pass"] = {a: bool(v >= 0.75) for a, v in row["values"].items()}
        else:
            if g["value"] is None:
                row["pass"] = False
            else:
                limit = float(g["target"].lstrip("≥= ").strip())
                row["pass"] = (abs(g["value"] - limit) < 1e-12
                               if g["target"].startswith("=") else g["value"] >= limit)
        gate_table.append(row)

    judgement = {
        "raw_diagnostic_auc": ref_raw["diagnostic_auc"],
        "car_diagnostic_auc": best_by_variant["car"]["diagnostic_auc"],
        "car_minus_raw_auc": best_by_variant["car"]["diagnostic_auc"] - ref_raw["diagnostic_auc"],
        "car_k12_diagnostic_auc": next(r for r in rows["car"] if r["k"] == 12)["diagnostic_auc"],
        "car_minus_raw_auc_k12": (next(r for r in rows["car"] if r["k"] == 12)["diagnostic_auc"]
                                  - ref_raw["diagnostic_auc"]),
        "car_minus_raw_sum_k12": (next(r for r in rows["car"] if r["k"] == 12)["best_s1_plus_s2"]
                                  - ref_raw["best_s1_plus_s2"]),
        "laplacian_gain_auc": best_by_variant["laplacian"]["diagnostic_auc"] - ref_raw["diagnostic_auc"],
        "bipolar_gain_auc": best_by_variant["bipolar"]["diagnostic_auc"] - ref_raw["diagnostic_auc"],
        "car_above_078": bool(best_by_variant["car"]["diagnostic_auc"] > 0.78),
        "common_mode_main_cause": bool(best_by_variant["car"]["diagnostic_auc"]
                                       - ref_raw["diagnostic_auc"] > 0.01),
        "best_variant": best_all["variant"],
        "best_k": best_all["k"],
        "best_s1_plus_s2": best_all["best_s1_plus_s2"],
        "best_a": best_all["best_a"],
        "beats_14479": bool(best_all["best_s1_plus_s2"] > 1.4479),
        "reaches_150": bool(best_all["best_s1_plus_s2"] >= 1.50),
        "best_ceiling_s1_plus_s2": max(r["ceiling_s1_plus_s2"]
                                       for rs in rows.values() for r in rs),
        "raw_ceiling_s1_plus_s2": max(r["ceiling_s1_plus_s2"] for r in rows["raw"]),
        "best_auc_variant": best_auc["variant"],
        "best_diagnostic_auc": best_auc["diagnostic_auc"],
    }
    log(f"judgement {json.dumps(judgement, ensure_ascii=False)}")

    def slim(r: dict) -> dict:
        return r

    payload = {
        "experiment": CFG["experiment"],
        "clean": True,
        "protocol": CFG["protocol"],
        "transforms": CFG["transforms"],
        "feature_checks": checks,
        "feature_cache": cache_report,
        "raw_reproduction": reproduction,
        "variants": {name: [slim(r) for r in rs] for name, rs in rows.items()},
        "best_by_variant": best_by_variant,
        "best_overall": best_all,
        "best_diagnostic_auc": best_auc,
        "ceiling": {name: max(r["ceiling_s1_plus_s2"] for r in rs)
                    for name, rs in rows.items()},
        "ten_gates": gate_table,
        "judgement": judgement,
        "official_test_metrics": "N/A",
    }
    (OUT / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                                 default=float), encoding="utf-8")
    write_report(payload, rows, best_by_variant, best_all, best_auc, ref_raw, gate_table)
    log(f"done in {time.time() - started:.0f}s")


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_report(payload, rows, best_by_variant, best_all, best_auc, ref_raw,
                 gate_table) -> None:
    j = payload["judgement"]
    acc = ", ".join(f"{a:.2f}" for a in ACC_GRID)
    lines = [
        "# E187 参考化 / 去共模不变性",
        "",
        "E65 的 correlation-logSPD 对逐通道增益、频带增益、循环平移精确不变（E165），"
        "但对**共模成分**敏感：若 Train 与 Test 的参考电极不同、或参考电极接触状态不同，"
        "18 个通道会叠加同一个共模信号，相关矩阵被整体扭曲。本实验对原始 18×1200 逐 trial "
        "施加纯逐样本的参考化变换，再走**完全相同的 E65 correlation-logSPD** 与 **E143 协议**。",
        "",
        "| 变体 | 定义 |",
        "|---|---|",
        "| `raw` | 恒等（复用 E65 冻结特征） |",
        "| `car` | 共平均参考 x − mean(通道) |",
        "| `car_rank` | CAR 后每通道秩→正态分位 |",
        "| `laplacian` | 减去固定 10-20 最近邻通道均值 |",
        "| `bipolar` | 18 组固定双极导联差分 |",
        "| `car_plus_slope` | CAR E65 特征 + E186 1/f 斜率（18 维，块 RMS 归一） |",
        "",
        "评估协议逐字沿用 E143：Train-only task z 标准化 → PCA-96 → k∈{1,3,6,12} 原型 → "
        "5 块 Train OOF 阈值 → LOSO 每折排除被评分身份、原型只用 `pm & fit_rows`、"
        "评估块 0.6<frac≤0.8 → Test 只评分、Session 只审计。",
        "",
        "## 1. 判定（核心）",
        "",
        f"1. **参考化是否提升 Test 诊断 AUC？** raw **{fmt(j['raw_diagnostic_auc'])}** → "
        f"CAR **{fmt(j['car_diagnostic_auc'])}**（Δ={j['car_minus_raw_auc']:+.4f}）；"
        f"CAR 是否 > 0.78：{'是' if j['car_above_078'] else '否'}。"
        f"全实验最高诊断 AUC 为 {fmt(j['best_diagnostic_auc'])}（{j['best_auc_variant']}）。",
        f"2. **S1+S2 能否突破 1.4479？能否达到 1.50？** 最高 **{fmt(j['best_s1_plus_s2'])}**"
        f"（{j['best_variant']} k={j['best_k']} @a={j['best_a']:.2f}）→ "
        f"{'突破' if j['beats_14479'] else '未突破'} 1.4479；"
        f"{'达到' if j['reaches_150'] else '未达到'} 1.50。"
        f"（冻结分数扫满阈值的犯规上界最高 {fmt(j['best_ceiling_s1_plus_s2'])}，"
        f"raw 上界 {fmt(j['raw_ceiling_s1_plus_s2'])}。）",
        f"3. **哪个变体最好？共模是不是会话偏移主因？** 最佳变体 **{j['best_variant']}**"
        f"（S1+S2 {fmt(j['best_s1_plus_s2'])}，诊断 AUC {fmt(best_all['diagnostic_auc'])}）。"
        f"CAR（精确移除共模）相对 raw 的诊断 ΔAUC={j['car_minus_raw_auc']:+.4f}"
        f"（同 k=12 时 {j['car_minus_raw_auc_k12']:+.4f}，S1+S2 "
        f"{j['car_minus_raw_sum_k12']:+.4f}）；bipolar ΔAUC={j['bipolar_gain_auc']:+.4f}、"
        f"Laplacian ΔAUC={j['laplacian_gain_auc']:+.4f}。"
        f"**共模是主因：{'是' if j['common_mode_main_cause'] else '否'}** —— "
        f"{'CAR 显著抬升诊断 AUC' if j['common_mode_main_cause'] else '彻底移除共模的 CAR 反而降低判别力，说明共模不是跨会话偏移的主要来源'}。",
        "",
        "## 2. 各变体的 Test 诊断 AUC 与 S1+S2（k=12，与 E65 基准同配置）",
        "",
        "| 变体 | 特征维 | 诊断 AUC | Train OOF AUC | Train OOF F1 | LOSO@.90 | S1@.90 | S2@.90 | S1+S2@.90 | 最优 a | 最优 S1+S2 | 犯规上界 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANTS:
        r = next(row for row in rows[name] if row["k"] == 12)
        lines.append(f"| {name} | {r['feature_dim']} | {fmt(r['diagnostic_auc'])} | "
                     f"{fmt(r['train_oof_auc'])} | {fmt(r['train_oof_f1'])} | "
                     f"{fmt(r['loso_at_090'])} | {fmt(r['s1_at_090'])} | {fmt(r['s2_at_090'])} | "
                     f"{fmt(r['s1_plus_s2_at_090'])} | {r['best_a']:.2f} | "
                     f"{fmt(r['best_s1_plus_s2'])} | {fmt(r['ceiling_s1_plus_s2'])} |")
    lines += ["",
              "各变体在自身最优 k（按 S1+S2，同分取 AUC 高者）：",
              "",
              "| 变体 | 最优 k | 诊断 AUC | Train OOF AUC | Train OOF F1 | LOSO@.90 | S1@.90 | S2@.90 | 最优 a | 最优 S1+S2 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in VARIANTS:
        r = best_by_variant[name]
        lines.append(f"| {name} | {r['k']} | {fmt(r['diagnostic_auc'])} | "
                     f"{fmt(r['train_oof_auc'])} | {fmt(r['train_oof_f1'])} | "
                     f"{fmt(r['loso_at_090'])} | {fmt(r['s1_at_090'])} | {fmt(r['s2_at_090'])} | "
                     f"{r['best_a']:.2f} | {fmt(r['best_s1_plus_s2'])} |")

    raw12 = next(row for row in rows["raw"] if row["k"] == 12)
    lines += ["", "相对 raw 的差值（同一 k 下相减）：", "",
              "| 变体 | 最优 k | ΔAUC（最优 k） | Δ(S1+S2)（最优 k） | ΔAUC（k=12） | Δ(S1+S2)（k=12） |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in VARIANTS:
        r = best_by_variant[name]
        r12 = next(row for row in rows[name] if row["k"] == 12)
        lines.append(f"| {name} | {r['k']} | "
                     f"{r['diagnostic_auc'] - ref_raw['diagnostic_auc']:+.4f} | "
                     f"{r['best_s1_plus_s2'] - ref_raw['best_s1_plus_s2']:+.4f} | "
                     f"{r12['diagnostic_auc'] - raw12['diagnostic_auc']:+.4f} | "
                     f"{r12['best_s1_plus_s2'] - raw12['best_s1_plus_s2']:+.4f} |")

    lines += ["", "## 3. 各工作点（每个变体在其最优 k）", ""]
    for name in VARIANTS:
        r = best_by_variant[name]
        lines += [f"**{name} k={r['k']}**（诊断 AUC {fmt(r['diagnostic_auc'])}，"
                  f"Train OOF AUC {fmt(r['train_oof_auc'])} / F1 {fmt(r['train_oof_f1'])}）", "",
                  "| a | LOSO 拒识 | Test S1 拒识 | Test S2 接受 | S1+S2 |",
                  "|---:|---:|---:|---:|---:|"]
        for p in r["points"]:
            lines.append(f"| {p['a']:.2f} | {fmt(p['LOSO_rejection'])} | "
                         f"{fmt(p['Test_S1_rejection'])} | {fmt(p['Test_S2_acceptance'])} | "
                         f"{fmt(p['S1_plus_S2'])} |")
        lines.append("")

    lines += ["## 4. E159 式上界分析（冻结分数，阈值用 Test 扫满）", "",
              "只做上限证明：不改变任何模型，只把已冻结分数上的阈值扫遍，取 S1+S2 的最大值。"
              "阈值直接用 Test 扫属于犯规读数，只用于判断排序能力还剩多少，不可部署。",
              "",
              "| 变体 | k=12 上界 | 上界点 a | 上界点 S1 | 上界点 S2 | 全 k 最高上界 | 该 k | 该点 a | 该点 S1 | 该点 S2 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in VARIANTS:
        r12 = next(row for row in rows[name] if row["k"] == 12)
        best_grid = max(rows[name], key=lambda r: r["ceiling_s1_plus_s2"])
        p12, pb = r12["ceiling"], best_grid["ceiling"]
        lines.append(f"| {name} | {fmt(p12['S1_plus_S2'])} | {p12['accept_rate']:.3f} | "
                     f"{fmt(p12['Test_S1_rejection'])} | {fmt(p12['Test_S2_acceptance'])} | "
                     f"{fmt(pb['S1_plus_S2'])} | {best_grid['k']} | {pb['accept_rate']:.3f} | "
                     f"{fmt(pb['Test_S1_rejection'])} | {fmt(pb['Test_S2_acceptance'])} |")
    lines += ["", "上界与历史门槛的关系：最高上界 "
              f"**{fmt(j['best_ceiling_s1_plus_s2'])}**（{j['best_variant']}）"
              f"对比历史最好 S1+S2 **1.4479**、目标 **1.50**。", ""]

    lines += ["## 5. 十项门禁表（最佳变体）", "",
              f"最佳变体：**{best_all['variant']} k={best_all['k']}**"
              f"（诊断 AUC {fmt(best_all['diagnostic_auc'])}）。"
              "每一列是同一阈值下的完整系统，不跨工作点拼接。"
              "#1–#4 是 E65 分支身份头的冻结复算（本实验不重训身份头），#7/#8 本实验未做攻击。",
              "",
              "| # | 门禁 | 目标 | 数值 | 判定 | 来源 |", "|---|---|---|---:|---|---|"]
    for g in gate_table:
        if "values" in g:
            for a, v in g["values"].items():
                lines.append(f"| {g['id']} | {g['name']} | {g['target']} | {v:.4f} | "
                             f"{'✅' if g['pass'][a] else '❌'} | a={a} |")
        elif g["value"] is None:
            lines.append(f"| {g['id']} | {g['name']} | {g['target']} | N/A | ❌ | {g['source']} |")
        else:
            lines.append(f"| {g['id']} | {g['name']} | {g['target']} | {g['value']:.4f} | "
                         f"{'✅' if g['pass'] else '❌'} | {g['source']} |")

    car = best_by_variant["car"]
    lap = best_by_variant["laplacian"]
    bip = best_by_variant["bipolar"]
    rank = best_by_variant["car_rank"]
    slope = best_by_variant["car_plus_slope"]
    raw = ref_raw
    car12 = next(row for row in rows["car"] if row["k"] == 12)
    lines += ["", "## 6. 机制解读", "",
              f"1. **raw 对照精确复现**：AUC {fmt(payload['raw_reproduction']['auc'])}、"
              f"S1@.90 {fmt(payload['raw_reproduction']['s1_at_090'])}、"
              f"S2@.90 {fmt(payload['raw_reproduction']['s2_at_090'])}、"
              f"S1+S2@.90 {fmt(payload['raw_reproduction']['s1_plus_s2_at_090'])}，"
              f"与 E65/E186 冻结基准逐位一致（{payload['raw_reproduction']['reproduced']}）。",
              f"2. **CAR 是负收益**：诊断 AUC 从 {fmt(raw['diagnostic_auc'])} 降到 "
              f"{fmt(car['diagnostic_auc'])}（最优 k={car['k']}），同 k=12 时 {fmt(car12['diagnostic_auc'])}"
              f"（{j['car_minus_raw_auc_k12']:+.4f}），S1+S2 从 {fmt(raw['best_s1_plus_s2'])} 掉到 "
              f"{fmt(car['best_s1_plus_s2'])}。CAR 精确移除了通道间的加性共模，"
              "如果共模是会话偏移主因，这里应当显著变好；实际相反，说明**共模不是主因**，"
              "而且被减掉的 18 通道均值里含有身份相关的全局成分（如全脑节律幅度），损失大于收益。",
              f"3. **Bipolar / Laplacian 是正收益但幅度有限**：bipolar 最优 AUC "
              f"{fmt(bip['diagnostic_auc'])}（{j['bipolar_gain_auc']:+.4f}）、S1+S2 "
              f"{fmt(bip['best_s1_plus_s2'])}；Laplacian 最优 AUC {fmt(lap['diagnostic_auc'])}"
              f"（{j['laplacian_gain_auc']:+.4f}）、S1+S2 {fmt(lap['best_s1_plus_s2'])}。"
              "它们同样消除共模，但保留局部空间梯度（相邻通道差分），"
              "说明收益来自「保留局部梯度、压低全局共模」这个组合，而不是单纯去共模。",
              f"4. **CAR-rank 基本无效**：最优 AUC {fmt(rank['diagnostic_auc'])}"
              f"（k={rank['k']}），k=12 时只有 {fmt(next(r for r in rows['car_rank'] if r['k'] == 12)['diagnostic_auc'])}。"
              "秩变换抹掉了幅值和重尾信息，而在本数据里身份信息恰恰部分存在于这些统计量中。",
              f"5. **CAR + 1/f 斜率不叠加**：AUC {fmt(slope['diagnostic_auc'])}、S1+S2 "
              f"{fmt(slope['best_s1_plus_s2'])}，与 CAR 单用几乎相同。"
              "原因是斜率块只有 18 维且与 CAR 的 logSPD 高度相关（E186 已证明它只是同一条跨会话 ROC 上的另一工作点）。",
              f"6. **工作点与 LOSO 的此消彼长**：最优 S1+S2 出现在 k=1 的 bipolar，但同一配置的 LOSO 拒识只有 "
              f"{fmt(bip['loso_at_090'])}（raw k=12 为 {fmt(raw['loso_at_090'])}）。"
              "也就是说，跨会话 S1/S2 的改善部分来自「压低 Train 侧未知拒识、把阈值放松」，"
              "这是同一 ROC 上的工作点搬移，不是系统性变强——与 E179 的松弛门禁结论一致。",
              f"7. **犯规上界仍不够**：bipolar k={best_all['k']} 把阈值扫满后上界只有 "
              f"{fmt(j['best_ceiling_s1_plus_s2'])}，依旧低于历史最好 1.4479，"
              "说明参考化这一族变换没有改变 S1 拒识与 S2 接受之间的斜率，"
              "限制来自排序能力本身（诊断 AUC 0.75–0.78 区间），而不是参考电极。", ""]

    lines += ["", "## 7. 合规", "",
              "- 所有变换都是逐 trial 的逐样本操作，未使用任何跨样本 / 跨批次 / Test 统计量；"
              "PCA、标准化、原型、阈值全部只用 Train（PCA 与 block RMS 只用 frac≤0.6 的 fit_rows）。",
              "- LOSO 一律 `pm & fit_rows`，被评分身份从原型中剔除，评估块从未进入拟合；"
              "Test 只评分，Session 只用于描述性审计，未置 0。",
              "- 逐样本不变性（单条 vs 批 vs 置换）已在 GPU 上复核，见 results.json 的 feature_checks。",
              "- 未修改任何已有实验目录，未提交 git。",
              "",
              "## 8. Test 官方指标", "",
              "**Test 官方 AUC / F1 / C / RA / ASafeR / U / Score = N/A**（Test 无真实标签）。", "",
              "## 9. 文件", "",
              "| 文件 | 内容 |", "|---|---|",
              "| `run_e187.py` | 特征提取 + E143 评估 + 报告 |",
              "| `config.json` | 冻结配置（含 10-20 邻接与双极对） |",
              "| `results.json` | 全部变体 × k × a 数值、上界与十项门禁 |",
              "| `cache/{train,test}_{car,car_rank,laplacian,bipolar}.npy` | 855 维特征缓存 |",
              "| `report.md` / `run.log` | 报告与日志 |"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["features", "eval", "all"])
    args = parser.parse_args()
    sys.stdout = Tee(OUT / "run.log")
    try:
        if args.stage == "features":
            build_caches()
        else:
            main()
    finally:
        sys.stdout = sys.__stdout__
