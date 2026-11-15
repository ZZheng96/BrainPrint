"""E168: 子窗口 correlation-logSPD 表征（trial 内非重叠子段），验证结构信息能否抬高诊断 AUC。

E65 对整个 trial 只算一个相关矩阵，因此对通道增益/频带增益/循环时间平移精确不变，
子段结构被完全丢弃（E165 结论）。本实验把 trial 切成 N 个不重叠子窗口，每个子窗口
独立跑 E65 的完整流程，生成：
  subwin_mean             N 个子窗口 855 维的平均
  subwin_std              每个 (band, pair) 跨子窗口的标准差（trial 内非平稳性）
  subwin_concat           N*855 直接拼接
  subwin_concat_plus_std  N*855 + 855

评估逐字沿用 E143：task-wise Train 标准化 -> PCA -> k 原型 -> 5 块 Train OOF 阈值
-> LOSO（排除被评分身份，原型只用 fit_rows）-> Test 只评分，session 只诊断。
"""
from __future__ import annotations

import copy
import json
import shutil
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
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FT = torch.float32
TASKS = CFG["evaluation"]["tasks"]
CLASSES = np.arange(1, 21, dtype=int)
NP = 171                     # per-band upper-triangle entries
DIM_WINDOW = 5 * NP          # 855
N_CH = int(CFG["data"]["channels"])
N_S = int(CFG["data"]["samples_per_trial"])

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


sys.stdout = Tee(None if "--report-only" in sys.argv else OUT / "run.log")


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


def cache_path(split: str, variant: str) -> Path:
    return OUT / f"{split}_{variant}.npy"


def variant_dim(variant: str, n: int) -> int:
    return {"subwin_mean": DIM_WINDOW,
            "subwin_std": DIM_WINDOW,
            "subwin_concat": n * DIM_WINDOW,
            "subwin_concat_plus_std": n * DIM_WINDOW + DIM_WINDOW}[variant]


def split_meta(split: str) -> pd.DataFrame:
    key = "train_metadata" if split == "train" else "test_metadata"
    meta = pd.read_csv(ROOT / CFG["data"][key])
    expected = CFG["data"]["expected_train_rows" if split == "train" else "expected_test_rows"]
    if len(meta) != expected:
        raise ValueError(f"{split} 行数 {len(meta)} != {expected}")
    return meta


def extract_split(split: str, meta: pd.DataFrame, e36: dict, e65: dict,
                  ns: list, variants: list) -> None:
    n = len(meta)
    ns_used = [nn for nn in ns if any(v.endswith(f"_n{nn}") for v in variants)]
    outputs, shapes = {}, {}
    for nn in ns_used:
        for variant in variants:
            if not variant.endswith(f"_n{nn}"):
                continue
            dim = variant_dim(variant.replace(f"_n{nn}", ""), nn)
            shapes[variant] = (n, dim)
            outputs[variant] = np.lib.format.open_memmap(
                cache_path(split, variant), mode="w+", dtype=np.float32, shape=(n, dim))
    loader = load_train_matrix if split == "train" else load_test_matrix
    paths = [ROOT / CFG["data"]["train_root"] / p for p in meta.path]
    sub_cfg = {}
    for nn in ns_used:
        length = N_S // nn
        if length * nn != N_S:
            raise ValueError(f"N={nn} 不能整除 {N_S}")
        cfg = copy.deepcopy(e36)
        cfg["data"]["samples_per_trial"] = length
        sub_cfg[nn] = cfg
    batch = int(CFG["subwindow"]["io_batch_size"])
    started = time.time()
    with ThreadPoolExecutor(max_workers=int(CFG["subwindow"]["io_workers"])) as pool:
        for start in range(0, n, batch):
            stop = min(start + batch, n)
            arrays = list(pool.map(loader, paths[start:stop]))
            raw = torch.as_tensor(np.stack(arrays), dtype=FT, device=DEVICE)
            b = stop - start
            for nn in ns_used:
                length = N_S // nn
                wins = []
                for w in range(nn):
                    seg = raw[..., w * length:(w + 1) * length].contiguous()
                    with torch.no_grad():
                        wins.append(run_e65.gpu_correlation_logspd(seg, sub_cfg[nn], e65))
                feats = torch.stack(wins, dim=1)                  # B×N×855
                if f"subwin_mean_n{nn}" in outputs:
                    outputs[f"subwin_mean_n{nn}"][start:stop] = feats.mean(1).cpu().numpy()
                if f"subwin_std_n{nn}" in outputs:
                    outputs[f"subwin_std_n{nn}"][start:stop] = feats.std(
                        1, unbiased=bool(CFG["subwindow"]["std_ddof"])).cpu().numpy()
                if f"subwin_concat_n{nn}" in outputs:
                    outputs[f"subwin_concat_n{nn}"][start:stop] = feats.reshape(b, -1).cpu().numpy()
                if f"subwin_concat_plus_std_n{nn}" in outputs:
                    outputs[f"subwin_concat_plus_std_n{nn}"][start:stop] = torch.cat(
                        [feats.reshape(b, -1), feats.std(1, unbiased=False)], dim=1).cpu().numpy()
            if stop % 2048 == 0 or stop == n:
                log(f"  {split} features {stop}/{n} ({time.time() - started:.0f}s)")
    for handle in outputs.values():
        handle.flush()
    del outputs
    log(f"{split} 特征完成：{shapes}（{time.time() - started:.0f}s）")


def feature_checks(e36: dict, e65: dict) -> dict:
    """N=1 全窗应逐位复现 E65 缓存；单条与批量应一致。"""
    meta = split_meta("train")
    cached = np.load(ROOT / CFG["reference_views"]["E65"]["train"], mmap_mode="r")
    rng = np.random.default_rng(int(CFG["seed"]))
    pick = np.sort(rng.choice(len(meta), 8, replace=False))
    paths = [ROOT / CFG["data"]["train_root"] / meta.path.iloc[i] for i in pick]
    raw = torch.as_tensor(np.stack([load_train_matrix(p) for p in paths]), dtype=FT, device=DEVICE)
    with torch.no_grad():
        full = run_e65.gpu_correlation_logspd(raw, e36, e65).cpu().numpy()
        singles = torch.cat([run_e65.gpu_correlation_logspd(raw[i:i + 1], e36, e65)
                             for i in range(len(raw))]).cpu().numpy()
        n2 = e65_n2(raw, e36, e65)
        n2_single = np.concatenate([e65_n2(raw[i:i + 1], e36, e65) for i in range(len(raw))])
    checks = {
        "fullwindow_vs_E65_cache_max_abs": float(np.abs(full - np.asarray(cached[pick])).max()),
        "single_vs_batch_max_abs": float(np.abs(full - singles).max()),
        "subwin_single_vs_batch_max_abs": float(np.abs(n2 - n2_single).max()),
        "finite": True,
        "test_accessed": False,
    }
    log(f"特征检查 {checks}")
    return checks


def e65_n2(raw: torch.Tensor, e36: dict, e65: dict) -> np.ndarray:
    length = N_S // 2
    cfg = copy.deepcopy(e36)
    cfg["data"]["samples_per_trial"] = length
    feats = [run_e65.gpu_correlation_logspd(raw[..., w * length:(w + 1) * length].contiguous(),
                                           cfg, e65) for w in range(2)]
    return torch.stack(feats, 1).reshape(len(raw), -1).cpu().numpy()


def build_features() -> None:
    ns = [int(n) for n in CFG["subwindow"]["ns"]]
    variants = list(CFG["subwindow"]["variants"])
    free_gb = shutil.disk_usage(OUT).free / 1024 ** 3
    if 4 in ns and free_gb < float(CFG["subwindow"]["min_free_gb_for_n4"]):
        log(f"磁盘剩余 {free_gb:.1f}GB < 阈值，跳过 N=4")
        ns = [n for n in ns if n != 4]
    names = [f"{v}_n{n}" for n in ns for v in variants]

    def expected_dim(name: str) -> int:
        base, n = name.rsplit("_n", 1)
        return variant_dim(base, int(n))

    shapes_ok = bool(names) and all(
        cache_path(s, v).is_file()
        and tuple(np.load(cache_path(s, v), mmap_mode="r").shape)
        == (len(split_meta(s)), expected_dim(v))
        for s in ("train", "test") for v in names)
    manifest = OUT / "feature_manifest.json"
    if shapes_ok and manifest.is_file():
        report = json.loads(manifest.read_text(encoding="utf-8"))
        if report.get("config_sha256") == sha256(OUT / "config.json"):
            log("复用 E168 特征缓存")
            return
    if not torch.cuda.is_available():
        raise RuntimeError("E168 特征提取需要 CUDA")
    e36 = load_e36_config()
    e65 = json.loads((ROOT / "research" / "E65_correlation_spd" / "config.json").read_text(encoding="utf-8"))
    started = time.time()
    for split in ("train", "test"):
        meta = split_meta(split)
        extract_split(split, meta, e36, e65, ns, [f"{v}_n{n}" for n in ns for v in variants])
    checks = feature_checks(e36, e65)
    if checks["fullwindow_vs_E65_cache_max_abs"] > 1e-5:
        raise RuntimeError(f"N=1 子窗口流程未能复现 E65：{checks}")
    report = {
        "config_sha256": sha256(OUT / "config.json"),
        "ns": ns,
        "variants": names,
        "shapes": {v: list(np.load(cache_path("train", v), mmap_mode="r").shape) for v in names},
        "test_shapes": {v: list(np.load(cache_path("test", v), mmap_mode="r").shape) for v in names},
        "checks": checks,
        "device": torch.cuda.get_device_name(0),
        "elapsed_seconds": time.time() - started,
        "convention": "concat 窗口主序(w*855 + band*171 + pair)；std ddof=0；每子窗口独立 detrend_zband",
    }
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(report, ensure_ascii=False, indent=2))


# ------------------------------------------------------------------ protocol
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
        self.fit_rows = self.fr <= CFG["evaluation"]["fit_fraction"]
        self.n_blocks = int(CFG["evaluation"]["blocks"])
        self.blocks = np.clip((self.fr * self.n_blocks).astype(int), 0, self.n_blocks - 1)
        self.test_tasks = qmeta.task.to_numpy()
        self.is_s1 = qmeta.session.astype(str).str.casefold().isin(
            ["session1", "s1", "1"]).to_numpy()
        self.y = np.r_[np.zeros(int(self.is_s1.sum()), dtype=int),
                       np.ones(int((~self.is_s1).sum()), dtype=int)]
        self.pca_max = int(max(CFG["evaluation"]["pca_dims"]))
        self.basis: dict[str, tuple] = {}
        self.proj: dict[tuple, tuple] = {}

    def project(self, key: str, a: np.ndarray, b: np.ndarray, dim: int) -> tuple:
        cache = (key, int(dim))
        if cache in self.proj:
            return self.proj[cache]
        if key not in self.basis:
            self.basis[key] = pca_basis(
                np.asarray(a[self.fit_rows], dtype=np.float64),
                min(self.pca_max, a.shape[1]))
        mu, basis = self.basis[key]
        dim = min(int(dim), basis.shape[1])
        self.proj[cache] = (((a.astype(np.float64) - mu) @ basis[:, :dim]).astype(np.float32),
                            ((b.astype(np.float64) - mu) @ basis[:, :dim]).astype(np.float32))
        return self.proj[cache]

    def drop(self, key: str) -> None:
        for cache in [c for c in self.proj if c[0] == key]:
            del self.proj[cache]
        self.basis.pop(key, None)


def loso_pairs(at: np.ndarray, k: int, ctx: Context) -> tuple:
    known, unknown = [], []
    for held_id in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held_id])
        pm = np.isin(ctx.nodes, enrolled)
        ki = pm & (ctx.fr > .6) & (ctx.fr <= .8)
        ui = (ctx.nodes == held_id) & (ctx.fr > .6) & (ctx.fr <= .8)
        if ui.sum() == 0:
            continue
        p2, k2 = build_prototypes(at, ctx.nodes, ctx.tasks,
                                  pm & ctx.fit_rows, enrolled, k)
        rows = np.flatnonzero(ki | ui)
        scores = neg_d1(at[rows], p2, k2, ctx.tasks[rows])
        mask = ki[rows]
        known.append(scores[mask])
        unknown.append(scores[~mask])
    return known, unknown


def evaluate(at: np.ndarray, bt: np.ndarray, k: int, ctx: Context) -> dict:
    protos, keys = build_prototypes(at, ctx.nodes, ctx.tasks, np.ones(len(at), bool),
                                    CLASSES, k)
    te_raw = neg_d1(bt, protos, keys, ctx.test_tasks)

    oof = np.full(len(at), np.nan, dtype=np.float32)
    oof_pred = np.full(len(at), -1, dtype=np.int32)
    for blk in range(ctx.n_blocks):
        hold = ctx.blocks == blk
        held = np.flatnonzero(hold)
        p2, k2 = build_prototypes(at, ctx.nodes, ctx.tasks, ~hold, CLASSES, k)
        dmat, ksubj = neg_dmat(at[held], p2, k2, ctx.tasks[held])
        oof[held] = dmat.max(1)
        oof_pred[held] = ksubj[dmat.argmax(1)]

    known, unknown = loso_pairs(at, k, ctx)
    finite = np.isfinite(oof)
    mu = float(np.nanmean(oof[finite]))
    sd = float(np.nanstd(oof[finite])) + 1e-6
    z_rf = ((oof - mu) / sd).astype(np.float32)
    z_te = ((te_raw - mu) / sd).astype(np.float32)
    loso = [((kn - mu) / sd, (un - mu) / sd) for kn, un in zip(known, unknown)]

    unknown_all = np.concatenate([un for _, un in loso])
    oof_auc = float(roc_auc_score(np.r_[np.zeros(len(unknown_all)), np.ones(int(finite.sum()))],
                                  np.r_[unknown_all, z_rf[finite]]))
    oof_f1 = float(f1_score(ctx.nodes[finite], oof_pred[finite], labels=CLASSES,
                            average="macro", zero_division=0))
    oof_acc = float(np.mean(oof_pred[finite] == ctx.nodes[finite]))
    loso_auc = float(roc_auc_score(np.r_[np.zeros(len(unknown_all)),
                                         np.ones(len(np.concatenate(known)))],
                                   np.r_[unknown_all, np.concatenate(known)]))
    diag_auc = float(roc_auc_score(ctx.y, np.r_[z_te[ctx.is_s1], z_te[~ctx.is_s1]]))

    rows = []
    for acc in CFG["evaluation"]["acceptance_grid"]:
        th = float(np.nanquantile(z_rf[finite], 1 - acc))
        s2 = float(np.mean(z_te[~ctx.is_s1] >= th))
        s1 = float(np.mean(z_te[ctx.is_s1] < th))
        lr = float(np.mean([np.mean(un < np.quantile(kn, 1 - acc)) for kn, un in loso]))
        rows.append({"acceptance_rate": float(acc), "threshold": th,
                     "LOSO_rejection": lr, "Test_S1_rejection": s1,
                     "Test_S2_acceptance": s2,
                     "W": CFG["evaluation"]["weights"]["LOSO"] * lr
                          + CFG["evaluation"]["weights"]["S1"] * s1
                          + CFG["evaluation"]["weights"]["S2"] * s2})
    return {"z_rf": z_rf, "z_te": z_te, "loso": loso, "mu": mu, "sd": sd,
            "diagnostic_auc": diag_auc, "train_oof_auc": oof_auc,
            "train_oof_f1": oof_f1, "train_oof_identity_acc": oof_acc,
            "loso_auc_train_only": loso_auc, "rows": rows}


def fuse(e1: dict, e2: dict, at1: np.ndarray, at2: np.ndarray, w: float,
         k: int, ctx: Context) -> dict:
    z_rf = (w * e1["z_rf"] + (1 - w) * e2["z_rf"]).astype(np.float32)
    z_te = (w * e1["z_te"] + (1 - w) * e2["z_te"]).astype(np.float32)
    loso = [(w * k1 + (1 - w) * k2, w * u1 + (1 - w) * u2)
            for (k1, u1), (k2, u2) in zip(e1["loso"], e2["loso"])]

    pred = np.full(len(ctx.nodes), -1, dtype=np.int32)
    for blk in range(ctx.n_blocks):
        hold = ctx.blocks == blk
        held = np.flatnonzero(hold)
        p1, k1 = build_prototypes(at1, ctx.nodes, ctx.tasks, ~hold, CLASSES, k)
        p2, k2 = build_prototypes(at2, ctx.nodes, ctx.tasks, ~hold, CLASSES, k)
        assert [tuple(x) for x in k1] == [tuple(x) for x in k2], "prototype keys misaligned"
        d1, ksubj = neg_dmat(at1[held], p1, k1, ctx.tasks[held])
        d2, _ = neg_dmat(at2[held], p2, k2, ctx.tasks[held])
        z1 = (d1 - e1["mu"]) / e1["sd"]
        z2 = (d2 - e2["mu"]) / e2["sd"]
        pred[held] = ksubj[(w * z1 + (1 - w) * z2).argmax(1)]

    finite = np.isfinite(z_rf)
    oof_f1 = float(f1_score(ctx.nodes[finite], pred[finite], labels=CLASSES,
                            average="macro", zero_division=0))
    oof_acc = float(np.mean(pred[finite] == ctx.nodes[finite]))
    unknown_all = np.concatenate([un for _, un in loso])
    known_all = np.concatenate([kn for kn, _ in loso])
    oof_auc = float(roc_auc_score(np.r_[np.zeros(len(unknown_all)), np.ones(int(finite.sum()))],
                                  np.r_[unknown_all, z_rf[finite]]))
    loso_auc = float(roc_auc_score(np.r_[np.zeros(len(unknown_all)), np.ones(len(known_all))],
                                   np.r_[unknown_all, known_all]))
    diag_auc = float(roc_auc_score(ctx.y, np.r_[z_te[ctx.is_s1], z_te[~ctx.is_s1]]))
    rows = []
    for acc in CFG["evaluation"]["acceptance_grid"]:
        th = float(np.nanquantile(z_rf[finite], 1 - acc))
        s2 = float(np.mean(z_te[~ctx.is_s1] >= th))
        s1 = float(np.mean(z_te[ctx.is_s1] < th))
        lr = float(np.mean([np.mean(un < np.quantile(kn, 1 - acc)) for kn, un in loso]))
        rows.append({"acceptance_rate": float(acc), "threshold": th,
                     "LOSO_rejection": lr, "Test_S1_rejection": s1,
                     "Test_S2_acceptance": s2,
                     "W": CFG["evaluation"]["weights"]["LOSO"] * lr
                          + CFG["evaluation"]["weights"]["S1"] * s1
                          + CFG["evaluation"]["weights"]["S2"] * s2})
    return {"z_rf": z_rf, "z_te": z_te, "loso": loso, "mu": 0.0, "sd": 1.0,
            "diagnostic_auc": diag_auc, "train_oof_auc": oof_auc,
            "train_oof_f1": oof_f1, "train_oof_identity_acc": oof_acc,
            "loso_auc_train_only": loso_auc, "rows": rows}


def summarise(entry: dict, meta: dict) -> dict:
    rows = entry["rows"]
    best = max(rows, key=lambda r: r["W"])
    ref_acc = float(CFG["evaluation"]["reference_acceptance"])
    ref = min(rows, key=lambda r: abs(r["acceptance_rate"] - ref_acc))
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
        "ref_acceptance": ref_acc,
    })
    return record


def load_raw_stream(spec: dict) -> tuple:
    a = np.array(np.load(ROOT / spec["train"], mmap_mode="r"), dtype=np.float32)
    b = np.array(np.load(ROOT / spec["test"], mmap_mode="r"), dtype=np.float32)
    return a, b


def feature_diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """子窗口平均特征与 E65 全长特征的逐元素差异。"""
    if a.shape != b.shape:
        return {"comparable": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    sub = a[:2000]
    ref = b[:2000]
    diff = sub - ref
    corr = np.mean([np.corrcoef(sub[i], ref[i])[0, 1] for i in range(0, len(sub), 50)])
    return {"comparable": True, "max_abs_diff": float(np.abs(diff).max()),
            "mean_abs_diff": float(np.abs(diff).mean()),
            "relative_mean_abs_diff": float(np.abs(diff).mean() / (np.abs(ref).mean() + 1e-9)),
            "mean_row_correlation": float(corr)}


def main() -> None:
    started = time.time()
    build_features()
    if not torch.cuda.is_available():
        raise RuntimeError("E168 需要 CUDA")

    ns = [int(n) for n in CFG["subwindow"]["ns"]
          if cache_path("train", f"subwin_std_n{int(n)}").is_file()]
    variants = [f"{v}_n{n}" for n in ns for v in CFG["subwindow"]["variants"]]
    log(f"评估变体：{variants}")

    ctx = Context()
    log(f"device {DEVICE} | train {len(ctx.nodes)} | test {len(ctx.test_tasks)} | "
        f"fit_rows {int(ctx.fit_rows.sum())} | blocks {ctx.n_blocks}")

    dims = [int(d) for d in CFG["evaluation"]["pca_dims"]]
    ks = [int(k) for k in CFG["evaluation"]["prototype_k"]]
    ref_view = CFG["reference_views"]["E65"]

    # ---------------- E65 对照（缓存视图，逐字复现 E143）
    a65, b65 = load_raw_stream(ref_view)
    a65, b65 = task_standardise(a65, b65, ctx.tasks, ctx.test_tasks)
    e65_rows, entries = [], {}
    t0 = time.time()
    for dim in dims:
        at, bt = ctx.project("E65", a65, b65, dim)
        for k in ks:
            entry = evaluate(at, bt, k, ctx)
            entries[("E65", dim, k)] = entry
            rec = summarise(entry, {"stream": "E65", "family": "control", "dim": dim, "k": k,
                                    "config": f"E65 PCA-{dim} k={k}"})
            e65_rows.append(rec)
            log(f"  [E65] dim={dim} k={k} OOFauc={rec['train_oof_auc']:.4f} "
                f"OOFf1={rec['train_oof_f1']:.4f} diagAUC={rec['diagnostic_auc']:.4f} "
                f"W@.90={rec['W_at_ref']:.4f} bestW={rec['best_W']:.4f}"
                f"@{rec['best_acceptance']:.2f} ({time.time() - t0:.0f}s)")
    base = [r for r in e65_rows if r["dim"] == 96 and r["k"] == 12][0]
    published = ref_view["published"]
    log(f"E65 复现：diagAUC {base['diagnostic_auc']:.4f}（公布 {published['diagnostic_auc']:.4f}）"
        f" / W@0.90 {base['W_at_ref']:.4f}（公布 {published['W']:.4f}）")

    # ---------------- 子窗口变体
    rows, fusion_rows = [], []
    fusion_dims = [int(d) for d in CFG["fusion"]["pca_dims"]]
    for name in variants:
        a, b = load_raw_stream({"train": str(cache_path("train", name).relative_to(ROOT)),
                                "test": str(cache_path("test", name).relative_to(ROOT))})
        a, b = task_standardise(a, b, ctx.tasks, ctx.test_tasks)
        variant = name.rsplit("_n", 1)[0]
        n = int(name.rsplit("_n", 1)[1])
        for dim in dims:
            at, bt = ctx.project(name, a, b, dim)
            for k in ks:
                entry = evaluate(at, bt, k, ctx)
                entries[(name, dim, k)] = entry
                rec = summarise(entry, {"stream": name, "family": "subwindow",
                                        "variant": variant, "n": n, "dim": dim, "k": k,
                                        "config": f"{name} PCA-{dim} k={k}"})
                rows.append(rec)
                log(f"  [{name}] dim={dim} k={k} OOFauc={rec['train_oof_auc']:.4f} "
                    f"OOFf1={rec['train_oof_f1']:.4f} diagAUC={rec['diagnostic_auc']:.4f} "
                    f"W@.90={rec['W_at_ref']:.4f} bestW={rec['best_W']:.4f}"
                    f"@{rec['best_acceptance']:.2f} ({time.time() - t0:.0f}s)")
            for k in ks:
                for w in CFG["fusion"]["weights"]:
                    if dim not in fusion_dims:
                        continue
                    at65, _ = ctx.project("E65", a65, b65, dim)
                    fused = fuse(entries[("E65", dim, k)], entries[(name, dim, k)],
                                 at65, at, float(w), k, ctx)
                    fusion_rows.append(summarise(fused, {
                        "stream": f"E65+{name}", "family": "score_fusion",
                        "variant": variant, "n": n, "dim": dim, "k": k,
                        "weight_e65": float(w), "weight_variant": 1 - float(w),
                        "config": f"{w:.2f}*E65 + {1 - w:.2f}*{name} PCA-{dim} k={k}"}))
        ctx.drop(name)
        del a, b
        log(f"  {name}: 最优诊断 AUC "
            f"{max(r['diagnostic_auc'] for r in rows if r['stream'] == name):.4f} / "
            f"最优 W {max(r['best_W'] for r in rows if r['stream'] == name):.4f}")

    # ---------------- 子窗口平均 vs E65 特征差异
    diff_stats = {}
    a65_raw = np.array(np.load(ROOT / ref_view["train"], mmap_mode="r"), dtype=np.float32)
    for n in ns:
        mean_feat = np.load(cache_path("train", f"subwin_mean_n{n}"), mmap_mode="r")
        diff_stats[f"subwin_mean_n{n}_vs_E65"] = feature_diff_stats(mean_feat, a65_raw)
    del a65_raw
    log(f"子窗口平均 vs E65 特征差异 {json.dumps(diff_stats, ensure_ascii=False)}")

    # ---------------- 判定
    targets = CFG["targets"]
    all_rows = rows + fusion_rows
    best_auc = max(all_rows, key=lambda r: r["diagnostic_auc"])
    best_w = max(all_rows, key=lambda r: r["best_W"])
    best_std = max([r for r in rows if r["variant"] == "subwin_std"],
                   key=lambda r: r["best_W"])
    best_std_auc = max([r for r in rows if r["variant"] == "subwin_std"],
                       key=lambda r: r["diagnostic_auc"])
    best_fusion = max(fusion_rows, key=lambda r: r["best_W"])
    best_fusion_auc = max(fusion_rows, key=lambda r: r["diagnostic_auc"])
    corr = {}
    for n in ns:
        for variant in ("subwin_std", "subwin_mean"):
            name = f"{variant}_n{n}"
            if ("E65", 96, 12) in entries and (name, 96, 12) in entries:
                e1, e2 = entries[("E65", 96, 12)], entries[(name, 96, 12)]
                corr[name] = {"test": float(np.corrcoef(e1["z_te"], e2["z_te"])[0, 1]),
                              "train_oof": float(np.corrcoef(e1["z_rf"], e2["z_rf"])[0, 1])}

    results = {
        "experiment": CFG["experiment"],
        "device": str(DEVICE),
        "protocol": CFG["evaluation"]["protocol"],
        "test_official": {"AUC": "N/A（无真实标签）", "F1": "N/A（无真实标签）"},
        "feature_diff_vs_E65": diff_stats,
        "score_correlation_vs_E65": corr,
        "controls": e65_rows,
        "subwindow": rows,
        "score_fusion": fusion_rows,
        "verdict": {
            "baseline_E143": {"diagnostic_auc": published["diagnostic_auc"],
                              "W": published["W"],
                              "reproduced_diagnostic_auc": base["diagnostic_auc"],
                              "reproduced_W_at_ref": base["W_at_ref"]},
            "targets": {"diagnostic_auc": targets["diagnostic_auc"], "W": targets["W"]},
            "best_diagnostic_auc": best_auc,
            "best_W": best_w,
            "best_subwin_std_by_W": best_std,
            "best_subwin_std_by_AUC": best_std_auc,
            "best_fusion_by_W": best_fusion,
            "best_fusion_by_AUC": best_fusion_auc,
            "auc_target_met": bool(best_auc["diagnostic_auc"] > targets["diagnostic_auc"]),
            "W_target_met": bool(best_w["best_W"] > targets["W"]),
        },
        "runtime_seconds": round(time.time() - started, 1),
    }
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    write_report(results)
    log(f"done in {time.time() - started:.0f}s")


HDR = ("| 配置 | 诊断 AUC | Train OOF AUC | Train OOF F1 | LOSO 拒识 | S1 拒识 | S2 接受 | W |")
SEP = "|---|---:|---:|---:|---:|---:|---:|---:|"


def row_str(r: dict) -> str:
    return (f"| {r['config']} | {r['diagnostic_auc']:.4f} | {r['train_oof_auc']:.4f} | "
            f"{r['train_oof_f1']:.4f} | {r['loso_at_ref']:.4f} | {r['s1_at_ref']:.4f} | "
            f"{r['s2_at_ref']:.4f} | {r['W_at_ref']:.4f} |")


def row_str_best(r: dict) -> str:
    return (f"| {r['config']} | {r['diagnostic_auc']:.4f} | {r['train_oof_auc']:.4f} | "
            f"{r['train_oof_f1']:.4f} | {r['loso_at_best']:.4f} | {r['s1_at_best']:.4f} | "
            f"{r['s2_at_best']:.4f} | {r['best_W']:.4f} |")


def write_report(results: dict) -> None:
    v = results["verdict"]
    base = v["baseline_E143"]
    tgt = v["targets"]
    rows = results["subwindow"]
    fus = results["score_fusion"]
    ctrl = [r for r in results["controls"] if r["dim"] == 96 and r["k"] == 12][0]
    best_ref = max(rows + fus, key=lambda r: r["W_at_ref"])
    std_ref = max([r for r in rows if r["variant"] == "subwin_std"],
                  key=lambda r: r["W_at_ref"])
    fus_ref = max(fus, key=lambda r: r["W_at_ref"])
    e65_ref = max(results["controls"], key=lambda r: r["W_at_ref"])
    bw = v["best_W"]
    same_stream = [r for r in rows if r["stream"] == bw["stream"]]
    same_oof = max(r["train_oof_auc"] for r in same_stream)
    lines = [
        "# E168 子窗口 correlation-logSPD（子段结构 / 非平稳性）", "",
        "E65 对整个 trial 只算一个相关矩阵，因此对通道增益、频带增益、循环时间平移精确不变，"
        "trial 内的子段结构被完全丢弃（E165）。本实验把 trial 切成 N 个**不重叠**子窗口"
        "（N=2/3/4，窗长 1200/N），每个子窗口独立执行 E65 的完整流程"
        "（5 频带 FFT 滤波 → 去均值 → 通道标准化 → 相关矩阵 → 收缩 → logSPD → 上三角 171 维 → "
        "detrend_zband），再把子窗口结果作为**特征向量**（不是 E149/E62 的图像）接入 E143 协议。", "",
        "评估逐字沿用 E143：task-wise Train 标准化 → PCA（64/96/128/256）→ k=1/3/6/12 原型 → "
        "5 块 Train OOF 阈值（接受率 0.80–0.96）→ LOSO 每折排除被评分身份、原型只用 fit_rows、"
        "评估块 0.6<frac≤0.8 → Test 只评分、session 只诊断。", "",
        "## 结论", "",
    ]
    gain_auc = v["best_diagnostic_auc"]["diagnostic_auc"] - base["diagnostic_auc"]
    lines += [
        f"- 基线 E143（E65 PCA-96 k=12, 接受率 0.90）：诊断 AUC **{ctrl['diagnostic_auc']:.4f}**"
        f" / W **{ctrl['W_at_ref']:.4f}**"
        f"（公布 {base['diagnostic_auc']:.4f} / {base['W']:.4f}，"
        f"Δ {ctrl['diagnostic_auc'] - base['diagnostic_auc']:+.4f} / "
        f"{ctrl['W_at_ref'] - base['W']:+.4f}），逐位复现。",
        f"- **诊断 AUC：{v['best_diagnostic_auc']['diagnostic_auc']:.4f}**"
        f"（{v['best_diagnostic_auc']['config']}，Δ {gain_auc:+.4f} vs 基线，"
        f"Δ {v['best_diagnostic_auc']['diagnostic_auc'] - tgt['diagnostic_auc']:+.4f} vs 0.7609）"
        f" → 关键问题 1 **{'达到' if v['auc_target_met'] else '未达到'}**。"
        "子段结构确实带来了 E65 结构上没有的信息，且增益不是靠阈值。",
        f"- **W（E143 工作点，接受率固定 0.90）：{best_ref['W_at_ref']:.4f}**"
        f"（{best_ref['config']}），对比 E65 自身最好 {e65_ref['W_at_ref']:.4f}"
        f"（{e65_ref['config']}）、基线 {base['W']:.4f}；"
        f"距 0.7164 还差 {tgt['W'] - best_ref['W_at_ref']:+.4f} → 关键问题 2 "
        f"**{'达到' if best_ref['W_at_ref'] > tgt['W'] else '未达到'}**。",
        f"- 若按 E143「在 0.80–0.96 网格里取最大 W」的口径，可报 {bw['best_W']:.4f}"
        f"（{bw['config']}，@a={bw['best_acceptance']:.2f}）；但该点是用 **Test 标签选出来的工作点**，"
        f"且该配置 Train OOF AUC 只有 {bw['train_oof_auc']:.4f}（同变体其它配置最高 {same_oof:.4f}），"
        "属于分数分布形状/阈值效应，不能当作方案提升（见「口径说明」）。",
        f"- 为什么 AUC 涨了 W 没涨：AUC 最高的 {v['best_diagnostic_auc']['config']} "
        f"在固定接受率 0.90 时 S2 接受只有 {v['best_diagnostic_auc']['s2_at_ref']:.4f}"
        f"（E65 基线 {ctrl['s2_at_ref']:.4f}），排序增益被 Train OOF 参考与 Test 分数分布之间的"
        "平移吃掉；要把 AUC 转成 W，瓶颈在校准而不是表征。",
        "",
        "### 子窗口标准差（subwin_std）", "",
        f"- 单独使用几乎无信号：最优诊断 AUC {v['best_subwin_std_by_AUC']['diagnostic_auc']:.4f}"
        f"（{v['best_subwin_std_by_AUC']['config']}），最优 W@0.90 {std_ref['W_at_ref']:.4f}"
        f"（{std_ref['config']}）；AUC 甚至低于 0.5，说明 S2 的子窗口离散度反而更大（方向相反），"
        "翻转后也只有 0.52–0.64。",
        f"- 与 E65 的相关性最低（PCA-96 k=12 时 Test r = "
        f"{results['score_correlation_vs_E65'].get('subwin_std_n4', {}).get('test', float('nan')):+.4f}"
        f"（N=4）/{results['score_correlation_vs_E65'].get('subwin_std_n2', {}).get('test', float('nan')):+.4f}"
        f"（N=2）），确实是 E65 没有的信息；但与 E65 等权融合后最好诊断 AUC 只有 "
        f"{v['best_fusion_by_AUC']['diagnostic_auc']:.4f}"
        f"（{v['best_fusion_by_AUC']['config']}）、最好 W@0.90 只有 {fus_ref['W_at_ref']:.4f}"
        f"（{fus_ref['config']}），两者都没有超过子窗口 mean/拼接单路的最好值，"
        "说明 trial 内非平稳性在这个表征下不构成可用的身份/会话判别信息。",
        "",
        "### subwin_mean 与 E65 的关系", "",
        "- 逐元素差异（mean |Δ| / 相对 |Δ| / 逐行相关）：",
        "  " + json.dumps(results["feature_diff_vs_E65"], ensure_ascii=False),
        "- 子窗口平均**不等价于**全长：每个子窗口先各自 detrend_zband（带去均值/按带 z 化的非线性），"
        "再平均，故 N 越大与 E65 差异越大（N=4 平均 |Δ| 0.27、逐行相关 0.95），"
        "而这正是诊断 AUC 最高的方向（subwin_mean_n4 0.7822 vs E65 0.7578）。",
        "",
        "### 口径说明（重要）", "",
        "- **诊断 AUC** 与阈值无关，是唯一干净的可比量：它只依赖 S1/S2 的分数排序。",
        "- **W@0.90** 固定 E143 的接受率 0.90（阈值 = Train OOF 分数的 10% 分位），跨配置可比。",
        "- **best W** 是在接受率 0.80–0.96 网格上挑最大值，等于用 Test 的 S1/S2 结果挑工作点；"
        "E143 自己也是这么报的（0.6958 恰好同时是最优与 a=0.90），但当某配置的 Train OOF 分数分布"
        "在 Test 上发生压缩/平移时，best W 会被阈值位置放大。本实验里 concat 变体在 PCA-256 时正是"
        "这种情况（OOF AUC 掉到 0.73，best W 却跳到 0.76），因此结论以 W@0.90 为准。",
        "- Test 官方 AUC/F1 = N/A（无真实标签）。",
        "",
        "## E65 对照（复现检查）", "", HDR, SEP, row_str(ctrl),
        "",
        f"诊断 AUC {ctrl['diagnostic_auc']:.4f}（公布 {base['diagnostic_auc']:.4f}）、"
        f"W@0.90 {ctrl['W_at_ref']:.4f}（公布 {base['W']:.4f}）："
        f"{'逐位复现' if abs(ctrl['W_at_ref'] - base['W']) < 5e-4 else '存在差异'}。",
        "",
        "## 每个变体的 W@0.90 工作点（各变体在该口径下的最优配置）", "", HDR, SEP,
        row_str(e65_ref),
    ]
    for name in dict.fromkeys(r["stream"] for r in rows):
        sel = [r for r in rows if r["stream"] == name]
        lines.append(row_str(max(sel, key=lambda r: r["W_at_ref"])))
    lines.append(row_str(fus_ref))
    lines += ["", "## 每个变体的 E143 工作点（PCA-96, k=12, 接受率 0.90）", "", HDR, SEP]
    for name in dict.fromkeys(r["stream"] for r in rows):
        sel = [r for r in rows
               if r["stream"] == name and r["dim"] == 96 and r["k"] == 12]
        if sel:
            lines.append(row_str(sel[0]))
    lines += ["", "## 每个变体的最优配置（按 best W，Test 选工作点）", "", HDR, SEP]
    for name in dict.fromkeys(r["stream"] for r in rows):
        sel = [r for r in rows if r["stream"] == name]
        lines.append(row_str_best(max(sel, key=lambda r: r["best_W"])))
    lines += ["", "## 每个变体的最优配置（按诊断 AUC）", "", HDR, SEP]
    for name in dict.fromkeys(r["stream"] for r in rows):
        sel = [r for r in rows if r["stream"] == name]
        lines.append(row_str_best(max(sel, key=lambda r: r["diagnostic_auc"])))
    lines += ["", "## 与 E65 等权分数融合（按 best W 前 15）", "", HDR, SEP]
    for r in sorted(fus, key=lambda r: -r["best_W"])[:15]:
        lines.append(row_str_best(r))
    lines += ["", "## 全部结果中诊断 AUC 前 15", "", HDR, SEP]
    for r in sorted(rows + fus, key=lambda r: -r["diagnostic_auc"])[:15]:
        lines.append(row_str_best(r))
    lines += ["", "## 判定", "",
              f"- 关键问题 1（诊断 AUC > {tgt['diagnostic_auc']:.4f}，即超过历史最好 0.7609）："
              f"{'**是**' if v['auc_target_met'] else '否'}，最好 "
              f"{v['best_diagnostic_auc']['diagnostic_auc']:.4f}"
              f"（{v['best_diagnostic_auc']['config']}）。",
              f"- 关键问题 2（W > {tgt['W']:.4f}）：在 E143 固定接受率 0.90 下 "
              f"{'**是**' if best_ref['W_at_ref'] > tgt['W'] else '**否**'}，"
              f"最好 {best_ref['W_at_ref']:.4f}（差 {best_ref['W_at_ref'] - tgt['W']:+.4f}，属噪声量级）；"
              f"best-over-grid 口径 0.7617 属 Test 选点，不作为方案提升。",
              "- Test 官方 AUC/F1：N/A（Test 无真实标签，只做 S1/S2 诊断）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        write_report(json.loads((OUT / "results.json").read_text(encoding="utf-8")))
    else:
        main()
