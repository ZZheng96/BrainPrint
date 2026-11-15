"""E178: measure the five gates E177 left unmeasured on the best-W E170 score.

Gates filled here
  #1 in-sample Train accuracy (capacity to fit)
  #5 pseudo-unknown AUROC under the verbatim LOSO protocol (pm & fit_rows)
  #6 U@95 / U@90 / U@98 on the same LOSO scores
  #7 RA and #8 ASafeR against a self-built end-to-end differentiable PGD surrogate
     (Train eval window 0.6 < frac <= 0.8, identity labels available; Test has none)

Everything is Train-only. No Test file and no Session column is read. The fused
score is frozen: 0.5*z(E65 PCA-64) + 0.1*z(logbp PCA-32) + 0.4*z(E162 PCA-64),
k=1 prototypes, acceptance 0.88.

Stages:  --stage gates | attack | all (default all)
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.io import loadmat
from sklearn.metrics import f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FT = torch.float32
TASKS = CFG["tasks"]
CLASSES = np.arange(1, 21, dtype=int)
NAMES = list(CFG["streams"].keys())
WEIGHTS = np.asarray(CFG["fusion"]["weights"], dtype=np.float64)
DIMS = {n: int(CFG["streams"][n]["pca_dim"]) for n in NAMES}
FE = CFG["features"]
N = int(CFG["data"]["samples_per_trial"])
CH = int(CFG["data"]["channels"])
FS = int(CFG["data"]["sampling_rate_hz"])
BANDS = FE["bands_hz"]
NB = len(BANDS)
ACCEPT = float(CFG["audited_scheme"]["acceptance_rate"])
OFF = -1e9


class Tee:
    def __init__(self, path: Path):
        self.file = open(path, "a", encoding="utf-8")
        self.stdout = sys.__stdout__

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


sys.stdout = Tee(OUT / "run.log")
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- primitives
def frac_of(meta: pd.DataFrame) -> np.ndarray:
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def pca_basis(x: np.ndarray, dim: int) -> tuple:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(0)
    xc = x - mu
    values, vectors = np.linalg.eigh(xc.T @ xc / max(len(xc) - 1, 1))
    return mu, vectors[:, np.argsort(-values)][:, : min(int(dim), x.shape[1])]


def task_stats(a: np.ndarray, tasks: np.ndarray) -> dict:
    """Per-task z-score statistics from all Train rows of that task (E170 rule)."""
    out = {}
    for task in TASKS:
        ref = tasks == task
        out[task] = (a[ref].mean(0), a[ref].std(0) + 1e-6)
    return out


def apply_stats(x: np.ndarray, tasks: np.ndarray, stats: dict) -> np.ndarray:
    out = x.astype(np.float64).copy()
    for task in TASKS:
        ref = tasks == task
        mu, sd = stats[task]
        out[ref] = (out[ref] - mu) / sd
    return out.astype(np.float32)


def build_prototypes(feat: np.ndarray, subjects: np.ndarray, tasks: np.ndarray,
                     mask: np.ndarray, classes: np.ndarray) -> tuple:
    """k=1 mean prototype per (task, identity); E170 k-means with k=1 is the mean."""
    protos, keys = [], []
    for task in TASKS:
        for subject in classes:
            sel = mask & (tasks == task) & (subjects == subject)
            if sel.sum() == 0:
                continue
            protos.append(np.asarray(feat[sel], dtype=np.float32).mean(0))
            keys.append((task, int(subject)))
    return np.stack(protos).astype(np.float32), keys


def neg_dmat(feat: np.ndarray, protos: np.ndarray, keys: list,
             tasks: np.ndarray, rows: np.ndarray = None) -> tuple:
    """row x prototype negative-distance matrix, off-task entries = -inf."""
    key_task = np.asarray([k[0] for k in keys])
    key_subject = np.asarray([k[1] for k in keys], dtype=np.int32)
    rows = np.arange(len(feat)) if rows is None else rows
    out = np.full((len(rows), len(keys)), -np.inf, dtype=np.float32)
    for task in TASKS:
        idx = rows[tasks[rows] == task]
        cols = np.flatnonzero(key_task == task)
        if len(idx) == 0 or len(cols) == 0:
            continue
        p = torch.as_tensor(protos[cols], dtype=FT, device=DEVICE)
        pos = np.searchsorted(rows, idx)
        for s in range(0, len(idx), 2048):
            chunk = idx[s:s + 2048]
            q = torch.as_tensor(np.asarray(feat[chunk], dtype=np.float32), device=DEVICE)
            out[np.ix_(pos[s:s + 2048], cols)] = (-torch.cdist(q, p)).cpu().numpy()
    return out, key_subject


def neg_d1(feat: np.ndarray, protos: np.ndarray, keys: list,
           tasks: np.ndarray) -> np.ndarray:
    out = np.full(len(feat), np.nan, dtype=np.float32)
    key_task = np.asarray([k[0] for k in keys])
    for task in TASKS:
        idx = np.flatnonzero(tasks == task)
        cols = np.flatnonzero(key_task == task)
        if len(idx) == 0 or len(cols) == 0:
            continue
        p = torch.as_tensor(protos[cols], dtype=FT, device=DEVICE)
        for s in range(0, len(idx), 2048):
            chunk = idx[s:s + 2048]
            q = torch.as_tensor(np.asarray(feat[chunk], dtype=np.float32), device=DEVICE)
            out[chunk] = (-torch.cdist(q, p)).max(1).values.cpu().numpy()
    return out


# ---------------------------------------------------------------- experiment state
class Ctx:
    def __init__(self) -> None:
        meta = pd.read_csv(ROOT / CFG["data"]["train_metadata"])
        self.meta = meta
        self.nodes = meta.subject.to_numpy(int)
        self.tasks = meta.task.to_numpy()
        self.paths = meta.path.to_numpy()
        self.frac = frac_of(meta)
        self.fit_rows = self.frac <= CFG["protocol"]["fit_fraction"]
        self.blocks = np.clip((self.frac * CFG["protocol"]["blocks"]).astype(int), 0,
                              CFG["protocol"]["blocks"] - 1)
        self.window = (self.frac > CFG["protocol"]["fit_fraction"]) & (self.frac <= 0.8)
        self.window_rows = np.flatnonzero(self.window)


def load_stream_features(ctx: Ctx) -> dict:
    """Cached raw stream features -> task-wise z-score -> Train-only PCA."""
    out = {}
    for name in NAMES:
        spec = CFG["streams"][name]
        raw = np.asarray(np.load(ROOT / spec["train"]), dtype=np.float32)
        assert raw.shape == (len(ctx.nodes), int(spec["raw_dim"])), (name, raw.shape)
        stats = task_stats(raw, ctx.tasks)
        std = apply_stats(raw, ctx.tasks, stats)
        mu, basis = pca_basis(std[ctx.fit_rows], DIMS[name])
        proj = ((std.astype(np.float64) - mu) @ basis).astype(np.float32)
        out[name] = {"stats": stats, "pca_mu": mu, "basis": basis, "proj": proj}
        log(f"stream {name}: raw {raw.shape} -> PCA-{DIMS[name]} "
            f"(basis from {int(ctx.fit_rows.sum())} fit rows)")
    return out


def stream_scores(ctx: Ctx, name: str, proj: np.ndarray) -> dict:
    """E170 evaluate() for one stream at k=1: in-sample, 5-block OOF, 20-fold LOSO."""
    # in-sample: prototypes from every Train row
    protos, keys = build_prototypes(proj, ctx.nodes, ctx.tasks,
                                   np.ones(len(proj), bool), CLASSES)
    zmat_all, ksubj = neg_dmat(proj, protos, keys, ctx.tasks)

    oof = np.full(len(proj), np.nan, dtype=np.float32)
    oof_pred = np.full(len(proj), -1, dtype=np.int32)
    for block in range(int(CFG["protocol"]["blocks"])):
        hold = ctx.blocks == block
        held = np.flatnonzero(hold)
        p2, k2 = build_prototypes(proj, ctx.nodes, ctx.tasks, ~hold, CLASSES)
        dmat, ksub = neg_dmat(proj, p2, k2, ctx.tasks, held)
        oof[held] = dmat.max(1)
        oof_pred[held] = ksub[dmat.argmax(1)]

    loso = []
    for held_id in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held_id])
        pm = np.isin(ctx.nodes, enrolled)
        p2, k2 = build_prototypes(proj, ctx.nodes, ctx.tasks, pm & ctx.fit_rows, enrolled)
        known = ctx.window_rows[pm[ctx.window_rows]]
        unknown = ctx.window_rows[ctx.nodes[ctx.window_rows] == held_id]
        sc = neg_d1(proj[np.r_[known, unknown]], p2, k2,
                    ctx.tasks[np.r_[known, unknown]])
        loso.append((sc[:len(known)], sc[len(known):]))

    finite = np.isfinite(oof)
    mu = float(np.nanmean(oof[finite]))
    sd = float(np.nanstd(oof[finite])) + 1e-6
    return {"mu": mu, "sd": sd,
            "z_oof": ((oof - mu) / sd).astype(np.float32), "oof_finite": finite,
            "oof_pred": oof_pred,
            "loso": [(((kn - mu) / sd).astype(np.float32),
                      ((un - mu) / sd).astype(np.float32)) for kn, un in loso],
            "zmat_all": np.where(np.isfinite(zmat_all), (zmat_all - mu) / sd, OFF)
            .astype(np.float32),
            "ksubj": ksubj, "proj": proj}


def fuse_z(parts: list, weights=WEIGHTS) -> np.ndarray:
    w = np.asarray(weights, float) / np.sum(weights)
    return sum(wi * p for wi, p in zip(w, parts)).astype(np.float32)


def gate_1(ctx: Ctx, stores: dict) -> dict:
    zmat = fuse_z([stores[n]["zmat_all"] for n in NAMES])
    pred = stores[NAMES[0]]["ksubj"][zmat.argmax(1)]
    acc = float(np.mean(pred == ctx.nodes))
    f1 = float(f1_score(ctx.nodes, pred, labels=CLASSES, average="macro",
                        zero_division=0))
    per_task = {t: float(np.mean(pred[ctx.tasks == t] == ctx.nodes[ctx.tasks == t]))
                for t in TASKS}
    return {"in_sample_accuracy": acc, "in_sample_macro_f1": f1,
            "in_sample_per_task_accuracy": per_task, "n": int(len(pred)),
            "note": "prototypes built from all 11304 Train rows, predicted on the same rows"}


def gate_5_6(ctx: Ctx, stores: dict) -> dict:
    fused = [(fuse_z([stores[n]["loso"][i][0] for n in NAMES]),
              fuse_z([stores[n]["loso"][i][1] for n in NAMES]))
             for i in range(len(CLASSES))]
    pooled_unknown = np.concatenate([u for _, u in fused])
    pooled_known = np.concatenate([k for k, _ in fused])
    auroc = float(roc_auc_score(np.r_[np.zeros(len(pooled_unknown)),
                                      np.ones(len(pooled_known))],
                                np.r_[pooled_unknown, pooled_known]))
    per_fold = []
    for held, (kn, un) in zip(CLASSES, fused):
        auc = float(roc_auc_score(np.r_[np.zeros(len(un)), np.ones(len(kn))],
                                  np.r_[un, kn]))
        rec = {"held_identity": int(held), "auroc": auc,
               "n_known": int(len(kn)), "n_unknown": int(len(un)),
               "known_median": float(np.median(kn)),
               "unknown_median": float(np.median(un))}
        for a in (0.90, 0.95, 0.98):
            thr = float(np.quantile(kn, 1 - a))
            rec[f"U@{int(round(a * 100))}"] = float(np.mean(un < thr))
        per_fold.append(rec)
    aucs = np.asarray([r["auroc"] for r in per_fold])
    out = {"gate5_auroc_pooled_unknown_positive": auroc,
           "gate5_auroc_per_fold_mean": float(aucs.mean()),
           "gate5_auroc_per_fold_std": float(aucs.std(ddof=1)),
           "gate5_auroc_per_fold_min": float(aucs.min()),
           "gate5_auroc_per_fold_max": float(aucs.max()),
           "n_pooled_known": int(len(pooled_known)),
           "n_pooled_unknown": int(len(pooled_unknown)),
           "per_fold": per_fold}
    for a in (90, 95, 98):
        vals = np.asarray([r[f"U@{a}"] for r in per_fold])
        out[f"gate6_U@{a}"] = {"mean": float(vals.mean()),
                               "std": float(vals.std(ddof=1)),
                               "min": float(vals.min()), "max": float(vals.max()),
                               "known_acceptance": a / 100.0}
    known_acc = np.asarray([np.mean(kn >= np.quantile(kn, 1 - ACCEPT))
                            for kn, _ in fused])
    out["reference_acceptance"] = ACCEPT
    out["U@88_reproduction"] = float(np.mean(
        [np.mean(un < np.quantile(kn, 1 - ACCEPT)) for kn, un in fused]))
    out["known_acceptance_check"] = float(known_acc.mean())
    return out


# ---------------------------------------------------------------- torch pipeline
def band_masks(bands) -> torch.Tensor:
    freq = torch.fft.rfftfreq(N, d=1.0 / FS, device=DEVICE)
    masks = []
    for i, (lo, hi) in enumerate(bands):
        upper = freq <= hi if i == len(bands) - 1 else freq < hi
        masks.append((freq >= lo) & upper)
    return torch.stack(masks).to(FT)


MASKS = band_masks(BANDS)
MASKS_LBP = band_masks(FE["logbp_bands_hz"])
TRI_ROWS, TRI_COLS = torch.triu_indices(CH, CH, device=DEVICE)
DIAG = (TRI_ROWS == TRI_COLS)
OFFMUL = torch.where(DIAG, torch.ones_like(TRI_ROWS, dtype=FT),
                     torch.full_like(TRI_ROWS, float(FE["off_diagonal_multiplier"]),
                                     dtype=FT))


def bandpass(x: torch.Tensor) -> torch.Tensor:
    spec = torch.fft.rfft(x, dim=-1)
    return torch.fft.irfft(spec[:, None] * MASKS[None, :, None, :], n=N, dim=-1)


def correlation_logspd(x: torch.Tensor) -> torch.Tensor:
    """Differentiable E65 gpu_correlation_logspd (B,18,1200) -> (B,855)."""
    f = bandpass(x)
    f = f - f.mean(-1, keepdim=True)
    scale = f.std(-1, keepdim=True, unbiased=True).clamp_min(
        float(FE["channel_std_epsilon"]))
    norm = (f - f.mean(-1, keepdim=True)) / scale
    corr = norm @ norm.transpose(-1, -2) / (N - int(FE["channel_std_ddof"]))
    trace = torch.diagonal(corr, dim1=-2, dim2=-1).sum(-1)
    eye = torch.eye(CH, dtype=x.dtype, device=x.device)
    rho = float(FE["covariance_shrinkage"])
    spd = (1.0 - rho) * corr + rho * trace[..., None, None] / CH * eye
    vals, vecs = torch.linalg.eigh(spd)
    vals = vals.clamp_min(float(FE["eigenvalue_floor"]))
    logspd = (vecs * vals.log().unsqueeze(-2)) @ vecs.transpose(-1, -2)
    vec = logspd[..., TRI_ROWS, TRI_COLS] * OFFMUL
    diag_mean = vec[..., DIAG].mean(-1, keepdim=True)
    vec = vec.clone()
    vec[..., DIAG] = vec[..., DIAG] - diag_mean
    mean = vec.mean(-1, keepdim=True)
    sd = vec.std(-1, keepdim=True, unbiased=False).clamp_min(
        float(FE["zband_epsilon"]))
    return ((vec - mean) / sd).flatten(-2)


def log_band_power(x: torch.Tensor) -> torch.Tensor:
    """Differentiable E146 log band power (B,18,1200) -> (B,90)."""
    xc = x - x.mean(-1, keepdim=True)
    spec = torch.fft.rfft(xc, dim=-1)
    parts = []
    for b in range(NB):
        sb = spec * MASKS_LBP[b][None, :].to(spec.dtype) * 2.0
        parts.append(torch.fft.ifft(sb, n=N, dim=-1))
    amp = torch.stack(parts, dim=1).abs()
    power = (amp ** 2).mean(-1)
    return torch.log(power + 1e-8).permute(0, 2, 1).reshape(len(x), CH * NB)


def normalize_eeg(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=-1, keepdim=True)
    rms = x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
    return x / rms


class Encoder(nn.Module):
    def __init__(self, widths, kernels, strides, embed_dim, dropout, in_channels=CH):
        super().__init__()
        blocks, channels = [], in_channels
        for width, kernel, stride in zip(widths, kernels, strides):
            blocks.extend([nn.Conv1d(channels, width, kernel, stride=stride,
                                     padding=kernel // 2, bias=False),
                           nn.BatchNorm1d(width), nn.ELU(inplace=True)])
            channels = width
        self.body = nn.Sequential(*blocks)
        self.pool_dim = 2 * channels
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.pool_dim, embed_dim)

    def forward(self, x):
        h = self.body(x)
        return self.head(self.drop(torch.cat([h.mean(-1), h.amax(-1)], 1)))


def load_encoder() -> nn.Module:
    payload = torch.load(ROOT / "research/E162_raw_eeg_contrastive_v2/model_lam0.0_t0.1.pt",
                         map_location="cpu", weights_only=False)
    spec = payload["encoder"]
    enc = Encoder(spec["widths"], spec["kernels"], spec["strides"],
                  spec["embed_dim"], spec["dropout"]).to(DEVICE)
    enc.load_state_dict(payload["state_dict"])
    enc.eval()
    enc.requires_grad_(False)
    return enc


def torch_features(x: torch.Tensor, encoder: nn.Module) -> dict:
    return {"E65": correlation_logspd(x), "logbp": log_band_power(x),
            "E162": encoder(normalize_eeg(x))}


class TorchScorer:
    """Frozen-score forward pass with gradients w.r.t. the raw EEG."""

    def __init__(self, ctx: Ctx, streams: dict) -> None:
        self.encoder = load_encoder()
        self.t = {}
        for name in NAMES:
            s = streams[name]
            self.t[name] = {
                "stats": {k: (torch.as_tensor(v[0], dtype=FT, device=DEVICE),
                              torch.as_tensor(v[1], dtype=FT, device=DEVICE))
                          for k, v in s["stats"].items()},
                "pca_mu": torch.as_tensor(s["pca_mu"], dtype=FT, device=DEVICE),
                "basis": torch.as_tensor(s["basis"], dtype=FT, device=DEVICE),
                "protos": {}, "mu": 0.0, "sd": 1.0,
            }

    def set_prototypes(self, ctx: Ctx, streams: dict, store: dict) -> None:
        for name in NAMES:
            proj = streams[name]["proj"]
            protos, keys = build_prototypes(proj, ctx.nodes, ctx.tasks,
                                           ctx.fit_rows & np.isin(ctx.nodes, CLASSES),
                                           CLASSES)
            key_task = np.asarray([k[0] for k in keys])
            key_subj = np.asarray([k[1] for k in keys])
            for task in TASKS:
                cols = np.flatnonzero(key_task == task)
                mat = np.zeros((len(CLASSES), proj.shape[1]), dtype=np.float32)
                mat[key_subj[cols] - 1] = protos[cols]
                self.t[name]["protos"][task] = torch.as_tensor(mat, dtype=FT,
                                                               device=DEVICE)
            self.t[name]["mu"] = float(store[name]["mu"])
            self.t[name]["sd"] = float(store[name]["sd"])

    def fine(self, x: torch.Tensor, task: str) -> torch.Tensor:
        feats = torch_features(x, self.encoder)
        total = None
        for name, w in zip(NAMES, WEIGHTS / WEIGHTS.sum()):
            cfg = self.t[name]
            mu, sd = cfg["stats"][task]
            f = (feats[name] - mu) / sd
            f = (f - cfg["pca_mu"]) @ cfg["basis"]
            score = -torch.cdist(f, cfg["protos"][task])
            z = (score - cfg["mu"]) / cfg["sd"]
            total = w * z if total is None else total + w * z
        return total


def load_raw(paths) -> np.ndarray:
    base = ROOT / CFG["data"]["train_root"]

    def one(p):
        data = np.asarray(loadmat(base / p, variable_names=["data"])["data"],
                          dtype=np.float32)
        assert data.shape == (N, CH), (p, data.shape)
        return np.ascontiguousarray(data.T)
    with ThreadPoolExecutor(max_workers=12) as pool:
        return np.stack(list(pool.map(one, paths)))


# ---------------------------------------------------------------- attack stage
def ci95(p: float, n: int) -> list:
    half = 1.96 * float(np.sqrt(max(p * (1 - p), 0.0) / max(n, 1)))
    return [max(0.0, p - half), min(1.0, p + half)]


def decide(scores: np.ndarray, thr: float) -> tuple:
    best = scores.max(1)
    idx = scores.argmax(1)
    accept = best >= thr
    pred = np.where(accept, idx + 1, 0)
    return pred, accept


def attack_metrics(scores: np.ndarray, truth: np.ndarray, thr: float) -> dict:
    pred, accept = decide(scores, thr)
    idx = scores.argmax(1)
    n = len(pred)
    ra = float(np.mean(pred == truth))
    asafe = float(np.mean((pred == truth) | (pred == 0)))
    return {"n": n, "RA": ra, "RA_ci95": ci95(ra, n), "ASafeR": asafe,
            "ASafeR_ci95": ci95(asafe, n),
            "closed_set_accuracy": float(np.mean(idx + 1 == truth)),
            "accept_rate": float(accept.mean()),
            "wrong_identity_rate": float(np.mean((pred != 0) & (pred != truth))),
            "reject_rate": float(np.mean(pred == 0)),
            "identity_accuracy_when_accepted": float(
                np.mean(pred[accept] == truth[accept])) if accept.any() else None}


def pgd(scorer: TorchScorer, x: torch.Tensor, task: str, truth_idx: np.ndarray,
        target_idx: np.ndarray, goal: str, eps: float, steps: int, seed: int) -> dict:
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    xc = x - x.mean(-1, keepdim=True)
    rms = xc.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-6)
    bound = (eps * rms).view(-1, 1, 1)
    x0 = x.clone()
    with torch.no_grad():
        x_adv = x0 + (torch.rand(x0.shape, generator=gen, device=DEVICE) * 2 - 1) * bound
        x_adv = torch.clamp(x_adv, x0 - bound, x0 + bound)
    alpha = eps / 2.0
    n_nan = 0
    for _ in range(steps):
        x_adv.requires_grad_(True)
        scores = scorer.fine(x_adv, task)
        if goal == "A":
            loss = -scores.gather(1, torch.as_tensor(target_idx, dtype=torch.long,
                                                    device=DEVICE)[:, None]).sum()
        elif goal == "C":
            tgt = torch.as_tensor(target_idx, dtype=torch.long, device=DEVICE)[:, None]
            tru = torch.as_tensor(truth_idx, dtype=torch.long, device=DEVICE)[:, None]
            loss = -(scores.gather(1, tgt) - scores.gather(1, tru)).sum()
        else:
            loss = -torch.logsumexp(scores, dim=1).sum()
        grad = torch.autograd.grad(loss, x_adv)[0]
        n_nan += int((~torch.isfinite(grad)).sum().item())
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        # minimise the loss -> move against the gradient sign
        step = -torch.sign(grad) * (alpha * rms).view(-1, 1, 1)
        x_adv = torch.clamp((x_adv + step).detach(), x0 - bound, x0 + bound)
    with torch.no_grad():
        out = scorer.fine(x_adv, task).cpu().numpy()
    rel = ((x_adv - x0).abs().amax(dim=(1, 2)) / rms).max().item()
    return {"scores": out, "nan_gradients": n_nan, "max_rel_linf": float(rel)}


def closed_set_reference(ctx: Ctx, streams: dict, stores: dict) -> dict:
    """How the window block behaves under three prototype protocols (2301 rows)."""
    w = ctx.window_rows

    def acc(mask_rows):
        parts = []
        for name in NAMES:
            p2, k2 = build_prototypes(streams[name]["proj"], ctx.nodes, ctx.tasks,
                                      mask_rows, CLASSES)
            d, ks = neg_dmat(streams[name]["proj"], p2, k2, ctx.tasks, w)
            parts.append(np.where(np.isfinite(d),
                                  (d - stores[name]["mu"]) / stores[name]["sd"], OFF))
        return float(np.mean(ks[fuse_z(parts).argmax(1)] == ctx.nodes[w]))

    oof_pred = np.full(len(ctx.nodes), -1, dtype=np.int32)
    for block in range(int(CFG["protocol"]["blocks"])):
        held = np.flatnonzero(ctx.blocks == block)
        hold = ctx.blocks == block
        parts = []
        for name in NAMES:
            p2, k2 = build_prototypes(streams[name]["proj"], ctx.nodes, ctx.tasks,
                                      ~hold, CLASSES)
            d, ks = neg_dmat(streams[name]["proj"], p2, k2, ctx.tasks, held)
            parts.append(np.where(np.isfinite(d),
                                  (d - stores[name]["mu"]) / stores[name]["sd"], OFF))
        oof_pred[held] = ks[fuse_z(parts).argmax(1)]
    return {"window_rows": int(len(w)),
            "oof_4of5_prototypes": float(np.mean(oof_pred[w] == ctx.nodes[w])),
            "fit_rows_prototypes": acc(ctx.fit_rows),
            "all_rows_prototypes_insample": acc(np.ones(len(ctx.nodes), bool)),
            "note": "gate #2 C=0.9592 uses the 5-block OOF protocol whose prototypes contain "
                    "the temporally adjacent 0.8-1.0 block; restricting prototypes to the "
                    "enrolment block frac<=0.6 (the deployment-honest protocol used for the "
                    "attack) is harder, which is why the attack clean baseline is lower"}


def stage_attack(ctx: Ctx, streams: dict, stores: dict, thr: float) -> dict:
    proxy = CFG["gates"]["gate7_8_proxy"]
    rng = np.random.default_rng(int(proxy["seed"]))
    rows = []
    for task in TASKS:
        cand = ctx.window_rows[ctx.tasks[ctx.window_rows] == task]
        take = min(int(proxy["cap_per_task"]), len(cand))
        rows.append(np.sort(rng.choice(cand, size=take, replace=False)))
    rows = np.sort(np.concatenate(rows))
    log(f"attack subset {len(rows)} rows | " +
        str({t: int(np.sum(ctx.tasks[rows] == t)) for t in TASKS}))

    raw = load_raw(ctx.paths[rows])
    log(f"raw loaded {raw.shape}")

    scorer = TorchScorer(ctx, streams)
    scorer.set_prototypes(ctx, streams, stores)

    # --- feature parity check against the cached extractors
    pick = np.sort(rng.choice(len(rows), 8, replace=False))
    xt = torch.as_tensor(raw[pick], dtype=FT, device=DEVICE)
    feats = torch_features(xt, scorer.encoder)
    parity = {}
    for name in NAMES:
        raw_cached = np.asarray(np.load(ROOT / CFG["streams"][name]["train"],
                                        mmap_mode="r"))[rows[pick]]
        got = feats[name].cpu().numpy()
        parity[name] = {"max_abs_diff": float(np.max(np.abs(got - raw_cached))),
                        "rel_fro": float(np.linalg.norm(got - raw_cached)
                                         / max(np.linalg.norm(raw_cached), 1e-8))}
    log(f"feature parity (torch vs cached): {json.dumps(parity)}")

    # --- clean scores: numpy path (cached features) vs torch path
    clean_np = np.full((len(rows), len(CLASSES)), -np.inf, dtype=np.float32)
    clean_torch = np.full((len(rows), len(CLASSES)), -np.inf, dtype=np.float32)
    for task in TASKS:
        sel = np.flatnonzero(ctx.tasks[rows] == task)
        parts = []
        for name in NAMES:
            s = streams[name]
            f = s["proj"][rows[sel]]
            protos, keys = build_prototypes(s["proj"], ctx.nodes, ctx.tasks,
                                           ctx.fit_rows, CLASSES)
            key_task = np.asarray([k[0] for k in keys])
            key_subj = np.asarray([k[1] for k in keys])
            cols = np.flatnonzero(key_task == task)
            p = torch.as_tensor(protos[cols], dtype=FT, device=DEVICE)
            d = -torch.cdist(torch.as_tensor(f, dtype=FT, device=DEVICE), p)
            z = np.full((len(sel), len(CLASSES)), OFF, dtype=np.float32)
            z[:, key_subj[cols] - 1] = ((d.cpu().numpy() - stores[name]["mu"])
                                        / stores[name]["sd"])
            parts.append(z)
        clean_np[sel] = fuse_z(parts)
        xt = torch.as_tensor(raw[sel], dtype=FT, device=DEVICE)
        with torch.no_grad():
            clean_torch[sel] = scorer.fine(xt, task).cpu().numpy()
    ok = np.isfinite(clean_np).all(1) & np.isfinite(clean_torch).all(1)
    agree = float(np.mean(clean_np[ok].argmax(1) == clean_torch[ok].argmax(1)))
    diff = float(np.max(np.abs(clean_np[ok] - clean_torch[ok])))
    log(f"clean score parity numpy vs torch: max|dz|={diff:.4f} argmax_agreement={agree:.4f}")

    truth = ctx.nodes[rows]
    result = {"subset": {"n": int(len(rows)),
                         "per_task": {t: int(np.sum(ctx.tasks[rows] == t)) for t in TASKS},
                         "rows": "Train 0.6 < frac <= 0.8, stratified cap "
                                 f"{int(proxy['cap_per_task'])} per task",
                         "threshold": thr, "acceptance_rate": ACCEPT},
              "feature_parity": parity,
              "clean_score_parity": {"max_abs_z_diff": diff,
                                     "argmax_agreement": agree},
              "closed_set_reference": closed_set_reference(ctx, streams, stores),
              "clean": attack_metrics(clean_np, truth, thr), "attacks": []}

    target = np.where(np.isfinite(clean_np), clean_np, -np.inf).argmax(1) + 1
    for i in range(len(target)):
        if target[i] == truth[i]:
            row = clean_np[i].copy()
            row[truth[i] - 1] = -np.inf
            target[i] = int(np.argmax(row)) + 1
    log(f"goal A target identity: {np.bincount(target, minlength=21)[1:].tolist()}")

    for eps in CFG["gates"]["gate7_8_proxy"]["eps_normalised"]:
        for goal in ("A", "B", "C"):
            scores = np.full((len(rows), len(CLASSES)), -np.inf, dtype=np.float32)
            nans, rel = 0, 0.0
            for task in TASKS:
                sel = np.flatnonzero(ctx.tasks[rows] == task)
                for s in range(0, len(sel), 128):
                    chunk = sel[s:s + 128]
                    xt = torch.as_tensor(raw[chunk], dtype=FT, device=DEVICE)
                    out = pgd(scorer, xt, task, truth[chunk] - 1, target[chunk] - 1,
                              goal, float(eps), int(proxy["steps"]), 1234 + s)
                    scores[chunk] = out["scores"]
                    nans += out["nan_gradients"]
                    rel = max(rel, out["max_rel_linf"])
            m = attack_metrics(scores, truth, thr)
            m.update({"eps_normalised": float(eps), "goal": goal,
                      "snr_db_lower_bound": float(-20 * np.log10(eps)),
                      "measured_rel_linf_max": rel,
                      "nan_gradient_entries": nans})
            result["attacks"].append(m)
            log(f"eps={eps} goal={goal}: RA={m['RA']:.4f} ASafeR={m['ASafeR']:.4f} "
                f"accept={m['accept_rate']:.4f} wrong={m['wrong_identity_rate']:.4f} "
                f"nan_grad={nans}")
    return result


# ---------------------------------------------------------------- main
def main() -> None:
    stage = "all"
    for a in sys.argv[1:]:
        if a.startswith("--stage"):
            stage = a.split("=")[1] if "=" in a else "gates"
    log(f"E178 start | stage={stage} | device={DEVICE} | "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    ctx = Ctx()
    log(f"train {len(ctx.nodes)} | fit {int(ctx.fit_rows.sum())} | "
        f"window {int(ctx.window.sum())} | tasks {dict(pd.Series(ctx.tasks).value_counts())}")
    streams = load_stream_features(ctx)
    stores = {}
    for name in NAMES:
        stores[name] = stream_scores(ctx, name, streams[name]["proj"])
        log(f"  {name}: oof mu={stores[name]['mu']:.4f} sd={stores[name]['sd']:.4f}")

    z_oof = fuse_z([stores[n]["z_oof"] for n in NAMES])
    finite = np.isfinite(z_oof)
    # 5-block OOF identity predictions from the fused score
    pred = np.full(len(ctx.nodes), -1, dtype=np.int32)
    for block in range(int(CFG["protocol"]["blocks"])):
        held = np.flatnonzero(ctx.blocks == block)
        hold = ctx.blocks == block
        parts, block_subj = [], None
        for name in NAMES:
            proj = streams[name]["proj"]
            p2, k2 = build_prototypes(proj, ctx.nodes, ctx.tasks, ~hold, CLASSES)
            dmat, block_subj = neg_dmat(proj, p2, k2, ctx.tasks, held)
            parts.append(np.where(np.isfinite(dmat),
                                  (dmat - stores[name]["mu"]) / stores[name]["sd"], OFF))
        pred[held] = block_subj[fuse_z(parts).argmax(1)]
    loso_fused = fuse_z_loso(stores)
    unknown_all = np.concatenate([u for _, u in loso_fused])
    oof_auc = float(roc_auc_score(np.r_[np.zeros(len(unknown_all)), np.ones(int(finite.sum()))],
                                  np.r_[unknown_all, z_oof[finite]]))
    oof_acc = float(np.mean(pred == ctx.nodes))
    oof_f1 = float(f1_score(ctx.nodes, pred, labels=CLASSES, average="macro",
                            zero_division=0))
    log(f"reproduction: OOF AUC={oof_auc:.4f} (pub 0.9229) | OOF acc={oof_acc:.4f} "
        f"(pub 0.9592) | OOF macro-F1={oof_f1:.4f} (pub 0.9590)")

    results = {"reproduction": {"train_oof_auc": oof_auc,
                                "train_oof_identity_accuracy": oof_acc,
                                "train_oof_macro_f1": oof_f1,
                                "published": CFG["audited_scheme"]["published"]}}
    if stage in ("gates", "all"):
        results["gate1"] = gate_1(ctx, stores)
        log(f"gate1 in-sample accuracy={results['gate1']['in_sample_accuracy']:.4f} "
            f"macro-F1={results['gate1']['in_sample_macro_f1']:.4f}")
        g56 = gate_5_6(ctx, stores)
        results["gate5_6"] = g56
        log(f"gate5 AUROC={g56['gate5_auroc_pooled_unknown_positive']:.4f} "
            f"(folds {g56['gate5_auroc_per_fold_mean']:.4f}+-"
            f"{g56['gate5_auroc_per_fold_std']:.4f}) | "
            f"U@95={g56['gate6_U@95']['mean']:.4f} "
            f"U@90={g56['gate6_U@90']['mean']:.4f} "
            f"U@98={g56['gate6_U@98']['mean']:.4f} | "
            f"U@88={g56['U@88_reproduction']:.4f}")

    if stage in ("attack", "all"):
        thr = float(np.quantile(z_oof[finite], 1 - ACCEPT))
        log(f"acceptance threshold @{ACCEPT}: {thr:.4f}")
        results["gate7_8"] = stage_attack(ctx, streams, stores, thr)

    path = OUT / "results.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing.update(results)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    log("results.json written")


def fuse_z_loso(stores: dict) -> list:
    return [(fuse_z([stores[n]["loso"][i][0] for n in NAMES]),
             fuse_z([stores[n]["loso"][i][1] for n in NAMES]))
            for i in range(len(CLASSES))]


if __name__ == "__main__":
    main()
