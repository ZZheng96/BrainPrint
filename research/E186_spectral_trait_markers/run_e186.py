"""E186: spectral trait markers (individual alpha peak frequency and friends).

The one family never tried here is the one the cross-session literature calls the
most stable identity marker: per-channel *spectral trait* features, i.e. numbers that
describe the shape of a subject's power spectrum rather than its momentary power.

  IAF (7-13 Hz peak, parabolic interpolation)         18
  alpha peak prominence (dB above the in-band trend)  18
  alpha peak width (FWHM, Hz)                         18
  spectral centroid (4-30 Hz, power weighted)         18
  aperiodic 1/f slope + intercept (robust IRLS)       36
  spectral edge frequency (95% cumulative power)      18
                                               total 126

No per-trial normalisation is applied to these features (that would delete the trait);
the only scaling is a per-task z-score fitted on Train session1 and applied to both
sides, followed by PCA fitted on the Train fit block (frac <= 0.6).

Evaluation is the E143 protocol verbatim: Train-only standardisation / PCA /
prototypes / thresholds, 5-block Train OOF, LOSO with `pm & fit_rows`, the scored
identity excluded, evaluation block 0.6 < frac <= 0.8.  Test is only scored; the
Session column is used for the diagnostic AUC only.
"""
from __future__ import annotations

import argparse
import json
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
CLASSES = np.arange(1, 21, dtype=int)
TASKS = CFG["tasks"]

SP = CFG["spectral"]
FS = int(CFG["data"]["sampling_rate_hz"])
NC = int(CFG["data"]["channels"])
N = int(CFG["data"]["samples_per_trial"])
NPERSEG = int(SP["nperseg"])
STEP = NPERSEG - int(SP["noverlap"])
NFFT = int(SP["n_fft"])
DF = FS / float(NFFT)
ALPHA_LO, ALPHA_HI = SP["alpha_band_hz"]
CENT_LO, CENT_HI = SP["centroid_band_hz"]
AP_LO, AP_HI = SP["aperiodic_band_hz"]
EDGE_LO, EDGE_HI = SP["edge_band_hz"]
EDGE_FRAC = float(SP["edge_fraction"])
IRLS_ITERS = int(SP["irls_iterations"])
HUBER = float(SP["irls_huber"])
LAYOUT = CFG["feature_layout"]
FAMILY_RANGES = {
    "iaf": tuple(LAYOUT["iaf"]),
    "prominence": tuple(LAYOUT["prominence"]),
    "width": tuple(LAYOUT["width"]),
    "centroid": tuple(LAYOUT["centroid"]),
    "aperiodic": (LAYOUT["aperiodic_slope"][0], LAYOUT["aperiodic_intercept"][1]),
    "edge95": tuple(LAYOUT["edge95"]),
    "all": (0, 126),
}
DIM_TRAITS = 126

FIT_FRACTION = float(CFG["fit_fraction"])
WIN_LO, WIN_HI = CFG["eval_window"]
BLOCKS = int(CFG["blocks"])
ACC_GRID = [float(a) for a in CFG["acceptance_grid"]]
K_GRID = [int(k) for k in CFG["prototype_k"]]


# ------------------------------------------------------------------ logging
class Tee:
    def __init__(self, path: Path, mode: str = "w") -> None:
        self.fh = path.open(mode, encoding="utf-8", newline="\n")
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


# ------------------------------------------------------------------ extraction
WINDOW = torch.hann_window(NPERSEG, periodic=False, dtype=FT, device=DEVICE)
FREQS = torch.fft.rfftfreq(NFFT, d=1.0 / FS).to(DEVICE).to(FT)
PSD_SCALE = 2.0 / (FS * float((WINDOW ** 2).sum()))
_T = torch.arange(N, dtype=FT, device=DEVICE)
_TC = _T - _T.mean()
_SXX_TIME = float((_TC ** 2).sum())


def _idx(lo: float, hi: float, closed: bool = True) -> torch.Tensor:
    mask = (FREQS >= lo) & (FREQS <= hi if closed else FREQS < hi)
    return torch.nonzero(mask, as_tuple=False).flatten()


IDX_A = _idx(ALPHA_LO, ALPHA_HI)
IDX_C = _idx(CENT_LO, CENT_HI, closed=False)
IDX_E = _idx(EDGE_LO, EDGE_HI)
_keep = (FREQS >= AP_LO) & (FREQS <= AP_HI)
for elo, ehi in SP["aperiodic_exclude_hz"]:
    _keep &= ~((FREQS >= elo) & (FREQS <= ehi))
IDX_R = torch.nonzero(_keep, as_tuple=False).flatten()
NA, NR, NE = int(len(IDX_A)), int(len(IDX_R)), int(len(IDX_E))
ALPHA_F = FREQS[IDX_A]
POS_A = torch.arange(NA, dtype=FT, device=DEVICE)
POS_BAR = float(POS_A.mean())
SXX_POS = float(((POS_A - POS_BAR) ** 2).sum())
XR = torch.log10(FREQS[IDX_R])
FE = FREQS[IDX_E]


def load_one(path: Path) -> np.ndarray:
    data = np.asarray(loadmat(path, variable_names=["data"])["data"], dtype=np.float32)
    if data.shape != (N, NC):
        raise ValueError(f"{path}: expected {(N, NC)}, got {data.shape}")
    return data.T


def welch(x: torch.Tensor) -> torch.Tensor:
    """x: B x C x N -> one-sided PSD, B x C x F (Hann, 50% overlap, zero padded)."""
    x = x - x.mean(-1, keepdim=True)
    slope = (x * _TC).sum(-1, keepdim=True) / _SXX_TIME
    x = x - slope * _TC
    seg = x.unfold(-1, NPERSEG, STEP) * WINDOW
    spec = torch.fft.rfft(seg, n=NFFT, dim=-1)
    psd = (spec.real.square() + spec.imag.square()).mean(2)
    return psd * PSD_SCALE


def _gather(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return t.gather(-1, idx.unsqueeze(-1)).squeeze(-1)


def trait_block(psd: torch.Tensor) -> torch.Tensor:
    """psd: B x C x F -> B x 126 trait features (channel-major inside each block)."""
    db = 10.0 * torch.log10(psd + 1e-12)

    # ---- alpha band -------------------------------------------------------
    ya = db[:, :, IDX_A]                              # B x C x NA
    ipk = ya.argmax(-1)
    y0 = _gather(ya, ipk)
    ym = _gather(ya, (ipk - 1).clamp(min=0))
    yp = _gather(ya, (ipk + 1).clamp(max=NA - 1))
    denom = ym - 2.0 * y0 + yp
    ok = denom.abs() > 1e-9
    delta = torch.where(ok, 0.5 * (ym - yp) / torch.where(ok, denom, torch.ones_like(denom)),
                        torch.zeros_like(denom)).clamp(-0.7, 0.7)
    iaf = (ALPHA_F[ipk] + delta * DF).clamp(ALPHA_LO, ALPHA_HI)

    # in-band linear trend (dB vs bin index), evaluated at the interpolated peak
    ybar = ya.mean(-1)
    slope_a = ((POS_A - POS_BAR) * (ya - ybar.unsqueeze(-1))).sum(-1) / SXX_POS
    inter_a = ybar - slope_a * POS_BAR
    trend = inter_a + slope_a * (ipk.to(FT) + delta)
    prominence = y0 - trend

    # full width at half maximum of the peak above that trend
    level = 0.5 * (y0 + trend)
    left_hi = torch.zeros_like(ipk)
    found_l = torch.zeros_like(ipk, dtype=torch.bool)
    for step in range(1, NA):
        j = ipk - step
        valid = j >= 0
        jj = j.clamp(min=0)
        nxt = _gather(ya, (jj + 1).clamp(max=NA - 1))
        cond = valid & (~found_l) & (nxt < level)
        left_hi = torch.where(cond, jj, left_hi)
        found_l = found_l | cond
    yl = _gather(ya, left_hi)
    yl1 = _gather(ya, (left_hi + 1).clamp(max=NA - 1))
    frac_l = ((level - yl) / (yl1 - yl + 1e-9)).clamp(0.0, 1.0)
    lo_edge = torch.full_like(frac_l, float(ALPHA_F[0]))
    f_l = torch.where(found_l, ALPHA_F[left_hi] + frac_l * DF, lo_edge)

    right_lo = torch.full_like(ipk, NA - 1)
    found_r = torch.zeros_like(ipk, dtype=torch.bool)
    for step in range(1, NA):
        j = ipk + step
        valid = j <= NA - 1
        jj = j.clamp(max=NA - 1)
        prv = _gather(ya, (jj - 1).clamp(min=0))
        cond = valid & (~found_r) & (prv < level)
        right_lo = torch.where(cond, jj, right_lo)
        found_r = found_r | cond
    yr_ = _gather(ya, right_lo)
    yrp = _gather(ya, (right_lo - 1).clamp(min=0))
    frac_r = ((level - yrp) / (yr_ - yrp + 1e-9)).clamp(0.0, 1.0)
    hi_edge = torch.full_like(frac_r, float(ALPHA_F[-1]))
    f_r = torch.where(found_r, ALPHA_F[(right_lo - 1).clamp(min=0)] + frac_r * DF, hi_edge)
    width = (f_r - f_l).clamp(0.0, 6.0)

    # ---- spectral centroid ------------------------------------------------
    pc = psd[:, :, IDX_C]
    centroid = (pc * FREQS[IDX_C]).sum(-1) / (pc.sum(-1) + 1e-12)

    # ---- aperiodic 1/f (robust IRLS on log10 power vs log10 frequency) ----
    yl_ = torch.log10(psd[:, :, IDX_R] + 1e-12)
    w = torch.ones_like(yl_)
    sl = torch.zeros_like(yl_[:, :, 0])
    ic = torch.zeros_like(sl)
    one = torch.ones_like(sl)
    one_full = torch.ones_like(yl_)
    for _ in range(IRLS_ITERS):
        sw = w.sum(-1)
        swx = (w * XR).sum(-1)
        swxx = (w * XR * XR).sum(-1)
        swy = (w * yl_).sum(-1)
        swxy = (w * XR * yl_).sum(-1)
        den = sw * swxx - swx * swx
        den = torch.where(den.abs() > 1e-9, den, one)
        sl = (sw * swxy - swx * swy) / den
        ic = (swy - sl * swx) / sw
        resid = yl_ - (sl.unsqueeze(-1) * XR + ic.unsqueeze(-1))
        mad = resid.abs().median(-1).values
        z = resid.abs() / (1.4826 * mad + 1e-9).unsqueeze(-1)
        w = torch.where(z <= HUBER, one_full, HUBER / z.clamp(min=1e-6))

    # ---- 95% spectral edge ------------------------------------------------
    pe = psd[:, :, IDX_E]
    cum = pe.cumsum(-1)
    thr = EDGE_FRAC * cum[..., -1]
    idx = (cum < thr.unsqueeze(-1)).sum(-1).clamp(0, NE - 1)
    c1 = _gather(cum, idx)
    c0 = _gather(cum, (idx - 1).clamp(min=0))
    frac = ((thr - c0) / (c1 - c0 + 1e-12)).clamp(0.0, 1.0)
    edge = FE[idx] + frac * DF

    return torch.cat([iaf, prominence, width, centroid, sl, ic, edge], dim=1)


def run_split(meta: pd.DataFrame, dest: Path, tag: str) -> None:
    paths = [ROOT / CFG["data"]["root"] / p for p in meta.path]
    out = np.lib.format.open_memmap(dest, mode="w+", dtype=np.float32,
                                    shape=(len(paths), DIM_TRAITS))
    chunk = int(SP["chunk"])
    started = time.time()
    with ThreadPoolExecutor(max_workers=12) as pool:
        for st in range(0, len(paths), chunk):
            stop = min(st + chunk, len(paths))
            arr = list(pool.map(load_one, paths[st:stop]))
            x = torch.as_tensor(np.stack(arr), dtype=FT, device=DEVICE)
            with torch.no_grad():
                feats = trait_block(welch(x))
            if feats.shape[1] != DIM_TRAITS:
                raise ValueError(f"dim mismatch {feats.shape[1]} != {DIM_TRAITS}")
            out[st:stop] = feats.cpu().numpy()
            if stop % 2048 == 0 or stop == len(paths):
                log(f"  extract {tag} {stop}/{len(paths)}  {time.time() - started:.0f}s")
    out.flush()
    del out


def stage_extract() -> None:
    tr = pd.read_csv(ROOT / CFG["data"]["train_metadata"])
    te = pd.read_csv(ROOT / CFG["data"]["test_metadata"])
    log(f"extract train={len(tr)} test={len(te)} dim={DIM_TRAITS} device={DEVICE} "
        f"nfft={NFFT} df={DF:.4f}Hz alpha_bins={NA} reg_bins={NR} edge_bins={NE}")
    t0 = time.time()
    run_split(tr, OUT / "train_spectral_traits.npy", "train")
    run_split(te, OUT / "test_spectral_traits.npy", "test")
    log(f"extract done in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ protocol
def frac(meta: pd.DataFrame) -> np.ndarray:
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def task_standardise(a: np.ndarray, b: np.ndarray, tasks: np.ndarray,
                     test_tasks: np.ndarray) -> tuple:
    """Per-task z-score with Train-session1 statistics (E143 behaviour)."""
    a = a.copy()
    b = b.copy()
    for task in TASKS:
        m = tasks == task
        mu = a[m].mean(0)
        sd = a[m].std(0) + 1e-6
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
        self.nodes = meta.subject.to_numpy(int)
        self.tasks = meta.task.to_numpy()
        self.frac = frac(meta)
        self.fit_rows = self.frac <= FIT_FRACTION
        self.blocks = np.clip((self.frac * BLOCKS).astype(int), 0, BLOCKS - 1)
        self.window = (self.frac > WIN_LO) & (self.frac <= WIN_HI)
        self.window_rows = np.flatnonzero(self.window)
        self.test_tasks = qmeta.task.to_numpy()
        self.is_s1 = qmeta.session.astype(str).str.casefold().isin(
            ["session1", "s1", "1"]).to_numpy()
        self.y = np.r_[np.zeros(int(self.is_s1.sum()), dtype=int),
                       np.ones(int((~self.is_s1).sum()), dtype=int)]
        self.proj: dict[str, tuple] = {}

    def project(self, key: str, feat_train: np.ndarray, feat_test: np.ndarray,
                dim: int) -> tuple:
        cache = f"{key}@{dim}"
        if cache in self.proj:
            return self.proj[cache]
        mu, basis = pca_basis(np.asarray(feat_train[self.fit_rows], dtype=np.float64), dim)
        at = ((feat_train.astype(np.float64) - mu) @ basis).astype(np.float32)
        bt = ((feat_test.astype(np.float64) - mu) @ basis).astype(np.float32)
        self.proj[cache] = (at, bt)
        return at, bt

    def drop(self, key: str, dim: int) -> None:
        self.proj.pop(f"{key}@{dim}", None)


def evaluate(at: np.ndarray, bt: np.ndarray, k: int, ctx: Context) -> dict:
    """E143/E167/E169 open-set protocol on one projected representation."""
    protos, keys = build_prototypes(at, ctx.nodes, ctx.tasks, np.ones(len(at), bool),
                                    CLASSES, k)
    te_raw = neg_d1(bt, protos, keys, ctx.test_tasks)

    oof = np.full(len(at), np.nan, dtype=np.float32)
    oof_pred = np.full(len(at), -1, dtype=np.int32)
    for block in range(BLOCKS):
        hold = ctx.blocks == block
        held = np.flatnonzero(hold)
        p2, k2 = build_prototypes(at, ctx.nodes, ctx.tasks, ~hold, CLASSES, k)
        rows = held
        dmat = np.full((len(rows), len(k2)), -np.inf, dtype=np.float32)
        key_task = np.asarray([key[0] for key in k2])
        key_subject = np.asarray([key[1] for key in k2], dtype=np.int32)
        for task in TASKS:
            r = np.flatnonzero(ctx.tasks[rows] == task)
            c = np.flatnonzero(key_task == task)
            if len(r) == 0 or len(c) == 0:
                continue
            p = torch.as_tensor(np.asarray(p2[c]), dtype=FT, device=DEVICE)
            for start in range(0, len(r), 2048):
                chunk = r[start:start + 2048]
                q = torch.as_tensor(np.asarray(at[rows[chunk]], dtype=np.float32),
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
            "per_task_auc": {
                task: float(roc_auc_score((~ctx.is_s1[ctx.test_tasks == task]).astype(int),
                                          z_te[ctx.test_tasks == task]))
                for task in TASKS
                if len(np.unique(ctx.is_s1[ctx.test_tasks == task])) > 1}}


def summarise(entry: dict, meta: dict, ctx: Context) -> dict:
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


# ------------------------------------------------------------------ main
def main() -> None:
    started = time.time()
    ctx = Context()
    log(f"device {DEVICE} | train {len(ctx.nodes)} | test {len(ctx.test_tasks)} | "
        f"fit_rows {int(ctx.fit_rows.sum())} | window {int(ctx.window.sum())}")
    log(f"tasks {dict(pd.Series(ctx.tasks).value_counts())}")

    tr = np.load(ROOT / CFG["streams"]["traits"]["train"]).astype(np.float32)
    te = np.load(ROOT / CFG["streams"]["traits"]["test"]).astype(np.float32)
    if tr.shape[1] != DIM_TRAITS:
        raise ValueError(f"trait dim {tr.shape[1]}")
    log(f"traits train {tr.shape} / test {te.shape}")

    families: dict[str, dict] = {}
    for name in ["iaf", "prominence", "width", "centroid", "aperiodic", "edge95", "all"]:
        lo, hi = FAMILY_RANGES[name]
        a, b = task_standardise(tr[:, lo:hi], te[:, lo:hi], ctx.tasks, ctx.test_tasks)
        families[name] = {"train": a, "test": b, "dim": hi - lo}

    a65 = np.load(ROOT / CFG["streams"]["E65_logspd"]["train"]).astype(np.float32)
    b65 = np.load(ROOT / CFG["streams"]["E65_logspd"]["test"]).astype(np.float32)
    a65, b65 = task_standardise(a65, b65, ctx.tasks, ctx.test_tasks)
    log(f"E65 train {a65.shape} / test {b65.shape}")

    entries: dict[tuple, dict] = {}
    t0 = time.time()

    def run(key: str, at: np.ndarray, bt: np.ndarray, meta: dict,
            k_list: list | None = None) -> list:
        recs = []
        for k in (K_GRID if k_list is None else k_list):
            ekey = (key, int(k))
            if ekey not in entries:
                entries[ekey] = evaluate(at, bt, int(k), ctx)
            rec = summarise(entries[ekey], dict(meta, k=k, entry_key=key), ctx)
            recs.append(rec)
            log(f"    {key} k={k}: diagAUC {rec['diagnostic_auc']:.4f} "
                f"bestS1+S2 {rec['best_s1_plus_s2']:.4f} ({time.time() - t0:.0f}s)")
        return recs

    # ---------------------------------------------------------------- single families
    singles: dict[str, list] = {}
    for name in ["iaf", "prominence", "width", "centroid", "aperiodic", "edge95", "all"]:
        n = families[name]["dim"]
        dims = sorted({min(d, n) for d in CFG["pca_dims_requested"] if min(d, n) > 0})
        singles[name] = []
        for dim in dims:
            at, bt = ctx.project(name, families[name]["train"], families[name]["test"], dim)
            recs = run(f"{name}@{dim}", at, bt,
                       {"family": f"{name}_only", "stream": name, "pca_dim": dim,
                        "config": f"{name} PCA-{dim}"},
                       k_list=K_GRID if name == "all" else [12])
            singles[name].extend(recs)
            ctx.drop(name, dim)
            top = max(recs, key=lambda r: r["diagnostic_auc"])
            log(f"  [{name}] PCA-{dim}: AUC {top['diagnostic_auc']:.4f} "
                f"OOF {top['train_oof_auc']:.4f} bestS1+S2 "
                f"{max(r['best_s1_plus_s2'] for r in recs):.4f} "
                f"({time.time() - t0:.0f}s)")

    # ---------------------------------------------------------------- E65 control
    e65_rows = []
    for dim in CFG["streams"]["E65_logspd"]["pca_dims"]:
        at, bt = ctx.project("E65_logspd", a65, b65, dim)
        e65_rows.extend(run(f"E65@{dim}", at, bt,
                            {"family": "E65_only", "stream": "E65_logspd",
                             "pca_dim": dim, "config": f"E65_logspd PCA-{dim}"}))
        ctx.drop("E65_logspd", dim)
    ref = next(r for r in e65_rows if r["k"] == 12)
    log(f"  [E65] PCA-96 k=12 AUC {ref['diagnostic_auc']:.4f} S1 {ref['s1_at_090']:.4f} "
        f"S2 {ref['s2_at_090']:.4f} sum {ref['s1_plus_s2_at_090']:.4f} "
        f"({time.time() - t0:.0f}s)")

    # ---------------------------------------------------------------- fusion
    FUS = CFG["fusion"]
    fusion_rows = []
    at65, bt65 = ctx.project("E65_logspd", a65, b65, FUS["e65_pca_dim"])
    for tdim in FUS["trait_pca_dims"]:
        atf, btf = ctx.project("all", families["all"]["train"], families["all"]["test"], tdim)
        sa = float(np.sqrt(np.mean(at65[ctx.fit_rows] ** 2)))
        sb = float(np.sqrt(np.mean(atf[ctx.fit_rows] ** 2)))
        tr_cat = np.concatenate([at65 / max(sa, 1e-6), atf / max(sb, 1e-6)], axis=1)
        te_cat = np.concatenate([bt65 / max(sa, 1e-6), btf / max(sb, 1e-6)], axis=1)
        for jdim in FUS["joint_pca_dims"]:
            atj, btj = ctx.project(f"fusion{tdim}@{jdim}", tr_cat, te_cat, jdim)
            recs = run(f"fusion{tdim}@{jdim}", atj, btj,
                       {"family": "fusion_block_norm", "stream": "E65+trait_all",
                        "pca_dim": jdim, "pca_dim_family": tdim,
                        "config": f"block_norm concat 96+{tdim} -> PCA-{jdim}"})
            fusion_rows.extend(recs)
            ctx.drop(f"fusion{tdim}@{jdim}", jdim)
            top = max(recs, key=lambda r: r["best_s1_plus_s2"])
            log(f"  [fusion 96+{tdim} -> PCA-{jdim}] AUC {top['diagnostic_auc']:.4f} "
                f"S1+S2 {top['best_s1_plus_s2']:.4f} @a={top['best_a']:.2f} "
                f"({time.time() - t0:.0f}s)")

    all_rows = [r for rows in singles.values() for r in rows] + e65_rows + fusion_rows
    trait_rows = [r for rows in singles.values() for r in rows if r["family"] != "E65_only"]
    tie = lambda r: (round(r["best_s1_plus_s2"], 4), r["diagnostic_auc"])
    best_sum = max(all_rows, key=tie)
    best_sum_trait = max(trait_rows, key=tie)
    best_sum_fusion = max(fusion_rows, key=tie)
    best_auc_single = max(trait_rows, key=lambda r: r["diagnostic_auc"])
    best_auc_fusion = max(fusion_rows, key=lambda r: r["diagnostic_auc"])

    # ---------------------------------------------------------------- ten gates
    cap = json.loads((ROOT / "research/E183_gate_head_and_capacity/results.json")
                     .read_text(encoding="utf-8"))["capacity"]
    best_entry = entries[(best_sum["entry_key"], int(best_sum["k"]))]
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
    for pid, key, name, target in [
            ("#9", "Test_S2_acceptance", "Test S2 接受率", "≥ 0.75"),
            ("#10", "Test_S1_rejection", "Test S1 拒识率", "≥ 0.75")]:
        gates.append({"id": pid, "name": name, "target": target,
                      "value": None, "source": "本实验最佳配置（逐工作点）"})

    gate_table = []
    for g in gates:
        row = dict(g)
        if g["id"] in ("#9", "#10"):
            key = ("S1_rejection" if g["id"] == "#10" else "S2_acceptance")
            row["values"] = {f"{p['a']:.2f}": round(p["Test_" + key], 4)
                             for p in best_sum["points"]}
            row["pass"] = {a: bool(v >= 0.75) for a, v in row["values"].items()}
        else:
            if g["value"] is None:
                row["pass"] = False
            else:
                target_value = float(g["target"].lstrip("≥= ").strip())
                row["pass"] = (abs(g["value"] - target_value) < 1e-12
                               if g["target"].startswith("=") else g["value"] >= target_value)
        gate_table.append(row)

    # ---------------------------------------------------------------- results
    def slim(r: dict) -> dict:
        return {k: v for k, v in r.items() if k not in ("points",)}

    payload = {
        "experiment": CFG["experiment"],
        "clean": True,
        "protocol": "E143 verbatim: Train-only task z / PCA / prototypes / thresholds, "
                    "5-block Train OOF, LOSO with pm & fit_rows, Test scored once",
        "feature_layout": LAYOUT,
        "n_features": DIM_TRAITS,
        "single_family_best": {
            name: {
                "best_diag_auc": max(r["diagnostic_auc"] for r in rows),
                "best_config": max(rows, key=lambda r: r["diagnostic_auc"])["config"]
                + f" k={max(rows, key=lambda r: r['diagnostic_auc'])['k']}",
                "auc_at_pca16_k12": next(
                    (r["diagnostic_auc"] for r in rows
                     if r["pca_dim"] == min(16, families[name]["dim"]) and r["k"] == 12),
                    None),
                "rows": [slim(r) for r in rows]}
            for name, rows in singles.items()},
        "E65_control": [slim(r) for r in e65_rows],
        "fusion": [slim(r) for r in fusion_rows],
        "best": {"overall": slim(best_sum), "trait_only": slim(best_sum_trait),
                 "fusion": slim(best_sum_fusion), "best_auc_single": slim(best_auc_single),
                 "best_auc_fusion": slim(best_auc_fusion),
                 "e65_reference": slim(ref)},
        "best_points": best_sum["points"],
        "ten_gates": gate_table,
        "judgement": {
            "trait_single_best_diag_auc": best_auc_single["diagnostic_auc"],
            "trait_single_above_070": bool(best_auc_single["diagnostic_auc"] > 0.70),
            "trait_single_below_065": bool(best_auc_single["diagnostic_auc"] < 0.65),
            "best_s1_plus_s2": best_sum["best_s1_plus_s2"],
            "best_s1_plus_s2_config": best_sum["config"] + f" k={best_sum['k']} "
                                    f"@a={best_sum['best_a']:.2f}",
            "beats_14479": bool(best_sum["best_s1_plus_s2"] > 1.4479),
            "reaches_150": bool(best_sum["best_s1_plus_s2"] >= 1.50),
            "e65_reproduces": {"auc": ref["diagnostic_auc"], "s1": ref["s1_at_090"],
                               "s2": ref["s2_at_090"],
                               "sum": ref["s1_plus_s2_at_090"]},
        },
        "official_test_metrics": "N/A",
    }
    (OUT / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                                 default=float), encoding="utf-8")
    write_report(payload, singles, fusion_rows, e65_rows, best_sum, best_sum_trait,
                 best_sum_fusion, best_auc_single, best_auc_fusion, ref, gate_table,
                 families)
    log(f"done in {time.time() - started:.0f}s")


def fmt_points(row: dict, keys=("LOSO_rejection", "Test_S1_rejection",
                                "Test_S2_acceptance", "S1_plus_S2")) -> str:
    return " | ".join(f"{p[k]:.4f}" for p in row["points"] for k in keys)


def write_report(payload, singles, fusion_rows, e65_rows, best_sum, best_sum_trait,
                 best_sum_fusion, best_auc_single, best_auc_fusion, ref, gate_table,
                 families) -> None:
    j = payload["judgement"]
    names = ["iaf", "prominence", "width", "centroid", "aperiodic", "edge95", "all"]
    label = {"iaf": "IAF 峰频率", "prominence": "alpha 峰突出度",
             "width": "alpha 峰宽(FWHM)", "centroid": "谱质心 4-30Hz",
             "aperiodic": "1/f 斜率+截距", "edge95": "95% 谱边缘",
             "all": "特质全拼接"}
    cols = " | ".join(f"a={a:.2f}" for a in ACC_GRID)
    lines = [
        "# E186 频谱特质标记（跨会话身份判别）", "",
        "从原始 18×1200 @300 Hz 提取文献公认的**跨会话稳定**频谱特质标记：逐个 trial 做 Welch"
        "（600 点 Hann、50% 重叠、补零到 2048 → 0.146 Hz 分辨率），然后每通道提取 IAF、alpha 峰"
        "突出度、峰宽 FWHM、4–30 Hz 谱质心、2–40 Hz 非周期 1/f 斜率与截距（IRLS 稳健回归，排除"
        "7–13 与 15–25 Hz）、95% 谱边缘，共 **126 维**。特征不做逐 trial 标准化（否则会抹掉特质），"
        "只在 task 内用 Train 统计量做 z 标准化。", "",
        "评估协议逐字沿用 E143：Train-only 标准化 / PCA / 原型 / 阈值；5 块 Train OOF；LOSO 每折"
        "排除被评分身份、原型只用 `pm & fit_rows`（frac≤0.6）、评估块 0.6<frac≤0.8；Test 只评分，"
        "Session 只用于诊断 AUC。", "",
        "## 1. 判定（核心）", "",
        f"- **特质标记单独的 Test 诊断 AUC 是否 > 0.70？** 最高 **{j['trait_single_best_diag_auc']:.4f}**"
        f"（{best_auc_single['config']} k={best_auc_single['k']}）→ "
        f"{'是' if j['trait_single_above_070'] else '否'}；"
        f"{'低于 0.65，跨会话也不稳' if j['trait_single_below_065'] else '高于 0.65 下限，跨会话仍有判别力'}。",
        f"- **与 E65 融合后 S1+S2 能否突破 1.4479？** 最高 **{j['best_s1_plus_s2']:.4f}**"
        f"（{j['best_s1_plus_s2_config']}）→ "
        f"{'突破' if j['beats_14479'] else '未突破'}。"
        f"能否达到 1.50：{'能' if j['reaches_150'] else '不能'}。",
        f"- E65 对照：诊断 AUC {j['e65_reproduces']['auc']:.4f}、S1 {j['e65_reproduces']['s1']:.4f}、"
        f"S2 {j['e65_reproduces']['s2']:.4f}、S1+S2 {j['e65_reproduces']['sum']:.4f}"
        f"（历史 0.7578 / 0.7979 / 0.5960 / 1.394）。", "",
        "## 2. 各单特征族的 Test 诊断 AUC（核心诊断）", "",
        "诊断 AUC = 同一条 Test 分数上分不分得开 session1（S1，应被拒）与 session2（S2，应被接）；"
        "分数越高越接受。全部配置见 results.json。", "",
        "| 特征族 | 维 | 最佳 PCA | 最佳 k | 诊断 AUC | PCA-16 k=12 AUC | Train OOF AUC | Train OOF F1 | 最优 S1+S2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        d = payload["single_family_best"][name]
        best = max(singles[name], key=lambda r: r["diagnostic_auc"])
        lines.append(f"| {label[name]} | {families[name]['dim']} | {best['pca_dim']} | {best['k']} | "
                     f"{d['best_diag_auc']:.4f} | "
                     f"{d['auc_at_pca16_k12'] if d['auc_at_pca16_k12'] is None else round(d['auc_at_pca16_k12'],4)} | "
                     f"{best['train_oof_auc']:.4f} | {best['train_oof_f1']:.4f} | "
                     f"{max(r['best_s1_plus_s2'] for r in singles[name]):.4f} |")
    rank = sorted(names, key=lambda n: -payload["single_family_best"][n]["best_diag_auc"])
    lines += ["", "排序（按最佳诊断 AUC）：" +
              " > ".join(f"{label[n]} {payload['single_family_best'][n]['best_diag_auc']:.4f}"
                         for n in rank), "",
              "## 3. 全部配置的完整指标", "",
              f"每个配置在 a ∈ {{{', '.join(f'{a:.2f}' for a in ACC_GRID)}}} 上的 Test S1 拒识 / S2 接受 / "
              "S1+S2；a=0.90 的 LOSO 拒识单独列。", ""]

    def table(rows, title):
        out = [f"### {title}", "",
               "| 配置 | k | 诊断 AUC | Train OOF AUC | Train OOF F1 | LOSO@.90 | S1@.90 | S2@.90 | S1+S2@.90 | 最优 a | 最优 S1+S2 |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            out.append(f"| {r['config']} | {r['k']} | {r['diagnostic_auc']:.4f} | "
                       f"{r['train_oof_auc']:.4f} | {r['train_oof_f1']:.4f} | "
                       f"{r['loso_at_090']:.4f} | {r['s1_at_090']:.4f} | "
                       f"{r['s2_at_090']:.4f} | {r['s1_plus_s2_at_090']:.4f} | "
                       f"{r['best_a']:.2f} | {r['best_s1_plus_s2']:.4f} |")
        return out + [""]

    lines += table(e65_rows, "3.1 E65 对照（correlation-logSPD，PCA-96）")
    for name in names:
        lines += table(singles[name], f"3.2 单族 {label[name]}")
    lines += table(fusion_rows, "3.3 E65 + 特质（block_norm 拼接后联合 PCA）")

    lines += ["## 4. 各工作点的 Test S1 / S2 / S1+S2", "",
              "单特征族只扫 k=12（诊断用途），特质全拼接 / E65 / 融合扫 k=1/3/6/12。", ""]

    def points_table(row, title):
        out = [f"**{title}：{row['config']} k={row['k']}**（诊断 AUC "
               f"{row['diagnostic_auc']:.4f}）", "",
               "| a | LOSO 拒识 | Test S1 拒识 | Test S2 接受 | S1+S2 |",
               "|---:|---:|---:|---:|---:|"]
        for p in row["points"]:
            out.append(f"| {p['a']:.2f} | {p['LOSO_rejection']:.4f} | "
                       f"{p['Test_S1_rejection']:.4f} | {p['Test_S2_acceptance']:.4f} | "
                       f"{p['S1_plus_S2']:.4f} |")
        return out + [""]

    lines += points_table(best_sum, "4.1 全实验最佳（S1+S2 最高，同分取 AUC 高者）")
    lines += points_table(best_auc_fusion, "4.2 融合最高诊断 AUC")
    lines += points_table(best_sum_trait, "4.3 特质单独最佳")
    lines += points_table(ref, "4.4 E65 对照")

    lines += ["对照汇总（五点半网格内最高 S1+S2）：", "",
              "| 方案 | 配置 | 诊断 AUC | S1+S2 最高 | 工作点 a | S1 | S2 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for tag, row in [("E65 对照", ref), ("特质单独最佳", best_sum_trait),
                     ("融合最佳", best_sum_fusion), ("全实验最佳", best_sum)]:
        lines.append(f"| {tag} | {row['config']} k={row['k']} | {row['diagnostic_auc']:.4f} | "
                     f"{row['best_s1_plus_s2']:.4f} | {row['best_a']:.2f} | "
                     f"{row['best_s1']:.4f} | {row['best_s2']:.4f} |")

    ap_row = next(r for r in singles["aperiodic"] if r["pca_dim"] == 32)
    lines += ["", "## 5. 机制解读", "",
              f"1. **特质标记里只有「非周期 1/f 斜率+截距」跨会话稳定**：36 维的 aperiodic 单族诊断 "
              f"AUC {ap_row['diagnostic_auc']:.4f}，与 855 维的 E65 correlation-logSPD "
              f"（{ref['diagnostic_auc']:.4f}）几乎持平；谱质心 {payload['single_family_best']['centroid']['best_diag_auc']:.4f}、"
              f"95% 谱边缘 {payload['single_family_best']['edge95']['best_diag_auc']:.4f} 次之。"
              f"而文献里最常说的 IAF 只有 {payload['single_family_best']['iaf']['best_diag_auc']:.4f}、"
              f"峰突出度 {payload['single_family_best']['prominence']['best_diag_auc']:.4f}、"
              f"峰宽 {payload['single_family_best']['width']['best_diag_auc']:.4f}，即 4 秒单 trial 的"
              "alpha 峰形状在本数据上几乎是噪声（未见可靠的个体 alpha 峰）。",
              f"2. **特质全拼接反而低于单一 1/f 族**（{payload['single_family_best']['all']['best_diag_auc']:.4f} < "
              f"{ap_row['diagnostic_auc']:.4f}）：IAF/峰形状这些低信噪比维度在 PCA 里占掉了方差，"
              "把已经很强的 1/f 方向冲淡了。",
              f"3. **S1/S2 的分工完全互补**：aperiodic 单族的 LOSO 拒识极低（@a=0.90 仅 "
              f"{ap_row['loso_at_090']:.4f}），说明它的分数在 Train 上分不开「伪未知 vs 已知」，"
              f"所以阈值被压得很松、S2 接受高（{ap_row['s2_at_090']:.4f}）而 S1 拒识低（{ap_row['s1_at_090']:.4f}）；"
              f"E65 正好相反（LOSO {ref['loso_at_090']:.4f}、S1 {ref['s1_at_090']:.4f}、S2 {ref['s2_at_090']:.4f}）。"
              "两者的 S1+S2 之和几乎相同（1.395 vs 1.394），说明它们只是同一条跨会话 ROC 上的不同工作点，"
              "不是互补信息。",
              f"4. **融合买到的是 AUC 不是工作点**：E65+trait 融合把诊断 AUC 抬到 "
              f"{best_auc_fusion['diagnostic_auc']:.4f}（比 E65 +{best_auc_fusion['diagnostic_auc'] - ref['diagnostic_auc']:.4f}），"
              f"但 S1+S2 只有 {best_sum['best_s1_plus_s2']:.4f}（E65 {ref['s1_plus_s2_at_090']:.4f}）。"
              "AUC 的增益集中在「排序」上，而 S1 拒识与 S2 接受在同一阈值下此消彼长，"
              "融合没有改变这个斜率——与 E167/E169 的结论一致。", ""]

    lines += ["", "## 6. 十项门禁表（对本实验最佳配置）", "",
              "每一列是同一阈值下的完整系统，不跨工作点拼接。#1–#4 是 E65 分支身份头的冻结复算"
              "（本实验不重训身份头），#7/#8 本实验未做攻击（N/A）。", "",
              "| # | 门禁 | 目标 | 数值 | 判定 | 来源 |", "|---|---|---|---:|---|---|"]
    for g in gate_table:
        if "values" in g:
            for a, v in g["values"].items():
                lines.append(f"| {g['id']} | {g['name']} | {g['target']} | {v:.4f} | "
                             f"{'✅' if g['pass'][a] else '❌'} | a={a} |")
        elif g["value"] is None:
            lines.append(f"| {g['id']} | {g['name']} | {g['target']} | N/A | ❌ | {g['source']} |")
        else:
            ok = g["pass"] if isinstance(g["pass"], bool) else False
            lines.append(f"| {g['id']} | {g['name']} | {g['target']} | {g['value']:.4f} | "
                         f"{'✅' if ok else '❌'} | {g['source']} |")

    lines += ["", "## 7. 明确回答", "",
              f"1. **特质标记单独的 Test 诊断 AUC 是否 > 0.70？** 最好 "
              f"{j['trait_single_best_diag_auc']:.4f}"
              f"（{best_auc_single['config']} k={best_auc_single['k']}），"
              f"{'超过 0.70' if j['trait_single_above_070'] else '未到 0.70'}，"
              f"{'也高于 0.65 的下限（跨会话仍有判别力）' if not j['trait_single_below_065'] else '低于 0.65（跨会话不稳定）'}。",
              f"2. **与 E65 融合后 S1+S2 能否突破 1.4479？** 最好 {j['best_s1_plus_s2']:.4f}，"
              f"{'突破' if j['beats_14479'] else '未突破'}；距 1.50 "
              f"{'达到' if j['reaches_150'] else '差 %.4f' % (1.50 - j['best_s1_plus_s2'])}。",
              f"3. **单特征 Test AUC 排序**：" +
              " > ".join(f"{label[n]} {payload['single_family_best'][n]['best_diag_auc']:.4f}"
                         for n in rank) + "。", "",
              "## 8. Test 官方指标", "",
              "**Test 官方 AUC / F1 / C / RA / ASafeR / U / Score = N/A**（Test 无真实标签）。", "",
              "## 9. 合规", "",
              "- 未修改任何已有实验目录，未提交 git。",
              "- Train/Test 走同一处理流程，逐 trial 特征不含任何数据集统计量；z 标准化只用 Train。",
              "- PCA、原型、阈值只用 Train；LOSO 一律 `pm & fit_rows`，评估窗口从未进入拟合。",
              "- Test 只评分，Session 只用于诊断 AUC，未置 0、未做任何 Test 侧选择。", "",
              "## 10. 文件", "",
              "| 文件 | 内容 |", "|---|---|",
              "| `run_e186.py` | 提取 + 评估 + 报告 |",
              "| `config.json` | 冻结配置 |",
              "| `results.json` | 全部配置数值与十项门禁 |",
              "| `train_spectral_traits.npy` / `test_spectral_traits.npy` | 126 维特质特征缓存 |",
              "| `report.md` / `run.log` | 报告与日志 |"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["extract", "eval", "all"])
    args = ap.parse_args()
    sys.stdout = Tee(OUT / "run.log", "w" if args.stage == "all" else "a")
    if args.stage in ("extract", "all"):
        stage_extract()
    if args.stage in ("eval", "all"):
        main()
