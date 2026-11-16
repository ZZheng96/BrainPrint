"""E200: package the fitted state of the E191 gate `subwin6_pca96_k1 @ a=0.90`.

The identity backbone weights were saved by E191, but the gate's *fitted* state
(task mean/sd, PCA basis, prototypes, threshold) only lived inside E189's cache
arrays.  This script re-derives that state from the frozen Train feature cache,
verifies it against E189's stored arrays, and writes one portable bundle that
`predict.py` can consume without importing any research/E* module.

Train only for every fitted quantity:
  * task-wise mean/sd   : all Train rows of that task (E143/E189 rule)
  * PCA mean / basis    : fit rows only (frac <= 0.6), dim 96
  * prototypes          : per (task, identity), fit rows only, k = 1
  * tau                 : 90% quantile of the 5-block Train-OOF distance (a = 0.90)
Test is read only to score the packaged system (reproduction check / S1-S2 audit).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BUNDLE_DIR = OUT / "bundle"
CACHE = OUT / "cache"
LOG_PATH = OUT / "run.log"
BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FT = torch.float32
TASKS = ["EC", "EO", "Imagery", "SSVEP"]
CLASSES = np.arange(1, 21, dtype=int)
N, CH, FS = 1200, 18, 300
NB = 5
SUBWIN = 6
A = 0.90
FIT_FRAC = 0.6
BLOCKS = 5

TRAIN_META = ROOT / "BrainPrint_cache/train_metadata.csv"
TEST_META = ROOT / "BrainPrint_cache/test_metadata.csv"
TRAIN_FEAT = ROOT / "research/E171_multiscale_fusion/train_subwin_mean_n6.npy"
TEST_FEAT = ROOT / "research/E171_multiscale_fusion/test_subwin_mean_n6.npy"
E189_SCORES = ROOT / "research/E189_balanced_gate_gates/cache/stream_scores.npz"
E191_AUDIT = ROOT / "research/E191_fix_gate1_gate7/audit_v3.json"
BACKBONES = [
    "research/E191_fix_gate1_gate7/model_branch_a_v3.pt",
    "research/E191_fix_gate1_gate7/cache/stage_a_v3.pt",
    "research/E191_fix_gate1_gate7/cache/fold_model_v3_0.pt",
    "research/E191_fix_gate1_gate7/cache/fold_model_v3_1.pt",
    "research/E191_fix_gate1_gate7/cache/fold_model_v3_2.pt",
    "research/E191_fix_gate1_gate7/cache/fold_model_v3_3.pt",
    "research/E191_fix_gate1_gate7/cache/fold_model_v3_4.pt",
]

# Frozen E65 -> E171 subwin6 pipeline constants (E36 + E65 + E171 configs).
FEATURE = {
    "variant": "subwin_mean_n6",
    "n_subwindows": SUBWIN,
    "samples_per_trial": N,
    "subwindow_length": N // SUBWIN,
    "channels": CH,
    "sampling_rate_hz": FS,
    "bands_hz": [[1, 8], [8, 13], [13, 20], [20, 30], [30, 40]],
    "band_edges": "lower-inclusive, upper-exclusive except the last band (<= 40 Hz)",
    "channel_std_ddof": 1,
    "channel_std_epsilon": 1e-12,
    "covariance_shrinkage": 0.05,
    "eigenvalue_floor": 1e-12,
    "off_diagonal_multiplier": 1.4142135623730951,
    "zband_epsilon": 1e-6,
    "dim": NB * 171,
    "aggregation": "each sub-window runs the full E65 pipeline independently "
                   "(band isolation -> channel standardisation -> correlation -> "
                   "shrinkage -> logSPD -> upper triangle -> detrend_zband); the 6 "
                   "855-d vectors are averaged",
    "detrend_rule": "subtract each band's mean diagonal log-variance from the "
                    "diagonal entries only, then z-score the 171 entries per band "
                    "(population std)",
}


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


def log(msg: str = "") -> None:
    print(msg, flush=True)


def f32(array) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float),
                    encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return "unavailable"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        return bool(out.strip())
    except Exception:  # noqa: BLE001
        return True


# ------------------------------------------------------------------ raw -> subwin6
def load_mat(path: Path) -> np.ndarray:
    data = np.asarray(loadmat(path, variable_names=["data"])["data"], dtype=np.float32)
    if data.shape != (N, CH):
        raise ValueError(f"{path}: expected ({N},{CH}) got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: non-finite")
    return np.ascontiguousarray(data.T)


def logspd_subwindow(x: torch.Tensor, length: int) -> torch.Tensor:
    """E65 gpu_correlation_logspd for one sub-window length (B x 18 x length) -> B x 855."""
    bands = FEATURE["bands_hz"]
    freq = torch.fft.rfftfreq(length, d=1.0 / float(FS), device=x.device)
    masks = []
    for index, (low, high) in enumerate(bands):
        upper = freq <= high if index == len(bands) - 1 else freq < high
        masks.append((freq >= low) & upper)
    mask = torch.stack(masks).to(x.dtype)
    spectrum = torch.fft.rfft(x, dim=-1)
    filtered = torch.fft.irfft(spectrum[:, None] * mask[None, :, None, :], n=length, dim=-1)
    filtered = filtered - filtered.mean(dim=-1, keepdim=True)
    ddof = int(FEATURE["channel_std_ddof"])
    scale = filtered.std(dim=-1, keepdim=True, unbiased=ddof == 1).clamp_min(
        float(FEATURE["channel_std_epsilon"]))
    normalised = (filtered - filtered.mean(dim=-1, keepdim=True)) / scale
    correlation = normalised @ normalised.transpose(-1, -2) / (length - ddof)
    trace = torch.diagonal(correlation, dim1=-2, dim2=-1).sum(dim=-1)
    eye = torch.eye(CH, dtype=x.dtype, device=x.device)
    shrinkage = float(FEATURE["covariance_shrinkage"])
    spd = ((1.0 - shrinkage) * correlation
           + shrinkage * trace[..., None, None] / CH * eye)
    values, vectors = torch.linalg.eigh(spd)
    values = values.clamp_min(float(FEATURE["eigenvalue_floor"]))
    logspd = (vectors * values.log().unsqueeze(-2)) @ vectors.transpose(-1, -2)
    rows, columns = torch.triu_indices(CH, CH, device=x.device)
    vec = logspd[..., rows, columns]
    vec[..., rows != columns] *= float(FEATURE["off_diagonal_multiplier"])
    vec = vec.reshape(len(vec), NB, 171)
    rows_np, cols_np = np.triu_indices(CH)
    diagonal = torch.as_tensor(rows_np == cols_np, device=x.device)
    diag_mean = vec[..., diagonal].mean(dim=-1, keepdim=True)
    vec[..., diagonal] = vec[..., diagonal] - diag_mean
    mean = vec.mean(dim=-1, keepdim=True)
    sd = vec.std(dim=-1, keepdim=True, unbiased=False).clamp_min(
        float(FEATURE["zband_epsilon"]))
    return ((vec - mean) / sd).flatten(-2)


def subwin6(x: torch.Tensor) -> torch.Tensor:
    length = FEATURE["subwindow_length"]
    parts = [logspd_subwindow(x[..., w * length:(w + 1) * length], length)
             for w in range(SUBWIN)]
    return torch.stack(parts, dim=1).mean(dim=1)


def rebuild_features(rows: np.ndarray, meta: pd.DataFrame, split: str) -> np.ndarray:
    base = ROOT / "Brainprint_Competition"
    paths = [base / str(p) for p in meta["path"].to_numpy()[rows]]
    with ThreadPoolExecutor(max_workers=8) as pool:
        arrays = list(pool.map(load_mat, paths))
    raw = torch.as_tensor(np.stack(arrays), dtype=FT, device=DEVICE)
    with torch.no_grad():
        return subwin6(raw).cpu().numpy()


# ------------------------------------------------------------------ stage 1 / stage 2
def task_stats(features: np.ndarray, tasks: np.ndarray) -> tuple:
    mu = np.stack([f32(features[tasks == task].mean(0)) for task in TASKS])
    sd = np.stack([f32(features[tasks == task].std(0) + 1e-6) for task in TASKS])
    return mu, sd


def apply_stats(features: np.ndarray, tasks: np.ndarray, mu: np.ndarray,
                sd: np.ndarray) -> np.ndarray:
    out = features.astype(np.float64).copy()
    for index, task in enumerate(TASKS):
        ref = tasks == task
        out[ref] = (out[ref] - mu[index].astype(np.float64)) / sd[index].astype(np.float64)
    return f32(out)


def pca_basis(x: np.ndarray, dim: int) -> tuple:
    data = np.asarray(x, dtype=np.float64)
    mu = data.mean(0)
    centred = data - mu
    values, vectors = np.linalg.eigh(centred.T @ centred / max(len(centred) - 1, 1))
    basis = vectors[:, np.argsort(-values)][:, : min(int(dim), data.shape[1])]
    return f32(mu), f32(basis)


def project(features: np.ndarray, tasks: np.ndarray, mu: np.ndarray, sd: np.ndarray,
            pca_mu: np.ndarray, basis: np.ndarray) -> np.ndarray:
    z = apply_stats(features, tasks, mu, sd)
    return f32((z.astype(np.float64) - pca_mu.astype(np.float64)) @ basis)


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


def build_prototypes(proj: np.ndarray, subjects: np.ndarray, tasks: np.ndarray,
                     mask: np.ndarray, k: int = 1) -> np.ndarray:
    """(len(TASKS), 20, k, 96) same construction as E180.build_prototypes."""
    out = np.zeros((len(TASKS), len(CLASSES), k, proj.shape[1]), dtype=np.float32)
    for task_index, task in enumerate(TASKS):
        for subject in CLASSES:
            sel = mask & (tasks == task) & (subjects == subject)
            if sel.sum() == 0:
                continue
            x = torch.as_tensor(f32(proj[sel]), device=DEVICE)
            centres = kmeans_t(x, k, int(subject) * 17 + task_index)
            out[task_index, subject - 1] = centres.cpu().numpy()
    return out


def branch_scores(proj: np.ndarray, tasks: np.ndarray, proto: np.ndarray,
                  batch: int = 4096) -> np.ndarray:
    """score = -nearest euclidean distance to the same-task prototypes (k=1)."""
    out = np.full(len(proj), np.nan, dtype=np.float32)
    for task_index, task in enumerate(TASKS):
        rows = np.flatnonzero(tasks == task)
        if len(rows) == 0:
            continue
        p = torch.as_tensor(f32(proto[task_index].reshape(len(CLASSES), -1)), device=DEVICE)
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            q = torch.as_tensor(f32(proj[chunk]), device=DEVICE)
            out[chunk] = -torch.cdist(q, p).min(dim=1).values.cpu().numpy()
    return out


def identity_of(proj: np.ndarray, tasks: np.ndarray, proto: np.ndarray,
                batch: int = 4096) -> np.ndarray:
    out = np.full(len(proj), -1, dtype=np.int32)
    for task_index, task in enumerate(TASKS):
        rows = np.flatnonzero(tasks == task)
        p = torch.as_tensor(f32(proto[task_index].reshape(len(CLASSES), -1)), device=DEVICE)
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            q = torch.as_tensor(f32(proj[chunk]), device=DEVICE)
            out[chunk] = CLASSES[torch.cdist(q, p).argmin(dim=1).cpu().numpy()]
    return out


def neighbours(proj: np.ndarray, tasks: np.ndarray, proto: np.ndarray,
               batch: int = 4096) -> tuple:
    """(nearest distance, nearest identity) in one pass."""
    dist = np.full(len(proj), np.nan, dtype=np.float32)
    ident = np.full(len(proj), -1, dtype=np.int32)
    for task_index, task in enumerate(TASKS):
        rows = np.flatnonzero(tasks == task)
        p = torch.as_tensor(f32(proto[task_index].reshape(len(CLASSES), -1)), device=DEVICE)
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            q = torch.as_tensor(f32(proj[chunk]), device=DEVICE)
            values, index = torch.cdist(q, p).min(dim=1)
            dist[chunk] = values.cpu().numpy()
            ident[chunk] = CLASSES[index.cpu().numpy()]
    return dist, ident


# ------------------------------------------------------------------ protocol
def load_protocol() -> dict:
    meta = pd.read_csv(TRAIN_META)
    assert len(meta) == 11304, len(meta)
    maximum = meta.groupby(["subject", "task"])["trial"].transform("max").to_numpy(float)
    frac = meta["trial"].to_numpy(float) / np.maximum(maximum, 1.0)
    blocks = np.clip((frac * BLOCKS).astype(int), 0, BLOCKS - 1)
    return {"meta": meta, "frac": frac, "blocks": blocks, "fit_rows": frac <= FIT_FRAC,
            "window": (frac > FIT_FRAC) & (frac <= 0.8),
            "subjects": meta["subject"].to_numpy(int),
            "tasks": meta["task"].to_numpy(dtype=object)}


def main() -> None:
    if not LOG_PATH.exists():
        LOG_PATH.write_text("", encoding="utf-8")
    sys.stdout = Tee(LOG_PATH)
    sys.stderr = sys.stdout
    started = time.time()
    log(f"\n===== E200 build_bundle {time.strftime('%Y-%m-%d %H:%M:%S')} device {DEVICE} =====")
    P = load_protocol()
    meta, frac, blocks = P["meta"], P["frac"], P["blocks"]
    subjects, tasks = P["subjects"], P["tasks"]
    fit_rows, window = P["fit_rows"], P["window"]
    log(f"Train rows {len(meta)}, fit_rows {int(fit_rows.sum())}, window {int(window.sum())}")

    # ---- frozen Train features (E171 cache, read-only)
    train_feat = f32(np.load(TRAIN_FEAT, mmap_mode="r"))
    test_meta = pd.read_csv(TEST_META)
    test_feat = f32(np.load(TEST_FEAT, mmap_mode="r"))
    log(f"features train {train_feat.shape} test {test_feat.shape}")

    # ---- feature rebuild check: raw .mat -> subwin6 must reproduce the cache
    rng = np.random.default_rng(20261116)
    rows = np.sort(rng.choice(len(meta), 8, replace=False))
    rebuilt = rebuild_features(rows, meta, "train")
    feature_check = float(np.max(np.abs(rebuilt - train_feat[rows])))
    log(f"feature rebuild vs E171 cache max|d| = {feature_check:.3e}")
    if feature_check > 1e-3:
        raise RuntimeError(f"subwin6 rebuild mismatch {feature_check}")

    # ---- stage 1: task mean/sd + PCA (Train only)
    mu, sd = task_stats(train_feat, tasks)
    train_std = apply_stats(train_feat, tasks, mu, sd)
    pca_mu, basis = pca_basis(train_std[fit_rows], 96)
    log(f"stage1: task mean/sd {mu.shape}, pca_mu {pca_mu.shape}, basis {basis.shape}")

    train_proj = project(train_feat, tasks, mu, sd, pca_mu, basis)
    test_tasks = test_meta["task"].to_numpy(dtype=object)
    test_proj = project(test_feat, test_tasks, mu, sd, pca_mu, basis)

    # ---- stage 2: prototypes from fit rows (Train only), k = 1
    proto = build_prototypes(train_proj, subjects, tasks, fit_rows, k=1)
    filled = np.load(E189_SCORES)["subwin6_pca96_k1__filled"]
    if not bool(np.asarray(filled).all()):
        raise RuntimeError("some (task, identity) prototype is missing in E189's state")
    log(f"stage2: prototypes {proto.shape} (task x identity x k x pca), "
        f"all (task, identity) cells filled")

    # ---- 5-block Train OOF score (prototypes exclude the scored block)
    oof = np.full(len(meta), np.nan, dtype=np.float32)
    for block in range(BLOCKS):
        keep = blocks == block
        p = build_prototypes(train_proj, subjects, tasks, fit_rows & ~keep, k=1)
        rows_b = np.flatnonzero(keep)
        oof[rows_b] = branch_scores(train_proj[rows_b], tasks[rows_b], p)
    if not np.isfinite(oof).all():
        raise RuntimeError("incomplete OOF")
    oof_mu = float(np.mean(oof.astype(np.float64)))
    oof_sd = float(np.std(oof.astype(np.float64))) + 1e-6

    # ---- threshold: a = 0.90 on the Train OOF distance (Train only)
    tau_dist = float(np.quantile(-oof.astype(np.float64), A))
    tau_norm = (-tau_dist - oof_mu) / oof_sd
    known_acceptance = float(np.mean(-oof.astype(np.float64) <= tau_dist))
    window_known = window
    tau_dist_window = float(np.quantile(-oof[window_known].astype(np.float64), A))
    log(f"tau_dist(OOF 90%) = {tau_dist:.6f}; tau_norm = {tau_norm:.6f}; "
        f"Train OOF acceptance = {known_acceptance:.4f}")
    log(f"reference: E191 tau = {read_json(E191_AUDIT)['subwin6_pca96_k1']['points']['0.9']['tau']:.6f}")
    log(f"sensitivity: window-only 90% quantile tau_dist = {tau_dist_window:.6f}")

    # ---- verification against E189's stored frozen state (read-only)
    stored = np.load(E189_SCORES, allow_pickle=False)
    ref = {key: stored[f"subwin6_pca96_k1__{key}"] for key in
           ("stats_mu", "stats_sd", "pca_mu", "basis", "proto", "oof")}
    stored_mu, stored_sd = float(stored["subwin6_pca96_k1__oof_mu"]), \
        float(stored["subwin6_pca96_k1__oof_sd"])
    checks = {
        "stats_mu_max_abs": float(np.max(np.abs(mu - ref["stats_mu"]))),
        "stats_sd_max_abs": float(np.max(np.abs(sd - ref["stats_sd"]))),
        "pca_mu_max_abs": float(np.max(np.abs(pca_mu - ref["pca_mu"]))),
        "basis_max_abs": float(np.max(np.abs(basis - ref["basis"]))),
        "prototypes_max_abs": float(np.max(np.abs(proto - ref["proto"]))),
        "oof_max_abs": float(np.max(np.abs(oof - ref["oof"]))),
        "oof_mu_delta": oof_mu - stored_mu,
        "oof_sd_delta": oof_sd - stored_sd,
    }
    log("E189 frozen-state check: " + ", ".join(f"{k}={v:.3e}" if isinstance(v, float)
                                                else f"{k}={v}" for k, v in checks.items()))
    if max(checks[key] for key in ("stats_mu_max_abs", "stats_sd_max_abs",
                                   "pca_mu_max_abs", "basis_max_abs",
                                   "prototypes_max_abs", "oof_max_abs")) > 1e-4:
        raise RuntimeError(f"packaged state does not match E189: {checks}")

    # ---- Test scoring (score only; session used later for the audit only)
    test_dist, test_ident = neighbours(test_proj, test_tasks, proto)
    labels = np.where(test_dist <= tau_dist, test_ident, 0).astype(np.int64)
    labels_alt = np.where(test_dist <= tau_dist_window, test_ident, 0).astype(np.int64)
    is_s1 = test_meta["session"].astype(str).str.casefold().isin(
        ["session1", "s1", "1"]).to_numpy()
    s1 = float(np.mean(labels[is_s1] == 0))
    s2 = float(np.mean(labels[~is_s1] != 0))
    s1_alt = float(np.mean(labels_alt[is_s1] == 0))
    s2_alt = float(np.mean(labels_alt[~is_s1] != 0))
    log(f"Test (OOF tau): rejected {int((labels == 0).sum())}/{len(labels)}; "
        f"S1 rejection {s1:.4f}, S2 acceptance {s2:.4f}")
    log(f"Test (window tau): S1 rejection {s1_alt:.4f}, S2 acceptance {s2_alt:.4f}")
    e191 = read_json(E191_AUDIT)["subwin6_pca96_k1"]["points"]["0.9"]
    log(f"E191 reference: S1 {e191['test_s1_rejection']:.4f}, "
        f"S2 {e191['test_s2_acceptance']:.4f}")

    # ---- cross-check against E189's stored Test scores
    stored_test = stored["subwin6_pca96_k1__test"].astype(np.float64)
    stored_label = np.where((stored_test - stored_mu) / stored_sd >= e191["tau"],
                            identity_of(test_proj, test_tasks, proto).astype(np.int64), 0)
    agree = float(np.mean(stored_label == labels))
    log(f"label agreement with E189 stored Test scores: {agree:.6f}")

    # ---- bundle
    backbones = []
    for relative in BACKBONES:
        path = ROOT / relative
        backbones.append({"path": relative, "sha256": sha256(path),
                          "bytes": int(path.stat().st_size),
                          "exists": path.is_file()})
    np.savez_compressed(CACHE / "reference_test.npz",
                        sample=test_meta["sample"].to_numpy(dtype=str),
                        task=test_tasks.astype(str), session=test_meta["session"].to_numpy(dtype=str),
                        distance=test_dist, identity=test_ident,
                        label=labels, label_window_tau=labels_alt,
                        tau_dist=np.float64(tau_dist),
                        tau_dist_window=np.float64(tau_dist_window))

    payload = {
        "format": "E200-frozen-gate-v1",
        "stage1": {
            "feature": FEATURE,
            "task_stats_mu": mu, "task_stats_sd": sd,
            "task_order": TASKS, "stats_rows": "all Train rows of that task (frac<=1.0)",
            "pca_mean": pca_mu, "pca_basis": basis, "pca_dim": int(basis.shape[1]),
            "pca_fit_rows": "fit rows only (frac<=0.6), 11304-row Train metadata",
            "standardisation": "z_t = (x - task_mean) / task_sd; p = (z_t - pca_mean) @ basis",
        },
        "stage2": {
            "prototypes": proto, "task_order": TASKS, "classes": CLASSES,
            "k_proto": 1, "index": "(task, identity) -> prototype[task, identity-1, 0]",
            "fit_rows": "frac<=0.6 only", "distance": "euclidean in the 96-d PCA space",
            "score": "score = -min_identity distance (same task only)",
        },
        "stage3": {
            "work_point_a": A,
            "tau_dist": tau_dist,
            "tau_norm": tau_norm,
            "oof_mu": oof_mu, "oof_sd": oof_sd,
            "decision": "nearest distance > tau_dist -> 0 (reject), else the nearest identity",
            "normalised_equivalent": "(oof_mu - dist) / oof_sd >= tau_norm",
            "threshold_source": "90% quantile of the 5-block Train-OOF nearest distance "
                                "(a = 0.90 known acceptance); Test never read",
            "train_known_acceptance": known_acceptance,
            "train_oof_quantile": A,
            "sensitivity_window_tau_dist": tau_dist_window,
            "sensitivity_window_source": "90% quantile of the OOF distance restricted to "
                                         "the Train evaluation window 0.6<frac<=0.8",
            "e191_reference": {"tau_norm": e191["tau"],
                               "test_s1_rejection": e191["test_s1_rejection"],
                               "test_s2_acceptance": e191["test_s2_acceptance"]},
            "class_rule": "Label 0 = unregistered; 1..20 = registered identity",
            "identity_backbone": {
                "note": "the subwin6 gate does NOT use the CNN identity head; these weights "
                        "belong to the E191 alternatives (CN closed-set readout, #1-#4)",
                "files": backbones},
        },
        "meta": {
            "experiment": "E200_final_freeze",
            "system": "subwin6_pca96_k1 @ a=0.90",
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": git_commit(),
            "git_dirty_worktree": git_dirty(),
            "feature_dim": int(FEATURE["dim"]),
            "pca_dim": int(basis.shape[1]),
            "classes": CLASSES,
            "tasks": TASKS,
            "train_rows": int(len(meta)),
            "test_rows": int(len(test_meta)),
            "session_used_for_decision": False,
            "source_checksums": {
                "train_features": sha256(TRAIN_FEAT),
                "test_features": sha256(TEST_FEAT),
                "e189_frozen_state": sha256(E189_SCORES),
                "train_metadata": sha256(TRAIN_META),
                "test_metadata": sha256(TEST_META),
            },
            "state_checks_vs_e189": checks,
            "feature_rebuild_max_abs": feature_check,
            "test_reproduction": {
                "tau_dist": tau_dist,
                "rejected": int((labels == 0).sum()),
                "s1_rejection": s1, "s2_acceptance": s2,
                "window_tau": {"tau_dist": tau_dist_window, "s1_rejection": s1_alt,
                               "s2_acceptance": s2_alt},
                "agreement_with_e189_stored_scores": agree,
            },
        },
    }
    bundle_path = BUNDLE_DIR / "frozen_bundle.pt"
    torch.save(payload, bundle_path)
    np.savez_compressed(BUNDLE_DIR / "prototypes.npz", prototypes=proto,
                        tasks=np.asarray(TASKS), classes=CLASSES,
                        pca_basis=basis, pca_mean=pca_mu,
                        task_stats_mu=mu, task_stats_sd=sd)
    bundle_sha = sha256(bundle_path)
    proto_sha = sha256(BUNDLE_DIR / "prototypes.npz")
    manifest = {
        "experiment": "E200_final_freeze",
        "system": "subwin6_pca96_k1 @ a=0.90",
        "format": payload["format"],
        "built_at": payload["meta"]["built_at"],
        "git_commit": payload["meta"]["git_commit"],
        "files": [
            {"path": "bundle/frozen_bundle.pt", "bytes": int(bundle_path.stat().st_size),
             "sha256": bundle_sha,
             "content": "stage1 + stage2 + stage3 + meta (torch.save dict)"},
            {"path": "bundle/prototypes.npz", "bytes": int((BUNDLE_DIR / 'prototypes.npz').stat().st_size),
             "sha256": proto_sha,
             "content": "prototypes (4,20,1,96) + pca basis/mean + task stats, npz"},
        ],
        "sizes": {
            "pca_basis_bytes": int(basis.nbytes), "prototypes_bytes": int(proto.nbytes),
            "total_inline_bytes": int(basis.nbytes + proto.nbytes + mu.nbytes + sd.nbytes
                                      + pca_mu.nbytes),
        },
        "stage1": {"feature": FEATURE, "task_stats": "4 x 855 mean + 4 x 855 sd",
                   "pca_mean": 855, "pca_basis": [855, 96], "pca_fit_rows": "frac<=0.6"},
        "stage2": {"k_proto": 1, "prototypes": [4, 20, 1, 96],
                   "rows": "fit rows (frac<=0.6) of all 20 identities"},
        "stage3": {"a": A, "tau_dist": tau_dist, "tau_norm": tau_norm,
                   "decision": payload["stage3"]["decision"],
                   "threshold_source": payload["stage3"]["threshold_source"],
                   "train_known_acceptance": known_acceptance,
                   "sensitivity_window_tau_dist": tau_dist_window},
        "identity_backbone": backbones,
        "source_checksums": payload["meta"]["source_checksums"],
        "state_checks_vs_e189": checks,
        "test_reproduction": payload["meta"]["test_reproduction"],
        "class_rule": "0 = unregistered, 1..20 = registered identity",
        "session_used_for_decision": False,
        "test_official_metrics": "N/A (Test has no identity labels)",
    }
    write_json(BUNDLE_DIR / "manifest.json", manifest)
    log(f"bundle written: {bundle_path} ({bundle_path.stat().st_size / 2**20:.2f} MiB) "
        f"sha256 {bundle_sha}")
    log(f"prototypes.npz sha256 {proto_sha}")
    log(f"build done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
