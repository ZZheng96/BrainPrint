"""Euclidean open-set gate:
  x (18 x 1200) -> 6 sub-windows of 200 samples -> per sub-window: 5 FFT bands ->
  channel standardisation -> correlation -> shrinkage -> logSPD -> upper triangle ->
  per-band detrend/z-score -> mean of the 6 855-d vectors -> per-task z-score ->
  PCA-96 -> euclidean distance to per-task identity prototypes.

Score matrix (N, 24): column 0 = -tau (unknown), column c = -distance to identity c
(-inf if identity c is not enrolled). argmax == nearest identity if its distance
< tau, otherwise unknown. No pretrained parameter is changed by enroll().
"""
from __future__ import annotations

import os

import numpy as np
import torch

N_COLS = 24
N_BASE = 20
NEW_IDS = (21, 22, 23)
TASKS = ("EC", "EO", "Imagery", "SSVEP")
_TASK_ALIASES = {
    "ec": 0, "eyesclosed": 0, "eyeclosed": 0, "restec": 0, "restingec": 0, "closed": 0,
    "task1": 0, "1": 0,
    "eo": 1, "eyesopen": 1, "eyeopen": 1, "resteo": 1, "restingeo": 1, "open": 1,
    "task2": 1, "2": 1,
    "imagery": 2, "mi": 2, "vi": 2, "visualimagery": 2, "imagination": 2, "task3": 2, "3": 2,
    "ssvep": 3, "task4": 3, "4": 3,
}
_TASK_KEYS = ("task", "Task", "TASK", "paradigm", "Paradigm", "task_name")


def _task_index(meta_row) -> int:
    """Map a meta dict to a task index 0..3, or -1 if the task is unknown."""
    if not isinstance(meta_row, dict):
        return -1
    for key in _TASK_KEYS:
        if key in meta_row and meta_row[key] is not None:
            text = str(meta_row[key]).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
            if text in _TASK_ALIASES:
                return _TASK_ALIASES[text]
            if text in ("闭眼", "睁眼", "想象", "视觉想象"):
                return {"闭眼": 0, "睁眼": 1, "想象": 2, "视觉想象": 2}[text]
    return -1


def _task_indices(meta, n: int) -> np.ndarray:
    if meta is None:
        return np.full(n, -1, dtype=np.int64)
    if isinstance(meta, dict):
        meta = [meta] * n if n != 1 else [meta]
    rows = list(meta)
    if len(rows) != n:
        return np.full(n, -1, dtype=np.int64)
    return np.asarray([_task_index(r) for r in rows], dtype=np.int64)


def _subwindow_feature(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    """(B, C, L) -> (B, 855) for one sub-window."""
    length = x.shape[-1]
    channels = int(cfg["channels"])
    bands = cfg["bands_hz"]
    freq = torch.fft.rfftfreq(length, d=1.0 / float(cfg["sampling_rate_hz"]), device=x.device)
    masks = []
    for i, (low, high) in enumerate(bands):
        upper = freq <= high if i == len(bands) - 1 else freq < high
        masks.append((freq >= low) & upper)
    mask = torch.stack(masks).to(x.dtype)
    spectrum = torch.fft.rfft(x, dim=-1)
    filtered = torch.fft.irfft(spectrum[:, None] * mask[None, :, None, :], n=length, dim=-1)
    filtered = filtered - filtered.mean(dim=-1, keepdim=True)
    ddof = int(cfg["channel_std_ddof"])
    scale = filtered.std(dim=-1, keepdim=True, unbiased=ddof == 1).clamp_min(
        float(cfg["channel_std_epsilon"]))
    normalised = (filtered - filtered.mean(dim=-1, keepdim=True)) / scale
    corr = normalised @ normalised.transpose(-1, -2) / (length - ddof)
    trace = torch.diagonal(corr, dim1=-2, dim2=-1).sum(dim=-1)
    eye = torch.eye(channels, dtype=x.dtype, device=x.device)
    shrink = float(cfg["covariance_shrinkage"])
    spd = (1.0 - shrink) * corr + shrink * trace[..., None, None] / channels * eye
    values, vectors = torch.linalg.eigh(spd)
    values = values.clamp_min(float(cfg["eigenvalue_floor"]))
    logspd = (vectors * values.log().unsqueeze(-2)) @ vectors.transpose(-1, -2)
    rows, cols = torch.triu_indices(channels, channels, device=x.device)
    vec = logspd[..., rows, cols]
    vec[..., rows != cols] *= float(cfg["off_diagonal_multiplier"])
    vec = vec.reshape(len(vec), len(bands), -1)
    diag = (rows == cols)
    vec[..., diag] = vec[..., diag] - vec[..., diag].mean(dim=-1, keepdim=True)
    mean = vec.mean(dim=-1, keepdim=True)
    sd = vec.std(dim=-1, keepdim=True, unbiased=False).clamp_min(float(cfg["zband_epsilon"]))
    return ((vec - mean) / sd).flatten(-2)


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.array(x, dtype=np.float32, copy=True)  # never touch the caller's array


def _features(art: dict, x) -> torch.Tensor:
    cfg = art["feature"]
    data = _as_numpy(x)
    if data.ndim == 2:
        data = data[None]
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    t = torch.from_numpy(data).to(art["device"])
    count, length = int(cfg["n_subwindows"]), int(cfg["subwindow_length"])
    parts = [_subwindow_feature(t[..., w * length:(w + 1) * length], cfg) for w in range(count)]
    feat = torch.stack(parts, dim=1).mean(dim=1)
    return torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


def _project(art: dict, feat: torch.Tensor, task: int) -> torch.Tensor:
    z = (feat - art["task_mu"][task]) / art["task_sd"][task]
    return (z - art["pca_mean"]) @ art["pca_basis"]


def from_weights(w, device: str = "cpu"):
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)
    base = w["prototypes"].float()  # (4, 20, 96)
    protos = torch.zeros(len(TASKS), N_COLS - 1, base.shape[-1])
    protos[:, :N_BASE] = base
    valid = torch.zeros(len(TASKS), N_COLS - 1, dtype=torch.bool)
    valid[:, :N_BASE] = True
    return {
        "device": dev,
        "feature": w["feature"],
        "task_mu": w["task_mu"].float().to(dev),
        "task_sd": w["task_sd"].float().to(dev),
        "pca_mean": w["pca_mean"].float().to(dev),
        "pca_basis": w["pca_basis"].float().to(dev),
        "prototypes": protos.to(dev),   # identity c -> index c-1
        "valid": valid.to(dev),
        "tau": float(w["tau_dist"]),
    }


@torch.no_grad()
def predict_batch(artifact, x, meta):
    feat = _features(artifact, x)
    n = int(feat.shape[0])
    tasks = _task_indices(meta, n)
    dist = torch.full((n, N_COLS - 1), float("inf"), device=artifact["device"])
    for t in range(len(TASKS)):
        # known task -> that task only; unknown task -> best match over all tasks
        take = torch.as_tensor((tasks == t) | (tasks < 0), device=artifact["device"])
        if not bool(take.any()):
            continue
        proj = _project(artifact, feat[take], t)
        d = torch.cdist(proj, artifact["prototypes"][t])
        d = d.masked_fill(~artifact["valid"][t][None], float("inf"))
        dist[take] = torch.minimum(dist[take], d)
    scores = torch.empty((n, N_COLS), dtype=torch.float32, device=artifact["device"])
    scores[:, 0] = -artifact["tau"]
    scores[:, 1:] = -dist
    out = scores.cpu().numpy()
    out[np.isnan(out)] = -np.inf
    out[:, 0] = -artifact["tau"]
    return out


@torch.no_grad()
def enroll(artifact, x, meta, labels):
    # All enrollment samples of a user are pooled under every task's standardisation.
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    feat = _features(artifact, x)
    for uid in np.unique(labels):
        uid = int(uid)
        if uid not in NEW_IDS:
            continue
        rows = torch.as_tensor(labels == uid, device=artifact["device"])
        for t in range(len(TASKS)):
            artifact["prototypes"][t, uid - 1] = _project(artifact, feat[rows], t).mean(dim=0)
            artifact["valid"][t, uid - 1] = True
    return artifact
