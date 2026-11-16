"""E200 frozen gate predictor: `subwin6_pca96_k1 @ a=0.90`.

Self-contained: the only inputs are
  * `bundle/frozen_bundle.pt` (task mean/sd, PCA mean/basis, 20 x 4 prototypes, tau),
  * the raw `.mat` trials listed in a metadata CSV.
Nothing from `research/E*` is imported; no statistic is estimated at inference time.

Decision (single threshold, no session term):
  raw (18 x 1200) -> 6 sub-windows -> each runs the full E65 pipeline -> mean 855-d
  -> per-task z-score -> PCA-96 -> nearest euclidean distance to the 20 same-task
  prototypes -> distance > tau -> Label 0 (unregistered), else that identity (1..20).

Usage
  python predict.py --meta <metadata.csv> --root <data root> --out <submission.csv>
                    [--device cuda] [--limit N] [--batch-size 256] [--io-workers 8]
                    [--work <dir>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
DEFAULT_BUNDLE = HERE / "bundle" / "frozen_bundle.pt"
SAVE_EVERY = 1024
TASKS = ["EC", "EO", "Imagery", "SSVEP"]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mat(path: Path, n: int, channels: int) -> np.ndarray:
    data = np.asarray(loadmat(path, variable_names=["data"])["data"], dtype=np.float32)
    if data.shape != (n, channels):
        raise ValueError(f"{path}: expected ({n},{channels}) got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: non-finite")
    return np.ascontiguousarray(data.T)


def logspd_subwindow(x: torch.Tensor, length: int, feature: dict) -> torch.Tensor:
    """Full E65 flow on one sub-window (B x 18 x length) -> B x 855."""
    channels = int(feature["channels"])
    fs = float(feature["sampling_rate_hz"])
    bands = feature["bands_hz"]
    freq = torch.fft.rfftfreq(length, d=1.0 / fs, device=x.device)
    masks = []
    for index, (low, high) in enumerate(bands):
        upper = freq <= high if index == len(bands) - 1 else freq < high
        masks.append((freq >= low) & upper)
    mask = torch.stack(masks).to(x.dtype)
    spectrum = torch.fft.rfft(x, dim=-1)
    filtered = torch.fft.irfft(spectrum[:, None] * mask[None, :, None, :], n=length, dim=-1)
    filtered = filtered - filtered.mean(dim=-1, keepdim=True)
    ddof = int(feature["channel_std_ddof"])
    scale = filtered.std(dim=-1, keepdim=True, unbiased=ddof == 1).clamp_min(
        float(feature["channel_std_epsilon"]))
    normalised = (filtered - filtered.mean(dim=-1, keepdim=True)) / scale
    correlation = normalised @ normalised.transpose(-1, -2) / (length - ddof)
    trace = torch.diagonal(correlation, dim1=-2, dim2=-1).sum(dim=-1)
    eye = torch.eye(channels, dtype=x.dtype, device=x.device)
    shrinkage = float(feature["covariance_shrinkage"])
    spd = ((1.0 - shrinkage) * correlation
           + shrinkage * trace[..., None, None] / channels * eye)
    values, vectors = torch.linalg.eigh(spd)
    values = values.clamp_min(float(feature["eigenvalue_floor"]))
    logspd = (vectors * values.log().unsqueeze(-2)) @ vectors.transpose(-1, -2)
    rows, columns = torch.triu_indices(channels, channels, device=x.device)
    vec = logspd[..., rows, columns]
    vec[..., rows != columns] *= float(feature["off_diagonal_multiplier"])
    vec = vec.reshape(len(vec), len(bands), -1)
    rows_np, cols_np = np.triu_indices(channels)
    diagonal = torch.as_tensor(rows_np == cols_np, device=x.device)
    diag_mean = vec[..., diagonal].mean(dim=-1, keepdim=True)
    vec[..., diagonal] = vec[..., diagonal] - diag_mean
    mean = vec.mean(dim=-1, keepdim=True)
    sd = vec.std(dim=-1, keepdim=True, unbiased=False).clamp_min(
        float(feature["zband_epsilon"]))
    return ((vec - mean) / sd).flatten(-2)


def subwin6(x: torch.Tensor, feature: dict) -> torch.Tensor:
    count = int(feature["n_subwindows"])
    length = int(feature["subwindow_length"])
    parts = [logspd_subwindow(x[..., w * length:(w + 1) * length], length, feature)
             for w in range(count)]
    return torch.stack(parts, dim=1).mean(dim=1)


def extract_features(meta: pd.DataFrame, root: Path, feature: dict, device: torch.device,
                     work: Path, io_workers: int, batch_size: int,
                     meta_sha: str, limit: int) -> np.ndarray:
    """Subwin6 features for the first `limit` metadata rows; resumes from `work`."""
    n_rows = limit
    dim = int(feature["dim"])
    store = work / "features.npy"
    progress_path = work / "progress.json"
    work.mkdir(parents=True, exist_ok=True)
    done = 0
    if store.is_file() and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (progress.get("meta_sha256") == meta_sha
                and int(progress.get("total_rows", -1)) == n_rows
                and tuple(np.load(store, mmap_mode="r").shape) == (n_rows, dim)):
            done = int(progress.get("rows_done", 0))
            log(f"resume: {done}/{n_rows} feature rows already on disk")
    if done == 0:
        memmap = np.lib.format.open_memmap(store, mode="w+", dtype=np.float32,
                                           shape=(n_rows, dim))
        memmap.flush()
        del memmap
        progress_path.write_text(json.dumps({"meta_sha256": meta_sha,
                                             "total_rows": n_rows, "rows_done": 0}),
                                 encoding="utf-8")
    output = np.lib.format.open_memmap(store, mode="r+")
    paths = meta["path"].to_numpy(dtype=object)
    n_samples = int(feature["samples_per_trial"])
    channels = int(feature["channels"])
    started = time.time()
    if done >= n_rows:
        log(f"features complete: {n_rows} rows")
        return np.asarray(output)
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        for start in range(done, n_rows, batch_size):
            stop = min(start + batch_size, n_rows)
            chunk = [root / str(p) for p in paths[start:stop]]
            arrays = list(pool.map(lambda p: load_mat(p, n_samples, channels), chunk))
            raw = torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=device)
            with torch.no_grad():
                output[start:stop] = subwin6(raw, feature).cpu().numpy()
            if stop % SAVE_EVERY < batch_size or stop == n_rows:
                output.flush()
                progress_path.write_text(json.dumps({"meta_sha256": meta_sha,
                                                     "total_rows": n_rows,
                                                     "rows_done": int(stop)}),
                                         encoding="utf-8")
                log(f"  features {stop}/{n_rows} ({time.time() - started:.0f}s)")
    output.flush()
    return np.asarray(output)


def score(features: np.ndarray, tasks: np.ndarray, bundle: dict,
          device: torch.device, batch: int = 4096) -> tuple:
    stage1, stage2 = bundle["stage1"], bundle["stage2"]
    order = list(stage1["task_order"])
    mu = torch.as_tensor(np.asarray(stage1["task_stats_mu"], dtype=np.float32), device=device)
    sd = torch.as_tensor(np.asarray(stage1["task_stats_sd"], dtype=np.float32), device=device)
    pca_mu = torch.as_tensor(np.asarray(stage1["pca_mean"], dtype=np.float32), device=device)
    basis = torch.as_tensor(np.asarray(stage1["pca_basis"], dtype=np.float32), device=device)
    proto = np.asarray(stage2["prototypes"], dtype=np.float32)
    classes = np.asarray(stage2["classes"], dtype=np.int64)
    task_index = np.asarray([order.index(str(t)) for t in tasks], dtype=np.int64)
    protos = [torch.as_tensor(proto[i].reshape(len(classes), -1), device=device)
              for i in range(len(order))]
    dist = np.full(len(features), np.nan, dtype=np.float32)
    ident = np.full(len(features), -1, dtype=np.int64)
    for start in range(0, len(features), batch):
        stop = min(start + batch, len(features))
        x = torch.as_tensor(np.asarray(features[start:stop], dtype=np.float32), device=device)
        index = torch.as_tensor(task_index[start:stop], device=device)
        z = (x - mu[index]) / sd[index]
        proj = (z - pca_mu) @ basis
        for i in range(len(order)):
            take = index == i
            if not bool(take.any()):
                continue
            values, neighbours = torch.cdist(proj[take], protos[i]).min(dim=1)
            rows = torch.nonzero(take).squeeze(1).cpu().numpy()
            dist[start + rows] = values.cpu().numpy()
            ident[start + rows] = classes[neighbours.cpu().numpy()]
    return dist, ident


def formatted_sample(value: object) -> str:
    text = str(value).strip()
    match = re.fullmatch(r"(?i)test[_\-]?(\d+)", text)
    if match:
        return f"test_{int(match.group(1)):03d}"
    if text.isdigit():
        return f"test_{int(text):03d}"
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", required=True, help="metadata CSV with sample/path/task")
    parser.add_argument("--root", required=True, help="data root containing the .mat paths")
    parser.add_argument("--out", required=True, help="submission CSV (Sample,Label)")
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="first N metadata rows only")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--work", default="", help="resume directory (default: <out>_work)")
    args = parser.parse_args()

    bundle_path = Path(args.bundle).resolve()
    if not bundle_path.is_file():
        raise SystemExit(f"bundle not found: {bundle_path}")
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    log(f"bundle {bundle_path.name} sha256 {sha256(bundle_path)}")
    log(f"system {bundle['meta']['system']}, built {bundle['meta']['built_at']}, "
        f"git {bundle['meta']['git_commit'][:12]}")

    meta_path = Path(args.meta).resolve()
    frame = pd.read_csv(meta_path)
    missing = {"sample", "path", "task"} - set(frame.columns)
    if missing:
        raise SystemExit(f"metadata missing columns: {sorted(missing)}")
    n_take = len(frame) if args.limit <= 0 else min(args.limit, len(frame))
    n_total = len(frame)
    meta = frame[["sample", "path", "task"]].iloc[:n_take].reset_index(drop=True)
    del frame
    root = Path(args.root).resolve()
    log(f"metadata {meta_path} rows={len(meta)} of {n_total}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    if device.type == "cuda":
        log(f"device {device} {torch.cuda.get_device_name(0)}")
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = Path(args.work).resolve() if args.work else out_path.with_name(
        out_path.stem + "_work")

    started = time.time()
    features = extract_features(meta, root, bundle["stage1"]["feature"], device, work,
                                args.io_workers, args.batch_size, sha256(meta_path),
                                len(meta))
    dist, ident = score(features, meta["task"].to_numpy(dtype=object), bundle, device)

    tau = float(bundle["stage3"]["tau_dist"])
    labels = np.where(dist <= tau, ident, 0).astype(np.int64)
    if not np.isfinite(dist).all():
        raise RuntimeError("non-finite distances")

    submission = pd.DataFrame({"Sample": [formatted_sample(v) for v in meta["sample"]],
                               "Label": labels})
    submission.to_csv(out_path, index=False)
    log(f"wrote {out_path} ({len(submission)} rows) in {time.time() - started:.0f}s")
    if len(submission) != len(meta):
        raise RuntimeError("row count mismatch")
    if not submission["Label"].isin(range(0, 21)).all():
        raise RuntimeError("label out of range")

    log("")
    log(f"rows            : {len(submission)}")
    log(f"tau_dist        : {tau:.6f} (a={bundle['stage3']['work_point_a']:.2f}, "
        f"{bundle['stage3']['threshold_source']})")
    log(f"rejected (0)    : {int((labels == 0).sum())} "
        f"({(labels == 0).mean():.4f})")
    counts = pd.Series(labels).value_counts().sort_index()
    log("label counts    : " + ", ".join(f"{int(k)}:{int(v)}" for k, v in counts.items()))
    log(f"distance        : mean {dist.mean():.3f}, median {np.median(dist):.3f}, "
        f"min {dist.min():.3f}, max {dist.max():.3f}")

    # ---------------- descriptive audit only (after the CSV is written; never a decision)
    # The Session column is read for the first time here, i.e. after every label exists.
    session = None
    try:
        session = pd.read_csv(meta_path, usecols=["session"]).iloc[:len(meta)][
            "session"].astype(str).to_numpy()
    except ValueError:
        pass
    if session is not None:
        log("session audit (descriptive only, no branch in the decision):")
        for name in sorted(set(session.tolist())):
            rows = session == name
            log(f"  {name}: N={int(rows.sum())} rejected={int((labels[rows] == 0).sum())} "
                f"rejection_rate={(labels[rows] == 0).mean():.4f} "
                f"acceptance_rate={(labels[rows] != 0).mean():.4f}")


if __name__ == "__main__":
    main()
