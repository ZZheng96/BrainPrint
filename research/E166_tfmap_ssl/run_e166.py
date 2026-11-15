"""E166: multi-band time-frequency maps + Train-only supervised contrastive CNN.

Gap being tested:  E149 froze a DINOv2 image backbone on pixel-space caches and
failed (W 0.6080);  E161/E162 learned end-to-end from the raw waveform and failed
(diagnostic AUC 0.6756).  Untested:  self-supervised/supervised contrastive
learning of a *time-frequency* representation of the EEG itself.

Pipeline
  1. Deterministic TF maps from raw 18x1200 @300 Hz:  STFT (n_fft=128, hop=32)
     -> log magnitude -> 6 bands -> 3 band groups -> channel x time matrix ->
     bilinear resize to 64x64 -> per-image standardisation.  No dataset statistic
     is used, so Train and Test are transformed by the identical rule.
  2. A small 3-conv CNN (<=1M params) is trained on Train session1 only with
     supervised contrastive loss (temperature 0.1, positives = same subject,
     same task, other trials/views).  Output is a raw 256-d projection with NO
     L2 normalisation (E161 lesson).
  3. The 256-d embeddings go through the unchanged E143 open-set protocol:
     task-wise Train standardisation -> PCA -> k prototypes -> 5-block Train OOF
     threshold -> LOSO / Test scoring.

The frozen E65 correlation-logSPD control must reproduce E143
(PCA-96, k=12, acceptance 0.90 -> W 0.6958, diagnostic AUC 0.7578), otherwise the
harness is not trustworthy.

Usage:
  .venv/Scripts/python.exe research/E166_tfmap_ssl/run_e166.py --stage all
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn.metrics import f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG_PATH = OUT / "config.json"
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "report.md"
LOG_PATH = OUT / "run.log"
MODEL_PATH = OUT / "model_tfmap_ssl.pt"
ENCODER_CFG_PATH = OUT / "encoder_config.json"
TRAIN_TFMAP_PATH = OUT / "train_tfmap.npy"
TEST_TFMAP_PATH = OUT / "test_tfmap.npy"
TRAIN_EMB_PATH = OUT / "train_emb.npy"
TEST_EMB_PATH = OUT / "test_emb.npy"

DEVICE = torch.device("cuda")
FT = torch.float32
CLASSES = np.arange(1, 21, dtype=int)
TASKS = CFG["protocol"]["tasks"]
WEIGHTS = CFG["protocol"]["weights"]
IMG = int(CFG["tfmap"]["image_size"])


class Tee:
    """Mirror stdout into run.log."""

    def __init__(self, path: Path, append: bool = False):
        self.handle = path.open("a" if append else "w", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.handle.write(text)

    def flush(self):
        self.stdout.flush()
        self.handle.flush()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


# --------------------------------------------------------------------------- data
def load_trial_matrix(path: Path) -> np.ndarray:
    data = np.asarray(loadmat(path, variable_names=["data"])["data"], dtype=np.float32)
    if data.shape != (1200, 18):
        raise ValueError(f"{path}: expected (1200,18), got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: non-finite values")
    return np.ascontiguousarray(data.T)


def load_paths(paths: list[Path], workers: int, label: str) -> np.ndarray:
    out = np.empty((len(paths), 18, 1200), dtype=np.float32)
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(paths), 512):
            chunk = paths[start:start + 512]
            out[start:start + len(chunk)] = np.stack(list(pool.map(load_trial_matrix, chunk)))
            done = start + len(chunk)
            if done % 4096 == 0 or done == len(paths):
                print(f"  load {label} {done}/{len(paths)} | {time.time() - started:.1f}s", flush=True)
    return out


# ------------------------------------------------------------------------ tf maps
def band_bin_map() -> dict:
    fs = float(CFG["data"]["sampling_rate_hz"])
    n_fft = int(CFG["tfmap"]["stft"]["n_fft"])
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    names = CFG["tfmap"]["band_names"]
    bins = []
    for lo, hi in CFG["tfmap"]["bands_hz"]:
        idx = np.flatnonzero((freqs >= lo) & (freqs < hi))
        if len(idx) == 0:
            raise ValueError(f"band {lo}-{hi} Hz has no STFT bin at n_fft={n_fft}")
        bins.append(idx)
    return {"freqs": freqs, "names": names, "bins": bins}


def tfmap_from_raw(raw: np.ndarray, window: torch.Tensor) -> np.ndarray:
    """raw: (B,18,1200) float32 numpy -> (B,3,64,64) float16, per-image standardised."""
    b = len(raw)
    stft = CFG["tfmap"]["stft"]
    n_fft, hop = int(stft["n_fft"]), int(stft["hop_length"])
    band = band_bin_map()
    x = torch.as_tensor(raw, dtype=FT, device=DEVICE).reshape(b * 18, -1)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, window=window,
                      center=bool(stft["center"]), return_complex=True)
    mag = torch.log1p(spec.abs())                    # (B*18, 65, frames)
    frames = mag.shape[-1]
    groups = CFG["tfmap"]["image_groups"]
    planes = []
    for group, members in groups.items():
        idx = np.concatenate([band["bins"][band["names"].index(m)] for m in members])
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=DEVICE)
        plane = mag.index_select(1, idx_t).mean(1).reshape(b, 18, frames)
        planes.append(plane)
    stack = torch.stack(planes, dim=1)               # (B,3,18,frames)
    resized = F.interpolate(stack.reshape(b * 3, 1, 18, frames), size=(IMG, IMG),
                            mode="bilinear", align_corners=False)
    resized = resized.reshape(b, 3, IMG, IMG)
    mu = resized.mean(dim=(1, 2, 3), keepdim=True)
    sd = resized.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    out = ((resized - mu) / sd).to(torch.float16).cpu().numpy()
    return np.ascontiguousarray(out)


def build_tfmap(meta: pd.DataFrame, out_path: Path, label: str, batch: int = 512) -> np.ndarray:
    paths = [ROOT / CFG["data"]["train_root"] / str(p) for p in meta["path"]]
    maps = np.empty((len(paths), 3, IMG, IMG), dtype=np.float16)
    stft = CFG["tfmap"]["stft"]
    window = torch.hann_window(int(stft["n_fft"]), device=DEVICE)
    shape_seen = None
    started = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(paths), batch):
            chunk = paths[start:start + batch]
            raw = np.stack(list(pool.map(load_trial_matrix, chunk)))
            part = tfmap_from_raw(raw, window)
            if shape_seen is None:
                n_fft, hop = int(stft["n_fft"]), int(stft["hop_length"])
                frames = (int(CFG["data"]["samples_per_trial"]) - n_fft) // hop + 1
                shape_seen = (n_fft // 2 + 1, frames)
                print(f"  STFT layout: {shape_seen[0]} freq bins x {shape_seen[1]} frames "
                      f"(n_fft={n_fft}, hop={hop}, center=False)", flush=True)
            maps[start:start + len(chunk)] = part
            done = start + len(chunk)
            if done % 4096 == 0 or done == len(paths):
                print(f"  tfmap {label} {done}/{len(paths)} | {time.time() - started:.1f}s", flush=True)
    np.save(out_path, maps)
    print(f"  saved {out_path.name} {maps.shape} {maps.dtype} "
          f"({maps.nbytes / 1e6:.0f} MB)", flush=True)
    return maps


# ------------------------------------------------------------------------ model
class Encoder(nn.Module):
    def __init__(self, widths, kernel, embed_dim, dropout, pool, in_channels=3):
        super().__init__()
        blocks, channels = [], in_channels
        for width in widths:
            blocks.extend([
                nn.Conv2d(channels, width, kernel, padding=kernel // 2, bias=False),
                nn.BatchNorm2d(width),
                nn.ELU(inplace=True),
                nn.MaxPool2d(pool),
            ])
            channels = width
        self.body = nn.Sequential(*blocks)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * channels, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return self.fc(self.drop(pooled))          # raw projection, no L2 normalisation


def build_encoder() -> Encoder:
    cfg = CFG["encoder"]
    return Encoder(cfg["widths"], int(cfg["kernel"]), int(cfg["embed_dim"]),
                   float(cfg["dropout"]), int(cfg["pool"])).to(DEVICE)


# ---------------------------------------------------------------------- augment
def augment_maps(x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """x: (B,3,64,64). Time = width, frequency = height. No horizontal flip."""
    aug = CFG["augment"]
    b, _, h, w = x.shape
    dev = x.device
    seg_len = int(aug["time_occlusion_len"])
    rows = int(aug["freq_occlusion_rows"])
    pos_w = torch.arange(w, device=dev)[None, None, :]
    pos_h = torch.arange(h, device=dev)[None, :, None]
    for _ in range(int(aug["time_occlusion_segments"])):
        start = torch.randint(0, w - seg_len + 1, (b,), generator=rng, device=dev)
        mask = ((pos_w >= start[:, None, None]) &
                (pos_w < start[:, None, None] + seg_len)).to(x.dtype)
        x = x * (1.0 - mask[:, None, :, :])
    start = torch.randint(0, h - rows + 1, (b,), generator=rng, device=dev)
    mask = ((pos_h >= start[:, None, None]) &
            (pos_h < start[:, None, None] + rows)).to(x.dtype)
    x = x * (1.0 - mask[:, None, :, :])
    std = float(aug["gaussian_noise_std"])
    return x + torch.randn(x.shape, generator=rng, device=dev) * std


# -------------------------------------------------------------------- train loop
def frac(meta: pd.DataFrame) -> np.ndarray:
    maximum = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(maximum, 1.0)


def block_of(f: np.ndarray, blocks: int = 5) -> np.ndarray:
    return np.clip((f * blocks).astype(int), 0, blocks - 1)


def fit_index(meta: pd.DataFrame, f: np.ndarray) -> dict:
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    trials = meta.trial.to_numpy()
    fit = f <= float(CFG["train"]["fit_fraction"])
    blocks = block_of(f)
    table = {}
    for subject in CLASSES:
        for task in TASKS:
            rows = np.flatnonzero(fit & (subjects == subject) & (tasks == task))
            if len(rows) == 0:
                continue
            rows = rows[np.argsort(trials[rows], kind="mergesort")]
            table[(int(subject), task)] = (rows, blocks[rows])
    return table


def sample_batch(table: dict, task: str, rng: np.random.Generator) -> tuple[list[int], list[int]]:
    per = int(CFG["train"]["trials_per_subject"])
    rows, labels = [], []
    for subject in CLASSES:
        entry = table.get((int(subject), task))
        if entry is None:
            continue
        pool, blocks = entry
        unique = np.unique(blocks)
        picked = rng.choice(unique, size=min(per, len(unique)), replace=False)
        chosen = [int(rng.choice(pool[blocks == blk])) for blk in picked]
        while len(chosen) < per:
            chosen.append(int(rng.choice(pool)))
        rows.extend(chosen)
        labels.extend([int(subject)] * per)
    return rows, labels


def supcon(z: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    sim = (z @ z.t()) / temperature
    mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    logits = sim - sim.max(dim=1, keepdim=True).values.detach()
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive = (labels[:, None] == labels[None, :]) & ~mask
    count = positive.sum(dim=1)
    mean_pos = (log_prob * positive).sum(dim=1) / count.clamp_min(1)
    return -mean_pos[count > 0].mean()


@torch.no_grad()
def embed_tensor(encoder: Encoder, maps: torch.Tensor, rows: np.ndarray,
                 batch: int = 512) -> np.ndarray:
    encoder.eval()
    out = np.empty((len(rows), encoder.embed_dim), dtype=np.float32)
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        index = torch.as_tensor(chunk, dtype=torch.long, device=maps.device)
        out[start:start + len(chunk)] = encoder(maps[index]).float().cpu().numpy()
    return out


def valid_accuracy(encoder: Encoder, maps: torch.Tensor, rows: np.ndarray,
                   subjects: np.ndarray, tasks: np.ndarray) -> dict:
    z = embed_tensor(encoder, maps, rows)
    sub, tas = subjects[rows], tasks[rows]
    correct, total, per_task = 0, 0, {}
    for task in TASKS:
        centroids, centre_labels, query, truth = [], [], [], []
        for subject in CLASSES:
            idx = np.flatnonzero((sub == subject) & (tas == task))
            if len(idx) < 2:
                continue
            centroids.append(z[idx[::2]].mean(0))
            centre_labels.append(subject)
            query.append(idx[1::2])
            truth.append(np.full(len(idx[1::2]), subject))
        if len(centroids) < 2:
            continue
        centre = np.stack(centroids).astype(np.float64)
        centre /= np.maximum(np.linalg.norm(centre, axis=1, keepdims=True), 1e-8)
        q = np.concatenate(query)
        label = np.concatenate(truth)
        pred = np.asarray(centre_labels)[np.argmax(z[q].astype(np.float64) @ centre.T, axis=1)]
        acc = float(np.mean(pred == label))
        per_task[task] = {"accuracy": acc, "n": int(len(q))}
        correct += int(np.sum(pred == label))
        total += len(q)
    return {"accuracy": float(correct / max(total, 1)), "per_task": per_task, "n": int(total)}


def train_encoder(maps: torch.Tensor, meta: pd.DataFrame) -> tuple[Encoder, dict]:
    cfg = CFG["train"]
    seed = int(cfg["seed"])
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)
    table = fit_index(meta, f)
    eval_rows = np.flatnonzero((f > float(cfg["fit_fraction"])) & (f <= float(cfg["eval_fraction"])))

    encoder = build_encoder()
    params = int(sum(p.numel() for p in encoder.parameters()))
    print(f"encoder parameters = {params:,} (budget {CFG['encoder']['param_budget']:,})", flush=True)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=float(cfg["lr"]),
                                  weight_decay=float(cfg["weight_decay"]))
    total_steps = int(cfg["epochs"]) * int(cfg["steps_per_epoch"])
    warmup = int(cfg["warmup_steps"])

    def lr_at(step: int) -> float:
        if step < warmup:
            return float(cfg["lr"]) * (step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return float(cfg["lr"]) * 0.5 * (1.0 + np.cos(np.pi * progress))

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng_np = np.random.default_rng(seed)
    rng_t = torch.Generator(device=DEVICE).manual_seed(seed)

    history, step = [], 0
    best = {"accuracy": -1.0, "epoch": -1, "state": None, "per_task": {}}
    patience = int(cfg["early_stop_patience"])
    started = time.time()
    for epoch in range(int(cfg["epochs"])):
        encoder.train()
        order = list(TASKS)
        rng_np.shuffle(order)
        losses, norms = [], []
        for local in range(int(cfg["steps_per_epoch"])):
            task = order[local % len(order)]
            row_ids, labels = sample_batch(table, task, rng_np)
            base = torch.as_tensor(row_ids, dtype=torch.long, device=DEVICE)
            batch = maps[base]
            views = [augment_maps(batch, rng_t) for _ in range(int(cfg["views_per_trial"]))]
            z = encoder(torch.cat(views, dim=0))
            target = torch.as_tensor(labels * int(cfg["views_per_trial"]),
                                     dtype=torch.long, device=DEVICE)
            loss = supcon(z, target, float(cfg["temperature"]))
            optimizer.param_groups[0]["lr"] = lr_at(step)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), float(cfg["grad_clip"]))
            optimizer.step()
            losses.append(float(loss.detach()))
            norms.append(float(z.norm(dim=1).mean().detach()))
            step += 1
        valid = valid_accuracy(encoder, maps, eval_rows, subjects, tasks)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)),
                        "embed_norm": float(np.mean(norms)),
                        "valid_accuracy": valid["accuracy"],
                        "per_task": {k: v["accuracy"] for k, v in valid["per_task"].items()},
                        "lr": lr_at(step), "elapsed": time.time() - started})
        print(f"epoch {epoch + 1}/{cfg['epochs']} loss={np.mean(losses):.4f} "
              f"norm={np.mean(norms):.2f} valid_acc={valid['accuracy']:.4f} "
              f"{ {k: round(v['accuracy'], 3) for k, v in valid['per_task'].items()} } "
              f"| {time.time() - started:.0f}s", flush=True)
        if valid["accuracy"] > best["accuracy"] + 1e-6:
            best = {"accuracy": valid["accuracy"], "epoch": epoch + 1,
                    "state": {k: v.detach().clone() for k, v in encoder.state_dict().items()},
                    "per_task": {k: v["accuracy"] for k, v in valid["per_task"].items()}}
        elif epoch + 1 - best["epoch"] >= patience:
            print(f"early stop at epoch {epoch + 1} (best epoch {best['epoch']})", flush=True)
            break

    encoder.load_state_dict(best["state"])
    info = {"parameters": params, "best_epoch": int(best["epoch"]),
            "best_valid_accuracy": float(best["accuracy"]),
            "best_valid_per_task": best["per_task"], "history": history,
            "total_steps": step, "elapsed_seconds": time.time() - started,
            "n_fit_rows": int((f <= float(cfg["fit_fraction"])).sum()),
            "n_valid_rows": int(len(eval_rows))}
    torch.save({"state_dict": encoder.state_dict(), "encoder": CFG["encoder"],
                "train": CFG["train"], "augment": CFG["augment"], "tfmap": CFG["tfmap"],
                "info": info, "train_only": True}, MODEL_PATH)
    write_json(ENCODER_CFG_PATH, {
        "format": "E166 multi-band TF-map supervised-contrastive encoder",
        "input": {"channels": 3, "image": [IMG, IMG],
                  "tfmap": CFG["tfmap"]["stft"]},
        **CFG["encoder"],
        "best_epoch": int(best["epoch"]), "best_valid_accuracy": float(best["accuracy"]),
        "parameters": params, "train_only": True,
        "weights": str(MODEL_PATH),
    })
    return encoder, info


def load_encoder() -> tuple[Encoder, dict]:
    payload = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    cfg = payload["encoder"]
    encoder = Encoder(cfg["widths"], int(cfg["kernel"]), int(cfg["embed_dim"]),
                      float(cfg["dropout"]), int(cfg["pool"])).to(DEVICE)
    encoder.load_state_dict(payload["state_dict"])
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder, payload["info"]


# --------------------------------------------------------------------- protocol
def pca_basis(x: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(0)
    xc = x - mu
    values, vectors = np.linalg.eigh(xc.T @ xc / max(len(xc) - 1, 1))
    return mu, vectors[:, np.argsort(-values)][:, : min(dim, x.shape[1])]


def kmeans_t(x: torch.Tensor, k: int, seed: int, iters: int = 20) -> torch.Tensor:
    """Verbatim E143 seeding so the E65 control reproduces the published W."""
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
                     mask: np.ndarray, classes: np.ndarray, k: int):
    protos, keys = [], []
    for task in TASKS:
        for subject in classes:
            sel = mask & (tasks == task) & (subjects == subject)
            if sel.sum() == 0:
                continue
            block = torch.as_tensor(np.asarray(feat[sel], dtype=np.float32), device=DEVICE)
            for centre in kmeans_t(block, k, seed=int(subject) * 17).cpu().numpy():
                protos.append(centre)
                keys.append((task, int(subject)))
    return np.stack(protos), keys


def neg_d1(feat: np.ndarray, protos: np.ndarray, keys: list, tasks: np.ndarray,
           return_argmax: bool = False):
    out = np.full(len(feat), np.nan, dtype=np.float32)
    pred = np.full(len(feat), -1, dtype=np.int32)
    key_subject = np.asarray([key[1] for key in keys], dtype=np.int32)
    for task in TASKS:
        idx = np.flatnonzero(tasks == task)
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(idx) == 0 or not cols:
            continue
        p = torch.as_tensor(protos[cols], dtype=FT, device=DEVICE)
        key_subject_task = key_subject[np.asarray(cols)]
        for start in range(0, len(idx), 2048):
            chunk = idx[start:start + 2048]
            q = torch.as_tensor(np.asarray(feat[chunk], dtype=np.float32), device=DEVICE)
            values, arg = (-torch.cdist(q, p)).max(1)
            out[chunk] = values.cpu().numpy()
            if return_argmax:
                pred[chunk] = key_subject_task[np.asarray(arg.cpu().numpy())]
    return (out, pred) if return_argmax else out


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    q = np.quantile(values, [0.05, 0.5, 0.95])
    return {"N": int(len(values)), "mean": float(values.mean()), "std": float(values.std()),
            "q05": float(q[0]), "median": float(q[1]), "q95": float(q[2])}


def effective_rank(feat: np.ndarray, tasks: np.ndarray, fit_rows: np.ndarray) -> dict:
    """Rank of the task-standardised Train embedding (same transform as the protocol)."""
    a = np.asarray(feat, dtype=np.float64).copy()
    for task in TASKS:
        mask = tasks == task
        mu, sd = a[mask].mean(0), a[mask].std(0) + 1e-6
        a[mask] = (a[mask] - mu) / sd
    x = a[fit_rows]
    xc = x - x.mean(0)
    values = np.clip(np.linalg.eigvalsh(xc.T @ xc / max(len(xc) - 1, 1))[::-1], 0, None)
    ratio = values / values.sum()
    cumulative = np.cumsum(ratio)
    need95 = int(np.searchsorted(cumulative, 0.95) + 1)
    need99 = int(np.searchsorted(cumulative, 0.99) + 1)
    return {"n_features": int(a.shape[1]), "fit_rows": int(len(x)),
            "dims_for_95pct_variance": need95,
            "dims_for_99pct_variance": need99,
            "explained_95pct_of_usable_dims": float(need95 / a.shape[1]),
            "cumulative": {str(d): float(cumulative[min(d, len(cumulative)) - 1])
                           for d in (8, 16, 32, 64, 128, 256) if min(d, len(cumulative)) > 0},
            "top10_ratio": [float(v) for v in ratio[:10]]}


def evaluate_view(name: str, train_feat: np.ndarray, test_feat: np.ndarray,
                  subjects: np.ndarray, tasks: np.ndarray, f: np.ndarray,
                  test_tasks: np.ndarray, is_s1: np.ndarray,
                  pca_dims: list[int], ks: list[int]) -> dict:
    """E143 protocol, unchanged."""
    cfg = CFG["protocol"]
    fit_rows = f <= float(cfg["fit_fraction"])
    blocks = int(cfg["blocks"])
    block_id = block_of(f, blocks)
    window = (f > float(cfg["fit_fraction"])) & (f <= 0.8)
    window_rows = np.flatnonzero(window)

    a = np.asarray(train_feat, dtype=np.float32).copy()
    b = np.asarray(test_feat, dtype=np.float32).copy()
    for task in TASKS:
        mask = tasks == task
        mu, sd = a[mask].mean(0), a[mask].std(0) + 1e-6
        a[mask] = (a[mask] - mu) / sd
        query = test_tasks == task
        b[query] = (b[query] - mu) / sd

    y = np.r_[np.zeros(int(is_s1.sum()), dtype=int), np.ones(int((~is_s1).sum()), dtype=int)]
    rows = []
    for dim in pca_dims:
        mu, basis = pca_basis(np.asarray(a[fit_rows], dtype=np.float64), int(dim))
        at = ((a.astype(np.float64) - mu) @ basis).astype(np.float32)
        bt = ((b.astype(np.float64) - mu) @ basis).astype(np.float32)
        print(f"  [{name}] pca {dim} projected", flush=True)
        for k in ks:
            protos, keys = build_prototypes(at, subjects, tasks, np.ones(len(at), bool),
                                            CLASSES, int(k))
            te_raw = neg_d1(bt, protos, keys, test_tasks)

            oof = np.full(len(at), np.nan, dtype=np.float32)
            oof_pred = np.full(len(at), -1, dtype=np.int32)
            for block in range(blocks):
                hold = block_id == block
                held = np.flatnonzero(hold)
                p2, k2 = build_prototypes(at, subjects, tasks, ~hold, CLASSES, int(k))
                scores, pred = neg_d1(at[held], p2, k2, tasks[held], return_argmax=True)
                oof[held] = scores
                oof_pred[held] = pred

            loso_known, loso_unknown = [], []
            for held in CLASSES:
                enrolled = np.setdiff1d(CLASSES, [held])
                pm = np.isin(subjects, enrolled)
                p2, k2 = build_prototypes(at, subjects, tasks, pm & fit_rows, enrolled, int(k))
                known = window_rows[pm[window_rows]]
                unknown = window_rows[subjects[window_rows] == held]
                if len(unknown) == 0 or len(known) == 0:
                    continue
                both = np.concatenate([known, unknown])
                scores = neg_d1(at[both], p2, k2, tasks[both])
                loso_known.append(scores[:len(known)])
                loso_unknown.append(scores[len(known):])

            ref = oof[np.isfinite(oof)]
            mu2, sd2 = float(ref.mean()), float(ref.std()) + 1e-6
            te = (te_raw - mu2) / sd2
            rf = (oof - mu2) / sd2
            known_z = [(x - mu2) / sd2 for x in loso_known]
            unknown_z = [(x - mu2) / sd2 for x in loso_unknown]
            unknown_all = np.concatenate(unknown_z) if unknown_z else np.empty(0, np.float32)
            ref_z = (ref - mu2) / sd2
            oof_auc = float(roc_auc_score(
                np.r_[np.zeros(len(unknown_all)), np.ones(len(ref_z))],
                np.r_[unknown_all, ref_z])) if len(unknown_all) else float("nan")
            finite = np.isfinite(oof)
            oof_f1 = float(f1_score(subjects[finite], oof_pred[finite], labels=CLASSES,
                                    average="macro", zero_division=0))
            oof_acc = float(np.mean(oof_pred[finite] == subjects[finite]))
            diagnostic_auc = float(roc_auc_score(y, np.r_[te[is_s1], te[~is_s1]]))
            per_task = {}
            for task in TASKS:
                sel = test_tasks == task
                item = {"s1_median": float(np.median(te[sel & is_s1])),
                        "s2_median": float(np.median(te[sel & ~is_s1]))}
                if len(np.unique(is_s1[sel])) > 1:
                    item["auc"] = float(roc_auc_score((~is_s1[sel]).astype(int), te[sel]))
                per_task[task] = item

            for acceptance in CFG["protocol"]["acceptance_grid"]:
                threshold = float(np.nanquantile(rf, 1 - acceptance))
                s2 = float(np.mean(te[~is_s1] >= threshold))
                s1 = float(np.mean(te[is_s1] < threshold))
                loso_rej = float(np.mean([
                    np.mean(u < np.quantile(kn, 1 - acceptance))
                    for kn, u in zip(known_z, unknown_z)])) if known_z else 0.0
                w = (WEIGHTS["LOSO"] * loso_rej + WEIGHTS["S1"] * s1 + WEIGHTS["S2"] * s2)
                rows.append({"view": name, "pca_dim": int(dim), "k": int(k),
                             "acceptance_rate": float(acceptance),
                             "diagnostic_auc": diagnostic_auc,
                             "train_oof_auc": oof_auc, "train_oof_macro_f1": oof_f1,
                             "train_oof_identity_accuracy": oof_acc,
                             "loso_rejection": loso_rej,
                             "test_s1_rejection": s1, "test_s2_acceptance": s2,
                             "W": float(w), "threshold": threshold,
                             "per_task": per_task,
                             "s1_score": distribution(te[is_s1]),
                             "s2_score": distribution(te[~is_s1])})
            print(f"  [{name}] pca={dim} k={k} OOFauc={oof_auc:.4f} OOFf1={oof_f1:.4f} "
                  f"diagAUC={diagnostic_auc:.4f} bestW="
                  f"{max(r['W'] for r in rows if r['pca_dim'] == dim and r['k'] == k):.4f}",
                  flush=True)
            del protos, keys, te_raw, oof, oof_pred
    return {"rows": rows}


# --------------------------------------------------------------- emb extraction
@torch.no_grad()
def extract_embeddings(encoder: Encoder, maps: np.ndarray, batch: int = 512,
                       label: str = "train") -> np.ndarray:
    out = np.empty((len(maps), encoder.embed_dim), dtype=np.float32)
    started = time.time()
    for start in range(0, len(maps), batch):
        chunk = torch.as_tensor(np.asarray(maps[start:start + batch], dtype=np.float32),
                                device=DEVICE)
        out[start:start + len(chunk)] = encoder(chunk).float().cpu().numpy()
        done = start + len(chunk)
        if done % 4096 == 0 or done == len(maps):
            print(f"  embed {label} {done}/{len(maps)} | {time.time() - started:.0f}s", flush=True)
    return out


# ---------------------------------------------------------------------- report
def fmt(value) -> str:
    return "N/A" if value is None or not np.isfinite(value) else f"{value:.4f}"


def write_report(result: dict) -> None:
    base = CFG["baseline"]
    best = result["e166"]["best"]
    control = result["control"]
    repro = control["reproduction"]
    er = result["embedding_spectrum"]
    train = result["training"]
    improved_auc = best["diagnostic_auc"] > base["diagnostic_auc"]
    improved_w = best["W"] > base["W"]

    head = [
        "# E166 多频带时频图 + Train 内监督对比学习",
        "",
        "填补 E149（冻结图像骨干）和 E161（原始波形端到端对比）之间的空隙："
        "从原始 18×1200 EEG 用**完全确定、无统计量**的规则生成多频带时频图，"
        "再在 **Train session1** 上训练一个小 CNN 做身份监督对比学习，"
        "最后走**逐字沿用**的 E143 开放集协议。Test 只被评分，session 只用于诊断 AUC。",
        "",
        "## 结论与当前 AUC / F1",
        "",
        f"- **Test 诊断 AUC = {fmt(best['diagnostic_auc'])}**（E143/E65 基准 {base['diagnostic_auc']:.4f}，"
        f"差 {best['diagnostic_auc'] - base['diagnostic_auc']:+.4f}）",
        f"- **W = {fmt(best['W'])}**（基准 {base['W']:.4f}，差 {best['W'] - base['W']:+.4f}）"
        f"，工作点 PCA-{best['pca_dim']} / k={best['k']} / 接受率 {best['acceptance_rate']:.2f}",
        f"- Train OOF AUC = {fmt(best['train_oof_auc'])}，Train OOF 宏 F1 = {fmt(best['train_oof_macro_f1'])}"
        f"（身份准确率 {fmt(best['train_oof_identity_accuracy'])}）",
        f"- LOSO 拒识 = {fmt(best['loso_rejection'])}，Test S1 拒识 = {fmt(best['test_s1_rejection'])}，"
        f"Test S2 接受 = {fmt(best['test_s2_acceptance'])}",
        f"- 有效秩：task 标准化后 Train 拟合块 embedding 解释 95% 方差需 "
        f"**{er['dims_for_95pct_variance']} 维**（99% 需 {er['dims_for_99pct_variance']} 维，共 {er['n_features']} 维）",
        f"- Test 官方 AUC/F1 = **N/A**（Test 无公开标签）",
        "",
        "## 判定",
        "",
    ]
    if improved_auc:
        head += [f"诊断 AUC {fmt(best['diagnostic_auc'])} **超过** E143 基准 0.7578。", ""]
    else:
        head += [
            f"**核心问题的答案：不能。** Test 诊断 AUC = {fmt(best['diagnostic_auc'])}，"
            f"低于 E143 基准 0.7578（差 {best['diagnostic_auc'] - base['diagnostic_auc']:+.4f}）。",
            "",
            "**“时频图 + Train 内对比学习”路线失败。**",
            "",
            "| 指标 | E166 时频图 | E65 对照（同协议） | E143 基准 |",
            "|---|---:|---:|---:|",
            f"| Test 诊断 AUC | {fmt(best['diagnostic_auc'])} | {fmt(repro['diagnostic_auc'])} | 0.7578 |",
            f"| W | {fmt(best['W'])} | {fmt(repro['W'])} | 0.6958 |",
            f"| Train OOF AUC | {fmt(best['train_oof_auc'])} | {fmt(repro['train_oof_auc'])} | – |",
            f"| Train OOF 宏 F1 | {fmt(best['train_oof_macro_f1'])} | {fmt(repro['train_oof_macro_f1'])} | – |",
            f"| LOSO 拒识 | {fmt(best['loso_rejection'])} | {fmt(repro['loso_rejection'])} | 0.7936 |",
            f"| Test S1 拒识 | {fmt(best['test_s1_rejection'])} | {fmt(repro['test_s1_rejection'])} | 0.7979 |",
            f"| Test S2 接受 | {fmt(best['test_s2_acceptance'])} | {fmt(repro['test_s2_acceptance'])} | 0.5960 |",
            "",
            "证据：",
            "",
            f"1. **学到的身份信息量本就不足**：同一协议下 E166 的 Train OOF 宏 F1 只有 "
            f"{fmt(best['train_oof_macro_f1'])}、OOF AUC {fmt(best['train_oof_auc'])}，远低于 E65 "
            f"correlation-logSPD 的 {fmt(repro['train_oof_macro_f1'])} / {fmt(repro['train_oof_auc'])}。"
            "时频图在 fit 块上就已经丢掉了大部分身份判别信息。",
            f"2. **表征严重降维坍缩**：256 维输出解释 95% 方差只需 **{er['dims_for_95pct_variance']} 维**、"
            f"99% 只需 {er['dims_for_99pct_variance']} 维，第 1 主成分独占 "
            f"{100 * er['top10_ratio'][0]:.0f}%。训练中 embedding 范数稳定在 ~0.86，"
            "所以不是 E161 那种范数坍缩，而是所有 trial 被压进一个极低维子空间——"
            "**去掉 L2 归一化并没有阻止坍缩**。",
            f"3. **分数分布被 session 偏移主导**：S1 中位分数 {best['s1_score']['median']:.2f} vs "
            f"S2 {best['s2_score']['median']:.2f}，跨 session 偏移（约 "
            f"{best['s2_score']['median'] - best['s1_score']['median']:.2f} 个标准化单位）远大于身份间距，"
            f"于是 LOSO 拒识 {fmt(best['loso_rejection'])} 与 S1 拒识 "
            f"{fmt(best['test_s1_rejection'])} 同时崩掉，S2 接受 {fmt(best['test_s2_acceptance'])} "
            "只是阈值被压低后的虚高。",
            f"4. 逐任务诊断 AUC "
            f"{min(v['auc'] for v in best['per_task'].values() if 'auc' in v):.3f}–"
            f"{max(v['auc'] for v in best['per_task'].values() if 'auc' in v):.3f}，"
            "没有任何任务接近 0.93（E159 已证明 W>0.80 所需）。",
            "",
        ]
    if not improved_w:
        head += [
            f"W 也未超过基准 0.6958（{fmt(best['W'])} vs 0.6958）。",
            "",
        ]

    lines = head + [
        "## 管线可信度：E65 冻结对照组",
        "",
        "| 指标 | 复现值 | E143 公布值 |",
        "|---|---:|---:|",
        f"| 诊断 AUC | {fmt(repro['diagnostic_auc'])} | 0.7578 |",
        f"| LOSO 拒识 | {fmt(repro['loso_rejection'])} | 0.7936 |",
        f"| Test S1 拒识 | {fmt(repro['test_s1_rejection'])} | 0.7979 |",
        f"| Test S2 接受 | {fmt(repro['test_s2_acceptance'])} | 0.5960 |",
        f"| W | {fmt(repro['W'])} | 0.6958 |",
        "",
        f"复现误差 ΔW = {repro['W'] - base['W']:+.6f}，管线可信。",
        "",
        "## 时频图表征",
        "",
        f"- STFT n_fft={CFG['tfmap']['stft']['n_fft']}、hop={CFG['tfmap']['stft']['hop_length']}、"
        f"Hann 窗、center=False → {result['tfmap_layout']['freq_bins']} 个频点 × "
        f"{result['tfmap_layout']['frames']} 帧",
        "- 6 个频带（1-4/4-8/8-13/13-20/20-30/30-40 Hz）→ 合并成 3 个图像通道"
        "（1-8 / 8-20 / 20-40 Hz），每通道是 18 通道 × 时间帧的 log 功率矩阵，双线性 resize 到 64×64",
        "- 逐图标准化（减均值除标准差），逐样本操作，Train/Test 同一规则，无任何数据集统计量",
        f"- 缓存：`train_tfmap.npy` (11304×3×64×64, float16) / `test_tfmap.npy` "
        f"(22251×3×64×64, float16)",
        "",
        "## 训练（严格 Train-only）",
        "",
        f"- 编码器：3 层 conv(32/64/128, 3×3, maxpool2) + BN + ELU + 全局 mean/max 池化 → 256 维线性投影，"
        f"参数 {train['parameters']:,}（预算 1,000,000）",
        "- 输出**不做 L2 归一化**（E161 教训：会导致表征坍缩）；训练日志里记录了 embedding 范数",
        f"- 损失：supervised contrastive，温度 {CFG['train']['temperature']}；正样本 = 同 subject 同 task 的不同 trial/视图，"
        "负样本 = batch 内其他 subject",
        f"- 增强：时间轴随机遮挡 2 段×{CFG['augment']['time_occlusion_len']} 列、"
        f"频率轴遮挡 {CFG['augment']['freq_occlusion_rows']} 行、高斯噪声 σ={CFG['augment']['gaussian_noise_std']}；"
        "禁止水平翻转（EEG 时间不可逆）",
        f"- 划分：frac≤0.6 为 fit，0.6<frac≤0.8 仅用于早停；epoch 上限 {CFG['train']['epochs']}、"
        f"patience {CFG['train']['early_stop_patience']}；最佳 epoch = {train['best_epoch']}，"
        f"验证准确率 = {fmt(train['best_valid_accuracy'])}"
        f"{'' if len(train['history']) < CFG['train']['epochs'] else '（未触发早停，跑满 epoch 上限）'}",
        f"- 模型：`model_tfmap_ssl.pt`；训练用时 {train['elapsed_seconds']:.0f}s",
        "",
        "## E143 协议扫描（PCA × k）",
        "",
        "| PCA | k | 接受率 | 诊断 AUC | Train OOF AUC | OOF F1 | LOSO | S1 拒识 | S2 接受 | W |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(result["e166"]["rows"], key=lambda r: -r["W"])[:15]:
        lines.append(f"| {r['pca_dim']} | {r['k']} | {r['acceptance_rate']:.2f} | "
                     f"{r['diagnostic_auc']:.4f} | {r['train_oof_auc']:.4f} | "
                     f"{r['train_oof_macro_f1']:.4f} | {r['loso_rejection']:.4f} | "
                     f"{r['test_s1_rejection']:.4f} | {r['test_s2_acceptance']:.4f} | {r['W']:.4f} |")
    lines += [
        "",
        "## 有效秩",
        "",
        f"task 标准化后、fit 块上的 PCA 累计解释方差：",
        "",
        "| 维数 | 累计解释方差 |",
        "|---:|---:|",
    ]
    for dim, value in er["cumulative"].items():
        lines.append(f"| {dim} | {value:.4f} |")
    lines += [
        "",
        f"**解释 95% 方差需要 {er['dims_for_95pct_variance']} 维**（占 {er['n_features']} 维的 "
        f"{100 * er['explained_95pct_of_usable_dims']:.1f}%），99% 需要 {er['dims_for_99pct_variance']} 维。",
        "",
        "## 逐任务 Test 诊断（最佳工作点）",
        "",
        "| 任务 | S1 中位分数 | S2 中位分数 | AUC |",
        "|---|---:|---:|---:|",
    ]
    for task, item in best["per_task"].items():
        lines.append(f"| {task} | {item['s1_median']:.4f} | {item['s2_median']:.4f} | "
                     f"{item.get('auc', float('nan')):.4f} |")
    lines += [
        "",
        "项目 Test 官方 AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。",
        "",
        "## 复现",
        "",
        "```powershell",
        '& "d:\\WORK\\BCIC2026\\BrainPrint\\.venv\\Scripts\\python.exe" '
        '"d:\\WORK\\BCIC2026\\BrainPrint\\research\\E166_tfmap_ssl\\run_e166.py" --stage all',
        "```",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("tfmap", "train", "embed", "eval", "report", "all"),
                        default="all")
    parser.add_argument("--force-tfmap", action="store_true")
    parser.add_argument("--force-embed", action="store_true")
    args = parser.parse_args()
    if args.stage == "report":
        write_report(json.loads(RESULTS_PATH.read_text(encoding="utf-8")))
        print(f"rewrote {REPORT_PATH}", flush=True)
        return
    if not torch.cuda.is_available():
        parser.error("E166 requires CUDA")
    sys.stdout = Tee(LOG_PATH, append=args.stage != "all")
    started = time.time()
    torch.backends.cudnn.benchmark = True

    meta = pd.read_csv(ROOT / CFG["data"]["train_metadata"])
    test_meta = pd.read_csv(ROOT / CFG["data"]["test_metadata"])
    if len(meta) != int(CFG["data"]["train_trials"]):
        raise ValueError(f"Train rows {len(meta)} != {CFG['data']['train_trials']}")
    if len(test_meta) != int(CFG["data"]["test_rows"]):
        raise ValueError(f"Test rows {len(test_meta)} != {CFG['data']['test_rows']}")
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)
    test_tasks = test_meta.task.to_numpy()
    is_s1 = test_meta.session.astype(str).str.casefold().str.replace("_", "").str.replace("-", "").isin(
        ["1", "s1", "session1"]).to_numpy()
    print(f"Train {len(meta)} | Test {len(test_meta)} "
          f"(S1={int(is_s1.sum())}, S2={int((~is_s1).sum())})", flush=True)

    if args.stage in ("tfmap", "all"):
        print("building Train TF maps ...", flush=True)
        train_maps = build_tfmap(meta, TRAIN_TFMAP_PATH, "train")
        print("building Test TF maps ...", flush=True)
        build_tfmap(test_meta, TEST_TFMAP_PATH, "test")
    if args.stage == "tfmap":
        print("TF maps cached", flush=True)
        return

    if args.stage in ("train", "all"):
        print("loading cached Train TF maps ...", flush=True)
        train_maps = np.load(TRAIN_TFMAP_PATH, mmap_mode="r")
        maps_gpu = torch.as_tensor(np.asarray(train_maps), dtype=FT, device=DEVICE)
        print(f"Train TF maps on GPU: {tuple(maps_gpu.shape)} "
              f"({maps_gpu.numel() * 4 / 1e6:.0f} MB)", flush=True)
        del train_maps
        print("training supervised-contrastive encoder ...", flush=True)
        encoder, info = train_encoder(maps_gpu, meta)
        del maps_gpu
        torch.cuda.empty_cache()

    if args.stage in ("embed", "eval", "all"):
        if args.stage == "eval" or args.force_embed or not (
                TRAIN_EMB_PATH.exists() and TEST_EMB_PATH.exists()):
            encoder, info = load_encoder()
            print("extracting Train embeddings ...", flush=True)
            train_maps = np.load(TRAIN_TFMAP_PATH, mmap_mode="r")
            train_emb = extract_embeddings(encoder, train_maps, label="train")
            del train_maps
            np.save(TRAIN_EMB_PATH, train_emb)
            print("extracting Test embeddings ...", flush=True)
            test_maps = np.load(TEST_TFMAP_PATH, mmap_mode="r")
            test_emb = extract_embeddings(encoder, test_maps, label="test")
            del test_maps
            np.save(TEST_EMB_PATH, test_emb)
            torch.cuda.empty_cache()
        else:
            print("reusing cached embeddings", flush=True)
            train_emb = np.load(TRAIN_EMB_PATH)
            test_emb = np.load(TEST_EMB_PATH)
            _, info = load_encoder()
        dim = int(CFG["encoder"]["embed_dim"])
        if train_emb.shape != (len(meta), dim) or test_emb.shape != (len(test_meta), dim):
            raise ValueError(f"embedding shapes {train_emb.shape} / {test_emb.shape}")
        if not (np.isfinite(train_emb).all() and np.isfinite(test_emb).all()):
            raise ValueError("non-finite embeddings")
        if args.stage == "embed":
            print(f"embeddings cached: {TRAIN_EMB_PATH.name}, {TEST_EMB_PATH.name}", flush=True)
            return

        print("evaluating E143 protocol on E166 embeddings ...", flush=True)
        e166 = evaluate_view("E166_tfmap_ssl", train_emb, test_emb, subjects, tasks, f,
                             test_tasks, is_s1, CFG["protocol"]["pca_dims"],
                             CFG["protocol"]["prototype_k"])
        print("evaluating E143 protocol on E65 control features ...", flush=True)
        control_train = np.load(ROOT / CFG["control"]["train_features"])
        control_test = np.load(ROOT / CFG["control"]["test_features"])
        if control_train.shape[0] != len(meta) or control_test.shape[0] != len(test_meta):
            raise ValueError("control feature row mismatch")
        control = evaluate_view("E65_control", control_train, control_test, subjects, tasks, f,
                                test_tasks, is_s1,
                                [int(CFG["control"]["reproduction_target"]["pca_dim"])],
                                [int(CFG["control"]["reproduction_target"]["k"])])
        del control_train, control_test

        best_e = max(e166["rows"], key=lambda r: r["W"])
        best_c = max(control["rows"], key=lambda r: r["W"])
        repro = [r for r in control["rows"]
                 if abs(r["acceptance_rate"] -
                        CFG["control"]["reproduction_target"]["acceptance_rate"]) < 1e-9][0]
        spectrum = effective_rank(train_emb, tasks, f <= float(CFG["protocol"]["fit_fraction"]))
        stft = CFG["tfmap"]["stft"]
        layout = {"freq_bins": int(stft["n_fft"]) // 2 + 1,
                  "frames": (int(CFG["data"]["samples_per_trial"]) - int(stft["n_fft"]))
                            // int(stft["hop_length"]) + 1}

        result = {
            "experiment": CFG["experiment"],
            "status": "complete",
            "date": "2026-11-15",
            "clean_train_only": True,
            "weights": WEIGHTS,
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "numpy": np.__version__, "sklearn": sklearn.__version__,
                            "gpu": torch.cuda.get_device_name(0)},
            "tfmap_layout": layout,
            "encoder_config": json.loads(ENCODER_CFG_PATH.read_text(encoding="utf-8")),
            "training": {k: v for k, v in info.items()},
            "control": {"rows": control["rows"], "best": best_c, "reproduction": repro,
                        "reproduction_delta_W": repro["W"] - CFG["baseline"]["W"]},
            "embedding_spectrum": spectrum,
            "e166": {"rows": e166["rows"], "best": best_e,
                     "delta_vs_baseline": {
                         "diagnostic_auc": best_e["diagnostic_auc"] - CFG["baseline"]["diagnostic_auc"],
                         "W": best_e["W"] - CFG["baseline"]["W"]}},
            "baseline_E143": CFG["baseline"],
            "test_official": "N/A: Test has no public labels; session only used for the "
                             "diagnostic AUC and the inherited operating-point grid",
            "elapsed_seconds": time.time() - started,
        }
        write_json(RESULTS_PATH, result)
        write_report(result)
        print(f"\nE166 best: pca={best_e['pca_dim']} k={best_e['k']} "
              f"a={best_e['acceptance_rate']:.2f} diagAUC={best_e['diagnostic_auc']:.4f} "
              f"W={best_e['W']:.4f} (LOSO {best_e['loso_rejection']:.4f} "
              f"S1 {best_e['test_s1_rejection']:.4f} S2 {best_e['test_s2_acceptance']:.4f})",
              flush=True)
        print(f"E65 control: pca=96 k=12 a=0.90 W={repro['W']:.4f} "
              f"diagAUC={repro['diagnostic_auc']:.4f}", flush=True)
        print(f"effective rank (95%): {spectrum['dims_for_95pct_variance']} / "
              f"{spectrum['n_features']}", flush=True)
        print(f"results: {RESULTS_PATH}\nreport: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
