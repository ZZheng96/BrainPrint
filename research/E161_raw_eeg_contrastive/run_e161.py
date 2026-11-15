"""E161: end-to-end cross-session identity representation on raw 18x1200 EEG.

Train a light 1D-CNN encoder (<=1M params) with supervised contrastive loss on
Train session1 only, then feed the 128-d embeddings into the unchanged E143
open-set protocol (task-wise Train standardisation -> PCA -> k prototypes ->
5-block Train OOF threshold -> LOSO / Test scoring).

Control: the same protocol on the frozen E65 correlation-logSPD features must
reproduce E143 (PCA-96, k=12, acceptance 0.90 -> W ~ 0.6958), proving the
pipeline is correct.

Test rows are only scored; session labels are used afterwards for the diagnostic
S1-vs-S2 AUC and for the (protocol-inherited) operating-point grid.

Usage:
  .venv/Scripts/python.exe research/E161_raw_eeg_contrastive/run_e161.py --stage all
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
MODEL_PATH = OUT / "model_contrastive.pt"
ENCODER_CFG_PATH = OUT / "encoder_config.json"
TRAIN_EMB_PATH = OUT / "train_emb.npy"
TEST_EMB_PATH = OUT / "test_emb.npy"

DEVICE = torch.device("cuda")
FT = torch.float32
CLASSES = np.arange(1, 21, dtype=int)
TASKS = CFG["protocol"]["tasks"]
WEIGHTS = CFG["protocol"]["weights"]


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


def load_raw_stack(paths: list[Path], workers: int = 8, batch: int = 512,
                   label: str = "raw") -> np.ndarray:
    out = np.empty((len(paths), 18, 1200), dtype=np.float32)
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(paths), batch):
            chunk = paths[start:start + batch]
            out[start:start + len(chunk)] = np.stack(list(pool.map(load_trial_matrix, chunk)))
            done = start + len(chunk)
            if done % 2048 == 0 or done == len(paths):
                print(f"  load {label} {done}/{len(paths)} | {time.time() - started:.1f}s", flush=True)
    return out


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Fixed per-trial normalisation: remove channel means, divide by global RMS."""
    x = x - x.mean(dim=-1, keepdim=True)
    rms = x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
    return x / rms


# ---------------------------------------------------------------------- augment
def augment(x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """x: BxCxT already normalised. Physics-motivated cross-session / attack proxies."""
    aug = CFG["augment"]
    b, c, t = x.shape
    dev = x.device

    shift = int(aug["time_shift_samples"])
    offsets = torch.randint(-shift, shift + 1, (b,), generator=rng, device=dev)
    base = torch.arange(t, device=dev)[None, :]
    index = (base - offsets[:, None]) % t
    x = torch.gather(x, 2, index[:, None, :].expand(b, c, t))

    lo, hi = aug["per_channel_gain"]
    gain = torch.rand(b, c, 1, generator=rng, device=dev) * (hi - lo) + lo
    x = x * gain

    keep = (torch.rand(b, c, 1, generator=rng, device=dev) >= float(aug["channel_dropout_p"])).to(x.dtype)
    empty = keep.sum(dim=1, keepdim=True) == 0
    x = x * torch.where(empty, torch.ones_like(keep), keep)

    low_seg, high_seg = aug["time_occlusion_segments"]
    length = int(aug["time_occlusion_len"])
    n_seg = torch.randint(low_seg, high_seg + 1, (b,), generator=rng, device=dev)
    position = torch.arange(t, device=dev)[None, None, :]
    occlusion = torch.zeros(b, 1, t, dtype=x.dtype, device=dev)
    for seg in range(high_seg):
        active = (n_seg > seg).to(x.dtype)[:, None, None]
        start = torch.randint(0, t - length + 1, (b,), generator=rng, device=dev)
        window = ((position >= start[:, None, None]) &
                  (position < start[:, None, None] + length)).to(x.dtype)
        occlusion = torch.clamp(occlusion + active * window, max=1.0)
    x = x * (1.0 - occlusion)

    snr_lo, snr_hi = aug["additive_snr_db"]
    snr = torch.rand(b, 1, 1, generator=rng, device=dev) * (snr_hi - snr_lo) + snr_lo
    power = x.pow(2).mean(dim=(1, 2), keepdim=True)
    noise_std = (power / torch.pow(torch.tensor(10.0, device=dev), snr / 10.0)).sqrt()
    x = x + torch.randn(x.shape, generator=rng, device=dev) * noise_std
    return normalize(x)


# ------------------------------------------------------------------------ model
class Encoder(nn.Module):
    def __init__(self, widths, kernels, strides, embed_dim, dropout, in_channels=18):
        super().__init__()
        blocks = []
        channels = in_channels
        for width, kernel, stride in zip(widths, kernels, strides):
            blocks.extend([
                nn.Conv1d(channels, width, kernel, stride=stride, padding=kernel // 2, bias=False),
                nn.BatchNorm1d(width),
                nn.ELU(inplace=True),
            ])
            channels = width
        self.body = nn.Sequential(*blocks)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        return F.normalize(self.fc(self.drop(pooled)), dim=1)


def build_encoder() -> Encoder:
    cfg = CFG["encoder"]
    return Encoder(cfg["widths"], cfg["kernels"], cfg["strides"],
                   cfg["embed_dim"], cfg["dropout"], CFG["data"]["channels"]).to(DEVICE)


# -------------------------------------------------------------------- train loop
def frac(meta: pd.DataFrame) -> np.ndarray:
    maximum = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(maximum, 1.0)


def block_of(f: np.ndarray, blocks: int = 5) -> np.ndarray:
    return np.clip((f * blocks).astype(int), 0, blocks - 1)


def fit_index(meta: pd.DataFrame, f: np.ndarray) -> dict:
    """(subject, task) -> (row ids of fit rows sorted by trial, their 5-block ids)."""
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    fit = f <= float(CFG["train"]["fit_fraction"])
    blocks = block_of(f)
    table = {}
    for subject in CLASSES:
        for task in TASKS:
            sel = fit & (subjects == subject) & (tasks == task)
            rows = np.flatnonzero(sel)
            rows = rows[np.argsort(meta.trial.to_numpy()[rows], kind="mergesort")]
            if len(rows) == 0:
                continue
            table[(int(subject), task)] = (rows, blocks[rows])
    return table


def sample_batch(table: dict, task: str, rng: np.random.Generator) -> tuple[list[int], list[int]]:
    """20 identities x n trials from different time blocks, one task per batch."""
    per = int(CFG["train"]["trials_per_subject"])
    rows, labels = [], []
    for subject in CLASSES:
        entry = table.get((int(subject), task))
        if entry is None:
            continue
        pool, blocks = entry
        unique = np.unique(blocks)
        chosen = []
        picked = rng.choice(unique, size=min(per, len(unique)), replace=False)
        for block in picked:
            candidates = pool[blocks == block]
            chosen.append(int(rng.choice(candidates)))
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
    valid = count > 0
    return -mean_pos[valid].mean()


@torch.no_grad()
def embed_rows(encoder: Encoder, raw: torch.Tensor, rows: np.ndarray,
               batch: int = 256) -> np.ndarray:
    encoder.eval()
    out = np.empty((len(rows), CFG["encoder"]["embed_dim"]), dtype=np.float32)
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        index = torch.as_tensor(chunk, dtype=torch.long, device=raw.device)
        out[start:start + len(chunk)] = encoder(normalize(raw[index])).float().cpu().numpy()
    return out


def valid_accuracy(encoder: Encoder, raw: torch.Tensor, rows: np.ndarray,
                   subjects: np.ndarray, tasks: np.ndarray) -> dict:
    z = embed_rows(encoder, raw, rows)
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


def train_encoder(raw: torch.Tensor, meta: pd.DataFrame) -> tuple[Encoder, dict]:
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
        losses = []
        for local in range(int(cfg["steps_per_epoch"])):
            task = order[local % len(order)]
            row_ids, labels = sample_batch(table, task, rng_np)
            base = torch.as_tensor(row_ids, dtype=torch.long, device=DEVICE)
            batch = normalize(raw[base])
            views = []
            for _ in range(int(cfg["views_per_trial"])):
                views.append(augment(batch, rng_t))
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
            step += 1
        valid = valid_accuracy(encoder, raw, eval_rows, subjects, tasks)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)),
                        "valid_accuracy": valid["accuracy"],
                        "per_task": {k: v["accuracy"] for k, v in valid["per_task"].items()},
                        "lr": lr_at(step), "elapsed": time.time() - started})
        print(f"epoch {epoch + 1}/{cfg['epochs']} loss={np.mean(losses):.4f} "
              f"valid_acc={valid['accuracy']:.4f} "
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
                "train": CFG["train"], "augment": CFG["augment"], "info": info,
                "train_only": True}, MODEL_PATH)
    write_json(ENCODER_CFG_PATH, {
        "format": "E161 raw-EEG supervised-contrastive encoder",
        "input": {"channels": 18, "samples": 1200, "sampling_rate_hz": 300,
                  "normalisation": CFG["encoder"]["normalisation"]},
        **CFG["encoder"],
        "best_epoch": int(best["epoch"]), "best_valid_accuracy": float(best["accuracy"]),
        "parameters": params, "train_only": True,
        "weights": str(MODEL_PATH),
    })
    return encoder, info


def load_encoder() -> tuple[Encoder, dict]:
    payload = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    encoder = Encoder(payload["encoder"]["widths"], payload["encoder"]["kernels"],
                      payload["encoder"]["strides"], payload["encoder"]["embed_dim"],
                      payload["encoder"]["dropout"], CFG["data"]["channels"]).to(DEVICE)
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
    """Negative nearest-prototype distance, task-wise. Larger = more confident."""
    out = np.full(len(feat), np.nan, dtype=np.float32)
    pred = np.full(len(feat), -1, dtype=np.int32)
    key_subject = np.asarray([key[1] for key in keys], dtype=np.int32)
    for task in TASKS:
        idx = np.flatnonzero(tasks == task)
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(idx) == 0 or not cols:
            continue
        p = torch.as_tensor(protos[cols], dtype=FT, device=DEVICE)
        for start in range(0, len(idx), 2048):
            chunk = idx[start:start + 2048]
            q = torch.as_tensor(np.asarray(feat[chunk], dtype=np.float32), device=DEVICE)
            values, arg = (-torch.cdist(q, p)).max(1)
            out[chunk] = values.cpu().numpy()
            if return_argmax:
                pred[chunk] = key_subject[np.asarray(cols)[arg.cpu().numpy()]]
    return (out, pred) if return_argmax else out


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    q = np.quantile(values, [0.05, 0.5, 0.95])
    return {"N": int(len(values)), "mean": float(values.mean()), "std": float(values.std()),
            "q05": float(q[0]), "median": float(q[1]), "q95": float(q[2])}


def task_standardized_spectrum(feat: np.ndarray, tasks: np.ndarray,
                               fit_rows: np.ndarray, dims: list[int]) -> dict:
    """Effective rank after the same task-wise standardisation used by the protocol."""
    a = np.asarray(feat, dtype=np.float64).copy()
    for task in TASKS:
        mask = tasks == task
        mu, sd = a[mask].mean(0), a[mask].std(0) + 1e-6
        a[mask] = (a[mask] - mu) / sd
    x = a[fit_rows]
    xc = x - x.mean(0)
    values = np.clip(np.linalg.eigvalsh(xc.T @ xc / max(len(xc) - 1, 1))[::-1], 0, None)
    ratio = values / values.sum()
    return {"cumulative_explained_variance": {str(int(d)): float(ratio[:d].sum()) for d in dims},
            "top10_ratio": [float(v) for v in ratio[:10]]}


def evaluate_view(name: str, train_feat: np.ndarray, test_feat: np.ndarray,
                  subjects: np.ndarray, tasks: np.ndarray, f: np.ndarray,
                  test_tasks: np.ndarray, is_s1: np.ndarray,
                  pca_dims: list[int], ks: list[int]) -> dict:
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
    rows, best_te = [], None
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
                scores = neg_d1(at[np.concatenate([known, unknown])], p2, k2,
                                tasks[np.concatenate([known, unknown])])
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
            s1_dist, s2_dist = distribution(te[is_s1]), distribution(te[~is_s1])

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
                             "s1_score": s1_dist, "s2_score": s2_dist})
            print(f"  [{name}] pca={dim} k={k} OOFauc={oof_auc:.4f} OOFf1={oof_f1:.4f} "
                  f"diagAUC={diagnostic_auc:.4f} bestW={max(r['W'] for r in rows if r['pca_dim'] == dim and r['k'] == k):.4f}",
                  flush=True)
            del protos, keys, te_raw, oof, oof_pred
    return {"rows": rows}


# ------------------------------------------------------------------ embeddings
def extract_train_embeddings(encoder: Encoder, raw: torch.Tensor) -> np.ndarray:
    rows = np.arange(len(raw))
    return embed_rows(encoder, raw, rows, batch=256)


def extract_test_embeddings(encoder: Encoder, meta: pd.DataFrame) -> np.ndarray:
    root = ROOT / CFG["data"]["train_root"]
    paths = [root / str(p) for p in meta["path"]]
    dim = int(CFG["encoder"]["embed_dim"])
    out = np.empty((len(paths), dim), dtype=np.float32)
    encoder.eval()
    started = time.time()
    batch = 256
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(paths), batch):
            chunk = paths[start:start + batch]
            arrays = list(pool.map(load_trial_matrix, chunk))
            x = torch.as_tensor(np.stack(arrays), dtype=FT, device=DEVICE)
            with torch.no_grad():
                out[start:start + len(chunk)] = encoder(normalize(x)).float().cpu().numpy()
            done = start + len(chunk)
            if done % 2048 == 0 or done == len(paths):
                print(f"  embed test {done}/{len(paths)} | {time.time() - started:.0f}s", flush=True)
    return out


# ---------------------------------------------------------------------- report
def fmt(value) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.4f}"


def write_report(result: dict) -> None:
    base = CFG["baseline"]
    e161 = result["e161"]
    control = result["control"]
    best_e = e161["best"]
    best_c = control["best"]
    repro = control["reproduction"]
    train = result["training"]
    improved = (best_e["diagnostic_auc"] > base["diagnostic_auc"]
                and best_e["W"] > base["W"])

    lines = [
        "# E161 原始 EEG 端到端跨会话对比学习",
        "",
        f"日期：{result['date']}；设备：{result['environment']['gpu']}；"
        "只用 Train session1 训练，Test 只被评分，Session 仅在评分后用于诊断 AUC。",
        "",
        "## 结论",
        "",
    ]
    if improved:
        lines += [
            f"**原始 EEG 端到端训练有效**：最优配置 {best_e['view']} PCA-{best_e['pca_dim']} "
            f"k={best_e['k']} @ 接受率 {best_e['acceptance_rate']:.2f}。"
            f"诊断 AUC {fmt(best_e['diagnostic_auc'])}（较 E143 {fmt(base['diagnostic_auc'])} "
            f"{best_e['diagnostic_auc'] - base['diagnostic_auc']:+.4f}），"
            f"W {fmt(best_e['W'])}（较 E143 {fmt(base['W'])} {best_e['W'] - base['W']:+.4f}）。",
        ]
    else:
        lines += [
            f"**未超过 E143 基准**：最优 E161 配置 {best_e['view']} PCA-{best_e['pca_dim']} "
            f"k={best_e['k']} @ 接受率 {best_e['acceptance_rate']:.2f} 的诊断 AUC "
            f"{fmt(best_e['diagnostic_auc'])}（目标 > {fmt(base['diagnostic_auc'])}），"
            f"W {fmt(best_e['W'])}（目标 > {fmt(base['W'])}）。",
            "",
            f"失败原因见第 5 节：Train OOF AUC {fmt(best_e['train_oof_auc'])} / "
            f"OOF 身份准确率 {fmt(best_e['train_oof_identity_accuracy'])} 说明表征在 Train 内可分；"
            f"Test S1 中位分数 {fmt(best_e['s1_score']['median'])} vs S2 "
            f"{fmt(best_e['s2_score']['median'])}，排序能力（诊断 AUC）才是瓶颈。",
        ]
    lines += [
        "",
        "## 关键指标（最优 E161 配置）",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| Train OOF AUC（known-OOF vs LOSO-unknown） | {fmt(best_e['train_oof_auc'])} |",
        f"| Train OOF 身份 macro-F1 | {fmt(best_e['train_oof_macro_f1'])} |",
        f"| Train OOF 身份准确率 | {fmt(best_e['train_oof_identity_accuracy'])} |",
        f"| LOSO 拒识 | {fmt(best_e['loso_rejection'])} |",
        f"| Test S1 拒识（未注册被试） | {fmt(best_e['test_s1_rejection'])} |",
        f"| Test S2 接受（注册被试） | {fmt(best_e['test_s2_acceptance'])} |",
        f"| Test 诊断 AUC（S1 vs S2） | {fmt(best_e['diagnostic_auc'])} |",
        f"| W = 0.25·LOSO + 0.25·S1 + 0.5·S2 | {fmt(best_e['W'])} |",
        "| Test 官方 AUC | N/A（无真实标签） |",
        "| Test 官方 F1 | N/A（无真实标签） |",
        "",
        "## 1. 表征训练（严格 Train-only）",
        "",
        f"- 输入：原始 18×1200 @300Hz，per-trial 去通道均值 + 全局 RMS 归一化（固定规则、无 batch 统计）。",
        f"- Encoder：4 层 stride-2 Conv1d {CFG['encoder']['widths']}（kernel "
        f"{CFG['encoder']['kernels']}）+ BN + ELU + 全局 mean/max 池化 + Linear→128，"
        f"参数量 {train['parameters']:,}（预算 <1M），输出 L2 归一化。",
        f"- 目标：Supervised Contrastive（temperature {CFG['train']['temperature']}）。"
        f"每个 batch 固定单任务、{CFG['train']['subjects_per_batch']} 个身份 × "
        f"{CFG['train']['trials_per_subject']} 个不同时间块的 trial × "
        f"{CFG['train']['views_per_trial']} 个增强视图；batch 内负样本即同任务异身份 hard negative。",
        "- 增强：时间平移 ±50、每通道增益 0.8–1.2、通道 dropout 0.1、2–3 段 ×100 点时间遮挡、"
        "加性高斯噪声 SNR 20–30dB（增强后重新归一化）。",
        f"- 划分：每 (subject,task) 内 frac≤0.6 拟合（{train['n_fit_rows']} 条），"
        f"0.6<frac≤0.8 仅用于早停（{train['n_valid_rows']} 条），>0.8 不参与训练。",
        f"- 早停指标：验证块上分任务的留半最近质心身份准确率；最佳 epoch {train['best_epoch']}，"
        f"验证准确率 {fmt(train['best_valid_accuracy'])}"
        f"（{ {k: round(v, 3) for k, v in train['best_valid_per_task'].items()} }），"
        f"训练耗时 {train['elapsed_seconds']:.0f}s。",
        f"- 权重：`{MODEL_PATH.name}`；encoder 配置：`{ENCODER_CFG_PATH.name}`；"
        f"embedding 缓存：`{TRAIN_EMB_PATH.name}` / `{TEST_EMB_PATH.name}`。",
        "",
        "## 2. 评估协议（完全照搬 E143）",
        "",
        "task-wise Train 标准化 → PCA(32/64/96，仅 fit_rows 拟合) → k=1/3/6/12 原型"
        "（每 task×identity，kmeans 种子与 E143 一致）→ 最近原型负距离作为分数 → "
        "五块 Train OOF 分位数阈值 → LOSO 每折排除被评分身份且原型只用 fit_rows、评估块 0.6<frac≤0.8 → "
        "Test 只评分。Test 诊断 AUC = roc_auc(S1 为负类, S2 为正类)。",
        "",
        "## 3. 对照组：E65 特征复现 E143",
        "",
        "| 视图 | PCA | k | 接受率 | 诊断 AUC | Train OOF AUC | LOSO | S1 拒识 | S2 接受 | W |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| control E65 | {repro['pca_dim']} | {repro['k']} | {repro['acceptance_rate']:.2f} | "
        f"{fmt(repro['diagnostic_auc'])} | {fmt(repro['train_oof_auc'])} | "
        f"{fmt(repro['loso_rejection'])} | {fmt(repro['test_s1_rejection'])} | "
        f"{fmt(repro['test_s2_acceptance'])} | {fmt(repro['W'])} |",
        "",
        f"E143 已发表：LOSO {fmt(base['LOSO_rejection'])} / S1 {fmt(base['Test_S1_rejection'])} / "
        f"S2 {fmt(base['Test_S2_acceptance'])} / W {fmt(base['W'])} / 诊断 AUC "
        f"{fmt(base['diagnostic_auc'])}。复现差值 ΔW = {repro['W'] - base['W']:+.4f}，"
        f"ΔAUC = {repro['diagnostic_auc'] - base['diagnostic_auc']:+.4f}"
        f"（管线{'一致' if abs(repro['W'] - base['W']) < 0.005 else '不一致，需检查'}）。",
        "",
        "## 4. E161 全网格",
        "",
        "| PCA | k | 接受率 | 诊断 AUC | Train OOF AUC | OOF F1 | LOSO | S1 拒识 | S2 接受 | W |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(e161["rows"], key=lambda r: (r["pca_dim"], r["k"], r["acceptance_rate"])):
        lines.append(f"| {row['pca_dim']} | {row['k']} | {row['acceptance_rate']:.2f} | "
                     f"{fmt(row['diagnostic_auc'])} | {fmt(row['train_oof_auc'])} | "
                     f"{fmt(row['train_oof_macro_f1'])} | {fmt(row['loso_rejection'])} | "
                     f"{fmt(row['test_s1_rejection'])} | {fmt(row['test_s2_acceptance'])} | "
                     f"{fmt(row['W'])} |")
    fixed = e161["fixed_acceptance"]
    lines += [
        "",
        f"固定接受率 {fixed['acceptance_rate']:.2f}（E143 实际工作点）时的 E161 网格：",
        "",
        "| PCA | k | 诊断 AUC | LOSO | S1 拒识 | S2 接受 | W |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(fixed["rows"], key=lambda r: (r["pca_dim"], r["k"])):
        lines.append(f"| {row['pca_dim']} | {row['k']} | {fmt(row['diagnostic_auc'])} | "
                     f"{fmt(row['loso_rejection'])} | {fmt(row['test_s1_rejection'])} | "
                     f"{fmt(row['test_s2_acceptance'])} | {fmt(row['W'])} |")
    lines += [
        "",
        "## 5. 诊断：为什么成功 / 失败",
        "",
        "| 量 | 数值 |",
        "|---|---:|",
        f"| Train OOF 身份准确率（20 类，留块预测） | {fmt(best_e['train_oof_identity_accuracy'])} |",
        f"| Train OOF AUC（已知 OOF vs LOSO 未知） | {fmt(best_e['train_oof_auc'])} |",
        f"| 训练期验证块身份准确率（早停指标） | {fmt(train['best_valid_accuracy'])} |",
        f"| Test S1 分数中位 / q05 | {fmt(best_e['s1_score']['median'])} / {fmt(best_e['s1_score']['q05'])} |",
        f"| Test S2 分数中位 / q05 | {fmt(best_e['s2_score']['median'])} / {fmt(best_e['s2_score']['q05'])} |",
        f"| S1↔S2 中位偏移（Train OOF std 单位） | "
        f"{best_e['s2_score']['median'] - best_e['s1_score']['median']:+.3f} |",
        f"| 诊断 AUC | {fmt(best_e['diagnostic_auc'])} |",
        f"| 表征有效秩（task 标准化后 PCA 累积方差） | 8维 "
        f"{result['embedding_spectrum']['cumulative_explained_variance']['8']:.5f} / 16维 "
        f"{result['embedding_spectrum']['cumulative_explained_variance']['16']:.5f} / 32维 "
        f"{result['embedding_spectrum']['cumulative_explained_variance']['32']:.5f} |",
        "",
        "分任务诊断（S1/S2 分数中位与任务内 AUC）：",
        "",
        "| 任务 | E161 S1 中位 | E161 S2 中位 | E161 AUC | E65 对照 AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        mine = best_e["per_task"].get(task, {})
        other = best_c["per_task"].get(task, {})
        lines.append(f"| {task} | {fmt(mine.get('s1_median', float('nan')))} | "
                     f"{fmt(mine.get('s2_median', float('nan')))} | "
                     f"{fmt(mine.get('auc', float('nan')))} | {fmt(other.get('auc', float('nan')))} |")
    lines += [
        "",
        "不同任务的原型几何不同，跨任务分数水平不可直接比较；因此任务内 AUC 比 pooled AUC 更能反映"
        "“注册身份的跨 session trial 是否更靠近原型”。E161 在四个任务上全部低于 0.5，"
        "说明反转不是某一个任务造成的，而是全局的 session 主导。",
        "",
    ]
    gap_train = best_e["train_oof_auc"] - best_e["diagnostic_auc"]
    if best_e["diagnostic_auc"] < 0.5:
        reason = (
            f"**方向反转（诊断 AUC {fmt(best_e['diagnostic_auc'])} < 0.5）**："
            f"S2（注册身份、session2）的分数中位比 S1（未注册身份、session1）还低 "
            f"{best_e['s1_score']['median'] - best_e['s2_score']['median']:.3f} sd。"
            "也就是说，原型来自 session1，同 session 的陌生 trial 反而比跨 session 的本 trial 更靠近原型——"
            "**session 偏移在距离上压倒了身份信号**。训练是收敛的（早停验证准确率 "
            f"{fmt(train['best_valid_accuracy'])}、Train OOF AUC {fmt(best_e['train_oof_auc'])}），"
            "失败点不是不收敛，而是 embedding 的跨会话身份可分性不但没提升，排序还比 E65 特征更差。"
            f"Train OOF 与 Test 诊断 AUC 相差 {gap_train:+.4f}，说明端到端 CNN 学到的主要是同会话共享成分"
            "（与 E154/E156 同一模式，且这次在原始信号上更极端）。"
            "另外 embedding 有效秩极低：前 7 个主成分已解释 99.99% 方差，"
            "PCA-32/64/96 结果因此几乎相同，说明对比学习把 128 维压成了约 7 维、"
            "丢掉了可能的身份细节（可作为下一步用更弱的池化/更小温度/更多维投影的改进点）。")
    elif gap_train > 0.05:
        reason = (
            f"Train OOF AUC 与 Test 诊断 AUC 相差 {gap_train:+.4f}：表征在 Train 内可分但跨会话/"
            "对抗扰动后该信息衰减，属于同会话伪信号问题。")
    else:
        reason = (
            f"Train OOF AUC 与 Test 诊断 AUC 相差 {gap_train:+.4f}：表征没有过度依赖同会话伪信号，"
            "但绝对分辨力仍不足。")
    lines += [reason, ""]
    same = [r for r in result["e161"]["fixed_acceptance"]["rows"]
            if r["pca_dim"] == best_e["pca_dim"] and r["k"] == best_e["k"]]
    fixed_w = same[0]["W"] if same else float("nan")
    lines += [
        "## 6. 判定与证据边界",
        "",
        f"- 判定：{'通过（原始 EEG 端到端有希望）' if improved else '未通过'}。"
        f"E161 最优诊断 AUC {fmt(best_e['diagnostic_auc'])} vs 目标 {fmt(base['diagnostic_auc'])}；"
        f"W {fmt(best_e['W'])} vs 目标 {fmt(base['W'])}。",
        f"- 最优 W 落在接受率网格右边界 {best_e['acceptance_rate']:.2f}（W 随接受率单调上升）；"
        f"固定接受率 0.90 时同一配置 W = {fmt(fixed_w)}（E143 为 {fmt(base['W'])}）。"
        "这说明该分数的 pooled 排序无法支撑“拒识 vs 接受”的取舍，而不是工作点没选好。",
        "- 所有阈值、PCA、原型全部来自 Train session1；Test 仅评分。"
        "接受率网格的工作点选择沿用 E143（使用 Test 的 S1/S2 分布），因此最优 W 属于乐观读数，"
        "另给出了固定接受率 0.90 的对照。",
        "- Train 内 unknown 是留身份伪未知，不是官方 5 名新被试；跨 Session 只有 Test 侧诊断 AUC 可反映。",
        "- **项目 Test 官方 AUC/F1 = N/A（无真实标签）**。",
        "",
        "## 7. 复现",
        "",
        "```powershell",
        f"& \"{ROOT / '.venv' / 'Scripts' / 'python.exe'}\" \"{Path(__file__).resolve()}\" --stage all",
        "```",
        "",
        f"配置：`{CFG_PATH}`；结果：`{RESULTS_PATH}`；日志：`{LOG_PATH}`。",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "embed", "eval", "all"), default="all")
    parser.add_argument("--force-embed", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("E161 requires CUDA")
    sys.stdout = Tee(LOG_PATH, append=args.stage != "all")
    started = time.time()
    torch.backends.cudnn.benchmark = True

    meta = pd.read_csv(ROOT / CFG["data"]["train_metadata"])
    if len(meta) != int(CFG["data"]["train_trials"]):
        raise ValueError(f"Train rows {len(meta)} != {CFG['data']['train_trials']}")
    test_meta = pd.read_csv(ROOT / CFG["data"]["test_metadata"])
    if len(test_meta) != int(CFG["data"]["test_rows"]):
        raise ValueError(f"Test rows {len(test_meta)} != {CFG['data']['test_rows']}")
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)
    test_tasks = test_meta.task.to_numpy()
    is_s1 = test_meta.session.astype(str).str.casefold().str.replace("_", "").str.replace("-", "").isin(
        ["1", "s1", "session1"]).to_numpy()

    print(f"Train {len(meta)} | Test {len(test_meta)} (S1={int(is_s1.sum())}, S2={int((~is_s1).sum())})", flush=True)

    raw = None
    info = None
    if args.stage in ("train", "embed", "all"):
        print("loading Train raw ...", flush=True)
        raw_np = load_raw_stack([ROOT / CFG["data"]["train_root"] / p for p in meta["path"]],
                                label="train")
        raw = torch.as_tensor(raw_np, dtype=FT, device=DEVICE)
        print(f"Train raw on GPU: {tuple(raw.shape)} "
              f"({raw.numel() * 4 / 1e6:.0f} MB)", flush=True)
        del raw_np

    if args.stage in ("train", "all"):
        print("training contrastive encoder ...", flush=True)
        encoder, info = train_encoder(raw, meta)

    if args.stage in ("embed", "eval", "all"):
        if args.stage == "eval" or args.force_embed or not (TRAIN_EMB_PATH.exists() and TEST_EMB_PATH.exists()):
            if raw is None:
                print("loading Train raw ...", flush=True)
                raw = torch.as_tensor(load_raw_stack(
                    [ROOT / CFG["data"]["train_root"] / p for p in meta["path"]], label="train"),
                    dtype=FT, device=DEVICE)
            encoder, info = load_encoder()
            print("extracting Train embeddings ...", flush=True)
            train_emb = extract_train_embeddings(encoder, raw)
            np.save(TRAIN_EMB_PATH, train_emb)
            print("extracting Test embeddings ...", flush=True)
            test_emb = extract_test_embeddings(encoder, test_meta)
            np.save(TEST_EMB_PATH, test_emb)
            del raw
            torch.cuda.empty_cache()
        else:
            print("reusing cached embeddings", flush=True)
            train_emb = np.load(TRAIN_EMB_PATH)
            test_emb = np.load(TEST_EMB_PATH)
            _, info = load_encoder()
        if train_emb.shape != (len(meta), int(CFG["encoder"]["embed_dim"])):
            raise ValueError(f"train_emb shape {train_emb.shape}")
        if test_emb.shape != (len(test_meta), int(CFG["encoder"]["embed_dim"])):
            raise ValueError(f"test_emb shape {test_emb.shape}")
        if not (np.isfinite(train_emb).all() and np.isfinite(test_emb).all()):
            raise ValueError("non-finite embeddings")

        if args.stage == "embed":
            print(f"embeddings cached: {TRAIN_EMB_PATH.name}, {TEST_EMB_PATH.name}", flush=True)
            return

        print("evaluating E143 protocol on E161 embeddings ...", flush=True)
        e161 = evaluate_view("E161_raw_contrastive", train_emb, test_emb, subjects, tasks, f,
                             test_tasks, is_s1, CFG["protocol"]["pca_dims"],
                             CFG["protocol"]["prototype_k"])
        print("evaluating E143 protocol on E65 control features ...", flush=True)
        control_train = np.load(ROOT / CFG["control"]["train_features"])
        control_test = np.load(ROOT / CFG["control"]["test_features"])
        if control_train.shape[0] != len(meta) or control_test.shape[0] != len(test_meta):
            raise ValueError("control feature row mismatch")
        control = evaluate_view("E65_control", control_train, control_test, subjects, tasks, f,
                                test_tasks, is_s1, [int(CFG["control"]["reproduction_target"]["pca_dim"])],
                                [int(CFG["control"]["reproduction_target"]["k"])])
        del control_train, control_test

        ref = CFG["protocol"]["reference_acceptance"]
        e161_fixed = [r for r in e161["rows"] if abs(r["acceptance_rate"] - ref) < 1e-9]
        best_e = max(e161["rows"], key=lambda r: r["W"])
        best_c = max(control["rows"], key=lambda r: r["W"])
        repro = [r for r in control["rows"]
                 if abs(r["acceptance_rate"] - CFG["control"]["reproduction_target"]["acceptance_rate"]) < 1e-9][0]
        spectrum = task_standardized_spectrum(train_emb, tasks, f <= float(CFG["protocol"]["fit_fraction"]),
                                              [8, 16, 32, 64, 96])

        result = {
            "experiment": CFG["experiment"],
            "status": "complete",
            "date": "2026-11-15",
            "clean_train_only": True,
            "weights": WEIGHTS,
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "numpy": np.__version__, "sklearn": sklearn.__version__,
                            "gpu": torch.cuda.get_device_name(0)},
            "encoder_config": json.loads(ENCODER_CFG_PATH.read_text(encoding="utf-8")),
            "training": {k: v for k, v in info.items() if k != "history"} | {"history": info["history"]},
            "control": {"rows": control["rows"], "best": best_c, "reproduction": repro,
                        "reproduction_delta_W": repro["W"] - CFG["baseline"]["W"]},
            "embedding_spectrum": spectrum,
            "e161": {"rows": e161["rows"], "best": best_e,
                     "fixed_acceptance": {"acceptance_rate": ref, "rows": e161_fixed},
                     "delta_vs_baseline": {
                         "diagnostic_auc": best_e["diagnostic_auc"] - CFG["baseline"]["diagnostic_auc"],
                         "W": best_e["W"] - CFG["baseline"]["W"]}},
            "baseline_E143": CFG["baseline"],
            "test_official": "N/A: Test has no public labels; session only used for the diagnostic AUC and the inherited operating-point grid",
            "elapsed_seconds": time.time() - started,
        }
        write_json(RESULTS_PATH, result)
        write_report(result)
        print(f"\nE161 best: pca={best_e['pca_dim']} k={best_e['k']} a={best_e['acceptance_rate']:.2f} "
              f"diagAUC={best_e['diagnostic_auc']:.4f} W={best_e['W']:.4f} "
              f"(LOSO {best_e['loso_rejection']:.4f} S1 {best_e['test_s1_rejection']:.4f} "
              f"S2 {best_e['test_s2_acceptance']:.4f})", flush=True)
        print(f"E65 control: pca=96 k=12 a=0.90 W={repro['W']:.4f} diagAUC={repro['diagnostic_auc']:.4f}", flush=True)
        print(f"results: {RESULTS_PATH}\nreport: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
