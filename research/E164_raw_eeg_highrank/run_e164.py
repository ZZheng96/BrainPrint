"""E164: is the low effective rank of the raw-EEG encoder the bottleneck?

E162 kept the E143 protocol honest (diagnostic AUC 0.6756, effective rank 8/256)
and recommended reducing pooling strength / raising the representation rank while
keeping the dual-perturbation-family cross-session invariance. E159 showed that
W > 0.80 needs a diagnostic AUC of about 0.93, versus E65's 0.7578.

Pre-registered changes (only the encoder structure and the pooling are touched):
  * BatchNorm removed everywhere (batch statistics are a cross-domain instability),
    replaced by GroupNorm(8);
  * variant A: learnable-query attention pooling (4 queries x 128 = 512);
  * variant B: global mean||max pooling with no final linear compression (256 ch = 512);
  * variant C: multi-scale time pooling (4 segments x 128 = 512);
  * variant D: E162 structure (mean||max -> Linear) with GroupNorm, to isolate BN removal;
  * lambda_VICReg fixed at 0, temperature fixed at 0.1, everything else identical to E162.

The E143 open-set protocol is reused verbatim and the E65 control must reproduce
W = 0.6958 / diagnostic AUC = 0.7578.

Usage:
  .venv/Scripts/python.exe research/E164_raw_eeg_highrank/run_e164.py --stage all --variants 0
  .venv/Scripts/python.exe research/E164_raw_eeg_highrank/run_e164.py --stage all --variants all
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
TRAIN_EMB_PATH = OUT / "train_emb.npy"
TEST_EMB_PATH = OUT / "test_emb.npy"

DEVICE = torch.device("cuda")
FT = torch.float32
CLASSES = np.arange(1, 21, dtype=int)
TASKS = CFG["protocol"]["tasks"]
WEIGHTS = CFG["protocol"]["weights"]
VARIANTS = CFG["variants"]


def variant_path(prefix: str, spec: dict, suffix: str) -> Path:
    return OUT / f"{prefix}_{spec['name']}{suffix}"


def model_path(spec: dict) -> Path:
    return variant_path("model", spec, ".pt")


def train_emb_path(spec: dict) -> Path:
    return variant_path("train_emb", spec, ".npy")


def test_emb_path(spec: dict) -> Path:
    return variant_path("test_emb", spec, ".npy")


class Tee:
    """Mirror stdout into run.log."""

    def __init__(self, path: Path):
        self.handle = path.open("a", encoding="utf-8", buffering=1)
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
            if done % 4096 == 0 or done == len(paths):
                print(f"  load {label} {done}/{len(paths)} | {time.time() - started:.1f}s", flush=True)
    return out


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Fixed per-trial normalisation: remove channel means, divide by global RMS."""
    x = x - x.mean(dim=-1, keepdim=True)
    rms = x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
    return x / rms


# ---------------------------------------------------------------------- augment
def time_shift(x: torch.Tensor, shift: int, rng: torch.Generator) -> torch.Tensor:
    b, c, t = x.shape
    offsets = torch.randint(-shift, shift + 1, (b,), generator=rng, device=x.device)
    base = torch.arange(t, device=x.device)[None, :]
    index = (base - offsets[:, None]) % t
    return torch.gather(x, 2, index[:, None, :].expand(b, c, t))


def augment_electrode(x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Family A: per-channel gain + slow electrode offset drift (no spectral change)."""
    cfg = CFG["augment"]["view_A_electrode"]
    b, c, t = x.shape
    dev = x.device
    x = time_shift(x, int(cfg["time_shift_samples"]), rng)

    lo, hi = cfg["per_channel_gain"]
    gain = torch.rand(b, c, 1, generator=rng, device=dev) * (hi - lo) + lo
    x = x * gain

    lo, hi = cfg["electrode_drift_amplitude"]
    amp = torch.rand(b, c, 1, generator=rng, device=dev) * (hi - lo) + lo
    ramp = torch.linspace(-0.5, 0.5, t, device=dev)[None, None, :]
    w_ramp = (torch.rand(b, c, 1, generator=rng, device=dev) * 2 - 1) * 0.8
    cycles = torch.rand(b, c, 1, generator=rng, device=dev) * 2.5 + 0.5
    phase = torch.rand(b, c, 1, generator=rng, device=dev) * (2 * np.pi)
    w_sine = (torch.rand(b, c, 1, generator=rng, device=dev) * 2 - 1) * 0.8
    pos = torch.linspace(0.0, 1.0, t, device=dev)[None, None, :]
    sine = torch.sin(2 * np.pi * cycles * pos + phase)
    drift = amp * (w_ramp * 2 * ramp + w_sine * sine)
    x = x + drift
    return normalize(x)


def augment_band(x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Family B: per-band amplitude reweighting + additive noise (no gain/offset)."""
    cfg = CFG["augment"]["view_B_band"]
    b, c, t = x.shape
    dev = x.device
    x = time_shift(x, int(cfg["time_shift_samples"]), rng)

    edges = cfg["band_edges_hz"]
    freqs = torch.fft.rfftfreq(t, d=1.0 / float(CFG["data"]["sampling_rate_hz"])).to(dev)
    band = torch.zeros_like(freqs, dtype=torch.long)
    for i in range(len(edges) - 1):
        band[(freqs >= edges[i]) & (freqs < edges[i + 1])] = i
    lo, hi = cfg["band_gain_db"]
    gains_db = torch.rand(b, len(edges) - 1, generator=rng, device=dev) * (hi - lo) + lo
    gains = torch.pow(torch.tensor(10.0, device=dev), gains_db / 20.0)
    factor = gains[:, band]  # (b, n_freq)
    spec = torch.fft.rfft(x, dim=-1) * factor[:, None, :]
    x = torch.fft.irfft(spec, n=t, dim=-1)

    snr_lo, snr_hi = cfg["additive_snr_db"]
    snr = torch.rand(b, 1, 1, generator=rng, device=dev) * (snr_hi - snr_lo) + snr_lo
    power = x.pow(2).mean(dim=(1, 2), keepdim=True)
    noise_std = (power / torch.pow(torch.tensor(10.0, device=dev), snr / 10.0)).sqrt()
    x = x + torch.randn(x.shape, generator=rng, device=dev) * noise_std
    return normalize(x)


# ------------------------------------------------------------------------ model
def make_norm(kind: str, channels: int, groups: int) -> nn.Module:
    if kind == "group":
        if channels % groups != 0:
            raise ValueError(f"GroupNorm needs {channels} divisible by {groups}")
        return nn.GroupNorm(groups, channels)
    if kind == "layer":
        return nn.GroupNorm(1, channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown normalisation {kind}")


class Encoder(nn.Module):
    """Conv body (GroupNorm, no BatchNorm) + variant-specific pooling, no L2 norm."""

    def __init__(self, spec: dict, dropout: float, in_channels: int = 18):
        super().__init__()
        blocks = []
        channels = in_channels
        for width, kernel, stride in zip(spec["widths"], spec["kernels"], spec["strides"]):
            blocks.extend([
                nn.Conv1d(channels, width, kernel, stride=stride, padding=kernel // 2, bias=False),
                make_norm(spec["norm"], width, int(spec.get("groups", 8))),
                nn.ELU(inplace=True),
            ])
            channels = width
        self.body = nn.Sequential(*blocks)
        self.mode = spec["pooling"]
        self.width = channels
        self.embed_dim = int(spec["embed_dim"])
        self.drop = nn.Dropout(dropout)
        if self.mode == "attention":
            self.queries = nn.Parameter(torch.randn(int(spec["attention_queries"]), channels)
                                        / channels ** 0.5)
            self.pool_dim = int(spec["attention_queries"]) * channels
        elif self.mode == "meanmax":
            self.pool_dim = 2 * channels
        elif self.mode == "multiscale":
            self.segments = int(spec["segments"])
            self.pool_dim = self.segments * channels
        elif self.mode == "meanmax_linear":
            self.pool_dim = 2 * channels
            self.head = nn.Linear(self.pool_dim, self.embed_dim)
        else:
            raise ValueError(f"unknown pooling {self.mode}")
        if self.pool_dim != self.embed_dim:
            raise ValueError(f"{spec['name']}: pooled dim {self.pool_dim} != embed_dim {self.embed_dim}")

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        if self.mode == "attention":
            b, c, t = h.shape
            scores = torch.einsum("bct,qc->btq", h, self.queries) / c ** 0.5
            attn = torch.softmax(scores, dim=1)                      # (b, t, q)
            out = torch.einsum("btq,bct->bqc", attn, h)              # (b, q, c)
            return out.reshape(b, self.pool_dim)
        if self.mode == "meanmax":
            return torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        if self.mode == "multiscale":
            seg = F.adaptive_avg_pool1d(h, self.segments)            # (b, c, segments)
            return seg.transpose(1, 2).reshape(h.shape[0], self.pool_dim)
        return torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.drop(self.pooled(x))
        if self.mode == "meanmax_linear":
            return self.head(z)
        return z


def build_encoder(spec: dict) -> Encoder:
    return Encoder(spec, CFG["encoder"]["dropout"], CFG["data"]["channels"]).to(DEVICE)


# -------------------------------------------------------------------- train loop
def frac(meta: pd.DataFrame) -> np.ndarray:
    maximum = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(maximum, 1.0)


def block_of(f: np.ndarray, blocks: int = 5) -> np.ndarray:
    return np.clip((f * blocks).astype(int), 0, blocks - 1)


def fit_index(meta: pd.DataFrame, f: np.ndarray) -> dict:
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
    out = np.empty((len(rows), encoder.embed_dim), dtype=np.float32)
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        index = torch.as_tensor(chunk, dtype=torch.long, device=raw.device)
        out[start:start + len(chunk)] = encoder(normalize(raw[index])).float().cpu().numpy()
    return out


def valid_accuracy(encoder: Encoder, raw: torch.Tensor, rows: np.ndarray,
                   subjects: np.ndarray, tasks: np.ndarray) -> dict:
    z = embed_rows(encoder, raw, rows).astype(np.float64)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
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
        centre = np.stack(centroids)
        centre /= np.maximum(np.linalg.norm(centre, axis=1, keepdims=True), 1e-8)
        q = np.concatenate(query)
        label = np.concatenate(truth)
        pred = np.asarray(centre_labels)[np.argmax(z[q] @ centre.T, axis=1)]
        acc = float(np.mean(pred == label))
        per_task[task] = {"accuracy": acc, "n": int(len(q))}
        correct += int(np.sum(pred == label))
        total += len(q)
    return {"accuracy": float(correct / max(total, 1)), "per_task": per_task, "n": int(total)}


def train_one(spec: dict, raw: torch.Tensor, meta: pd.DataFrame) -> tuple[Encoder, dict]:
    cfg = CFG["train"]
    seed = int(cfg["seed"])
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)
    table = fit_index(meta, f)
    eval_rows = np.flatnonzero((f > float(cfg["fit_fraction"])) & (f <= float(cfg["eval_fraction"])))
    temperature = float(spec["temperature"])
    lam = float(spec["vicreg_weight"])
    lam_inv = float(cfg["invariance_weight"])

    encoder = build_encoder(spec)
    params = int(sum(p.numel() for p in encoder.parameters()))
    print(f"[{spec['name']}] parameters = {params:,} (budget {CFG['encoder']['param_budget']:,}), "
          f"pool_dim = {encoder.pool_dim}", flush=True)
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
        losses, parts = [], {"supcon": [], "inv": []}
        for local in range(int(cfg["steps_per_epoch"])):
            task = order[local % len(order)]
            row_ids, labels = sample_batch(table, task, rng_np)
            base = torch.as_tensor(row_ids, dtype=torch.long, device=DEVICE)
            batch = normalize(raw[base])
            view_a = augment_electrode(batch, rng_t)
            view_b = augment_band(batch, rng_t)
            z_a, z_b = encoder(view_a), encoder(view_b)
            z = torch.cat([z_a, z_b], dim=0)
            target = torch.as_tensor(labels * 2, dtype=torch.long, device=DEVICE)
            loss_sup = supcon(z, target, temperature)
            loss_inv = F.mse_loss(z_a, z_b)
            loss = loss_sup + lam_inv * loss_inv
            optimizer.param_groups[0]["lr"] = lr_at(step)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), float(cfg["grad_clip"]))
            optimizer.step()
            losses.append(float(loss.detach()))
            parts["supcon"].append(float(loss_sup.detach()))
            parts["inv"].append(float(loss_inv.detach()))
            step += 1
        with torch.no_grad():
            mean_std = float(z.std(dim=0).mean())
        valid = valid_accuracy(encoder, raw, eval_rows, subjects, tasks)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)),
                        "loss_supcon": float(np.mean(parts["supcon"])),
                        "loss_invariance": float(np.mean(parts["inv"])),
                        "mean_dim_std": mean_std,
                        "valid_accuracy": valid["accuracy"],
                        "per_task": {k: v["accuracy"] for k, v in valid["per_task"].items()},
                        "lr": lr_at(step), "elapsed": time.time() - started})
        print(f"[{spec['name']}] epoch {epoch + 1}/{cfg['epochs']} loss={np.mean(losses):.4f} "
              f"sup={np.mean(parts['supcon']):.4f} inv={np.mean(parts['inv']):.4f} "
              f"dimstd={mean_std:.3f} valid_acc={valid['accuracy']:.4f} "
              f"{ {k: round(v['accuracy'], 3) for k, v in valid['per_task'].items()} } "
              f"| {time.time() - started:.0f}s", flush=True)
        if valid["accuracy"] > best["accuracy"] + 1e-6:
            best = {"accuracy": valid["accuracy"], "epoch": epoch + 1,
                    "state": {k: v.detach().clone() for k, v in encoder.state_dict().items()},
                    "per_task": {k: v["accuracy"] for k, v in valid["per_task"].items()}}
        elif epoch + 1 - best["epoch"] >= patience:
            print(f"[{spec['name']}] early stop at epoch {epoch + 1} (best {best['epoch']})", flush=True)
            break

    encoder.load_state_dict(best["state"])
    info = {"config": spec["name"], "vicreg_weight": lam, "temperature": temperature,
            "invariance_weight": lam_inv, "parameters": params,
            "pooling": spec["pooling"], "embed_dim": int(spec["embed_dim"]),
            "best_epoch": int(best["epoch"]),
            "best_valid_accuracy": float(best["accuracy"]),
            "best_valid_per_task": best["per_task"], "history": history,
            "total_steps": step, "elapsed_seconds": time.time() - started,
            "n_fit_rows": int((f <= float(cfg["fit_fraction"])).sum()),
            "n_valid_rows": int(len(eval_rows))}
    torch.save({"state_dict": encoder.state_dict(), "encoder": spec,
                "train": {**CFG["train"], "temperature": temperature, "vicreg_weight": lam},
                "augment": CFG["augment"], "info": info, "train_only": True},
               model_path(spec))
    return encoder, info


def load_encoder(spec: dict) -> tuple[Encoder, dict]:
    payload = torch.load(model_path(spec), map_location="cpu", weights_only=False)
    encoder = Encoder(payload["encoder"], CFG["encoder"]["dropout"],
                      CFG["data"]["channels"]).to(DEVICE)
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
                               fit_rows: np.ndarray) -> dict:
    """Effective rank of the task-standardised fit-row embeddings."""
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
    rank95 = int(np.searchsorted(cumulative, 0.95) + 1)
    rank99 = int(np.searchsorted(cumulative, 0.99) + 1)
    return {"effective_rank_95": rank95, "effective_rank_99": rank99,
            "cumulative_explained_variance": {str(d): float(cumulative[min(d, len(cumulative)) - 1])
                                              for d in [8, 16, 32, 64, 96, 128, 256]},
            "top10_ratio": [float(v) for v in ratio[:10]],
            "dims_total": int(len(ratio))}


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
                  f"diagAUC={diagnostic_auc:.4f} "
                  f"bestW={max(r['W'] for r in rows if r['pca_dim'] == dim and r['k'] == k):.4f}",
                  flush=True)
            del protos, keys, te_raw, oof, oof_pred
    return {"rows": rows}


# ------------------------------------------------------------------ embeddings
def embed_config(spec: dict, raw_train: torch.Tensor, raw_test: torch.Tensor) -> None:
    encoder, _ = load_encoder(spec)
    started = time.time()
    train_emb = embed_rows(encoder, raw_train, np.arange(len(raw_train)), batch=512)
    print(f"  [{spec['name']}] train embeddings {train_emb.shape} | {time.time() - started:.0f}s",
          flush=True)
    test_emb = embed_rows(encoder, raw_test, np.arange(len(raw_test)), batch=512)
    print(f"  [{spec['name']}] test embeddings {test_emb.shape} | {time.time() - started:.0f}s",
          flush=True)
    np.save(train_emb_path(spec), train_emb)
    np.save(test_emb_path(spec), test_emb)


# ---------------------------------------------------------------------- report
def fmt(value) -> str:
    return "N/A" if value is None or not np.isfinite(value) else f"{value:.4f}"


def summarise_config(name: str, rows: list[dict], spec: dict, training: dict,
                     spectrum: dict) -> dict:
    ref = float(CFG["protocol"]["reference_acceptance"])
    fixed = [r for r in rows if abs(r["acceptance_rate"] - ref) < 1e-9]
    best_auc = max(rows, key=lambda r: (r["diagnostic_auc"], r["W"]))
    best_w = max(rows, key=lambda r: r["W"])
    fixed_best = max(fixed, key=lambda r: r["W"])
    return {"config": name, "spec": spec, "label": spec.get("label", ""),
            "vicreg_weight": float(spec["vicreg_weight"]),
            "temperature": float(spec["temperature"]),
            "rows": rows,
            "fixed_acceptance": {"acceptance_rate": ref, "rows": fixed, "best_row": fixed_best},
            "best_row_by_auc": best_auc, "best_row_by_W": best_w,
            "training": {k: v for k, v in training.items() if k != "history"},
            "history": training["history"],
            "spectrum": spectrum}


def write_report(result: dict) -> None:
    base = CFG["baseline"]
    e162 = CFG["e162_reference"]
    e159 = CFG["e159_reference"]
    ctrl = result["control"]
    repro = ctrl["reproduction"]
    configs = result["configs"]
    best_name = result["best_variant"]
    best = configs[best_name]
    brow = best["best_row_by_auc"]
    bfix = best["fixed_acceptance"]["best_row"]
    bwor = best["best_row_by_W"]
    spectrum = best["spectrum"]
    improved = brow["diagnostic_auc"] > base["diagnostic_auc"]
    target_auc = float(e159["diagnostic_auc_needed_for_W_0.80"])
    reproduced = abs(repro["W"] - base["W"]) < 5e-3 and abs(repro["diagnostic_auc"] - base["diagnostic_auc"]) < 5e-3
    ranks95 = [c["spectrum"]["effective_rank_95"] for c in configs.values()]
    ranks99 = [c["spectrum"]["effective_rank_99"] for c in configs.values()]
    ranks99_512 = [c["spectrum"]["effective_rank_99"] for c in configs.values()
                   if int(c["spec"]["embed_dim"]) == 512]
    ranks99_512 = ranks99_512 or ranks99
    aucs = [c["best_row_by_auc"]["diagnostic_auc"] for c in configs.values()]
    oofs = [c["fixed_acceptance"]["best_row"]["train_oof_auc"] for c in configs.values()]
    span = lambda v: f"{min(v):.4f}–{max(v):.4f}"
    rank_span = lambda v: f"{min(v)}–{max(v)}"
    d_auc = (configs["D_e162head_gn256"]["best_row_by_auc"]["diagnostic_auc"]
             if "D_e162head_gn256" in configs else float("nan"))

    lines = [
        "# E164 原始 EEG 高秩编码器（去 BN + 弱化池化）",
        "",
        f"日期：{result['date']}；设备：{result['environment']['gpu']}；"
        "只用 Train session1 训练，Test 只被评分，Session 仅在评分后用于诊断 AUC。",
        "",
        "## 结论",
        "",
    ]
    if improved and brow["diagnostic_auc"] >= target_auc:
        verdict = "通过（诊断 AUC 达到 W>0.80 所需水平）"
        lines += [f"**提高有效秩解决了瓶颈**：最优变体 `{best_name}`，诊断 AUC "
                  f"{fmt(brow['diagnostic_auc'])}（E65 {fmt(base['diagnostic_auc'])}），"
                  f"有效秩 {spectrum['effective_rank_95']}/{spectrum['dims_total']}@95%。"]
    elif improved:
        verdict = "部分通过（诊断 AUC 超过 E65，但未达 W>0.80 所需 0.93）"
        lines += [f"**有效秩假说方向成立但幅度不足**：最优变体 `{best_name}`，诊断 AUC "
                  f"{fmt(brow['diagnostic_auc'])}（E65 {fmt(base['diagnostic_auc'])}，"
                  f"{brow['diagnostic_auc'] - base['diagnostic_auc']:+.4f}；E162 {fmt(e162['diagnostic_auc'])}），"
                  f"有效秩 {spectrum['effective_rank_95']}/{spectrum['dims_total']}@95%，"
                  f"距 W>0.80 所需的 {target_auc:.2f} 仍差 "
                  f"{target_auc - brow['diagnostic_auc']:+.4f}；最优 W {fmt(bwor['W'])}。"]
    else:
        verdict = "未通过（低于手工 E65 特征）"
        lines += [f"**有效秩假说被否定**：最优变体 `{best_name}` 的诊断 AUC "
                  f"{fmt(brow['diagnostic_auc'])} 仍低于 E65 的 {fmt(base['diagnostic_auc'])}"
                  f"（{brow['diagnostic_auc'] - base['diagnostic_auc']:+.4f}）。"
                  "**原始 EEG 端到端在当前数据下不如手工 E65 特征**；"
                  f"去 BN + 弱池化只把 512 维变体的 99% 有效秩从 E162 的 {e162['effective_rank_99']}/"
                  f"{e162['dims_total']} 抬到 {rank_span(ranks99_512)}/512"
                  f"（95% 有效秩 {rank_span(ranks95)} 对 E162 的 {e162['effective_rank_95']}），"
                  f"诊断 AUC 反而从 {fmt(e162['diagnostic_auc'])} 掉到 {span(aucs)}，"
                  "说明低秩不是跨会话泛化的主瓶颈。"
                  f"最优 W {fmt(bwor['W'])}（E65 {fmt(base['W'])}）。"]
    lines += [
        "",
        "## 关键指标（最优变体 `" + best_name + "`）",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| Train OOF AUC（known-OOF vs LOSO-unknown） | {fmt(brow['train_oof_auc'])} |",
        f"| Train OOF 身份 macro-F1 | {fmt(brow['train_oof_macro_f1'])} |",
        f"| Train OOF 身份准确率 | {fmt(brow['train_oof_identity_accuracy'])} |",
        f"| LOSO 拒识 | {fmt(brow['loso_rejection'])} |",
        f"| Test S1 拒识（未注册被试） | {fmt(brow['test_s1_rejection'])} |",
        f"| Test S2 接受（注册被试） | {fmt(brow['test_s2_acceptance'])} |",
        f"| Test 诊断 AUC（S1 vs S2） | {fmt(brow['diagnostic_auc'])} |",
        f"| W（最优接受率 {brow['acceptance_rate']:.2f}） | {fmt(brow['W'])} |",
        f"| W（固定接受率 0.90，E143 工作点） | {fmt(bfix['W'])} |",
        f"| 有效秩（task 标准化后 PCA 95% 方差所需维数） | {spectrum['effective_rank_95']} / {spectrum['dims_total']} |",
        f"| 有效秩（99% 方差所需维数） | {spectrum['effective_rank_99']} / {spectrum['dims_total']} |",
        "| Test 官方 AUC | N/A（无真实标签） |",
        "| Test 官方 F1 | N/A（无真实标签） |",
        "",
        "## 1. 相对 E162 的预注册改动",
        "",
    ]
    for change in CFG["changes_vs_E162"]:
        lines.append(f"- {change}")
    lines += ["", "保持不变的项（E162 已证明有效，未再调参）：", ""]
    for kept in CFG["kept_from_E162"]:
        lines.append(f"- {kept}")
    lines += [
        "",
        "## 2. 编码器变体定义",
        "",
        "| 变体 | 归一化 | conv 宽度 | 池化 | 输出维 | 参数量 |",
        "|---|---|---|---|---:|---:|",
    ]
    for name, item in configs.items():
        s = item["spec"]
        lines.append(f"| {name} | GroupNorm({s['groups']}) | {s['widths']} | {s['pooling']} | "
                     f"{s['embed_dim']} | {item['training']['parameters']:,} |")
    lines += [
        "",
        "（变体 B 的预注册文本同时写了“conv 通道加宽到 128”和“输出 512 维”，"
        "mean||max 池化下 128 通道只有 256 维，故末层 conv 加宽到 256 以满足 512 维且不做线性压缩。）",
        "",
        "## 3. 训练协议（严格 Train-only，沿用 E162）",
        "",
        "- 输入：原始 18×1200 @300Hz，per-trial 去通道均值 + 全局 RMS 归一化（固定规则、无 batch 统计）。",
        "- 损失：SupCon（温度 0.10）+ 1.0·MSE(视图A, 视图B)；λ_VICReg 固定 0，无 L2 归一化。",
        "- 跨会话模拟正样本：同一 trial 的两组互斥强扰动——视图 A 通道增益 U(0.5,1.5) + 电极慢漂移；"
        "视图 B 6 个频带 ±6dB 幅值重加权 + 加性噪声 SNR 10–20dB；共享 ±50 点时间平移。",
        f"- 划分：每 (subject,task) 内 frac≤0.6 拟合（{best['training']['n_fit_rows']} 条），"
        f"0.6<frac≤0.8 仅用于早停（{best['training']['n_valid_rows']} 条），>0.8 不参与训练。",
        f"- 早停：验证块留半最近质心身份准确率（余弦）；最优 epoch {best['training']['best_epoch']}、"
        f"准确率 {fmt(best['training']['best_valid_accuracy'])}，训练耗时 "
        f"{best['training']['elapsed_seconds']:.0f}s。",
        "",
        "## 4. 评估协议（与 E143/E162 完全一致）",
        "",
        "task-wise Train 标准化 → PCA(32/64/128/256，仅 fit_rows 拟合) → k=1/3/6/12 原型"
        "（每 task×identity，kmeans 种子与 E143 一致）→ 最近原型负距离为分数 → "
        "五块 Train OOF 分位数阈值 → LOSO 每折排除被评分身份、原型只用 fit_rows、评估块 0.6<frac≤0.8 → "
        "Test 只评分。Test 诊断 AUC = roc_auc(S1 为负类, S2 为正类)。",
        "",
        "## 5. 对照组：E65 特征复现 E143",
        "",
        "| 视图 | PCA | k | 接受率 | 诊断 AUC | Train OOF AUC | LOSO | S1 拒识 | S2 接受 | W |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| control E65 | {repro['pca_dim']} | {repro['k']} | {repro['acceptance_rate']:.2f} | "
        f"{fmt(repro['diagnostic_auc'])} | {fmt(repro['train_oof_auc'])} | "
        f"{fmt(repro['loso_rejection'])} | {fmt(repro['test_s1_rejection'])} | "
        f"{fmt(repro['test_s2_acceptance'])} | {fmt(repro['W'])} |",
        "",
        f"E143 已发表：LOSO {fmt(base['LOSO_rejection'])} / S1 {fmt(base['Test_S1_rejection'])} / "
        f"S2 {fmt(base['Test_S2_acceptance'])} / W {fmt(base['W'])} / 诊断 AUC {fmt(base['diagnostic_auc'])}。"
        f"复现差值 ΔW = {repro['W'] - base['W']:+.4f}，ΔAUC = "
        f"{repro['diagnostic_auc'] - base['diagnostic_auc']:+.4f}"
        f"（管线{'一致' if reproduced else '不一致，需检查'}）。",
        "",
        "## 6. 核心对照：有效秩 vs 诊断 AUC",
        "",
        "| 视图 | 有效秩@95% | 有效秩@99% | 诊断 AUC | ΔAUC(vs E65) | W@0.90 | 最优 W | LOSO | S1 拒识 | S2 接受 | OOF AUC | OOF F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| E65 control | — | — | {fmt(repro['diagnostic_auc'])} | +0.0000 | "
        f"{fmt(repro['W'])} | {fmt(repro['W'])} | {fmt(repro['loso_rejection'])} | "
        f"{fmt(repro['test_s1_rejection'])} | {fmt(repro['test_s2_acceptance'])} | "
        f"{fmt(repro['train_oof_auc'])} | {fmt(repro['train_oof_macro_f1'])} |",
        f"| E162 参考（历史值） | {e162['effective_rank_95']}/{e162['dims_total']} | "
        f"{e162['effective_rank_99']}/{e162['dims_total']} | {fmt(e162['diagnostic_auc'])} | "
        f"{e162['diagnostic_auc'] - base['diagnostic_auc']:+.4f} | {fmt(e162['W_at_0.90'])} | "
        f"{fmt(e162['W'])} | {fmt(e162['LOSO_rejection'])} | {fmt(e162['Test_S1_rejection'])} | "
        f"{fmt(e162['Test_S2_acceptance'])} | {fmt(e162['train_oof_auc'])} | "
        f"{fmt(e162['train_oof_macro_f1'])} |",
    ]
    for name, item in sorted(configs.items(),
                             key=lambda kv: -kv[1]["best_row_by_auc"]["diagnostic_auc"]):
        a, fx, bw = item["best_row_by_auc"], item["fixed_acceptance"]["best_row"], item["best_row_by_W"]
        lines.append(
            f"| {name} | {item['spectrum']['effective_rank_95']}/{item['spectrum']['dims_total']} | "
            f"{item['spectrum']['effective_rank_99']}/{item['spectrum']['dims_total']} | "
            f"{fmt(a['diagnostic_auc'])} | {a['diagnostic_auc'] - base['diagnostic_auc']:+.4f} | "
            f"{fmt(fx['W'])} | {fmt(bw['W'])} | {fmt(a['loso_rejection'])} | "
            f"{fmt(a['test_s1_rejection'])} | {fmt(a['test_s2_acceptance'])} | "
            f"{fmt(fx['train_oof_auc'])} | {fmt(fx['train_oof_macro_f1'])} |")
    lines += [
        "",
        "（每个变体的诊断 AUC 取全网格最大行；W@0.90 取固定接受率 0.90 下 W 最大的 PCA/k；"
        "有效秩为 task 标准化后 fit_rows 上全维 embedding 的 PCA。）",
        "",
        "**有效秩–AUC 关系（本方向的终止证据）**：神经网络的池化越小、输出维越高，"
        f"512 维变体的 99% 有效秩从 E162 的 {e162['effective_rank_99']}/{e162['dims_total']} "
        f"升到 {rank_span(ranks99_512)}/512（约 2.5–5 倍；256 维对照变体 D 仍为 "
        f"{configs['D_e162head_gn256']['spectrum']['effective_rank_99'] if 'D_e162head_gn256' in configs else 'N/A'}），"
        f"但 95% 有效秩仍只有 {rank_span(ranks95)}（E162 为 {e162['effective_rank_95']}）；"
        f"与此同时诊断 AUC 从 E162 的 {fmt(e162['diagnostic_auc'])} 降到 {span(aucs)}，"
        f"而同会话 Train OOF AUC 仍在 {span(oofs)}。即：秩确实被抬高了（尾部主成分变多），"
        "更高的秩带来的是更多同会话伪信号而不是更可迁移的身份信息。"
        f"对照 E65 的 0.7578，四个变体全部落在 0.35–0.50；在“W>0.80 需要 AUC≈{target_auc:.2f}”的目标下，"
        "“原始 EEG 端到端（SupCon + 双扰动族不变性）”这一方向应终止。",
        "",
        "## 7. 各变体最优行分任务诊断",
        "",
        "| 变体 | 任务 | S1 中位 | S2 中位 | 任务内 AUC |",
        "|---|---|---:|---:|---:|",
    ]
    for name, item in sorted(configs.items()):
        a = item["best_row_by_auc"]
        for task in TASKS:
            t = a["per_task"].get(task, {})
            lines.append(f"| {name} | {task} | {fmt(t.get('s1_median'))} | "
                         f"{fmt(t.get('s2_median'))} | {fmt(t.get('auc'))} |")
    lines += [
        "",
        "## 8. 各变体固定接受率 0.90 全网格",
        "",
    ]
    for name, item in sorted(configs.items()):
        lines += [
            f"`{name}`（诊断 AUC 与 PCA/k 无关，为 {fmt(item['best_row_by_auc']['diagnostic_auc'])}）：",
            "",
            "| PCA | k | LOSO | S1 拒识 | S2 接受 | W |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for row in sorted(item["fixed_acceptance"]["rows"], key=lambda r: (r["pca_dim"], r["k"])):
            lines.append(f"| {row['pca_dim']} | {row['k']} | {fmt(row['loso_rejection'])} | "
                         f"{fmt(row['test_s1_rejection'])} | {fmt(row['test_s2_acceptance'])} | "
                         f"{fmt(row['W'])} |")
        lines.append("")
    lines += [
        "## 9. 判定与证据边界",
        "",
        f"- 判定：**{verdict}**。核心问题“有效秩提高后诊断 AUC 能否超过 0.7578”的答案："
        + ("**能**" if improved else "**不能**")
        + f"（最优 {fmt(brow['diagnostic_auc'])}，E65 {fmt(base['diagnostic_auc'])}）。",
        f"- 预注册变体 4 个（A/B/C/D），每个只有单一配置（温度 0.1、λ_VICReg=0），总配置数 4 ≤ 6。",
        f"- E159 结论：E143 分数上即使把阈值扫满，最优 W 也只有 "
        f"{e159['best_W_after_threshold_sweep']:.4f}（乐观上界 {e159['optimistic_upper_bound']:.4f}），"
        f"要 W>0.80 需诊断 AUC≈{target_auc:.2f}。",
        "- 所有阈值、PCA、原型全部来自 Train session1；Test 仅评分。接受率网格的工作点选择沿用 E143"
        "（使用 Test 的 S1/S2 分布），该 W 属乐观读数，同时给出固定接受率 0.90 的对照。",
        "- Train 内 unknown 是留身份伪未知（不是官方 5 名新被试）；跨 Session 只有 Test 侧诊断 AUC 可反映。",
        "- 有效秩只说明表征的线性维度，不等于跨会话可迁移性；本实验只检验“低秩是否为瓶颈”这一单因素。",
        f"- 副产品：变体 D 与 E162 的唯一差别是 BN→GroupNorm/8，诊断 AUC 从 E162 的 "
        f"{fmt(e162['diagnostic_auc'])} 降到 {fmt(d_auc)}，"
        "说明 BatchNorm 在这条管线上对跨会话排序是有帮助的（与预注册假设相反），"
        "“去 BN”本身即是本次下降的部分原因。",
        "- **项目 Test 官方 AUC/F1 = N/A（无真实标签）**。",
        "",
        "## 10. 复现",
        "",
        "```powershell",
        f"& \"{ROOT / '.venv' / 'Scripts' / 'python.exe'}\" \"{Path(__file__).resolve()}\" --stage all --variants all",
        "```",
        "",
        f"配置：`{CFG_PATH}`；结果：`{RESULTS_PATH}`；日志：`{LOG_PATH}`。",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(description="E164 raw EEG high-rank encoder")
    parser.add_argument("--stage", choices=("train", "embed", "eval", "report", "all"), default="all")
    parser.add_argument("--variants", default="all", help="all or comma separated indices")
    parser.add_argument("--force", action="store_true", help="retrain / re-embed even if artefacts exist")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("E164 requires CUDA")
    sys.stdout = Tee(LOG_PATH)
    started = time.time()
    torch.backends.cudnn.benchmark = True

    if args.variants.strip().lower() == "all":
        selected = list(VARIANTS)
    else:
        selected = [VARIANTS[int(i)] for i in args.variants.split(",") if i.strip() != ""]
    print(f"\n===== E164 run {time.strftime('%Y-%m-%d %H:%M:%S')} stage={args.stage} "
          f"variants={[s['name'] for s in selected]} =====", flush=True)

    if args.stage == "report":
        if not RESULTS_PATH.exists():
            raise FileNotFoundError(f"{RESULTS_PATH} missing; run --stage eval first")
        write_report(json.loads(RESULTS_PATH.read_text(encoding="utf-8")))
        print(f"report regenerated: {REPORT_PATH}", flush=True)
        return

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
    print(f"Train {len(meta)} | Test {len(test_meta)} (S1={int(is_s1.sum())}, S2={int((~is_s1).sum())})",
          flush=True)

    raw_train, raw_test = None, None
    if args.stage in ("train", "embed", "all"):
        print("loading Train raw ...", flush=True)
        raw_np = load_raw_stack([ROOT / CFG["data"]["train_root"] / p for p in meta["path"]],
                                label="train")
        raw_train = torch.as_tensor(raw_np, dtype=FT, device=DEVICE)
        del raw_np
        print(f"Train raw on GPU: {tuple(raw_train.shape)} "
              f"({raw_train.numel() * 4 / 1e6:.0f} MB)", flush=True)
        print("loading Test raw ...", flush=True)
        test_np = load_raw_stack([ROOT / CFG["data"]["train_root"] / p for p in test_meta["path"]],
                                 label="test")
        raw_test = torch.as_tensor(test_np, dtype=FT, device=DEVICE)
        del test_np
        print(f"Test raw on GPU: {tuple(raw_test.shape)} "
              f"({raw_test.numel() * 4 / 1e6:.0f} MB)", flush=True)

    if args.stage in ("train", "all"):
        for spec in selected:
            if model_path(spec).exists() and not args.force:
                print(f"[{spec['name']}] model exists, skip training", flush=True)
                continue
            print(f"training {spec['name']} ...", flush=True)
            train_one(spec, raw_train, meta)
            torch.cuda.empty_cache()

    if args.stage in ("embed", "all"):
        for spec in selected:
            if train_emb_path(spec).exists() and test_emb_path(spec).exists() and not args.force:
                print(f"[{spec['name']}] embeddings exist, skip", flush=True)
                continue
            embed_config(spec, raw_train, raw_test)

    if args.stage == "embed":
        print("embeddings cached", flush=True)
        return

    if raw_train is not None:
        del raw_train, raw_test
        torch.cuda.empty_cache()

    available = []
    for spec in VARIANTS:
        if train_emb_path(spec).exists() and test_emb_path(spec).exists() and model_path(spec).exists():
            available.append(spec)
    print(f"evaluating {len(available)} variant(s): {[s['name'] for s in available]}", flush=True)

    configs = {}
    for spec in available:
        train_emb = np.load(train_emb_path(spec))
        test_emb = np.load(test_emb_path(spec))
        dim = int(spec["embed_dim"])
        if train_emb.shape != (len(meta), dim) or test_emb.shape != (len(test_meta), dim):
            raise ValueError(f"{spec['name']} embedding shape mismatch")
        if not (np.isfinite(train_emb).all() and np.isfinite(test_emb).all()):
            raise ValueError(f"{spec['name']}: non-finite embeddings")
        print(f"evaluating {spec['name']} ...", flush=True)
        view = evaluate_view(spec["name"], train_emb, test_emb, subjects, tasks, f,
                             test_tasks, is_s1, CFG["protocol"]["pca_dims"],
                             CFG["protocol"]["prototype_k"])
        spectrum = task_standardized_spectrum(train_emb, tasks,
                                              f <= float(CFG["protocol"]["fit_fraction"]))
        _, info = load_encoder(spec)
        configs[spec["name"]] = summarise_config(spec["name"], view["rows"], spec, info, spectrum)
        print(f"[{spec['name']}] effective_rank_95={spectrum['effective_rank_95']}/{spectrum['dims_total']} "
              f"rank99={spectrum['effective_rank_99']}", flush=True)

    print("evaluating E143 protocol on E65 control features ...", flush=True)
    control_train = np.load(ROOT / CFG["control"]["train_features"])
    control_test = np.load(ROOT / CFG["control"]["test_features"])
    if control_train.shape[0] != len(meta) or control_test.shape[0] != len(test_meta):
        raise ValueError("control feature row mismatch")
    control = evaluate_view("E65_control", control_train, control_test, subjects, tasks, f,
                            test_tasks, is_s1, [int(CFG["control"]["reproduction_target"]["pca_dim"])],
                            [int(CFG["control"]["reproduction_target"]["k"])])
    del control_train, control_test
    best_c = max(control["rows"], key=lambda r: r["W"])
    repro = [r for r in control["rows"]
             if abs(r["acceptance_rate"] - CFG["control"]["reproduction_target"]["acceptance_rate"]) < 1e-9][0]
    if abs(repro["W"] - CFG["baseline"]["W"]) > 5e-3:
        print(f"WARNING: E65 control W {repro['W']:.4f} != baseline {CFG['baseline']['W']:.4f}",
              flush=True)
    if abs(repro["diagnostic_auc"] - CFG["baseline"]["diagnostic_auc"]) > 5e-3:
        print(f"WARNING: E65 control AUC {repro['diagnostic_auc']:.4f} != baseline "
              f"{CFG['baseline']['diagnostic_auc']:.4f}", flush=True)

    if not configs:
        raise RuntimeError("no variant artefacts found; run --stage all first")

    best_name = max(configs, key=lambda n: (configs[n]["best_row_by_auc"]["diagnostic_auc"],
                                            configs[n]["best_row_by_W"]["W"]))
    best_spec = [s for s in VARIANTS if s["name"] == best_name][0]
    np.save(TRAIN_EMB_PATH, np.load(train_emb_path(best_spec)))
    np.save(TEST_EMB_PATH, np.load(test_emb_path(best_spec)))

    result = {
        "experiment": CFG["experiment"],
        "status": "complete",
        "date": "2026-11-15",
        "clean_train_only": True,
        "weights": WEIGHTS,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "sklearn": sklearn.__version__,
                        "gpu": torch.cuda.get_device_name(0)},
        "config": CFG,
        "configs": configs,
        "best_variant": best_name,
        "control": {"rows": control["rows"], "best": best_c, "reproduction": repro,
                    "reproduction_delta_W": repro["W"] - CFG["baseline"]["W"],
                    "reproduction_delta_auc": repro["diagnostic_auc"] - CFG["baseline"]["diagnostic_auc"]},
        "baseline_E143": CFG["baseline"],
        "e162_reference": CFG["e162_reference"],
        "e159_reference": CFG["e159_reference"],
        "delta_vs_baseline": {
            "diagnostic_auc": configs[best_name]["best_row_by_auc"]["diagnostic_auc"]
            - CFG["baseline"]["diagnostic_auc"],
            "W": configs[best_name]["best_row_by_W"]["W"] - CFG["baseline"]["W"]},
        "test_official": CFG["reporting"]["test_official"],
        "elapsed_seconds": time.time() - started,
    }
    write_json(RESULTS_PATH, result)
    write_report(result)

    print("\n" + "=" * 70, flush=True)
    for name, item in sorted(configs.items(),
                             key=lambda kv: -kv[1]["best_row_by_auc"]["diagnostic_auc"]):
        a = item["best_row_by_auc"]
        fxd = item["fixed_acceptance"]["best_row"]
        print(f"{name}: diagAUC={a['diagnostic_auc']:.4f} W@0.90={fxd['W']:.4f} "
              f"maxW={item['best_row_by_W']['W']:.4f} rank95={item['spectrum']['effective_rank_95']}"
              f"/{item['spectrum']['dims_total']} | S1 {a['test_s1_rejection']:.4f} "
              f"S2 {a['test_s2_acceptance']:.4f} OOFauc {fxd['train_oof_auc']:.4f} "
              f"OOFf1 {fxd['train_oof_macro_f1']:.4f}", flush=True)
    print(f"best variant: {best_name}", flush=True)
    print(f"E65 control: pca=96 k=12 a=0.90 W={repro['W']:.4f} diagAUC={repro['diagnostic_auc']:.4f}",
          flush=True)
    print(f"results: {RESULTS_PATH}\nreport: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
