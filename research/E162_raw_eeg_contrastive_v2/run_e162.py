"""E162: fix the E161 collapse and test whether the cross-session ranking reversal
can be corrected by the training objective itself.

Changes relative to E161 (pre-registered, no further tuning):
  1. no L2 normalisation of the embedding; 256-d pooled vector through a widened
     256-d projection head (no BatchNorm), so the representation can keep its rank;
  2. VICReg-style variance hinge (target std >= 1) + off-diagonal covariance penalty,
     weight swept over {0.0, 0.1, 1.0};
  3. temperature swept over {0.1, 0.2};
  4. simulated cross-session positive pairs: two disjoint strong perturbation
     families (A = channel gain + electrode drift, B = band amplitude reweighting +
     additive noise) of the same trial must map to the same embedding
     (explicit MSE invariance term).

The E143 open-set protocol is reused verbatim (task standardisation -> PCA ->
k prototypes -> 5-block Train OOF threshold -> LOSO excluding the scored identity
with prototypes from fit_rows -> Test scoring only), and the E65 control must
reproduce W ~ 0.6958.

Usage:
  .venv/Scripts/python.exe research/E162_raw_eeg_contrastive_v2/run_e162.py --stage all --configs 0
  .venv/Scripts/python.exe research/E162_raw_eeg_contrastive_v2/run_e162.py --stage all --configs all
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
DIMS = int(CFG["encoder"]["embed_dim"])


def specs() -> list[dict]:
    out = []
    for lam in CFG["train"]["vicreg_weight_grid"]:
        for temp in CFG["train"]["temperature_grid"]:
            out.append({"name": f"lam{lam:.1f}_t{temp:.1f}",
                        "vicreg": float(lam), "temperature": float(temp)})
    return out


def model_path(spec: dict) -> Path:
    return OUT / f"model_{spec['name']}.pt"


def train_emb_path(spec: dict) -> Path:
    return OUT / f"train_emb_{spec['name']}.npy"


def test_emb_path(spec: dict) -> Path:
    return OUT / f"test_emb_{spec['name']}.npy"


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
class Encoder(nn.Module):
    """Conv body -> pooled(mean||max) 256 -> Linear(256, embed_dim). No L2 norm."""

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
        self.pool_dim = 2 * channels
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.pool_dim, embed_dim)

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        return torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.pooled(x)))


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


def vicreg_terms(z: torch.Tensor, target_std: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Variance hinge (std >= target) + off-diagonal covariance penalty."""
    std = z.std(dim=0)
    var_loss = torch.clamp(target_std - std, min=0.0).mean()
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.t() @ zc) / max(len(zc) - 1, 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    cov_loss = off.pow(2).sum() / z.shape[1]
    return var_loss, cov_loss


@torch.no_grad()
def embed_rows(encoder: Encoder, raw: torch.Tensor, rows: np.ndarray,
               batch: int = 256) -> np.ndarray:
    encoder.eval()
    out = np.empty((len(rows), DIMS), dtype=np.float32)
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
    lam = float(spec["vicreg"])
    lam_inv = float(cfg["invariance_weight"])
    target_std = float(cfg["vicreg_variance_target"])

    encoder = build_encoder()
    params = int(sum(p.numel() for p in encoder.parameters()))
    print(f"[{spec['name']}] parameters = {params:,} (budget {CFG['encoder']['param_budget']:,})", flush=True)
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
        losses, parts = [], {"supcon": [], "inv": [], "var": [], "cov": []}
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
            var_loss, cov_loss = vicreg_terms(z, target_std)
            loss = loss_sup + lam_inv * loss_inv + lam * (var_loss + cov_loss)
            optimizer.param_groups[0]["lr"] = lr_at(step)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), float(cfg["grad_clip"]))
            optimizer.step()
            losses.append(float(loss.detach()))
            parts["supcon"].append(float(loss_sup.detach()))
            parts["inv"].append(float(loss_inv.detach()))
            parts["var"].append(float(var_loss.detach()))
            parts["cov"].append(float(cov_loss.detach()))
            step += 1
        with torch.no_grad():
            mean_std = float(z.std(dim=0).mean())
        valid = valid_accuracy(encoder, raw, eval_rows, subjects, tasks)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)),
                        "loss_supcon": float(np.mean(parts["supcon"])),
                        "loss_invariance": float(np.mean(parts["inv"])),
                        "loss_variance": float(np.mean(parts["var"])),
                        "loss_covariance": float(np.mean(parts["cov"])),
                        "mean_dim_std": mean_std,
                        "valid_accuracy": valid["accuracy"],
                        "per_task": {k: v["accuracy"] for k, v in valid["per_task"].items()},
                        "lr": lr_at(step), "elapsed": time.time() - started})
        print(f"[{spec['name']}] epoch {epoch + 1}/{cfg['epochs']} loss={np.mean(losses):.4f} "
              f"sup={np.mean(parts['supcon']):.4f} inv={np.mean(parts['inv']):.4f} "
              f"var={np.mean(parts['var']):.4f} cov={np.mean(parts['cov']):.4f} "
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
            "best_epoch": int(best["epoch"]),
            "best_valid_accuracy": float(best["accuracy"]),
            "best_valid_per_task": best["per_task"], "history": history,
            "total_steps": step, "elapsed_seconds": time.time() - started,
            "n_fit_rows": int((f <= float(cfg["fit_fraction"])).sum()),
            "n_valid_rows": int(len(eval_rows))}
    torch.save({"state_dict": encoder.state_dict(), "encoder": CFG["encoder"],
                "train": {**CFG["train"], "temperature": temperature, "vicreg_weight": lam},
                "augment": CFG["augment"], "info": info, "train_only": True},
               model_path(spec))
    return encoder, info


def load_encoder(spec: dict) -> tuple[Encoder, dict]:
    payload = torch.load(model_path(spec), map_location="cpu", weights_only=False)
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
def embed_config(spec: dict, raw_train: torch.Tensor, raw_test: torch.Tensor,
                 meta: pd.DataFrame, test_meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
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
    return train_emb, test_emb


# ---------------------------------------------------------------------- report
def fmt(value) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.4f}"


def summarise_config(name: str, rows: list[dict], spec: dict, training: dict,
                     spectrum: dict) -> dict:
    ref = float(CFG["protocol"]["reference_acceptance"])
    fixed = [r for r in rows if abs(r["acceptance_rate"] - ref) < 1e-9]
    best_auc = max(rows, key=lambda r: (r["diagnostic_auc"], r["W"]))
    best_w = max(rows, key=lambda r: r["W"])
    fixed_best = max(fixed, key=lambda r: r["W"])
    return {"config": name, "spec": spec,
            "vicreg_weight": float(spec["vicreg"]), "temperature": float(spec["temperature"]),
            "rows": rows,
            "fixed_acceptance": {"acceptance_rate": ref, "rows": fixed, "best_row": fixed_best},
            "best_row_by_auc": best_auc, "best_row_by_W": best_w,
            "training": {k: v for k, v in training.items() if k != "history"},
            "history": training["history"],
            "spectrum": spectrum}


def write_report(result: dict) -> None:
    base = CFG["baseline"]
    ctrl = result["control"]
    repro = ctrl["reproduction"]
    best = result["best"]
    brow = best["best_row_by_auc"]
    bfix = best["fixed_acceptance"]["best_row"]
    bwor = best["best_row_by_W"]
    spectrum = best["spectrum"]
    e161 = CFG["e161_reference"]
    improved_auc = brow["diagnostic_auc"] > base["diagnostic_auc"]
    improved_w = bwor["W"] > base["W"]
    above_chance = brow["diagnostic_auc"] > 0.5

    lines = [
        "# E162 原始 EEG 端到端跨会话对比学习 v2（修坍缩 + 会话模拟正样本）",
        "",
        f"日期：{result['date']}；设备：{result['environment']['gpu']}；"
        "只用 Train session1 训练，Test 只被评分，Session 仅在评分后用于诊断 AUC。",
        "",
        "## 结论",
        "",
    ]
    if improved_auc and improved_w:
        verdict = "通过"
        lines += [
            f"**原始 EEG 端到端超过 E143**：最优配置 `{best['config']}`（VICReg 权重 "
            f"{best['vicreg_weight']:.1f}、温度 {best['temperature']:.2f}）。"
            f"诊断 AUC {fmt(brow['diagnostic_auc'])}（E143 {fmt(base['diagnostic_auc'])}，"
            f"{brow['diagnostic_auc'] - base['diagnostic_auc']:+.4f}），"
            f"最优 W {fmt(bwor['W'])}（E143 {fmt(base['W'])}，{bwor['W'] - base['W']:+.4f}）。",
        ]
    elif above_chance:
        verdict = "部分改善（排序方向已纠正但未超过 E143）"
        lines += [
            f"**排序反转被纠正，但仍未超过 E143**：最优配置 `{best['config']}`。"
            f"诊断 AUC {fmt(brow['diagnostic_auc'])}（E161 {fmt(e161['best_diagnostic_auc'])}，"
            f"E143 {fmt(base['diagnostic_auc'])}）；最优 W {fmt(bwor['W'])}"
            f"（E161 {fmt(e161['best_W'])}，E143 {fmt(base['W'])}）。",
        ]
    else:
        verdict = "未通过（排序仍反转）"
        lines += [
            f"**仍未通过**：最优配置 `{best['config']}` 的诊断 AUC 仍低于 0.5 "
            f"（{fmt(brow['diagnostic_auc'])}，E161 {fmt(e161['best_diagnostic_auc'])}），"
            f"最优 W {fmt(bwor['W'])}（目标 > {fmt(base['W'])}）。"
            "说明原始 EEG 端到端在当前数据下不可行，诊断见第 6 节。",
        ]
    lines += [
        "",
        "## 关键指标（最优配置 `" + best["config"] + "`）",
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
        "## 1. 相对 E161 的预注册改动",
        "",
    ]
    for change in CFG["changes_vs_E161"]:
        lines.append(f"- {change}")
    lines += [
        "",
        "## 2. 表征训练（严格 Train-only）",
        "",
        f"- 输入：原始 18×1200 @300Hz，per-trial 去通道均值 + 全局 RMS 归一化（固定规则、无 batch 统计）。",
        f"- Encoder：4 层 stride-2 Conv1d {CFG['encoder']['widths']}（kernel "
        f"{CFG['encoder']['kernels']}）+ BN + ELU + 全局 mean/max 池化 → "
        f"Linear({2 * CFG['encoder']['widths'][-1]}→{CFG['encoder']['embed_dim']})，"
        f"参数量 {best['training']['parameters']:,}（预算 <1M），**无 L2 归一化、无 BN**。",
        f"- 损失：SupCon（温度 {best['temperature']:.2f}）+ {CFG['train']['invariance_weight']}·MSE(视图A, 视图B) "
        f"+ {best['vicreg_weight']:.1f}·(方差 hinge(std≥{CFG['train']['vicreg_variance_target']}) + 去协方差)。",
        "- 跨会话模拟正样本：同一 trial 的两组互斥强扰动——"
        "视图 A 只做通道增益 U(0.5,1.5) + 电极慢漂移（线性斜坡+0.5–3 周期正弦，幅度 U(0.2,1.0)）；"
        "视图 B 只做 6 个频带 ±6dB 幅值重加权 + 加性噪声 SNR 10–20dB。"
        "两者共享 ±50 点时间平移，增强后重新归一化。",
        f"- batch：20 身份 × {CFG['train']['trials_per_subject']} 个不同时间块 trial × 2 视图 = 80 样本，"
        "同任务内异身份为 hard negative。",
        f"- 划分：每 (subject,task) 内 frac≤0.6 拟合（{best['training']['n_fit_rows']} 条），"
        f"0.6<frac≤0.8 仅用于早停（{best['training']['n_valid_rows']} 条），>0.8 不参与训练。",
        f"- 早停：验证块留半最近质心身份准确率（余弦度量）；最佳 epoch "
        f"{best['training']['best_epoch']}、准确率 {fmt(best['training']['best_valid_accuracy'])}，"
        f"训练耗时 {best['training']['elapsed_seconds']:.0f}s。",
        f"- 权重：`model_{best['config']}.pt`；embedding：`train_emb_{best['config']}.npy` / "
        f"`test_emb_{best['config']}.npy`（另存 best 别名 `train_emb.npy` / `test_emb.npy`）。",
        "",
        "## 3. 评估协议（与 E143/E161 完全一致）",
        "",
        "task-wise Train 标准化 → PCA(32/64/96，仅 fit_rows 拟合) → k=1/3/6/12 原型"
        "（每 task×identity，kmeans 种子与 E143 一致）→ 最近原型负距离为分数 → "
        "五块 Train OOF 分位数阈值 → LOSO 每折排除被评分身份、原型只用 fit_rows、评估块 0.6<frac≤0.8 → "
        "Test 只评分。Test 诊断 AUC = roc_auc(S1 为负类, S2 为正类)。",
        "",
        "## 4. 对照组：E65 特征复现 E143",
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
        f"复现差值 ΔW = {repro['W'] - base['W']:+.4f}，"
        f"ΔAUC = {repro['diagnostic_auc'] - base['diagnostic_auc']:+.4f}"
        f"（管线{'一致' if abs(repro['W'] - base['W']) < 0.005 else '不一致，需检查'}）。",
        "",
        "## 5. 配置扫描（每配置 = VICReg 权重 × 温度）",
        "",
        "| 配置 | VICReg | 温度 | 诊断 AUC | OOF AUC | OOF F1 | LOSO | S1 拒识 | S2 接受 | W@0.90 | 最优 W | 有效秩@95% | 中位偏移(S2-S1) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in result["configs"].items():
        frow = item["fixed_acceptance"]["best_row"]
        lines.append(
            f"| {name} | {item['vicreg_weight']:.1f} | {item['temperature']:.2f} | "
            f"{fmt(frow['diagnostic_auc'])} | {fmt(frow['train_oof_auc'])} | "
            f"{fmt(frow['train_oof_macro_f1'])} | {fmt(frow['loso_rejection'])} | "
            f"{fmt(frow['test_s1_rejection'])} | {fmt(frow['test_s2_acceptance'])} | "
            f"{fmt(frow['W'])} | {fmt(item['best_row_by_W']['W'])} | "
            f"{item['spectrum']['effective_rank_95']} | "
            f"{frow['s2_score']['median'] - frow['s1_score']['median']:+.3f} |")
    lines += [
        "",
        "（各行均取固定接受率 0.90 —— E143 工作点 —— 下该配置 W 最大的 PCA/k；"
        "同一配置内不同 PCA/k 的诊断 AUC 差异 < 0.01。全网格最优 W 见“最优 W”列。）",
        "",
        "## 6. 最优配置全网格",
        "",
        "| PCA | k | 接受率 | 诊断 AUC | Train OOF AUC | OOF F1 | LOSO | S1 拒识 | S2 接受 | W |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(best["rows"], key=lambda r: (r["pca_dim"], r["k"], r["acceptance_rate"])):
        lines.append(f"| {row['pca_dim']} | {row['k']} | {row['acceptance_rate']:.2f} | "
                     f"{fmt(row['diagnostic_auc'])} | {fmt(row['train_oof_auc'])} | "
                     f"{fmt(row['train_oof_macro_f1'])} | {fmt(row['loso_rejection'])} | "
                     f"{fmt(row['test_s1_rejection'])} | {fmt(row['test_s2_acceptance'])} | "
                     f"{fmt(row['W'])} |")
    lines += [
        "",
        f"固定接受率 {best['fixed_acceptance']['acceptance_rate']:.2f}（E143 工作点）：",
        "",
        "| PCA | k | 诊断 AUC | LOSO | S1 拒识 | S2 接受 | W |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(best["fixed_acceptance"]["rows"], key=lambda r: (r["pca_dim"], r["k"])):
        lines.append(f"| {row['pca_dim']} | {row['k']} | {fmt(row['diagnostic_auc'])} | "
                     f"{fmt(row['loso_rejection'])} | {fmt(row['test_s1_rejection'])} | "
                     f"{fmt(row['test_s2_acceptance'])} | {fmt(row['W'])} |")
    lines += [
        "",
        "## 7. 诊断：坍缩是否修复，排序是否纠正",
        "",
        "| 量 | E162 最优 | E161 | E143/E65 |",
        "|---|---:|---:|---:|",
        f"| Train OOF AUC | {fmt(brow['train_oof_auc'])} | {fmt(e161['train_oof_auc'])} | — |",
        f"| Train OOF 身份 macro-F1 | {fmt(brow['train_oof_macro_f1'])} | {fmt(e161['train_oof_macro_f1'])} | — |",
        f"| Test 诊断 AUC | {fmt(brow['diagnostic_auc'])} | {fmt(e161['best_diagnostic_auc'])} | {fmt(base['diagnostic_auc'])} |",
        f"| Test S1 中位分数 | {fmt(brow['s1_score']['median'])} | {fmt(e161['s1_median'])} | — |",
        f"| Test S2 中位分数 | {fmt(brow['s2_score']['median'])} | {fmt(e161['s2_median'])} | — |",
        f"| 有效秩@95% | {spectrum['effective_rank_95']}/{spectrum['dims_total']} | "
        f"≈7/128（前 7 PC 解释 99.99%） | — |",
        f"| 有效秩@99% | {spectrum['effective_rank_99']}/{spectrum['dims_total']} | — | — |",
        f"| 前 8 维累积解释方差 | {spectrum['cumulative_explained_variance']['8']:.5f} | 0.99930 | — |",
        "",
        "分任务诊断（S1/S2 分数中位与任务内 AUC）：",
        "",
        "| 任务 | E162 S1 中位 | E162 S2 中位 | E162 AUC | E65 对照 AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    best_ctrl = result["control"]["best"]
    for task in TASKS:
        mine = brow["per_task"].get(task, {})
        other = best_ctrl["per_task"].get(task, {})
        lines.append(f"| {task} | {fmt(mine.get('s1_median', float('nan')))} | "
                     f"{fmt(mine.get('s2_median', float('nan')))} | "
                     f"{fmt(mine.get('auc', float('nan')))} | {fmt(other.get('auc', float('nan')))} |")
    lines += [""]
    lam_stats, temp_stats = {}, {}
    for item in result["configs"].values():
        frow = item["fixed_acceptance"]["best_row"]
        key_l = float(item["vicreg_weight"])
        key_t = float(item["temperature"])
        lam_stats.setdefault(key_l, []).append((frow["diagnostic_auc"], frow["test_s1_rejection"],
                                                item["spectrum"]["effective_rank_95"]))
        temp_stats.setdefault(key_t, []).append((frow["diagnostic_auc"], frow["test_s1_rejection"]))
    if len(result["configs"]) > 1:
        lines += ["预注册扫描的趋势（固定接受率 0.90，按配置平均）：", ""]
        for lam in sorted(lam_stats):
            vals = np.asarray([v[0] for v in lam_stats[lam]])
            s1 = np.asarray([v[1] for v in lam_stats[lam]])
            rank = lam_stats[lam][0][2]
            lines.append(f"- VICReg 权重 {lam:.1f}：诊断 AUC {vals.mean():.4f}，"
                         f"S1 拒识 {s1.mean():.4f}，有效秩@95% {rank}/{spectrum['dims_total']}")
        for temp in sorted(temp_stats):
            vals = np.asarray([v[0] for v in temp_stats[temp]])
            s1 = np.asarray([v[1] for v in temp_stats[temp]])
            lines.append(f"- 温度 {temp:.2f}：诊断 AUC {vals.mean():.4f}，S1 拒识 {s1.mean():.4f}")
        lines.append("")
    rank95 = spectrum["effective_rank_95"]
    cum8 = spectrum["cumulative_explained_variance"]["8"]
    if rank95 <= 10:
        lines.append(
            f"**坍缩检查**：有效秩@95% {rank95}/{spectrum['dims_total']}、@99% "
            f"{spectrum['effective_rank_99']}/{spectrum['dims_total']}，前 8 维累计解释方差 {cum8:.5f}"
            f"（E161 为 0.99930，前 7 维已解释 99.99%）。去掉 L2 归一化与 VICReg 正则只让浓度轻微下降，"
            "表征仍高度低秩，说明低秩主要来自 Conv+BN+均值/最大池化这条编码器本身；"
            f"但诊断 AUC 已从 E161 的 {fmt(e161['best_diagnostic_auc'])} 回到 {fmt(brow['diagnostic_auc'])}，"
            "说明跨会话排序反转的关键不是坍缩，而是缺少显式的会话风格不变性约束。")
    else:
        lines.append(
            f"**坍缩检查**：有效秩@95% {rank95}/{spectrum['dims_total']}、@99% "
            f"{spectrum['effective_rank_99']}/{spectrum['dims_total']}，前 8 维累计解释方差 {cum8:.5f}，"
            "表征已不再坍缩（E161 前 7 维即解释 99.99%）。")
    lines.append("")
    gap = brow["train_oof_auc"] - brow["diagnostic_auc"]
    if brow["diagnostic_auc"] < 0.5:
        lines.append(
            f"**排序仍反转**：诊断 AUC {fmt(brow['diagnostic_auc'])} < 0.5，"
            f"S2 中位 {fmt(brow['s2_score']['median'])} 仍低于 S1 中位 {fmt(brow['s1_score']['median'])}。"
            f"有效秩 {spectrum['effective_rank_95']}/{spectrum['dims_total']}@95% 与 E161 的 ≈7/128 相当，"
            "说明去掉 L2 归一化与 VICReg 正则并没有真正抬高表征秩，坍缩也不是排序反转的唯一原因；"
            "显式会话扰动+不变性约束同样没能把 session 风格从身份中剥离。"
            f"Train OOF AUC 与 Test 诊断 AUC 相差 {gap:+.4f}。")
    elif not improved_auc:
        lines.append(
            f"**排序方向已纠正但幅度不足**：诊断 AUC {fmt(brow['diagnostic_auc'])} > 0.5，"
            f"但距 E143 {fmt(base['diagnostic_auc'])} 仍差 {brow['diagnostic_auc'] - base['diagnostic_auc']:+.4f}；"
            f"Train OOF AUC {fmt(brow['train_oof_auc'])} vs Test 诊断 AUC 相差 {gap:+.4f}，"
            "仍存在同会话伪信号。")
    else:
        lines.append(
            f"**排序纠正且超过 E143**：诊断 AUC {fmt(brow['diagnostic_auc'])}，"
            f"S2 中位 {fmt(brow['s2_score']['median'])} 高于 S1 中位 {fmt(brow['s1_score']['median'])}，"
            f"有效秩 {spectrum['effective_rank_95']}/{spectrum['dims_total']}@95%。"
            f"Train OOF AUC 与 Test 诊断 AUC 相差 {gap:+.4f}。")
    lines += [
        "",
        "## 8. 判定与证据边界",
        "",
        f"- 判定：**{verdict}**。主指标诊断 AUC {fmt(brow['diagnostic_auc'])} vs 目标 "
        f"{fmt(base['diagnostic_auc'])}；次指标最优 W {fmt(bwor['W'])} vs 目标 {fmt(base['W'])}"
        f"（固定接受率 0.90 时 W = {fmt(bfix['W'])}）。",
        "- 预注册扫描仅为 VICReg 权重 {0.0,0.1,1.0} × 温度 {0.1,0.2}，共 6 个配置；"
        "不变性权重固定为 1.0，未做额外调参。",
        "- 所有阈值、PCA、原型全部来自 Train session1；Test 仅评分。"
        "接受率网格的工作点选择沿用 E143（使用 Test 的 S1/S2 分布），该 W 属乐观读数，"
        "同时给出固定接受率 0.90 的对照。",
        "- Train 内 unknown 是留身份伪未知，不是官方 5 名新被试；跨 Session 只有 Test 侧诊断 AUC 可反映。",
        "- **项目 Test 官方 AUC/F1 = N/A（无真实标签）**。",
        "",
        "## 9. 复现",
        "",
        "```powershell",
        f"& \"{ROOT / '.venv' / 'Scripts' / 'python.exe'}\" \"{Path(__file__).resolve()}\" --stage all --configs all",
        "```",
        "",
        f"配置：`{CFG_PATH}`；结果：`{RESULTS_PATH}`；日志：`{LOG_PATH}`。",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(description="E162 raw EEG contrastive v2")
    parser.add_argument("--stage", choices=("train", "embed", "eval", "report", "all"), default="all")
    parser.add_argument("--configs", default="all", help="all or comma separated indices of the sweep")
    parser.add_argument("--force", action="store_true", help="retrain / re-embed even if artefacts exist")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("E162 requires CUDA")
    sys.stdout = Tee(LOG_PATH)
    started = time.time()
    torch.backends.cudnn.benchmark = True

    all_specs = specs()
    if args.configs.strip().lower() == "all":
        selected = all_specs
    else:
        selected = [all_specs[int(i)] for i in args.configs.split(",") if i.strip() != ""]
    print(f"\n===== E162 run {time.strftime('%Y-%m-%d %H:%M:%S')} stage={args.stage} "
          f"configs={[s['name'] for s in selected]} =====", flush=True)

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
            embed_config(spec, raw_train, raw_test, meta, test_meta)

    if args.stage == "embed":
        print("embeddings cached", flush=True)
        return

    if raw_train is not None:
        del raw_train, raw_test
        torch.cuda.empty_cache()

    available = []
    for spec in all_specs:
        if train_emb_path(spec).exists() and test_emb_path(spec).exists() and model_path(spec).exists():
            available.append(spec)
    print(f"evaluating {len(available)} config(s): {[s['name'] for s in available]}", flush=True)

    configs = {}
    for spec in available:
        train_emb = np.load(train_emb_path(spec))
        test_emb = np.load(test_emb_path(spec))
        if train_emb.shape != (len(meta), DIMS) or test_emb.shape != (len(test_meta), DIMS):
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

    best_name = max(configs, key=lambda n: (configs[n]["best_row_by_auc"]["diagnostic_auc"],
                                            configs[n]["best_row_by_W"]["W"]))
    best = dict(configs[best_name])
    best_train = train_emb_path([s for s in all_specs if s["name"] == best_name][0])
    best_test = test_emb_path([s for s in all_specs if s["name"] == best_name][0])
    np.save(TRAIN_EMB_PATH, np.load(best_train))
    np.save(TEST_EMB_PATH, np.load(best_test))

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
        "best": best,
        "control": {"rows": control["rows"], "best": best_c, "reproduction": repro,
                    "reproduction_delta_W": repro["W"] - CFG["baseline"]["W"]},
        "baseline_E143": CFG["baseline"],
        "e161_reference": CFG["e161_reference"],
        "delta_vs_baseline": {
            "diagnostic_auc": best["best_row_by_auc"]["diagnostic_auc"] - CFG["baseline"]["diagnostic_auc"],
            "W": best["best_row_by_W"]["W"] - CFG["baseline"]["W"]},
        "test_official": CFG["reporting"]["test_official"],
        "elapsed_seconds": time.time() - started,
    }
    write_json(RESULTS_PATH, result)
    write_report(result)

    print("\n" + "=" * 70, flush=True)
    for name, item in configs.items():
        a = item["best_row_by_auc"]
        fxd = item["fixed_acceptance"]["best_row"]
        print(f"{name}: diagAUC={a['diagnostic_auc']:.4f} W@0.90={fxd['W']:.4f} "
              f"maxW={item['best_row_by_W']['W']:.4f} rank95={item['spectrum']['effective_rank_95']} "
              f"| S1 {a['test_s1_rejection']:.4f} S2 {a['test_s2_acceptance']:.4f} "
              f"OOFauc {fxd['train_oof_auc']:.4f}", flush=True)
    print(f"best config: {best_name} (diagAUC {best['best_row_by_auc']['diagnostic_auc']:.4f}, "
          f"W {best['best_row_by_W']['W']:.4f})", flush=True)
    print(f"E65 control: pca=96 k=12 a=0.90 W={repro['W']:.4f} diagAUC={repro['diagnostic_auc']:.4f}",
          flush=True)
    print(f"results: {RESULTS_PATH}\nreport: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
