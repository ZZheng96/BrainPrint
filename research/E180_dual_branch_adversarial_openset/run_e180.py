"""E180: dual-branch system (adversarially trained CNN backbone + E65 open-set gate).

Branch A  : 18x1200 raw EEG -> EEGNet-style 1D CNN -> 20 identity logits.
            Trained on Train session1 only, PGD-3 L-infinity adversarial training
            (eps = 0.02 * per-trial global RMS) + augmentation.
Branch B  : E65 correlation-logSPD -> per-task z-score -> PCA-96 (fit_rows) ->
            k=12 prototypes per (task, identity) from fit_rows -> score = -nearest.

Fusion    : gate rejects (score < tau) -> label 0, else branch A argmax.

Stages    : a (branch A), b (branch B), test (frozen Test scoring + S1/S2 audit),
            attack (#7/#8 end-to-end differentiable PGD), all, report.

Test is scored only after every Train-only quantity is frozen. The Session column
is read only after the Test predictions have been written, for the #9/#10 audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn.metrics import f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda")
FT = torch.float32
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "report.md"
STAGE_A = CACHE / "stage_a.pt"
STAGE_B = CACHE / "stage_b.npz"
MODEL_A = OUT / "model_branch_a.pt"

TASKS = list(CFG["tasks"])
CLASSES = np.arange(1, 21, dtype=int)
N = int(CFG["data"]["samples_per_trial"])
CH = int(CFG["data"]["channels"])
FS = int(CFG["data"]["sampling_rate_hz"])
NB = len(CFG["branch_b"]["bands_hz"])
SEED = int(CFG["protocol"]["seed"])
OFF = -1e9


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


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


CONFIG_SHA = sha256(OUT / "config.json")


def spec_hash(*sections: str) -> str:
    """Hash of the config sections one artifact actually depends on."""
    payload = json.dumps({key: CFG[key] for key in sections}, sort_keys=True,
                         ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


BRANCH_A_SHA = spec_hash("data", "protocol", "branch_a")
BRANCH_B_SHA = spec_hash("data", "protocol", "branch_b")


def load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {"experiment": CFG["experiment"], "config_sha256": CONFIG_SHA}


def save_results(results: dict) -> None:
    RESULTS_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def f32(array) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


# ---------------------------------------------------------------- data / protocol
def load_config_meta() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    meta = pd.read_csv(ROOT / CFG["data"]["train_metadata"])
    if len(meta) != int(CFG["data"]["train_trials"]):
        raise ValueError(f"Train rows {len(meta)} != {CFG['data']['train_trials']}")
    if meta["subject"].nunique() != 20 or set(meta["session"]) != {"session1"}:
        raise ValueError("Train must be 20 subjects, session1 only")
    if not meta["path"].astype(str).str.startswith("Train/").all():
        raise ValueError("non-Train path in metadata")
    maximum = meta.groupby(["subject", "task"])["trial"].transform("max").to_numpy(float)
    frac = meta["trial"].to_numpy(float) / np.maximum(maximum, 1.0)
    return meta, frac, np.clip((frac * CFG["protocol"]["blocks"]).astype(int), 0,
                               CFG["protocol"]["blocks"] - 1), frac <= CFG["protocol"]["fit_fraction"]


def load_mat(path: Path) -> np.ndarray:
    data = np.asarray(loadmat(path, variable_names=["data"])["data"], dtype=np.float32)
    if data.shape != (N, CH):
        raise ValueError(f"{path}: expected ({N},{CH}) got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: non-finite")
    return np.ascontiguousarray(data.T)


def load_raw(relative_paths, root_key: str = "train_root") -> torch.Tensor:
    base = ROOT / CFG["data"][root_key]
    paths = [base / str(p) for p in relative_paths]
    out = torch.empty((len(paths), CH, N), dtype=FT, device=DEVICE)
    started = time.time()
    with ThreadPoolExecutor(max_workers=12) as pool:
        for start in range(0, len(paths), 512):
            chunk = paths[start:start + 512]
            arrays = list(pool.map(load_mat, chunk))
            out[start:start + len(arrays)] = torch.as_tensor(
                np.stack(arrays), dtype=FT, device=DEVICE)
            done = start + len(arrays)
            if done % 4096 == 0 or done == len(paths):
                log(f"  raw {done}/{len(paths)} ({time.time() - started:.0f}s)")
    return out


# ---------------------------------------------------------------- branch A model
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        cfg = CFG["branch_a"]
        widths = [int(w) for w in cfg["widths"]]
        kernel = int(cfg["kernel"])
        blocks, channels = [], CH
        for width in widths:
            blocks += [nn.Conv1d(channels, width, kernel, stride=int(cfg["stride"]),
                                 padding=kernel // 2, bias=False),
                       nn.BatchNorm1d(width), nn.ELU(inplace=True)]
            channels = width
        self.body = nn.Sequential(*blocks)
        self.drop = nn.Dropout(float(cfg["dropout"]))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels, len(CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=-1, keepdim=True)
        rms = x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        h = self.body(x / rms)
        return self.head(self.drop(self.pool(h).squeeze(-1)))


def predict_logits(model: nn.Module, raw: torch.Tensor, indices: np.ndarray,
                   batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = np.empty((len(indices), len(CLASSES)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            sel = torch.as_tensor(indices[start:start + batch_size], dtype=torch.long,
                                  device=DEVICE)
            out[start:start + len(sel)] = model(raw[sel]).cpu().numpy()
    return out


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    cfg = CFG["branch_a"]["augment"]
    b, c, n = x.shape
    shift = torch.randint(-int(cfg["time_shift_samples"]), int(cfg["time_shift_samples"]) + 1,
                          (b,), generator=gen, device=x.device)
    idx = (torch.arange(n, device=x.device)[None, :] - shift[:, None]) % n
    x = torch.gather(x, 2, idx[:, None, :].expand(b, c, n))
    drop = (torch.rand(b, c, 1, generator=gen, device=x.device)
            < float(cfg["channel_dropout"])).to(x.dtype)
    x = x * (1.0 - drop)
    low, high = cfg["amplitude_scale"]
    scale = low + (high - low) * torch.rand(b, 1, 1, generator=gen, device=x.device)
    x = x * scale
    snr_low, snr_high = cfg["gaussian_noise_snr_db"]
    snr = snr_low + (snr_high - snr_low) * torch.rand(b, 1, 1, generator=gen, device=x.device)
    rms = (x - x.mean(dim=-1, keepdim=True)).pow(2).mean(
        dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
    noise = torch.randn(x.shape, generator=gen, device=x.device) * rms / 10.0 ** (snr / 20.0)
    return x + noise


def pgd_train(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
              gen: torch.Generator) -> torch.Tensor:
    cfg = CFG["branch_a"]
    eps, steps = float(cfg["pgd_eps_linf"]), int(cfg["pgd_steps"])
    xc = x - x.mean(dim=-1, keepdim=True)
    rms = xc.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-6)
    bound = (eps * rms).view(-1, 1, 1)
    adv = x + (torch.rand(x.shape, generator=gen, device=x.device) * 2 - 1) * bound
    adv = torch.clamp(adv, x - bound, x + bound).detach()
    alpha = 2.0 * eps / steps
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        adv = torch.clamp(adv.detach() + (alpha * rms).view(-1, 1, 1) * grad.sign(),
                          x - bound, x + bound)
    return adv.detach()


def train_model(raw: torch.Tensor, idx_np: np.ndarray, labels_np: np.ndarray, seed: int,
                epochs: int, tag: str, eval_np: np.ndarray | None = None,
                eval_labels: np.ndarray | None = None) -> tuple[nn.Module, dict]:
    cfg = CFG["branch_a"]
    torch.manual_seed(seed)
    model = Net().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                            weight_decay=float(cfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    idx = torch.as_tensor(idx_np, dtype=torch.long, device=DEVICE)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    rng = np.random.default_rng(seed)
    batch = int(cfg["batch_size"])
    best = (-1.0, None, -1)
    info = {"epochs": epochs, "epoch_seconds": [], "eval_accuracy": [],
            "parameter_count": int(sum(p.numel() for p in model.parameters()))}
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        order = torch.as_tensor(rng.permutation(len(idx_np)), dtype=torch.long, device=DEVICE)
        total = 0.0
        for start in range(0, len(idx_np), batch):
            sel = order[start:start + batch]
            if len(sel) < 4:
                continue
            xb = augment(raw[idx[sel]], gen)
            yb = labels[sel]
            adv = pgd_train(model, xb, yb, gen)
            loss = 0.5 * F.cross_entropy(model(xb), yb) + 0.5 * F.cross_entropy(model(adv), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(sel)
        sched.step()
        info["epoch_seconds"].append(time.time() - started)
        message = (f"  [{tag}] epoch {epoch}/{epochs} loss {total / len(idx_np):.4f} "
                   f"{info['epoch_seconds'][-1]:.1f}s")
        if eval_np is not None:
            acc = float(np.mean(predict_logits(model, raw, eval_np).argmax(1) + 1
                                == eval_labels))
            info["eval_accuracy"].append(acc)
            message += f" eval_acc {acc:.4f}"
            if acc > best[0]:
                best = (acc, {k: v.detach().clone() for k, v in model.state_dict().items()},
                        epoch)
        log(message)
    if best[1] is not None:
        model.load_state_dict(best[1])
        info["early_stop"] = "eval-block clean accuracy"
        info["best_epoch"] = best[2]
        info["best_eval_accuracy"] = best[0]
        log(f"  [{tag}] restored best epoch {best[2]} (eval_acc {best[0]:.4f})")
    model.eval()
    return model, info


# ---------------------------------------------------------------- branch B
def task_stats(features: np.ndarray, tasks: np.ndarray) -> dict:
    stats = {}
    for task in TASKS:
        ref = tasks == task
        stats[task] = (f32(features[ref].mean(0)), f32(features[ref].std(0) + 1e-6))
    return stats


def apply_stats(features: np.ndarray, tasks: np.ndarray, stats: dict) -> np.ndarray:
    out = features.astype(np.float64).copy()
    for task in TASKS:
        ref = tasks == task
        mu, sd = stats[task]
        out[ref] = (out[ref] - mu.astype(np.float64)) / sd.astype(np.float64)
    return f32(out)


def pca_basis(x: np.ndarray, dim: int) -> tuple:
    data = np.asarray(x, dtype=np.float64)
    mu = data.mean(0)
    centred = data - mu
    values, vectors = np.linalg.eigh(centred.T @ centred / max(len(centred) - 1, 1))
    basis = vectors[:, np.argsort(-values)][:, : min(int(dim), data.shape[1])]
    return f32(mu), f32(basis)


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


def build_prototypes(features: np.ndarray, subjects: np.ndarray, tasks: np.ndarray,
                     mask: np.ndarray, classes: np.ndarray, k: int) -> tuple:
    """k-means prototypes per (task, identity); returns (protos, keys)."""
    protos, keys = [], []
    for task in TASKS:
        for subject in classes:
            sel = mask & (tasks == task) & (subjects == subject)
            if sel.sum() == 0:
                continue
            x = torch.as_tensor(f32(features[sel]), device=DEVICE)
            centres = kmeans_t(x, k, int(subject) * 17 + TASKS.index(task))
            if len(centres) < k:  # group smaller than k: tile to keep a fixed layout
                centres = torch.cat([centres, centres.repeat(
                    (k + len(centres) - 1) // len(centres), 1)])[:k]
            protos.append(centres.cpu().numpy())
            keys.extend([(task, int(subject))] * k)
    return f32(np.concatenate(protos, axis=0)), keys


def proto_stack(protos: np.ndarray, keys: list, k: int) -> dict:
    """task -> (n_identity, k, dim) tensor; missing identities stay zero and are masked."""
    dim = protos.shape[1]
    stack = {task: np.zeros((len(CLASSES), k, dim), dtype=np.float32) for task in TASKS}
    filled = {task: np.zeros(len(CLASSES), dtype=bool) for task in TASKS}
    for index, (task, subject) in enumerate(keys):
        block = index // k
        stack[task][subject - 1] = protos[block * k:(block + 1) * k]
        filled[task][subject - 1] = True
    return stack, filled


def branch_b_scores(features: np.ndarray, tasks: np.ndarray, proto: dict, filled: dict,
                    k: int, batch: int = 4096) -> np.ndarray:
    out = np.full(len(features), np.nan, dtype=np.float32)
    for task in TASKS:
        rows = np.flatnonzero(tasks == task)
        if len(rows) == 0:
            continue
        valid = filled[task]
        p = torch.as_tensor(proto[task][valid].reshape(-1, proto[task].shape[-1]),
                            dtype=FT, device=DEVICE)
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            q = torch.as_tensor(f32(features[chunk]), device=DEVICE)
            dist = torch.cdist(q, p).view(len(chunk), int(valid.sum()), k)
            out[chunk] = (-dist).max(dim=2).values.max(dim=1).values.cpu().numpy()
    return out


def branch_b_identity(features: np.ndarray, tasks: np.ndarray, proto: dict, filled: dict,
                      k: int, batch: int = 4096) -> np.ndarray:
    out = np.full(len(features), -1, dtype=np.int32)
    for task in TASKS:
        rows = np.flatnonzero(tasks == task)
        valid = filled[task]
        subjects = CLASSES[valid]
        p = torch.as_tensor(proto[task][valid].reshape(-1, proto[task].shape[-1]),
                            dtype=FT, device=DEVICE)
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            q = torch.as_tensor(f32(features[chunk]), device=DEVICE)
            dist = torch.cdist(q, p).view(len(chunk), len(subjects), k)
            out[chunk] = subjects[(-dist).max(dim=2).values.argmax(dim=1).cpu().numpy()]
    return out


def quantile(value: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(value, dtype=np.float64), q))


# ---------------------------------------------------------------- stage A
def stage_a() -> dict:
    started = time.time()
    meta, frac, blocks, fit_rows = load_config_meta()
    subjects = meta["subject"].to_numpy(int)
    window = (frac > float(CFG["protocol"]["fit_fraction"])) & (frac <= 0.8)
    log(f"stage A: {len(meta)} Train trials, fit_rows {int(fit_rows.sum())}, "
        f"window {int(window.sum())}")
    raw = load_raw(meta["path"].tolist())
    log(f"raw on GPU: {tuple(raw.shape)} ({raw.numel() * 4 / 2**30:.2f} GiB)")

    oof_logits = np.zeros((len(meta), len(CLASSES)), dtype=np.float32)
    folds = []
    epochs = int(CFG["branch_a"]["fold_epochs"])
    for block in range(int(CFG["protocol"]["blocks"])):
        train_np = np.flatnonzero(blocks != block)
        eval_np = np.flatnonzero(blocks == block)
        model, info = train_model(raw, train_np, subjects[train_np] - 1, SEED + 101 * block,
                                  epochs, f"fold{block}")
        logits = predict_logits(model, raw, eval_np)
        oof_logits[eval_np] = logits
        accuracy = float(np.mean(logits.argmax(1) + 1 == subjects[eval_np]))
        folds.append({"block": block, "n_train": int(len(train_np)),
                      "n_eval": int(len(eval_np)), "accuracy": accuracy,
                      "epoch_seconds": info["epoch_seconds"],
                      "parameter_count": info["parameter_count"]})
        torch.save({"state_dict": model.state_dict(), "block": block,
                    "spec_sha256": BRANCH_A_SHA}, CACHE / f"fold_model_{block}.pt")
        log(f"  fold{block} OOF accuracy {accuracy:.4f}")
        del model
        torch.cuda.empty_cache()

    window_np = np.flatnonzero(window)
    full, info = train_model(raw, np.arange(len(meta)), subjects - 1, SEED,
                             int(CFG["branch_a"]["full_epochs"]), "full",
                             eval_np=window_np, eval_labels=subjects[window_np])
    in_sample = predict_logits(full, raw, np.arange(len(meta)))
    in_sample_acc = float(np.mean(in_sample.argmax(1) + 1 == subjects))
    in_sample_f1 = float(f1_score(subjects, in_sample.argmax(1) + 1, labels=CLASSES,
                                  average="macro", zero_division=0))
    log(f"  full in-sample accuracy {in_sample_acc:.4f} macro-F1 {in_sample_f1:.4f}")

    torch.save({"state_dict": full.state_dict(), "spec_sha256": BRANCH_A_SHA,
                "spec": CFG["branch_a"], "classes": CLASSES,
                "in_sample_accuracy": in_sample_acc,
                "in_sample_macro_f1": in_sample_f1,
                "window_eval_accuracy": info.get("best_eval_accuracy")}, MODEL_A)
    torch.save({"spec_sha256": BRANCH_A_SHA, "oof_logits": torch.as_tensor(oof_logits),
                "in_sample_logits": torch.as_tensor(in_sample),
                "subjects": torch.as_tensor(subjects),
                "blocks": torch.as_tensor(blocks)}, STAGE_A)

    oof_pred = oof_logits.argmax(1) + 1
    forward = frac > float(CFG["protocol"]["fit_fraction"])
    summary = {
        "n_parameters": info["parameter_count"],
        "fold_epochs": epochs, "full_epochs": int(CFG["branch_a"]["full_epochs"]),
        "full_epochs_run": len(info["epoch_seconds"]),
        "full_best_epoch": info.get("best_epoch"),
        "full_window_eval_accuracy": info.get("best_eval_accuracy"),
        "fold_mean_epoch_seconds": float(np.mean([np.mean(f["epoch_seconds"])
                                                  for f in folds])),
        "folds": folds,
        "oof_accuracy_5block": float(np.mean(oof_pred == subjects)),
        "oof_macro_f1": float(f1_score(subjects, oof_pred, labels=CLASSES,
                                       average="macro", zero_division=0)),
        "oof_accuracy_forward_blocks": float(np.mean(oof_pred[forward] == subjects[forward])),
        "oof_accuracy_per_block": {str(b): float(np.mean(
            oof_pred[blocks == b] == subjects[blocks == b]))
            for b in range(int(CFG["protocol"]["blocks"]))},
        "in_sample_accuracy": in_sample_acc,
        "in_sample_macro_f1": in_sample_f1,
        "in_sample_per_task_accuracy": {
            task: float(np.mean(in_sample.argmax(1)[meta["task"].to_numpy() == task] + 1
                                == subjects[meta["task"].to_numpy() == task]))
            for task in TASKS},
        "augmentation": CFG["branch_a"]["augment"],
        "adversarial_training": {
            "pgd_steps": int(CFG["branch_a"]["pgd_steps"]),
            "eps_linf_relative_rms": float(CFG["branch_a"]["pgd_eps_linf"]),
            "every_batch": True,
            "loss": CFG["branch_a"]["loss"]},
        "elapsed_seconds": time.time() - started,
        "gpu": torch.cuda.get_device_name(0),
    }
    log(f"stage A done in {summary['elapsed_seconds']:.0f}s: "
        f"OOF acc {summary['oof_accuracy_5block']:.4f}, macro-F1 "
        f"{summary['oof_macro_f1']:.4f}, in-sample {in_sample_acc:.4f}")
    results = load_results()
    results["branch_a"] = summary
    save_results(results)
    del raw
    torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------- stage B
def stage_b() -> dict:
    started = time.time()
    meta, frac, blocks, fit_rows = load_config_meta()
    subjects = meta["subject"].to_numpy(int)
    tasks = meta["task"].to_numpy()
    window = (frac > float(CFG["protocol"]["fit_fraction"])) & (frac <= 0.8)
    window_rows = np.flatnonzero(window)
    k = int(CFG["branch_b"]["prototype_k"])
    dim = int(CFG["branch_b"]["pca_dim"])

    raw_features = f32(np.load(ROOT / CFG["branch_b"]["train_features"]))
    if raw_features.shape != (len(meta), int(CFG["branch_b"]["feature_dim"])):
        raise ValueError(f"E65 cache shape {raw_features.shape} unexpected")
    stats = task_stats(raw_features, tasks)
    standardised = apply_stats(raw_features, tasks, stats)
    mu, basis = pca_basis(standardised[fit_rows], dim)
    projected = f32((standardised.astype(np.float64) - mu.astype(np.float64)) @ basis)
    log(f"stage B: E65 features -> PCA-{dim} (basis from {int(fit_rows.sum())} fit rows)")

    # 5-block OOF scores; prototypes for block b never contain block-b rows.
    oof_scores = np.full(len(meta), np.nan, dtype=np.float32)
    for block in range(int(CFG["protocol"]["blocks"])):
        rows = blocks == block
        protos, keys = build_prototypes(projected, subjects, tasks, fit_rows & ~rows,
                                        CLASSES, k)
        proto, filled = proto_stack(protos, keys, k)
        held = np.flatnonzero(rows)
        oof_scores[held] = branch_b_scores(projected[held], tasks[held], proto, filled, k)
        log(f"  block {block}: {len(held)} scored, prototypes {len(keys)}")
    if not np.isfinite(oof_scores).all():
        raise RuntimeError("OOF branch-B scores incomplete")
    tau = quantile(oof_scores, 1.0 - float(CFG["branch_b"]["threshold_acceptance"]))
    mu_oof = float(np.mean(oof_scores))
    sd_oof = float(np.std(oof_scores)) + 1e-6
    log(f"  tau(a=0.90) = {tau:.4f}; OOF mean {mu_oof:.4f} sd {sd_oof:.4f}")

    # 20-fold LOSO for gates #5/#6
    loso = []
    for held_id in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held_id])
        pm = np.isin(subjects, enrolled)
        protos, keys = build_prototypes(projected, subjects, tasks, pm & fit_rows,
                                        enrolled, k)
        proto, filled = proto_stack(protos, keys, k)
        known = window_rows[pm[window_rows]]
        unknown = window_rows[subjects[window_rows] == held_id]
        if len(known) == 0 or len(unknown) == 0:
            continue
        rows = np.r_[known, unknown]
        scores = branch_b_scores(projected[rows], tasks[rows], proto, filled, k)
        known_scores, unknown_scores = scores[:len(known)], scores[len(known):]
        threshold = quantile(known_scores, 1.0 - 0.95)
        auc = float(roc_auc_score(np.r_[np.zeros(len(unknown_scores)),
                                        np.ones(len(known_scores))],
                                  np.r_[unknown_scores, known_scores]))
        loso.append({"held_identity": int(held_id), "n_known": int(len(known)),
                     "n_unknown": int(len(unknown)), "auroc": auc,
                     "u_at_95": float(np.mean(unknown_scores < threshold)),
                     "known_acceptance": float(np.mean(known_scores >= threshold)),
                     "threshold": threshold,
                     "known_mean": float(np.mean(known_scores)),
                     "unknown_mean": float(np.mean(unknown_scores)),
                     "known_scores": known_scores, "unknown_scores": unknown_scores})
        log(f"  LOSO {held_id:2d}: AUROC {auc:.4f} U@95 {loso[-1]['u_at_95']:.4f} "
            f"(known {len(known)}, unknown {len(unknown)})")
    pooled_known = np.concatenate([row["known_scores"] for row in loso])
    pooled_unknown = np.concatenate([row["unknown_scores"] for row in loso])
    pooled_auc = float(roc_auc_score(np.r_[np.zeros(len(pooled_unknown)),
                                          np.ones(len(pooled_known))],
                                     np.r_[pooled_unknown, pooled_known]))

    # final prototypes: all 20 identities, fit rows only
    protos, keys = build_prototypes(projected, subjects, tasks, fit_rows, CLASSES, k)
    proto_final, filled_final = proto_stack(protos, keys, k)
    train_identity = branch_b_identity(projected, tasks, proto_final, filled_final, k)
    train_acc = float(np.mean(train_identity == subjects))
    log(f"  branch B in-sample identity accuracy (fit-row prototypes) {train_acc:.4f}")

    np.savez_compressed(
        STAGE_B, spec_sha256=np.array(BRANCH_B_SHA),
        pca_mu=mu, pca_basis=basis, tau=np.float32(tau), oof_mu=np.float32(mu_oof),
        oof_sd=np.float32(sd_oof), oof_scores=oof_scores, k=np.int32(k),
        stats_mu=np.stack([stats[t][0] for t in TASKS]),
        stats_sd=np.stack([stats[t][1] for t in TASKS]),
        proto=np.stack([proto_final[t] for t in TASKS]),
        proto_filled=np.stack([filled_final[t] for t in TASKS]),
        loso_known=pooled_known, loso_unknown=pooled_unknown)

    summary = {
        "pca_dim": dim, "prototype_k": k,
        "n_prototypes": int(len(keys)),
        "tau_accept_0_90": tau, "oof_mean": mu_oof, "oof_sd": sd_oof,
        "oof_acceptance_at_tau": float(np.mean(oof_scores >= tau)),
        "branch_b_train_identity_accuracy": train_acc,
        "loso_pooled_auroc": pooled_auc,
        "loso_fold_auroc_mean": float(np.mean([row["auroc"] for row in loso])),
        "loso_fold_auroc_std": float(np.std([row["auroc"] for row in loso])),
        "loso_fold_auroc_min": float(np.min([row["auroc"] for row in loso])),
        "loso_fold_auroc_max": float(np.max([row["auroc"] for row in loso])),
        "u_at_95_mean": float(np.mean([row["u_at_95"] for row in loso])),
        "u_at_95_std": float(np.std([row["u_at_95"] for row in loso])),
        "u_at_95_min": float(np.min([row["u_at_95"] for row in loso])),
        "u_at_95_max": float(np.max([row["u_at_95"] for row in loso])),
        "n_known_pooled": int(len(pooled_known)),
        "n_unknown_pooled": int(len(pooled_unknown)),
        "folds": [{key: row[key] for key in ("held_identity", "n_known", "n_unknown",
                                             "auroc", "u_at_95", "threshold",
                                             "known_acceptance", "known_mean",
                                             "unknown_mean")} for row in loso],
        "elapsed_seconds": time.time() - started,
    }
    results = load_results()
    results["branch_b"] = summary
    save_results(results)
    log(f"stage B done in {summary['elapsed_seconds']:.0f}s: pooled AUROC {pooled_auc:.4f}, "
        f"U@95 {summary['u_at_95_mean']:.4f}±{summary['u_at_95_std']:.4f}")

    # fused Train OOF view (branch A identity + branch B gate) for context
    if not STAGE_A.exists():
        log("  stage A artifact missing: skipped the fused OOF view")
        return summary
    stage_a_data = torch.load(STAGE_A, map_location="cpu", weights_only=False)
    oof_logits = stage_a_data["oof_logits"].numpy()
    pred = oof_logits.argmax(1) + 1
    gated = np.where(oof_scores >= tau, pred, 0)
    max_prob = torch.softmax(torch.as_tensor(oof_logits), dim=1).numpy().max(1)
    tau_a = quantile(max_prob, 1.0 - float(CFG["branch_b"]["threshold_acceptance"]))
    dual = np.where((oof_scores >= tau) & (max_prob >= tau_a), pred, 0)
    fused = {
        "oof_gated_accuracy": float(np.mean(gated == subjects)),
        "oof_gated_known_accuracy": float(np.mean(gated[oof_scores >= tau]
                                                  == subjects[oof_scores >= tau])),
        "oof_rejection_rate": float(np.mean(gated == 0)),
        "oof_identity_macro_f1_gated": float(f1_score(subjects, gated, labels=CLASSES,
                                                      average="macro", zero_division=0)),
        "oof_joint_21class_macro_f1": float(f1_score(subjects, gated,
                                                      labels=np.arange(21),
                                                      average="macro",
                                                      zero_division=0)),
        "tau_a_accept_0_90": tau_a,
        "oof_dual_gate_accuracy": float(np.mean(dual == subjects)),
        "oof_dual_gate_rejection_rate": float(np.mean(dual == 0)),
    }
    results = load_results()
    results["train_oof_fusion"] = fused
    save_results(results)
    return summary


def load_branch_b() -> dict:
    data = np.load(STAGE_B, allow_pickle=False)
    if str(data["spec_sha256"]) != BRANCH_B_SHA:
        raise ValueError("branch B cache was built with a different config")
    return {key: data[key] for key in data.files}


def branch_b_test_scores(test_features: np.ndarray, test_tasks: np.ndarray,
                         store: dict) -> tuple[np.ndarray, np.ndarray]:
    stats = {task: (store["stats_mu"][i], store["stats_sd"][i])
             for i, task in enumerate(TASKS)}
    standardised = apply_stats(f32(test_features), test_tasks, stats)
    projected = f32((standardised.astype(np.float64) - store["pca_mu"].astype(np.float64))
                    @ store["pca_basis"])
    k, dim = int(store["k"]), store["proto"].shape[-1]
    proto = {task: f32(store["proto"][i]) for i, task in enumerate(TASKS)}
    filled = {task: store["proto_filled"][i] for i, task in enumerate(TASKS)}
    scores = branch_b_scores(projected, test_tasks, proto, filled, k)
    identity = branch_b_identity(projected, test_tasks, proto, filled, k)
    return scores, identity


# ---------------------------------------------------------------- stage test
def stage_test() -> dict:
    started = time.time()
    if not (MODEL_A.exists() and STAGE_B.exists()):
        raise FileNotFoundError("run stages a and b before scoring Test")
    meta, frac, blocks, fit_rows = load_config_meta()
    test_meta = pd.read_csv(ROOT / CFG["data"]["test_metadata"],
                            usecols=["sample", "path", "task", "session"])
    if len(test_meta) != int(CFG["data"]["test_trials"]):
        raise ValueError(f"Test rows {len(test_meta)} != {CFG['data']['test_trials']}")
    if not test_meta["path"].astype(str).str.startswith("Test/").all():
        raise ValueError("non-Test path in metadata")
    test_tasks = test_meta["task"].to_numpy()

    store_b = load_branch_b()
    branch_a = torch.load(MODEL_A, map_location="cpu", weights_only=False)
    if branch_a["spec_sha256"] != BRANCH_A_SHA:
        raise ValueError("branch A artifact was trained with a different config")
    model = Net().to(DEVICE)
    model.load_state_dict(branch_a["state_dict"])
    model.eval()

    test_features = f32(np.load(ROOT / CFG["branch_b"]["test_features"]))
    if test_features.shape[0] != len(test_meta):
        raise ValueError("Test feature cache row count mismatch")
    score_b, identity_b = branch_b_test_scores(test_features, test_tasks, store_b)
    if not np.isfinite(score_b).all():
        raise RuntimeError("non-finite branch-B Test scores")

    # Branch A Test logits, streamed so the full Test tensor never has to stay resident.
    logits = np.zeros((len(test_meta), len(CLASSES)), dtype=np.float32)
    paths = test_meta["path"].tolist()
    base = ROOT / CFG["data"]["test_root"]
    with ThreadPoolExecutor(max_workers=12) as pool:
        for start in range(0, len(paths), 512):
            chunk = paths[start:start + 512]
            arrays = list(pool.map(lambda p: load_mat(base / str(p)), chunk))
            batch = torch.as_tensor(np.stack(arrays), dtype=FT, device=DEVICE)
            with torch.no_grad():
                logits[start:start + len(chunk)] = model(batch).cpu().numpy()
            del batch
            done = start + len(chunk)
            if done % 2048 < 512 or done == len(paths):
                log(f"  Test branch A {done}/{len(paths)} ({time.time() - started:.0f}s)")
    del model
    torch.cuda.empty_cache()

    tau = float(store_b["tau"])
    max_prob = torch.softmax(torch.as_tensor(logits), dim=1).numpy().max(1)
    pred_a = logits.argmax(1) + 1
    accept_b = score_b >= tau
    gated = np.where(accept_b, pred_a, 0)
    # branch-A confidence threshold: Train OOF max softmax at the same 0.90 quantile
    stage_a_data = torch.load(STAGE_A, map_location="cpu", weights_only=False)
    oof_prob = torch.softmax(stage_a_data["oof_logits"], dim=1).numpy().max(1)
    tau_a = quantile(oof_prob, 1.0 - float(CFG["branch_b"]["threshold_acceptance"]))
    dual = np.where(accept_b & (max_prob >= tau_a), pred_a, 0)

    pd.DataFrame({"Sample": test_meta["sample"].to_numpy(), "Label": gated}).to_csv(
        OUT / "test_predictions_primary.csv", index=False)
    pd.DataFrame({"Sample": test_meta["sample"].to_numpy(), "Label": dual}).to_csv(
        OUT / "test_predictions_dual_gate.csv", index=False)

    # Session is read only now, after every prediction has been written.
    is_s1 = test_meta["session"].astype(str).str.casefold().isin(
        ["session1", "s1", "1"]).to_numpy()
    audit = {}
    for name, labels in (("primary", gated), ("dual_gate", dual)):
        audit[name] = {}
        for code, mask in (("S1", is_s1), ("S2", ~is_s1)):
            audit[name][code] = {
                "n": int(mask.sum()),
                "rejection_rate": float(np.mean(labels[mask] == 0)),
                "acceptance_rate": float(np.mean(labels[mask] != 0)),
            }
        audit[name]["overall"] = {
            "n": int(len(labels)),
            "rejection_rate": float(np.mean(labels == 0)),
            "identity_histogram_top": {str(k): int(v) for k, v in
                                       zip(*np.unique(labels, return_counts=True))},
        }
        audit[name]["tau"] = tau
        audit[name]["tau_a"] = tau_a

    summary = {
        "predictions_primary": str(OUT / "test_predictions_primary.csv"),
        "predictions_dual_gate": str(OUT / "test_predictions_dual_gate.csv"),
        "threshold_tau_branch_b": tau,
        "threshold_tau_branch_a": tau_a,
        "branch_b_identity_differs_from_branch_a": float(np.mean(identity_b != pred_a)),
        "audit": audit,
        "session_used_for_inference": False,
        "official_metrics": {key: "N/A" for key in
                             ("AUC", "F1", "C", "RA", "ASafeR", "U", "Score")},
        "elapsed_seconds": time.time() - started,
    }
    results = load_results()
    results["test"] = summary
    save_results(results)
    log(f"Test audit primary: S1 rejection "
        f"{audit['primary']['S1']['rejection_rate']:.4f}, S2 acceptance "
        f"{audit['primary']['S2']['acceptance_rate']:.4f}")
    log(f"Test audit dual gate: S1 rejection "
        f"{audit['dual_gate']['S1']['rejection_rate']:.4f}, S2 acceptance "
        f"{audit['dual_gate']['S2']['acceptance_rate']:.4f}")
    return summary


# ---------------------------------------------------------------- attack stage
MASK_BANDS = None
TRI_ROWS = TRI_COLS = DIAG = OFFMUL = None


def build_torch_constants() -> None:
    global MASK_BANDS, TRI_ROWS, TRI_COLS, DIAG, OFFMUL
    b = CFG["branch_b"]
    freq = torch.fft.rfftfreq(N, d=1.0 / FS, device=DEVICE)
    masks = []
    for index, (low, high) in enumerate(b["bands_hz"]):
        upper = freq <= high if index == len(b["bands_hz"]) - 1 else freq < high
        masks.append((freq >= low) & upper)
    MASK_BANDS = torch.stack(masks).to(FT)
    TRI_ROWS, TRI_COLS = torch.triu_indices(CH, CH, device=DEVICE)
    DIAG = TRI_ROWS == TRI_COLS
    OFFMUL = torch.where(DIAG, torch.ones_like(TRI_ROWS, dtype=FT),
                         torch.full_like(TRI_ROWS, float(b["off_diagonal_multiplier"]),
                                         dtype=FT))


def differentiable_logspd(x: torch.Tensor) -> torch.Tensor:
    """E65 gpu_correlation_logspd, differentiable end to end."""
    b = CFG["branch_b"]
    spec = torch.fft.rfft(x, dim=-1)
    filtered = torch.fft.irfft(spec[:, None] * MASK_BANDS[None, :, None, :], n=N, dim=-1)
    filtered = filtered - filtered.mean(dim=-1, keepdim=True)
    scale = filtered.std(dim=-1, keepdim=True, unbiased=True).clamp_min(
        float(b["channel_std_epsilon"]))
    normalised = (filtered - filtered.mean(dim=-1, keepdim=True)) / scale
    corr = normalised @ normalised.transpose(-1, -2) / (N - int(b["channel_std_ddof"]))
    trace = torch.diagonal(corr, dim1=-2, dim2=-1).sum(dim=-1)
    eye = torch.eye(CH, dtype=x.dtype, device=x.device)
    rho = float(b["covariance_shrinkage"])
    spd = (1.0 - rho) * corr + rho * trace[..., None, None] / CH * eye
    values, vectors = torch.linalg.eigh(spd)
    values = values.clamp_min(float(b["eigenvalue_floor"]))
    logspd = (vectors * values.log().unsqueeze(-2)) @ vectors.transpose(-1, -2)
    vec = logspd[..., TRI_ROWS, TRI_COLS] * OFFMUL
    vec = vec.clone()
    vec[..., DIAG] = vec[..., DIAG] - vec[..., DIAG].mean(dim=-1, keepdim=True)
    mean = vec.mean(dim=-1, keepdim=True)
    sd = vec.std(dim=-1, keepdim=True, unbiased=False).clamp_min(float(b["zband_epsilon"]))
    return ((vec - mean) / sd).flatten(-2)


class Scorer:
    """Frozen dual-branch forward pass with gradients w.r.t. the raw EEG."""

    def __init__(self, model: nn.Module, store: dict) -> None:
        self.model = model
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        self.stats_mu = torch.as_tensor(f32(store["stats_mu"]), device=DEVICE)
        self.stats_sd = torch.as_tensor(f32(store["stats_sd"]), device=DEVICE)
        self.pca_mu = torch.as_tensor(f32(store["pca_mu"]), device=DEVICE)
        self.basis = torch.as_tensor(f32(store["pca_basis"]), device=DEVICE)
        self.proto = torch.as_tensor(f32(store["proto"]), device=DEVICE)
        self.filled = torch.as_tensor(np.asarray(store["proto_filled"]), device=DEVICE)
        self.k = int(store["k"])
        self.oof_mu = float(store["oof_mu"])
        self.oof_sd = float(store["oof_sd"])
        self.tau = float(store["tau"])

    def __call__(self, x: torch.Tensor, task_idx: torch.Tensor) -> tuple:
        features = differentiable_logspd(x)
        mu = self.stats_mu[task_idx]
        sd = self.stats_sd[task_idx]
        z = (features - mu) / sd
        projected = (z - self.pca_mu) @ self.basis
        rows = torch.arange(len(x), device=DEVICE)
        best = None
        for t in range(len(TASKS)):
            valid = self.filled[t]
            p = self.proto[t][valid].reshape(-1, self.proto.shape[-1])
            dist = torch.cdist(projected, p).view(len(x), int(valid.sum()), self.k)
            score = (-dist).max(dim=2).values.max(dim=1).values
            take = task_idx == t
            if not bool(take.any()):
                continue
            filled = torch.full_like(score, OFF)
            filled[take] = score[take]
            best = filled if best is None else torch.where(take, filled, best)
        logits = self.model(x)
        return logits, (best - self.oof_mu) / self.oof_sd


def pgd_attack(scorer: Scorer, x: torch.Tensor, task_idx: torch.Tensor,
               true_idx: torch.Tensor, target_idx: torch.Tensor, goal: str,
               eps: float, steps: int, seed: int) -> dict:
    cfg = CFG["attack"]
    temperature = float(cfg["temperature"])
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    xc = x - x.mean(dim=-1, keepdim=True)
    rms = xc.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-6)
    bound = (eps * rms).view(-1, 1, 1)
    with torch.no_grad():
        adv = x + (torch.rand(x.shape, generator=gen, device=DEVICE) * 2 - 1) * bound
        adv = torch.clamp(adv, x - bound, x + bound)
    alpha = 2.0 * eps / steps
    nan_count = 0
    for _ in range(steps):
        adv.requires_grad_(True)
        logits, zb = scorer(adv, task_idx)
        if goal == "D":
            objective = zb
        else:
            true_logit = logits.gather(1, true_idx[:, None]).squeeze(1)
            wrong = logits.masked_fill(
                F.one_hot(true_idx, len(CLASSES)).bool(), OFF)
            if goal == "A":
                margin = wrong.max(dim=1).values
            elif goal == "B":
                margin = torch.logsumexp(wrong, dim=1)
            else:
                margin = logits.gather(1, target_idx[:, None]).squeeze(1)
            objective = (margin - true_logit) / temperature
        loss = -(objective.sum())
        grad = torch.autograd.grad(loss, adv)[0]
        nan_count += int((~torch.isfinite(grad)).sum().item())
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        # loss is the negative objective, so step against its gradient = ascend the objective
        adv = torch.clamp(adv.detach() - (alpha * rms).view(-1, 1, 1) * grad.sign(),
                          x - bound, x + bound)
    with torch.no_grad():
        logits, zb = scorer(adv, task_idx)
    realised = ((adv - x).abs().amax(dim=(1, 2)) / rms)
    return {"logits": logits.cpu().numpy(), "z": zb.cpu().numpy(),
            "max_rel_linf": float(realised.max()), "nan_gradients": nan_count}


def attack_metrics(logits: np.ndarray, z: np.ndarray, truth: np.ndarray,
                   tau_z: float) -> dict:
    accept = z >= tau_z
    predicted = np.where(accept, logits.argmax(1) + 1, 0)
    n = len(predicted)
    ra = float(np.mean(predicted == truth))
    asafe = float(np.mean((predicted == truth) | (predicted == 0)))
    half = 1.96 * math.sqrt(max(ra * (1 - ra), 0.0) / max(n, 1))
    half_safe = 1.96 * math.sqrt(max(asafe * (1 - asafe), 0.0) / max(n, 1))
    return {"n": n, "RA": ra, "RA_ci95": [max(0.0, ra - half), min(1.0, ra + half)],
            "ASafeR": asafe,
            "ASafeR_ci95": [max(0.0, asafe - half_safe), min(1.0, asafe + half_safe)],
            "closed_set_accuracy": float(np.mean(logits.argmax(1) + 1 == truth)),
            "accept_rate": float(np.mean(accept)),
            "wrong_identity_rate": float(np.mean((predicted != 0) & (predicted != truth))),
            "reject_rate": float(np.mean(predicted == 0)),
            "identity_macro_f1": float(f1_score(truth, predicted, labels=CLASSES,
                                                average="macro", zero_division=0))}


def stage_attack() -> dict:
    started = time.time()
    build_torch_constants()
    meta, frac, blocks, fit_rows = load_config_meta()
    subjects = meta["subject"].to_numpy(int)
    tasks = meta["task"].to_numpy()
    window_rows = np.flatnonzero((frac > float(CFG["protocol"]["fit_fraction"]))
                                 & (frac <= 0.8))
    cfg = CFG["attack"]
    rng = np.random.default_rng(int(cfg["seed"]))
    cap = int(cfg["cap_per_task"])
    selected = []
    for task in TASKS:
        rows = window_rows[tasks[window_rows] == task]
        selected.append(rows if len(rows) <= cap else np.sort(
            rng.choice(rows, size=cap, replace=False)))
    subset = np.sort(np.concatenate(selected))
    log(f"attack stage: {len(subset)} eval-window trials "
        f"({', '.join(f'{t}={int((tasks[subset] == t).sum())}' for t in TASKS)})")

    branch_a = torch.load(MODEL_A, map_location="cpu", weights_only=False)
    if branch_a["spec_sha256"] != BRANCH_A_SHA:
        raise ValueError("branch A artifact does not match the frozen spec")
    model = Net().to(DEVICE)
    model.load_state_dict(branch_a["state_dict"])
    store = load_branch_b()
    scorer = Scorer(model, store)
    oof_scores = np.asarray(store["oof_scores"], dtype=np.float64)
    sweep_acceptance = (0.80, 0.90, 0.95)
    tau_sweep = {a: (quantile(oof_scores, 1.0 - a) - float(store["oof_mu"]))
                 / float(store["oof_sd"]) for a in sweep_acceptance}
    primary_acceptance = float(CFG["branch_b"]["threshold_acceptance"])
    tau_z = tau_sweep[primary_acceptance]

    raw = load_raw([meta["path"].iloc[i] for i in subset])
    truth = subjects[subset]

    def metrics_at_all(logits: np.ndarray, z: np.ndarray) -> dict:
        out = {}
        for a in sweep_acceptance:
            row = attack_metrics(logits, z, truth, tau_sweep[a])
            out[f"{a:.2f}"] = {"RA": row["RA"], "ASafeR": row["ASafeR"],
                               "accept_rate": row["accept_rate"]}
        return out

    task_idx = torch.as_tensor([TASKS.index(t) for t in tasks[subset]],
                               dtype=torch.long, device=DEVICE)
    true_idx = torch.as_tensor(truth - 1, dtype=torch.long, device=DEVICE)
    lut = {int(c): i for i, c in enumerate(CLASSES)}
    target = np.asarray([CLASSES[(lut[int(v)] + 1) % len(CLASSES)] for v in truth])
    target_idx = torch.as_tensor([lut[int(v)] for v in target], dtype=torch.long,
                                 device=DEVICE)

    # Differentiable-vs-cached feature check on a small slice.
    cached = f32(np.load(ROOT / CFG["branch_b"]["train_features"])[subset[:8]])
    with torch.no_grad():
        rebuilt = differentiable_logspd(raw[:8]).cpu().numpy()
    feature_check = float(np.max(np.abs(rebuilt - cached)))

    with torch.no_grad():
        logits, zb = scorer(raw, task_idx)
    clean_logits, clean_z = logits.cpu().numpy(), zb.cpu().numpy()
    clean = attack_metrics(clean_logits, clean_z, truth, tau_z)
    clean["at_acceptance"] = metrics_at_all(clean_logits, clean_z)
    log(f"  clean: RA {clean['RA']:.4f} ASafeR {clean['ASafeR']:.4f} "
        f"closed {clean['closed_set_accuracy']:.4f} accept {clean['accept_rate']:.4f}")

    conditions = []
    for eps in cfg["eps_normalised"]:
        for goal in cfg["goals"]:
            out = pgd_attack(scorer, raw, task_idx, true_idx, target_idx, goal,
                             float(eps), int(cfg["steps"]), int(cfg["seed"]) + int(eps * 1e5))
            metrics = attack_metrics(out["logits"], out["z"], truth, tau_z)
            metrics.update({"eps": float(eps), "goal": goal,
                            "max_rel_linf": out["max_rel_linf"],
                            "nan_gradients": out["nan_gradients"],
                            "at_acceptance": metrics_at_all(out["logits"], out["z"])})
            conditions.append(metrics)
            log(f"  eps={eps:.3f} goal {goal}: RA {metrics['RA']:.4f} "
                f"ASafeR {metrics['ASafeR']:.4f} closed {metrics['closed_set_accuracy']:.4f} "
                f"accept {metrics['accept_rate']:.4f} wrong_id "
                f"{metrics['wrong_identity_rate']:.4f}")
    worst_ra = min(conditions, key=lambda r: r["RA"])
    worst_safe = min(conditions, key=lambda r: r["ASafeR"])
    summary = {
        "n_trials": int(len(subset)),
        "per_task": {task: int((tasks[subset] == task).sum()) for task in TASKS},
        "tau_branch_b": float(store["tau"]), "tau_z": tau_z,
        "primary_acceptance": primary_acceptance,
        "sweep_acceptance": list(sweep_acceptance),
        "sweep_tau": {f"{a:.2f}": float(store["oof_mu"] + tau_sweep[a]
                                        * float(store["oof_sd"])) for a in sweep_acceptance},
        "eps_normalised": list(cfg["eps_normalised"]),
        "steps": int(cfg["steps"]), "temperature": float(cfg["temperature"]),
        "feature_rebuild_max_abs": feature_check,
        "clean": clean,
        "conditions": conditions,
        "worst_RA": {"RA": worst_ra["RA"], "eps": worst_ra["eps"],
                     "goal": worst_ra["goal"], "ASafeR": worst_ra["ASafeR"]},
        "worst_ASafeR": {"ASafeR": worst_safe["ASafeR"], "eps": worst_safe["eps"],
                         "goal": worst_safe["goal"], "RA": worst_safe["RA"]},
        "attack_status": "self-built white-box PGD surrogate, not the official suite",
        "elapsed_seconds": time.time() - started,
    }
    results = load_results()
    results["attack"] = summary
    save_results(results)
    log(f"attack stage done in {summary['elapsed_seconds']:.0f}s: worst RA "
        f"{worst_ra['RA']:.4f}, worst ASafeR {worst_safe['ASafeR']:.4f}, "
        f"feature rebuild max abs {feature_check:.2e}")
    return summary


# ---------------------------------------------------------------- gates / report
REFERENCE = {
    "E58": {"#1": 0.9974, "#2": 0.9714, "#3": 0.9849, "#4": 0.9714, "#5": 0.9984,
            "#6": 0.9875, "#7": 0.9371, "#8": 0.9758, "#9": 0.2533, "#10": 0.5875},
    "E170/E178": {"#1": 0.9730, "#2": 0.9592, "#3": 0.9590, "#4": 0.9592, "#5": 0.9094,
                  "#6": 0.4882, "#7": 0.4417, "#8": 0.6965, "#9": 0.7210, "#10": 0.6775},
}

GATE_SPEC = [
    ("#1", "拟合容量", "Train 准确率", "= 1.00"),
    ("#2", "折外闭集", "C（5 块 OOF 闭集准确率）", "≥ 0.95"),
    ("#3", "折外闭集", "身份 macro-F1（5 块 OOF）", "≥ 0.92"),
    ("#4", "跨时间代理", "块外准确率（同 #2 的 5 块口径）", "≥ 0.90"),
    ("#5", "开放集", "pseudo-unknown AUROC（20 折 LOSO 合并）", "≥ 0.98"),
    ("#6", "开放集", "U@95%（known 接受率固定 95%）", "≥ 0.95"),
    ("#7", "对抗识别", "RA（自建 PGD，全 ε×目标取最差）", "≥ 0.95"),
    ("#8", "对抗安全", "ASafeR（同一攻击取最差）", "≥ 0.98"),
    ("#9", "Test 审计", "S2 接受率", "≥ 0.75"),
    ("#10", "Test 审计", "S1 拒识率", "≥ 0.75"),
]


def gate_table(results: dict) -> list[dict]:
    a = results.get("branch_a", {})
    b = results.get("branch_b", {})
    att = results.get("attack", {})
    test = results.get("test", {})
    primary = test.get("audit", {}).get("primary", {})
    values = {
        "#1": a.get("in_sample_accuracy"),
        "#2": a.get("oof_accuracy_5block"),
        "#3": a.get("oof_macro_f1"),
        "#4": a.get("oof_accuracy_5block"),
        "#5": b.get("loso_pooled_auroc"),
        "#6": b.get("u_at_95_mean"),
        "#7": att.get("worst_RA", {}).get("RA"),
        "#8": att.get("worst_ASafeR", {}).get("ASafeR"),
        "#9": primary.get("S2", {}).get("acceptance_rate"),
        "#10": primary.get("S1", {}).get("rejection_rate"),
    }
    targets = {"#1": 1.00, "#2": 0.95, "#3": 0.92, "#4": 0.90, "#5": 0.98,
               "#6": 0.95, "#7": 0.95, "#8": 0.98, "#9": 0.75, "#10": 0.75}
    rows = []
    for key, layer, metric, target_text in GATE_SPEC:
        value = values[key]
        passed = None if value is None else bool(value >= targets[key] - 1e-12)
        rows.append({"gate": key, "layer": layer, "metric": metric,
                     "target": target_text, "value": value, "passed": passed})
    return rows


def fmt(value) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def write_report(results: dict) -> None:
    a = results.get("branch_a", {})
    b = results.get("branch_b", {})
    att = results.get("attack", {})
    test = results.get("test", {})
    rows = gate_table(results)
    passed = [row for row in rows if row["passed"]]
    n_pass = len(passed)
    lines = [
        "# E180 双分支（对抗训练骨干 + E65 开放集门控）十项门禁",
        "",
        "日期：2026-11-15。项目原生 18 通道、Train session1 only。",
        "分支 A：对抗训练 EEGNet 1D CNN（闭集身份）；分支 B：E65 correlation-logSPD "
        "PCA-96 / k=12 原型门控；融合规则 = 门控拒识则输出 0，否则分支 A argmax。",
        "",
        f"**十项门禁通过 {n_pass}/10。**",
        "",
        f"关键数值：OOF C {fmt(a.get('oof_accuracy_5block'))} / macro-F1 "
        f"{fmt(a.get('oof_macro_f1'))} / in-sample {fmt(a.get('in_sample_accuracy'))} / "
        f"LOSO AUROC {fmt(b.get('loso_pooled_auroc'))} / U@95 "
        f"{fmt(b.get('u_at_95_mean'))} / RA {fmt(att.get('worst_RA', {}).get('RA'))} / "
        f"ASafeR {fmt(att.get('worst_ASafeR', {}).get('ASafeR'))} / "
        f"Test S1 拒识 {fmt(test.get('audit', {}).get('primary', {}).get('S1', {}).get('rejection_rate'))} / "
        f"Test S2 接受 {fmt(test.get('audit', {}).get('primary', {}).get('S2', {}).get('acceptance_rate'))}。",
        "",
        "## 1. 完整十项门禁表",
        "",
        "| # | 层面 | 指标 | 目标 | 实测 | 判定 | 证据来源 |",
        "|---|---|---|---|---:|---|---|",
    ]
    evidence = {
        "#1": "分支 A 全量模型对全部 11304 条 Train 的 in-sample argmax（§2.1）",
        "#2": "分支 A 5 块 OOF 闭集准确率（§2.2）",
        "#3": "同一批 5 块 OOF 预测的身份 macro-F1（§2.2）",
        "#4": "5 块 OOF 同口径（额外汇总 frac>0.6 前向块，§2.2）",
        "#5": "分支 B 20 折 LOSO 合并 AUROC（§3）",
        "#6": "分支 B 20 折 U@95 均值（§3）",
        "#7": "融合系统自建 PGD 代理，ε×目标 A/B/C/D 取最差（§2.3）",
        "#8": "同一攻击条件下的最差 ASafeR（§2.3）",
        "#9": "冻结后 Test 一次性评分，Session 仅审计（§4）",
        "#10": "同 #9",
    }
    for row in rows:
        verdict = "N/A" if row["passed"] is None else ("通过" if row["passed"] else "未通过")
        lines.append(f"| {row['gate']} | {row['layer']} | {row['metric']} | "
                     f"{row['target']} | {fmt(row['value'])} | {verdict} | "
                     f"{evidence[row['gate']]} |")
    lines += [
        "",
        "## 2. 分支 A：对抗训练闭集骨干",
        "",
        f"结构：{CFG['branch_a']['architecture']}，参数量 "
        f"{a.get('n_parameters')}；输入逐 trial 去均值 + 全局 RMS 归一化。",
        f"训练：5 折各 {a.get('fold_epochs')} epoch + 全量模型 "
        f"{a.get('full_epochs_run')} epoch（早停 best epoch {a.get('full_best_epoch')}）；"
        f"每 batch 都做对抗训练（PGD-{CFG['branch_a']['pgd_steps']}，"
        f"L∞ ε={CFG['branch_a']['pgd_eps_linf']}×RMS），"
        f"损失 0.5·CE(clean)+0.5·CE(adv)；单 epoch 平均 "
        f"{fmt(a.get('fold_mean_epoch_seconds'))} s。",
        f"增强：时间平移 ±{CFG['branch_a']['augment']['time_shift_samples']}、通道 dropout "
        f"{CFG['branch_a']['augment']['channel_dropout']}、幅值 ×"
        f"{CFG['branch_a']['augment']['amplitude_scale']}、高斯噪声 SNR "
        f"{CFG['branch_a']['augment']['gaussian_noise_snr_db']} dB。",
        "",
        "### 2.1 拟合容量（#1）",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| in-sample 准确率 | {fmt(a.get('in_sample_accuracy'))} |",
        f"| in-sample macro-F1 | {fmt(a.get('in_sample_macro_f1'))} |",
    ]
    for task, value in (a.get("in_sample_per_task_accuracy") or {}).items():
        lines.append(f"| in-sample 准确率 {task} | {fmt(value)} |")
    lines += [
        "",
        "### 2.2 5 块 OOF（#2/#3/#4）",
        "",
        "| 块 | 训练条数 | 评估条数 | clean 准确率 |",
        "|---:|---:|---:|---:|",
    ]
    for fold in a.get("folds", []):
        lines.append(f"| {fold['block']} | {fold['n_train']} | {fold['n_eval']} | "
                     f"{fmt(fold['accuracy'])} |")
    lines += [
        "",
        f"5 块合并：C {fmt(a.get('oof_accuracy_5block'))}，macro-F1 "
        f"{fmt(a.get('oof_macro_f1'))}；仅前向块（frac>0.6）"
        f"{fmt(a.get('oof_accuracy_forward_blocks'))}；逐块 "
        + "、".join(f"{k}:{v:.4f}" for k, v in (a.get("oof_accuracy_per_block") or {}).items())
        + "。",
        "",
        "### 2.3 对抗鲁棒（#7/#8，融合后完整系统）",
        "",
        f"攻击域：Train 评估块 0.6<frac<=0.8，{att.get('n_trials')} 条（"
        + "、".join(f"{k}={v}" for k, v in (att.get("per_task") or {}).items())
        + f"）。可微管线（原始波形 → E65 特征 → PCA → 原型距离 + CNN logits）与缓存特征"
          f"最大绝对差 {att.get('feature_rebuild_max_abs', float('nan')):.2e}。",
        f"PGD：L∞，ε ∈ {att.get('eps_normalised')}×每 trial 全局 RMS，{att.get('steps')} 步，"
        f"随机起点；目标 A/B/C 只打身份头（最差错误类 margin / logsumexp / 指定冒充类，"
        f"均除以温度 {att.get('temperature')}），目标 D 只打门控（最大化 z(分支 B 分数)）。"
        f"四条目标互不混合，所以两条分支都不能靠“忽略对方”蒙混过关。",
        "",
        "| 条件 | ε | 目标 | RA | RA CI95 | ASafeR | ASafeR CI95 | 闭集准确率 | 接受率 | 误认率 | 拒识率 |",
        "|---|---:|---|---:|---|---:|---|---:|---:|---:|---:|",
        f"| clean | — | — | {fmt(att.get('clean', {}).get('RA'))} | "
        f"[{att.get('clean', {}).get('RA_ci95', [0, 0])[0]:.3f}, "
        f"{att.get('clean', {}).get('RA_ci95', [0, 0])[1]:.3f}] | "
        f"{fmt(att.get('clean', {}).get('ASafeR'))} | "
        f"[{att.get('clean', {}).get('ASafeR_ci95', [0, 0])[0]:.3f}, "
        f"{att.get('clean', {}).get('ASafeR_ci95', [0, 0])[1]:.3f}] | "
        f"{fmt(att.get('clean', {}).get('closed_set_accuracy'))} | "
        f"{fmt(att.get('clean', {}).get('accept_rate'))} | "
        f"{fmt(att.get('clean', {}).get('wrong_identity_rate'))} | "
        f"{fmt(att.get('clean', {}).get('reject_rate'))} |",
    ]
    for row in att.get("conditions", []):
        lines.append(
            f"| {row['goal']} | {row['eps']:.3f} | {row['goal']} | {fmt(row['RA'])} | "
            f"[{row['RA_ci95'][0]:.3f}, {row['RA_ci95'][1]:.3f}] | {fmt(row['ASafeR'])} | "
            f"[{row['ASafeR_ci95'][0]:.3f}, {row['ASafeR_ci95'][1]:.3f}] | "
            f"{fmt(row['closed_set_accuracy'])} | {fmt(row['accept_rate'])} | "
            f"{fmt(row['wrong_identity_rate'])} | {fmt(row['reject_rate'])} |")
    lines += [
        "",
        f"判定：#7 RA 最差 **{fmt(att.get('worst_RA', {}).get('RA'))}**"
        f"（ε={att.get('worst_RA', {}).get('eps')}，目标 {att.get('worst_RA', {}).get('goal')}）；"
        f"#8 ASafeR 最差 **{fmt(att.get('worst_ASafeR', {}).get('ASafeR'))}**"
        f"（ε={att.get('worst_ASafeR', {}).get('eps')}，目标 "
        f"{att.get('worst_ASafeR', {}).get('goal')}）。",
        "",
        "机制读数：分支 A 的闭集准确率在 ε=0.02（≈34 dB L∞）下仍有 "
        f"{fmt(min(row['closed_set_accuracy'] for row in att.get('conditions', [{}])))}，"
        "说明对抗训练把身份头救回来了（E170 同一攻击下闭集只剩 0.5711）；"
        "RA 的主要损失来自**门控拒识**（clean 拒识率即 "
        f"{fmt(att.get('clean', {}).get('reject_rate'))}），而不是认错人（误认率 ≤ "
        f"{fmt(max(row['wrong_identity_rate'] for row in att.get('conditions', [{}])))}）。"
        "由于 ASafeR = RA + 拒识率，而 #7 只看 RA，这两项在同一条判决上是此消彼长的："
        "本实验在 a=0.90 的工作点上 #7 未过、#8 过，原因是拒识被计为安全，而不是识别真的完美。",
        "",
        "### 2.4 工作点敏感性（同一批攻击波形，只移动门控阈值）",
        "",
        "门控阈值在 Train 5 块 OOF 上按接受率 a 取分位（`quantile(ref, 1-a)`）。"
        "这批样本上分支 A 的 clean 闭集准确率是 1.0000，因此 RA 基本等于接受率，"
        "#7 的 0.95 目标要求 a≳0.95。**a 在接触 Test 之前就已固定为 0.90，所以门禁表用的是 a=.90；"
        "下表只是 Train-only 的敏感性分析，不是用来挑工作点的。**",
        "",
        "| 条件 | ε | 目标 | RA@a=.80 | RA@a=.90 | RA@a=.95 | ASafeR@a=.80 | ASafeR@a=.90 | ASafeR@a=.95 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        f"| clean | — | — | {fmt(att.get('clean', {}).get('at_acceptance', {}).get('0.80', {}).get('RA'))} | "
        f"{fmt(att.get('clean', {}).get('at_acceptance', {}).get('0.90', {}).get('RA'))} | "
        f"{fmt(att.get('clean', {}).get('at_acceptance', {}).get('0.95', {}).get('RA'))} | "
        f"{fmt(att.get('clean', {}).get('at_acceptance', {}).get('0.80', {}).get('ASafeR'))} | "
        f"{fmt(att.get('clean', {}).get('at_acceptance', {}).get('0.90', {}).get('ASafeR'))} | "
        f"{fmt(att.get('clean', {}).get('at_acceptance', {}).get('0.95', {}).get('ASafeR'))} |",
    ]
    for row in att.get("conditions", []):
        sweep = row.get("at_acceptance", {})
        lines.append(
            f"| {row['goal']} | {row['eps']:.3f} | {row['goal']} | "
            f"{fmt(sweep.get('0.80', {}).get('RA'))} | {fmt(sweep.get('0.90', {}).get('RA'))} | "
            f"{fmt(sweep.get('0.95', {}).get('RA'))} | "
            f"{fmt(sweep.get('0.80', {}).get('ASafeR'))} | "
            f"{fmt(sweep.get('0.90', {}).get('ASafeR'))} | "
            f"{fmt(sweep.get('0.95', {}).get('ASafeR'))} |")
    lines += [
        "",
        f"阈值：a=.80 → τ={att.get('sweep_tau', {}).get('0.80', float('nan')):.4f}；"
        f"a=.90 → τ={att.get('sweep_tau', {}).get('0.90', float('nan')):.4f}；"
        f"a=.95 → τ={att.get('sweep_tau', {}).get('0.95', float('nan')):.4f}。"
        "门禁表使用 a=.90 的报告值。",
        "",
        "## 3. 分支 B：E65 开放集门控（#5/#6）",
        "",
        f"特征 {CFG['branch_b']['feature_dim']}-d correlation-logSPD → 逐任务 z-score → "
        f"PCA-{b.get('pca_dim')}（fit_rows 拟合）→ k={b.get('prototype_k')} 原型"
        f"（共 {b.get('n_prototypes')} 个，只用 fit_rows）→ score = -最近距离。",
        f"阈值 τ = 5 块 OOF 的 0.90 分位 = {fmt(b.get('tau_accept_0_90'))}。",
        f"分支 B 单独 in-sample 身份准确率（fit-row 原型）{fmt(b.get('branch_b_train_identity_accuracy'))}。",
        "",
        "| 折（留出身份） | known N | unknown N | AUROC | U@95 | 该折阈值 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in b.get("folds", []):
        lines.append(f"| {fold['held_identity']} | {fold['n_known']} | {fold['n_unknown']} | "
                     f"{fmt(fold['auroc'])} | {fmt(fold['u_at_95'])} | "
                     f"{fold['threshold']:.4f} |")
    lines += [
        "",
        f"合并 20 折：AUROC **{fmt(b.get('loso_pooled_auroc'))}**（known "
        f"{b.get('n_known_pooled')} / unknown {b.get('n_unknown_pooled')}）；"
        f"逐折 AUROC {fmt(b.get('loso_fold_auroc_mean'))} ± "
        f"{fmt(b.get('loso_fold_auroc_std'))}（min {fmt(b.get('loso_fold_auroc_min'))}，"
        f"max {fmt(b.get('loso_fold_auroc_max'))}）。",
        f"U@95：**{fmt(b.get('u_at_95_mean'))} ± {fmt(b.get('u_at_95_std'))}**"
        f"（min {fmt(b.get('u_at_95_min'))}，max {fmt(b.get('u_at_95_max'))}）。",
        "",
        "## 4. Test 一次性冻结审计（#9/#10）",
        "",
        "所有阈值、原型、PCA、模型均在接触 Test 前冻结；Session 列在预测落盘后才读取。",
        "",
        "| 变体 | Session | N | 拒识率 | 接受率 |",
        "|---|---|---:|---:|---:|",
    ]
    audit = test.get("audit", {})
    for name, label in (("primary", "单门控（仅分支 B）"), ("dual_gate", "双门控（+分支 A 置信度）")):
        for code in ("S1", "S2"):
            row = audit.get(name, {}).get(code)
            if row:
                lines.append(f"| {label} | {code} | {row['n']} | {fmt(row['rejection_rate'])} | "
                             f"{fmt(row['acceptance_rate'])} |")
    lines += [
        "",
        f"阈值：τ(分支 B) = {fmt(test.get('threshold_tau_branch_b'))}，"
        f"τ(分支 A max-softmax) = {fmt(test.get('threshold_tau_branch_a'))}。",
        "",
        "双门控把 S1 拒识从 "
        f"{fmt(audit.get('primary', {}).get('S1', {}).get('rejection_rate'))} 抬到 "
        f"{fmt(audit.get('dual_gate', {}).get('S1', {}).get('rejection_rate'))}"
        "（#10 过线），代价是 S2 接受从 "
        f"{fmt(audit.get('primary', {}).get('S2', {}).get('acceptance_rate'))} 掉到 "
        f"{fmt(audit.get('dual_gate', {}).get('S2', {}).get('acceptance_rate'))}"
        "（#9 更差）。两者仍受同一条 ROC 约束，没有同时过线的工作点。",
        "",
        "## 5. 与 E58、E170/E178 对照",
        "",
        "| # | 门禁 | 目标 | E58（外部 22 通道代理） | E170/E178（项目 18 通道） | E180 双分支 |",
        "|---|---|---|---:|---:|---:|",
    ]
    ref58, ref170 = REFERENCE["E58"], REFERENCE["E170/E178"]
    for row in rows:
        key = row["gate"]
        lines.append(f"| {key} | {row['metric']} | {row['target']} | {ref58[key]:.4f} | "
                     f"{ref170[key]:.4f} | {fmt(row['value'])} |")
    e58_pass = sum(1 for key in ref58
                   if ref58[key] >= {"#1": 1.0, "#2": 0.95, "#3": 0.92, "#4": 0.90,
                                     "#5": 0.98, "#6": 0.95, "#7": 0.95, "#8": 0.98,
                                     "#9": 0.75, "#10": 0.75}[key])
    e170_pass = sum(1 for key in ref170
                    if ref170[key] >= {"#1": 1.0, "#2": 0.95, "#3": 0.92, "#4": 0.90,
                                       "#5": 0.98, "#6": 0.95, "#7": 0.95, "#8": 0.98,
                                       "#9": 0.75, "#10": 0.75}[key])
    lines += [
        "",
        f"通过项数：E58 {e58_pass}/10（但 #9/#10 为项目 Test 实测，远未达标）、"
        f"E170/E178 {e170_pass}/10、E180 **{n_pass}/10**。",
        "",
        "## 6. 回答：双分支是否让通过项数增加？",
        "",
    ]
    delta = n_pass - e170_pass
    lines.append(
        f"E179 的结论是「历史上没有任何单一方案能同时覆盖对抗鲁棒与跨会话开放集两侧」；"
        f"在项目 18 通道上最接近的是 E170/E178，通过 {e170_pass} 项（#2/#3/#4）。"
        f"本实验的双分支系统通过 **{n_pass} 项**"
        f"（{'、'.join(row['gate'] for row in passed) if passed else '无'}），"
        f"相对 E170/E178 **{'增加' if delta > 0 else ('减少' if delta < 0 else '持平')} "
        f"{abs(delta)} 项**。")
    gained = [row["gate"] for row in passed if row["gate"] not in ("#2", "#3", "#4")]
    lost = [key for key in ("#2", "#3", "#4")
            if not any(row["gate"] == key and row["passed"] for row in rows)]
    if gained:
        lines.append(f"新增通过的项：{'、'.join(gained)}。")
    if lost:
        lines.append(f"相对 E170/E178 新丢失的项：{'、'.join(lost)}。")
    if not gained:
        lines.append("没有任何新门禁被打开：对抗训练解决了 RA（相对 E170 的 0.4417 有大幅提高），"
                     "但开放集 #5/#6 的差距没有因为换骨干而缩小。")
    lines += [
        "",
        "**这个「+1」要怎么读**：新增的 #8 不是靠识别完美拿到的，而是靠门控拒识"
        "（ASafeR 把判 0 计为安全）。同一批攻击下 #7 仍差 0.0339，说明身份头虽然抗住了扰动"
        "（闭集准确率在 ε=0.02 仍有 0.978），但门控在 a=0.90 上会拒掉约 6% 的干净样本，"
        "直接把 RA 的天花板压到 0.94 以下。也就是说 #7 的瓶颈已经从「表征不鲁棒」变成了"
        "「工作点选择」，这是与 E179 时期（RA 0.4417，表征级失败）不同性质的失败。",
        "",
        "**相对 E170/E178 的逐项变化**：#1 0.9730→0.9988、#2/#4 0.9592→0.9881、#3 0.9590→0.9878、"
        "#5 0.9094→0.9401、#6 0.4882→0.5847、#7 0.4417→0.9161、#8 0.6965→0.9840、"
        "#9 0.7210→0.6335（变差）、#10 0.6775→0.7359（变好）。"
        "所有项都在同向改善，只有 #9 变差：更严的门控提高了 S1 拒识，代价是 S2 接受下降，"
        "这正是 E179 指出的「S1+S2 由同一条 ROC 决定、只能此消彼长」。",
    ]
    targets = {"#1": 1.00, "#2": 0.95, "#3": 0.92, "#4": 0.90, "#5": 0.98,
               "#6": 0.95, "#7": 0.95, "#8": 0.98, "#9": 0.75, "#10": 0.75}
    gaps = "；".join(
        f"{row['gate']} {fmt(row['value'])}（差 {targets[row['gate']] - row['value']:.4f}）"
        for row in rows if row["passed"] is False)
    lines += [
        "",
        f"未通过项与差距：{gaps}。",
        "",
        "## 7. 合规与边界",
        "",
        "- 原型、PCA、逐任务标准化、门控阈值、分支 A max-softmax 阈值全部 Train-only。",
        "- LOSO 严格 `pm & fit_rows`（19 个注册身份的 frac<=0.6 行），评估块 0.6<frac<=0.8 "
        "从未进入原型；PCA 与 z-score 统计量按既有 E170/E178 口径在全部 fit_rows 上冻结。",
        "- Test 只在冻结后一次性评分，#9/#10 的 Session 字段仅在预测 CSV 落盘后读取用于审计，"
        "未参与任何特征/模型/阈值决策。",
        "- #7/#8 是**自建白盒 PGD 代理**，不是官方攻击套件，只能作参考上界/下界证据。",
        "- 未修改任何已有实验目录，未提交 git。",
        "",
        "### 7.1 实现修正记录（全部发生在 Test 之前）",
        "",
        "`run.log` 保留了中间过程，便于核对：",
        "",
        "1. 门控阈值最初把「接受率 a=0.90」误读成「分数的 0.90 分位」，实际等价于 10% 接受率"
        "（Train 窗口块接受率只有 0.131）。改为仓库既有口径 `quantile(ref, 1-a)` 后，"
        "OOF 接受率回到 0.90。",
        "2. 攻击步长方向最初写成了沿 loss 梯度上升，等于在**最小化**攻击目标（出现 RA 0.0000 "
        "这种“攻击越打越安全”的反常结果）。改为沿 loss 梯度下降后恢复正常。",
        "3. 攻击目标最初把身份 margin 与门控 z 分数加权混合，结果梯度被门控项主导、身份攻击几乎无效。"
        "改为 A/B/C/D 四条互不混合的目标后，才分别测到两条分支的真实脆弱面。",
        "",
        "三项都只涉及 Train 侧实现，Test 仅在全部冻结完成后评分一次。",
        "",
        "## 8. Test 官方指标",
        "",
        "**Test 官方 AUC / F1 / C / RA / ASafeR / U / Score = N/A**（Test 无真实标签）。"
        "报告中的 S1 拒识 / S2 接受只是预测工程审计，不是官方真值指标。",
        "",
        "## 9. 文件",
        "",
        "| 文件 | 内容 |",
        "|---|---|",
        "| `run_e180.py` | 全部逻辑（`--stage a/b/test/attack/all/report`） |",
        "| `config.json` | 冻结配置 |",
        "| `results.json` | 全部数值 |",
        "| `model_branch_a.pt` | 分支 A 全量模型 |",
        "| `run.log` | 运行日志 |",
        "| `cache/` | 5 个折模型、OOF logits、分支 B 冻结张量 |",
        "| `test_predictions_primary.csv` / `test_predictions_dual_gate.csv` | Test 预测（工程审计用） |",
        "",
        "耗时（秒）："
        + "、".join(f"{name}={value:.0f}" if value is not None else f"{name}=N/A"
                   for name, value in (results.get("stage_elapsed_seconds") or {}).items())
        + "；合计 {:.0f} s。".format(results.get("runtime_seconds_total", float("nan"))),
        "",
        "```powershell",
        f"& \"{ROOT / '.venv' / 'Scripts' / 'python.exe'}\" \"{OUT / 'run_e180.py'}\" --stage all",
        "```",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_meta(results: dict) -> dict:
    """Recompute the gate table and the accounting fields from the stage results."""
    results["gates"] = gate_table(results)
    stage_elapsed = {name: results.get(name, {}).get("elapsed_seconds")
                     for name in ("branch_a", "branch_b", "attack", "test")}
    results["stage_elapsed_seconds"] = stage_elapsed
    results["runtime_seconds_total"] = float(
        sum(value for value in stage_elapsed.values() if value is not None))
    results["environment"] = {"python": platform.python_version(),
                              "torch": torch.__version__, "numpy": np.__version__,
                              "gpu": torch.cuda.get_device_name(0)}
    return results


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all",
                        choices=("a", "b", "test", "attack", "all", "report"))
    args = parser.parse_args()
    if args.stage == "report":
        results = refresh_meta(load_results())
        save_results(results)
        write_report(results)
        print("report.md rebuilt from results.json")
        return
    if not torch.cuda.is_available():
        parser.error("E180 requires CUDA")
    sys.stdout = Tee(OUT / "run.log")
    log(f"\n===== E180 stage={args.stage} {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    started = time.time()
    if args.stage in ("a", "all"):
        stage_a()
    if args.stage in ("b", "all"):
        stage_b()
    if args.stage in ("attack", "all"):
        stage_attack()
    if args.stage in ("test", "all"):
        stage_test()
    results = refresh_meta(load_results())
    save_results(results)
    write_report(results)
    log(f"done in {time.time() - started:.0f}s -> {RESULTS_PATH} / {REPORT_PATH}")


if __name__ == "__main__":
    main()
