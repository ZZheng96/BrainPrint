"""E53: raw-waveform EEGNet, reciprocal-point open-set loss and robust nested evaluation.

Only Train/session1 is read.  All normalization is trial-local; unknown is never a
21st training class.  Formal attacks are always raw-waveform PGD20/FGSM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CONFIG_PATH = OUT / "config.json"
RESULTS_PATH = OUT / "results.json"
CACHE = ROOT / "BrainPrint_cache" / "e53_raw_eeg_open_robust"
ATTACK_CODES = {"untargeted_fgsm": 1, "untargeted_pgd": 2, "targeted_pgd": 3}
EARLY_ATTACKS = ("untargeted_fgsm_snr20", "untargeted_pgd_snr20", "targeted_pgd_snr20")
COMPONENT_KEYS = (
    "C", "RA", "ASafeR", "A", "U", "direct_RA", "known_acceptance",
    "identity_macro_f1_direct", "identity_macro_f1", "unknown_auroc",
    "unknown_aupr", "unknown_f1", "joint_f1",
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_score(c: float, a: float, u: float) -> float:
    return float(100 * c**0.40 * a**0.35 * u**0.25) if min(c, a, u) > 0 else 0.0


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def train_metadata(config: dict) -> pd.DataFrame:
    metadata = pd.read_csv(ROOT / config["data"]["train_metadata"])
    expected = config["data"]
    if len(metadata) != expected["expected_trials"]:
        raise ValueError("Train trial count mismatch")
    if metadata["subject"].nunique() != expected["expected_subjects"]:
        raise ValueError("Train subject count mismatch")
    if set(metadata["session"]) != {expected["session"]}:
        raise ValueError("Only Train session1 is allowed")
    if set(metadata["fs"]) != {expected["sampling_rate_hz"]}:
        raise ValueError("Sampling rate mismatch")
    if not metadata["path"].str.startswith("Train/").all():
        raise ValueError("Non-Train path in metadata")
    return metadata


def load_train_matrix(path: Path) -> np.ndarray:
    data = np.asarray(loadmat(path, variable_names=["data"])["data"], dtype=np.float32)
    if data.shape != (1200, 18) or not np.isfinite(data).all():
        raise ValueError(f"Invalid Train MAT: {path}")
    return data.T


def load_all_raw(metadata: pd.DataFrame, config: dict, device: torch.device) -> torch.Tensor:
    started = time.time()
    paths = [ROOT / config["data"]["train_root"] / path for path in metadata["path"]]
    output = torch.empty((len(paths), 18, 1200), dtype=torch.float32, device=device)
    with ThreadPoolExecutor(max_workers=int(config["data"]["io_workers"])) as pool:
        batch = int(config["data"]["io_batch_size"])
        for start in range(0, len(paths), batch):
            arrays = list(pool.map(load_train_matrix, paths[start:start + batch]))
            output[start:start + len(arrays)] = torch.as_tensor(
                np.stack(arrays), dtype=torch.float32, device=device
            )
    print(f"Train raw GPU loaded {tuple(output.shape)} in {time.time()-started:.1f}s", flush=True)
    return output


def robust_preprocess(raw: torch.Tensor, floor: float = 1e-8) -> torch.Tensor:
    """B×1×18×1200 -> trial-local differentiable linear detrend + median/MAD scale."""
    x = raw.float()
    single = x.ndim == 2
    if single:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(1)
    if x.ndim != 4 or x.shape[1:] != (1, 18, 1200):
        raise ValueError(f"Expected Bx1x18x1200, got {tuple(x.shape)}")
    t = torch.linspace(-1.0, 1.0, 1200, device=x.device, dtype=x.dtype)
    centered = x - x.mean(dim=-1, keepdim=True)
    slope = (centered * t).sum(dim=-1, keepdim=True) / (t*t).sum()
    detrended = centered - slope*t
    median = detrended.median(dim=-1, keepdim=True).values
    centered = detrended - median
    mad = 1.4826 * centered.abs().median(dim=-1, keepdim=True).values
    rms = centered.square().mean(dim=-1, keepdim=True).sqrt()
    scale = torch.where(mad > floor, mad, rms).clamp_min(floor)
    result = torch.nan_to_num(centered / scale)
    return result[0, 0] if single else result


class EEGNetEmbedding(nn.Module):
    def __init__(self, n_classes: int, config: dict):
        super().__init__()
        cfg = config["model"]
        f1, depth, f2 = int(cfg["temporal_filters"]), int(cfg["depthwise_multiplier"]), int(cfg["separable_filters"])
        self.mad_floor = float(cfg["mad_floor"])
        self.temporal = nn.Conv2d(1, f1, (1, int(cfg["temporal_kernel"])), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.depthwise = nn.Conv2d(f1, f1*depth, (18, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f1*depth)
        self.separable_depth = nn.Conv2d(f1*depth, f1*depth, (1, int(cfg["separable_kernel"])),
                                         padding="same", groups=f1*depth, bias=False)
        self.separable_point = nn.Conv2d(f1*depth, f2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool = nn.AvgPool2d((1, int(cfg["pool"][0])))
        self.dropout = nn.Dropout(float(cfg["dropout"]))
        self.projection = nn.Linear(f2, int(cfg["embedding_dim"]))
        self.prototypes = nn.Parameter(torch.empty(n_classes, int(cfg["embedding_dim"])))
        nn.init.orthogonal_(self.prototypes)
        initial = float(cfg["arpl_radius_initial"])
        self.radius_raw = nn.Parameter(torch.full((n_classes,), math.log(math.expm1(initial))))
        self.n_classes = n_classes

    def embed(self, raw: torch.Tensor) -> torch.Tensor:
        x = robust_preprocess(raw, self.mad_floor)
        x = self.bn1(self.temporal(x))
        x = self.bn2(self.depthwise(x))
        x = self.dropout(self.pool(F.elu(x)))
        x = self.separable_point(self.separable_depth(x))
        x = self.bn3(x)
        x = self.dropout(self.pool(F.elu(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        x = self.projection(x)
        return F.normalize(x, dim=1, eps=1e-12)

    def normalized_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, dim=1, eps=1e-12)

    def scores(self, raw: torch.Tensor) -> torch.Tensor:
        return self.embed(raw) @ self.normalized_prototypes().T

    def open_space_loss(self, raw: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
        embedding = self.embed(raw)
        prototypes = self.normalized_prototypes()
        distance = 1.0 - embedding @ prototypes.T
        true_distance = distance.gather(1, labels[:, None]).squeeze(1)
        other_distance = distance.masked_fill(
            F.one_hot(labels, self.n_classes).bool(), float("inf")
        ).min(dim=1).values
        radius = F.softplus(self.radius_raw)
        true_radius = radius[labels]
        # True distances lie inside the class radius; nearest reciprocal/other
        # distances lie outside it.  This is the preregistered two-sided ARPL term.
        inside = F.relu(true_distance - true_radius + margin)
        outside = F.relu(true_radius - other_distance + margin)
        return (inside + outside).mean()


def initialize_prototypes(model: EEGNetEmbedding, raw: torch.Tensor, labels: torch.Tensor) -> None:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(raw), 256):
            chunks.append(model.embed(raw[start:start+256]))
        embedding = torch.cat(chunks)
        means = torch.stack([embedding[labels == index].mean(0) for index in range(model.n_classes)])
        model.prototypes.copy_(F.normalize(means, dim=1, eps=1e-12))


def stratified_subset(metadata: pd.DataFrame, indices: np.ndarray, per_group: int) -> np.ndarray:
    frame = metadata.iloc[indices][["subject", "task", "trial", "path"]].copy()
    frame["matrix_index"] = indices
    selected = (frame.sort_values(["subject", "task", "trial", "path"])
                .groupby(["subject", "task"], sort=True, as_index=False).head(per_group))
    return np.sort(selected["matrix_index"].to_numpy(dtype=int))


def attack_seed(config: dict, stage: int, outer: int, inner: int, name: str, snr: int, start: int = 0) -> int:
    return int(config["seed_base"] + 100000*stage + 10000*outer + 1000*inner
               + 100*ATTACK_CODES[name] + snr + start)


def raw_attack(model: EEGNetEmbedding, raw: torch.Tensor, labels: torch.Tensor,
               snr: float | torch.Tensor, attack_name: str, steps: int, seed: int,
               target: torch.Tensor | None = None, batch_size: int = 64,
               return_waveforms: bool = False):
    """Seeded per-trial L2 FGSM/PGD through detrend/MAD and the complete network."""
    targeted = attack_name == "targeted_pgd"
    scores_output = torch.empty((len(raw), model.n_classes), device=raw.device, dtype=torch.float32)
    waveforms = torch.empty_like(raw) if return_waveforms else None
    realized, radius_errors = [], []
    was_training = model.training
    model.eval()
    snr_tensor = torch.as_tensor(snr, dtype=torch.float32, device=raw.device)
    if snr_tensor.ndim == 0:
        snr_tensor = snr_tensor.expand(len(raw))
    for start in range(0, len(raw), batch_size):
        stop = min(start + batch_size, len(raw))
        clean = raw[start:stop].detach().float()
        centered = clean - clean.mean(dim=-1, keepdim=True)
        signal_norm = centered.reshape(len(clean), -1).norm(dim=1).clamp_min(1e-12)
        radius = signal_norm / (10.0 ** (snr_tensor[start:stop] / 20.0))
        generator = torch.Generator(device=raw.device)
        generator.manual_seed(int(seed + start))
        delta = torch.zeros_like(clean)
        if steps > 1:
            delta = torch.randn(clean.shape, generator=generator, device=raw.device)
            delta = delta / delta.reshape(len(clean), -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
            random_radius = torch.rand(len(clean), generator=generator, device=raw.device)
            delta = delta * radius.view(-1, 1, 1) * random_radius.view(-1, 1, 1)
        objective_labels = target[start:stop] if targeted else labels[start:stop]
        if targeted and objective_labels is None:
            raise ValueError("Targeted PGD requires targets")
        for _ in range(steps):
            delta = delta.detach().requires_grad_(True)
            scores = model.scores(clean + delta)
            loss = F.cross_entropy(scores / 0.05, objective_labels)
            gradient = torch.autograd.grad(-loss if targeted else loss, delta)[0]
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("Non-finite raw-waveform attack gradient")
            direction = gradient / gradient.reshape(len(gradient), -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
            step_size = radius if steps == 1 else 2.0*radius/steps
            delta = delta.detach() + step_size.view(-1, 1, 1)*direction
            norm = delta.reshape(len(delta), -1).norm(dim=1).clamp_min(1e-12)
            factor = torch.minimum(torch.ones_like(norm), radius/norm)
            delta = delta*factor.view(-1, 1, 1)
        with torch.no_grad():
            final_scores = model.scores(clean + delta)
            if not torch.isfinite(final_scores).all() or not torch.isfinite(delta).all():
                raise FloatingPointError("Non-finite attack output")
            scores_output[start:stop] = final_scores
            norm = delta.reshape(len(delta), -1).norm(dim=1)
            realized.extend((20*torch.log10(signal_norm/norm.clamp_min(1e-12))).cpu().tolist())
            radius_errors.extend(((norm-radius).abs()/radius.clamp_min(1e-12)).cpu().tolist())
            if waveforms is not None:
                waveforms[start:stop] = clean + delta
    if was_training:
        model.train()
    diagnostics = {
        "n": int(len(raw)), "steps": int(steps), "finite_gradients_and_scores": True,
        "realized_snr_db_mean": float(np.mean(realized)), "realized_snr_db_min": float(np.min(realized)),
        "radius_relative_error_max": float(np.max(radius_errors)),
    }
    return scores_output, diagnostics, waveforms


def mixed_training_attack(model: EEGNetEmbedding, raw: torch.Tensor, labels: torch.Tensor,
                          config: dict, seed: int, targeted_fraction: float, steps: int) -> torch.Tensor:
    generator = torch.Generator(device=raw.device)
    generator.manual_seed(int(seed))
    snr = torch.where(torch.rand(len(raw), generator=generator, device=raw.device) < 0.5, 20.0, 30.0)
    targeted = torch.rand(len(raw), generator=generator, device=raw.device) < targeted_fraction
    targets = (labels + 1) % model.n_classes
    objective_labels = torch.where(targeted, targets, labels)
    # Attack all samples in one call; a per-sample sign gives mixed targeted and
    # untargeted gradients without changing their subsequent true training labels.
    clean = raw.detach().float()
    centered = clean - clean.mean(dim=-1, keepdim=True)
    signal_norm = centered.reshape(len(clean), -1).norm(dim=1).clamp_min(1e-12)
    radius = signal_norm / (10.0**(snr/20.0))
    delta = torch.randn(clean.shape, generator=generator, device=raw.device)
    delta = delta/delta.reshape(len(clean), -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
    delta = delta*radius.view(-1, 1, 1)*torch.rand(len(clean), generator=generator, device=raw.device).view(-1, 1, 1)
    was_training = model.training
    model.eval()
    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        per_sample = F.cross_entropy(model.scores(clean+delta)/0.05, objective_labels, reduction="none")
        objective = (per_sample*torch.where(targeted, -1.0, 1.0)).sum()
        gradient = torch.autograd.grad(objective, delta)[0]
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Non-finite mixed PGD gradient")
        direction = gradient/gradient.reshape(len(gradient), -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
        delta = delta.detach() + (2.0*radius/steps).view(-1, 1, 1)*direction
        norm = delta.reshape(len(delta), -1).norm(dim=1).clamp_min(1e-12)
        delta = delta*torch.minimum(torch.ones_like(norm), radius/norm).view(-1, 1, 1)
    if was_training:
        model.train()
    return (clean+delta).detach()


def predict_scores(model: EEGNetEmbedding, raw: torch.Tensor, indices: np.ndarray | None = None,
                   batch_size: int = 512) -> np.ndarray:
    source = raw if indices is None else raw[torch.as_tensor(indices, device=raw.device)]
    output = np.empty((len(source), model.n_classes), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(source), batch_size):
            output[start:start+batch_size] = model.scores(source[start:start+batch_size]).cpu().numpy()
    return output


def confidence_values(scores: np.ndarray) -> dict[str, np.ndarray]:
    ordered = np.sort(np.asarray(scores, dtype=np.float64), axis=1)
    maximum, second = ordered[:, -1], ordered[:, -2]
    return {"max_cosine": maximum, "margin": maximum-second,
            "distance_ratio": (1.0-second)/np.maximum(1.0-maximum, 1e-6)}


def make_run(model: EEGNetEmbedding, raw: torch.Tensor, metadata: pd.DataFrame,
             subjects: np.ndarray, known_np: np.ndarray, unknown_np: np.ndarray,
             classes: np.ndarray, config: dict, stage: int, outer: int, inner: int,
             per_group: int, steps: int, attack_names: tuple[str, ...] | list[str],
             include_full: bool) -> dict:
    known_subset = stratified_subset(metadata, known_np, per_group)
    unknown_subset = stratified_subset(metadata, unknown_np, per_group)
    known_index = torch.as_tensor(known_subset, device=raw.device)
    unknown_index = torch.as_tensor(unknown_subset, device=raw.device)
    run = {
        "classes": np.asarray(classes, dtype=int), "known_true": subjects[known_subset],
        "known_indices": known_subset, "unknown_indices": unknown_subset,
        "known_scores": predict_scores(model, raw, known_subset),
        "unknown_scores": predict_scores(model, raw, unknown_subset), "attacks": {},
    }
    if include_full:
        run["clean_full"] = {
            "known_true": subjects[known_np], "known_scores": predict_scores(model, raw, known_np),
            "unknown_scores": predict_scores(model, raw, unknown_np),
            "n_known": int(len(known_np)), "n_unknown": int(len(unknown_np)),
        }
    true = subjects[known_subset]
    class_to_index = {int(label): index for index, label in enumerate(classes)}
    labels = torch.as_tensor([class_to_index[int(v)] for v in true], dtype=torch.long, device=raw.device)
    target = np.asarray([classes[(class_to_index[int(v)]+1) % len(classes)] for v in true], dtype=int)
    target_indices = torch.as_tensor([class_to_index[int(v)] for v in target], dtype=torch.long, device=raw.device)
    raw_known = raw[known_index]
    for full_name in attack_names:
        base_name, snr_text = full_name.rsplit("_snr", 1)
        snr = int(snr_text)
        actual_steps = 1 if base_name == "untargeted_fgsm" else steps
        seed = attack_seed(config, stage, outer, inner, base_name, snr)
        scores, diagnostics, _ = raw_attack(
            model, raw_known, labels, snr, base_name, actual_steps, seed,
            target=target_indices if base_name == "targeted_pgd" else None,
            batch_size=int(config["attacks"]["batch_size"]),
        )
        run["attacks"][full_name] = {
            "scores": scores.cpu().numpy(), "diagnostics": diagnostics,
            "target": target if base_name == "targeted_pgd" else None,
        }
        print(f"  {full_name} N={len(true)} SNR={diagnostics['realized_snr_db_mean']:.2f}", flush=True)
    del raw_known, known_index, unknown_index
    return run


def embedding_loss(model: EEGNetEmbedding, embedding: torch.Tensor, labels: torch.Tensor,
                   temperature: float, arpl_weight: float, margin: float) -> torch.Tensor:
    prototypes = model.normalized_prototypes()
    scores = embedding @ prototypes.T
    loss = F.cross_entropy(scores/temperature, labels)
    if arpl_weight <= 0:
        return loss
    distance = 1.0-scores
    true_distance = distance.gather(1, labels[:, None]).squeeze(1)
    other_distance = distance.masked_fill(F.one_hot(labels, model.n_classes).bool(), float("inf")).min(1).values
    radius = F.softplus(model.radius_raw)[labels]
    open_loss = (F.relu(true_distance-radius+margin)+F.relu(radius-other_distance+margin)).mean()
    return loss + arpl_weight*open_loss


def refresh_training_cache(model: EEGNetEmbedding, fit_raw: torch.Tensor, fit_labels: torch.Tensor,
                           config: dict, seed: int, targeted_fraction: float) -> torch.Tensor:
    cache = torch.empty_like(fit_raw)
    was_training = model.training
    model.eval()
    batch = int(config["training"]["batch_size"])
    for start in range(0, len(fit_raw), batch):
        stop = min(start+batch, len(fit_raw))
        cache[start:stop] = mixed_training_attack(
            model, fit_raw[start:stop], fit_labels[start:stop], config, seed+start,
            targeted_fraction, int(config["training"]["cache_pgd_steps"]),
        )
    if was_training:
        model.train()
    return cache


def train_model(variant: str, raw: torch.Tensor, subjects: np.ndarray, fit_np: np.ndarray,
                known_np: np.ndarray | None, unknown_np: np.ndarray | None,
                classes: np.ndarray, metadata: pd.DataFrame, config: dict,
                outer: int, inner: int, fixed_epochs: int | None = None):
    seed = int(config["seed_base"] + 1000*outer + inner)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = EEGNetEmbedding(len(classes), config).to(raw.device)
    class_to_index = {int(label): index for index, label in enumerate(classes)}
    fit_index = torch.as_tensor(fit_np, device=raw.device)
    fit_raw = raw[fit_index]
    fit_labels = torch.as_tensor([class_to_index[int(v)] for v in subjects[fit_np]],
                                 dtype=torch.long, device=raw.device)
    initialize_prototypes(model, fit_raw, fit_labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]),
                                  weight_decay=float(config["training"]["weight_decay"]))
    arpl_weight = 0.0 if variant == "no_arpl" else float(config["model"]["arpl_weight"])
    targeted_fraction = 0.0 if variant == "untargeted_at" else float(config["training"]["main_targeted_fraction"])
    temperature = float(config["model"]["cosine_temperature"])
    margin = float(config["model"]["arpl_margin"])
    batch = int(config["training"]["batch_size"])
    max_epochs = int(fixed_epochs or config["training"]["max_epochs"])
    online = config["training"]["attack_training_mode"] == "online_pgd7"
    rng = np.random.default_rng(seed)
    best = {"key": None, "epoch": max_epochs, "state": None, "early": None}
    stale = 0
    history, adversarial_cache = [], None
    started = time.time()
    for epoch in range(1, max_epochs+1):
        model.train()
        epoch_started = time.time()
        if not online and (adversarial_cache is None or (epoch-1) % int(config["training"]["cache_refresh_epochs"]) == 0):
            refresh_seed = int(config["seed_base"] + 300000 + 10000*outer + 1000*inner + 100*epoch)
            adversarial_cache = refresh_training_cache(model, fit_raw, fit_labels, config, refresh_seed, targeted_fraction)
        order = rng.permutation(len(fit_np))
        epoch_loss = 0.0
        for start in range(0, len(order), batch):
            selected = torch.as_tensor(order[start:start+batch], device=raw.device)
            labels = fit_labels[selected]
            clean = fit_raw[selected]
            if online:
                attacked = mixed_training_attack(
                    model, clean, labels, config,
                    int(config["seed_base"] + 300000 + 10000*outer + 1000*inner + 100000*epoch + start),
                    targeted_fraction, int(config["training"]["online_pgd_steps"]),
                )
            else:
                attacked = adversarial_cache[selected]
            clean_embedding = model.embed(clean)
            adv_embedding = model.embed(attacked)
            clean_loss = embedding_loss(model, clean_embedding, labels, temperature, arpl_weight, margin)
            adv_loss = embedding_loss(model, adv_embedding, labels, temperature, arpl_weight, margin)
            loss = float(config["training"]["clean_ce_weight"])*clean_loss + float(config["training"]["adversarial_ce_weight"])*adv_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            optimizer.step()
            epoch_loss += float(loss.detach())*len(selected)
        row = {"epoch": epoch, "loss": epoch_loss/len(fit_np), "seconds": time.time()-epoch_started}
        if fixed_epochs is None and (epoch == 1 or epoch % int(config["training"]["early_eval_every_epochs"]) == 0):
            # Epoch 1 is diagnostic only; model selection starts at epoch 5.
            if epoch > 1:
                early = make_run(model, raw, metadata, subjects, known_np, unknown_np, classes,
                                 config, 10+epoch, outer, inner,
                                 int(config["training"]["early_trials_per_subject_task"]),
                                 int(config["training"]["early_pgd_steps"]), EARLY_ATTACKS, False)
                selection = select_threshold([early], EARLY_ATTACKS, config)
                clean = clean_metrics([early], selection["score_name"], selection["threshold"],
                                      selection["q05_threshold"])
                key = (selection["macro"]["Score"], clean["direct_C"], clean["unknown_auroc"],
                       clean["q05_U"], -epoch)
                row.update({"Score": key[0], "direct_C": key[1], "unknown_auroc": key[2], "U_at_95": key[3]})
                if best["key"] is None or key > best["key"]:
                    best = {"key": key, "epoch": epoch, "early": {"selection": selection, "clean": clean},
                            "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
                    stale = 0
                else:
                    stale += 1
        history.append(row)
        print(f"  {variant} o{outer}i{inner} epoch={epoch} loss={row['loss']:.4f} "
              f"sec={row['seconds']:.1f} best={best['epoch']}", flush=True)
        if fixed_epochs is None and stale >= int(config["training"]["early_stopping_patience_evaluations"]):
            break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    fit_scores = predict_scores(model, fit_raw)
    fit_pred = np.asarray(classes)[fit_scores.argmax(1)]
    info = {
        "variant": variant, "seed": seed, "best_epoch": int(best["epoch"]), "history": history,
        "best_early": best["early"], "attack_training_mode": config["training"]["attack_training_mode"],
        "n_fit": int(len(fit_np)), "fit_accuracy": float(np.mean(fit_pred == subjects[fit_np])),
        "fit_identity_macro_f1": float(f1_score(subjects[fit_np], fit_pred, labels=classes,
                                                   average="macro", zero_division=0)),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "elapsed_seconds": time.time()-started,
    }
    del fit_raw, fit_index, adversarial_cache, optimizer
    torch.cuda.empty_cache()
    return model, info


def concat_runs(runs: list[dict], accessor) -> np.ndarray:
    return np.concatenate([accessor(run) for run in runs])


def components_many(runs: list[dict], attack_name: str, score_name: str, threshold: float) -> dict:
    known = concat_runs(runs, lambda r: confidence_values(r["known_scores"])[score_name])
    unknown = concat_runs(runs, lambda r: confidence_values(r["unknown_scores"])[score_name])
    adv = concat_runs(runs, lambda r: confidence_values(r["attacks"][attack_name]["scores"])[score_name])
    known_pred = concat_runs(runs, lambda r: r["classes"][r["known_scores"].argmax(1)])
    adv_pred = concat_runs(runs, lambda r: r["classes"][r["attacks"][attack_name]["scores"].argmax(1)])
    true = concat_runs(runs, lambda r: r["known_true"])
    clean_accept, adv_accept = known >= threshold, adv >= threshold
    clean_correct, adv_correct = known_pred == true, adv_pred == true
    c = float(np.mean(clean_correct & clean_accept))
    ra = float(np.mean(adv_correct & adv_accept))
    asafer = float(np.mean(adv_correct | ~adv_accept))
    a = 0.7*ra+0.3*asafer
    u = float(np.mean(unknown < threshold))
    return {"C": c, "RA": ra, "ASafeR": asafer, "A": a, "U": u,
            "direct_RA": float(np.mean(adv_correct)), "known_acceptance": float(np.mean(clean_accept)),
            "Score": official_score(c, a, u)}


def select_threshold(runs: list[dict], attack_names: tuple[str, ...] | list[str], config: dict) -> dict:
    best = None
    summaries = []
    for score_index, score_name in enumerate(config["threshold"]["score_candidates"]):
        known = concat_runs(runs, lambda r: confidence_values(r["known_scores"])[score_name])
        thresholds = np.unique(np.quantile(known, np.linspace(0.0, 0.05, 101)))
        for threshold in thresholds:
            per_attack = {name: components_many(runs, name, score_name, float(threshold)) for name in attack_names}
            macro = {key: float(np.mean([row[key] for row in per_attack.values()]))
                     for key in ("C", "RA", "ASafeR", "A", "U", "direct_RA", "known_acceptance")}
            macro["Score"] = official_score(macro["C"], macro["A"], macro["U"])
            if macro["known_acceptance"] < 0.95-1e-12:
                continue
            key = (macro["Score"], macro["RA"], macro["ASafeR"], macro["C"], macro["U"], -score_index)
            if best is None or key > best[0]:
                best = (key, score_name, float(threshold), macro)
        summaries.append({"score_name": score_name})
    if best is None:
        raise RuntimeError("No threshold satisfies clean-known acceptance")
    known = concat_runs(runs, lambda r: confidence_values(r["known_scores"])[best[1]])
    return {"score_name": best[1], "threshold": best[2],
            "q05_threshold": float(np.quantile(known, 0.05, method="lower")),
            "macro": best[3], "score_candidate_summary": summaries,
            "selection_attacks": list(attack_names)}


def detection_summary(known: np.ndarray, unknown: np.ndarray) -> dict:
    labels = np.concatenate([np.ones(len(known), dtype=int), np.zeros(len(unknown), dtype=int)])
    values = np.concatenate([known, unknown])
    fpr, tpr, _ = roc_curve(labels, values)
    fnr = 1.0-tpr
    index = int(np.argmin(np.abs(fpr-fnr)))
    return {"unknown_auroc": float(roc_auc_score(labels, values)),
            "unknown_aupr": float(average_precision_score(1-labels, -values)),
            "known_aupr": float(average_precision_score(labels, values)),
            "eer": float((fpr[index]+fnr[index])/2)}


def evaluate_attack(runs: list[dict], attack_name: str, score_name: str, threshold: float) -> dict:
    result = components_many(runs, attack_name, score_name, threshold)
    true = concat_runs(runs, lambda r: r["known_true"])
    direct = concat_runs(runs, lambda r: r["classes"][r["attacks"][attack_name]["scores"].argmax(1)])
    confidence = concat_runs(runs, lambda r: confidence_values(r["attacks"][attack_name]["scores"])[score_name])
    unknown_direct = concat_runs(runs, lambda r: r["classes"][r["unknown_scores"].argmax(1)])
    unknown_conf = concat_runs(runs, lambda r: confidence_values(r["unknown_scores"])[score_name])
    clean_direct = concat_runs(runs, lambda r: r["classes"][r["known_scores"].argmax(1)])
    clean_conf = concat_runs(runs, lambda r: confidence_values(r["known_scores"])[score_name])
    pred = direct.copy()
    pred[confidence < threshold] = 0
    unknown_pred = unknown_direct.copy()
    unknown_pred[unknown_conf < threshold] = 0
    clean_pred = clean_direct.copy()
    clean_pred[clean_conf < threshold] = 0
    classes = np.unique(np.concatenate([r["classes"] for r in runs]))
    detection = np.concatenate([np.ones(len(pred), dtype=int), np.zeros(len(unknown_pred), dtype=int)])
    accepted = np.concatenate([pred != 0, unknown_pred != 0])
    result.update({
        "identity_macro_f1_direct": float(f1_score(true, direct, labels=classes, average="macro", zero_division=0)),
        "identity_macro_f1": float(f1_score(true, pred, labels=classes, average="macro", zero_division=0)),
        "clean_identity_macro_f1": float(f1_score(true, clean_pred, labels=classes, average="macro", zero_division=0)),
        **detection_summary(confidence, unknown_conf),
        "unknown_f1": float(f1_score(detection, accepted, pos_label=0, zero_division=0)),
        "joint_f1": float(f1_score(np.concatenate([true, np.zeros(len(unknown_pred), dtype=int)]),
                                     np.concatenate([pred, unknown_pred]), labels=np.concatenate([[0], classes]),
                                     average="macro", zero_division=0)),
        "N_clean": int(len(clean_pred)), "N_adversarial": int(len(pred)), "N_unknown": int(len(unknown_pred)),
        "all_rejection": bool(np.all(pred == 0) and np.all(clean_pred == 0)),
    })
    if attack_name.startswith("targeted"):
        target = concat_runs(runs, lambda r: r["attacks"][attack_name]["target"])
        result["targeted_direct_false_authentication"] = float(np.mean(direct == target))
        result["targeted_false_authentication"] = float(np.mean(pred == target))
    return result


def macro_from_attacks(per_attack: dict[str, dict]) -> dict:
    result = {key: float(np.mean([row[key] for row in per_attack.values()])) for key in COMPONENT_KEYS}
    result["Score"] = official_score(result["C"], result["A"], result["U"])
    result["attack_score_mean"] = float(np.mean([row["Score"] for row in per_attack.values()]))
    result["N_clean_per_attack_sum"] = int(sum(row["N_clean"] for row in per_attack.values()))
    result["N_adversarial_total"] = int(sum(row["N_adversarial"] for row in per_attack.values()))
    result["N_unknown_per_attack_sum"] = int(sum(row["N_unknown"] for row in per_attack.values()))
    targeted = [row for name, row in per_attack.items() if name.startswith("targeted")]
    result["targeted_direct_false_authentication"] = float(np.mean([row["targeted_direct_false_authentication"] for row in targeted]))
    result["targeted_false_authentication"] = float(np.mean([row["targeted_false_authentication"] for row in targeted]))
    result["all_rejection"] = bool(all(row["all_rejection"] for row in per_attack.values()))
    return result


def evaluate_point(runs: list[dict], score_name: str, threshold: float,
                   attack_names: tuple[str, ...] | list[str]) -> dict:
    per_attack = {name: evaluate_attack(runs, name, score_name, threshold) for name in attack_names}
    return {"per_attack": per_attack, "macro": macro_from_attacks(per_attack)}


def clean_metrics(runs: list[dict], score_name: str, threshold: float, q05_threshold: float) -> dict:
    views = [run.get("clean_full", run) for run in runs]
    true = concat_runs(views, lambda r: r["known_true"])
    direct = concat_runs(runs, lambda r, i=None: r["classes"][r.get("clean_full", r)["known_scores"].argmax(1)])
    unknown_direct = concat_runs(runs, lambda r: r["classes"][r.get("clean_full", r)["unknown_scores"].argmax(1)])
    known = concat_runs(views, lambda r: confidence_values(r["known_scores"])[score_name])
    unknown = concat_runs(views, lambda r: confidence_values(r["unknown_scores"])[score_name])
    classes = np.unique(np.concatenate([r["classes"] for r in runs]))
    detection = np.concatenate([np.ones(len(known), dtype=int), np.zeros(len(unknown), dtype=int)])
    result = {
        "direct_C": float(np.mean(direct == true)), "continuous_block_accuracy": float(np.mean(direct == true)),
        "identity_macro_f1_direct": float(f1_score(true, direct, labels=classes, average="macro", zero_division=0)),
        **detection_summary(known, unknown), "N_known": int(len(known)), "N_unknown": int(len(unknown)),
    }
    for prefix, value in (("", threshold), ("q05_", q05_threshold)):
        known_pred, unknown_pred = direct.copy(), unknown_direct.copy()
        known_pred[known < value] = 0
        unknown_pred[unknown < value] = 0
        accepted = np.concatenate([known_pred != 0, unknown_pred != 0])
        result.update({
            f"{prefix}threshold": float(value), f"{prefix}C": float(np.mean(known_pred == true)),
            f"{prefix}U": float(np.mean(unknown_pred == 0)),
            f"{prefix}known_acceptance": float(np.mean(known_pred != 0)),
            f"{prefix}identity_macro_f1": float(f1_score(true, known_pred, labels=classes, average="macro", zero_division=0)),
            f"{prefix}unknown_f1": float(f1_score(detection, accepted, pos_label=0, zero_division=0)),
            f"{prefix}joint_f1": float(f1_score(np.concatenate([true, np.zeros(len(unknown_pred), dtype=int)]),
                                                  np.concatenate([known_pred, unknown_pred]),
                                                  labels=np.concatenate([[0], classes]), average="macro", zero_division=0)),
            f"{prefix}all_rejection": bool(np.all(known_pred == 0)),
        })
    return result


def fold_mean(rows: list[dict], keys: list[str] | tuple[str, ...] | None = None) -> dict:
    keys = list(keys or rows[0].keys())
    output = {}
    for key in keys:
        if isinstance(rows[0][key], (float, int, np.floating, np.integer)) and not isinstance(rows[0][key], bool):
            output[key] = float(np.mean([row[key] for row in rows]))
    return output


def artifact_path(variant: str, outer: int, inner: int) -> Path:
    name = f"{variant}_outer{outer}_inner{inner}.pt" if inner else f"{variant}_outer{outer}_full.pt"
    return CACHE / name


def save_artifact(path: Path, model: EEGNetEmbedding, info: dict, run: dict | None) -> None:
    torch.save({"config_sha256": sha256(CONFIG_PATH), "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                "classes": run["classes"] if run is not None else np.arange(1, 21),
                "info": info, "run": run}, path)


def load_artifact(path: Path, device: torch.device, need_model: bool = False):
    item = torch.load(path, map_location="cpu", weights_only=False)
    if item["config_sha256"] != sha256(CONFIG_PATH):
        raise ValueError(f"Frozen config changed; rebuild required: {path}")
    model = None
    if need_model:
        model = EEGNetEmbedding(len(item["classes"]), load_config()).to(device)
        model.load_state_dict(item["model_state"])
        model.eval()
    return item["info"], item["run"], model


def numerical_checks(model: EEGNetEmbedding, raw: torch.Tensor, indices: np.ndarray) -> dict:
    selected = torch.as_tensor(indices[:8], device=raw.device)
    batch = model.scores(raw[selected])
    singles = torch.cat([model.scores(raw[selected[i:i+1]]) for i in range(len(selected))])
    permutation = torch.randperm(len(selected), device=raw.device)
    permuted = model.scores(raw[selected[permutation]])
    inverse = torch.argsort(permutation)
    check_raw = raw[selected[:2]].detach().clone().requires_grad_(True)
    generator = torch.Generator(device=raw.device)
    generator.manual_seed(20261113)
    direction = torch.randn(check_raw.shape, generator=generator, device=raw.device)
    direction = direction/direction.reshape(2, -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
    weights = torch.randn((2, model.n_classes), generator=generator, device=raw.device)
    analytic = (torch.autograd.grad((model.scores(check_raw)*weights).sum(), check_raw)[0]*direction).sum().item()
    epsilon = 1e-3
    with torch.no_grad():
        plus = (model.scores(raw[selected[:2]]+epsilon*direction)*weights).sum().item()
        minus = (model.scores(raw[selected[:2]]-epsilon*direction)*weights).sum().item()
    finite_difference = (plus-minus)/(2*epsilon)
    return {
        "single_vs_batch_max_abs": float((batch-singles).abs().max()),
        "permutation_max_abs": float((batch-permuted[inverse]).abs().max()),
        "single_vs_batch_labels_equal": bool(torch.equal(batch.argmax(1), singles.argmax(1))),
        "finite_difference_analytic": float(analytic), "finite_difference_numeric": float(finite_difference),
        "finite_difference_relative_error": float(abs(analytic-finite_difference)/max(abs(analytic), abs(finite_difference), 1e-8)),
        "all_scores_finite": bool(torch.isfinite(batch).all()), "n": int(len(selected)),
    }


def get_inner_fold(variant: str, outer: int, inner: int, raw: torch.Tensor, metadata: pd.DataFrame,
                   subjects: np.ndarray, forward: np.ndarray, inner_block: np.ndarray,
                   known_subjects: np.ndarray, inner_unknown: np.ndarray, config: dict,
                   checks: bool = False):
    path = artifact_path(variant, outer, inner)
    if path.exists():
        info, run, model = load_artifact(path, raw.device, need_model=checks)
        print(f"Reuse {path.name}", flush=True)
    else:
        enrolled = np.setdiff1d(known_subjects, inner_unknown)
        enrolled_mask = np.isin(subjects, enrolled)
        fit_np = np.flatnonzero(enrolled_mask & ~forward & ~inner_block)
        known_np = np.flatnonzero(enrolled_mask & inner_block)
        unknown_np = np.flatnonzero(np.isin(subjects, inner_unknown) & ~forward)
        model, info = train_model(variant, raw, subjects, fit_np, known_np, unknown_np,
                                  enrolled, metadata, config, outer, inner)
        run = make_run(model, raw, metadata, subjects, known_np, unknown_np, enrolled, config,
                       1, outer, inner, int(config["protocol"]["formal_trials_per_subject_task"]),
                       int(config["attacks"]["pgd_steps"]), config["attacks"]["order"], True)
        run.update({"registered_subjects": enrolled.tolist(), "unknown_subjects": inner_unknown.tolist(),
                    "n_fit": int(len(fit_np))})
        save_artifact(path, model, info, run)
    if checks:
        indices = run["known_indices"]
        info["numerical_checks"] = numerical_checks(model, raw, indices)
        torch.cuda.synchronize()
        started = time.time()
        sample = raw[torch.as_tensor(indices[:64], device=raw.device)]
        true = run["known_true"][:64]
        labels = torch.as_tensor([list(run["classes"]).index(int(v)) for v in true], device=raw.device)
        mixed_training_attack(model, sample, labels, config, 20261113, 0.5,
                              int(config["training"]["online_pgd_steps"]))
        torch.cuda.synchronize()
        info["online_pgd7_batch64_seconds_after_warmup"] = time.time()-started
        save_artifact(path, model, info, run)
    if model is not None:
        del model
    torch.cuda.empty_cache()
    return info, run


def get_full_train(variant: str, raw: torch.Tensor, metadata: pd.DataFrame, subjects: np.ndarray,
                   best_epochs: list[int], config: dict):
    path = CACHE / f"{variant}_all20_capacity.pt"
    if path.exists():
        info, _, _ = load_artifact(path, raw.device)
        return info
    epochs = max(1, int(np.rint(np.median(best_epochs))))
    classes = np.unique(subjects)
    model, info = train_model(variant, raw, subjects, np.arange(len(subjects)), None, None,
                              classes, metadata, config, 0, 0, fixed_epochs=epochs)
    info["fixed_epochs_from_inner_median"] = epochs
    info["strict_100pct"] = bool(info["fit_accuracy"] == 1.0)
    save_artifact(path, model, info, None)
    del model
    torch.cuda.empty_cache()
    return info


def gates_for(summary: dict, full_train: dict, targets: dict) -> dict:
    macro, clean = summary["macro_fold_mean"], summary["clean_fold_mean"]
    gates = {
        "train_fit_100pct": bool(full_train["fit_accuracy"] == 1.0),
        "direct_C": bool(clean["direct_C"] >= targets["C"]),
        "threshold_C": bool(macro["C"] >= targets["C"]),
        "identity_f1_direct": bool(clean["identity_macro_f1_direct"] >= targets["identity_f1"]),
        "continuous_block_accuracy": bool(clean["continuous_block_accuracy"] >= targets["continuous_block_accuracy"]),
        "unknown_auroc": bool(clean["unknown_auroc"] >= targets["unknown_auroc"]),
        "U_at_95": bool(clean["q05_U"] >= targets["U_at_95"]),
        "direct_RA": bool(macro["direct_RA"] >= targets["direct_RA"]),
        "threshold_RA": bool(macro["RA"] >= targets["threshold_RA"]),
        "ASafeR": bool(macro["ASafeR"] >= targets["ASafeR"]),
        "not_all_rejection": not bool(summary["all_rejection"]),
    }
    gates["all_pass"] = bool(all(gates.values()))
    return gates


def base_result(config: dict) -> dict:
    return {
        "status": "running", "experiment": "E53 raw EEG open robust", "date": "2026-11-13",
        "data_evidence": config["reporting"]["evidence_boundary"],
        "config_sha256": sha256(CONFIG_PATH), "script_sha256": sha256(Path(__file__).resolve()),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                        "gpu": torch.cuda.get_device_name(0), "compute": "raw data/model/attacks on CUDA; metrics on CPU"},
        "variants": {}, "outer": "N/A: gated until all main inner gates pass",
        "test_metrics": "N/A: Test was not read and has no true labels",
    }


def run_inner_stage(raw: torch.Tensor, metadata: pd.DataFrame, config: dict,
                    variants: list[str], smoke: bool = False) -> dict:
    subjects = metadata["subject"].to_numpy(dtype=int)
    trial = metadata["trial"].to_numpy()
    maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    forward = trial > 0.8*maximum
    inner_block = (trial > 0.6*maximum) & (trial <= 0.8*maximum)
    all_subjects = np.unique(subjects)
    outer_groups = list(np.array_split(np.random.default_rng(42).permutation(all_subjects), 5))
    if smoke:
        outer_groups = outer_groups[:1]
    result = base_result(config)
    for variant in variants:
        pairs, all_infos = [], []
        for outer_index, outer_unknown in enumerate(outer_groups):
            outer = outer_index+1
            known_subjects = np.setdiff1d(all_subjects, outer_unknown)
            inner_order = np.random.default_rng(100+outer).permutation(known_subjects)
            splits = list(np.array_split(inner_order, 2))
            if smoke:
                splits = splits[:1]
            runs, infos = [], []
            for inner_index, inner_unknown in enumerate(splits):
                inner = inner_index+1
                info, run = get_inner_fold(variant, outer, inner, raw, metadata, subjects, forward,
                                           inner_block, known_subjects, inner_unknown, config,
                                           checks=smoke)
                infos.append(info)
                runs.append(run)
                all_infos.append(info)
            selection = select_threshold(runs, config["attacks"]["order"], config)
            evaluated = evaluate_point(runs, selection["score_name"], selection["threshold"], config["attacks"]["order"])
            clean = clean_metrics(runs, selection["score_name"], selection["threshold"], selection["q05_threshold"])
            pair = {"outer_fold": outer, "train_info": infos, "selection": selection,
                    "clean": clean, **evaluated,
                    "attack_diagnostics": [{name: run["attacks"][name]["diagnostics"]
                                            for name in config["attacks"]["order"]} for run in runs]}
            pairs.append(pair)
            print(f"INNER {variant} outer={outer}: Score={evaluated['macro']['Score']:.3f} "
                  f"directRA={evaluated['macro']['direct_RA']:.4f} RA={evaluated['macro']['RA']:.4f} "
                  f"ASafeR={evaluated['macro']['ASafeR']:.4f} C={clean['direct_C']:.4f} "
                  f"AUROC={clean['unknown_auroc']:.4f} U@95={clean['q05_U']:.4f}", flush=True)
            result["variants"][variant] = {"folds": pairs}
            result["status"] = "smoke_partial" if smoke else "inner_partial"
            write_json(RESULTS_PATH, result)
        summary = {
            "macro_fold_mean": fold_mean([p["macro"] for p in pairs]),
            "clean_fold_mean": fold_mean([p["clean"] for p in pairs]),
            "per_attack_fold_mean": {name: fold_mean([p["per_attack"][name] for p in pairs])
                                     for name in config["attacks"]["order"]},
            "all_rejection": bool(any(p["macro"]["all_rejection"] for p in pairs)),
        }
        result["variants"][variant].update(summary)
        if not smoke:
            full = get_full_train(variant, raw, metadata, subjects,
                                  [info["best_epoch"] for info in all_infos], config)
            result["variants"][variant]["full_train_fit"] = {
                "accuracy": full["fit_accuracy"], "identity_20class_macro_f1": full["fit_identity_macro_f1"],
                "error_count": int(round(len(subjects)*(1-full["fit_accuracy"]))), "N": len(subjects),
                "strict_100pct": full["strict_100pct"], "fixed_epochs": full["fixed_epochs_from_inner_median"],
                "parameter_count": full["parameter_count"], "history": full["history"],
            }
            result["variants"][variant]["gates"] = gates_for(summary, full, config["targets"])
        result["status"] = "smoke_complete" if smoke else ("inner_partial" if variant != variants[-1] else "inner_complete")
        write_json(RESULTS_PATH, result)
    return result


def run_outer_stage(result: dict, raw: torch.Tensor, metadata: pd.DataFrame, config: dict) -> dict:
    main = result["variants"]["main"]
    if not main["gates"]["all_pass"]:
        result["outer"] = "N/A: at least one preregistered main inner gate failed; outer was not run"
        result["status"] = "complete_inner_gates_failed"
        write_json(RESULTS_PATH, result)
        return result
    subjects = metadata["subject"].to_numpy(dtype=int)
    trial = metadata["trial"].to_numpy()
    maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    forward = trial > 0.8*maximum
    all_subjects = np.unique(subjects)
    outer_groups = list(np.array_split(np.random.default_rng(42).permutation(all_subjects), 5))
    folds = []
    for outer_index, unknown_subjects in enumerate(outer_groups):
        outer = outer_index+1
        path = artifact_path("main", outer, 0)
        selection = main["folds"][outer_index]["selection"]
        if path.exists():
            info, run, _ = load_artifact(path, raw.device)
        else:
            classes = np.setdiff1d(all_subjects, unknown_subjects)
            enrolled_mask = np.isin(subjects, classes)
            fit_np = np.flatnonzero(enrolled_mask & ~forward)
            known_np = np.flatnonzero(enrolled_mask & forward)
            unknown_np = np.flatnonzero(np.isin(subjects, unknown_subjects))
            epochs = max(1, int(np.rint(np.median([x["best_epoch"] for x in main["folds"][outer_index]["train_info"]]))))
            model, info = train_model("main", raw, subjects, fit_np, None, None, classes,
                                      metadata, config, outer, 0, fixed_epochs=epochs)
            info["fixed_epochs_from_corresponding_inner_median"] = epochs
            run = make_run(model, raw, metadata, subjects, known_np, unknown_np, classes, config,
                           2, outer, 0, int(config["protocol"]["formal_trials_per_subject_task"]),
                           int(config["attacks"]["pgd_steps"]), config["attacks"]["order"], True)
            run.update({"registered_subjects": classes.tolist(), "unknown_subjects": unknown_subjects.tolist(),
                        "n_fit": int(len(fit_np))})
            save_artifact(path, model, info, run)
            del model
        evaluated = evaluate_point([run], selection["score_name"], selection["threshold"], config["attacks"]["order"])
        clean = clean_metrics([run], selection["score_name"], selection["threshold"], selection["q05_threshold"])
        folds.append({"outer_fold": outer, "unknown_subjects": unknown_subjects.tolist(),
                      "frozen_inner_selection": selection, "train_info": info, "clean": clean, **evaluated})
        print(f"OUTER {outer}: Score={evaluated['macro']['Score']:.3f} directRA={evaluated['macro']['direct_RA']:.4f} "
              f"RA={evaluated['macro']['RA']:.4f} ASafeR={evaluated['macro']['ASafeR']:.4f}", flush=True)
        result["outer"] = {"folds": folds, "rule": "one frozen confirmation; no outer threshold/model selection"}
        result["status"] = "outer_partial"
        write_json(RESULTS_PATH, result)
    result["outer"].update({
        "macro_fold_mean": fold_mean([f["macro"] for f in folds]),
        "clean_fold_mean": fold_mean([f["clean"] for f in folds]),
        "per_attack_fold_mean": {name: fold_mean([f["per_attack"][name] for f in folds])
                                 for name in config["attacks"]["order"]},
    })
    result["status"] = "complete"
    write_json(RESULTS_PATH, result)
    return result


def fmt(value: float) -> str:
    return f"{float(value):.4f}"


def write_report(result: dict, config: dict) -> None:
    lines = [
        "# E53 原始 EEG 轻量开放集与对抗鲁棒实验", "",
        "日期：2026-11-13。仅使用 Train Session1；Test 未读取。", "",
        "## 实际执行与复现", "",
        "```powershell",
        f"& \"{ROOT / '.venv' / 'Scripts' / 'python.exe'}\" \"{OUT / 'run_e53.py'}\" --stage smoke",
        f"& \"{ROOT / '.venv' / 'Scripts' / 'python.exe'}\" \"{OUT / 'run_e53.py'}\" --stage inner",
        f"& \"{ROOT / '.venv' / 'Scripts' / 'python.exe'}\" \"{OUT / 'run_e53.py'}\" --stage outer  # 仅主方案全部门禁通过时执行",
        "```", "",
        f"状态：`{result['status']}`。实现：`{OUT / 'run_e53.py'}`；配置：`{CONFIG_PATH}`；完整结果：`{RESULTS_PATH}`。", "",
        "- 输入为单条 1×18×1200；逐通道线性去趋势、median/1.4826×MAD（RMS 回退）均逐样本可微。",
        "- 8 时间卷积(k=65)→depthwise 空间卷积(D=2)→16 separable(k=15)，BN/ELU/池化/GAP，64 维 L2 embedding；归一化可训练身份原型+cosine logits。",
        "- 主方案双侧 ARPL/reciprocal-point 开空间损失，不设第 21 类；两个消融仅去 ARPL、或目标 AT 改全非目标 AT。",
        f"- 实际训练攻击路径：`{config['training']['attack_training_mode']}`；clean/adv 损失各 0.5，标签始终为真实身份。正式评估始终原始波形 FGSM/PGD20。",
        "- 阈值仅用对应 inner clean、六攻击和 pseudo-unknown，在 clean known acceptance≥0.95 下最大化完整本地代理 Score。direct RA 与阈值 RA 分开判定。", "",
        "## 十 inner 折汇总（五个 outer-group 内各两折）", "",
        "AUC 为 known=1、unknown=0 的检测 AUROC（置信度越大越像 known）；AUPR 为 unknown 类。clean F1 在连续块全量已知上计算；对抗 F1/AUC 为六攻击宏均值。N 为每折六攻击对抗样本合计的折均值。", "",
        "| 变体 | Train fit | direct C | 阈值C | clean F1 direct/拒识 | 连续块 | 攻击F1 direct/拒识 | clean AUROC/AUPR | U@95 | 工作点U | unknown F1 | joint F1 | direct RA | 阈值RA | ASafeR | A | Score | N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in result.get("variants", {}).items():
        if "full_train_fit" not in item:
            continue
        clean, macro = item["clean_fold_mean"], item["macro_fold_mean"]
        lines.append(
            f"| {name} | {fmt(item['full_train_fit']['accuracy'])} | {fmt(clean['direct_C'])} | {fmt(macro['C'])} | "
            f"{fmt(clean['identity_macro_f1_direct'])}/{fmt(clean['identity_macro_f1'])} | {fmt(clean['continuous_block_accuracy'])} | "
            f"{fmt(macro['identity_macro_f1_direct'])}/{fmt(macro['identity_macro_f1'])} | "
            f"{fmt(clean['unknown_auroc'])}/{fmt(clean['unknown_aupr'])} | {fmt(clean['q05_U'])} | {fmt(macro['U'])} | "
            f"{fmt(macro['unknown_f1'])} | {fmt(macro['joint_f1'])} | {fmt(macro['direct_RA'])} | {fmt(macro['RA'])} | "
            f"{fmt(macro['ASafeR'])} | {fmt(macro['A'])} | {fmt(macro['Score'])} | {macro['N_adversarial_total']:.0f} |"
        )
    main = result.get("variants", {}).get("main")
    if main and "per_attack_fold_mean" in main:
        lines.extend(["", "## 主方案逐攻击（inner 折均值）", "",
                      "| 攻击 | C | direct RA | 阈值RA | ASafeR | A | F1 direct/拒识 | AUROC | AUPR | unknown F1 | U | joint F1 | Score | N攻击 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for attack, row in main["per_attack_fold_mean"].items():
            lines.append(f"| {attack} | {fmt(row['C'])} | {fmt(row['direct_RA'])} | {fmt(row['RA'])} | {fmt(row['ASafeR'])} | "
                         f"{fmt(row['A'])} | {fmt(row['identity_macro_f1_direct'])}/{fmt(row['identity_macro_f1'])} | "
                         f"{fmt(row['unknown_auroc'])} | {fmt(row['unknown_aupr'])} | {fmt(row['unknown_f1'])} | {fmt(row['U'])} | "
                         f"{fmt(row['joint_f1'])} | {fmt(row['Score'])} | {row['N_adversarial']:.0f} |")
    lines.extend(["", "## 门禁与消融定位", "",
                  "| 变体 | Train fit | direct/阈值C | 身份F1 | 连续块 | AUROC | U@95 | direct RA | 阈值RA | ASafeR | 非全拒识 | 全部 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"])
    for name, item in result.get("variants", {}).items():
        gates = item.get("gates")
        if not gates:
            continue
        keys = ("train_fit_100pct", "direct_C", "threshold_C", "identity_f1_direct", "continuous_block_accuracy",
                "unknown_auroc", "U_at_95", "direct_RA", "threshold_RA", "ASafeR", "not_all_rejection", "all_pass")
        values = []
        for key in keys:
            mark = "通过" if gates[key] else "未通过"
            values.append(mark if key != "direct_C" else mark+"/"+("通过" if gates["threshold_C"] else "未通过"))
        # direct/threshold C occupy one table column.
        values.pop(2)
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    if main and "macro_fold_mean" in main:
        lines.extend(["", f"- 主方案 all_rejection={main['macro_fold_mean']['all_rejection'] if 'all_rejection' in main['macro_fold_mean'] else any(f['macro']['all_rejection'] for f in main['folds'])}。",
                      "- 两个消融只用于定位 ARPL 与目标 AT 的贡献；未做网格搜索，未据 outer/Test 调参。",
                      "- direct RA<0.95 时，即使阈值 RA 或 ASafeR 较高，也不宣称鲁棒门禁达标。"])
    smoke_info = None
    smoke_path = artifact_path("main", 1, 1)
    if smoke_path.exists():
        info, _, _ = load_artifact(smoke_path, torch.device("cpu"))
        smoke_info = info
    if smoke_info and "numerical_checks" in smoke_info:
        check = smoke_info["numerical_checks"]
        lines.extend(["", "## SMOKE 数值与不变性", "",
                      f"- 参数量：{smoke_info['parameter_count']}；在线 PGD7 batch64 实测 {smoke_info.get('online_pgd7_batch64_seconds_after_warmup', float('nan')):.3f}s（仅速度基准）。",
                      f"- 单样本/批次最大分数差 {check['single_vs_batch_max_abs']:.3g}；置换差 {check['permutation_max_abs']:.3g}；标签一致={check['single_vs_batch_labels_equal']}。",
                      f"- 方向导数有限差分相对误差 {check['finite_difference_relative_error']:.3g}；所有正式攻击梯度与分数均要求 finite。"])
    if isinstance(result.get("outer"), dict) and "macro_fold_mean" in result["outer"]:
        outer = result["outer"]
        lines.extend(["", "## Outer 一次性冻结确认", "",
                      f"macro：direct RA={fmt(outer['macro_fold_mean']['direct_RA'])}，RA={fmt(outer['macro_fold_mean']['RA'])}，"
                      f"ASafeR={fmt(outer['macro_fold_mean']['ASafeR'])}，Score={fmt(outer['macro_fold_mean']['Score'])}；"
                      f"clean AUROC={fmt(outer['clean_fold_mean']['unknown_auroc'])}，direct F1={fmt(outer['clean_fold_mean']['identity_macro_f1_direct'])}。"])
    else:
        lines.extend(["", "## Outer 与 Test", "", f"- Outer：{result.get('outer', 'N/A')}。", "- Test AUC、F1、C、RA、ASafeR、A、U、Score：N/A（无真实标签，且未读取 Test）。"])
    lines.extend(["", "## 证据边界", "",
                  "- unknown 全部是 Train 留被试 pseudo-unknown，不等于官方新被试；连续文件块是时间隔离代理，不是真实跨天/跨 Session 证据。",
                  "- Train fit=100% 仅为容量与训练流程检查，不是泛化证据。SMOKE 只用于速度、梯度和数值检查，不用于模型选择。",
                  "- 本地攻击覆盖 FGSM、非目标/目标 PGD20×SNR30/20，不能保证覆盖官方全部扰动；Score 是 Train 内代理，不是官方 Test 得分。",
                  "- 若门禁失败，本报告即为负结果归档；不通过削弱攻击、全拒识或外层/Test 调参补救。", ""])
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "inner", "outer", "all", "report"), default="all")
    parser.add_argument("--variants", nargs="*", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("E53 requires CUDA")
    config = load_config()
    CACHE.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed_base"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    variants = args.variants or list(config["training"]["variants"])
    if any(v not in config["training"]["variants"] for v in variants):
        parser.error("Only preregistered main/no_arpl/untargeted_at variants are allowed")
    if args.stage == "report":
        result = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        write_report(result, config)
        return
    metadata = train_metadata(config)
    raw = load_all_raw(metadata, config, device)
    if args.stage in ("smoke", "all"):
        result = run_inner_stage(raw, metadata, config, ["main"], smoke=True)
        write_report(result, config)
    if args.stage in ("inner", "all"):
        result = run_inner_stage(raw, metadata, config, variants, smoke=False)
        write_report(result, config)
    if args.stage in ("outer", "all"):
        if args.stage == "outer":
            result = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
            if result["status"] not in ("inner_complete", "complete_inner_gates_failed"):
                raise ValueError("Run the complete inner stage first")
            if result["config_sha256"] != sha256(CONFIG_PATH):
                raise ValueError("Config changed after inner freeze")
        result = run_outer_stage(result, raw, metadata, config)
        write_report(result, config)
    print(f"Results: {RESULTS_PATH}", flush=True)
    print(f"Report: {OUT / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
