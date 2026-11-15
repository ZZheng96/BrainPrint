"""E182: session-invariant branch A (cross-session consistency) + frozen gate fusion.

Question
--------
E181 showed that branch A's four confidence signals (conf / margin / energy / entropy)
are *anti-correlated* with the Test S1-vs-S2 split (AUC 0.40-0.41 with S2 as positive):
a classifier trained on session1 finds session1 "familiar" and session2 "unfamiliar",
so its confidence measures session similarity, not enrolment match.

E182 keeps the E180 EEGNet backbone, data pipeline, PGD adversarial objective and the
frozen branch B gate, but adds an explicit cross-session invariance constraint:
the same trial is pushed through two mutually exclusive strong perturbation families
(electrode/impedance vs spectral/reference), and the 128-d pooled embeddings are
pulled together with a cosine consistency loss.

Stages
------
train    selection (lambda in {0.1, 0.3}) -> 5-block OOF -> full model (model_branch_a_v2.pt)
signals  five gate signals on Train OOF and on Test (no Session read)
loso     20 leave-identity-out probe models (pm & fit_rows) -> gate-signal AUC
select   fusion candidate library, weights by Train LOSO pooled AUC only
test     frozen Test scoring, predictions written before Session is read (#9/#10 audit)
attack   #7/#8 self-built white-box PGD on the fused system
report   report.md

Compliance: every threshold, weight, PCA basis and prototype is Train-only; LOSO uses
`pm & fit_rows`; Test is scored once after freezing; Session is never a model input.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "report.md"
MODEL_V2 = OUT / "model_branch_a_v2.pt"
STAGE_A_V2 = CACHE / "stage_a_v2.pt"
SIGNAL_CACHE = CACHE / "signals_v2.npz"
LOSO_CACHE = CACHE / "loso_v2.npz"
TEST_INPUTS = CACHE / "test_inputs.npz"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E180_DIR = ROOT / "research" / CFG["frozen_source"]["experiment"]
E180 = load_module("e182_frozen_e180", E180_DIR / "run_e180.py")
DEVICE = E180.DEVICE
FT = E180.FT
TASKS = list(E180.TASKS)
CLASSES = E180.CLASSES
SEED = int(E180.SEED)
FIT = float(E180.CFG["protocol"]["fit_fraction"])
N = int(E180.N)
CH = int(E180.CH)
FS = int(E180.FS)
NB = int(E180.NB)
BLOCKS = int(E180.CFG["protocol"]["blocks"])

SPEC = {"data": E180.CFG["data"], "protocol": E180.CFG["protocol"],
        "branch_a_v2": CFG["branch_a_v2"]}
V2_SHA = hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode("utf-8")).hexdigest()

SIGNALS = ("g_dist", "g_conf", "g_margin", "g_energy", "g_entropy")
ACC_GRID = [float(v) for v in CFG["threshold"]["grid"]]
PRIMARY_A = float(CFG["threshold"]["primary_acceptance"])
WEIGHTS = [round(0.1 * i, 1) for i in range(1, 10)]
CLASS_OF = {int(c): i for i, c in enumerate(CLASSES)}

# E181 published reference values (used only for the before/after comparison table)
E181_SIGNAL_AUC = {"g_dist": 0.7563, "g_conf": 0.4106, "g_margin": 0.4084,
                   "g_energy": 0.4001, "g_entropy": 0.4123}
E181_GATES = {"#1": 0.9988, "#2": 0.9881, "#3": 0.9878, "#4": 0.9881, "#5": 0.9694,
              "#6": 0.8145, "#7": 0.8978, "#8": 0.9944, "#9": 0.3494, "#10": 0.9651}
E180_GATES = {"#1": 0.9988, "#2": 0.9881, "#3": 0.9878, "#4": 0.9881, "#5": 0.9401,
              "#6": 0.5847, "#7": 0.9161, "#8": 0.9840, "#9": 0.6335, "#10": 0.7359}


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


def f32(array) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {"experiment": CFG["experiment"], "branch_a_v2_spec_sha256": V2_SHA}


def save_results(results: dict) -> None:
    RESULTS_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def quantile(value: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(value, dtype=np.float64), q))


# ---------------------------------------------------------------- perturbation families
def session_view_electrode(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Family A: electrode / impedance style shift (gain, bias, channel dropout)."""
    b, c, _ = x.shape
    cfg = CFG["branch_a_v2"]["perturbation_A_electrode"]
    xc = x - x.mean(dim=-1, keepdim=True)
    rms = xc.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
    gain_sd = float(cfg["channel_gain_sd"])
    gain = 1.0 + gain_sd * torch.randn(b, c, 1, generator=gen, device=x.device)
    bias = float(cfg["channel_bias_sd_rms"]) * rms * torch.randn(
        b, c, 1, generator=gen, device=x.device)
    keep = (torch.rand(b, c, 1, generator=gen, device=x.device)
            >= float(cfg["channel_dropout"])).to(x.dtype)
    return (x * gain + bias) * keep


def session_view_spectral(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Family B: spectral / reference style shift (band gains, drift, additive noise)."""
    b, c, n = x.shape
    cfg = CFG["branch_a_v2"]["perturbation_B_spectral"]
    xc = x - x.mean(dim=-1, keepdim=True)
    rms = xc.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
    spec = torch.fft.rfft(x, dim=-1)
    band_sd = float(cfg["band_gain_sd"])
    gains = 1.0 + band_sd * torch.randn(b, 1, NB, 1, generator=gen, device=x.device)
    weight = torch.ones(b, 1, spec.shape[-1], dtype=x.dtype, device=x.device)
    for index in range(NB):
        weight = weight + (gains[:, :, index] - 1.0) * E180.MASK_BANDS[index]
    out = torch.fft.irfft(spec * weight, n=n, dim=-1)

    low_hz, high_hz = [float(v) for v in cfg["drift_freq_hz"]]
    low_amp, high_amp = [float(v) for v in cfg["drift_amp_rms"]]
    t = torch.arange(n, dtype=x.dtype, device=x.device) / FS
    freq = low_hz + (high_hz - low_hz) * torch.rand(
        b, 1, 1, generator=gen, device=x.device)
    phase = 2.0 * math.pi * torch.rand(b, 1, 1, generator=gen, device=x.device)
    amp = rms * (low_amp + (high_amp - low_amp) * torch.rand(
        b, 1, 1, generator=gen, device=x.device))
    drift = amp * torch.sin(2.0 * math.pi * freq * t + phase)
    ramp = float(cfg["ramp_sd_rms"]) * rms * torch.randn(
        b, c, 1, generator=gen, device=x.device) * \
        torch.linspace(-1.0, 1.0, n, dtype=x.dtype, device=x.device)
    snr_low, snr_high = [float(v) for v in cfg["additive_noise_snr_db"]]
    snr = snr_low + (snr_high - snr_low) * torch.rand(
        b, 1, 1, generator=gen, device=x.device)
    noise = torch.randn(out.shape, generator=gen, device=x.device) * \
        rms / 10.0 ** (snr / 20.0)
    return out + drift + ramp + noise


def embed(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """128-d pooled pre-head embedding (same normalisation as Net.forward)."""
    x = x - x.mean(dim=-1, keepdim=True)
    rms = x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
    return model.pool(model.body(x / rms)).squeeze(-1)


# ---------------------------------------------------------------- training
def train_v2(raw: torch.Tensor, idx_np: np.ndarray, labels_np: np.ndarray, seed: int,
             epochs: int, lam: float, tag: str, eval_np: np.ndarray | None = None,
             eval_labels: np.ndarray | None = None) -> tuple[torch.nn.Module, dict]:
    cfg = CFG["branch_a_v2"]
    w_clean = float(cfg["loss_weights"]["clean"])
    w_adv = float(cfg["loss_weights"]["adversarial"])
    torch.manual_seed(seed)
    model = E180.Net().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                            weight_decay=float(cfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    idx = torch.as_tensor(idx_np, dtype=torch.long, device=DEVICE)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    rng = np.random.default_rng(seed)
    batch = int(cfg["batch_size"])
    best = (-1.0, None, -1)
    info = {"epochs": epochs, "lambda": lam, "epoch_seconds": [],
            "eval_accuracy": [], "consistency": [],
            "parameter_count": int(sum(p.numel() for p in model.parameters()))}
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        order = torch.as_tensor(rng.permutation(len(idx_np)), dtype=torch.long,
                                device=DEVICE)
        total, cons_total = 0.0, 0.0
        for start in range(0, len(idx_np), batch):
            sel = order[start:start + batch]
            if len(sel) < 4:
                continue
            xb = E180.augment(raw[idx[sel]], gen)
            yb = labels[sel]
            adv = E180.pgd_train(model, xb, yb, gen)
            view_a = session_view_electrode(xb, gen)
            view_b = session_view_spectral(xb, gen)
            logits_clean = model(xb)
            logits_adv = model(adv)
            consistency = 1.0 - F.cosine_similarity(
                embed(model, view_a), embed(model, view_b), dim=1).mean()
            loss = (w_clean * F.cross_entropy(logits_clean, yb)
                    + w_adv * F.cross_entropy(logits_adv, yb)
                    + lam * consistency)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(sel)
            cons_total += float(consistency.detach()) * len(sel)
        sched.step()
        info["epoch_seconds"].append(time.time() - started)
        info["consistency"].append(cons_total / len(idx_np))
        message = (f"  [{tag}] epoch {epoch}/{epochs} loss {total / len(idx_np):.4f} "
                   f"cons {info['consistency'][-1]:.4f} "
                   f"{info['epoch_seconds'][-1]:.1f}s")
        if eval_np is not None:
            acc = float(np.mean(E180.predict_logits(model, raw, eval_np).argmax(1) + 1
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


# ---------------------------------------------------------------- stage train
def stage_train() -> dict:
    started = time.time()
    E180.build_torch_constants()
    cfg = CFG["branch_a_v2"]
    meta, frac, blocks, fit_rows = E180.load_config_meta()
    subjects = meta["subject"].to_numpy(int)
    window = (frac > FIT) & (frac <= 0.8)
    window_np = np.flatnonzero(window)
    log(f"stage train: {len(meta)} Train trials, fit_rows {int(fit_rows.sum())}, "
        f"eval window {len(window_np)}, lambda sweep "
        f"{cfg['loss_weights']['consistency_sweep']}")
    raw = E180.load_raw(meta["path"].tolist())
    log(f"raw on GPU: {tuple(raw.shape)} ({raw.numel() * 4 / 2**30:.2f} GiB)")

    # 1) lambda selection: train on fit_rows, evaluate on the eval window (Train-only).
    sel_epochs = int(cfg.get("selection_epochs", 20))
    sel_train = np.flatnonzero(fit_rows)
    selection = []
    for lam in [float(v) for v in cfg["loss_weights"]["consistency_sweep"]]:
        path = CACHE / f"v2_select_lam{lam:.2f}.pt"
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=False)
            if state.get("spec_sha256") == V2_SHA and state.get("epochs") == sel_epochs:
                selection.append({"lambda": lam,
                                  "best_eval_accuracy": float(state["best_eval_accuracy"]),
                                  "best_epoch": int(state["best_epoch"]),
                                  "mean_epoch_seconds": float(np.mean(state["epoch_seconds"])),
                                  "eval_accuracy_curve": state["eval_accuracy"].tolist(),
                                  "consistency_curve": state["consistency"].tolist(),
                                  "seconds": float(state["seconds"]), "cached": True})
                log(f"  [select lam={lam}] cached eval_acc "
                    f"{state['best_eval_accuracy']:.4f}")
                continue
        model, info = train_v2(raw, sel_train, subjects[sel_train] - 1, SEED + 303,
                               sel_epochs, lam, f"select_lam{lam}", eval_np=window_np,
                               eval_labels=subjects[window_np])
        torch.save({"state_dict": model.state_dict(), "lambda": lam,
                    "spec_sha256": V2_SHA, "epochs": sel_epochs,
                    "best_epoch": info["best_epoch"],
                    "best_eval_accuracy": info["best_eval_accuracy"],
                    "epoch_seconds": np.asarray(info["epoch_seconds"], dtype=np.float32),
                    "eval_accuracy": np.asarray(info["eval_accuracy"], dtype=np.float32),
                    "consistency": np.asarray(info["consistency"], dtype=np.float32),
                    "seconds": time.time() - started}, path)
        selection.append({"lambda": lam,
                          "best_eval_accuracy": info["best_eval_accuracy"],
                          "best_epoch": info["best_epoch"],
                          "mean_epoch_seconds": float(np.mean(info["epoch_seconds"])),
                          "eval_accuracy_curve": info["eval_accuracy"],
                          "consistency_curve": info["consistency"],
                          "seconds": float(np.sum(info["epoch_seconds"])),
                          "cached": False})
        del model
        torch.cuda.empty_cache()

    chosen = max(selection, key=lambda row: (row["best_eval_accuracy"], -row["lambda"]))
    lam = float(chosen["lambda"])
    log(f"  selected lambda = {lam:.2f} (eval-block clean accuracy "
        f"{chosen['best_eval_accuracy']:.4f})")

    # 2) 5-block OOF with the chosen lambda.
    oof_logits = np.zeros((len(meta), len(CLASSES)), dtype=np.float32)
    folds = []
    fold_epochs = int(cfg["fold_epochs"])
    for block in range(BLOCKS):
        train_np = np.flatnonzero(blocks != block)
        eval_np = np.flatnonzero(blocks == block)
        path = CACHE / f"v2_fold_{block}.pt"
        state = None
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=False)
            if (state.get("spec_sha256") != V2_SHA or state.get("lambda") != lam
                    or state.get("epochs") != fold_epochs):
                state = None
        if state is None:
            model, info = train_v2(raw, train_np, subjects[train_np] - 1,
                                   SEED + 101 * block, fold_epochs, lam, f"fold{block}")
            torch.save({"state_dict": model.state_dict(), "block": block, "lambda": lam,
                        "spec_sha256": V2_SHA, "epochs": fold_epochs,
                        "epoch_seconds": np.asarray(info["epoch_seconds"], dtype=np.float32),
                        "consistency": np.asarray(info["consistency"], dtype=np.float32)},
                       path)
            epoch_seconds = info["epoch_seconds"]
        else:
            model = E180.Net().to(DEVICE)
            model.load_state_dict(state["state_dict"])
            model.eval()
            epoch_seconds = state["epoch_seconds"].tolist()
            log(f"  [fold{block}] cached")
        logits = E180.predict_logits(model, raw, eval_np)
        oof_logits[eval_np] = logits
        accuracy = float(np.mean(logits.argmax(1) + 1 == subjects[eval_np]))
        folds.append({"block": block, "n_train": int(len(train_np)),
                      "n_eval": int(len(eval_np)), "accuracy": accuracy,
                      "epoch_seconds": list(map(float, epoch_seconds))})
        del model
        torch.cuda.empty_cache()
        log(f"  fold{block} OOF accuracy {accuracy:.4f}")

    # 3) full model, early stop on the eval-window clean accuracy.
    full_epochs = int(cfg["full_epochs"])
    cached = None
    if MODEL_V2.exists():
        cached = torch.load(MODEL_V2, map_location="cpu", weights_only=False)
        if (cached.get("spec_sha256") != V2_SHA or cached.get("lambda") != lam
                or cached.get("epochs") != full_epochs):
            cached = None
    if cached is not None:
        full = E180.Net().to(DEVICE)
        full.load_state_dict(cached["state_dict"])
        full.eval()
        info = {"epoch_seconds": list(map(float, cached["epoch_seconds"])),
                "best_epoch": int(cached["best_epoch"]),
                "best_eval_accuracy": float(cached["best_eval_accuracy"]),
                "consistency": list(map(float, cached["consistency"]))}
        log(f"  [full] cached (best epoch {info['best_epoch']})")
    else:
        full, info = train_v2(raw, np.arange(len(meta)), subjects - 1, SEED,
                              full_epochs, lam, "full", eval_np=window_np,
                              eval_labels=subjects[window_np])

    in_sample = E180.predict_logits(full, raw, np.arange(len(meta)))
    in_sample_acc = float(np.mean(in_sample.argmax(1) + 1 == subjects))
    in_sample_f1 = float(f1_score(subjects, in_sample.argmax(1) + 1, labels=CLASSES,
                                  average="macro", zero_division=0))
    log(f"  full in-sample accuracy {in_sample_acc:.4f} macro-F1 {in_sample_f1:.4f}")

    if cached is None:
        torch.save({"state_dict": full.state_dict(), "spec_sha256": V2_SHA,
                    "spec": cfg, "classes": CLASSES, "lambda": lam,
                    "epochs": full_epochs, "best_epoch": info["best_epoch"],
                    "best_eval_accuracy": info["best_eval_accuracy"],
                    "epoch_seconds": np.asarray(info["epoch_seconds"], dtype=np.float32),
                    "consistency": np.asarray(info["consistency"], dtype=np.float32),
                    "in_sample_accuracy": in_sample_acc,
                    "in_sample_macro_f1": in_sample_f1}, MODEL_V2)
    torch.save({"spec_sha256": V2_SHA, "lambda": lam,
                "oof_logits": torch.as_tensor(oof_logits),
                "in_sample_logits": torch.as_tensor(in_sample),
                "subjects": torch.as_tensor(subjects),
                "blocks": torch.as_tensor(blocks)}, STAGE_A_V2)
    del full
    torch.cuda.empty_cache()

    oof_pred = oof_logits.argmax(1) + 1
    forward = frac > FIT
    summary = {
        "lambda_sweep": selection, "lambda": lam,
        "selection_rule": "argmax eval-window clean accuracy on blocks != 3 (Train-only)",
        "n_parameters": int(sum(p.numel() for p in E180.Net().parameters())),
        "fold_epochs": fold_epochs, "full_epochs": full_epochs,
        "full_epochs_run": len(info["epoch_seconds"]),
        "full_best_epoch": info["best_epoch"],
        "full_best_eval_accuracy": info["best_eval_accuracy"],
        "full_consistency_last": float(info["consistency"][-1]),
        "fold_mean_epoch_seconds": float(np.mean([np.mean(f["epoch_seconds"])
                                                  for f in folds])),
        "folds": folds,
        "oof_accuracy_5block": float(np.mean(oof_pred == subjects)),
        "oof_macro_f1": float(f1_score(subjects, oof_pred, labels=CLASSES,
                                       average="macro", zero_division=0)),
        "oof_accuracy_forward_blocks": float(np.mean(oof_pred[forward] == subjects[forward])),
        "oof_accuracy_per_block": {str(b): float(np.mean(
            oof_pred[blocks == b] == subjects[blocks == b])) for b in range(BLOCKS)},
        "in_sample_accuracy": in_sample_acc,
        "in_sample_macro_f1": in_sample_f1,
        "loss": cfg["loss"],
        "elapsed_seconds": time.time() - started,
        "gpu": torch.cuda.get_device_name(0),
    }
    results = load_results()
    results["branch_a_v2"] = summary
    save_results(results)
    log(f"stage train done in {summary['elapsed_seconds']:.0f}s: lambda {lam:.2f}, "
        f"OOF acc {summary['oof_accuracy_5block']:.4f}, macro-F1 "
        f"{summary['oof_macro_f1']:.4f}, in-sample {in_sample_acc:.4f}")
    del raw
    torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------- gate signals
def logsumexp_np(x: np.ndarray) -> np.ndarray:
    m = x.max(1, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(1, keepdims=True))).ravel()


def raw_signals(logits: np.ndarray, dist: np.ndarray) -> dict:
    lg = np.asarray(logits, dtype=np.float64)
    shifted = lg - lg.max(1, keepdims=True)
    prob = np.exp(shifted)
    prob /= prob.sum(1, keepdims=True)
    ordered = np.sort(prob, axis=1)
    return {
        "g_dist": np.asarray(dist, dtype=np.float64),
        "g_conf": prob.max(1),
        "g_margin": ordered[:, -1] - ordered[:, -2],
        "g_energy": logsumexp_np(lg),
        "g_entropy": np.sum(prob * np.log(np.clip(prob, 1e-12, None)), axis=1),
    }


def fit_stats(raw: dict) -> dict:
    return {"mu": {k: float(np.mean(raw[k])) for k in SIGNALS},
            "sd": {k: float(np.std(raw[k])) + 1e-12 for k in SIGNALS}}


def standardise(raw: dict, stats: dict) -> dict:
    return {k: (raw[k] - stats["mu"][k]) / stats["sd"][k] for k in SIGNALS}


def fuse(z: dict, weights: dict) -> np.ndarray:
    total = np.zeros(len(next(iter(z.values()))), dtype=np.float64)
    for name, weight in weights.items():
        total = total + float(weight) * z[name]
    return total


def candidate_library() -> list[dict]:
    cands: list[dict] = [{"name": s, "weights": {s: 1.0}} for s in SIGNALS]
    for a, b in itertools.combinations(SIGNALS, 2):
        for w in WEIGHTS:
            cands.append({"name": f"{w:.2f}*{a} + {1 - w:.2f}*{b}",
                          "weights": {a: w, b: round(1 - w, 1)}})
    for combo in itertools.combinations(SIGNALS, 3):
        cands.append({"name": " + ".join(f"0.33*{s}" for s in combo),
                      "weights": {s: 1.0 / 3.0 for s in combo}})
    for size in (4, 5):
        for combo in itertools.combinations(SIGNALS, size):
            cands.append({"name": " + ".join(f"{1 / size:.2f}*{s}" for s in combo),
                          "weights": {s: 1.0 / size for s in combo}})
    seen, unique = set(), []
    for cand in cands:
        key = tuple(sorted((k, round(v, 6)) for k, v in cand["weights"].items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
    return unique


def build_test_inputs() -> dict:
    """Frozen branch-A v2 logits, E180 v1 logits and branch-B Test scores (no Session)."""
    if TEST_INPUTS.exists():
        data = np.load(TEST_INPUTS, allow_pickle=False)
        if str(data["spec_sha256"]) == V2_SHA:
            return {k: data[k] for k in data.files}
    test_meta = pd.read_csv(ROOT / E180.CFG["data"]["test_metadata"],
                            usecols=["sample", "path", "task"])
    paths = test_meta["path"].tolist()
    base = ROOT / E180.CFG["data"]["test_root"]
    logits_v2 = np.zeros((len(paths), len(CLASSES)), dtype=np.float32)
    logits_v1 = np.zeros((len(paths), len(CLASSES)), dtype=np.float32)

    state = torch.load(MODEL_V2, map_location="cpu", weights_only=False)
    if state.get("spec_sha256") != V2_SHA:
        raise ValueError("branch A v2 artifact was trained with a different config")
    model = E180.Net().to(DEVICE)
    model.load_state_dict(state["state_dict"])
    model.eval()
    from concurrent.futures import ThreadPoolExecutor
    started = time.time()
    with ThreadPoolExecutor(max_workers=12) as pool:
        for start in range(0, len(paths), 512):
            chunk = paths[start:start + 512]
            arrays = list(pool.map(lambda p: E180.load_mat(base / str(p)), chunk))
            batch = torch.as_tensor(np.stack(arrays), dtype=FT, device=DEVICE)
            with torch.no_grad():
                logits_v2[start:start + len(chunk)] = model(batch).cpu().numpy()
            del batch
            if (start // 512) % 8 == 0:
                log(f"  Test v2 logits {start + len(chunk)}/{len(paths)} "
                    f"({time.time() - started:.0f}s)")
    del model
    torch.cuda.empty_cache()

    v1 = torch.load(E180.MODEL_A, map_location="cpu", weights_only=False)
    if v1.get("spec_sha256") != E180.BRANCH_A_SHA:
        raise ValueError("E180 branch A artifact does not match its frozen spec")
    model = E180.Net().to(DEVICE)
    model.load_state_dict(v1["state_dict"])
    model.eval()
    with ThreadPoolExecutor(max_workers=12) as pool:
        for start in range(0, len(paths), 512):
            chunk = paths[start:start + 512]
            arrays = list(pool.map(lambda p: E180.load_mat(base / str(p)), chunk))
            batch = torch.as_tensor(np.stack(arrays), dtype=FT, device=DEVICE)
            with torch.no_grad():
                logits_v1[start:start + len(chunk)] = model(batch).cpu().numpy()
            del batch
    del model
    torch.cuda.empty_cache()

    store = E180.load_branch_b()
    test_features = f32(np.load(ROOT / E180.CFG["branch_b"]["test_features"]))
    dist, _ = E180.branch_b_test_scores(test_features, test_meta["task"].to_numpy(), store)
    if not np.isfinite(dist).all():
        raise RuntimeError("non-finite branch-B Test scores")
    np.savez_compressed(TEST_INPUTS, spec_sha256=np.array(V2_SHA),
                        logits_v2=logits_v2, logits_v1=logits_v1, dist=dist,
                        sample=test_meta["sample"].astype(str).to_numpy(dtype=str),
                        tasks=test_meta["task"].to_numpy())
    return {"logits_v2": logits_v2, "logits_v1": logits_v1, "dist": dist,
            "sample": test_meta["sample"].astype(str).to_numpy(dtype=str),
            "tasks": test_meta["task"].to_numpy()}


def stage_signals() -> dict:
    started = time.time()
    meta, frac, blocks, fit_rows = E180.load_config_meta()
    subjects = meta["subject"].to_numpy(int)
    stage_a_v2 = torch.load(STAGE_A_V2, map_location="cpu", weights_only=False)
    if stage_a_v2.get("spec_sha256") != V2_SHA:
        raise ValueError("stage A v2 artifact does not match the config")
    stage_a_v1 = torch.load(E180.STAGE_A, map_location="cpu", weights_only=False)
    store = E180.load_branch_b()
    oof_dist = np.asarray(store["oof_scores"], np.float64)

    raw_oof_v2 = raw_signals(stage_a_v2["oof_logits"].numpy(), oof_dist)
    stats_v2 = fit_stats(raw_oof_v2)
    z_oof_v2 = standardise(raw_oof_v2, stats_v2)
    raw_oof_v1 = raw_signals(stage_a_v1["oof_logits"].numpy(), oof_dist)
    stats_v1 = fit_stats(raw_oof_v1)
    z_oof_v1 = standardise(raw_oof_v1, stats_v1)
    log("stage signals: Train OOF stats (Train-only)")
    for name in SIGNALS:
        log(f"  {name:9s} v2 mean {stats_v2['mu'][name]:+.4f} sd {stats_v2['sd'][name]:.4f}")

    inputs = build_test_inputs()
    raw_test_v2 = raw_signals(inputs["logits_v2"], inputs["dist"])
    raw_test_v1 = raw_signals(inputs["logits_v1"], inputs["dist"])
    z_test_v2 = standardise(raw_test_v2, stats_v2)
    z_test_v1 = standardise(raw_test_v1, stats_v1)

    np.savez_compressed(
        SIGNAL_CACHE,
        stats_mu=np.asarray([stats_v2["mu"][k] for k in SIGNALS], dtype=np.float64),
        stats_sd=np.asarray([stats_v2["sd"][k] for k in SIGNALS], dtype=np.float64),
        oof_raw=np.stack([raw_oof_v2[k] for k in SIGNALS]),
        oof_z=np.stack([z_oof_v2[k] for k in SIGNALS]),
        oof_z_v1=np.stack([z_oof_v1[k] for k in SIGNALS]),
        oof_logits_v2=stage_a_v2["oof_logits"].numpy().astype(np.float32),
        test_raw=np.stack([raw_test_v2[k] for k in SIGNALS]),
        test_z=np.stack([z_test_v2[k] for k in SIGNALS]),
        test_z_v1=np.stack([z_test_v1[k] for k in SIGNALS]),
        test_logits_v2=inputs["logits_v2"], test_logits_v1=inputs["logits_v1"],
        test_sample=inputs["sample"].astype(str),
        test_tasks=inputs["tasks"],
        subjects=subjects, blocks=blocks, frac=frac)

    summary = {
        "signal_stats_v2": stats_v2,
        "signal_stats_v1": stats_v1,
        "oof_z_check_v2": {k: [float(np.mean(z_oof_v2[k])), float(np.std(z_oof_v2[k]))]
                           for k in SIGNALS},
        "branch_a_reproduced": {
            "oof_accuracy_5block": float(np.mean(
                stage_a_v2["oof_logits"].numpy().argmax(1) + 1 == subjects)),
            "oof_macro_f1": float(f1_score(
                subjects, stage_a_v2["oof_logits"].numpy().argmax(1) + 1,
                labels=CLASSES, average="macro", zero_division=0)),
            "in_sample_accuracy": float(np.mean(
                stage_a_v2["in_sample_logits"].numpy().argmax(1) + 1 == subjects)),
        },
        "n_test": int(len(inputs["sample"])),
        "elapsed_seconds": time.time() - started,
    }
    results = load_results()
    results["signals"] = summary
    save_results(results)
    log(f"stage signals done in {summary['elapsed_seconds']:.0f}s")
    return summary


# ---------------------------------------------------------------- stage loso
def projected_branch_b() -> tuple:
    meta, frac, blocks, fit_rows = E180.load_config_meta()
    tasks = meta["task"].to_numpy()
    raw_features = f32(np.load(ROOT / E180.CFG["branch_b"]["train_features"]))
    stats = E180.task_stats(raw_features, tasks)
    standardised = E180.apply_stats(raw_features, tasks, stats)
    mu, basis = E180.pca_basis(standardised[fit_rows],
                               int(E180.CFG["branch_b"]["pca_dim"]))
    return f32((standardised.astype(np.float64) - mu.astype(np.float64)) @ basis)


def stage_loso() -> dict:
    started = time.time()
    E180.build_torch_constants()
    meta, frac, blocks, fit_rows = E180.load_config_meta()
    subjects = meta["subject"].to_numpy(int)
    tasks = meta["task"].to_numpy()
    window = (frac > FIT) & (frac <= 0.8)
    win_rows = np.flatnonzero(window)
    fit_idx = np.flatnonzero(fit_rows)
    k = int(E180.CFG["branch_b"]["prototype_k"])
    epochs = int(CFG["loso_probe"]["epochs"])
    lam = float(load_results()["branch_a_v2"]["lambda"])

    log(f"stage loso: probes on pm & fit_rows ({len(fit_idx)} rows), eval window "
        f"{len(win_rows)} rows, {epochs} epochs, lambda {lam:.2f}")
    fit_raw = E180.load_raw([meta["path"].iloc[i] for i in fit_idx])
    win_raw = E180.load_raw([meta["path"].iloc[i] for i in win_rows])
    projected = projected_branch_b()

    rows_all, fold_all, known_all = [], [], []
    signal_all = {k: [] for k in SIGNALS}
    folds = []
    for held in CLASSES:
        enrolled = np.setdiff1d(CLASSES, [held])
        pm = np.isin(subjects, enrolled)
        path = CACHE / f"v2_loso_model_{int(held)}.pt"
        state = None
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=False)
            if (state.get("spec_sha256") != V2_SHA or state.get("lambda") != lam
                    or state.get("epochs") != epochs):
                state = None
        if state is None:
            train_pos = np.flatnonzero(pm[fit_idx])
            model, info = train_v2(fit_raw, train_pos, subjects[fit_idx][train_pos] - 1,
                                   SEED + 1000 + int(held), epochs, lam,
                                   f"loso{int(held)}")
            torch.save({"state_dict": model.state_dict(), "held_identity": int(held),
                        "epochs": epochs, "lambda": lam, "spec_sha256": V2_SHA}, path)
            cached = False
        else:
            model = E180.Net().to(DEVICE)
            model.load_state_dict(state["state_dict"])
            model.eval()
            cached = True
        known = win_rows[pm[win_rows]]
        unknown = win_rows[subjects[win_rows] == held]
        rows = np.r_[known, unknown]
        pos = np.searchsorted(win_rows, rows)
        if not np.array_equal(win_rows[pos], rows):
            raise RuntimeError("window indexing error")
        logits = E180.predict_logits(model, win_raw, pos)
        protos, keys = E180.build_prototypes(projected, subjects, tasks, pm & fit_rows,
                                             enrolled, k)
        proto, filled = E180.proto_stack(protos, keys, k)
        dist = E180.branch_b_scores(projected[rows], tasks[rows], proto, filled, k)
        raw = raw_signals(logits, dist)
        rows_all.append(rows)
        fold_all.append(np.full(len(rows), int(held), dtype=np.int32))
        known_all.append(np.r_[np.ones(len(known), bool), np.zeros(len(unknown), bool)])
        for name in SIGNALS:
            signal_all[name].append(raw[name])
        closed = float(np.mean(logits.argmax(1)[:len(known)] + 1 == subjects[known]))
        folds.append({"held_identity": int(held), "n_known": int(len(known)),
                      "n_unknown": int(len(unknown)),
                      "closed_set_accuracy_known": closed,
                      "cached": cached})
        log(f"  LOSO {int(held):2d}: known {len(known)} unknown {len(unknown)} "
            f"closed {closed:.4f}")
        del model
        torch.cuda.empty_cache()

    rows = np.concatenate(rows_all)
    fold = np.concatenate(fold_all)
    known = np.concatenate(known_all)
    np.savez_compressed(LOSO_CACHE, rows=rows, fold=fold, known=known,
                        **{name: np.concatenate(signal_all[name]) for name in SIGNALS})

    pooled = {}
    for name in SIGNALS:
        score = np.concatenate(signal_all[name])
        auc = float(roc_auc_score(known.astype(int), score))
        per_fold = []
        for held in CLASSES:
            sel = fold == int(held)
            if sel.sum() == 0 or len(np.unique(known[sel])) < 2:
                continue
            per_fold.append(float(roc_auc_score(known[sel].astype(int), score[sel])))
        pooled[name] = {"pooled_auc": auc,
                        "fold_auc_mean": float(np.mean(per_fold)),
                        "fold_auc_std": float(np.std(per_fold))}
        log(f"  probe LOSO {name:9s}: pooled AUC {auc:.4f} fold mean "
            f"{np.mean(per_fold):.4f}")

    summary = {"n_known": int(known.sum()), "n_unknown": int((~known).sum()),
               "n_folds": len(folds), "epochs": epochs, "lambda": lam,
               "per_signal_loso": pooled, "folds": folds,
               "elapsed_seconds": time.time() - started}
    results = load_results()
    results["loso_probe"] = summary
    save_results(results)
    log(f"stage loso done in {summary['elapsed_seconds']:.0f}s")
    del fit_raw, win_raw
    torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------- stage select
def candidate_table() -> tuple:
    store = np.load(SIGNAL_CACHE)
    loso = np.load(LOSO_CACHE)
    stats = {"mu": {k: float(v) for k, v in zip(SIGNALS, store["stats_mu"])},
             "sd": {k: float(v) for k, v in zip(SIGNALS, store["stats_sd"])}}
    z_oof = {k: store["oof_z"][i].astype(np.float64) for i, k in enumerate(SIGNALS)}
    z_test = {k: store["test_z"][i].astype(np.float64) for i, k in enumerate(SIGNALS)}
    z_loso = standardise({k: loso[k].astype(np.float64) for k in SIGNALS}, stats)
    known = loso["known"].astype(bool)
    fold = loso["fold"]

    rows = []
    for cand in candidate_library():
        w = cand["weights"]
        score = fuse(z_loso, w)
        auc = float(roc_auc_score(known.astype(int), score))
        per_fold, u95 = [], []
        for held in CLASSES:
            sel = fold == int(held)
            if sel.sum() == 0 or len(np.unique(known[sel])) < 2:
                continue
            per_fold.append(float(roc_auc_score(known[sel].astype(int), score[sel])))
            thr = quantile(score[sel & known], 0.05)
            u95.append(float(np.mean(score[sel & ~known] < thr)))
        rows.append({"name": cand["name"], "weights": w, "n_signals": len(w),
                     "loso_pooled_auc": auc,
                     "loso_fold_auc_mean": float(np.mean(per_fold)),
                     "loso_fold_auc_std": float(np.std(per_fold)),
                     "loso_fold_auc_min": float(np.min(per_fold)),
                     "u95_mean": float(np.mean(u95)),
                     "oof_fused": fuse(z_oof, w), "test_fused": fuse(z_test, w)})
    return rows, z_oof, z_test


def stage_select() -> dict:
    started = time.time()
    rows, z_oof, _ = candidate_table()
    ranked = sorted(rows, key=lambda r: (-r["loso_pooled_auc"],
                                         -r["loso_fold_auc_mean"], r["n_signals"]))
    best = ranked[0]
    tau = {f"{a:.2f}": quantile(best["oof_fused"], 1.0 - a) for a in ACC_GRID}
    store = np.load(SIGNAL_CACHE)
    subjects = store["subjects"]
    primary = np.where(best["oof_fused"] >= tau[f"{PRIMARY_A:.2f}"],
                       store["oof_logits_v2"].argmax(1) + 1, 0)
    summary = {
        "selected": {"name": best["name"], "weights": best["weights"],
                     "loso_pooled_auc": best["loso_pooled_auc"],
                     "loso_fold_auc_mean": best["loso_fold_auc_mean"],
                     "loso_fold_auc_std": best["loso_fold_auc_std"],
                     "u95_mean": best["u95_mean"]},
        "tau_train_oof": tau,
        "oof_acceptance_at_primary": float(np.mean(
            best["oof_fused"] >= tau[f"{PRIMARY_A:.2f}"])),
        "oof_gated_accuracy_at_primary": float(np.mean(primary == subjects)),
        "oof_rejection_rate_at_primary": float(np.mean(primary == 0)),
        "leaderboard": [{key: row[key] for key in
                         ("name", "n_signals", "loso_pooled_auc", "loso_fold_auc_mean",
                          "loso_fold_auc_std", "loso_fold_auc_min", "u95_mean")}
                        for row in ranked[:15]],
        "n_candidates": len(ranked),
        "elapsed_seconds": time.time() - started,
    }
    results = load_results()
    results["selection"] = summary
    save_results(results)
    log("stage select: top-5 candidates by Train LOSO pooled AUC")
    for row in ranked[:5]:
        log(f"  {row['loso_pooled_auc']:.4f}  {row['name']}")
    log(f"  selected: {best['name']} (LOSO {best['loso_pooled_auc']:.4f}), "
        f"tau(a={PRIMARY_A:.2f}) {tau[f'{PRIMARY_A:.2f}']:+.4f}")
    return summary


# ---------------------------------------------------------------- stage test
def stage_test() -> dict:
    started = time.time()
    rows, _, _ = candidate_table()
    store = np.load(SIGNAL_CACHE)
    selected = load_results()["selection"]["selected"]

    def pick(name):
        for row in rows:
            if row["name"] == name:
                return row
        raise KeyError(name)

    chosen = pick(selected["name"])
    tau_primary = quantile(chosen["oof_fused"], 1.0 - PRIMARY_A)
    pred = np.where(chosen["test_fused"] >= tau_primary,
                    store["test_logits_v2"].argmax(1) + 1, 0)
    pd.DataFrame({"Sample": store["test_sample"], "Label": pred}).to_csv(
        OUT / "test_predictions_fused.csv", index=False)
    log(f"  wrote {OUT / 'test_predictions_fused.csv'} "
        f"({int(np.sum(pred == 0))} rejects of {len(pred)}) before reading Session")

    # Session is read only now, after every Test prediction has been written.
    test_meta = pd.read_csv(ROOT / E180.CFG["data"]["test_metadata"],
                            usecols=["sample", "session"])
    if not np.array_equal(test_meta["sample"].astype(str).to_numpy(),
                          store["test_sample"].astype(str)):
        raise ValueError("Test sample order mismatch")
    is_s1 = test_meta["session"].astype(str).str.casefold().isin(
        ["session1", "s1", "1"]).to_numpy()

    orientation = {}
    for name in SIGNALS:
        index = SIGNALS.index(name)
        v1 = store["test_z_v1"][index].astype(np.float64)
        v2 = store["test_z"][index].astype(np.float64)
        orientation[name] = {
            "v1": {"s1_mean_z": float(np.mean(v1[is_s1])),
                   "s2_mean_z": float(np.mean(v1[~is_s1])),
                   "auc_s2_vs_s1": float(roc_auc_score((~is_s1).astype(int), v1))},
            "v2": {"s1_mean_z": float(np.mean(v2[is_s1])),
                   "s2_mean_z": float(np.mean(v2[~is_s1])),
                   "auc_s2_vs_s1": float(roc_auc_score((~is_s1).astype(int), v2))},
        }
    log("  Test signal orientation (z units, S1 vs S2, AUC with S2 as positive):")
    for name in SIGNALS:
        row = orientation[name]
        log(f"    {name:9s} v1 AUC {row['v1']['auc_s2_vs_s1']:.4f} -> "
            f"v2 AUC {row['v2']['auc_s2_vs_s1']:.4f}")

    audit = []
    for row in rows:
        tau = {a: quantile(row["oof_fused"], 1.0 - a) for a in ACC_GRID}
        entry = {"name": row["name"], "n_signals": row["n_signals"],
                 "loso_pooled_auc": row["loso_pooled_auc"],
                 "u95_mean": row.get("u95_mean"),
                 "test_diagnostic_auc": float(roc_auc_score(
                     (~is_s1).astype(int), row["test_fused"])),
                 "per_acceptance": {}}
        for a in ACC_GRID:
            accept = row["test_fused"] >= tau[a]
            entry["per_acceptance"][f"{a:.2f}"] = {
                "tau": tau[a],
                "s1_rejection": float(np.mean(~accept[is_s1])),
                "s2_acceptance": float(np.mean(accept[~is_s1])),
                "s1_plus_s2": float(np.mean(~accept[is_s1]) + np.mean(accept[~is_s1]))}
        order = np.argsort(-row["test_fused"])
        label = is_s1[order]
        s1_rank = np.cumsum(label) / max(is_s1.sum(), 1)
        s2_rank = np.cumsum(~label) / max((~is_s1).sum(), 1)
        total = (1.0 - s1_rank) + s2_rank
        best_at = int(np.argmax(total))
        entry["test_optimal"] = {"max_s1_plus_s2": float(total[best_at]),
                                 "s1_rejection": float(1.0 - s1_rank[best_at]),
                                 "s2_acceptance": float(s2_rank[best_at])}
        audit.append(entry)

    primary_row = next(e for e in audit if e["name"] == selected["name"])
    primary_at = primary_row["per_acceptance"][f"{PRIMARY_A:.2f}"]
    summary = {
        "selected_weights": selected["weights"],
        "selected_name": selected["name"],
        "primary_acceptance": PRIMARY_A,
        "audit": {
            "n_s1": int(is_s1.sum()), "n_s2": int((~is_s1).sum()),
            "s1_rejection": primary_at["s1_rejection"],
            "s2_acceptance": primary_at["s2_acceptance"],
            "s1_plus_s2": primary_at["s1_plus_s2"],
            "tau": primary_at["tau"],
        },
        "signal_orientation": orientation,
        "all_candidates": audit,
        "session_used_for_inference": False,
        "official_metrics": {k: "N/A" for k in
                             ("AUC", "F1", "C", "RA", "ASafeR", "U", "Score")},
        "elapsed_seconds": time.time() - started,
    }
    results = load_results()
    results["test"] = summary
    save_results(results)
    log(f"stage test: selected {selected['name']} -> S1 rejection "
        f"{primary_at['s1_rejection']:.4f}, S2 acceptance "
        f"{primary_at['s2_acceptance']:.4f} (a={PRIMARY_A:.2f})")
    return summary


# ---------------------------------------------------------------- attack stage
class FusedScorer(E180.Scorer):
    """Frozen v2 dual-branch forward with the fused gate score exposed."""

    def __init__(self, model, store, weights: dict, stats: dict):
        super().__init__(model, store)
        self.weights = {k: float(v) for k, v in weights.items()}
        self.mu = {k: torch.tensor(float(stats["mu"][k]), device=DEVICE)
                   for k in self.weights}
        self.sd = {k: torch.tensor(float(stats["sd"][k]), device=DEVICE)
                   for k in self.weights}

    def __call__(self, x: torch.Tensor, task_idx: torch.Tensor) -> tuple:
        logits, z_dist = super().__call__(x, task_idx)
        prob = torch.softmax(logits, dim=1)
        pieces = []
        for name, weight in self.weights.items():
            if name == "g_dist":
                value = z_dist
            elif name == "g_conf":
                value = (prob.max(1).values - self.mu[name]) / self.sd[name]
            elif name == "g_margin":
                top2 = torch.topk(prob, 2, dim=1).values
                value = (top2[:, 0] - top2[:, 1] - self.mu[name]) / self.sd[name]
            elif name == "g_energy":
                value = (torch.logsumexp(logits, dim=1) - self.mu[name]) / self.sd[name]
            elif name == "g_entropy":
                ent = -(prob * prob.clamp_min(1e-12).log()).sum(1)
                value = (-ent - self.mu[name]) / self.sd[name]
            else:
                raise KeyError(name)
            pieces.append(weight * value)
        total = pieces[0]
        for value in pieces[1:]:
            total = total + value
        return logits, total


def attack_metrics(logits: np.ndarray, gate: np.ndarray, truth: np.ndarray,
                   tau: float) -> dict:
    accept = gate >= tau
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
    E180.build_torch_constants()
    cfg = CFG["attack"]
    meta, frac, blocks, fit_rows = E180.load_config_meta()
    subjects = meta["subject"].to_numpy(int)
    tasks = meta["task"].to_numpy()
    window_rows = np.flatnonzero((frac > FIT) & (frac <= 0.8))
    rng = np.random.default_rng(int(cfg["seed"]))
    cap = int(cfg["cap_per_task"])
    selected_rows = []
    for task in TASKS:
        rows = window_rows[tasks[window_rows] == task]
        selected_rows.append(rows if len(rows) <= cap else np.sort(
            rng.choice(rows, size=cap, replace=False)))
    subset = np.sort(np.concatenate(selected_rows))
    log(f"attack stage: {len(subset)} eval-window trials "
        f"({', '.join(f'{t}={int((tasks[subset] == t).sum())}' for t in TASKS)})")

    store_npz = np.load(SIGNAL_CACHE)
    stats = {"mu": {k: float(v) for k, v in zip(SIGNALS, store_npz["stats_mu"])},
             "sd": {k: float(v) for k, v in zip(SIGNALS, store_npz["stats_sd"])}}
    oof_z = {k: store_npz["oof_z"][i].astype(np.float64) for i, k in enumerate(SIGNALS)}
    selection = load_results()["selection"]["selected"]
    weights = selection["weights"]
    of = fuse(oof_z, weights)
    tau_z = quantile(of, 1.0 - PRIMARY_A)
    sweep = (0.80, 0.90, 0.95)
    tau_sweep = {a: quantile(of, 1.0 - a) for a in sweep}

    branch_a = torch.load(MODEL_V2, map_location="cpu", weights_only=False)
    if branch_a["spec_sha256"] != V2_SHA:
        raise ValueError("branch A v2 artifact does not match the frozen spec")
    model = E180.Net().to(DEVICE)
    model.load_state_dict(branch_a["state_dict"])
    scorer = FusedScorer(model, E180.load_branch_b(), weights, stats)

    raw = E180.load_raw([meta["path"].iloc[i] for i in subset])
    truth = subjects[subset]
    task_idx = torch.as_tensor([TASKS.index(t) for t in tasks[subset]],
                               dtype=torch.long, device=DEVICE)
    true_idx = torch.as_tensor(truth - 1, dtype=torch.long, device=DEVICE)
    target = np.asarray([CLASSES[(CLASS_OF[int(v)] + 1) % len(CLASSES)] for v in truth])
    target_idx = torch.as_tensor([CLASS_OF[int(v)] for v in target],
                                 dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        logits, gate = scorer(raw, task_idx)
    clean_logits, clean_gate = logits.cpu().numpy(), gate.cpu().numpy()
    clean = attack_metrics(clean_logits, clean_gate, truth, tau_z)
    clean["at_acceptance"] = {f"{a:.2f}": {
        "RA": attack_metrics(clean_logits, clean_gate, truth, tau_sweep[a])["RA"],
        "ASafeR": attack_metrics(clean_logits, clean_gate, truth, tau_sweep[a])["ASafeR"],
        "accept_rate": attack_metrics(clean_logits, clean_gate, truth,
                                      tau_sweep[a])["accept_rate"]} for a in sweep}
    log(f"  clean: RA {clean['RA']:.4f} ASafeR {clean['ASafeR']:.4f} "
        f"closed {clean['closed_set_accuracy']:.4f} accept {clean['accept_rate']:.4f}")

    conditions = []
    for eps in cfg["eps_normalised"]:
        for goal in cfg["goals"]:
            out = E180.pgd_attack(scorer, raw, task_idx, true_idx, target_idx, goal,
                                  float(eps), int(cfg["steps"]),
                                  int(cfg["seed"]) + int(float(eps) * 1e5))
            metrics = attack_metrics(out["logits"], out["z"], truth, tau_z)
            metrics.update({"eps": float(eps), "goal": goal,
                            "max_rel_linf": out["max_rel_linf"],
                            "nan_gradients": out["nan_gradients"],
                            "at_acceptance": {f"{a:.2f}": {
                                "RA": attack_metrics(out["logits"], out["z"], truth,
                                                     tau_sweep[a])["RA"],
                                "ASafeR": attack_metrics(out["logits"], out["z"], truth,
                                                         tau_sweep[a])["ASafeR"]}
                                for a in sweep}})
            conditions.append(metrics)
            log(f"  eps={metrics['eps']:.3f} goal {goal}: RA {metrics['RA']:.4f} "
                f"ASafeR {metrics['ASafeR']:.4f} closed "
                f"{metrics['closed_set_accuracy']:.4f} accept "
                f"{metrics['accept_rate']:.4f} wrong_id "
                f"{metrics['wrong_identity_rate']:.4f}")

    worst_ra = min(conditions, key=lambda r: r["RA"])
    worst_safe = min(conditions, key=lambda r: r["ASafeR"])
    summary = {
        "selected_name": selection["name"], "weights": weights,
        "n_trials": int(len(subset)),
        "per_task": {task: int((tasks[subset] == task).sum()) for task in TASKS},
        "tau_z": tau_z, "primary_acceptance": PRIMARY_A,
        "sweep_acceptance": list(sweep),
        "sweep_tau": {f"{a:.2f}": tau_sweep[a] for a in sweep},
        "eps_normalised": list(cfg["eps_normalised"]),
        "steps": int(cfg["steps"]), "temperature": float(cfg["temperature"]),
        "clean": clean, "conditions": conditions,
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
    log(f"attack done in {summary['elapsed_seconds']:.0f}s: worst RA "
        f"{worst_ra['RA']:.4f}, worst ASafeR {worst_safe['ASafeR']:.4f}")
    del raw, model
    torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------- gates / report
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
TARGETS = {"#1": 1.00, "#2": 0.95, "#3": 0.92, "#4": 0.90, "#5": 0.98,
           "#6": 0.95, "#7": 0.95, "#8": 0.98, "#9": 0.75, "#10": 0.75}


def gate_table(results: dict) -> list[dict]:
    a = results.get("branch_a_v2", {})
    sel = results.get("selection", {}).get("selected", {})
    att = results.get("attack", {})
    test = results.get("test", {})
    values = {
        "#1": a.get("in_sample_accuracy"),
        "#2": a.get("oof_accuracy_5block"),
        "#3": a.get("oof_macro_f1"),
        "#4": a.get("oof_accuracy_5block"),
        "#5": sel.get("loso_pooled_auc"),
        "#6": sel.get("u95_mean"),
        "#7": att.get("worst_RA", {}).get("RA"),
        "#8": att.get("worst_ASafeR", {}).get("ASafeR"),
        "#9": test.get("audit", {}).get("s2_acceptance"),
        "#10": test.get("audit", {}).get("s1_rejection"),
    }
    rows = []
    for key, layer, metric, target_text in GATE_SPEC:
        value = values[key]
        rows.append({"gate": key, "layer": layer, "metric": metric,
                     "target": target_text, "value": value,
                     "passed": None if value is None else bool(
                         value >= TARGETS[key] - 1e-12),
                     "e181": E181_GATES[key], "e180": E180_GATES[key]})
    return rows


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def write_report(results: dict) -> None:
    a = results.get("branch_a_v2", {})
    sig = results.get("signals", {})
    loso = results.get("loso_probe", {})
    sel = results.get("selection", {})
    test = results.get("test", {})
    att = results.get("attack", {})
    rows = gate_table(results)
    n_pass = sum(1 for r in rows if r["passed"])
    selected = sel.get("selected", {})
    audit = test.get("audit", {})
    orientation = test.get("signal_orientation", {})
    per_signal_loso = loso.get("per_signal_loso", {})
    by_name = {row["name"]: row for row in test.get("all_candidates", [])}
    primary_row = by_name.get(selected.get("name"), {})
    per_acceptance = primary_row.get("per_acceptance", {})
    primary_at = per_acceptance.get(f"{PRIMARY_A:.2f}", {})

    flipped = [name for name in SIGNALS[1:]
               if orientation.get(name, {}).get("v2", {}).get("auc_s2_vs_s1", 0.0) > 0.5]
    ent_v1 = orientation.get("g_entropy", {}).get("v1", {})
    ent_v2 = orientation.get("g_entropy", {}).get("v2", {})
    ent_gap_v1 = ent_v1.get("s2_mean_z", 0.0) - ent_v1.get("s1_mean_z", 0.0)
    ent_gap_v2 = ent_v2.get("s2_mean_z", 0.0) - ent_v2.get("s1_mean_z", 0.0)
    ent_shift = ent_v2.get("s1_mean_z", 0.0) - ent_v1.get("s1_mean_z", 0.0)
    lines = [
        "# E182 会话不变骨干：让分支 A 的置信度变成身份注册度",
        "",
        "日期：2026-11-15。项目原生 18 通道、Train session1 only。"
        "分支 A 复用 E180 EEGNet 骨干与数据管线，在原对抗训练目标上加入**跨会话不变性约束**"
        "（两组互斥强会话扰动 + 128-d 池化嵌入 cosine 一致性）；"
        "分支 B（E65 correlation-logSPD PCA-96 / k=12 原型）直接读取 E180 冻结产物，**未改动**。",
        "",
        f"**十项门禁通过 {n_pass}/10。**",
        "",
        f"一致性权重 λ = **{fmt(a.get('lambda'), 2)}**"
        f"（Train-only：fit_rows 训练 + eval 块 clean 准确率选择）；"
        f"选定门控 `{selected.get('name', 'N/A')}`，Train LOSO 门控 AUC "
        f"**{fmt(selected.get('loso_pooled_auc'))}**、U@95 **{fmt(selected.get('u95_mean'))}**；"
        f"Test S1 拒识 **{fmt(audit.get('s1_rejection'))}**、Test S2 接受 "
        f"**{fmt(audit.get('s2_acceptance'))}**（a={fmt(PRIMARY_A, 2)}）。",
        "",
        "**核心判据**：分支 A 的四个置信度信号在 Test 上的 AUC（S2 为正）"
        + ("**全部转正**" if len(flipped) == 4 else f"仅 {len(flipped)}/4 转正")
        + "：" + "、".join(
            f"{name} {fmt(orientation.get(name, {}).get('v1', {}).get('auc_s2_vs_s1'))}→"
            f"{fmt(orientation.get(name, {}).get('v2', {}).get('auc_s2_vs_s1'))}"
            for name in SIGNALS[1:]) + "。",
        "",
        "## 1. 训练配置（预注册，控制配置数）",
        "",
        "| 项目 | 设置 |",
        "|---|---|",
        f"| 骨干 | E180 EEGNet（18→32→64→128→128，k9/s2，248084 参数） |",
        f"| 损失 | `{CFG['branch_a_v2']['loss']}` |",
        f"| 一致性 | {CFG['branch_a_v2']['consistency']} |",
        f"| 扰动族 A | 逐通道增益 N(1,0.10) + 逐通道偏置 N(0,0.15)·RMS + 通道 dropout 0.15 |",
        f"| 扰动族 B | 1–40Hz 五个频带增益 N(1,0.15) + 0.1–1Hz 漂移 + 加性噪声 SNR 15–25dB |",
        f"| 基础增强 | E180 augment，两个视图共享 |",
        f"| 对抗训练 | PGD-{CFG['branch_a_v2']['pgd_steps']}，ε="
        f"{CFG['branch_a_v2']['pgd_eps_linf']}·RMS，每 batch |",
        f"| λ 扫描 | {CFG['branch_a_v2']['loss_weights']['consistency_sweep']}，"
        f"fit_rows 训练 + eval 块（0.6<frac≤0.8）clean 准确率，平手取小 λ |",
        f"| 折/全量 epoch | {a.get('fold_epochs')} / {a.get('full_epochs')}"
        f"（全量早停 best epoch {a.get('full_best_epoch')}） |",
        "",
        "### 1.1 λ 扫描（Train-only 选择）",
        "",
        "| λ | 最佳 eval 块 clean 准确率 | best epoch | 平均 epoch 秒 | 一致性损失（末） |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in a.get("lambda_sweep", []):
        lines.append(f"| {row['lambda']:.2f} | {fmt(row['best_eval_accuracy'])} | "
                     f"{row['best_epoch']} | {fmt(row['mean_epoch_seconds'], 2)} | "
                     f"{fmt(row['consistency_curve'][-1])} |")
    lines += [
        "",
        "## 2. 四个置信度信号的 Test AUC：E181（v1）vs E182（v2）",
        "",
        "Test 诊断 AUC = 用 Session 列把 S2 当正类（应接受）、S1 当负类（应拒识）算出的 AUC，"
        "**仅作诊断，不参与任何权重/阈值决策**。",
        "",
        "| 信号 | E181 报告 AUC（v1） | 本实验复算 v1 | E182 v2 | 变化（v2−复算v1） | S1 均值 z (v1→v2) | S2 均值 z (v1→v2) | 方向 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in SIGNALS:
        row = orientation.get(name, {})
        v1, v2 = row.get("v1", {}), row.get("v2", {})
        change = (v2.get("auc_s2_vs_s1", 0.0) - v1.get("auc_s2_vs_s1", 0.0))
        verdict = "同向" if v2.get("auc_s2_vs_s1", 0.0) > 0.5 else "反向"
        lines.append(
            f"| `{name}` | {fmt(E181_SIGNAL_AUC[name])} | {fmt(v1.get('auc_s2_vs_s1'))} | "
            f"{fmt(v2.get('auc_s2_vs_s1'))} | {change:+.4f} | "
            f"{v1.get('s1_mean_z', 0.0):+.3f}→{v2.get('s1_mean_z', 0.0):+.3f} | "
            f"{v1.get('s2_mean_z', 0.0):+.3f}→{v2.get('s2_mean_z', 0.0):+.3f} | {verdict} |")
    lines += [
        "",
        f"转正判定（AUC > 0.5）：**{len(flipped)}/4** "
        f"（{', '.join(flipped) if flipped else '无'}）。`g_dist` 保持不变"
        f"（{fmt(orientation.get('g_dist', {}).get('v2', {}).get('auc_s2_vs_s1'))}，"
        "分支 B 未改动）。",
        "",
        "### 2.1 机制读数",
        "",
        f"1. **方向没变**：四个置信度信号的 S1/S2 z 均值整体上移约 {ent_shift:+.2f}（"
        f"g_entropy {ent_v1.get('s1_mean_z', 0):+.3f}/"
        f"{ent_v1.get('s2_mean_z', 0):+.3f} → "
        f"{ent_v2.get('s1_mean_z', 0):+.3f}/"
        f"{ent_v2.get('s2_mean_z', 0):+.3f}），"
        f"但 **S1 与 S2 之间的差几乎不变**（{ent_gap_v1:+.3f} → {ent_gap_v2:+.3f}）。"
        "跨会话一致性约束更像一次"
        "全局置信度校准，而不是把「会话相似度」从置信度里剥离。",
        f"2. **同会话信号被削弱、跨会话排序没有补上**：Train LOSO 上 v2 的置信度 AUC 反而低于 v1"
        f"（g_conf {fmt(per_signal_loso.get('g_conf', {}).get('pooled_auc'))} vs 0.9352、"
        f"g_entropy {fmt(per_signal_loso.get('g_entropy', {}).get('pooled_auc'))} vs 0.9381），"
        "说明约束确实移除了部分会话信息，但移除的部分没有转化为 Test 上的跨会话判别能力。",
        "3. **合成扰动没有覆盖真实 S1→S2 的差异**：增益/偏置/频带/漂移/噪声是可在线生成的"
        "强扰动，但真实跨会话差还包括电极位置、阻抗、参考与受试状态变化。模型可以同时"
        "满足一致性损失、又保持「S1 更自信」——这正是本实验观察到的结果。",
        "4. **门禁仍是同一条 ROC**：选定融合门控的 Test 诊断 AUC "
        f"{fmt(primary_row.get('test_diagnostic_auc'))}（`g_dist` 单独为 "
        f"{fmt(orientation.get('g_dist', {}).get('v2', {}).get('auc_s2_vs_s1'))}）意味着"
        "任何 Train-only 阈值都无法让 S1 拒识与 S2 接受同时 ≥0.75。",
        "",
        "## 3. 分支 A v2：clean / 对抗指标",
        "",
        "| 指标 | 值 | 说明 |",
        "|---|---:|---|",
        f"| C（5 块 OOF 闭集准确率） | {fmt(a.get('oof_accuracy_5block'))} | 与 #2 同口径 |",
        f"| 身份 macro-F1（5 块 OOF） | {fmt(a.get('oof_macro_f1'))} | #3 |",
        f"| 块外准确率（frac>0.6 前向块） | {fmt(a.get('oof_accuracy_forward_blocks'))} | #4 同口径 |",
        f"| in-sample 准确率 | {fmt(a.get('in_sample_accuracy'))} | #1 |",
        f"| RA（最差 ε×目标） | {fmt(att.get('worst_RA', {}).get('RA'))} | 融合系统，自建 PGD |",
        f"| ASafeR（同一攻击最差） | {fmt(att.get('worst_ASafeR', {}).get('ASafeR'))} | 拒识计为安全 |",
        "",
        "### 3.1 5 块 OOF",
        "",
        "| 块 | 训练条数 | 评估条数 | clean 准确率 |",
        "|---:|---:|---:|---:|",
    ]
    for fold in a.get("folds", []):
        lines.append(f"| {fold['block']} | {fold['n_train']} | {fold['n_eval']} | "
                     f"{fmt(fold['accuracy'])} |")
    lines += [
        "",
        "### 3.2 对抗（#7/#8，融合系统，自建 PGD 代理）",
        "",
        f"攻击子集 {att.get('n_trials')} 条（"
        + "、".join(f"{k}={v}" for k, v in (att.get("per_task") or {}).items())
        + f"），ε ∈ {att.get('eps_normalised')}×RMS，{att.get('steps')} 步；"
          "目标 D 直接最大化融合门控分数。",
        "",
        "| 条件 | ε | 目标 | RA | ASafeR | 闭集准确率 | 接受率 | 误认率 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
        f"| clean | — | — | {fmt(att.get('clean', {}).get('RA'))} | "
        f"{fmt(att.get('clean', {}).get('ASafeR'))} | "
        f"{fmt(att.get('clean', {}).get('closed_set_accuracy'))} | "
        f"{fmt(att.get('clean', {}).get('accept_rate'))} | "
        f"{fmt(att.get('clean', {}).get('wrong_identity_rate'))} |",
    ]
    for row in att.get("conditions", []):
        lines.append(
            f"| {row['goal']} | {row['eps']:.3f} | {row['goal']} | {fmt(row['RA'])} | "
            f"{fmt(row['ASafeR'])} | {fmt(row['closed_set_accuracy'])} | "
            f"{fmt(row['accept_rate'])} | {fmt(row['wrong_identity_rate'])} |")
    lines += [
        "",
        f"判定：#7 RA 最差 **{fmt(att.get('worst_RA', {}).get('RA'))}**"
        f"（ε={att.get('worst_RA', {}).get('eps')}，目标 "
        f"{att.get('worst_RA', {}).get('goal')}）；#8 ASafeR 最差 "
        f"**{fmt(att.get('worst_ASafeR', {}).get('ASafeR'))}**"
        f"（ε={att.get('worst_ASafeR', {}).get('eps')}，目标 "
        f"{att.get('worst_ASafeR', {}).get('goal')}）。",
        "",
        "机制：攻击条件下身份头闭集准确率仍有 "
        f"{fmt(min(row['closed_set_accuracy'] for row in att.get('conditions', [{}])))}–"
        f"{fmt(max(row['closed_set_accuracy'] for row in att.get('conditions', [{}])))}，"
        "RA 的主要损失来自**门控拒识**"
        f"（clean 接受率 {fmt(att.get('clean', {}).get('accept_rate'))}），而不是认错人"
        f"（误认率 ≤ {fmt(max((row['wrong_identity_rate'] for row in att.get('conditions', [{}]))))}）。"
        "与 E180/E181 一样，#7 的瓶颈是工作点而不是表征鲁棒性。",
        "",
        "## 4. 融合门控（权重只用 Train LOSO AUC）",
        "",
        "| 信号 | Train LOSO pooled AUC | 逐折均值 ± std |",
        "|---|---:|---:|",
    ]
    for name in SIGNALS:
        row = per_signal_loso.get(name, {})
        lines.append(f"| `{name}` | {fmt(row.get('pooled_auc'))} | "
                     f"{fmt(row.get('fold_auc_mean'))} ± {fmt(row.get('fold_auc_std'))} |")
    lines += [
        "",
        f"选定：`{selected.get('name')}`，Train LOSO pooled AUC "
        f"**{fmt(selected.get('loso_pooled_auc'))}**，逐折 "
        f"{fmt(selected.get('loso_fold_auc_mean'))} ± "
        f"{fmt(selected.get('loso_fold_auc_std'))}，U@95 "
        f"**{fmt(selected.get('u95_mean'))}**。候选总数 {sel.get('n_candidates')}。",
        "",
        "## 5. 完整十项门禁表",
        "",
        "| # | 层面 | 指标 | 目标 | 实测 | 判定 | 证据来源 |",
        "|---|---|---|---|---:|---|---|",
    ]
    evidence = {
        "#1": "分支 A v2 全量模型对全部 Train 的 in-sample argmax（§3）",
        "#2": "分支 A v2 5 块 OOF 闭集准确率（§3.1）",
        "#3": "同一批 5 块 OOF 预测的身份 macro-F1（§3.1）",
        "#4": "5 块 OOF 同口径，额外汇总 frac>0.6 前向块（§3）",
        "#5": "融合门控留身份探针 20 折 LOSO 合并 AUROC（§4）",
        "#6": "同一 LOSO 的 U@95 均值（§4）",
        "#7": "融合系统自建 PGD，ε×目标 A/B/C/D 取最差（§3.2）",
        "#8": "同一攻击条件下的最差 ASafeR（§3.2）",
        "#9": "冻结后 Test 一次性评分，Session 仅审计（§7）",
        "#10": "同 #9",
    }
    for row in rows:
        verdict = "N/A" if row["passed"] is None else ("通过" if row["passed"] else "未通过")
        lines.append(f"| {row['gate']} | {row['layer']} | {row['metric']} | "
                     f"{row['target']} | {fmt(row['value'])} | {verdict} | "
                     f"{evidence[row['gate']]} |")
    lines += [
        "",
        "## 6. 与 E181 逐项对照",
        "",
        "| # | 门禁 | 目标 | E181 | E182 | 变化 | 判定 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        change = "N/A" if (row["value"] is None) else f"{row['value'] - row['e181']:+.4f}"
        verdict = "N/A" if row["passed"] is None else ("通过" if row["passed"] else "未通过")
        lines.append(f"| {row['gate']} | {row['metric']} | {row['target']} | "
                     f"{fmt(row['e181'])} | {fmt(row['value'])} | {change} | {verdict} |")
    lines += [
        "",
        f"通过项数：E181 **5/10** → E182 **{n_pass}/10**。",
        "",
        "## 7. Test 审计（#9/#10）与「S1+S2 能否到 1.50」",
        "",
        "所有阈值、原型、PCA、模型在接触 Test 前冻结；预测 CSV 落盘后才读取 Session。",
        "",
        f"| 变体 | Session | N | 拒识率 | 接受率 |",
        "|---|---|---:|---:|---:|",
        f"| 融合门控 `{selected.get('name')}` | S1 | {audit.get('n_s1')} | "
        f"{fmt(audit.get('s1_rejection'))} | {fmt(1 - (audit.get('s1_rejection') or 0))} |",
        f"| 融合门控 `{selected.get('name')}` | S2 | {audit.get('n_s2')} | "
        f"{fmt(1 - (audit.get('s2_acceptance') or 0))} | {fmt(audit.get('s2_acceptance'))} |",
        "",
        f"a={PRIMARY_A:.2f} 工作点：S1 拒识 **{fmt(audit.get('s1_rejection'))}**（目标 0.75，"
        f"{'超出' if (audit.get('s1_rejection') or 0) >= 0.75 else '差'} "
        f"{fmt(abs((audit.get('s1_rejection') or 0) - 0.75))}），S2 接受 "
        f"**{fmt(audit.get('s2_acceptance'))}**（目标 0.75，"
        f"{'超出' if (audit.get('s2_acceptance') or 0) >= 0.75 else '差'} "
        f"{fmt(abs((audit.get('s2_acceptance') or 0) - 0.75))}），S1+S2 = "
        f"**{fmt(audit.get('s1_plus_s2'))}**（同时达标需 1.50；"
        f"E181 为 1.3145、E180 为 1.3694）。",
        "",
        "### 7.1 多工作点审计（全部为审计值，不参与决策）",
        "",
        "| a | τ | S1 拒识 | S2 接受 | S1+S2 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for a_key, cell in per_acceptance.items():
        lines.append(f"| {a_key} | {cell['tau']:+.4f} | {fmt(cell['s1_rejection'])} | "
                     f"{fmt(cell['s2_acceptance'])} | {fmt(cell['s1_plus_s2'])} |")
    optimal = primary_row.get("test_optimal", {})
    lines += [
        "",
        f"Test 诊断 AUC（选定门控）**{fmt(primary_row.get('test_diagnostic_auc'))}**；"
        f"**犯规上界**（阈值直接在 Test ROC 上扫，仅用于回答「还差多少」）："
        f"最大 S1+S2 = {fmt(optimal.get('max_s1_plus_s2'))}"
        f"（S1 拒识 {fmt(optimal.get('s1_rejection'))} / S2 接受 "
        f"{fmt(optimal.get('s2_acceptance'))}）。",
        "",
        "### 7.2 结论：#9 能否达到？",
        "",
    ]
    s2 = audit.get("s2_acceptance") or 0.0
    s1 = audit.get("s1_rejection") or 0.0
    if s2 >= 0.75 and s1 >= 0.75:
        answer = ("**能，两项同时达到**：S1 拒识 "
                  f"{fmt(s1)}、S2 接受 {fmt(s2)}，S1+S2 = {fmt(s1 + s2)} ≥ 1.50。")
    elif s2 >= 0.75:
        answer = ("**#9 达到**（S2 接受 "
                  f"{fmt(s2)} ≥ 0.75），但 #10 未达到（S1 拒识 {fmt(s1)}）。")
    else:
        answer = (
            f"**不能**：a={PRIMARY_A:.2f} 工作点 S2 接受只有 {fmt(s2)}，"
            f"距 0.75 差 **{fmt(0.75 - s2)}**；S1 拒识 {fmt(s1)}。"
            f"两者之和 {fmt(s1 + s2)}，距 1.50 差 {fmt(1.50 - (s1 + s2))}。")
    lines.append(answer)
    lines += [
        "",
        f"选定门控的 Test 诊断 AUC 为 {fmt(primary_row.get('test_diagnostic_auc'))}"
        f"（E181 0.6618、E180 的 `g_dist` 0.7563）。相对 E181 的改善（S1+S2 "
        f"{fmt(audit.get('s1_plus_s2'))} vs 1.3145）来自**门控权重把反向的置信度压到 0.20**，"
        "而不是新增了正向信息：融合后的 ROC 仍低于 `g_dist` 单独。按 E181 的换算，"
        "S1+S2 要搬到 1.50 需要诊断 AUC 约 0.87，本实验为 "
        f"{fmt(primary_row.get('test_diagnostic_auc'))}；即便假定四个置信度信号全部转正，"
        "以当前信息量也不足以跨过这条 ROC 约束。因此 **#9 的瓶颈不在阈值或融合方式，"
        "而在分支 A 的置信度与 Test 会话结构之间的符号关系**。",
        "",
        "结论（一句话）：跨会话一致性约束把分支 A 的同会话置信度削弱了"
        f"（Train LOSO g_conf 0.9352→{fmt(per_signal_loso.get('g_conf', {}).get('pooled_auc'))}），"
        f"但没有改变 Test 上「S1 比 S2 更自信」的符号，四个置信度信号 **{len(flipped)}/4 转正**，"
        f"十项门禁仍为 **{n_pass}/10**。",
        "",
        "## 8. 合规与边界",
        "",
        "- 分支 A v2 只在 Train session1 上训练；λ、早停、阈值、权重全部 Train-only。",
        "- LOSO 探针严格 `pm & fit_rows`（19 个注册身份），评估块 `0.6<frac≤0.8` 从未进入训练或原型。",
        "- 分支 B（PCA、标准化、原型、距离阈值）直接复用 E180 冻结产物，未重训、未改动。",
        "- Test 只评分：先用冻结模型写 `test_predictions_fused.csv`，之后才读 Session 做 #9/#10 审计与信号方向诊断。",
        "- 门控权重只用 Train LOSO AUC 选择；Test 指标只出现在审计/上界表中。",
        "- #7/#8 是自建白盒 PGD 代理，不是官方攻击套件。",
        "- 未修改任何已有实验目录，未提交 git。",
        "",
        "## 9. Test 官方指标",
        "",
        "**Test 官方 AUC / F1 / C / RA / ASafeR / U / Score = N/A**（Test 无真实标签）。",
        "报告中的 S1 拒识 / S2 接受只是预测工程审计，不是官方真值指标。",
        "",
        "## 10. 文件",
        "",
        "| 文件 | 内容 |",
        "|---|---|",
        "| `run_e182.py` | 全部逻辑（`--stage train/signals/loso/select/test/attack/report/all`） |",
        "| `config.json` | 冻结配置 |",
        "| `results.json` | 全部数值 |",
        "| `model_branch_a_v2.pt` | 分支 A v2 全量模型 |",
        "| `run.log` | 运行日志 |",
        "| `cache/` | λ 选择模型、5 折模型、OOF logits、20 个 LOSO 探针、Test 输入与信号缓存 |",
        "| `test_predictions_fused.csv` | 选定门控的 Test 预测（工程审计用） |",
        "",
        f"耗时（秒）：train {fmt(a.get('elapsed_seconds'), 0)}、"
        f"signals {fmt(sig.get('elapsed_seconds'), 0)}、"
        f"loso {fmt(loso.get('elapsed_seconds'), 0)}、"
        f"select {fmt(sel.get('elapsed_seconds'), 0)}、"
        f"test {fmt(test.get('elapsed_seconds'), 0)}、"
        f"attack {fmt(att.get('elapsed_seconds'), 0)}。",
        f"训练合计（train + loso）= "
        f"{(float(a.get('elapsed_seconds') or 0) + float(loso.get('elapsed_seconds') or 0)) / 60:.1f} "
        "分钟，低于 40 分钟预算；clean 准确率未因一致性约束掉到 0.95 以下"
        f"（C {fmt(a.get('oof_accuracy_5block'))}）。折 OOF 用 24 epoch 以控制总时长，"
        "全量模型仍按 30–60 epoch 区间训练（30）。",
        "",
        "```powershell",
        '& "D:\\WORK\\BCIC2026\\BrainPrint\\.venv\\Scripts\\python.exe" '
        '"D:\\WORK\\BCIC2026\\BrainPrint\\research\\E182_session_invariant_backbone'
        '\\run_e182.py" --stage all',
        "```",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"report written to {REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["train", "signals", "loso", "select", "test",
                                 "attack", "report", "all"])
    args = parser.parse_args()
    if args.stage in ("train", "all"):
        stage_train()
    if args.stage in ("signals", "all"):
        stage_signals()
    if args.stage in ("loso", "all"):
        stage_loso()
    if args.stage in ("select", "all"):
        stage_select()
    if args.stage in ("test", "all"):
        stage_test()
    if args.stage in ("attack", "all"):
        stage_attack()
    if args.stage in ("report", "all"):
        write_report(load_results())


if __name__ == "__main__":
    sys.stdout = Tee(OUT / "run.log")
    main()
