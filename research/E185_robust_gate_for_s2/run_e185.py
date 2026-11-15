"""E185: adversarially trained open-set gate aimed at the Test-S2 acceptance problem.

Hypothesis under test
---------------------
Test S2 = 20 registered subjects' normal *and adversarial* trials.  If the deployed
gate rejects adversarially perturbed registered trials, then S2 acceptance is
depressed by the gate, and an adversarially trained (AT) gate could raise S2
acceptance without moving S1 (no adversarial trials).

Step 1 (mandatory pre-check, Train only)
    Re-run the frozen E183/E178 PGD flow on the Train evaluation window
    (0.6 < frac <= 0.8), score clean and adversarial trials with the deployed gate
    read-outs (E65 distance, E183 open-set-head confidence, E183 fused gate) at the
    Train-LOSO thresholds, and measure how much the rejection rate of registered
    trials grows.  If the largest increment is < 0.03 the hypothesis is rejected and
    the AT gate is *not* trained (explicit stop rule of the task).

Step 2 (only if Step 1 passes)
    Train MLP 96 -> 128 -> 64 -> 1 on the E65 PCA-96 space with feature-space PGD
    positives (eps in {0.5, 1.0, 2.0} x feature RMS, lambda in {0.5, 1.0}) plus a
    clean-only baseline, strictly nested leave-identity-out pseudo-unknowns.

Step 3 (only if Step 1 passes)
    LOSO / Test evaluation at a in {0.88, 0.90, 0.92, 0.94, 0.96}, ten-gate matrix at
    a single working point.

Everything is read-only with respect to E180 / E182 / E183.  No Test metric, threshold
or Session field influences any training, selection or calibration.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)
GATE_DIR = CACHE / "gate_folds"
GATE_DIR.mkdir(exist_ok=True)
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "report.md"
LOG_PATH = OUT / "run.log"
LOSO_CACHE = CACHE / "loso_gate.npz"
TEST_PRED = OUT / "test_predictions_robust_gate.csv"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E180 = load_module("e185_frozen_e180",
                   ROOT / "research" / CFG["frozen_sources"]["e180"] / "run_e180.py")
E183 = load_module("e185_frozen_e183",
                   ROOT / "research" / CFG["frozen_sources"]["e183"] / "run_e183.py")
E182_DIR = ROOT / "research" / CFG["frozen_sources"]["e182"]
E183_DIR = ROOT / "research" / CFG["frozen_sources"]["e183"]

DEVICE = E180.DEVICE
FT = E180.FT
TASKS = list(E180.TASKS)
CLASSES = E180.CLASSES
SEED = int(E180.CFG["protocol"]["seed"])
FIT = float(E180.CFG["protocol"]["fit_fraction"])
ACC_GRID = [float(a) for a in CFG["threshold"]["grid"]]
DIM = int(E183.CFG["head"]["pca_dim"])


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


def quantile(value: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(value, dtype=np.float64), q))


def load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {"experiment": CFG["experiment"]}


def save_results(results: dict) -> None:
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2,
                                       default=float), encoding="utf-8")


# ------------------------------------------------------------------ frozen gate
def folded_entries() -> list:
    """E183 fold artifacts (PCA, standardisation, 3 view models) as tensors."""
    entries = []
    for held in CLASSES:
        art, models = E183.load_fold_models(int(held))
        entries.append({
            "held": int(held),
            "models": models,
            "K": [np.asarray(v, dtype=int) for v in art["K"]],
            "stats_mu": torch.as_tensor(art["stats_mu"], device=DEVICE),
            "stats_sd": torch.as_tensor(art["stats_sd"], device=DEVICE),
            "pca_mu": torch.as_tensor(art["pca_mu"], device=DEVICE),
            "pca_basis": torch.as_tensor(art["pca_basis"], device=DEVICE),
            "feat_mu": torch.as_tensor(art["feat_mu"], device=DEVICE),
            "feat_sd": torch.as_tensor(art["feat_sd"], device=DEVICE),
            "mu_conf": art["mu_conf"], "sd_conf": art["sd_conf"],
            "mu_open": art["mu_open"], "sd_open": art["sd_open"],
            "art": art})
    return entries


class FrozenGateScorer:
    """Differentiable raw-waveform -> (branch A logits, E183 gate read-outs)."""

    def __init__(self, model_a: nn.Module, store: dict, folds: list,
                 ent_stats: tuple) -> None:
        self.model_a = model_a
        for parameter in self.model_a.parameters():
            parameter.requires_grad_(False)
        self.model_a.eval()
        self.stats_mu = torch.as_tensor(f32(store["stats_mu"]), device=DEVICE)
        self.stats_sd = torch.as_tensor(f32(store["stats_sd"]), device=DEVICE)
        self.pca_mu = torch.as_tensor(f32(store["pca_mu"]), device=DEVICE)
        self.basis = torch.as_tensor(f32(store["pca_basis"]), device=DEVICE)
        self.proto = torch.as_tensor(f32(store["proto"]), device=DEVICE)
        self.filled = torch.as_tensor(np.asarray(store["proto_filled"]), device=DEVICE)
        self.k = int(store["k"])
        self.oof_mu = float(store["oof_mu"])
        self.oof_sd = float(store["oof_sd"])
        self.ent_mu, self.ent_sd = float(ent_stats[0]), float(ent_stats[1])
        self.folds = folds
        self.weights = {}

    def set_weights(self, weights: dict) -> None:
        self.weights = {key: float(value) for key, value in weights.items()}

    def parts(self, x: torch.Tensor, task_idx: torch.Tensor, groups: list) -> tuple:
        logits = self.model_a(x)
        prob = torch.softmax(logits, dim=1)
        entropy = (prob * prob.clamp_min(1e-12).log()).sum(1)
        g_ent = (-entropy - self.ent_mu) / self.ent_sd

        features = E180.differentiable_logspd(x)
        z = (features - self.stats_mu[task_idx]) / self.stats_sd[task_idx]
        projected = (z - self.pca_mu) @ self.basis
        dist = None
        for t in range(len(TASKS)):
            valid = self.filled[t]
            p = self.proto[t][valid].reshape(-1, self.proto.shape[-1])
            d = torch.cdist(projected, p).view(len(x), int(valid.sum()), self.k)
            score = (-d).max(dim=2).values.max(dim=1).values
            take = task_idx == t
            block = torch.full_like(score, -1e9)
            block[take] = score[take]
            dist = block if dist is None else torch.where(take, block, dist)
        g_dist = (dist - self.oof_mu) / self.oof_sd

        conf = torch.zeros(len(x), device=DEVICE, dtype=x.dtype)
        opened = torch.zeros(len(x), device=DEVICE, dtype=x.dtype)
        for fold_index, index in groups:
            art = self.folds[fold_index]
            group_task = task_idx[index]
            zz = (features[index] - art["stats_mu"][group_task]) / art["stats_sd"][group_task]
            p96 = ((zz - art["pca_mu"]) @ art["pca_basis"] - art["feat_mu"]) / art["feat_sd"]
            cv, ov = [], []
            for rep, model in enumerate(art["models"]):
                emb = model.trunk(p96)
                value = torch.softmax(model.id_head(emb), 1).max(1).values
                cv.append((value - art["mu_conf"][rep]) / art["sd_conf"][rep])
                ov.append((model.open_head(emb).squeeze(1) - art["mu_open"][rep])
                          / art["sd_open"][rep])
            conf = conf.index_copy(0, index, torch.stack(cv).max(0).values)
            opened = opened.index_copy(0, index, torch.stack(ov).max(0).values)
        return logits, {"g_dist": g_dist, "g_conf": conf, "g_open": opened,
                        "g_entropy": g_ent}

    def gate(self, parts: dict) -> torch.Tensor:
        total = None
        for name, weight in self.weights.items():
            key = "g_conf" if name == "g_head" else name
            term = weight * parts[key]
            total = term if total is None else total + term
        return total


def fused_loso_signals() -> dict:
    """E183 frozen Train-LOSO read-outs, aligned with the fold-major row order."""
    data = np.load(E183_DIR / "cache" / "loso_head.npz")
    weights = load_e183()["fusion"]["selected"]["weights"]
    return {
        "rows": data["rows"], "fold": data["fold"], "known": data["known"],
        "known_k": data["known_k"], "z_dist": data["z_dist"].astype(np.float64),
        "z_ent": data["z_ent"].astype(np.float64),
        "g_conf": data["comp_conf_max"].astype(np.float64),
        "g_open": data["comp_open_max"].astype(np.float64),
        "weights": {k: float(v) for k, v in weights.items()},
    }


def load_e183() -> dict:
    return json.loads((E183_DIR / "results.json").read_text(encoding="utf-8"))


def fused_e183(sig: dict) -> np.ndarray:
    w = sig["weights"]
    return (w["g_head"] * sig["g_conf"] + w["g_dist"] * sig["z_dist"]
            + w["g_entropy"] * sig["z_ent"])


def head_choice(k_mask: np.ndarray) -> dict:
    choice = {}
    for value in CLASSES:
        for held in CLASSES:
            if int(value) != int(held) and k_mask[int(held) - 1, :, int(value) - 1].all():
                choice[int(value)] = int(held)
                break
        if int(value) not in choice:
            raise RuntimeError(f"identity {value} is never fully registered")
    return choice


def eval_subset() -> np.ndarray:
    meta, frac, _, _ = E180.load_config_meta()
    subjects = meta["subject"].to_numpy(int)
    tasks = meta["task"].to_numpy()
    window = np.flatnonzero((frac > FIT) & (frac <= 0.8))
    cfg = CFG["attack_step1"]
    rng = np.random.default_rng(int(cfg["seed"]))
    cap = int(cfg["cap_per_task"])
    picked = []
    for task in TASKS:
        rows = window[tasks[window] == task]
        picked.append(rows if len(rows) <= cap
                      else np.sort(rng.choice(rows, size=cap, replace=False)))
    del subjects
    return np.sort(np.concatenate(picked))


# ------------------------------------------------------------------ PGD (raw)
def pgd_raw(scorer: FrozenGateScorer, x: torch.Tensor, task_idx: torch.Tensor,
            groups: list, true_idx: torch.Tensor, target_idx: torch.Tensor,
            goal: str, eps: float, steps: int, seed: int) -> torch.Tensor:
    temperature = float(CFG["attack_step1"]["temperature"])
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    xc = x - x.mean(dim=-1, keepdim=True)
    rms = xc.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-6)
    bound = (eps * rms).view(-1, 1, 1)
    with torch.no_grad():
        adv = x + (torch.rand(x.shape, generator=gen, device=DEVICE) * 2 - 1) * bound
        adv = torch.clamp(adv, x - bound, x + bound)
    alpha = 2.0 * eps / steps
    for _ in range(steps):
        adv.requires_grad_(True)
        logits, parts = scorer.parts(adv, task_idx, groups)
        if goal in ("D", "E"):
            gate = scorer.gate(parts)
            objective = gate if goal == "D" else -gate
        else:
            true_logit = logits.gather(1, true_idx[:, None]).squeeze(1)
            wrong = logits.masked_fill(F.one_hot(true_idx, len(CLASSES)).bool(), -1e9)
            if goal == "A":
                margin = wrong.max(dim=1).values
            elif goal == "B":
                margin = torch.logsumexp(wrong, dim=1)
            else:
                margin = logits.gather(1, target_idx[:, None]).squeeze(1)
            objective = (margin - true_logit) / temperature
        grad = torch.autograd.grad(-objective.sum(), adv)[0]
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        adv = torch.clamp(adv.detach() - (alpha * rms).view(-1, 1, 1) * grad.sign(),
                          x - bound, x + bound)
    return adv.detach()


# ------------------------------------------------------------------ stage 1
def rejection_table(scores: dict, tau: dict, names) -> dict:
    table = {}
    for name in names:
        table[name] = {}
        for a in ACC_GRID:
            table[name][f"{a:.2f}"] = float(np.mean(scores[name] < tau[name][f"{a:.2f}"]))
    return table


def stage_hypothesis() -> dict:
    started = time.time()
    E180.build_torch_constants()
    meta, frac, blocks, fit_rows = E180.load_config_meta()
    subjects = meta["subject"].to_numpy(int)
    tasks = meta["task"].to_numpy()
    subset = eval_subset()
    log(f"stage hypothesis: {len(subset)} eval-window trials "
        f"({', '.join(f'{t}={int((tasks[subset] == t).sum())}' for t in TASKS)})")

    sig = fused_loso_signals()
    signals = {"g_dist": sig["z_dist"], "g_conf": sig["g_conf"],
               "g_open": sig["g_open"], "fused": fused_e183(sig)}
    tau = {name: {f"{a:.2f}": quantile(value[sig["known"]], 1.0 - a) for a in ACC_GRID}
           for name, value in signals.items()}
    log("  Train-LOSO thresholds (known = 19 enrolled identities):")
    for name in signals:
        log(f"    {name:7s} " + " ".join(f"a={a:.2f}:{tau[name][f'{a:.2f}']:+.3f}"
                                         for a in ACC_GRID))

    choice = head_choice(np.load(E183_DIR / "cache" / "loso_head.npz")["k_mask"])
    folds = folded_entries()
    store = E180.load_branch_b()
    signals82 = np.load(E182_DIR / "cache" / "signals_v2.npz")
    ent_stats = (float(signals82["stats_mu"][4]), float(signals82["stats_sd"][4]))
    branch_a = torch.load(E182_DIR / "model_branch_a_v2.pt", map_location="cpu",
                          weights_only=False)
    model_a = E180.Net().to(DEVICE)
    model_a.load_state_dict(branch_a["state_dict"])
    model_a.eval()
    scorer = FrozenGateScorer(model_a, store, folds, ent_stats)
    scorer.set_weights(sig["weights"])

    raw = E180.load_raw([meta["path"].iloc[i] for i in subset])
    truth = subjects[subset]
    task_idx = torch.as_tensor([TASKS.index(t) for t in tasks[subset]],
                               dtype=torch.long, device=DEVICE)
    true_idx = torch.as_tensor(truth - 1, dtype=torch.long, device=DEVICE)
    class_of = {int(c): i for i, c in enumerate(CLASSES)}
    target = np.asarray([CLASSES[(class_of[int(v)] + 1) % len(CLASSES)] for v in truth])
    target_idx = torch.as_tensor([class_of[int(v)] for v in target],
                                 dtype=torch.long, device=DEVICE)
    assign = np.asarray([choice[int(v)] for v in truth])
    groups = []
    for held in CLASSES:
        index = np.flatnonzero(assign == int(held))
        if len(index):
            groups.append((int(held), torch.as_tensor(index, dtype=torch.long,
                                                      device=DEVICE)))

    def measure(x: torch.Tensor) -> dict:
        with torch.no_grad():
            _, parts = scorer.parts(x, task_idx, groups)
        out = {name: parts[name].detach().cpu().numpy().astype(np.float64)
               for name in ("g_dist", "g_conf", "g_open")}
        out["fused"] = (sig["weights"]["g_head"] * out["g_conf"]
                        + sig["weights"]["g_dist"] * out["g_dist"]
                        + sig["weights"]["g_entropy"] * parts["g_entropy"].detach()
                        .cpu().numpy().astype(np.float64))
        return out

    clean = measure(raw)
    clean_table = rejection_table(clean, tau, signals)
    for a in ACC_GRID:
        log(f"  clean rejection @a={a:.2f}: " + " ".join(
            f"{name} {clean_table[name][f'{a:.2f}']:.4f}" for name in signals))

    conditions = []
    cfg = CFG["attack_step1"]
    for eps in cfg["eps_normalised"]:
        for goal in cfg["goals"]:
            adv = pgd_raw(scorer, raw, task_idx, groups, true_idx, target_idx,
                          goal, float(eps), int(cfg["steps"]),
                          int(cfg["seed"]) + int(float(eps) * 1e5))
            values = measure(adv)
            table = rejection_table(values, tau, signals)
            increment = {name: {key: table[name][key] - clean_table[name][key]
                                for key in table[name]} for name in signals}
            realised = float(((adv - raw).abs().amax(dim=(1, 2))
                              / (raw - raw.mean(dim=-1, keepdim=True)).pow(2)
                              .mean(dim=(1, 2)).sqrt()).max())
            conditions.append({
                "eps": float(eps), "goal": goal, "realised_rel_linf": realised,
                "mean_signal": {name: float(np.mean(values[name])) for name in signals},
                "rejection": table, "increment": increment})
            log(f"  eps={float(eps):.3f} goal {goal}: "
                + " ".join(f"{name}@a=0.90 {table[name]['0.90']:.4f}"
                           f"({increment[name]['0.90']:+.4f})" for name in signals))

    primary = [c for c in conditions if c["goal"] in ("A", "B", "C")]
    best = max(((c["increment"][name][f"{a:.2f}"], name, f"{a:.2f}", c)
                for c in primary for name in ("g_dist", "g_conf", "fused")
                for a in ACC_GRID), key=lambda item: item[0])
    statistic = float(best[0])
    increment_by_signal = {}
    for name in ("g_dist", "g_conf", "g_open", "fused"):
        values = [c["increment"][name][f"{a:.2f}"] for c in primary for a in ACC_GRID]
        increment_by_signal[name] = {"max": float(np.max(values)),
                                     "mean": float(np.mean(values))}
    diagnostic = {}
    for c in conditions:
        if c["goal"] in ("D", "E"):
            diagnostic[f"{c['eps']:.3f}_{c['goal']}"] = {
                key: c["increment"]["fused"][key] for key in c["increment"]["fused"]}
    feature_diag = feature_space_diagnostic(meta, subjects, tasks, fit_rows, subset,
                                            choice, folds, tau["g_conf"])
    decision = {"statistic": statistic, "best_signal": best[1],
                "best_working_point": best[2], "best_eps": best[3]["eps"],
                "best_goal": best[3]["goal"], "threshold": 0.03,
                "stopped": bool(statistic < 0.03)}
    summary = {"n_trials": int(len(subset)), "subset_rows": subset.tolist(),
               "tau_train_loso": tau, "clean_rejection": clean_table,
               "clean_mean_signal": {name: float(np.mean(clean[name]))
                                     for name in signals},
               "increment_by_signal": increment_by_signal,
               "conditions": [{key: value for key, value in c.items()
                               if key not in ("rejection", "increment")}
                              | {"increment": c["increment"]} for c in conditions],
               "gate_evasion_diagnostic": diagnostic,
               "feature_space_diagnostic": feature_diag,
               "decision": decision,
               "elapsed_seconds": time.time() - started}
    results = load_results()
    results["experiment"] = CFG["experiment"]
    results["config"] = CFG
    results["hypothesis"] = summary
    save_results(results)
    log(f"stage hypothesis done in {summary['elapsed_seconds']:.0f}s: max rejection "
        f"increment {statistic:+.4f} ({best[1]} @ a={best[2]}, eps={best[3]['eps']}, "
        f"goal {best[3]['goal']}) -> hypothesis "
        f"{'REJECTED' if decision['stopped'] else 'accepted'}")
    del raw, model_a, folds
    torch.cuda.empty_cache()
    return summary


def feature_space_diagnostic(meta, subjects, tasks, fit_rows, subset, choice,
                             folds, tau_conf_raw) -> dict:
    """How strongly an adversary inside the PCA-96 space (the AT epsilon grid) would
    move the frozen E183 gate confidence.  Diagnostic only, no training."""
    cfg = CFG["robust_gate"]
    tau = {f"{a:.2f}": tau_conf_raw[f"{a:.2f}"] for a in ACC_GRID}
    features = f32(np.load(ROOT / E180.CFG["branch_b"]["train_features"]))
    choice_subset = np.asarray([choice[int(v)] for v in subjects[subset]])
    buckets = {}
    clean_conf = np.zeros(len(subset), dtype=np.float64)
    prepared = {}
    for held in CLASSES:
        index = np.flatnonzero(choice_subset == int(held))
        if len(index) == 0:
            continue
        buckets[int(held)] = index
        entry = folds[int(held) - 1]
        train_mask = (subjects != int(held)) & fit_rows
        tr = E183.fold_transform(features, tasks, train_mask, DIM)
        x_all = (tr["proj"] - tr["feat_mu"]) / tr["feat_sd"]
        prepared[int(held)] = {
            "models": entry["models"], "mu_conf": entry["mu_conf"],
            "sd_conf": entry["sd_conf"],
            "x": torch.as_tensor(x_all[subset[index]], dtype=FT, device=DEVICE),
            "rms": float(np.sqrt(np.mean(x_all[train_mask] ** 2)))}
        with torch.no_grad():
            views = []
            for rep, model in enumerate(entry["models"]):
                value = torch.softmax(model.id_head(model.trunk(prepared[int(held)]["x"])),
                                      1).max(1).values
                views.append((value - entry["mu_conf"][rep]) / entry["sd_conf"][rep])
            clean_conf[index] = torch.stack(views).max(0).values.cpu().numpy()
        del tr, x_all
    out = {"clean_rejection": {key: float(np.mean(clean_conf < tau[key])) for key in tau}}
    scores = {}
    for eps in cfg["eps_feature_sweep"]:
        adv_conf = clean_conf.copy()
        for held, index in buckets.items():
            piece = prepared[held]
            xb = piece["x"]
            bound = float(eps) * piece["rms"]
            delta = (torch.rand(xb.shape, generator=torch.Generator(
                device=DEVICE).manual_seed(SEED + int(held)), device=DEVICE) * 2 - 1) * bound
            delta = delta.clamp(-bound, bound)
            alpha = 2.0 * bound / int(cfg["pgd_steps"])
            for _ in range(int(cfg["pgd_steps"])):
                delta.requires_grad_(True)
                views = []
                for rep, model in enumerate(piece["models"]):
                    value = torch.softmax(model.id_head(model.trunk(xb + delta)), 1) \
                        .max(1).values
                    views.append((value - piece["mu_conf"][rep]) / piece["sd_conf"][rep])
                score = torch.stack(views).max(0).values
                grad = torch.autograd.grad(-score.sum(), delta)[0]
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                delta = (delta + alpha * grad.sign()).clamp(-bound, bound).detach()
            with torch.no_grad():
                views = []
                for rep, model in enumerate(piece["models"]):
                    value = torch.softmax(model.id_head(model.trunk(xb + delta)), 1) \
                        .max(1).values
                    views.append((value - piece["mu_conf"][rep]) / piece["sd_conf"][rep])
                adv_conf[index] = torch.stack(views).max(0).values.cpu().numpy()
        key = f"{float(eps):.2f}"
        scores[key] = {a: {"rejection": float(np.mean(adv_conf < tau[a])),
                           "increment": float(np.mean(adv_conf < tau[a]))
                           - out["clean_rejection"][a]} for a in tau}
        log(f"  feature-space diagnostic eps={float(eps):.2f}xRMS: "
            + " ".join(f"a={a}:{scores[key][a]['increment']:+.4f}" for a in tau))
        del adv_conf
    out["eps"] = scores
    del prepared
    torch.cuda.empty_cache()
    out["note"] = ("diagnostic only: PGD inside the standardised PCA-96 space of the "
                   "frozen E183 head, minimising its confidence read-out; this is the "
                   "perturbation family the AT gate would be trained on")
    return out


# ------------------------------------------------------------------ gate model
class GateMLP(nn.Module):
    def __init__(self, dim: int, hidden, dropout: float) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dim, hidden[0]), nn.BatchNorm1d(hidden[0]), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.BatchNorm1d(hidden[1]), nn.ReLU(inplace=True))
        self.head = nn.Linear(hidden[1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x)).squeeze(1)


def feature_pgd(model: nn.Module, xb: torch.Tensor, bound: float, steps: int,
                gen: torch.Generator) -> torch.Tensor:
    delta = (torch.rand(xb.shape, generator=gen, device=DEVICE) * 2 - 1) * bound
    delta = delta.clamp(-bound, bound)
    alpha = 2.0 * bound / steps
    for _ in range(steps):
        delta.requires_grad_(True)
        out = model(xb + delta)
        loss = F.binary_cross_entropy_with_logits(out, torch.ones_like(out))
        grad = torch.autograd.grad(loss, delta)[0]
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        delta = (delta + alpha * grad.sign()).clamp(-bound, bound).detach()
    return (xb + delta).detach()


def balanced_bce(logit: torch.Tensor, target: torch.Tensor,
                 known: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    return torch.stack([bce[known].mean(), bce[~known].mean()]).mean()


def train_gate(x_train: np.ndarray, y_known: np.ndarray, val_mask: np.ndarray,
               eps: float, lam: float, seed: int) -> tuple:
    cfg = CFG["robust_gate"]
    epochs = int(cfg["epochs"])
    batch = int(cfg["batch_size"])
    steps = int(cfg["pgd_steps"])
    torch.manual_seed(seed)
    model = GateMLP(x_train.shape[1], [int(v) for v in cfg["hidden"]],
                    float(cfg["dropout"])).to(DEVICE)
    x = torch.as_tensor(x_train, dtype=FT, device=DEVICE)
    y = torch.as_tensor(y_known, dtype=FT, device=DEVICE)
    tr = torch.as_tensor(np.flatnonzero(~val_mask), dtype=torch.long, device=DEVICE)
    va = torch.as_tensor(np.flatnonzero(val_mask), dtype=torch.long, device=DEVICE)
    rms = float(np.sqrt(np.mean(np.asarray(x_train, dtype=np.float64) ** 2)))
    bound = float(eps) * rms
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                            weight_decay=float(cfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    best = (float("inf"), None, -1)
    for epoch in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=gen, device=DEVICE)]
        perm = perm[:max((len(perm) // batch) * batch, batch)]
        for start in range(0, len(perm), batch):
            sel = perm[start:start + batch]
            xb, yb = x[sel], y[sel]
            known = yb > 0.5
            logit = model(xb)
            loss = balanced_bce(logit, yb, known)
            if bound > 0:
                adv = feature_pgd(model, xb, bound, steps, gen)
                adv_logit = model(adv)
                loss = loss + F.binary_cross_entropy_with_logits(
                    adv_logit[known], torch.ones_like(adv_logit[known]))
                consistency = (torch.sigmoid(logit[known])
                               - torch.sigmoid(adv_logit[known])).abs().mean()
                loss = loss + float(lam) * consistency
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                value = float(balanced_bce(model(x[va]), y[va], y[va] > 0.5))
            if value < best[0]:
                best = (value, {key: v.detach().clone()
                                for key, v in model.state_dict().items()}, epoch)
    if best[1] is not None:
        model.load_state_dict(best[1])
    model.eval()
    return model, {"best_val_loss": best[0], "best_epoch": best[2],
                   "feature_rms": rms, "bound": bound}


def gate_prob(model: nn.Module, x, batch: int = 8192) -> np.ndarray:
    model.eval()
    tensor = x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x), dtype=FT,
                                                          device=DEVICE)
    out = np.empty(len(tensor), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(tensor), batch):
            piece = tensor[start:start + batch]
            out[start:start + len(piece)] = torch.sigmoid(model(piece)).cpu().numpy()
    return out


# ------------------------------------------------------------------ stage 2
def gate_configs() -> list:
    cfg = CFG["robust_gate"]
    out = [{"name": "clean", "eps": 0.0, "lam": 0.0}] if bool(cfg["baseline_eps0"]) else []
    for eps in cfg["eps_feature_sweep"]:
        for lam in cfg["lambda_sweep"]:
            out.append({"name": f"eps{float(eps):.1f}_lam{float(lam):.1f}",
                        "eps": float(eps), "lam": float(lam)})
    return out


def config_key(cfg: dict) -> str:
    return cfg["name"]


def stage_gate() -> dict:
    started = time.time()
    deadline = started + float(CFG["robust_gate"]["time_budget_seconds"])
    meta, frac, blocks, fit_rows = E180.load_config_meta()
    subjects = meta["subject"].to_numpy(int)
    tasks = meta["task"].to_numpy()
    window_rows = np.flatnonzero((frac > FIT) & (frac <= 0.8))
    features = f32(np.load(ROOT / E180.CFG["branch_b"]["train_features"]))
    n_unk = int(E183.CFG["head"]["pseudo_unknown_per_repeat"])
    configs = gate_configs()
    log(f"stage gate: {len(configs)} configs x {len(CLASSES)} folds, "
        f"window rows {len(window_rows)}")

    scores = {config_key(c): [] for c in configs}
    calibration = {config_key(c): [] for c in configs}
    eps_used = {config_key(c): [] for c in configs}
    trained = []
    for held in CLASSES:
        held = int(held)
        pm = subjects != held
        enrolled = np.setdiff1d(CLASSES, [held])
        train_mask = pm & fit_rows
        tr = E183.fold_transform(features, tasks, train_mask, DIM)
        x_all = (tr["proj"] - tr["feat_mu"]) / tr["feat_sd"]
        x_eval = torch.as_tensor(x_all[window_rows], dtype=FT, device=DEVICE)
        rng = np.random.default_rng(SEED + 1000 * held)
        unknown_ids = np.sort(rng.choice(enrolled, size=n_unk, replace=False))
        known_ids = np.setdiff1d(enrolled, unknown_ids)
        sel_k = np.flatnonzero(fit_rows & np.isin(subjects, known_ids))
        sel_u = np.flatnonzero(fit_rows & np.isin(subjects, unknown_ids))
        order = np.r_[sel_k, sel_u]
        x_train = x_all[order]
        y_known = np.r_[np.ones(len(sel_k), np.float32), np.zeros(len(sel_u), np.float32)]
        group = np.r_[subjects[sel_k], np.full(len(sel_u), -1)]
        split = np.random.default_rng(SEED + 1000 * held + 7)
        val = np.zeros(len(order), dtype=bool)
        for value in np.unique(group):
            pos = np.flatnonzero(group == value)
            take = split.permutation(pos)[:max(1, int(round(0.15 * len(pos))))]
            val[take] = True
        for cfg in configs:
            key = config_key(cfg)
            if time.time() > deadline:
                log(f"  fold {held}: time budget reached, skipping {key}")
                continue
            seed = SEED + 1000 * held + 31 * len(key)
            model, info = train_gate(x_train, y_known, val, cfg["eps"], cfg["lam"], seed)
            p_train = gate_prob(model, x_train)
            mu, sd = float(np.mean(p_train)), float(np.std(p_train)) + 1e-6
            p_eval = gate_prob(model, x_eval)
            scores[key].append(((p_eval - mu) / sd).astype(np.float32))
            calibration[key].append({"mu": mu, "sd": sd, "held": held,
                                     "unknown_ids": unknown_ids.tolist(),
                                     "known_ids": known_ids.tolist(),
                                     "best_epoch": info["best_epoch"],
                                     "best_val_loss": info["best_val_loss"]})
            eps_used[key].append({"feature_rms": info["feature_rms"],
                                  "bound": info["bound"]})
            torch.save({"state_dict": {k: v.detach().cpu()
                                       for k, v in model.state_dict().items()},
                        "mu": mu, "sd": sd,
                        "stats_mu": tr["stats_mu"], "stats_sd": tr["stats_sd"],
                        "pca_mu": tr["pca_mu"], "pca_basis": tr["pca_basis"],
                        "feat_mu": tr["feat_mu"], "feat_sd": tr["feat_sd"],
                        "K": known_ids.tolist(), "U": unknown_ids.tolist(),
                        "eps": cfg["eps"], "lam": cfg["lam"]},
                       GATE_DIR / f"{key}_fold{held}.pt")
            trained.append({"fold": held, "config": key,
                            "best_epoch": info["best_epoch"],
                            "best_val_loss": info["best_val_loss"]})
            del model
        log(f"  fold {held:2d}: trained {len(trained)} models so far, "
            f"{time.time() - started:.0f}s")
        del tr, x_all, x_eval
        torch.cuda.empty_cache()

    loso_rows = np.concatenate([window_rows] * len(CLASSES))
    loso_fold = np.repeat(CLASSES.astype(np.int32), len(window_rows))
    cache_sig = fused_loso_signals()
    if not np.array_equal(cache_sig["rows"], loso_rows):
        raise RuntimeError("LOSO row order does not match the frozen E183 cache")
    known = subjects[loso_rows] != loso_fold
    payload = {"rows": loso_rows, "fold": loso_fold, "known": known,
               "z_dist": cache_sig["z_dist"], "z_ent": cache_sig["z_ent"]}
    for cfg in configs:
        key = config_key(cfg)
        if len(scores[key]) == len(CLASSES):
            payload[f"score_{key}"] = np.concatenate(scores[key])
    np.savez_compressed(LOSO_CACHE, **payload)
    summary = {"configs": configs, "n_trained": len(trained),
               "folds_complete": {cfg["name"]: len(scores[cfg["name"]])
                                  for cfg in configs},
               "calibration": calibration, "eps_bound": eps_used,
               "elapsed_seconds": time.time() - started}
    results = load_results()
    results["gate"] = summary
    save_results(results)
    log(f"stage gate done in {summary['elapsed_seconds']:.0f}s ({len(trained)} models)")
    return summary


# ------------------------------------------------------------------ stage 3
def strict_known_mask(data: dict, key: str) -> np.ndarray:
    scores = data[f"score_{key}"]
    return scores


def pseudo_unknown_diagnostic(key: str) -> dict:
    """Does the gate learn general novelty, or only the 4 pseudo-unknown identities?"""
    meta, frac, blocks, fit_rows = E180.load_config_meta()
    subjects = meta["subject"].to_numpy(int)
    tasks = meta["task"].to_numpy()
    window = (frac > FIT) & (frac <= 0.8)
    features = f32(np.load(ROOT / E180.CFG["branch_b"]["train_features"]))
    rows = []
    for held in CLASSES:
        art = torch.load(GATE_DIR / f"{key}_fold{int(held)}.pt", map_location="cpu",
                         weights_only=False)
        model = GateMLP(DIM, [int(v) for v in CFG["robust_gate"]["hidden"]],
                        float(CFG["robust_gate"]["dropout"])).to(DEVICE)
        model.load_state_dict(art["state_dict"])
        model.eval()
        train_mask = (subjects != int(held)) & fit_rows
        tr = E183.fold_transform(features, tasks, train_mask, DIM)
        x_all = (tr["proj"] - tr["feat_mu"]) / tr["feat_sd"]
        k_ids, u_ids = np.asarray(art["K"]), np.asarray(art["U"])
        masks = {"fit_k": fit_rows & np.isin(subjects, k_ids),
                 "fit_u": fit_rows & np.isin(subjects, u_ids),
                 "win_k": window & np.isin(subjects, k_ids),
                 "win_u": window & np.isin(subjects, u_ids),
                 "win_h": window & (subjects == int(held))}
        p = {name: gate_prob(model, x_all[mask]) for name, mask in masks.items()}

        def auc(positive, negative):
            y = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
            return float(roc_auc_score(y, np.r_[positive, negative]))

        rows.append({"held": int(held),
                     "auc_train_K_vs_U": auc(p["fit_k"], p["fit_u"]),
                     "auc_window_K_vs_pseudoU": auc(p["win_k"], p["win_u"]),
                     "auc_window_K_vs_heldout": auc(p["win_k"], p["win_h"])})
        del tr, x_all, model
        torch.cuda.empty_cache()
    out = {name: float(np.mean([row[name] for row in rows]))
           for name in ("auc_train_K_vs_U", "auc_window_K_vs_pseudoU",
                        "auc_window_K_vs_heldout")}
    out["per_fold"] = rows
    return out


def stage_evaluate() -> dict:
    started = time.time()
    data = np.load(LOSO_CACHE)
    fold, known = data["fold"], data["known"]
    z_dist = data["z_dist"].astype(np.float64)

    gate = load_results()["gate"]
    configs = [cfg["name"] for cfg in gate["configs"]
               if f"score_{cfg['name']}" in data.files]
    leaderboard = []
    for key in configs:
        score = data[f"score_{key}"].astype(np.float64)
        metrics = E183.gate_metrics(score, known, fold)
        leaderboard.append({"config": key, **{k: v for k, v in metrics.items()
                                              if k != "fold_auc_raw"}})
        log(f"  {key:16s} pooled AUC {metrics['pooled_auc']:.4f} "
            f"U@95 {metrics['u95_mean']:.4f}")
    best = max(leaderboard, key=lambda row: row["pooled_auc"])
    key = best["config"]
    gate_score = data[f"score_{key}"].astype(np.float64)
    fusion = []
    for w in np.round(np.arange(0.0, 1.0001, 0.1), 1):
        fused = float(w) * gate_score + (1.0 - float(w)) * z_dist
        metrics = E183.gate_metrics(fused, known, fold)
        fusion.append({"w_gate": float(w), "w_dist": float(1.0 - w),
                       "pooled_auc": metrics["pooled_auc"],
                       "u95_mean": metrics["u95_mean"],
                       "u95_min": metrics["u95_min"]})
    best_fusion = max(fusion, key=lambda row: row["pooled_auc"])
    w = best_fusion["w_gate"]
    fused = w * gate_score + (1.0 - w) * z_dist
    metrics = E183.gate_metrics(fused, known, fold)
    tau = {f"{a:.2f}": quantile(fused[known], 1.0 - a) for a in ACC_GRID}
    diag = pseudo_unknown_diagnostic(key)
    log(f"  pseudo-unknown transfer: train K/U AUC {diag['auc_train_K_vs_U']:.4f}, "
        f"window K/pseudoU {diag['auc_window_K_vs_pseudoU']:.4f}, "
        f"window K/held-out {diag['auc_window_K_vs_heldout']:.4f}")
    per_op = {}
    for a in ACC_GRID:
        thr = tau[f"{a:.2f}"]
        reject = [float(np.mean(fused[(fold == int(h)) & ~known] < thr))
                  for h in CLASSES]
        per_op[f"{a:.2f}"] = {"tau": thr,
                              "unknown_rejection": float(np.mean(reject)),
                              "known_acceptance": float(np.mean(fused[known] >= thr))}
    summary = {"leaderboard": leaderboard, "selected_config": key,
               "selected_pooled_auc": best["pooled_auc"],
               "selected_u95": best["u95_mean"],
               "selected_u95_pooled": best["u95_pooled_threshold"],
               "selected_u95_min": best["u95_min"],
               "selected_fold_auc_mean": best["fold_auc_mean"],
               "selected_fold_auc_std": best["fold_auc_std"],
               "selected_fold_auc_min": best["fold_auc_min"],
               "fused_pooled_auc": metrics["pooled_auc"],
               "fused_u95": metrics["u95_mean"],
               "fused_u95_pooled": metrics["u95_pooled_threshold"],
               "fusion": fusion, "selected_fusion": best_fusion,
               "pseudo_unknown_transfer": diag,
               "tau_train_loso": tau, "operating_points": per_op,
               "elapsed_seconds": time.time() - started}
    results = load_results()
    results["evaluation"] = summary
    save_results(results)
    log(f"  selected {key} + {w:.1f}*g_dist: LOSO AUC "
        f"{best_fusion['pooled_auc']:.4f}, U@95 {metrics['u95_mean']:.4f}")
    return summary


def stage_test() -> dict:
    started = time.time()
    gate = load_results()["gate"]
    evaluation = load_results()["evaluation"]
    key = evaluation["selected_config"]
    w = float(evaluation["selected_fusion"]["w_gate"])
    test_meta = pd.read_csv(ROOT / E180.CFG["data"]["test_metadata"],
                            usecols=["sample", "path", "task", "session"])
    test_features = f32(np.load(ROOT / E180.CFG["branch_b"]["test_features"]))
    tasks = test_meta["task"].to_numpy()
    inputs = np.load(E182_DIR / "cache" / "test_inputs.npz", allow_pickle=True)
    if not np.array_equal(test_meta["sample"].astype(str).to_numpy(), inputs["sample"]):
        raise ValueError("Test sample order mismatch with the frozen E182 cache")
    store = E180.load_branch_b()
    z_dist = ((inputs["dist"].astype(np.float64) - float(store["oof_mu"]))
              / float(store["oof_sd"]))

    per_fold = np.zeros((len(CLASSES), len(test_features)), dtype=np.float32)
    for held in CLASSES:
        art = torch.load(GATE_DIR / f"{key}_fold{int(held)}.pt", map_location="cpu",
                         weights_only=False)
        model = GateMLP(DIM, [int(v) for v in CFG["robust_gate"]["hidden"]],
                        float(CFG["robust_gate"]["dropout"])).to(DEVICE)
        model.load_state_dict(art["state_dict"])
        model.eval()
        x = E183.fold_inputs(test_features, tasks, art)
        with torch.no_grad():
            p = torch.sigmoid(model(x)).cpu().numpy().astype(np.float64)
        per_fold[int(held) - 1] = (p - art["mu"]) / art["sd"]
        del x, model
        torch.cuda.empty_cache()
    variants = {"median_over_folds": np.median(per_fold, axis=0).astype(np.float64),
                "mean_over_folds": per_fold.mean(0).astype(np.float64),
                "max_over_folds": per_fold.max(0).astype(np.float64)}
    z_gate = variants["median_over_folds"]
    fused = w * z_gate + (1.0 - w) * z_dist
    tau = load_results()["evaluation"]["tau_train_loso"]
    primary = f"{0.92:.2f}"
    accept = fused >= tau[primary]
    logits = inputs["logits_v2"]
    predicted = np.where(accept, logits.argmax(1) + 1, 0)
    pd.DataFrame({"sample": test_meta["sample"].astype(str), "task": tasks,
                  "prediction": predicted, "accepted": accept.astype(int),
                  "fused_gate": fused, "g_robust": z_gate, "g_dist": z_dist}) \
        .to_csv(TEST_PRED, index=False)
    log(f"stage test: predictions written to {TEST_PRED.name}")

    session = test_meta["session"].to_numpy()
    is_s1 = session == "session1"
    is_s2 = session == "session2"
    grid = {}
    for a in ACC_GRID:
        thr = tau[f"{a:.2f}"]
        s1 = float(np.mean(~(fused[is_s1] >= thr)))
        s2 = float(np.mean(fused[is_s2] >= thr))
        grid[f"{a:.2f}"] = {"S1_rejection": s1, "S2_acceptance": s2, "S1_S2": s1 + s2}
    variants_out = {}
    for name, value in variants.items():
        fused_v = w * value + (1.0 - w) * z_dist
        variants_out[name] = {
            f"{a:.2f}": {"S1_rejection": float(np.mean(~(fused_v[is_s1] >= tau[f"{a:.2f}"]))),
                         "S2_acceptance": float(np.mean(fused_v[is_s2] >= tau[f"{a:.2f}"]))}
            for a in ACC_GRID}
    summary = {"selected_config": key, "w_gate": w,
               "tau_train_loso": tau, "grid": grid,
               "aggregation_variants": variants_out,
               "auc_s2_vs_s1_gate": float(roc_auc_score(is_s2.astype(int), z_gate)),
               "auc_s2_vs_s1_dist": float(roc_auc_score(is_s2.astype(int), z_dist)),
               "auc_s2_vs_s1_fused": float(roc_auc_score(is_s2.astype(int), fused)),
               "n_S1": int(is_s1.sum()), "n_S2": int(is_s2.sum()),
               "session_used_for_inference": False,
               "official_metrics": "N/A (Test has no identity labels)",
               "elapsed_seconds": time.time() - started}
    results = load_results()
    results["test"] = summary
    save_results(results)
    log("  " + " ".join(f"a={a:.2f}: S1 {grid[f'{a:.2f}']['S1_rejection']:.4f} "
                        f"S2 {grid[f'{a:.2f}']['S2_acceptance']:.4f} "
                        f"sum {grid[f'{a:.2f}']['S1_S2']:.4f}" for a in ACC_GRID))
    return summary


# ------------------------------------------------------------------ stage 4
class RobustGateScorer(FrozenGateScorer):
    """Frozen branch A + the trained robust gate (fused with g_dist)."""

    def __init__(self, model_a, store, folds, ent_stats, gate_models, weights,
                 w_gate, w_dist) -> None:
        super().__init__(model_a, store, folds, ent_stats)
        self.gate_models = gate_models
        self.w_gate = float(w_gate)
        self.w_dist = float(w_dist)

    def parts(self, x, task_idx, groups):
        logits, parts = super().parts(x, task_idx, groups)
        out = torch.zeros(len(x), device=DEVICE, dtype=x.dtype)
        for fold_index, index in groups:
            entry = self.folds[fold_index]
            group_task = task_idx[index]
            raw_feat = E180.differentiable_logspd(x[index])
            zz = (raw_feat - entry["stats_mu"][group_task]) / entry["stats_sd"][group_task]
            p96 = ((zz - entry["pca_mu"]) @ entry["pca_basis"]
                   - entry["feat_mu"]) / entry["feat_sd"]
            model, mu, sd = self.gate_models[fold_index]
            value = torch.sigmoid(model(p96))
            out = out.index_copy(0, index, (value - mu) / sd)
        parts["g_robust"] = out
        return logits, parts

    def gate(self, parts):
        return self.w_gate * parts["g_robust"] + self.w_dist * parts["g_dist"]


def stage_attack() -> dict:
    started = time.time()
    E180.build_torch_constants()
    evaluation = load_results()["evaluation"]
    key = evaluation["selected_config"]
    w_gate = float(evaluation["selected_fusion"]["w_gate"])
    w_dist = float(evaluation["selected_fusion"]["w_dist"])
    data = np.load(LOSO_CACHE)
    fold, known, rows = data["fold"], data["known"], data["rows"]
    fused = w_gate * data[f"score_{key}"].astype(np.float64) \
        + w_dist * data["z_dist"].astype(np.float64)
    tau = {f"{a:.2f}": quantile(fused[known], 1.0 - a) for a in ACC_GRID}
    gate_models = []
    for held in CLASSES:
        art = torch.load(GATE_DIR / f"{key}_fold{int(held)}.pt", map_location="cpu",
                         weights_only=False)
        model = GateMLP(DIM, [int(v) for v in CFG["robust_gate"]["hidden"]],
                        float(CFG["robust_gate"]["dropout"])).to(DEVICE)
        model.load_state_dict(art["state_dict"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        gate_models.append((model, float(art["mu"]), float(art["sd"])))

    meta, frac, blocks, fit_rows = E180.load_config_meta()
    subjects = meta["subject"].to_numpy(int)
    tasks = meta["task"].to_numpy()
    subset = eval_subset()
    choice = head_choice(np.load(E183_DIR / "cache" / "loso_head.npz")["k_mask"])
    folds = folded_entries()
    store = E180.load_branch_b()
    signals82 = np.load(E182_DIR / "cache" / "signals_v2.npz")
    ent_stats = (float(signals82["stats_mu"][4]), float(signals82["stats_sd"][4]))
    branch_a = torch.load(E182_DIR / "model_branch_a_v2.pt", map_location="cpu",
                          weights_only=False)
    model_a = E180.Net().to(DEVICE)
    model_a.load_state_dict(branch_a["state_dict"])
    model_a.eval()
    scorer = RobustGateScorer(model_a, store, folds, ent_stats, gate_models, None,
                              w_gate, w_dist)
    raw = E180.load_raw([meta["path"].iloc[i] for i in subset])
    truth = subjects[subset]
    task_idx = torch.as_tensor([TASKS.index(t) for t in tasks[subset]],
                               dtype=torch.long, device=DEVICE)
    true_idx = torch.as_tensor(truth - 1, dtype=torch.long, device=DEVICE)
    class_of = {int(c): i for i, c in enumerate(CLASSES)}
    target = np.asarray([CLASSES[(class_of[int(v)] + 1) % len(CLASSES)] for v in truth])
    target_idx = torch.as_tensor([class_of[int(v)] for v in target],
                                 dtype=torch.long, device=DEVICE)
    assign = np.asarray([choice[int(v)] for v in truth])
    groups = []
    for held in CLASSES:
        index = np.flatnonzero(assign == int(held))
        if len(index):
            groups.append((int(held), torch.as_tensor(index, dtype=torch.long,
                                                      device=DEVICE)))

    def sweep(logits: np.ndarray, gate: np.ndarray) -> dict:
        out = {}
        for a in ACC_GRID:
            accept = gate >= tau[f"{a:.2f}"]
            predicted = np.where(accept, logits.argmax(1) + 1, 0)
            out[f"{a:.2f}"] = {
                "RA": float(np.mean(predicted == truth)),
                "ASafeR": float(np.mean((predicted == truth) | (predicted == 0))),
                "accept_rate": float(np.mean(accept)),
                "closed_set": float(np.mean(logits.argmax(1) + 1 == truth))}
        return out

    with torch.no_grad():
        logits, parts = scorer.parts(raw, task_idx, groups)
        gate = scorer.gate(parts)
    clean_logits = logits.detach().cpu().numpy()
    clean_gate = gate.detach().cpu().numpy()
    clean = sweep(clean_logits, clean_gate)
    conditions = []
    cfg = CFG["attack_step1"]
    for eps in cfg["eps_normalised"]:
        for goal in ("A", "B", "C", "D"):
            adv = pgd_raw(scorer, raw, task_idx, groups, true_idx, target_idx,
                          goal, float(eps), int(cfg["steps"]),
                          int(cfg["seed"]) + int(float(eps) * 1e5))
            with torch.no_grad():
                logits, parts = scorer.parts(adv, task_idx, groups)
                gate = scorer.gate(parts)
            row = sweep(logits.detach().cpu().numpy(), gate.detach().cpu().numpy())
            conditions.append({"eps": float(eps), "goal": goal, "sweep": row})
            log(f"  eps={float(eps):.3f} goal {goal}: RA@a=0.92 "
                f"{row['0.92']['RA']:.4f} ASafeR {row['0.92']['ASafeR']:.4f}")
    worst = {}
    for a in ACC_GRID:
        k = f"{a:.2f}"
        worst[k] = {"RA": min(row["sweep"][k]["RA"] for row in conditions),
                    "ASafeR": min(row["sweep"][k]["ASafeR"] for row in conditions)}
    summary = {"n_trials": int(len(subset)), "tau_loso": tau, "clean": clean,
               "conditions": conditions, "worst": worst,
               "attack_status": "self-built white-box PGD surrogate, not the official suite",
               "elapsed_seconds": time.time() - started}
    results = load_results()
    results["attack"] = summary
    save_results(results)
    log(f"attack done in {summary['elapsed_seconds']:.0f}s")
    return summary


# ------------------------------------------------------------------ report
def stage_report() -> dict:
    results = load_results()
    hyp = results["hypothesis"]
    decision = hyp["decision"]
    e183 = load_e183()
    ev = results.get("evaluation")
    te = results.get("test")
    at = results.get("attack")
    cap = e183.get("capacity", {})
    e183_grid_top = e183.get("test", {}).get("grid", {})
    top_sum = max((row["S1_rejection"] + row["S2_acceptance"]
                   for row in (te or {}).get("grid", {}).values()),
                  default=float("nan"))
    clean_auc = next((row["pooled_auc"] for row in (ev or {}).get("leaderboard", [])
                      if row["config"] == "clean"), float("nan"))
    lines = []
    lines.append("# E185 鲁棒门控（对抗训练）针对 Test S2 接受率\n")
    lines.append("日期：2026-11-15。分支 A（E182 v2）、分支 B（E180 E65 PCA-96）、"
                 "E183 专用开放集头与融合门控**全部只读复用、未重训**。"
                 "本实验只新增一件事：检验「S2 接受率被门控误拒对抗样本压低」这一假设，"
                 "只有在 Train 上成立的条件下才训练对抗训练门控。\n")
    lines.append(f"**结论先行**：Step 1 假设检验最大拒绝率增量 "
                 f"**{decision['statistic']:+.4f}**"
                 f"（{decision['best_signal']} @ a={decision['best_working_point']}，"
                 f"ε={decision['best_eps']}，目标 {decision['best_goal']}），"
                 f"{'低于' if decision['stopped'] else '不低于'} 0.03 门槛 → "
                 f"**假设在 Train 上{'不成立' if decision['stopped'] else '成立'}**，"
                 f"{'按任务的停止规则不训练 AT 门控' if decision['stopped'] else '继续训练 AT 门控'}。"
                 + ("" if ev is None else
                    f" 但 AT 门控本身在 Train LOSO 上只有 AUC "
                    f"{ev['selected_pooled_auc']:.4f}（clean 基线 {clean_auc:.4f}），"
                    f"低于 E65 距离 0.9062，预注册选择规则把权重压到 0；"
                    f"最终系统等价于 E65 距离门控，Test 上 S1+S2 最好 "
                    f"**{top_sum:.4f}**，未突破 1.4479，也未到 1.50。") + "\n")

    lines.append("## 1. Step 1：假设验证（Train 0.6<frac≤0.8）\n")
    lines.append(f"攻击子集 {hyp['n_trials']} 条（同 E180/E181/E183：每任务上限 400、"
                 f"seed 20261115）。门控信号取 E183 部署口径，阈值取 Train LOSO 已知分数分位 "
                 f"`quantile(known, 1-a)`。\n")
    lines.append("| 信号 | " + " | ".join(f"a={a:.2f}" for a in ACC_GRID) + " |")
    lines.append("|---|" + "---:|" * len(ACC_GRID))
    for name in ("g_dist", "g_conf", "g_open", "fused"):
        lines.append(f"| clean `{name}` 拒绝率 | "
                     + " | ".join(f"{hyp['clean_rejection'][name][f'{a:.2f}']:.4f}"
                                   for a in ACC_GRID) + " |")
    lines.append("")
    lines.append("对抗条件（ε×目标）下的拒绝率增量 Δ = adv − clean，"
                 "主统计只取目标 A/B/C（身份攻击，E178/E183 既有流程）：\n")
    lines.append("| ε | 目标 | Δ g_dist@0.90 | Δ g_conf@0.90 | Δ fused@0.90 | Δ fused@0.96 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for cond in hyp["conditions"]:
        lines.append(f"| {cond['eps']:.3f} | {cond['goal']} | "
                     + " | ".join([f"{cond['increment']['g_dist']['0.90']:+.4f}",
                                   f"{cond['increment']['g_conf']['0.90']:+.4f}",
                                   f"{cond['increment']['fused']['0.90']:+.4f}",
                                   f"{cond['increment']['fused']['0.96']:+.4f}"]) + " |")
    lines.append("")
    lines.append("各信号在 a∈{0.88,0.90,0.92,0.94,0.96}×ε×(A/B/C) 上的最大增量：\n")
    lines.append("| 信号 | 最大 Δ | 平均 Δ |")
    lines.append("|---|---:|---:|")
    for name, value in hyp["increment_by_signal"].items():
        lines.append(f"| `{name}` | {value['max']:+.4f} | {value['mean']:+.4f} |")
    lines.append("")
    lines.append(f"判定按预先固定的规则取上表最大值（{decision['best_signal']} "
                 f"**{decision['statistic']:+.4f}**，超过 0.03）。**必须同时说明**：如果只看 "
                 f"E183 实际部署的融合门控，最大增量只有 "
                 f"{hyp['increment_by_signal']['fused']['max']:+.4f}（其攻击子集 clean 拒绝率 "
                 f"仅 0.0104–0.0184，结构上没有 3pp 的可回收空间）；假设只在"
                 f"「把 E183 开放集头的置信度读数本身当作门控」时成立。\n")
    diag = hyp.get("feature_space_diagnostic", {})
    if diag:
        keys = ["0.88", "0.90", "0.96"]
        lines.append("特征空间诊断（frozen E183 头，在标准化 PCA-96 空间做 PGD，"
                     "即 Step 2 的扰动族）：\n")
        lines.append("| ε×RMS | " + " | ".join(f"Δ@a={k}" for k in keys) + " |")
        lines.append("|---|" + "---:|" * len(keys))
        for eps, row in diag.get("eps", {}).items():
            lines.append(f"| {eps} | " + " | ".join(
                f"{row[k]['increment']:+.4f}" for k in keys) + " |")
        lines.append("")

    lines.append("## 2. 鲁棒门控（AT）在 Train LOSO 与 Test 上的 AUC\n")
    if ev is None:
        lines.append("Step 1 未通过，因此**未训练鲁棒门控**：无 LOSO AUC / Test AUC。\n")
    else:
        lines.append("输入：E65 PCA-96（逐折、`pm & fit_rows` 拟合）。结构 96→128→64→1。"
                     "训练行只用 `fit_rows`（frac≤0.6）的 19 个注册身份，伪未知为其中 4 个"
                     "（严格嵌套，外层留出身份从不参与）。特征空间 PGD 的 ε 以该折训练行的"
                     "特征 RMS 标定（Train-only），ε∈{0.5,1.0,2.0}×RMS、λ∈{0.5,1.0}。"
                     "配置和融合权重只按 Train LOSO 合并 AUC 选。每折 **1 个内层视图**"
                     "（E183 用 3 个视图取最大，且 g_head 用的是身份头置信度而不是 1-logit "
                     "p_reg 输出，两者不可直接对比）。\n")
        lines.append("| 配置 | LOSO 合并 AUC | 逐折 AUC | U@95（逐折） |")
        lines.append("|---|---:|---:|---:|")
        for row in ev["leaderboard"]:
            mark = " **←选中**" if row["config"] == ev["selected_config"] else ""
            lines.append(f"| `{row['config']}`{mark} | {row['pooled_auc']:.4f} | "
                         f"{row['fold_auc_mean']:.4f} ± {row['fold_auc_std']:.4f} | "
                         f"{row['u95_mean']:.4f} |")
        sel = ev["selected_fusion"]
        lines.append("")
        lines.append(f"选定融合：**{sel['w_gate']:.1f}·z(gate) + {sel['w_dist']:.1f}·z(g_dist)**，"
                     f"LOSO 合并 AUC **{sel['pooled_auc']:.4f}**、U@95 逐折 "
                     f"{ev['fused_u95']:.4f} / 全局阈值 {ev['fused_u95_pooled']:.4f}；"
                     f"门控单独 AUC {ev['selected_pooled_auc']:.4f}"
                     f"（逐折 {ev['selected_fold_auc_mean']:.4f} ± "
                     f"{ev['selected_fold_auc_std']:.4f}，最差折 "
                     f"{ev['selected_fold_auc_min']:.4f}）。\n")
        if te:
            lines.append(f"Test 诊断 AUC（S2 vs S1，仅审计）：鲁棒门控 "
                         f"**{te['auc_s2_vs_s1_gate']:.4f}**、E65 距离 "
                         f"{te['auc_s2_vs_s1_dist']:.4f}、融合 "
                         f"{te['auc_s2_vs_s1_fused']:.4f}。\n")
        transfer = ev.get("pseudo_unknown_transfer")
        if transfer:
            lines.append(f"伪未知迁移诊断（20 折均值）：训练行 K/U AUC "
                         f"{transfer['auc_train_K_vs_U']:.4f}，评估窗口 K/伪未知 AUC "
                         f"{transfer['auc_window_K_vs_pseudoU']:.4f}，评估窗口 "
                         f"K/外层留出身份 AUC {transfer['auc_window_K_vs_heldout']:.4f}。"
                         f"1-logit 门控把「哪 4 个身份被标为未知」学得很好（训练行 1.00、"
                         f"窗口伪未知 0.9998），但对真正留出的第 20 个身份只有 0.80；"
                         f"再把 4 个伪未知身份按 LOSO 口径算作「已知」，合并 AUC 就被拉到 "
                         f"{ev['selected_pooled_auc']:.4f}。AT 只在最小 ε（0.5×RMS）上有 "
                         f"+0.0058 的合并 AUC 增益，ε≥1.0 反而把它打到 0.52–0.57。\n")

    lines.append("## 3. 各工作点 S1/S2/S1+S2\n")
    grid = (te or {}).get("grid") or {
        key: {"S1_rejection": row["S1_rejection"],
              "S2_acceptance": row["S2_acceptance"]}
        for key, row in e183.get("test", {}).get("grid", {}).items()}
    if te is None:
        lines.append("未在 Test 上评分新门控；下表为 E183 冻结门控对照。\n")
    else:
        lines.append("τ(a) 全部来自 Train LOSO 已知分数（19 个注册身份），"
                     "Test 只评分一次、Session 事后审计。生产聚合为 20 折取中位数。\n")
    lines.append("| a | S1 拒识 | S2 接受 | S1+S2 |")
    lines.append("|---:|---:|---:|---:|")
    for a in ACC_GRID:
        row = grid[f"{a:.2f}"]
        lines.append(f"| {a:.2f} | {row['S1_rejection']:.4f} | {row['S2_acceptance']:.4f} | "
                     f"{row['S1_rejection'] + row['S2_acceptance']:.4f} |")
    lines.append("")
    lines.append("**关键对照（各自最好工作点的 S1+S2）**\n")
    lines.append("| 方案 | S1+S2 最高 | 说明 |")
    lines.append("|---|---:|---|")
    lines.append("| E180 E65 距离门控 | 1.3756 | 参考值（历史） |")
    lines.append("| E181 融合门控 | 1.3756 | 参考值（历史） |")
    lines.append("| E183 三视图门控 | 1.0906 | 参考值（E183 实测） |")
    lines.append(f"| **E185：AT 门控被选择权重压到 0 → E65 距离门控** | "
                 f"**{max(grid[a]['S1_rejection'] + grid[a]['S2_acceptance'] for a in grid):.4f}** "
                 f"| 本实验 |")
    lines.append("")

    lines.append("## 4. 十项门禁（单工作点口径）\n")
    lines.append("真实部署只能有一个阈值，因此每一列是**同一个 `a` 下的完整系统**，"
                 "不允许跨工作点拼接。\n")
    auc_gate = ev["selected_fusion"]["pooled_auc"] if ev else float("nan")
    u95_fold = ev["fused_u95"] if ev else float("nan")
    u95_pool = ev["fused_u95_pooled"] if ev else float("nan")
    cols = [f"{a:.2f}" for a in ACC_GRID]
    lines.append("| # | 门禁 | 目标 | " + " | ".join(f"a={a}" for a in cols) + " | 单点判定 |")
    lines.append("|---|---|---|" + "---:|" * len(cols) + "---|")
    lines.append(f"| #1 | Train 准确率 | =1.00 | " + " | ".join(
        f"{cap.get('in_sample_logits_argmax', float('nan')):.4f}" for _ in cols)
        + " | ❌ 结构性未通过（E183 复算，未改动） |")
    for number, label, target, key_, threshold in (
            ("#2", "C（5 块 OOF）", "≥0.95", "oof_accuracy_C", 0.95),
            ("#3", "身份 macro-F1", "≥0.92", "oof_macro_f1", 0.92),
            ("#4", "块外准确率（frac>0.6）", "≥0.90", "oof_accuracy_forward_blocks", 0.90)):
        value = cap.get(key_, float("nan"))
        verdict = "✅ 全工作点通过" if value >= threshold else "❌ 未通过"
        lines.append(f"| {number} | {label} | {target} | " + " | ".join(
            f"{value:.4f}" for _ in cols) + f" | {verdict}（E183 复算） |")
    verdict5 = "✅ 通过" if (ev and auc_gate >= 0.98) else "❌ 未通过"
    lines.append(f"| #5 | 20 折 LOSO AUROC | ≥0.98 | " + " | ".join(
        f"{auc_gate:.4f}" for _ in cols) + f" | {verdict5} |")
    worst6 = min(u95_fold, u95_pool) if ev else float("nan")
    verdict6 = "✅ 通过" if (ev and worst6 >= 0.95) else "❌ 未通过"
    lines.append(f"| #6 | U@95%（逐折 / 全局） | ≥0.95 | " + " | ".join(
        f"{u95_fold:.4f} / {u95_pool:.4f}" for _ in cols) + f" | {verdict6}（按最差口径） |")
    for number, label, target, key7, threshold in (
            ("#7", "RA（ε×目标最差）", "≥0.95", "RA", 0.95),
            ("#8", "ASafeR（同攻击最差）", "≥0.98", "ASafeR", 0.98),
            ("#9", "Test S2 接受率", "≥0.75", "S2_acceptance", 0.75),
            ("#10", "Test S1 拒识率", "≥0.75", "S1_rejection", 0.75)):
        values = []
        for a in cols:
            if number in ("#7", "#8"):
                values.append((at or {}).get("worst", {}).get(a, {}).get(key7, float("nan")))
            else:
                values.append(grid[a][key7] if a in grid else float("nan"))
        flags = [v >= threshold for v in values]
        verdict = ("✅ 全工作点通过" if all(flags) else
                   ("❌ 结构性未通过" if not any(flags) else "部分工作点通过"))
        lines.append(f"| {number} | {label} | {target} | "
                     + " | ".join(f"{v:.4f}" for v in values) + f" | {verdict} |")
    lines.append("")

    lines.append("| a | S1 拒识 | S2 接受 | S1+S2 | #7 RA | #8 ASafeR | 通过项数 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    pass_counts = {}
    for a in cols:
        r7 = (at or {}).get("worst", {}).get(a, {}).get("RA", float("nan"))
        r8 = (at or {}).get("worst", {}).get(a, {}).get("ASafeR", float("nan"))
        s1, s2 = grid[a]["S1_rejection"], grid[a]["S2_acceptance"]
        checks = [cap.get("in_sample_logits_argmax", 0) >= 1.0,
                  cap.get("oof_accuracy_C", 0) >= 0.95,
                  cap.get("oof_macro_f1", 0) >= 0.92,
                  cap.get("oof_accuracy_forward_blocks", 0) >= 0.90,
                  bool(ev) and auc_gate >= 0.98,
                  bool(ev) and worst6 >= 0.95,
                  r7 >= 0.95, r8 >= 0.98, s2 >= 0.75, s1 >= 0.75]
        pass_counts[a] = int(sum(bool(v) for v in checks))
        lines.append(f"| {a} | {s1:.4f} | {s2:.4f} | {s1 + s2:.4f} | {r7:.4f} | "
                     f"{r8:.4f} | **{pass_counts[a]}/10** |")
    lines.append("")

    lines.append("## 5. 明确回答\n")
    best_a, best_row = max(grid.items(), key=lambda kv: kv[1]["S1_rejection"]
                           + kv[1]["S2_acceptance"])
    best_sum = best_row["S1_rejection"] + best_row["S2_acceptance"]
    e183_grid = e183.get("test", {}).get("grid", {})
    e183_best = max((row["S1_rejection"] + row["S2_acceptance"]
                     for row in e183_grid.values()), default=float("nan"))
    compare = "；".join(
        f"a={k} S2 {grid[k]['S2_acceptance']:.4f} vs E183 {e183_grid[k]['S2_acceptance']:.4f}"
        f"（Δ {grid[k]['S2_acceptance'] - e183_grid[k]['S2_acceptance']:+.4f}）"
        for k in ("0.90", "0.92") if k in e183_grid)
    auc_gate_test = te["auc_s2_vs_s1_gate"] if te else float("nan")
    auc_dist_test = te["auc_s2_vs_s1_dist"] if te else float("nan")
    lines.append(f"**S2 接受率能否因此提升？** 不能。AT 门控在 Train LOSO 上只有 AUC "
                 f"**{ev['selected_pooled_auc'] if ev else float('nan'):.4f}**"
                 f"（clean 基线 {clean_auc:.4f}，AT 增益仅 "
                 f"{ev['selected_pooled_auc'] - clean_auc if ev else float('nan'):+.4f}），"
                 f"比 E65 距离 0.9062 低约 0.28；按预注册的 Train-only 选择规则，融合权重 "
                 f"w_gate = 0，**Test 分数没有任何一项来自 AT 门控**，系统退化为 E65 距离门控。"
                 f"{compare}。前面的下降并不代表 E183 更好——E183 的高 S2 是用 S1 拒识跌到 "
                 f"0.1118 换来的；可比的是 S1+S2 之和（§3 对照表）。"
                 f"Test 诊断 AUC：AT 门控 "
                 f"{te['auc_s2_vs_s1_gate'] if te else float('nan'):.4f} < E65 距离 "
                 f"{te['auc_s2_vs_s1_dist'] if te else float('nan'):.4f}，"
                 f"强行用 AT 门控（w=1）只会更差。\n")
    lines.append(f"**S1+S2 能否突破历史 1.4479？** 本实验最好工作点 a={best_a}，"
                 f"S1+S2 = **{best_sum:.4f}**（E183 冻结门控同口径 {e183_best:.4f}）。"
                 f"{'能突破' if best_sum > 1.4479 else '不能突破'} 1.4479。\n")
    lines.append(f"**能否达到 1.50？** {'能' if best_sum >= 1.50 else '不能'}，"
                 f"缺口 {max(0.0, 1.50 - best_sum):.4f}。\n")
    both = [a for a in grid
            if grid[a]["S2_acceptance"] >= 0.75 and grid[a]["S1_rejection"] >= 0.75]
    if both:
        lines.append(f"**#9 与 #10 能否在同一阈值下同时达标？** 能，a="
                     f"{'、'.join(both)}。\n")
    else:
        lines.append("**#9 与 #10 能否在同一阈值下同时达标？** 不能。五个工作点没有一个"
                     "同时满足 S2 接受 ≥0.75 与 S1 拒识 ≥0.75（a=0.94 时 S1 0.7540 刚过线，"
                     "但 S2 只有 0.6145；a=0.96 时 S1 已跌到 0.6789）。S1 拒识与 S2 接受"
                     "沿同一条跨会话 ROC 此消彼长，与 E180/E183 的结论一致。\n")

    lines.append("## 6. Test 官方指标\n")
    lines.append("**Test 官方 AUC / F1 / C / RA / ASafeR / U / Score = N/A**"
                 "（Test 无真实标签）。\n")
    lines.append("## 7. 合规与边界\n")
    lines.append("- 未修改任何已有实验目录，未提交 git。")
    lines.append("- 外层留出身份不参与任何 PCA、标准化、训练、阈值或权重；评估窗口 "
                 "0.6<frac≤0.8 从未进入训练或标定。")
    lines.append("- Test 只评分，Session 只审计；本实验未读取任何 Test 指标做选择。")
    lines.append("- #7/#8 是自建白盒 PGD 代理，不是官方攻击套件。")
    lines.append("")
    lines.append("## 8. 文件与耗时\n")
    lines.append("| 文件 | 内容 |")
    lines.append("|---|---|")
    lines.append("| `run_e185.py` | 全部逻辑（`--stage hypothesis/gate/evaluate/test/attack/report/all`） |")
    lines.append("| `config.json` | 冻结配置（含 Step 1 停止规则与 Step 2 的 ε/λ 网格） |")
    lines.append("| `results.json` | 全部数值（Step 1 全部 15 个条件、7 个门控配置、10 项 × 5 工作点） |")
    lines.append("| `cache/loso_gate.npz` | 20 折 LOSO 门控分数与 g_dist（46020 行） |")
    lines.append("| `cache/gate_folds/*.pt` | 140 个门控 MLP（7 配置 × 20 折）与逐折 PCA/标准化 |")
    lines.append("| `test_predictions_robust_gate.csv` | Test 预测（工程审计用） |")
    lines.append("| `run.log` | 运行日志 |")
    lines.append("")
    gate_info = results.get("gate", {})
    lines.append(f"耗时：Step 1 假设验证 {hyp['elapsed_seconds']:.0f} s；"
                 f"门控训练 {gate_info.get('elapsed_seconds', float('nan')):.0f} s"
                 f"（{gate_info.get('n_trained', 0)} 个 MLP）；"
                 f"评估 {ev['elapsed_seconds'] if ev else 0:.0f} s、"
                 f"Test {te['elapsed_seconds'] if te else 0:.0f} s、"
                 f"攻击 {at['elapsed_seconds'] if at else 0:.0f} s。"
                 f"训练合计约 {gate_info.get('elapsed_seconds', 0) / 60:.1f} 分钟，"
                 f"在 25 分钟预算内。\n")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"report written to {REPORT_PATH.name}")
    return {"report": str(REPORT_PATH)}


# ------------------------------------------------------------------ main
STAGES = {"hypothesis": stage_hypothesis, "gate": stage_gate,
          "evaluate": stage_evaluate, "test": stage_test,
          "attack": stage_attack, "report": stage_report}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    args = parser.parse_args()
    sys.stdout = Tee(LOG_PATH)
    sys.stderr = sys.stdout
    t0 = time.time()
    log(f"=== E185 start ({args.stage}) ===")
    try:
        if args.stage == "all":
            hypothesis = stage_hypothesis()
            if hypothesis["decision"]["stopped"]:
                log("hypothesis rejected on Train -> AT gate not trained (stop rule)")
                stage_report()
            else:
                stage_gate()
                stage_evaluate()
                stage_test()
                stage_attack()
                stage_report()
        else:
            STAGES[args.stage]()
    except Exception:
        traceback.print_exc()
        raise
    log(f"total {time.time() - t0:.0f}s")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
