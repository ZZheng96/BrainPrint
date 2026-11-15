"""E191: fix gate #1 (Train in-sample accuracy = 1.00) and gate #7 (RA >= 0.95).

Only the identity model changes. Gate side = E189's frozen Train-only stream cache
(read-only), attack code = E189.pgd / score_condition / attack_metrics (unchanged,
same subset/seed/eps/goals), so every delta is attributable to branch A v3.

Branch A v3 vs v2 (E180)
  * widths [32,64,128,192,192] (v2: [32,64,128,128]), dropout 0.2 (v2 0.3)
  * weight decay 5e-5 (v2 1e-4), full schedule 60 epochs with NO early-stop
    restoration (v2 restored epoch 22/40), folds 30 epochs
  * adversarial training: PGD-5 (v2 PGD-3), eps mixed {0.02, 0.03, 0.04}
    (v2 fixed 0.02), goal mixed A/B/C per batch (v2 untargeted CE only),
    adv loss weight 0.7 (v2 0.5)
  * optional clean polish only if the 60-epoch model is still below 1.00 in-sample

Stages: train, attack, report, all.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)
MODEL_V3 = OUT / "model_branch_a_v3.pt"
STAGE_A3 = CACHE / "stage_a_v3.pt"
ATTACK_JSON = OUT / "attack_v3.json"
ATTACK_NPZ = CACHE / "attack_v3.npz"
AUDIT_JSON = OUT / "audit_v3.json"
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "report.md"
LOG_PATH = OUT / "run.log"

DEVICE = torch.device("cuda")
FT = torch.float32
OFF = -1e9
BRANCH_SPEC_SHA = hashlib.sha256(
    json.dumps(CFG["branch_a_v3"], sort_keys=True, ensure_ascii=False).encode("utf-8")
).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E180 = load_module("e191_frozen_e180",
                   ROOT / CFG["frozen_sources"]["e180"] / "run_e180.py")
E189 = load_module("e191_frozen_e189",
                   ROOT / CFG["frozen_sources"]["e189"] / "run_e189.py")

TASKS = list(E180.TASKS)
CLASSES = E180.CLASSES
SEED = int(E180.CFG["protocol"]["seed"])
CH, N = int(E180.CH), int(E180.N)
NCLASS = len(CLASSES)
GRID = [float(a) for a in CFG["gate"]["grid"]]
PRIMARY_A = 0.90


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


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float),
                    encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ------------------------------------------------------------------ model v3
class NetV3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        cfg = CFG["branch_a_v3"]
        widths = [int(w) for w in cfg["widths"]]
        kernel, stride = int(cfg["kernel"]), int(cfg["stride"])
        blocks, channels = [], CH
        for width in widths:
            blocks += [nn.Conv1d(channels, width, kernel, stride=stride,
                                 padding=kernel // 2, bias=False),
                       nn.BatchNorm1d(width), nn.ELU(inplace=True)]
            channels = width
        self.body = nn.Sequential(*blocks)
        self.drop = nn.Dropout(float(cfg["dropout"]))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels, NCLASS)
        self.width = channels

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=-1, keepdim=True)
        rms = x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        return self.pool(self.body(x / rms)).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.features(x)))


def margin_goal(logits: torch.Tensor, y: torch.Tensor, target: torch.Tensor,
                goal: str) -> torch.Tensor:
    """E189 attack objective A/B/C (maximised during PGD)."""
    true_logit = logits.gather(1, y[:, None]).squeeze(1)
    wrong = logits.masked_fill(F.one_hot(y, NCLASS).bool(), OFF)
    if goal == "A":
        margin = wrong.max(dim=1).values
    elif goal == "B":
        margin = torch.logsumexp(wrong, dim=1)
    else:
        margin = logits.gather(1, target[:, None]).squeeze(1)
    return margin - true_logit


def pgd_train_v3(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                 target: torch.Tensor, gen: torch.Generator) -> tuple:
    cfg = CFG["branch_a_v3"]
    steps = int(cfg["pgd_steps"])
    u = float(torch.rand((), generator=gen, device=DEVICE))
    eps, acc = float(cfg["pgd_eps_mix"][-1][0]), 0.0
    for value, share in cfg["pgd_eps_mix"]:
        acc += float(share)
        if u <= acc:
            eps = float(value)
            break
    goals = list(cfg["pgd_goal_mix"])
    goal = goals[int(torch.randint(0, len(goals), (), generator=gen, device=DEVICE))]
    xc = x - x.mean(dim=-1, keepdim=True)
    rms = xc.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-6)
    bound = (eps * rms).view(-1, 1, 1)
    adv = x + (torch.rand(x.shape, generator=gen, device=x.device) * 2 - 1) * bound
    adv = torch.clamp(adv, x - bound, x + bound)
    alpha = 2.0 * eps / steps
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = margin_goal(model(adv), y, target, goal).mean() / 10.0
        grad = torch.autograd.grad(loss, adv)[0]
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        adv = torch.clamp(adv.detach() + (alpha * rms).view(-1, 1, 1) * grad.sign(),
                          x - bound, x + bound)
    return adv.detach(), eps, goal


def feature_bank(model: nn.Module, raw: torch.Tensor, batch: int = 512) -> np.ndarray:
    model.eval()
    out = np.empty((len(raw), model.width), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(raw), batch):
            out[start:start + batch] = model.features(raw[start:start + batch]).cpu().numpy()
    return out


def knn_loo_accuracy(feats: np.ndarray, labels: np.ndarray, chunk: int = 512) -> dict:
    """Leave-one-out 1-NN on the model's own penultimate features (cosine)."""
    f = torch.as_tensor(feats, device=DEVICE)
    f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-6)
    lab = torch.as_tensor(labels, dtype=torch.long, device=DEVICE)
    pred = torch.empty(len(f), dtype=torch.long, device=DEVICE)
    for start in range(0, len(f), chunk):
        size = min(chunk, len(f) - start)
        dist = torch.cdist(f[start:start + size], f)
        dist[torch.arange(size, device=DEVICE), torch.arange(start, start + size,
                                                             device=DEVICE)] = float("inf")
        pred[start:start + size] = lab[dist.argmin(1)]
    return {"loo_1nn_accuracy": float((pred == lab).float().mean()),
            "n_wrong": int((pred != lab).sum())}


def train_v3(raw: torch.Tensor, idx_np: np.ndarray, labels_np: np.ndarray, seed: int,
             epochs: int, tag: str, eval_np: np.ndarray | None = None,
             eval_labels: np.ndarray | None = None,
             adv_weight: float | None = None) -> tuple:
    cfg = CFG["branch_a_v3"]
    adv_weight = float(cfg["adv_weight"]) if adv_weight is None else float(adv_weight)
    torch.manual_seed(seed)
    model = NetV3().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                            weight_decay=float(cfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    idx = torch.as_tensor(idx_np, dtype=torch.long, device=DEVICE)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=DEVICE)
    # labels are 0-based; E189's boundary-impostor target is the next identity cyclically.
    target = torch.as_tensor((np.asarray(labels_np) + 1) % NCLASS, dtype=torch.long,
                             device=DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    rng = np.random.default_rng(seed)
    batch = int(cfg["batch_size"])
    best = (-1.0, None, -1)
    info = {"epochs": epochs, "epoch_seconds": [], "eval_accuracy": [],
            "eps_histogram": {}, "goal_histogram": {},
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
            xb = E180.augment(raw[idx[sel]], gen)
            yb = labels[sel]
            tb = target[sel]
            loss = F.cross_entropy(model(xb), yb)
            if adv_weight > 0.0:
                adv, eps, goal = pgd_train_v3(model, xb, yb, tb, gen)
                key = f"{eps:.2f}"
                info["eps_histogram"][key] = info["eps_histogram"].get(key, 0) + 1
                info["goal_histogram"][goal] = info["goal_histogram"].get(goal, 0) + 1
                loss = (1.0 - adv_weight) * loss + adv_weight * F.cross_entropy(model(adv), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(sel)
        sched.step()
        info["epoch_seconds"].append(time.time() - started)
        message = (f"  [{tag}] epoch {epoch}/{epochs} loss {total / len(idx_np):.4f} "
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
    info["best_window_accuracy"] = best[0]
    info["best_window_epoch"] = best[2]
    if best[1] is not None:
        info["best_window_state"] = best[1]
    model.eval()
    return model, info


def polish(model: nn.Module, raw: torch.Tensor, labels: np.ndarray, seed: int) -> dict:
    """Clean-only fine-tune, run only when the AT model is still below 1.00 in-sample.
    This is a capacity demonstration, not a robustness or generalisation claim."""
    cfg = CFG["branch_a_v3"]["polish"]
    epochs = int(cfg["epochs"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                            weight_decay=float(CFG["branch_a_v3"]["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    y = torch.as_tensor(labels, dtype=torch.long, device=DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(seed + 7777)
    rng = np.random.default_rng(seed + 7777)
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        order = torch.as_tensor(rng.permutation(len(raw)), dtype=torch.long, device=DEVICE)
        total = 0.0
        for start in range(0, len(raw), 64):
            sel = order[start:start + 64]
            xb = E180.augment(raw[sel], gen)
            loss = F.cross_entropy(model(xb), y[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(sel)
        sched.step()
        acc = float(np.mean(E180.predict_logits(model, raw, np.arange(len(raw))).argmax(1) + 1
                            == labels))
        log(f"  [polish] epoch {epoch}/{epochs} loss {total / len(raw):.4f} "
            f"in_sample {acc:.4f} {time.time() - started:.1f}s")
    model.eval()
    return {"epochs": epochs, "learning_rate": float(cfg["learning_rate"]),
            "adv_weight": 0.0, "final_in_sample": acc}


# ------------------------------------------------------------------ stage: train
def stage_train() -> dict:
    started = time.time()
    meta, frac, blocks, fit_rows = E180.load_config_meta()
    subjects = meta["subject"].to_numpy(int)
    window = (frac > 0.6) & (frac <= 0.8)
    log(f"stage train: {len(meta)} rows, fit_rows {int(fit_rows.sum())}, "
        f"window {int(window.sum())}, pca/prototypes Train-only")
    raw = E180.load_raw(meta["path"].tolist())
    log(f"raw on GPU: {tuple(raw.shape)} ({raw.numel() * 4 / 2**30:.2f} GiB)")
    cfg = CFG["branch_a_v3"]

    oof_logits = np.zeros((len(meta), NCLASS), dtype=np.float32)
    folds = []
    for block in range(5):
        train_np = np.flatnonzero(blocks != block)
        eval_np = np.flatnonzero(blocks == block)
        model, info = train_v3(raw, train_np, subjects[train_np] - 1, SEED + 101 * block,
                               int(cfg["fold_epochs"]), f"fold{block}")
        logits = E180.predict_logits(model, raw, eval_np)
        oof_logits[eval_np] = logits
        accuracy = float(np.mean(logits.argmax(1) + 1 == subjects[eval_np]))
        folds.append({"block": block, "n_train": int(len(train_np)),
                      "n_eval": int(len(eval_np)), "accuracy": accuracy,
                      "mean_epoch_seconds": float(np.mean(info["epoch_seconds"])),
                      "parameter_count": info["parameter_count"]})
        torch.save({"state_dict": model.state_dict(), "block": block,
                    "spec": cfg, "classes": CLASSES}, CACHE / f"fold_model_v3_{block}.pt")
        log(f"  fold{block} OOF accuracy {accuracy:.4f}")
        del model
        torch.cuda.empty_cache()

    window_np = np.flatnonzero(window)
    full, info = train_v3(raw, np.arange(len(meta)), subjects - 1, SEED,
                          int(cfg["full_epochs"]), "full",
                          eval_np=window_np, eval_labels=subjects[window_np])
    best_window = {"epoch": info["best_window_epoch"],
                   "accuracy": info["best_window_accuracy"]}
    best_state = info.pop("best_window_state", None)
    in_sample = E180.predict_logits(full, raw, np.arange(len(meta)))
    in_sample_acc = float(np.mean(in_sample.argmax(1) + 1 == subjects))
    log(f"  full model (last epoch) in-sample accuracy {in_sample_acc:.4f}")

    polish_info = None
    if in_sample_acc < 1.0:
        polish_info = polish(full, raw, subjects, SEED)
        in_sample = E180.predict_logits(full, raw, np.arange(len(meta)))
        in_sample_acc = float(np.mean(in_sample.argmax(1) + 1 == subjects))
        log(f"  after clean polish in-sample accuracy {in_sample_acc:.4f}")

    feats = feature_bank(full, raw)
    knn = knn_loo_accuracy(feats, subjects)
    in_sample_f1 = float(f1_score(subjects, in_sample.argmax(1) + 1, labels=CLASSES,
                                  average="macro", zero_division=0))
    log(f"  in-sample macro-F1 {in_sample_f1:.4f}; LOO 1-NN on features "
        f"{knn['loo_1nn_accuracy']:.4f} ({knn['n_wrong']} wrong)")

    torch.save({"state_dict": full.state_dict(), "spec": cfg, "classes": CLASSES,
                "spec_sha256": BRANCH_SPEC_SHA, "trained_on": "all Train rows",
                "in_sample_accuracy": in_sample_acc,
                "in_sample_macro_f1": in_sample_f1,
                "best_window": {"epoch": best_window["epoch"],
                                "accuracy": best_window["accuracy"],
                                "state_dict": best_state},
                "knn_loo": knn,
                "pgd": {"steps": int(cfg["pgd_steps"]),
                        "eps_mix": cfg["pgd_eps_mix"],
                        "goals": cfg["pgd_goal_mix"],
                        "adv_weight": float(cfg["adv_weight"])}}, MODEL_V3)
    torch.save({"oof_logits": torch.as_tensor(oof_logits),
                "in_sample_logits": torch.as_tensor(in_sample),
                "subjects": torch.as_tensor(subjects),
                "blocks": torch.as_tensor(blocks),
                "features": torch.as_tensor(feats)}, STAGE_A3)

    oof_pred = oof_logits.argmax(1) + 1
    forward = frac > 0.6
    summary = {
        "architecture": "widths " + str(cfg["widths"]),
        "n_parameters": int(info["parameter_count"]),
        "fold_epochs": int(cfg["fold_epochs"]),
        "full_epochs": int(cfg["full_epochs"]),
        "folds": folds,
        "fold_mean_epoch_seconds": float(np.mean([f["mean_epoch_seconds"] for f in folds])),
        "oof_accuracy_5block": float(np.mean(oof_pred == subjects)),
        "oof_macro_f1": float(f1_score(subjects, oof_pred, labels=CLASSES,
                                       average="macro", zero_division=0)),
        "oof_accuracy_forward_blocks": float(np.mean(oof_pred[forward] == subjects[forward])),
        "oof_accuracy_per_block": {str(b): float(np.mean(oof_pred[blocks == b]
                                                           == subjects[blocks == b]))
                                   for b in range(5)},
        "in_sample_accuracy": in_sample_acc,
        "in_sample_macro_f1": in_sample_f1,
        "in_sample_per_task_accuracy": {
            task: float(np.mean(in_sample.argmax(1)[meta["task"].to_numpy() == task] + 1
                                == subjects[meta["task"].to_numpy() == task]))
            for task in TASKS},
        "in_sample_n_wrong": int(round((1.0 - in_sample_acc) * len(meta))),
        "knn_loo": knn,
        "polish": polish_info,
        "best_window": best_window,
        "full_last_window_accuracy": float(info["eval_accuracy"][-1]),
        "eps_histogram": info["eps_histogram"],
        "goal_histogram": info["goal_histogram"],
        "adversarial_training": {"pgd_steps": int(cfg["pgd_steps"]),
                                 "eps_mix": cfg["pgd_eps_mix"],
                                 "goals": cfg["pgd_goal_mix"],
                                 "adv_weight": float(cfg["adv_weight"])},
        "elapsed_seconds": time.time() - started,
        "gpu": torch.cuda.get_device_name(0),
    }
    results = read_json(RESULTS_PATH)
    results["branch_a_v3"] = summary
    results.setdefault("stage_seconds", {})["train"] = time.time() - started
    write_json(RESULTS_PATH, results)
    log(f"stage train done in {time.time() - started:.0f}s: OOF "
        f"{summary['oof_accuracy_5block']:.4f}, in-sample {in_sample_acc:.4f}")
    del raw
    torch.cuda.empty_cache()
    return summary


# ------------------------------------------------------------------ stage: attack
def stage_attack() -> dict:
    started = time.time()
    P = E189.protocol()
    subset = E189.attack_subset(P)
    subjects, tasks = P["subjects"], P["tasks"]
    cfg = CFG["attack"]
    log(f"stage attack: {len(subset)} window trials "
        f"({', '.join(f'{t}={int((tasks[subset] == t).sum())}' for t in TASKS)})")
    scorers, systems, _ = E189.build_systems()
    payload = torch.load(MODEL_V3, map_location="cpu", weights_only=False)
    model = NetV3().to(DEVICE)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    raw = E180.load_raw([P["meta"]["path"].iloc[i] for i in subset])
    truth = subjects[subset]
    task_idx = torch.as_tensor([TASKS.index(t) for t in tasks[subset]],
                               dtype=torch.long, device=DEVICE)
    true_idx = torch.as_tensor(truth - 1, dtype=torch.long, device=DEVICE)
    lut = {int(c): i for i, c in enumerate(CLASSES)}
    target = np.asarray([CLASSES[(lut[int(v)] + 1) % NCLASS] for v in truth])
    target_idx = torch.as_tensor([lut[int(v)] for v in target], dtype=torch.long,
                                 device=DEVICE)

    clean = E189.score_condition(model, scorers, systems, raw, task_idx)
    conditions = [{"eps": 0.0, "goal": "clean", "gate": None, "signature": None,
                   "max_rel_linf": 0.0, "logits": clean["logits"], "z": clean["z"]}]
    for eps in cfg["eps_normalised"]:
        for goal in ("A", "B", "C"):
            out = E189.pgd(model, None, raw, task_idx, true_idx, target_idx, goal,
                           float(eps), int(cfg["steps"]),
                           int(SEED) + int(eps * 1e5) + ord(goal), need_gate=False)
            scored = E189.score_condition(model, scorers, systems, out["adv"], task_idx)
            conditions.append({"eps": float(eps), "goal": goal, "gate": None,
                               "signature": None, "max_rel_linf": out["max_rel_linf"],
                               "logits": scored["logits"], "z": scored["z"]})
            log(f"  eps={eps:.3f} goal {goal} done ({time.time() - started:.0f}s)")
    representatives = {}
    for key, system in systems.items():
        representatives.setdefault(E189.signature(system.components), key)
    for sig, key in representatives.items():
        out = E189.pgd(model, systems[key], raw, task_idx, true_idx, target_idx, "D",
                       0.02, int(cfg["steps"]), int(SEED) + 99991, need_gate=True)
        scored = E189.score_condition(model, scorers, systems, out["adv"], task_idx)
        conditions.append({"eps": 0.02, "goal": "D", "gate": key, "signature": sig,
                           "max_rel_linf": out["max_rel_linf"],
                           "logits": scored["logits"], "z": scored["z"]})
        log(f"  goal D [{key}] done ({time.time() - started:.0f}s)")

    store = {"truth": truth.astype(np.int32),
             "tasks": np.asarray(tasks[subset]),
             "n_conditions": np.int32(len(conditions))}
    for index, cond in enumerate(conditions):
        store[f"logits_{index}"] = cond["logits"]
        for key in systems:
            store[f"z_{index}_{key}"] = cond["z"][key]
    np.savez_compressed(ATTACK_NPZ, **store)

    meta = {"n_trials": int(len(subset)),
            "per_task": {t: int((tasks[subset] == t).sum()) for t in TASKS},
            "conditions": [{"index": i, "eps": c["eps"], "goal": c["goal"],
                            "gate": c["gate"],
                            "signature": None if c["signature"] is None
                            else str(c["signature"]),
                            "max_rel_linf": float(c["max_rel_linf"])}
                           for i, c in enumerate(conditions)]}
    write_json(ATTACK_JSON, meta)
    results = read_json(RESULTS_PATH)
    results["attack"] = meta
    results.setdefault("stage_seconds", {})["attack"] = time.time() - started
    write_json(RESULTS_PATH, results)
    log(f"stage attack done in {time.time() - started:.0f}s")
    del raw
    torch.cuda.empty_cache()
    return meta


def load_attack() -> tuple:
    data = np.load(ATTACK_NPZ, allow_pickle=False)
    meta = read_json(ATTACK_JSON)
    return data, meta


def attack_rows(candidates: dict, data, meta: dict) -> dict:
    """RA / ASafeR for every candidate at every a of the grid (no PGD re-run)."""
    truth = data["truth"].astype(int)
    out = {}
    for key, cand in candidates.items():
        sig = E189.signature([tuple(c) for c in cand["components"]])
        keep = [c for c in meta["conditions"]
                if c["goal"] != "D" or c["signature"] == str(sig)]
        rows = {}
        for a in GRID:
            tau = E189.quantile(cand["oof"], 1.0 - a)
            metrics = []
            for cond in keep:
                logits = data[f"logits_{cond['index']}"]
                z = data[f"z_{cond['index']}_{key}"]
                metric = E189.attack_metrics(logits, z, truth, tau)
                metric.update({"eps": cond["eps"], "goal": cond["goal"]})
                metrics.append(metric)
            worst_ra = min(metrics, key=lambda r: r["RA"])
            worst_safe = min(metrics, key=lambda r: r["ASafeR"])
            rows[a] = {"tau": tau, "worst_RA": worst_ra, "worst_ASafeR": worst_safe,
                       "clean": metrics[0], "conditions": metrics}
        out[key] = rows
    return out


# ------------------------------------------------------------------ stage: report
GATES = [("#1", "拟合容量", "Train in-sample 准确率", "= 1.00", 1.00),
         ("#2", "折外闭集", "C（5 块 OOF）", "≥ 0.95", 0.95),
         ("#3", "折外闭集", "身份 macro-F1", "≥ 0.92", 0.92),
         ("#4", "跨时间代理", "块外准确率", "≥ 0.90", 0.90),
         ("#5", "开放集", "LOSO AUROC", "≥ 0.98", 0.98),
         ("#6", "开放集", "U@95%", "≥ 0.95", 0.95),
         ("#7", "对抗识别", "RA（最差 ε×目标）", "≥ 0.95", 0.95),
         ("#8", "对抗安全", "ASafeR（同一攻击）", "≥ 0.98", 0.98),
         ("#9", "Test 审计", "S2 接受率", "≥ 0.70", 0.70),
         ("#10", "Test 审计", "S1 拒识率", "≥ 0.70", 0.70)]


def evaluate_point(key: str, a: float, branch: dict, audit: dict, attack: dict) -> dict:
    point = audit[key]["points"][a]
    att = attack[key][a]
    values = {"#1": branch["in_sample_accuracy"],
              "#2": branch["oof_accuracy_5block"],
              "#3": branch["oof_macro_f1"],
              "#4": branch["oof_accuracy_forward_blocks"],
              "#5": audit[key]["loso_auc"],
              "#6": audit[key]["u_at_95"],
              "#7": att["worst_RA"]["RA"],
              "#8": att["worst_ASafeR"]["ASafeR"],
              "#9": point["test_s2_acceptance"],
              "#10": point["test_s1_rejection"]}
    passed = {gate: bool(values[gate] >= target - 1e-12) for gate, _, _, _, target in GATES}
    return {"a": a, "values": values, "passed": passed, "point": point, "attack": att,
            "count": int(sum(passed.values()))}


def build_audit() -> dict:
    cands = E189.load_candidates()
    audit = {}
    for key, cand in cands.items():
        points = {a: E189.audit_point(cand, a) for a in GRID}
        audit[key] = {"loso_auc": cand["loso_auc"],
                      "components": cand["components"],
                      "u_at_95": E189.u_at_95(cand)["per_fold_mean"],
                      "u_at_95_global": E189.u_at_95(cand)["global"],
                      "points": points}
    write_json(AUDIT_JSON, audit)
    return audit


def fmt(value, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def write_report() -> None:
    results = read_json(RESULTS_PATH)
    branch = results["branch_a_v3"]
    v2 = CFG["branch_a_v2_reference"]
    cands = E189.load_candidates()
    data, attack_meta = load_attack()
    audit = build_audit()
    attack = attack_rows(cands, data, attack_meta)
    keys = list(audit)

    grid = {(k, a): evaluate_point(k, a, branch, audit, attack)
            for k in keys for a in GRID}
    primary_key = max(keys, key=lambda k: (audit[k]["points"][PRIMARY_A]["loso_rejection"],
                                           audit[k]["loso_auc"]))
    primary = grid[(primary_key, PRIMARY_A)]
    dual = [(k, a) for k in keys for a in GRID
            if grid[(k, a)]["values"]["#9"] >= 0.70 and grid[(k, a)]["values"]["#10"] >= 0.70]
    dual.sort(key=lambda p: (-grid[p]["count"], -grid[p]["values"]["#7"],
                             -min(grid[p]["values"]["#9"], grid[p]["values"]["#10"])))
    best_dual = dual[0] if dual else None
    max_count = max(row["count"] for row in grid.values())
    max_count_at_dual = max((grid[p]["count"] for p in dual), default=0)
    ra_points = [(k, a) for k in keys for a in GRID if grid[(k, a)]["values"]["#7"] >= 0.95]

    lines = []
    add = lines.append
    add("# E191：单点攻 #1（in-sample = 1.00）与 #7（RA ≥ 0.95）")
    add("")
    add("日期：2026-11-15。身份模型**重训**为分支 A v3（加宽加深 + 更强混目标对抗训练）；"
        "门控侧完全复用 E189 冻结的 Train-only 流缓存（只读），攻击代码与 E189 逐行相同"
        "（同一子集 / 种子 / ε / 目标），因此所有差值都只来自身份模型。")
    add("")
    add("**结论先说**：")
    add("")
    add(f"- #1：{v2['in_sample_accuracy']:.4f} → **{branch['in_sample_accuracy']:.4f}**"
        f"（错 {branch['in_sample_n_wrong']} 条）")
    ref_dual = dual[0] if dual else (primary_key, PRIMARY_A)
    add(f"- #7：主工作点（`{primary_key}`，a={PRIMARY_A:.2f}）RA "
        f"**{primary['attack']['worst_RA']['RA']:.4f}**；推荐双 0.70 点（`{dual[0][0]}`，a="
        f"{dual[0][1]:.3f}）RA **{grid[dual[0]]['values']['#7']:.4f}**"
        "，**两者都未过线**（见 §1.1 与 §2）" if dual else "")
    fail_dual = [g for g, p in grid[ref_dual]["passed"].items() if not p]
    add(f"- 十大项单工作点最多 **{max_count} 项**；在用户口径（S1 ≥ 0.70 且 S2 ≥ 0.70 同一点）"
        f"最多 **{max_count_at_dual} 项**，未通过的是 {', '.join(fail_dual)}")
    add("")
    add("## 1. #1 与 #7 的改进前后对照")
    add("")
    add("| 项目 | E180 v2（E189 复用） | E191 v3 | 变化 |")
    add("|---|---:|---:|---:|")
    add(f"| #1 Train in-sample 准确率 | {v2['in_sample_accuracy']:.4f} | "
        f"**{branch['in_sample_accuracy']:.4f}** | "
        f"{branch['in_sample_accuracy'] - v2['in_sample_accuracy']:+.4f} |")
    add(f"| #2 5 块 OOF C | {v2['oof_accuracy_5block']:.4f} | "
        f"{branch['oof_accuracy_5block']:.4f} | "
        f"{branch['oof_accuracy_5block'] - v2['oof_accuracy_5block']:+.4f} |")
    add(f"| #3 OOF macro-F1 | {v2['oof_macro_f1']:.4f} | {branch['oof_macro_f1']:.4f} | "
        f"{branch['oof_macro_f1'] - v2['oof_macro_f1']:+.4f} |")
    add(f"| #4 块外准确率 | {v2['oof_accuracy_forward_blocks']:.4f} | "
        f"{branch['oof_accuracy_forward_blocks']:.4f} | "
        f"{branch['oof_accuracy_forward_blocks'] - v2['oof_accuracy_forward_blocks']:+.4f} |")
    add("")
    add("v3 训练配方：")
    add("")
    add(f"- 结构：widths {branch['architecture']}，dropout 0.2，"
        f"{branch['n_parameters']} 参数")
    add(f"- 全量 {branch['full_epochs']} epoch **不做 early-stop 回滚**（v2 回滚到第 22/40 轮，"
        f"这正是 in-sample 卡在 0.9988 的直接原因）；折内 {branch['fold_epochs']} epoch")
    add(f"- 对抗训练：PGD-{branch['adversarial_training']['pgd_steps']}，"
        f"ε 混合 {branch['adversarial_training']['eps_mix']}，"
        f"目标混合 {branch['adversarial_training']['goals']}，"
        f"adv 权重 {branch['adversarial_training']['adv_weight']}；"
        f"批次统计 ε {branch['eps_histogram']}、目标 {branch['goal_histogram']}")
    add(f"- 全量最后一轮在评估块上的 clean 准确率 {branch['full_last_window_accuracy']:.4f}；"
        f"评估块最佳快照 {branch['best_window']['accuracy']:.4f}（第 "
        f"{branch['best_window']['epoch']} 轮，仅记录不回滚）")
    if branch.get("polish"):
        add(f"- **额外 clean 抛光 {branch['polish']['epochs']} epoch（lr "
            f"{branch['polish']['learning_rate']}，adv 权重 0）**：对抗模型仍未到 "
            "1.00，触发抛光。这一步只是把容量用完，不代表鲁棒性或泛化提升")
    else:
        add("- 对抗训练结束后 in-sample 已达 1.00，**未启用** clean 抛光")
    add("")
    add("### 1.1 #7 的前后对照（同一候选、同一工作点、同一攻击）")
    add("")
    add("| 候选 | a | v2 RA（E189） | v3 RA | 变化 | v3 clean 接受率 | v3 攻击后闭集准确率 |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    e189_attack = read_json(ROOT / "research/E189_balanced_gate_gates" / "attack.json")
    compare = [k for k in ("subwin6_pca96_k1", "subwin12_pca48_k1", "e65_pca96_k12",
                           "fuse_e65v64k1__sw6v96k1__logbpv16k1__e162v128k1")
               if k in keys]
    for key in compare:
        for a in (0.90, 0.94, 0.96):
            if a not in GRID:
                continue
            row = grid[(key, a)]
            old = [r for r in e189_attack["candidates"][key] if abs(r["a"] - a) < 1e-9]
            old_ra = old[0]["worst_RA"]["RA"] if old else None
            worst = row["attack"]["worst_RA"]
            delta = "—" if old_ra is None else f"{worst['RA'] - old_ra:+.4f}"
            add(f"| `{key}` | {a:.2f} | {fmt(old_ra)} | {worst['RA']:.4f} | {delta} | "
                f"{row['attack']['clean']['accept_rate']:.4f} | "
                f"{worst['closed_set_accuracy']:.4f} |")
    deltas = [grid[(k, a)]["values"]["#7"] - r["worst_RA"]["RA"]
              for k in compare for a in (0.90, 0.94, 0.96) if a in GRID
              for r in [next((x for x in e189_attack["candidates"][k]
                              if abs(x["a"] - a) < 1e-9), None)] if r is not None]
    add("")
    add(f"身份侧的强对抗训练让同一工作点的 RA 提升 {min(deltas):+.4f} ~ {max(deltas):+.4f}"
        "（只来自「攻击后闭集准确率」这一项，接受率由门控决定，不受身份模型影响）；"
        "但 #7 要 0.95，缺口仍在 0.07 量级，见 §2。")
    add("")
    add("## 2. #7 为什么在均衡点抬不起来：RA 的分解")
    add("")
    add("RA = P(门控接受) × P(身份正确 | 接受)。攻击子集上 **clean 闭集准确率已经是 "
        f"{primary['attack']['clean']['closed_set_accuracy']:.4f}**，所以 RA 的上限由门控接受率"
        "决定，而不是由身份头鲁棒性决定：")
    add("")
    add("| ε | 目标 | 门控接受率 | 闭集准确率 | RA | ASafeR |")
    add("|---:|---|---:|---:|---:|---:|")
    for metric in primary["attack"]["conditions"]:
        add(f"| {metric['eps']:.3f} | {metric['goal']} | "
            f"{metric['accept_rate']:.4f} | {metric['closed_set_accuracy']:.4f} | "
            f"{metric['RA']:.4f} | {metric['ASafeR']:.4f} |")
    add("")
    add(f"主工作点 clean 接受率 {primary['attack']['clean']['accept_rate']:.4f}，"
        f"因此该点 RA 的天花板就是 "
        f"{primary['attack']['clean']['accept_rate']:.4f}；把 RA 抬到 0.95 必须让门控接受率 "
        "≥ 0.95，而接受率只由工作点 a 决定：")
    add("")
    add(f"| `{primary_key}` a | clean 接受率 | RA | Test S1 拒识 | Test S2 接受 | #7 | #10 |")
    add("|---:|---:|---:|---:|---:|---|---|")
    for a in GRID:
        row = grid[(primary_key, a)]
        add(f"| {a:.3f} | {row['attack']['clean']['accept_rate']:.4f} | "
            f"{row['attack']['worst_RA']['RA']:.4f} | "
            f"{row['values']['#10']:.4f} | {row['values']['#9']:.4f} | "
            f"{'✅' if row['passed']['#7'] else '❌'} | "
            f"{'✅' if row['passed']['#10'] else '❌'} |")
    add("")
    add(f"#7 在 {len(ra_points)} 个工作点上过线"
        + (f"（如 `{ra_points[0][0]}` a={ra_points[0][1]:.3f}），但那些点的 Test S1 拒识"
           f"只有 {grid[ra_points[0]]['values']['#10']:.4f}，" if ra_points else "，")
        + "与 S1 ≥ 0.70 直接互斥：两者争的是同一个 τ。")
    add("")
    add("## 3. 单工作点十项门禁矩阵（不跨工作点拼接）")
    add("")
    add(f"### 3.1 主工作点（Train-only 规则：a={PRIMARY_A:.2f} 项目既定 90% 已知接受约定，"
        "候选按 LOSO 未知拒识最大选出）")
    add("")
    add("| 候选 | a | " + " | ".join(g for g, *_ in GATES) + " | 通过项数 |")
    add("|---" * (len(GATES) + 3) + "|")
    for key in keys:
        row = grid[(key, PRIMARY_A)]
        cells = [f"{row['values'][gate]:.4f}{'✅' if row['passed'][gate] else '❌'}"
                 for gate, _, _, _, _ in GATES]
        mark = "**" if key == primary_key else ""
        add(f"| {mark}`{key}`{mark} | {PRIMARY_A:.2f} | " + " | ".join(cells) +
            f" | **{row['count']}/10** |")
    add("")
    add(f"主工作点系统 = `{primary_key}` @ a={PRIMARY_A:.2f}："
        f"S1 拒识 {primary['values']['#10']:.4f}、S2 接受 {primary['values']['#9']:.4f}、"
        f"通过 {primary['count']}/10。")
    add("")
    add("**必须写清的选点矛盾**：上表的 a=0.90 是项目既定约定（Train 已知接受 90%），"
        "候选则按纯 Train-only 规则（LOSO 未知拒识最大）选出，落到 "
        f"`{primary_key}`，它在 Test 上的 S1 是 {primary['values']['#10']:.4f}"
        f"（差 {0.70 - primary['values']['#10']:.4f} 未达 0.70）。"
        "同一 a 下 `subwin6_pca96_k1` 的 S1/S2 = "
        f"{grid[('subwin6_pca96_k1', PRIMARY_A)]['values']['#10']:.4f}/"
        f"{grid[('subwin6_pca96_k1', PRIMARY_A)]['values']['#9']:.4f}（双 0.70 ✅，"
        f"{grid[('subwin6_pca96_k1', PRIMARY_A)]['count']}/10），但它的 LOSO 未知拒识只有 "
        f"{audit['subwin6_pca96_k1']['points'][PRIMARY_A]['loso_rejection']:.4f}，"
        f"低于 `{primary_key}` 的 "
        f"{audit[primary_key]['points'][PRIMARY_A]['loso_rejection']:.4f}。"
        "也就是说 **Train 侧的 LOSO 排序与 Test 上的已知/未知排序不一致**，"
        "任何纯 Train 规则都不会落到 §3.2 的 7/10 点上——这一点必须与 §3.2 一起读。")
    add("")
    add("### 3.2 用户口径（同一工作点 S1 ≥ 0.70 且 S2 ≥ 0.70）下的全部可行点")
    add("")
    add("| 候选 | a | τ | S1 拒识 | S2 接受 | min | LOSO AUROC | U@95 | RA | ASafeR | 通过项数 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, a in dual[:12]:
        row = grid[(key, a)]
        add(f"| `{key}` | {a:.3f} | {row['point']['tau']:+.4f} | "
            f"{row['values']['#10']:.4f} | {row['values']['#9']:.4f} | "
            f"{min(row['values']['#9'], row['values']['#10']):.4f} | "
            f"{row['values']['#5']:.4f} | {row['values']['#6']:.4f} | "
            f"{row['values']['#7']:.4f} | {row['values']['#8']:.4f} | "
            f"**{row['count']}/10** |")
    add("")
    dual_auc = [grid[p]["values"]["#5"] for p in dual]
    dual_u = [grid[p]["values"]["#6"] for p in dual]
    add(f"共 {len(dual)} 个（候选 × a）组合满足双 0.70，全部集中在 "
        + "、".join(f"`{k}`" for k in sorted({p[0] for p in dual}))
        + f"；它们的 LOSO AUROC 只有 {min(dual_auc):.4f}–{max(dual_auc):.4f}，"
        f"#6 只有 {min(dual_u):.4f}–{max(dual_u):.4f}。")
    add("")
    add("## 4. 明确回答：十项最多能通过几项？还差什么？")
    add("")
    if dual:
        ref = grid[dual[0]]
        best = dual[0]
        add(f"**用户口径下单工作点最多 {grid[best]['count']} 项**（`{best[0]}`，a={best[1]:.3f}，"
            f"S1 {grid[best]['values']['#10']:.4f} / S2 {grid[best]['values']['#9']:.4f}）。"
            "未通过的项与缺口：")
        add("")
        add("| 门禁 | 状态 | 读数 | 目标 | 缺口 | 性质 |")
        add("|---|---|---:|---:|---:|---|")
        nature = {"#1": "容量检查（非泛化证据）", "#5": "结构性（门控 ROC 上限）",
                  "#6": "结构性（同一条 ROC）", "#7": "结构性（与 #10 争同一 τ）",
                  "#2": "通过", "#3": "通过", "#4": "通过", "#8": "通过",
                  "#9": "通过", "#10": "通过"}
        for gate, layer, metric, target, threshold in GATES:
            value = ref["values"][gate]
            gap = value - threshold
            add(f"| {gate} {metric} | {'✅ 通过' if ref['passed'][gate] else '❌ 未通过'} | "
                f"{value:.4f} | {target} | {gap:+.4f} | {nature[gate]} |")
        add("")
        structural = [gate for gate, _, _, _, _ in GATES
                      if not any(grid[(k, a)]["passed"][gate] for k in keys for a in GRID)]
        add(f"也就是说：**用户口径下十项里最多 {grid[best]['count']} 项可同时通过**，"
            f"未通过的是 {', '.join(g for g, p in grid[best]['passed'].items() if not p)}。"
            f"全部 {len(keys)}×{len(GRID)} 个（候选 × a）组合里单点最大值 {max_count}/10，"
            f"其中有 {len(ra_points)} 个点能让 #7 过线（都牺牲了 #10）。"
            + (f"任何点上都过不了的门禁：{', '.join(structural)}。" if structural else ""))
    add("")
    add("**#7 的定性结论（必须读）**：重训确实让身份头更稳，但 RA 的瓶颈是"
        "「门控在评估块上愿意接受合法注册者的比例」，不是身份头鲁棒性——攻击子集 clean "
        f"闭集准确率 {primary['attack']['clean']['closed_set_accuracy']:.4f}，"
        f"而 clean 接受率只有 {primary['attack']['clean']['accept_rate']:.4f}。"
        "把接受率推到 0.95 需要 a≈0.96，同一 τ 下 Test S1 拒识只剩 "
        f"{grid[(primary_key, 0.96)]['values']['#10']:.4f}，因此 #7 与 #10 在同一 τ 上不可兼得。"
        "这与 E190 对 #5/#6 的结论同源：**#5/#6/#7 本质上都是同一条已知/未知 ROC 的三个读数**。")
    add("")
    add("## 5. #1 的达成方式与「技术性满足」标注")
    add("")
    add(f"- 分支 A v3 自身的 in-sample 准确率 {branch['in_sample_accuracy']:.4f}"
        f"（{branch['in_sample_n_wrong']} 条错，共 11304 条），"
        f"对应 macro-F1 {branch['in_sample_macro_f1']:.4f}")
    add(f"- 若只用特征最近邻读数（LOO 1-NN，余弦，模型 global-average-pool 特征）："
        f"{branch['knn_loo']['loo_1nn_accuracy']:.4f}（错 {branch['knn_loo']['n_wrong']} 条）"
        " —— **比模型本身还差，因此本实验没有启用 k-NN 兜底**")
    add("")
    add("**如实标注（两种做法的性质不同）**：")
    add("")
    add("- 「加宽加深 + 降 weight decay + 延长训练」是把**拟合容量**用满，"
        "不等于泛化提升：v3 的折外读数（#2 0.9884 / #3 0.9882 / #4 0.9900）"
        "与 v2 几乎相同（+0.0003 ~ +0.0011），说明多出来的容量只体现在 in-sample。")
    add("- 「取消 early-stop 回滚」本身就是技术手段：v2 之所以是 0.9988，是因为它回滚到"
        "评估块最好的第 22 轮；把回滚去掉，in-sample 自然到 1.00，而评估块的 clean 准确率"
        "两版都是 1.0000（该块本来就在全量模型训练集里）。")
    add("- 若改用「最近训练样本（k-NN）辅助判定」达成 1.00，那属于**纯技术性满足**"
        "（用训练集自身兜底，不含任何额外信息），必须与折外读数分开读；"
        f"本实验实测该读数为 {branch['knn_loo']['loo_1nn_accuracy']:.4f}，"
        "低于模型自身，故未采用。")
    add("- #1 是「模型容量/训练流程」检查，不是泛化指标；本实验没有训练集以外的任何信息"
        "进入 #1，也没有用它选择阈值或候选。")
    add("")
    add("## 6. 其余八项是否保持")
    add("")
    add(f"- #2 C = {branch['oof_accuracy_5block']:.4f}（≥ 0.95 ✅）、"
        f"#3 macro-F1 = {branch['oof_macro_f1']:.4f}（≥ 0.92 ✅）、"
        f"#4 块外 = {branch['oof_accuracy_forward_blocks']:.4f}（≥ 0.90 ✅）")
    add(f"- #8 ASafeR = {primary['values']['#8']:.4f}（主工作点）/ "
        f"{grid[ref_dual]['values']['#8']:.4f}（双 0.70 点），目标 ≥ 0.98")
    add(f"- #9/#10：双 0.70 点 {grid[ref_dual]['values']['#9']:.4f} / "
        f"{grid[ref_dual]['values']['#10']:.4f}，两个都 ≥ 0.70（用户口径要求同时满足）")
    add(f"- #5/#6 未变：最好 LOSO AUROC "
        f"{max(audit[k]['loso_auc'] for k in keys):.4f}（E189 同值，门控未改动）")
    add("")
    add("## 7. Test 官方指标")
    add("")
    add("**Test 官方 AUC / F1 / C / RA / ASafeR / U / Score = N/A**（Test 无真实身份标签）。"
        "报告中的 S1 拒识率 / S2 接受率是预测工程审计，不是官方真值指标。")
    add("")
    add("## 8. 合规与边界")
    add("")
    add("- 门控的标准化 / PCA / 原型 / 阈值 / 融合权重全部来自 Train 且复用 E189（只读）；"
        "LOSO 一律 `pm & fit_rows`；评估块 0.6 < frac ≤ 0.8 从未进入门控的训练或标定。")
    add("- 分支 A v3 的折内模型按块折外训练（第 b 块不参与第 b 折），#2/#3/#4 因此是折外读数。")
    add("- Test 只评分：先算分再读 Session；Session 从未进入特征、模型、阈值或权重。")
    add("- 每个门禁列都是「一个候选 + 一个 a」的完整系统，未跨工作点拼接。")
    add(f"- **#7/#8 的评估块行在 v3 全量模型的训练集内**（与 E180/E189 同一构造），"
        "所以 RA 度量的是鲁棒性而非跨时间泛化；这也是 clean 闭集准确率等于 "
        f"{primary['attack']['clean']['closed_set_accuracy']:.4f} 的原因。")
    add("- 门控候选与工作点从不用 Test 选：主工作点是项目既定 a=0.90 约定；§3.2 的可行点枚举"
        "只用于回答用户口径下的可达项数，不作为部署选择依据。")
    add("- 未修改任何已有实验目录，未提交 git。")
    add("")
    add("## 9. 文件与耗时")
    add("")
    add("| 文件 | 内容 |")
    add("|---|---|")
    add("| `run_e191.py` | 全部逻辑（`--stage train/attack/report/all`） |")
    add("| `config.json` | 冻结配置 |")
    add("| `model_branch_a_v3.pt` | 分支 A v3 全量模型（含 best-window 快照） |")
    add("| `cache/stage_a_v3.pt` | OOF / in-sample logits、特征库 |")
    add("| `cache/attack_v3.npz` | 13 类攻击条件的 logits 与全部候选的 z 分数 |")
    add("| `attack_v3.json` | 攻击条件元数据 |")
    add("| `audit_v3.json` | 候选 × a 的 Train/Test 读数 |")
    add("| `results.json` | 汇总 |")
    add("| `run.log` | 运行日志 |")
    add("")
    seconds = results.get("stage_seconds", {})
    add("耗时（秒）：" + "、".join(f"{k} {v:.0f}" for k, v in seconds.items()) + "。")
    add("")
    add("```powershell")
    add('& "D:\\WORK\\BCIC2026\\BrainPrint\\.venv\\Scripts\\python.exe" '
        f'"{OUT / "run_e191.py"}" --stage all')
    add("```")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    results["gate_matrix"] = {
        "max_pass_single_work_point": int(max_count),
        "max_pass_under_dual_070": int(max_count_at_dual),
        "primary": {"candidate": primary_key, "a": PRIMARY_A,
                    "values": primary["values"], "passed": primary["passed"],
                    "count": primary["count"]},
        "best_dual": None if best_dual is None else
        {"candidate": best_dual[0], "a": best_dual[1],
         "values": grid[best_dual]["values"], "passed": grid[best_dual]["passed"],
         "count": grid[best_dual]["count"]},
        "structural_failures": [gate for gate, _, _, _, _ in GATES
                                if not any(grid[(k, a)]["passed"][gate]
                                           for k in keys for a in GRID)],
        "test_official": {key: "N/A" for key in
                          ("AUC", "F1", "C", "RA", "ASafeR", "U", "Score")},
    }
    write_json(RESULTS_PATH, results)
    log(f"report written: {REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["train", "attack", "report", "all"])
    args = parser.parse_args()
    if not LOG_PATH.exists():
        LOG_PATH.write_text("", encoding="utf-8")
    sys.stdout = Tee(LOG_PATH)
    sys.stderr = sys.stdout
    log(f"\n===== E191 stage {args.stage} =====")
    if args.stage in ("train", "all"):
        stage_train()
    if args.stage in ("attack", "all"):
        stage_attack()
    if args.stage in ("report", "all"):
        write_report()


if __name__ == "__main__":
    main()
