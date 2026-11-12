"""E45：原始波形 CNN + PGD7 对抗训练的完整嵌套评估。

每个 inner fold 独立训练：0.5*CE(clean) + 0.5*CE(adv)，adv 为原始波形 PGD7，
SNR20/30 各半、25% 目标攻击。评估沿用 E36C 六攻击（PGD20），
阈值取该 run 内层 clean known 的 distance_ratio q05。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
sys.path.insert(0, str(OUT.parent / "E40_attack_rejection"))

from run_e36 import load_config as load_e36_config, train_metadata  # noqa: E402
from run_attacks import ATTACK_ORDER, load_all_raw, raw_attack, stratified_subset  # noqa: E402
from run_e40 import confidence_values  # noqa: E402

CONFIG_PATH = OUT / "config.json"
RESULTS_PATH = OUT / "results.json"
ATTACK_CODES = {"untargeted_fgsm": 1, "untargeted_pgd": 2, "targeted_pgd": 3}
REFERENCE = {"E36C": {"C": 0.9395, "RA": 0.0609, "ASafeR": 0.6818, "U": 1.0, "Score": 59.78}}


def official_score(c: float, a: float, u: float) -> float:
    return float(100 * c**0.40 * a**0.35 * u**0.25) if min(c, a, u) > 0 else 0.0


class EEGNet1d(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, width: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, width, 9, stride=2, padding=4),
            nn.BatchNorm1d(width), nn.ELU(),
            nn.Conv1d(width, width * 2, 9, stride=2, padding=4),
            nn.BatchNorm1d(width * 2), nn.ELU(),
            nn.Conv1d(width * 2, width * 4, 9, stride=2, padding=4),
            nn.BatchNorm1d(width * 4), nn.ELU(),
            nn.Conv1d(width * 4, width * 4, 9, stride=2, padding=4),
            nn.BatchNorm1d(width * 4), nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(width * 4, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / x.reshape(len(x), -1).std(dim=1).view(-1, 1, 1).clamp_min(1e-6)
        return self.head(self.net(x).squeeze(-1))


class Wrapper:
    """把 CNN 包装成 raw_attack 需要的接口（float64 输入 → float32 前向）。"""

    def __init__(self, net: nn.Module, classes: np.ndarray):
        self.net = net
        self.classes = np.asarray(classes, dtype=int)

    def __call__(self, raw_input: torch.Tensor) -> torch.Tensor:
        return self.net(raw_input.float())


def pgd_raw(model, clean, labels, snr, steps, generator, targeted=False, target=None):
    centered = clean - clean.mean(dim=-1, keepdim=True)
    signal_norm = centered.reshape(len(clean), -1).norm(dim=1).clamp_min(1e-12)
    radius = signal_norm / (10.0 ** (float(snr) / 20.0))
    delta = torch.randn(clean.shape, generator=generator, device=clean.device, dtype=clean.dtype)
    delta = delta / delta.reshape(len(delta), -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
    delta = delta * radius.view(-1, 1, 1) * torch.rand(
        len(clean), generator=generator, device=clean.device, dtype=clean.dtype).view(-1, 1, 1)
    objective = target if targeted else labels
    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        loss = F.cross_entropy(model(clean + delta), objective)
        gradient = torch.autograd.grad(-loss if targeted else loss, delta)[0]
        direction = gradient / gradient.reshape(len(gradient), -1).norm(
            dim=1).view(-1, 1, 1).clamp_min(1e-12)
        delta = delta.detach() + (2.0 * radius / steps).view(-1, 1, 1) * direction
        norm = delta.reshape(len(delta), -1).norm(dim=1).clamp_min(1e-12)
        delta = delta * torch.minimum(torch.ones_like(norm), radius / norm).view(-1, 1, 1)
    return (clean + delta).detach()


def train_fold(raw, fit_index, fit_labels, n_classes, config, seed, device):
    torch.manual_seed(seed)
    model = EEGNet1d(18, n_classes, int(config["model"]["width"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(fit_index))
        total = 0.0
        for start in range(0, len(order), int(config["training"]["batch_size"])):
            sel = torch.as_tensor(order[start:start + int(config["training"]["batch_size"])],
                                  device=device)
            clean = raw[fit_index[sel]]
            labels = fit_labels[sel]
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed + 1000 * epoch + start))
            snr = 20 if rng.random() < 0.5 else 30
            targeted = rng.random() < float(config["training"]["targeted_mix"])
            target = (labels + 1) % n_classes if targeted else None
            attacked = pgd_raw(model, clean, labels, snr,
                               int(config["training"]["pgd_steps"]), generator,
                               targeted, target)
            loss = 0.5 * F.cross_entropy(model(clean), labels) + \
                0.5 * F.cross_entropy(model(attacked), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(sel)
        scheduler.step()
        if epoch % 10 == 0:
            print(f"    epoch={epoch} loss={total / len(fit_index):.4f}", flush=True)
    model.eval()
    return model


def predict(model, raw, indices, device, batch=512):
    output = []
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            selected = torch.as_tensor(indices[start:start + batch], device=device)
            output.append(model(raw[selected]).cpu().numpy())
    return np.concatenate(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "inner"), default="inner")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("E45 要求 CUDA GPU")
    started = time.time()
    device = torch.device("cuda")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    e36 = load_e36_config()
    metadata = train_metadata(e36)
    subjects = metadata["subject"].to_numpy(dtype=int)
    trial = metadata["trial"].to_numpy()
    maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    forward = trial > 0.8 * maximum
    inner_block = (trial > 0.6 * maximum) & (trial <= 0.8 * maximum)
    raw = load_all_raw(metadata, e36, device)
    all_subjects = np.unique(subjects)
    outer_groups = np.array_split(
        np.random.default_rng(e36["nested_protocol"]["outer_seed"]).permutation(all_subjects), 5)
    if args.stage == "smoke":
        outer_groups = outer_groups[:1]
    results = {"status": "running", "experiment": "E45", "stage": args.stage,
               "outer_folds": []}

    for outer_index, unknown_subjects in enumerate(outer_groups):
        outer_fold = outer_index + 1
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        inner_order = np.random.default_rng(100 + outer_fold).permutation(known_subjects)
        inner_splits = list(np.array_split(inner_order, 2))
        if args.stage == "smoke":
            inner_splits = inner_splits[:1]
        runs = []
        for inner_index, inner_unknown in enumerate(inner_splits):
            inner_fold = inner_index + 1
            enrolled = np.setdiff1d(known_subjects, inner_unknown)
            mask = np.isin(subjects, enrolled)
            fit_np = np.flatnonzero(mask & ~forward & ~inner_block)
            known_np = np.flatnonzero(mask & inner_block)
            unknown_np = np.flatnonzero(np.isin(subjects, inner_unknown) & ~forward)
            class_to_index = {int(v): i for i, v in enumerate(enrolled)}
            fit_index = torch.as_tensor(fit_np, device=device)
            fit_labels = torch.as_tensor([class_to_index[int(v)] for v in subjects[fit_np]],
                                         dtype=torch.long, device=device)
            seed = int(config["seed_base"] + 1000 * outer_fold + inner_fold)
            print(f"outer={outer_fold} inner={inner_fold} 训练 start", flush=True)
            model = train_fold(raw, fit_index, fit_labels, len(enrolled), config, seed, device)
            known_subset = stratified_subset(metadata, known_np, 8)
            unknown_subset = stratified_subset(metadata, unknown_np, 8)
            known_scores = predict(model, raw, known_subset, device)
            unknown_scores = predict(model, raw, unknown_subset, device)
            run = {
                "outer_fold": outer_fold, "inner_fold": inner_fold, "classes": enrolled,
                "known_true": subjects[known_subset], "known_scores": known_scores,
                "unknown_scores": unknown_scores, "attacks": {},
                "n_known": int(len(known_subset)), "n_unknown": int(len(unknown_subset)),
            }
            wrapper = Wrapper(model, enrolled)
            raw_known = raw[torch.as_tensor(known_subset, device=device)]
            true = subjects[known_subset]
            target = np.asarray([
                enrolled[(int(np.flatnonzero(enrolled == v)[0]) + 1) % len(enrolled)]
                for v in true
            ], dtype=int)
            for attack_name in ATTACK_ORDER:
                base_name, snr_text = attack_name.rsplit("_snr", 1)
                snr = int(snr_text)
                attack_seed = int(config["seed_base"] + 100000 + 10000 * outer_fold
                                  + 1000 * inner_fold + 100 * ATTACK_CODES[base_name] + snr)
                scores, diagnostics, _ = raw_attack(
                    wrapper, raw_known, true, snr, base_name, attack_seed,
                    {"attacks": {"pgd_steps": 20, "batch_size": 64, "temperature": 1.0,
                                 "seed_base": config["seed_base"]}},
                    target=target if base_name == "targeted_pgd" else None,
                )
                run["attacks"][attack_name] = {"scores": np.asarray(scores),
                                               "diagnostics": diagnostics}
            runs.append(run)
            print(f"outer={outer_fold} inner={inner_fold} 完成 "
                  f"clean_direct={np.mean(enrolled[known_scores.argmax(1)] == run['known_true']):.4f}",
                  flush=True)
            del model, wrapper
            torch.cuda.empty_cache()
        per_attack = {}
        for attack_name in ATTACK_ORDER:
            rows = []
            for run in runs:
                classes = np.asarray(run["classes"], dtype=int)
                threshold = float(np.quantile(
                    confidence_values(run["known_scores"])["distance_ratio"],
                    0.05, method="higher"))
                clean_conf = confidence_values(run["known_scores"])["distance_ratio"]
                clean_accept = clean_conf >= threshold
                clean_correct = classes[run["known_scores"].argmax(1)] == run["known_true"]
                unknown_accept = confidence_values(run["unknown_scores"])["distance_ratio"] >= threshold
                scores = run["attacks"][attack_name]["scores"]
                conf = confidence_values(scores)["distance_ratio"]
                accept = conf >= threshold
                correct = classes[scores.argmax(1)] == run["known_true"]
                labels = np.concatenate([np.ones(len(conf)), np.zeros(len(clean_conf))])
                values = np.concatenate([conf, clean_conf])
                rows.append({
                    "C": float(np.mean(clean_correct & clean_accept)),
                    "RA": float(np.mean(correct & accept)),
                    "ASafeR": float(np.mean(correct | ~accept)),
                    "U": float(np.mean(~unknown_accept)),
                    "direct_RA": float(np.mean(correct)),
                    "Rej": float(1.0 - np.mean(accept)),
                    "unknown_auroc": float(roc_auc_score(labels, values)),
                    "identity_macro_f1": float(f1_score(
                        run["known_true"], np.where(clean_accept,
                                                    classes[run["known_scores"].argmax(1)], 0),
                        labels=classes, average="macro", zero_division=0)),
                })
            per_attack[attack_name] = {key: float(np.mean([row[key] for row in rows]))
                                       for key in rows[0]}
        macro = {key: float(np.mean([row[key] for row in per_attack.values()]))
                 for key in per_attack[ATTACK_ORDER[0]]}
        macro["A"] = 0.7 * macro["RA"] + 0.3 * macro["ASafeR"]
        macro["Score"] = official_score(macro["C"], macro["A"], macro["U"])
        results["outer_folds"].append({"outer_fold": outer_fold, "per_attack": per_attack,
                                       "macro": macro})
        RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        print(f"E45 outer={outer_fold}: Score={macro['Score']:.2f} C={macro['C']:.4f} "
              f"RA={macro['RA']:.4f} ASafeR={macro['ASafeR']:.4f} A={macro['A']:.4f} "
              f"U={macro['U']:.4f} F1={macro['identity_macro_f1']:.4f}", flush=True)

    macro = {key: float(np.mean([f["macro"][key] for f in results["outer_folds"]]))
             for key in results["outer_folds"][0]["macro"]}
    results["macro_fold_mean"] = macro
    results["reference"] = REFERENCE
    results["status"] = "complete"
    results["elapsed_seconds"] = time.time() - started
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"E45 MACRO: Score={macro['Score']:.2f} C={macro['C']:.4f} RA={macro['RA']:.4f} "
          f"ASafeR={macro['ASafeR']:.4f} A={macro['A']:.4f} U={macro['U']:.4f} "
          f"F1={macro['identity_macro_f1']:.4f}", flush=True)
    print(f"完成：{RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
