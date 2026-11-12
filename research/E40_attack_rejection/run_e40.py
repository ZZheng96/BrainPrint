"""E40：双门控攻击感知拒识（冻结 E36 WCCN 模型，不重新训练）。

规则：distance_ratio >= t_dr 且 max_cosine >= t_mc 才接受。
两个阈值由 inner clean known 上的共同低分位 q 求解，使联合接受率≈0.95。
对照：confidence_only（只用 distance_ratio 的 q05，等价 E36C 工作点）。

指标口径与 E36C 一致：合并每个外层折的两个 inner run，再对五折取均值。
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
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))

from run_e36 import (  # noqa: E402
    ZBAND_PATH,
    detrend_zband,
    gpu_logspd,
    load_config as load_e36_config,
    train_metadata,
)
from run_attacks import (  # noqa: E402
    ATTACK_ORDER,
    DifferentiableWCCN,
    load_all_raw,
    raw_attack,
    stratified_subset,
)

CONFIG_PATH = OUT / "config.json"
RESULTS_PATH = OUT / "results.json"
ATTACK_CODES = {"untargeted_fgsm": 1, "untargeted_pgd": 2, "targeted_pgd": 3}
REFERENCE = {"C": 0.939540, "RA": 0.060891, "ASafeR": 0.681814, "A": 0.247168,
             "U": 1.0, "Score": 59.784}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def official_score(c: float, a: float, u: float) -> float:
    return float(100 * c**0.40 * a**0.35 * u**0.25) if min(c, a, u) > 0 else 0.0


def confidence_values(scores: np.ndarray) -> dict[str, np.ndarray]:
    ordered = np.sort(np.asarray(scores, dtype=np.float64), axis=1)
    maximum = ordered[:, -1]
    second = ordered[:, -2]
    return {
        "max_cosine": maximum,
        "margin": maximum - second,
        "distance_ratio": (1.0 - second) / np.maximum(1.0 - maximum, 1e-6),
    }


class FrozenPrototype:
    """E36 WCCN 原型的只读推理封装：原始波形 → 分数。"""

    def __init__(self, prototype, config: dict):
        self.classes = np.asarray(prototype.classes, dtype=int)
        self.whitening = prototype.whitening.detach().float()
        self.prototypes = prototype.prototypes.detach().float()
        self.whitened_means = getattr(prototype, "whitened_means", None)
        self.config = config

    def scores(self, raw: torch.Tensor, batch: int = 512) -> np.ndarray:
        output = np.empty((len(raw), len(self.classes)), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(raw), batch):
                chunk = raw[start:start + batch]
                features = detrend_zband(gpu_logspd(chunk, self.config), self.config)
                whitened = features @ self.whitening.to(features.dtype)
                normalized = whitened / whitened.norm(dim=1, keepdim=True).clamp_min(1e-12)
                output[start:start + len(chunk)] = (
                    normalized @ self.prototypes.to(features.dtype).T
                ).float().cpu().numpy()
        return output


def fit_prototype(zband: torch.Tensor, fit_np: np.ndarray, subjects: np.ndarray,
                  classes: np.ndarray, shrinkage: float, floor: float):
    index = torch.as_tensor(fit_np, device=zband.device)
    x = zband[index]
    y = subjects[fit_np]
    class_to_index = {int(label): i for i, label in enumerate(classes)}
    target = torch.as_tensor([class_to_index[int(v)] for v in y], dtype=torch.long,
                             device=x.device)
    means = torch.stack([x[target == i].mean(0) for i in range(len(classes))])
    residual = x - means[target]
    covariance = residual.T @ residual / max(len(x) - len(classes), 1)
    trace = torch.diagonal(covariance).sum()
    eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * trace / x.shape[1] * eye
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    whitening = (eigenvectors * eigenvalues.clamp_min(floor).rsqrt().unsqueeze(0)) @ eigenvectors.T
    whitened_means = means @ whitening
    prototypes = whitened_means / whitened_means.norm(dim=1, keepdim=True).clamp_min(1e-12)

    class _Proto:
        pass

    proto = _Proto()
    proto.classes = np.asarray(classes, dtype=int)
    proto.whitening = whitening
    proto.prototypes = prototypes
    proto.whitened_means = whitened_means
    return proto


def solve_two_gate(runs: list[dict]) -> dict:
    """只用 clean known 求解共享低分位 q，使联合接受率≈0.95。"""
    dr = np.concatenate([confidence_values(r["known_scores"])["distance_ratio"] for r in runs])
    mc = np.concatenate([confidence_values(r["known_scores"])["max_cosine"] for r in runs])
    best = None
    for q in np.linspace(0.001, 0.05, 200):
        t_dr = float(np.quantile(dr, q, method="higher"))
        t_mc = float(np.quantile(mc, q, method="higher"))
        acceptance = float(np.mean((dr >= t_dr) & (mc >= t_mc)))
        if best is None or abs(acceptance - 0.95) < abs(best[0] - 0.95):
            best = (acceptance, t_dr, t_mc, float(q))
    return {"kind": "two_gate", "quantile": best[3], "t_dr": best[1], "t_mc": best[2],
            "clean_acceptance": best[0]}


def solve_confidence_only(runs: list[dict]) -> dict:
    dr = np.concatenate([confidence_values(r["known_scores"])["distance_ratio"] for r in runs])
    threshold = float(np.quantile(dr, 0.05, method="higher"))
    return {"kind": "confidence_only", "threshold": threshold,
            "t_dr": threshold, "clean_acceptance": float(np.mean(dr >= threshold))}


def accept_mask(scores: np.ndarray, rule: dict) -> np.ndarray:
    conf = confidence_values(scores)
    mask = conf["distance_ratio"] >= rule["t_dr"]
    if rule["kind"] == "two_gate":
        mask &= conf["max_cosine"] >= rule["t_mc"]
    return mask


def run_metrics(run: dict, attack: str | None, rule: dict) -> dict:
    """单个 inner run 的指标。两个 inner run 的 enrolled 类集合不同，不能拼接分数矩阵。"""
    classes = np.asarray(run["classes"], dtype=int)
    known_scores = run["known_scores"]
    known_true = run["known_true"]
    clean_pred = classes[known_scores.argmax(1)]
    clean_correct = clean_pred == known_true
    known_accept = accept_mask(known_scores, rule)
    c = float(np.mean(clean_correct & known_accept))
    u = float(np.mean(~accept_mask(run["unknown_scores"], rule)))
    if attack is None:
        ra = c
        asafer = 1.0
        direct = float(np.mean(clean_correct))
        rej = 1.0 - float(np.mean(known_accept))
        auroc = best_f1 = float("nan")
    else:
        scores = run["attacks"][attack]["scores"]
        adv_correct = classes[scores.argmax(1)] == known_true
        accept = accept_mask(scores, rule)
        ra = float(np.mean(adv_correct & accept))
        asafer = float(np.mean(adv_correct | ~accept))
        direct = float(np.mean(adv_correct))
        rej = float(1.0 - np.mean(accept))
        labels = np.concatenate([np.ones(len(scores)), np.zeros(len(known_scores))])
        values = -np.concatenate([confidence_values(scores)["max_cosine"],
                                  confidence_values(known_scores)["max_cosine"]])
        auroc = float(roc_auc_score(labels, values))
        order = np.argsort(-values)
        ranked = labels[order]
        tp = np.cumsum(ranked)
        fp = np.cumsum(1 - ranked)
        total = tp[-1]
        best_f1 = float((2 * tp / np.maximum(tp + fp + total, 1e-12)).max())
    final = np.where(known_accept, clean_pred, 0)
    return {
        "C": c, "RA": ra, "ASafeR": asafer, "A": 0.7 * ra + 0.3 * asafer, "U": u,
        "direct_RA": direct, "Rej": rej,
        "known_acceptance": float(np.mean(known_accept)),
        "detector_auroc": auroc, "detector_best_f1": best_f1,
        "identity_macro_f1": float(f1_score(known_true, final, labels=classes,
                                            average="macro", zero_division=0)),
        "identity_macro_f1_direct": float(f1_score(known_true, clean_pred, labels=classes,
                                                   average="macro", zero_division=0)),
    }


def evaluate(runs: list[dict], attack: str | None, rule: dict) -> dict:
    rows = [run_metrics(run, attack, rule) for run in runs]
    macro = {}
    for key in rows[0]:
        values = [row[key] for row in rows if not np.isnan(row[key])]
        macro[key] = float(np.mean(values)) if values else float("nan")
    macro["Score"] = official_score(macro["C"], macro["A"], macro["U"])
    return macro


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "inner"), default="inner")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("E40 要求 CUDA GPU")
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
    zband = torch.as_tensor(np.load(ZBAND_PATH), dtype=torch.float32, device=device)
    all_subjects = np.unique(subjects)
    outer_groups = np.array_split(
        np.random.default_rng(e36["nested_protocol"]["outer_seed"]).permutation(all_subjects), 5)
    if args.stage == "smoke":
        outer_groups = outer_groups[:1]
    shrinkage = float(e36["models"]["wccn_shrinkage"])
    floor = float(e36["models"]["wccn_eigenvalue_floor"])
    raw = load_all_raw(metadata, e36, device)
    results = {"status": "running", "experiment": "E40", "date": "2026-11-13",
               "stage": args.stage, "outer_folds": []}

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
            proto = fit_prototype(zband, fit_np, subjects, enrolled, shrinkage, floor)
            model = FrozenPrototype(proto, e36)
            known_subset = stratified_subset(metadata, known_np, 8)
            unknown_subset = stratified_subset(metadata, unknown_np, 8)
            known_raw = raw[torch.as_tensor(known_subset, device=device)]
            run = {
                "outer_fold": outer_fold, "inner_fold": inner_fold,
                "classes": enrolled, "known_true": subjects[known_subset],
                "known_scores": model.scores(known_raw),
                "unknown_scores": model.scores(raw[torch.as_tensor(unknown_subset, device=device)]),
                "attacks": {},
                "n_known": int(len(known_subset)), "n_unknown": int(len(unknown_subset)),
            }
            differentiable = DifferentiableWCCN(proto, e36)
            true = subjects[known_subset]
            target = np.asarray([
                enrolled[(int(np.flatnonzero(enrolled == label)[0]) + 1) % len(enrolled)]
                for label in true
            ], dtype=int)
            for attack_name in ATTACK_ORDER:
                base_name, snr_text = attack_name.rsplit("_snr", 1)
                snr = int(snr_text)
                seed = int(config["seed_base"] + 100000 * 1 + 10000 * outer_fold
                           + 1000 * inner_fold + 100 * ATTACK_CODES[base_name] + snr)
                scores, diagnostics, _ = raw_attack(
                    differentiable, known_raw, true, snr, base_name, seed,
                    {"attacks": {"pgd_steps": 20, "batch_size": 64, "temperature": 0.05,
                                 "seed_base": config["seed_base"]}},
                    target=target if base_name == "targeted_pgd" else None,
                )
                run["attacks"][attack_name] = {"scores": scores,
                                               "diagnostics": diagnostics}
            runs.append(run)
            print(f"outer={outer_fold} inner={inner_fold} 完成 "
                  f"known={run['n_known']} unknown={run['n_unknown']}", flush=True)
            del model, differentiable
            torch.cuda.empty_cache()
        rules = {"two_gate": solve_two_gate(runs),
                 "confidence_only": solve_confidence_only(runs)}
        tables = {}
        for rule_name, rule in rules.items():
            per_attack = {name: evaluate(runs, name, rule) for name in ATTACK_ORDER}
            macro = {key: float(np.mean([row[key] for row in per_attack.values()]))
                     for key in per_attack[ATTACK_ORDER[0]]}
            macro["Score"] = official_score(macro["C"], macro["A"], macro["U"])
            clean = evaluate(runs, None, rule)
            tables[rule_name] = {"per_attack": per_attack, "macro": macro, "clean": clean}
        results["outer_folds"].append({"outer_fold": outer_fold, "rules": rules,
                                       "tables": tables})
        write_json(RESULTS_PATH, results)
        for rule_name in rules:
            macro = tables[rule_name]["macro"]
            print(f"E40 outer={outer_fold} {rule_name}: Score={macro['Score']:.2f} "
                  f"RA={macro['RA']:.4f} ASafeR={macro['ASafeR']:.4f} A={macro['A']:.4f} "
                  f"C={macro['C']:.4f} U={macro['U']:.4f} "
                  f"detAUROC={macro['detector_auroc']:.4f} detF1={macro['detector_best_f1']:.4f}",
                  flush=True)

    summary = {}
    for rule_name in ("two_gate", "confidence_only"):
        macro = {key: float(np.mean([f["tables"][rule_name]["macro"][key]
                                    for f in results["outer_folds"]]))
                 for key in results["outer_folds"][0]["tables"][rule_name]["macro"]}
        clean = {key: float(np.mean([f["tables"][rule_name]["clean"][key]
                                    for f in results["outer_folds"]]))
                 for key in results["outer_folds"][0]["tables"][rule_name]["clean"]}
        summary[rule_name] = {"macro": macro, "clean": clean}
    results["summary"] = summary
    results["reference_e36c_outer"] = REFERENCE
    results["delta_vs_e36c"] = {
        key: summary["two_gate"]["macro"][key] - REFERENCE[key]
        for key in ("C", "RA", "ASafeR", "A", "U", "Score")
    }
    results["status"] = "complete"
    results["elapsed_seconds"] = time.time() - started
    write_json(RESULTS_PATH, results)
    for rule_name in ("two_gate", "confidence_only"):
        macro = summary[rule_name]["macro"]
        print(f"E40 MACRO {rule_name}: Score={macro['Score']:.2f} RA={macro['RA']:.4f} "
              f"ASafeR={macro['ASafeR']:.4f} A={macro['A']:.4f} C={macro['C']:.4f} "
              f"U={macro['U']:.4f} detAUROC={macro['detector_auroc']:.4f} "
              f"detF1={macro['detector_best_f1']:.4f}", flush=True)
    print(f"完成：{RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
