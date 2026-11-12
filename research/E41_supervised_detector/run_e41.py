"""E41：fit 折内自训练的监督攻击检测器 + 拒识。

与 E40 的唯一区别：第二道门控的检测器不是无监督的 max_cosine，而是用
fit 折 clean + fit 折对抗增广样本训练的 logistic 回归。
门控方向仍为「distance_ratio 高 且 detector 低」，阈值在 inner clean known
上按共享分位求解，使联合接受率为 0.95。
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
sys.path.insert(0, str(ROOT / "research" / "E40_attack_rejection"))

from run_e36 import ZBAND_PATH, detrend_zband, gpu_logspd, load_config as load_e36_config, train_metadata  # noqa: E402
from run_attacks import (  # noqa: E402
    ATTACK_ORDER,
    DifferentiableWCCN,
    load_all_raw,
    raw_attack,
    stratified_subset,
)
from run_e40 import FrozenPrototype, fit_prototype, confidence_values  # noqa: E402

CONFIG_PATH = OUT / "config.json"
RESULTS_PATH = OUT / "results.json"
ATTACK_CODES = {"untargeted_fgsm": 1, "untargeted_pgd": 2, "targeted_pgd": 3}
TRAIN_ATTACKS = [
    ("untargeted_fgsm", 30), ("untargeted_pgd", 30), ("untargeted_pgd", 20),
    ("targeted_pgd", 30), ("targeted_pgd", 20),
]
REFERENCE = {"E36C": {"ASafeR": 0.6818, "Score": 59.78},
             "E40": {"ASafeR": 0.7082, "Score": 59.14}}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def official_score(c: float, a: float, u: float) -> float:
    return float(100 * c**0.40 * a**0.35 * u**0.25) if min(c, a, u) > 0 else 0.0


def trial_features(raw: torch.Tensor, model: FrozenPrototype, e36: dict) -> np.ndarray:
    """8 维、逐 trial 的特征；不使用任何标签。"""
    with torch.no_grad():
        features = detrend_zband(gpu_logspd(raw, e36), e36)
        whitened = features @ model.whitening.to(features.dtype)
        normalized = whitened / whitened.norm(dim=1, keepdim=True).clamp_min(1e-12)
        scores = normalized @ model.prototypes.to(features.dtype).T
    ordered = torch.sort(scores, dim=1, descending=True).values
    maximum = ordered[:, 0]
    second = ordered[:, 1]
    distance_ratio = (1.0 - second) / (1.0 - maximum).clamp_min(1e-6)
    means = model.whitened_means.to(features.dtype)
    distances = ((whitened[:, None, :] - means[None, :, :]) ** 2).mean(-1)
    spectrum = torch.fft.rfft(raw - raw.mean(-1, keepdim=True), dim=-1).abs() ** 2
    freqs = torch.linspace(0, 150, spectrum.shape[-1], device=raw.device)
    total = spectrum.sum(-1).clamp_min(1e-12)
    high_ratio = spectrum[..., freqs >= 20].sum(-1) / total
    flatness = torch.exp(torch.log(spectrum.clamp_min(1e-12)).mean(-1)) / \
        spectrum.mean(-1).clamp_min(1e-12)
    return np.stack([
        maximum.cpu().numpy(), second.cpu().numpy(), (maximum - second).cpu().numpy(),
        distance_ratio.cpu().numpy(), whitened.norm(dim=1).cpu().numpy(),
        distances.min(1).values.cpu().numpy(), flatness.mean(1).cpu().numpy(),
        high_ratio.mean(1).cpu().numpy(),
    ], axis=1)


class Detector:
    def __init__(self, model: FrozenPrototype, e36: dict):
        self.model = model
        self.e36 = e36
        self.pipeline = None

    def fit(self, clean_raw: torch.Tensor, attacked_raw: torch.Tensor) -> "Detector":
        features = np.concatenate([
            trial_features(clean_raw, self.model, self.e36),
            trial_features(attacked_raw, self.model, self.e36),
        ])
        labels = np.concatenate([np.zeros(len(clean_raw)), np.ones(len(attacked_raw))])
        self.pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ).fit(features, labels)
        return self

    def score(self, raw: torch.Tensor) -> np.ndarray:
        return self.pipeline.decision_function(trial_features(raw, self.model, self.e36))


def build_train_attacks(proto, raw, subset, true, enrolled, e36, config,
                        outer_fold, inner_fold):
    raw_subset = raw[torch.as_tensor(subset, device=raw.device)]
    attacked = []
    differentiable = DifferentiableWCCN(proto, e36)
    target = np.asarray([
        enrolled[(int(np.flatnonzero(enrolled == label)[0]) + 1) % len(enrolled)]
        for label in true
    ], dtype=int)
    for base_name, snr in TRAIN_ATTACKS:
        seed = int(config["seed_base"] + 300000 + 10000 * outer_fold + 1000 * inner_fold
                   + 100 * ATTACK_CODES[base_name] + snr)
        _, _, waveforms = raw_attack(
            differentiable, raw_subset, true, snr, base_name, seed,
            {"attacks": {"pgd_steps": 20, "batch_size": 64, "temperature": 0.05,
                         "seed_base": config["seed_base"]}},
            target=target if base_name == "targeted_pgd" else None,
            return_waveforms=True,
        )
        attacked.append(waveforms.float())
    return raw_subset, torch.cat(attacked, dim=0)


def solve_two_gate(dr: np.ndarray, det: np.ndarray) -> dict:
    best = None
    for q in np.linspace(0.001, 0.05, 200):
        t_dr = float(np.quantile(dr, q, method="higher"))
        t_det = float(np.quantile(det, 1.0 - q, method="lower"))
        acceptance = float(np.mean((dr >= t_dr) & (det <= t_det)))
        if best is None or abs(acceptance - 0.95) < abs(best[0] - 0.95):
            best = (acceptance, t_dr, t_det, float(q))
    return {"quantile": best[3], "t_dr": best[1], "t_det": best[2],
            "clean_acceptance": best[0]}


def run_metrics(run: dict, attack, t_dr: float, t_det: float) -> dict:
    """单 run 指标；clean 与 unknown 都用该 run 自己的数组，阈值来自池化求解。"""
    classes = np.asarray(run["classes"], dtype=int)
    dr_known = confidence_values(run["known_scores"])["distance_ratio"]
    det_known = run["known_detector"]
    clean_pred = classes[run["known_scores"].argmax(1)]
    clean_correct = clean_pred == run["known_true"]
    known_accept = (dr_known >= t_dr) & (det_known <= t_det)
    c = float(np.mean(clean_correct & known_accept))
    dr_unknown = confidence_values(run["unknown_scores"])["distance_ratio"]
    unknown_accept = (dr_unknown >= t_dr) & (run["unknown_detector"] <= t_det)
    u = float(np.mean(~unknown_accept))
    if attack is None:
        ra, asafer = c, 1.0
        direct = float(np.mean(clean_correct))
        rej = 1.0 - float(np.mean(known_accept))
        auroc = best_f1 = float("nan")
    else:
        scores = run["attacks"][attack]["scores"]
        dr = confidence_values(scores)["distance_ratio"]
        det = run["attacks"][attack]["detector"]
        adv_correct = classes[scores.argmax(1)] == run["known_true"]
        accept = (dr >= t_dr) & (det <= t_det)
        ra = float(np.mean(adv_correct & accept))
        asafer = float(np.mean(adv_correct | ~accept))
        direct = float(np.mean(adv_correct))
        rej = float(1.0 - np.mean(accept))
        labels = np.concatenate([np.ones(len(det)), np.zeros(len(det_known))])
        values = np.concatenate([det, det_known])
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
        "identity_macro_f1": float(f1_score(run["known_true"], final, labels=classes,
                                            average="macro", zero_division=0)),
        "identity_macro_f1_direct": float(f1_score(run["known_true"], clean_pred,
                                                   labels=classes, average="macro",
                                                   zero_division=0)),
    }


def evaluate(runs: list[dict], t_dr: float, t_det: float, attack) -> dict:
    rows = [run_metrics(run, attack, t_dr, t_det) for run in runs]
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
        raise SystemExit("E41 要求 CUDA GPU")
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
    results = {"status": "running", "experiment": "E41", "date": "2026-11-13",
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
            train_subset = stratified_subset(metadata, fit_np, 8)
            detector = Detector(model, e36)
            clean_raw, attacked_raw = build_train_attacks(
                proto, raw, train_subset, subjects[train_subset], enrolled,
                e36, config, outer_fold, inner_fold,
            )
            detector.fit(clean_raw, attacked_raw)
            known_subset = stratified_subset(metadata, known_np, 8)
            unknown_subset = stratified_subset(metadata, unknown_np, 8)
            known_raw = raw[torch.as_tensor(known_subset, device=device)]
            unknown_raw = raw[torch.as_tensor(unknown_subset, device=device)]
            known_scores = model.scores(known_raw)
            unknown_scores = model.scores(unknown_raw)
            run = {
                "outer_fold": outer_fold, "inner_fold": inner_fold, "classes": enrolled,
                "known_true": subjects[known_subset], "known_scores": known_scores,
                "unknown_scores": unknown_scores,
                "known_detector": detector.score(known_raw),
                "unknown_detector": detector.score(unknown_raw),
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
                seed = int(config["seed_base"] + 100000 + 10000 * outer_fold
                           + 1000 * inner_fold + 100 * ATTACK_CODES[base_name] + snr)
                scores, diagnostics, waveforms = raw_attack(
                    differentiable, known_raw, true, snr, base_name, seed,
                    {"attacks": {"pgd_steps": 20, "batch_size": 64, "temperature": 0.05,
                                 "seed_base": config["seed_base"]}},
                    target=target if base_name == "targeted_pgd" else None,
                    return_waveforms=True,
                )
                run["attacks"][attack_name] = {
                    "scores": scores, "detector": detector.score(waveforms),
                    "diagnostics": diagnostics,
                }
            runs.append(run)
            print(f"outer={outer_fold} inner={inner_fold} 完成 "
                  f"known={run['n_known']} unknown={run['n_unknown']}", flush=True)
            del model, differentiable
            torch.cuda.empty_cache()
        dr_known = np.concatenate([confidence_values(r["known_scores"])["distance_ratio"]
                                   for r in runs])
        det_known = np.concatenate([r["known_detector"] for r in runs])
        solved = solve_two_gate(dr_known, det_known)
        per_attack = {name: evaluate(runs, solved["t_dr"], solved["t_det"], name)
                      for name in ATTACK_ORDER}
        macro = {key: float(np.mean([row[key] for row in per_attack.values()]))
                 for key in per_attack[ATTACK_ORDER[0]]}
        macro["Score"] = official_score(macro["C"], macro["A"], macro["U"])
        clean = evaluate(runs, solved["t_dr"], solved["t_det"], None)
        results["outer_folds"].append({"outer_fold": outer_fold, "solve": solved,
                                       "per_attack": per_attack, "macro": macro,
                                       "clean": clean})
        write_json(RESULTS_PATH, results)
        print(f"E41 outer={outer_fold}: Score={macro['Score']:.2f} RA={macro['RA']:.4f} "
              f"ASafeR={macro['ASafeR']:.4f} C={macro['C']:.4f} U={macro['U']:.4f} "
              f"detAUROC={macro['detector_auroc']:.4f} detF1={macro['detector_best_f1']:.4f} "
              f"q={solved['quantile']:.3f}", flush=True)

    macro = {key: float(np.mean([f["macro"][key] for f in results["outer_folds"]]))
             for key in results["outer_folds"][0]["macro"]}
    clean = {key: float(np.mean([f["clean"][key] for f in results["outer_folds"]]))
             for key in results["outer_folds"][0]["clean"]}
    results["macro_fold_mean"] = macro
    results["clean_fold_mean"] = clean
    results["reference"] = REFERENCE
    results["delta"] = {
        "vs_E40_ASafeR": macro["ASafeR"] - REFERENCE["E40"]["ASafeR"],
        "vs_E40_Score": macro["Score"] - REFERENCE["E40"]["Score"],
        "vs_E36C_ASafeR": macro["ASafeR"] - REFERENCE["E36C"]["ASafeR"],
        "vs_E36C_Score": macro["Score"] - REFERENCE["E36C"]["Score"],
    }
    results["status"] = "complete"
    results["elapsed_seconds"] = time.time() - started
    write_json(RESULTS_PATH, results)
    print(f"E41 MACRO: Score={macro['Score']:.2f} RA={macro['RA']:.4f} "
          f"ASafeR={macro['ASafeR']:.4f} A={macro['A']:.4f} C={macro['C']:.4f} "
          f"U={macro['U']:.4f} detAUROC={macro['detector_auroc']:.4f} "
          f"detF1={macro['detector_best_f1']:.4f}", flush=True)
    print(f"完成：{RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
