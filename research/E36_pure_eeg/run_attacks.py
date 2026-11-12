"""E36C：Train-only 内层正则消融、原始 EEG 攻击校准与冻结外层评估。"""
from __future__ import annotations

import argparse
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from run_e36 import (
    CONFIG_PATH,
    OUT,
    RAW_PATH,
    ROOT,
    ZBAND_PATH,
    TorchPrototype,
    confidence_values,
    detrend_zband,
    gpu_logspd,
    load_config,
    load_train_matrix,
    sha256,
    train_metadata,
    write_json,
)

ATTACK_CONFIG_PATH = OUT / "attack_config.json"
RESULTS_PATH = OUT / "attack_results.json"
CANDIDATE = "detrend_zband_cov5_wccn_prototype"
ATTACK_CODES = {"untargeted_fgsm": 1, "untargeted_pgd": 2, "targeted_pgd": 3}
ATTACK_ORDER = [
    "untargeted_fgsm_snr30", "untargeted_fgsm_snr20",
    "untargeted_pgd_snr30", "untargeted_pgd_snr20",
    "targeted_pgd_snr30", "targeted_pgd_snr20",
]
COMPONENT_KEYS = (
    "C", "RA", "ASafeR", "A", "identity_macro_f1",
    "clean_identity_macro_f1", "unknown_auroc", "unknown_f1", "U",
    "Score", "known_acceptance", "direct_RA",
)


def load_attack_config() -> dict:
    return json.loads(ATTACK_CONFIG_PATH.read_text(encoding="utf-8"))


def official_score(c: float, a: float, u: float) -> float:
    return float(100 * c**0.40 * a**0.35 * u**0.25) if min(c, a, u) > 0 else 0.0


def stratified_subset(metadata: pd.DataFrame, indices: np.ndarray, per_group: int) -> np.ndarray:
    frame = metadata.iloc[indices][["subject", "task", "trial", "path"]].copy()
    frame["matrix_index"] = indices
    selected = (
        frame.sort_values(["subject", "task", "trial", "path"])
        .groupby(["subject", "task"], sort=True, as_index=False)
        .head(per_group)
    )
    result = np.sort(selected["matrix_index"].to_numpy(dtype=int))
    expected_groups = frame.groupby(["subject", "task"]).ngroups
    if selected.groupby(["subject", "task"]).ngroups != expected_groups:
        raise ValueError("分层子集未覆盖全部 subject/task 组")
    return result


def load_all_raw(metadata: pd.DataFrame, config: dict, device: torch.device) -> torch.Tensor:
    started = time.time()
    paths = [ROOT / config["data"]["train_root"] / path for path in metadata["path"]]
    output = torch.empty(
        (len(paths), config["data"]["channels"], config["data"]["samples_per_trial"]),
        dtype=torch.float32, device=device,
    )
    batch_size = int(config["features"]["io_batch_size"])
    with ThreadPoolExecutor(max_workers=int(config["features"]["io_workers"])) as pool:
        for start in range(0, len(paths), batch_size):
            arrays = list(pool.map(load_train_matrix, paths[start : start + batch_size]))
            output[start : start + len(arrays)] = torch.as_tensor(
                np.stack(arrays), dtype=torch.float32, device=device
            )
    print(f"Train raw loaded: {tuple(output.shape)} in {time.time() - started:.1f}s", flush=True)
    return output


def fit_clean_wccn(
    zband: torch.Tensor, fit_np: np.ndarray, classes: np.ndarray,
    shrinkage: float, config: dict,
) -> TorchPrototype:
    indices = torch.as_tensor(fit_np, device=zband.device)
    return TorchPrototype(
        zband[indices], config["_subjects"][fit_np], classes, True,
        shrinkage, config["models"]["wccn_eigenvalue_floor"],
    )


def predict_wccn(
    model: TorchPrototype, zband: torch.Tensor, indices: np.ndarray,
    config: dict, batch_size: int | None = None,
) -> np.ndarray:
    if len(indices) == 0:
        return np.empty((0, len(model.classes)), dtype=np.float32)
    batch_size = batch_size or int(config["models"]["prediction_batch_size"])
    output = np.empty((len(indices), len(model.classes)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = torch.as_tensor(indices[start : start + batch_size], device=zband.device)
            output[start : start + len(selected)] = model(zband[selected]).cpu().numpy()
    return output


class DifferentiableWCCN:
    def __init__(self, model: TorchPrototype, config: dict):
        self.config = config
        self.classes = np.asarray(model.classes, dtype=int)
        self.whitening = model.whitening.detach().double()
        self.prototypes = model.prototypes.detach().double()

    def features(self, raw: torch.Tensor) -> torch.Tensor:
        return detrend_zband(gpu_logspd(raw, self.config), self.config)

    def scores_from_features(self, features: torch.Tensor) -> torch.Tensor:
        whitened = features @ self.whitening
        normalized = whitened / whitened.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return normalized @ self.prototypes.T

    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        return self.scores_from_features(self.features(raw))


def attack_seed(
    attack_config: dict, stage_code: int, outer_fold: int, inner_fold: int,
    attack_name: str, snr_db: int,
) -> int:
    return int(
        attack_config["attacks"]["seed_base"] + 100000 * stage_code
        + 10000 * outer_fold + 1000 * inner_fold
        + 100 * ATTACK_CODES[attack_name] + snr_db
    )


def raw_attack(
    model: DifferentiableWCCN, raw: torch.Tensor, true: np.ndarray,
    snr_db: int, attack_name: str, seed: int, attack_config: dict,
    target: np.ndarray | None = None, return_waveforms: bool = False,
) -> tuple[np.ndarray, dict, torch.Tensor | None]:
    cfg = attack_config["attacks"]
    targeted = attack_name == "targeted_pgd"
    steps = 1 if attack_name == "untargeted_fgsm" else int(cfg["pgd_steps"])
    batch_size = int(cfg["batch_size"])
    score_output = np.empty((len(raw), len(model.classes)), dtype=np.float32)
    realized_snr = []
    radius_errors = []
    waveform_output = torch.empty_like(raw, dtype=torch.float32) if return_waveforms else None
    class_to_index = {int(label): index for index, label in enumerate(model.classes)}
    for start in range(0, len(raw), batch_size):
        stop = min(start + batch_size, len(raw))
        clean = raw[start:stop].detach().double()
        centered = clean - clean.mean(dim=-1, keepdim=True)
        signal_norm = centered.reshape(len(clean), -1).norm(dim=1).clamp_min(1e-12)
        radius = signal_norm / (10.0 ** (float(snr_db) / 20.0))
        delta = torch.zeros_like(clean)
        generator = torch.Generator(device=clean.device)
        generator.manual_seed(int(seed + start))
        if steps > 1:
            delta = torch.randn(clean.shape, generator=generator, device=clean.device, dtype=clean.dtype)
            delta = delta / delta.reshape(len(clean), -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
            random_radius = torch.rand(len(clean), generator=generator, device=clean.device, dtype=clean.dtype)
            delta = delta * radius.view(-1, 1, 1) * random_radius.view(-1, 1, 1)
        labels = target if targeted else true
        batch_labels = labels[start:stop]
        label_indices = torch.as_tensor(
            [class_to_index[int(label)] for label in batch_labels],
            dtype=torch.long, device=clean.device,
        )
        for _ in range(steps):
            delta = delta.detach().requires_grad_(True)
            scores = model(clean + delta)
            loss = F.cross_entropy(scores / float(cfg["temperature"]), label_indices)
            objective = -loss if targeted else loss
            gradient = torch.autograd.grad(objective, delta)[0]
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("E36 原始波形攻击出现非有限梯度")
            direction = gradient / gradient.reshape(len(gradient), -1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
            step_size = radius if steps == 1 else 2.0 * radius / steps
            delta = delta.detach() + step_size.view(-1, 1, 1) * direction
            delta_norm = delta.reshape(len(delta), -1).norm(dim=1).clamp_min(1e-12)
            factor = torch.minimum(torch.ones_like(delta_norm), radius / delta_norm)
            delta = delta * factor.view(-1, 1, 1)
        with torch.no_grad():
            final_scores = model(clean + delta)
            if not torch.isfinite(final_scores).all() or not torch.isfinite(delta).all():
                raise FloatingPointError("E36 原始波形攻击出现非有限输出")
            score_output[start:stop] = final_scores.float().cpu().numpy()
            final_norm = delta.reshape(len(delta), -1).norm(dim=1)
            realized_snr.extend((20 * torch.log10(signal_norm / final_norm.clamp_min(1e-12))).cpu().tolist())
            radius_errors.extend(((final_norm - radius).abs() / radius.clamp_min(1e-12)).cpu().tolist())
            if waveform_output is not None:
                waveform_output[start:stop] = (clean + delta).float()
    diagnostics = {
        "n": int(len(raw)), "snr_db_requested": int(snr_db), "steps": int(steps),
        "realized_snr_db_mean": float(np.mean(realized_snr)),
        "realized_snr_db_min": float(np.min(realized_snr)),
        "radius_relative_error_max": float(np.max(radius_errors)),
        "finite_gradients_and_scores": True,
    }
    return score_output, diagnostics, waveform_output


def fit_model(
    method: str, raw: torch.Tensor, zband: torch.Tensor, fit_np: np.ndarray,
    classes: np.ndarray, shrinkage: float, config: dict, attack_config: dict,
    metadata: pd.DataFrame, outer_fold: int, inner_fold: int,
) -> tuple[TorchPrototype, dict]:
    model = fit_clean_wccn(zband, fit_np, classes, shrinkage, config)
    info = {"method": method, "n_clean_fit": int(len(fit_np))}
    if method == "baseline":
        return model, info
    if method != "pgd_fgsm_augmentation":
        raise ValueError(method)
    per_group = int(attack_config["subset"]["training_augmentation_trials_per_subject_task"])
    aug_np = stratified_subset(metadata, fit_np, per_group)
    attack_model = DifferentiableWCCN(model, config)
    features = []
    diagnostics = {}
    for attack_name, snr_db in (("untargeted_pgd", 20), ("untargeted_fgsm", 30)):
        seed = attack_seed(attack_config, 3, outer_fold, inner_fold, attack_name, snr_db)
        _, diagnostic, waveforms = raw_attack(
            attack_model, raw[torch.as_tensor(aug_np, device=raw.device)],
            config["_subjects"][aug_np], snr_db, attack_name, seed,
            attack_config, return_waveforms=True,
        )
        with torch.no_grad():
            features.append(attack_model.features(waveforms.double()).float())
        diagnostics[f"{attack_name}_snr{snr_db}"] = diagnostic
    fit_indices = torch.as_tensor(fit_np, device=zband.device)
    augmented_zband = torch.cat([zband[fit_indices], *features], dim=0)
    augmented_y = np.concatenate([config["_subjects"][fit_np], config["_subjects"][aug_np], config["_subjects"][aug_np]])
    model = TorchPrototype(
        augmented_zband, augmented_y, classes, True, shrinkage,
        config["models"]["wccn_eigenvalue_floor"],
    )
    info.update({
        "n_augmentation_source": int(len(aug_np)), "n_total_fit": int(len(augmented_y)),
        "augmentation_diagnostics": diagnostics,
    })
    return model, info


def make_clean_run(
    model: TorchPrototype, zband: torch.Tensor, known_np: np.ndarray,
    unknown_np: np.ndarray, classes: np.ndarray, config: dict,
) -> dict:
    return {
        "known_true": config["_subjects"][known_np],
        "known_scores": predict_wccn(model, zband, known_np, config),
        "unknown_scores": predict_wccn(model, zband, unknown_np, config),
        "classes": np.asarray(classes, dtype=int),
    }


def make_attack_run(
    model: TorchPrototype, raw: torch.Tensor, zband: torch.Tensor,
    known_np: np.ndarray, unknown_np: np.ndarray, classes: np.ndarray,
    config: dict, attack_config: dict, metadata: pd.DataFrame,
    stage_code: int, outer_fold: int, inner_fold: int,
) -> dict:
    per_group = int(attack_config["subset"]["trials_per_subject_task"])
    known_subset = stratified_subset(metadata, known_np, per_group)
    unknown_subset = stratified_subset(metadata, unknown_np, per_group)
    run = make_clean_run(model, zband, known_subset, unknown_subset, classes, config)
    run["clean_full"] = make_clean_run(model, zband, known_np, unknown_np, classes, config)
    run["known_indices"] = known_subset
    run["unknown_indices"] = unknown_subset
    run["attacks"] = {}
    attack_model = DifferentiableWCCN(model, config)
    raw_known = raw[torch.as_tensor(known_subset, device=raw.device)]
    true = config["_subjects"][known_subset]
    target = np.asarray([
        classes[(int(np.flatnonzero(classes == label)[0]) + 1) % len(classes)] for label in true
    ], dtype=int)
    for attack_name in ATTACK_ORDER:
        base_name, snr_text = attack_name.rsplit("_snr", 1)
        snr_db = int(snr_text)
        seed = attack_seed(attack_config, stage_code, outer_fold, inner_fold, base_name, snr_db)
        scores, diagnostics, _ = raw_attack(
            attack_model, raw_known, true, snr_db, base_name, seed, attack_config,
            target=target if base_name == "targeted_pgd" else None,
        )
        run["attacks"][attack_name] = {"scores": scores, "diagnostics": diagnostics}
        if base_name == "targeted_pgd":
            run["attacks"][attack_name]["target"] = target
        print(
            f"  {stage_code=} outer={outer_fold} inner={inner_fold} {attack_name} "
            f"N={len(true)} SNR={diagnostics['realized_snr_db_mean']:.2f}", flush=True,
        )
    return run


def direct_predictions(run: dict, key: str) -> np.ndarray:
    return run["classes"][run[key].argmax(axis=1)]


def concatenate_run_values(runs: list[dict], accessor) -> np.ndarray:
    return np.concatenate([accessor(run) for run in runs])


def clean_component(runs: list[dict], score_name: str, threshold: float) -> dict:
    known = concatenate_run_values(runs, lambda r: confidence_values(r["known_scores"], CANDIDATE)[score_name])
    unknown = concatenate_run_values(runs, lambda r: confidence_values(r["unknown_scores"], CANDIDATE)[score_name])
    correct = concatenate_run_values(runs, lambda r: direct_predictions(r, "known_scores") == r["known_true"])
    accepted = known >= threshold
    c = float(np.mean(correct & accepted))
    u = float(np.mean(unknown < threshold))
    return {
        "C": c, "U": u, "known_acceptance": float(np.mean(accepted)),
        "C_U_proxy_not_official": float(100 * c**0.4 * u**0.25) if min(c, u) > 0 else 0.0,
    }


def select_clean_point(runs: list[dict], attack_config: dict) -> dict:
    candidates = []
    for score_index, score_name in enumerate(attack_config["threshold"]["score_candidates"]):
        known = concatenate_run_values(runs, lambda r: confidence_values(r["known_scores"], CANDIDATE)[score_name])
        thresholds = np.unique(np.quantile(known, np.linspace(0, 0.05, 101)))
        rows = []
        for threshold in thresholds:
            row = clean_component(runs, score_name, float(threshold))
            if row["known_acceptance"] >= 0.95 - 1e-12:
                rows.append((row, float(threshold)))
        work, threshold = max(rows, key=lambda item: (
            item[0]["C_U_proxy_not_official"], item[0]["C"], item[0]["U"],
        ))
        unknown = concatenate_run_values(runs, lambda r: confidence_values(r["unknown_scores"], CANDIDATE)[score_name])
        labels = np.concatenate([np.ones(len(known)), np.zeros(len(unknown))])
        direct = concatenate_run_values(runs, lambda r: direct_predictions(r, "known_scores"))
        true = concatenate_run_values(runs, lambda r: r["known_true"])
        enrolled = np.unique(np.concatenate([r["classes"] for r in runs]))
        q05_threshold = float(np.quantile(known, 0.05))
        candidates.append({
            "score_name": score_name, "threshold": threshold, "q05_threshold": q05_threshold,
            "direct_C": float(np.mean(direct == true)),
            "identity_macro_f1_direct": float(f1_score(true, direct, labels=enrolled, average="macro", zero_division=0)),
            "unknown_auroc": float(roc_auc_score(labels, np.concatenate([known, unknown]))),
            "U_at_q05": clean_component(runs, score_name, q05_threshold)["U"],
            "work": work, "score_index": score_index,
        })
    selected = max(candidates, key=lambda row: (
        row["work"]["C_U_proxy_not_official"], row["direct_C"],
        row["identity_macro_f1_direct"], row["unknown_auroc"], row["U_at_q05"],
        -row["score_index"],
    ))
    return {key: value for key, value in selected.items() if key != "score_index"}


def components_at_threshold(
    runs: list[dict], attack_name: str, score_name: str, threshold: float,
) -> dict:
    clean_conf = concatenate_run_values(runs, lambda r: confidence_values(r["known_scores"], CANDIDATE)[score_name])
    clean_correct = concatenate_run_values(runs, lambda r: direct_predictions(r, "known_scores") == r["known_true"])
    unknown_conf = concatenate_run_values(runs, lambda r: confidence_values(r["unknown_scores"], CANDIDATE)[score_name])
    adv_conf = concatenate_run_values(runs, lambda r: confidence_values(r["attacks"][attack_name]["scores"], CANDIDATE)[score_name])
    adv_correct = np.concatenate([
        r["classes"][r["attacks"][attack_name]["scores"].argmax(axis=1)] == r["known_true"]
        for r in runs
    ])
    clean_accepted = clean_conf >= threshold
    adv_accepted = adv_conf >= threshold
    c = float(np.mean(clean_correct & clean_accepted))
    ra = float(np.mean(adv_correct & adv_accepted))
    direct_ra = float(np.mean(adv_correct))
    asafer = float(np.mean(adv_correct | ~adv_accepted))
    a = 0.7 * ra + 0.3 * asafer
    u = float(np.mean(unknown_conf < threshold))
    return {
        "C": c, "RA": ra, "ASafeR": asafer, "A": a, "U": u,
        "direct_RA": direct_ra, "known_acceptance": float(np.mean(clean_accepted)),
        "Score": official_score(c, a, u),
    }


def macro_from_attacks(per_attack: dict[str, dict]) -> dict:
    result = {
        key: float(np.mean([row[key] for row in per_attack.values()])) for key in COMPONENT_KEYS
    }
    result["Score"] = official_score(result["C"], result["A"], result["U"])
    result["attack_score_mean"] = float(np.mean([row["Score"] for row in per_attack.values()]))
    result["N_clean_per_attack_sum"] = int(sum(row["N_clean"] for row in per_attack.values()))
    result["N_adversarial_total"] = int(sum(row["N_adversarial"] for row in per_attack.values()))
    result["N_unknown_per_attack_sum"] = int(sum(row["N_unknown"] for row in per_attack.values()))
    targeted = [row for name, row in per_attack.items() if name.startswith("targeted")]
    result["targeted_direct_false_authentication"] = float(np.mean([
        row["targeted_direct_false_authentication"] for row in targeted
    ]))
    result["targeted_false_authentication"] = float(np.mean([
        row["targeted_false_authentication"] for row in targeted
    ]))
    result["all_rejection"] = bool(all(row["all_rejection"] for row in per_attack.values()))
    return result


def evaluate_attack(
    runs: list[dict], attack_name: str, score_name: str, threshold: float,
) -> dict:
    result = components_at_threshold(runs, attack_name, score_name, threshold)
    true = concatenate_run_values(runs, lambda r: r["known_true"])
    direct = np.concatenate([r["classes"][r["attacks"][attack_name]["scores"].argmax(axis=1)] for r in runs])
    confidence = concatenate_run_values(runs, lambda r: confidence_values(r["attacks"][attack_name]["scores"], CANDIDATE)[score_name])
    pred = direct.copy()
    pred[confidence < threshold] = 0
    clean_true = true
    clean_direct = concatenate_run_values(runs, lambda r: direct_predictions(r, "known_scores"))
    clean_conf = concatenate_run_values(runs, lambda r: confidence_values(r["known_scores"], CANDIDATE)[score_name])
    clean_pred = clean_direct.copy()
    clean_pred[clean_conf < threshold] = 0
    unknown_direct = concatenate_run_values(runs, lambda r: direct_predictions(r, "unknown_scores"))
    unknown_conf = concatenate_run_values(runs, lambda r: confidence_values(r["unknown_scores"], CANDIDATE)[score_name])
    unknown_pred = unknown_direct.copy()
    unknown_pred[unknown_conf < threshold] = 0
    labels = np.unique(np.concatenate([r["classes"] for r in runs]))
    detection_true = np.concatenate([np.ones(len(pred)), np.zeros(len(unknown_pred))])
    accepted = np.concatenate([pred != 0, unknown_pred != 0])
    result.update({
        "identity_macro_f1": float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)),
        "clean_identity_macro_f1": float(f1_score(clean_true, clean_pred, labels=labels, average="macro", zero_division=0)),
        "unknown_auroc": float(roc_auc_score(detection_true, np.concatenate([confidence, unknown_conf]))),
        "unknown_f1": float(f1_score(detection_true, accepted, pos_label=0, zero_division=0)),
        "N_clean": int(len(clean_pred)), "N_adversarial": int(len(pred)), "N_unknown": int(len(unknown_pred)),
        "all_rejection": bool(np.all(pred == 0)),
    })
    if attack_name.startswith("targeted"):
        target = concatenate_run_values(runs, lambda r: r["attacks"][attack_name]["target"])
        result["targeted_direct_false_authentication"] = float(np.mean(direct == target))
        result["targeted_false_authentication"] = float(np.mean(pred == target))
    return result


def select_attack_point(runs: list[dict], attack_config: dict) -> dict:
    score_candidates = []
    for score_index, score_name in enumerate(attack_config["threshold"]["score_candidates"]):
        known = concatenate_run_values(runs, lambda r: confidence_values(r["known_scores"], CANDIDATE)[score_name])
        thresholds = np.unique(np.quantile(known, np.linspace(0, 0.05, 101)))
        rows = []
        for threshold in thresholds:
            per_attack = {
                name: components_at_threshold(runs, name, score_name, float(threshold))
                for name in ATTACK_ORDER
            }
            macro = {key: float(np.mean([row[key] for row in per_attack.values()])) for key in ("C", "RA", "ASafeR", "A", "U")}
            macro["Score"] = official_score(macro["C"], macro["A"], macro["U"])
            macro["attack_score_mean"] = float(np.mean([row["Score"] for row in per_attack.values()]))
            known_acceptance = float(np.mean([row["known_acceptance"] for row in per_attack.values()]))
            if known_acceptance >= 0.95 - 1e-12:
                rows.append((float(threshold), macro, known_acceptance))
        threshold, macro, _ = max(rows, key=lambda item: (
            item[1]["Score"], item[1]["attack_score_mean"], item[1]["RA"],
            item[1]["ASafeR"], item[1]["C"], item[1]["U"],
        ))
        score_candidates.append((score_index, score_name, threshold, macro))
    _, score_name, threshold, _ = max(score_candidates, key=lambda item: (
        item[3]["Score"], item[3]["attack_score_mean"], item[3]["RA"],
        item[3]["ASafeR"], item[3]["C"], item[3]["U"], -item[0],
    ))
    q05 = float(np.quantile(
        concatenate_run_values(runs, lambda r: confidence_values(r["known_scores"], CANDIDATE)[score_name]), 0.05
    ))
    per_attack = {name: evaluate_attack(runs, name, score_name, threshold) for name in ATTACK_ORDER}
    q05_per_attack = {name: evaluate_attack(runs, name, score_name, q05) for name in ATTACK_ORDER}
    return {
        "score_name": score_name, "threshold": float(threshold), "q05_threshold": q05,
        "per_attack": per_attack, "macro": macro_from_attacks(per_attack),
        "q05_per_attack": q05_per_attack, "q05_macro": macro_from_attacks(q05_per_attack),
        "score_candidate_summary": [
            {"score_name": name, "threshold": threshold_value, "macro": macro_value}
            for _, name, threshold_value, macro_value in score_candidates
        ],
    }


def fold_mean(rows: list[dict], keys: tuple[str, ...] | list[str]) -> dict:
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def clean_gate_summary(clean_selections: list[dict]) -> dict:
    return fold_mean(clean_selections, (
        "direct_C", "identity_macro_f1_direct", "unknown_auroc", "U_at_q05",
    ))


def gates_for_summary(summary: dict, attack_config: dict) -> dict:
    gates = attack_config["regularization_ablation"]["inner_gates"]
    return {
        "direct_C": bool(summary["direct_C"] >= gates["direct_C_min"]),
        "identity_macro_f1_direct": bool(summary["identity_macro_f1_direct"] >= gates["identity_macro_f1_direct_min"]),
        "unknown_auroc": bool(summary["unknown_auroc"] >= gates["unknown_auroc_min"]),
        "U_at_q05": bool(summary["U_at_q05"] >= gates["unknown_rejection_at_inner_q05_min"]),
    }


def full_train_fit(
    model: TorchPrototype, zband: torch.Tensor, config: dict,
) -> dict:
    indices = np.arange(len(zband))
    scores = predict_wccn(model, zband, indices, config)
    pred = model.classes[scores.argmax(axis=1)]
    true = config["_subjects"]
    accuracy = float(np.mean(pred == true))
    return {
        "accuracy": accuracy, "error_count": int(np.sum(pred != true)),
        "strict_100pct": bool(accuracy == 1.0),
        "identity_20class_macro_f1": float(f1_score(true, pred, labels=np.arange(1, 21), average="macro", zero_division=0)),
        "N": int(len(true)),
    }


def run_regularization(
    metadata: pd.DataFrame, raw: torch.Tensor, zband: torch.Tensor,
    config: dict, attack_config: dict, outer_groups: list[np.ndarray],
) -> tuple[dict, float]:
    subjects = config["_subjects"]
    trial = metadata["trial"].to_numpy()
    maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    forward = trial > 0.8 * maximum
    inner_block = (trial > 0.6 * maximum) & (trial <= 0.8 * maximum)
    all_subjects = np.unique(subjects)
    candidate_reports = []
    for shrinkage in attack_config["regularization_ablation"]["wccn_shrinkage_candidates"]:
        clean_selections = []
        for outer_index, outer_unknown in enumerate(outer_groups):
            outer_fold = outer_index + 1
            outer_known = np.setdiff1d(all_subjects, outer_unknown)
            runs = []
            inner_order = np.random.default_rng(100 + outer_fold).permutation(outer_known)
            for inner_unknown in np.array_split(inner_order, 2):
                enrolled = np.setdiff1d(outer_known, inner_unknown)
                enrolled_mask = np.isin(subjects, enrolled)
                fit_np = np.flatnonzero(enrolled_mask & ~forward & ~inner_block)
                known_np = np.flatnonzero(enrolled_mask & inner_block)
                unknown_np = np.flatnonzero(np.isin(subjects, inner_unknown) & ~forward)
                model = fit_clean_wccn(zband, fit_np, enrolled, float(shrinkage), config)
                runs.append(make_clean_run(model, zband, known_np, unknown_np, enrolled, config))
            clean_selections.append(select_clean_point(runs, attack_config))
        full_model = fit_clean_wccn(zband, np.arange(len(subjects)), all_subjects, float(shrinkage), config)
        train_fit = full_train_fit(full_model, zband, config)
        summary = clean_gate_summary(clean_selections)
        report = {
            "shrinkage": float(shrinkage), "full_train_fit": train_fit,
            "inner_clean_fold_mean": summary, "inner_clean_gates": gates_for_summary(summary, attack_config),
            "inner_clean_selections": clean_selections,
            "selection_proxy_fold_mean": float(np.mean([row["work"]["C_U_proxy_not_official"] for row in clean_selections])),
        }
        report["eligible"] = bool(train_fit["strict_100pct"] and all(report["inner_clean_gates"].values()))
        candidate_reports.append(report)
        print(
            f"regularization shrink={shrinkage}: fit={train_fit['accuracy']:.6f} "
            f"C={summary['direct_C']:.4f} F1={summary['identity_macro_f1_direct']:.4f} "
            f"AUROC={summary['unknown_auroc']:.4f} U@95={summary['U_at_q05']:.4f}", flush=True,
        )
    exact = [row for row in candidate_reports if row["full_train_fit"]["strict_100pct"]]
    pool = [row for row in exact if row["eligible"]] or exact or candidate_reports
    selected = max(pool, key=lambda row: (
        row["selection_proxy_fold_mean"], row["inner_clean_fold_mean"]["direct_C"],
        row["inner_clean_fold_mean"]["identity_macro_f1_direct"],
        row["inner_clean_fold_mean"]["unknown_auroc"], row["inner_clean_fold_mean"]["U_at_q05"],
        row["shrinkage"],
    ))
    return {
        "candidates": candidate_reports, "selected_shrinkage": selected["shrinkage"],
        "selection_conflict": not bool(selected["eligible"]),
        "selection_data": "inner folds and full Train only",
    }, float(selected["shrinkage"])


def run_inner_method(
    method: str, shrinkage: float, metadata: pd.DataFrame,
    raw: torch.Tensor, zband: torch.Tensor, config: dict, attack_config: dict,
    outer_groups: list[np.ndarray],
) -> dict:
    subjects = config["_subjects"]
    trial = metadata["trial"].to_numpy()
    maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    forward = trial > 0.8 * maximum
    inner_block = (trial > 0.6 * maximum) & (trial <= 0.8 * maximum)
    all_subjects = np.unique(subjects)
    folds = []
    for outer_index, outer_unknown in enumerate(outer_groups):
        outer_fold = outer_index + 1
        outer_known = np.setdiff1d(all_subjects, outer_unknown)
        runs = []
        fit_infos = []
        inner_order = np.random.default_rng(100 + outer_fold).permutation(outer_known)
        for inner_index, inner_unknown in enumerate(np.array_split(inner_order, 2)):
            inner_fold = inner_index + 1
            enrolled = np.setdiff1d(outer_known, inner_unknown)
            enrolled_mask = np.isin(subjects, enrolled)
            fit_np = np.flatnonzero(enrolled_mask & ~forward & ~inner_block)
            known_np = np.flatnonzero(enrolled_mask & inner_block)
            unknown_np = np.flatnonzero(np.isin(subjects, inner_unknown) & ~forward)
            model, fit_info = fit_model(
                method, raw, zband, fit_np, enrolled, shrinkage, config,
                attack_config, metadata, outer_fold, inner_fold,
            )
            runs.append(make_attack_run(
                model, raw, zband, known_np, unknown_np, enrolled, config,
                attack_config, metadata, 1, outer_fold, inner_fold,
            ))
            fit_infos.append(fit_info)
        selection = select_attack_point(runs, attack_config)
        clean_selection = select_clean_point([r["clean_full"] for r in runs], attack_config)
        folds.append({
            "outer_fold": outer_fold, "fit_info": fit_infos,
            "selection": selection, "clean_selection": clean_selection,
            "n_known_per_attack": int(sum(len(r["known_true"]) for r in runs)),
            "n_unknown_per_attack": int(sum(len(r["unknown_scores"]) for r in runs)),
            "attack_diagnostics": {
                name: [r["attacks"][name]["diagnostics"] for r in runs] for name in ATTACK_ORDER
            },
        })
        print(
            f"INNER method={method} outer={outer_fold}: Score={selection['macro']['Score']:.3f} "
            f"RA={selection['macro']['RA']:.4f} ASafeR={selection['macro']['ASafeR']:.4f}", flush=True,
        )
    per_attack = {
        name: fold_mean([fold["selection"]["per_attack"][name] for fold in folds], list(COMPONENT_KEYS) + ["N_clean", "N_adversarial", "N_unknown"])
        for name in ATTACK_ORDER
    }
    macro = fold_mean([fold["selection"]["macro"] for fold in folds], list(folds[0]["selection"]["macro"].keys()))
    clean_summary = clean_gate_summary([fold["clean_selection"] for fold in folds])
    return {
        "method": method, "folds": folds, "per_attack_fold_mean": per_attack,
        "macro_fold_mean": macro, "clean_inner_fold_mean": clean_summary,
        "clean_gates": gates_for_summary(clean_summary, attack_config),
    }


def evaluate_clean_full(
    runs: list[dict], score_name: str, threshold: float, q05_threshold: float,
) -> dict:
    true = concatenate_run_values(runs, lambda r: r["known_true"])
    direct = concatenate_run_values(runs, lambda r: direct_predictions(r, "known_scores"))
    known = concatenate_run_values(runs, lambda r: confidence_values(r["known_scores"], CANDIDATE)[score_name])
    unknown_direct = concatenate_run_values(runs, lambda r: direct_predictions(r, "unknown_scores"))
    unknown = concatenate_run_values(runs, lambda r: confidence_values(r["unknown_scores"], CANDIDATE)[score_name])
    labels = np.unique(np.concatenate([r["classes"] for r in runs]))
    detection = np.concatenate([np.ones(len(known)), np.zeros(len(unknown))])
    result = {
        "direct_C": float(np.mean(direct == true)),
        "identity_macro_f1_direct": float(f1_score(true, direct, labels=labels, average="macro", zero_division=0)),
        "unknown_auroc": float(roc_auc_score(detection, np.concatenate([known, unknown]))),
        "N_known": int(len(known)), "N_unknown": int(len(unknown)),
    }
    for prefix, value in (("", threshold), ("q05_", q05_threshold)):
        known_pred = direct.copy()
        unknown_pred = unknown_direct.copy()
        known_pred[known < value] = 0
        unknown_pred[unknown < value] = 0
        accepted = np.concatenate([known_pred != 0, unknown_pred != 0])
        result.update({
            f"{prefix}C": float(np.mean(known_pred == true)),
            f"{prefix}U": float(np.mean(unknown_pred == 0)),
            f"{prefix}known_acceptance": float(np.mean(known_pred != 0)),
            f"{prefix}identity_macro_f1": float(f1_score(true, known_pred, labels=labels, average="macro", zero_division=0)),
            f"{prefix}unknown_f1": float(f1_score(detection, accepted, pos_label=0, zero_division=0)),
        })
    return result


def run_outer(
    result: dict, metadata: pd.DataFrame, raw: torch.Tensor, zband: torch.Tensor,
    config: dict, attack_config: dict, outer_groups: list[np.ndarray],
) -> dict:
    method = result["selected_method"]
    shrinkage = float(result["regularization"]["selected_shrinkage"])
    inner_method = result["inner_methods"][method]
    subjects = config["_subjects"]
    trial = metadata["trial"].to_numpy()
    maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    forward = trial > 0.8 * maximum
    all_subjects = np.unique(subjects)
    folds = []
    for outer_index, unknown_subjects in enumerate(outer_groups):
        outer_fold = outer_index + 1
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        outer_known_mask = np.isin(subjects, known_subjects)
        fit_np = np.flatnonzero(outer_known_mask & ~forward)
        known_np = np.flatnonzero(outer_known_mask & forward)
        unknown_np = np.flatnonzero(np.isin(subjects, unknown_subjects))
        model, fit_info = fit_model(
            method, raw, zband, fit_np, known_subjects, shrinkage, config,
            attack_config, metadata, outer_fold, 0,
        )
        run = make_attack_run(
            model, raw, zband, known_np, unknown_np, known_subjects, config,
            attack_config, metadata, 2, outer_fold, 0,
        )
        selection = inner_method["folds"][outer_index]["selection"]
        per_attack = {
            name: evaluate_attack([run], name, selection["score_name"], selection["threshold"])
            for name in ATTACK_ORDER
        }
        clean = evaluate_clean_full(
            [run["clean_full"]], selection["score_name"], selection["threshold"], selection["q05_threshold"],
        )
        fit_scores = predict_wccn(model, zband, fit_np, config)
        fit_pred = known_subjects[fit_scores.argmax(axis=1)]
        fit_info["fit_accuracy"] = float(np.mean(fit_pred == subjects[fit_np]))
        folds.append({
            "outer_fold": outer_fold, "unknown_subjects": unknown_subjects.tolist(),
            "registered_subjects": known_subjects.tolist(), "fit_info": fit_info,
            "frozen_inner_selection": {
                "score_name": selection["score_name"], "threshold": selection["threshold"],
                "q05_threshold": selection["q05_threshold"],
            },
            "clean_full": clean, "per_attack": per_attack, "macro": macro_from_attacks(per_attack),
            "attack_diagnostics": {name: run["attacks"][name]["diagnostics"] for name in ATTACK_ORDER},
        })
        print(
            f"OUTER {outer_fold} {method}: C={folds[-1]['macro']['C']:.4f} "
            f"RA={folds[-1]['macro']['RA']:.4f} ASafeR={folds[-1]['macro']['ASafeR']:.4f} "
            f"U={folds[-1]['macro']['U']:.4f} Score={folds[-1]['macro']['Score']:.3f}", flush=True,
        )
        result["status"] = "outer_partial"
        result["outer"] = {"folds": folds}
        write_json(RESULTS_PATH, result)
    per_attack = {}
    for name in ATTACK_ORDER:
        rows = [fold["per_attack"][name] for fold in folds]
        per_attack[name] = {
            "fold_mean": fold_mean(rows, list(COMPONENT_KEYS)),
            "fold_std": {key: float(np.std([row[key] for row in rows])) for key in COMPONENT_KEYS},
            "N_clean_total": int(sum(row["N_clean"] for row in rows)),
            "N_adversarial_total": int(sum(row["N_adversarial"] for row in rows)),
            "N_unknown_total": int(sum(row["N_unknown"] for row in rows)),
        }
        if name.startswith("targeted"):
            per_attack[name]["fold_mean"]["targeted_direct_false_authentication"] = float(np.mean([
                row["targeted_direct_false_authentication"] for row in rows
            ]))
            per_attack[name]["fold_mean"]["targeted_false_authentication"] = float(np.mean([
                row["targeted_false_authentication"] for row in rows
            ]))
    macro_mean = fold_mean([fold["macro"] for fold in folds], list(folds[0]["macro"].keys()))
    clean_mean = fold_mean([fold["clean_full"] for fold in folds], list(folds[0]["clean_full"].keys()))
    return {
        "folds": folds, "per_attack": per_attack, "macro_fold_mean": macro_mean,
        "clean_full_fold_mean": clean_mean,
        "evaluation_rule": "outer only applied inner-frozen model, score and thresholds",
    }


def run_inner(
    metadata: pd.DataFrame, raw: torch.Tensor, zband: torch.Tensor,
    config: dict, attack_config: dict, outer_groups: list[np.ndarray],
) -> dict:
    started = time.time()
    regularization, shrinkage = run_regularization(metadata, raw, zband, config, attack_config, outer_groups)
    baseline = run_inner_method("baseline", shrinkage, metadata, raw, zband, config, attack_config, outer_groups)
    methods = {"baseline": baseline}
    baseline_train = next(
        row["full_train_fit"] for row in regularization["candidates"]
        if row["shrinkage"] == shrinkage
    )
    train_fits = {"baseline": baseline_train}
    trigger = (
        baseline["macro_fold_mean"]["RA"] < 0.95
        or baseline["macro_fold_mean"]["ASafeR"] < 0.98
        or not all(baseline["clean_gates"].values()) or not baseline_train["strict_100pct"]
    )
    if trigger:
        print("INNER preregistered single augmentation triggered", flush=True)
        augmented = run_inner_method(
            "pgd_fgsm_augmentation", shrinkage, metadata, raw, zband,
            config, attack_config, outer_groups,
        )
        methods["pgd_fgsm_augmentation"] = augmented
        all_subjects = np.unique(config["_subjects"])
        full_model, full_info = fit_model(
            "pgd_fgsm_augmentation", raw, zband, np.arange(len(config["_subjects"])),
            all_subjects, shrinkage, config, attack_config, metadata, 0, 0,
        )
        train_fits["pgd_fgsm_augmentation"] = {
            **full_train_fit(full_model, zband, config), "fit_info": full_info,
        }
    def method_rank(name: str) -> tuple:
        method = methods[name]
        clean_ok = all(method["clean_gates"].values())
        exact = train_fits[name]["strict_100pct"]
        macro = method["macro_fold_mean"]
        return (
            bool(exact and clean_ok), macro["Score"], macro["RA"], macro["ASafeR"],
        )
    selected_method = max(methods, key=method_rank)
    selected = methods[selected_method]
    selected_clean = selected["clean_inner_fold_mean"]
    result = {
        "status": "inner_complete", "experiment": "E36C", "date": "2026-11-12",
        "data_evidence": attack_config["data"]["source"] + "; Test not accessed",
        "validation_boundary": attack_config["reporting"]["validation_boundary"],
        "attack_config_sha256": sha256(ATTACK_CONFIG_PATH),
        "base_config_sha256": sha256(CONFIG_PATH), "script_sha256": sha256(Path(__file__).resolve()),
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "regularization": regularization, "inner_methods": methods,
        "full_train_fit": train_fits, "improvement_triggered": bool(trigger),
        "selected_method": selected_method,
        "targets": {
            "full_train_strict_100pct": bool(train_fits[selected_method]["strict_100pct"]),
            "inner_clean_gates": selected["clean_gates"],
            "inner_RA_at_least_0.95": bool(selected["macro_fold_mean"]["RA"] >= 0.95),
            "inner_ASafeR_at_least_0.98": bool(selected["macro_fold_mean"]["ASafeR"] >= 0.98),
            "inner_not_all_rejection": not bool(selected["macro_fold_mean"]["all_rejection"]),
            "conflict": bool(
                not train_fits[selected_method]["strict_100pct"]
                or not all(selected["clean_gates"].values())
                or selected["macro_fold_mean"]["RA"] < 0.95
                or selected["macro_fold_mean"]["ASafeR"] < 0.98
            ),
            "selected_inner_clean": selected_clean,
        },
        "test_metrics": "N/A: Test was not read and has no labels",
        "elapsed_inner_seconds": time.time() - started,
    }
    write_json(RESULTS_PATH, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("inner", "outer", "all"), default="all")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("E36 攻击要求 CUDA GPU")
    torch.manual_seed(20261112)
    np.random.seed(20261112)
    device = torch.device("cuda")
    config = load_config()
    attack_config = load_attack_config()
    metadata = train_metadata(config)
    config["_subjects"] = metadata["subject"].to_numpy(dtype=int)
    subjects = config["_subjects"]
    outer_groups = list(np.array_split(
        np.random.default_rng(config["nested_protocol"]["outer_seed"]).permutation(np.unique(subjects)), 5
    ))
    raw = load_all_raw(metadata, config, device)
    zband = torch.as_tensor(np.load(ZBAND_PATH), dtype=torch.float32, device=device)
    if args.stage in ("inner", "all"):
        result = run_inner(metadata, raw, zband, config, attack_config, outer_groups)
    else:
        result = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if result["status"] not in ("inner_complete", "outer_partial", "complete"):
            raise ValueError("先运行 --stage inner")
        if result["attack_config_sha256"] != sha256(ATTACK_CONFIG_PATH):
            raise ValueError("攻击配置在内层冻结后发生变化")
    if args.stage in ("outer", "all"):
        result["outer"] = run_outer(result, metadata, raw, zband, config, attack_config, outer_groups)
        result["status"] = "complete"
        macro = result["outer"]["macro_fold_mean"]
        result["targets"]["outer_RA_at_least_0.95_descriptive"] = bool(macro["RA"] >= 0.95)
        result["targets"]["outer_ASafeR_at_least_0.98_descriptive"] = bool(macro["ASafeR"] >= 0.98)
        result["targets"]["outer_not_all_rejection"] = not bool(macro["all_rejection"])
        result["test_metrics"] = "N/A: Test was not read and has no labels"
        write_json(RESULTS_PATH, result)
        print(f"完成：{RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
