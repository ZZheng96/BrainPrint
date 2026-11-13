"""E65：逐trial/频带通道标准化相关矩阵logSPD；先inner，门禁后一次Test。"""
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
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
sys.path.insert(0, str(ROOT / "research" / "E54_outlier_exposure"))
from run_e36 import detrend_zband, load_config as load_e36_config, load_train_matrix, sha256, train_metadata  # noqa: E402
import run_e54 as E54  # noqa: E402

CONFIG_PATH = OUT / "config.json"
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "report.md"
PREDICTIONS_PATH = OUT / "e65_predictions.csv"
CACHE = ROOT / "BrainPrint_cache" / "e65_correlation_spd"
CORR_PATH = CACHE / "train_correlation_logspd.npy"
FEATURE_MANIFEST = CACHE / "feature_manifest.json"
E36_ZBAND = ROOT / "BrainPrint_cache" / "e36_pure_eeg" / "train_detrend_zband_cov5.npy"
E54_CACHE = ROOT / "BrainPrint_cache" / "e54_outlier_exposure"
MODEL_PATH = CACHE / "frozen_model.pt"
FREEZE_MANIFEST = CACHE / "frozen_manifest.json"
CLASSES = np.arange(1, 21, dtype=int)
VARIANTS = ("e36_covariance", "e65_correlation")
TEST_METRICS = {key: "N/A（无真实标签）" for key in ("AUC", "F1", "C", "RA", "ASafeR", "U", "Score")}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def bandpass(raw: torch.Tensor, e36: dict) -> torch.Tensor:
    n = int(e36["data"]["samples_per_trial"])
    fs = float(e36["data"]["sampling_rate_hz"])
    bands = e36["features"]["bands_hz"]
    frequency = torch.fft.rfftfreq(n, d=1.0 / fs, device=raw.device)
    masks = []
    for index, (low, high) in enumerate(bands):
        upper = frequency <= high if index == len(bands) - 1 else frequency < high
        masks.append((frequency >= low) & upper)
    mask = torch.stack(masks).to(raw.dtype)
    spectrum = torch.fft.rfft(raw, dim=-1)
    return torch.fft.irfft(spectrum[:, None] * mask[None, :, None], n=n, dim=-1)


def gpu_correlation_logspd(raw: torch.Tensor, e36: dict, config: dict,
                           return_diagnostics: bool = False):
    """B×18×1200 -> B×855；通道统计只存在于各trial/频带内部。"""
    filtered = bandpass(raw, e36)
    filtered = filtered - filtered.mean(dim=-1, keepdim=True)
    ddof = int(config["features"]["channel_std_ddof"])
    scale = filtered.std(dim=-1, keepdim=True, unbiased=ddof == 1).clamp_min(
        float(config["features"]["channel_std_epsilon"]))
    normalized = (filtered - filtered.mean(dim=-1, keepdim=True)) / scale
    # filtered已去均值；第二次去均值仅抑制浮点残差，不引入任何跨样本统计。
    n = raw.shape[-1]
    correlation = normalized @ normalized.transpose(-1, -2) / (n - ddof)
    channels = raw.shape[-2]
    shrinkage = float(e36["features"]["covariance_shrinkage"])
    trace = torch.diagonal(correlation, dim1=-2, dim2=-1).sum(dim=-1)
    eye = torch.eye(channels, dtype=raw.dtype, device=raw.device)
    spd = ((1.0 - shrinkage) * correlation
           + shrinkage * trace[..., None, None] / channels * eye)
    eigenvalues, eigenvectors = torch.linalg.eigh(spd)
    eigenvalues = eigenvalues.clamp_min(float(e36["features"]["eigenvalue_floor"]))
    logspd = (eigenvectors * eigenvalues.log().unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)
    rows, columns = torch.triu_indices(channels, channels, device=raw.device)
    vectors = logspd[..., rows, columns]
    vectors[..., rows != columns] *= float(e36["features"]["off_diagonal_multiplier"])
    features = detrend_zband(vectors.flatten(1), e36)
    if return_diagnostics:
        diagonal = torch.diagonal(correlation, dim1=-2, dim2=-1)
        diagnostics = {
            "correlation_diag_max_abs_error": float((diagonal - 1).abs().max()),
            "correlation_max_asymmetry": float((correlation - correlation.transpose(-1, -2)).abs().max()),
            "normalized_max_abs_channel_mean": float(normalized.mean(dim=-1).abs().max()),
            "normalized_channel_std_max_abs_error": float((normalized.std(dim=-1, unbiased=True) - 1).abs().max()),
        }
        return features, diagnostics
    return features


def build_features(config: dict, e36: dict, device: torch.device) -> dict:
    metadata = train_metadata(e36)
    if CORR_PATH.exists() and FEATURE_MANIFEST.exists():
        report = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))
        if report["config_sha256"] == sha256(CONFIG_PATH):
            cached = np.load(CORR_PATH, mmap_mode="r")
            if cached.shape == (len(metadata), 855) and np.isfinite(cached).all():
                print("复用E65 correlation-logSPD Train缓存", flush=True)
                return report
    started = time.time()
    paths = [ROOT / config["data"]["train_root"] / path for path in metadata["path"]]
    output = np.lib.format.open_memmap(CORR_PATH, mode="w+", dtype=np.float32,
                                       shape=(len(metadata), 855))
    batch_size = int(config["features"]["io_batch_size"])
    with ThreadPoolExecutor(max_workers=int(config["features"]["io_workers"])) as pool:
        for start in range(0, len(paths), batch_size):
            arrays = list(pool.map(load_train_matrix, paths[start:start + batch_size]))
            raw = torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=device)
            with torch.no_grad():
                features = gpu_correlation_logspd(raw, e36, config)
            output[start:start + len(arrays)] = features.cpu().numpy()
            stop = min(start + batch_size, len(paths))
            if stop % 1024 == 0 or stop == len(paths):
                print(f"E65 features {stop}/{len(paths)}", flush=True)
    output.flush()
    del output
    rng = np.random.default_rng(int(config["seed"]))
    selected = np.sort(rng.choice(len(paths), 8, replace=False))
    selected_paths = [paths[index] for index in selected]
    raw = torch.as_tensor(np.stack([load_train_matrix(path) for path in selected_paths]),
                          dtype=torch.float32, device=device)
    with torch.no_grad():
        batch, diagnostics = gpu_correlation_logspd(raw, e36, config, True)
        singles = torch.cat([gpu_correlation_logspd(raw[i:i + 1], e36, config)
                             for i in range(len(raw))])
        permutation = rng.permutation(len(raw))
        inverse = np.argsort(permutation)
        permuted = gpu_correlation_logspd(raw[permutation], e36, config)
    cached = np.load(CORR_PATH, mmap_mode="r")
    checks = {
        **diagnostics,
        "cache_vs_rebuilt_max_abs": float(np.max(np.abs(cached[selected] - batch.cpu().numpy()))),
        "single_vs_batch_max_abs": float((batch - singles).abs().max()),
        "permutation_vs_batch_max_abs": float((batch - permuted[inverse]).abs().max()),
        "single_vs_batch_allclose": bool(torch.allclose(batch, singles, atol=1e-5, rtol=1e-5)),
        "permutation_allclose": bool(torch.allclose(batch, permuted[inverse], atol=1e-5, rtol=1e-5)),
        "cross_trial_or_batch_statistics": False,
    }
    if not checks["single_vs_batch_allclose"] or not checks["permutation_allclose"]:
        raise RuntimeError(f"E65特征逐样本不变性失败：{checks}")
    report = {
        "shape": [len(metadata), 855], "dtype": "float32", "finite": bool(np.isfinite(cached).all()),
        "cache": str(CORR_PATH), "cache_sha256": sha256(CORR_PATH),
        "config_sha256": sha256(CONFIG_PATH), "metadata_sha256": sha256(ROOT / config["data"]["train_metadata"]),
        "checks": checks, "test_accessed": False, "elapsed_seconds": time.time() - started,
        "gpu": torch.cuda.get_device_name(0),
    }
    write_json(FEATURE_MANIFEST, report)
    print(f"E65特征检查 {checks}", flush=True)
    return report


class WCCNPrototype:
    def __init__(self, x: torch.Tensor, y: np.ndarray, classes: np.ndarray, config: dict):
        self.classes = np.asarray(classes, dtype=int)
        lookup = {int(label): index for index, label in enumerate(self.classes)}
        target = torch.as_tensor([lookup[int(label)] for label in y], dtype=torch.long, device=x.device)
        means = torch.stack([x[target == index].mean(0) for index in range(len(classes))])
        residual = x - means[target]
        covariance = residual.T @ residual / max(len(x) - len(classes), 1)
        shrinkage = float(config["features"]["wccn_shrinkage"])
        eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
        covariance = ((1 - shrinkage) * covariance
                      + shrinkage * torch.diagonal(covariance).sum() / x.shape[1] * eye)
        values, vectors = torch.linalg.eigh(covariance)
        inverse_root = values.clamp_min(float(config["features"]["wccn_eigenvalue_floor"])).rsqrt()
        self.whitening = (vectors * inverse_root.unsqueeze(0)) @ vectors.T
        means = means @ self.whitening
        self.prototypes = means / means.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.whitening
        return (x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)) @ self.prototypes.T


def score_features(model: WCCNPrototype, features: torch.Tensor, indices: np.ndarray,
                   batch_size: int = 512) -> np.ndarray:
    result = np.empty((len(indices), len(model.classes)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = torch.as_tensor(indices[start:start + batch_size], dtype=torch.long,
                                       device=features.device)
            result[start:start + len(selected)] = model(features[selected]).cpu().numpy()
    return result


def score_raw(model: WCCNPrototype, raw: torch.Tensor, variant: str,
              e36: dict, config: dict) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(raw), 64):
            batch = raw[start:start + 64]
            if variant == "e65_correlation":
                features = gpu_correlation_logspd(batch, e36, config)
            else:
                from run_e36 import gpu_logspd
                features = detrend_zband(gpu_logspd(batch, e36), e36)
            outputs.append(model(features).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def distance_ratio(scores: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(scores, dtype=np.float64), axis=1)
    maximum, second = ordered[:, -1], ordered[:, -2]
    return (1 - second) / np.maximum(1 - maximum, 1e-6)


def thresholds(known_log: np.ndarray, method: str = "linear") -> tuple[float, float]:
    dr = distance_ratio(known_log)
    return (float(np.quantile(dr, 0.02, method=method)),
            float(np.quantile(dr, 0.05, method=method)))


def decide(identity_scores: np.ndarray, log_scores: np.ndarray, classes: np.ndarray,
           tau_low: float, tau_high: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dr = distance_ratio(log_scores)
    identity = classes[np.asarray(identity_scores).argmax(1)]
    log_identity = classes[np.asarray(log_scores).argmax(1)]
    direct = np.where(dr >= tau_high, log_identity, identity).astype(int)
    return np.where(dr < tau_low, 0, direct).astype(int), direct, dr


def detection(known_dr: np.ndarray, unknown_dr: np.ndarray) -> dict:
    labels = np.concatenate([np.ones(len(known_dr), dtype=int), np.zeros(len(unknown_dr), dtype=int)])
    values = np.concatenate([known_dr, unknown_dr])
    return {
        "AUROC": float(roc_auc_score(labels, values)),
        "unknown_AUPR": float(average_precision_score(1 - labels, -values)),
        "known_AUPR": float(average_precision_score(labels, values)),
    }


def dr_summary(values: np.ndarray) -> dict:
    quantiles = np.quantile(values, [0.02, 0.05, 0.5, 0.95])
    return {"N": int(len(values)), "mean": float(np.mean(values)), "std": float(np.std(values)),
            "q02": float(quantiles[0]), "q05": float(quantiles[1]),
            "median": float(quantiles[2]), "q95": float(quantiles[3])}


def clean_run_metrics(run: dict, variant: str) -> dict:
    item = run["variants"][variant]
    classes = run["classes"]
    true = run["known_true"]
    tau_low, tau_high = item["tau_low"], item["tau_high"]
    known_pred, direct, known_dr = decide(run["id_known"], item["log_known"], classes, tau_low, tau_high)
    unknown_pred, _, unknown_dr = decide(run["id_unknown"], item["log_unknown"], classes, tau_low, tau_high)
    detection_true = np.concatenate([np.ones(len(true), dtype=int), np.zeros(len(unknown_pred), dtype=int)])
    accepted = np.concatenate([known_pred != 0, unknown_pred != 0])
    return {
        "N_known": int(len(true)), "N_unknown": int(len(unknown_pred)),
        "direct_C": float(np.mean(direct == true)),
        "threshold_C": float(np.mean(known_pred == true)),
        "continuous_block_accuracy": float(np.mean(direct == true)),
        "identity_macro_F1_direct": float(f1_score(true, direct, labels=classes, average="macro", zero_division=0)),
        "identity_macro_F1_threshold": float(f1_score(true, known_pred, labels=classes, average="macro", zero_division=0)),
        **detection(known_dr, unknown_dr),
        "U_at_95": float(np.mean(unknown_dr < tau_high)),
        "working_U": float(np.mean(unknown_pred == 0)),
        "known_acceptance": float(np.mean(known_pred != 0)),
        "unknown_F1": float(f1_score(detection_true, accepted, pos_label=0, zero_division=0)),
        "joint_F1": float(f1_score(np.concatenate([true, np.zeros(len(unknown_pred), dtype=int)]),
                                    np.concatenate([known_pred, unknown_pred]),
                                    labels=np.concatenate([[0], classes]), average="macro", zero_division=0)),
        "tau_low": float(tau_low), "tau_high": float(tau_high),
    }


def attack_run_metrics(run: dict, variant: str, attack: str) -> dict:
    item = run["variants"][variant]
    scores = item["attacks"][attack]
    pred, direct, _ = decide(scores["id"], scores["log"], run["classes"], item["tau_low"], item["tau_high"])
    true = run["known_true"]
    ra = float(np.mean(pred == true))
    asafer = float(np.mean((pred == true) | (pred == 0)))
    return {
        "N": int(len(true)), "RA": ra, "ASafeR": asafer, "A": 0.7 * ra + 0.3 * asafer,
        "direct_RA": float(np.mean(direct == true)), "rejection_rate": float(np.mean(pred == 0)),
        "identity_macro_F1": float(f1_score(true, pred, labels=run["classes"], average="macro", zero_division=0)),
        "all_rejection": bool(np.all(pred == 0)),
    }


def mean_rows(rows: list[dict]) -> dict:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]
            if isinstance(rows[0][key], (int, float, np.integer, np.floating))
            and not isinstance(rows[0][key], (bool, np.bool_))}


def pooled_clean(runs: list[dict], variant: str) -> dict:
    known_true, known_pred, unknown_pred, direct = [], [], [], []
    known_dr, unknown_dr = [], []
    for run in runs:
        item = run["variants"][variant]
        kp, d, kd = decide(run["id_known"], item["log_known"], run["classes"], item["tau_low"], item["tau_high"])
        up, _, ud = decide(run["id_unknown"], item["log_unknown"], run["classes"], item["tau_low"], item["tau_high"])
        known_true.append(run["known_true"])
        known_pred.append(kp)
        unknown_pred.append(up)
        direct.append(d)
        known_dr.append(kd)
        unknown_dr.append(ud)
    true = np.concatenate(known_true)
    kp = np.concatenate(known_pred)
    up = np.concatenate(unknown_pred)
    direct = np.concatenate(direct)
    detection_true = np.concatenate([np.ones(len(true), dtype=int), np.zeros(len(up), dtype=int)])
    accepted = np.concatenate([kp != 0, up != 0])
    return {
        "N_known": int(len(true)), "N_unknown": int(len(up)),
        "direct_C": float(np.mean(direct == true)), "threshold_C": float(np.mean(kp == true)),
        "identity_20class_macro_F1_direct": float(f1_score(true, direct, labels=CLASSES, average="macro", zero_division=0)),
        "identity_20class_macro_F1_threshold": float(f1_score(true, kp, labels=CLASSES, average="macro", zero_division=0)),
        **detection(np.concatenate(known_dr), np.concatenate(unknown_dr)),
        "working_U": float(np.mean(up == 0)),
        "unknown_F1": float(f1_score(detection_true, accepted, pos_label=0, zero_division=0)),
        "joint_21class_F1": float(f1_score(np.concatenate([true, np.zeros(len(up), dtype=int)]),
                                           np.concatenate([kp, up]), labels=np.arange(21),
                                           average="macro", zero_division=0)),
    }


def drift_summary(runs: list[dict], variant: str) -> dict:
    block_rows = {block: [] for block in ("fit", "calibration", "sanity")}
    for run in runs:
        values = run["variants"][variant]["dr_blocks"]
        fit = dr_summary(values["fit"])
        for block in block_rows:
            row = dr_summary(values[block])
            row["drift_vs_fit"] = {key: float(row[key] - fit[key])
                                   for key in ("mean", "q02", "q05", "median", "q95")}
            block_rows[block].append(row)
    return {block: {
        "run_mean": mean_rows(rows),
        "drift_vs_fit_run_mean": mean_rows([row["drift_vs_fit"] for row in rows]),
    } for block, rows in block_rows.items()}


def summarize_runs(runs: list[dict], config: dict) -> dict:
    folds, overall = [], {}
    for variant in VARIANTS:
        overall[variant] = {}
    for outer in sorted(set(run["outer_fold"] for run in runs)):
        pair = [run for run in runs if run["outer_fold"] == outer]
        fold = {"outer_fold": outer, "inner_runs": [run["inner_fold"] for run in pair], "variants": {}}
        for variant in VARIANTS:
            clean_runs = [clean_run_metrics(run, variant) for run in pair]
            per_attack = {attack: mean_rows([attack_run_metrics(run, variant, attack) for run in pair])
                          for attack in config["attacks"]["order"]}
            macro = mean_rows(list(per_attack.values()))
            fold["variants"][variant] = {"clean_runs": clean_runs, "clean": mean_rows(clean_runs),
                                         "per_attack": per_attack, "attack_macro": macro}
        fold["delta_correlation_minus_covariance"] = {
            "clean": {key: float(fold["variants"]["e65_correlation"]["clean"][key]
                                 - fold["variants"]["e36_covariance"]["clean"][key])
                      for key in fold["variants"]["e65_correlation"]["clean"]},
            "attack_macro": {key: float(fold["variants"]["e65_correlation"]["attack_macro"][key]
                                        - fold["variants"]["e36_covariance"]["attack_macro"][key])
                             for key in fold["variants"]["e65_correlation"]["attack_macro"]},
        }
        folds.append(fold)
    for variant in VARIANTS:
        variant_folds = [fold["variants"][variant] for fold in folds]
        overall[variant] = {
            "clean_outer_mean": mean_rows([row["clean"] for row in variant_folds]),
            "attack_outer_mean": mean_rows([row["attack_macro"] for row in variant_folds]),
            "per_attack_outer_mean": {attack: mean_rows([row["per_attack"][attack] for row in variant_folds])
                                      for attack in config["attacks"]["order"]},
            "pooled_clean": pooled_clean(runs, variant),
            "dr_drift": drift_summary(runs, variant),
        }
    overall["delta_correlation_minus_covariance"] = {
        section: {key: float(overall["e65_correlation"][section][key]
                             - overall["e36_covariance"][section][key])
                  for key in overall["e65_correlation"][section]}
        for section in ("clean_outer_mean", "attack_outer_mean")
    }
    return {"outer_folds": folds, "summary": overall}


def load_inner_model(outer: int, inner: int, enrolled: np.ndarray,
                     e54_config: dict, device: torch.device):
    path = E54_CACHE / f"oe_outer{outer}_inner{inner}.pt"
    item = torch.load(path, map_location="cpu", weights_only=False)
    if item["config_sha256"] != sha256(E54.OUT / "config.json") or item["info"]["variant"] != "oe":
        raise ValueError(f"E54 artifact指纹或variant异常：{path}")
    if not np.array_equal(np.asarray(item["classes"], dtype=int), enrolled):
        raise ValueError(f"E54 artifact类别与E58角色不一致：{path}")
    model = E54.EEGNetEmbedding(len(enrolled), e54_config, task_adversarial=False).to(device)
    model.load_state_dict(item["model_state"])
    model.eval()
    model.requires_grad_(False)
    return model, item


def make_inner_run(raw: torch.Tensor, features: dict[str, torch.Tensor], metadata: pd.DataFrame,
                   outer: int, inner: int, unknown_subjects: np.ndarray, known_subjects: np.ndarray,
                   inner_unknown: np.ndarray, config: dict, e36: dict, e54_config: dict) -> dict:
    device = raw.device
    subjects = metadata["subject"].to_numpy(dtype=int)
    trial = metadata["trial"].to_numpy()
    maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    roles = E54.build_inner_roles(known_subjects, inner_unknown, unknown_subjects, outer, inner, e54_config)
    enrolled = roles["enrolled"]
    masks = {
        "fit": trial <= 0.6 * maximum,
        "calibration": (trial > 0.6 * maximum) & (trial <= 0.7 * maximum),
        "sanity": (trial > 0.7 * maximum) & (trial <= 0.8 * maximum),
        "known": (trial > 0.6 * maximum) & (trial <= 0.8 * maximum),
        "unknown": trial <= 0.8 * maximum,
    }
    fit_np = np.flatnonzero(np.isin(subjects, enrolled) & masks["fit"])
    known_np = np.flatnonzero(np.isin(subjects, enrolled) & masks["known"])
    unknown_np = np.flatnonzero(np.isin(subjects, roles["calibration"]) & masks["unknown"])
    known_subset = E54.stratified_subset(metadata, known_np, 8)
    unknown_subset = E54.stratified_subset(metadata, unknown_np, 8)
    model, artifact = load_inner_model(outer, inner, enrolled, e54_config, device)
    id_known = E54.predict_scores(model, raw, known_subset)
    id_unknown = E54.predict_scores(model, raw, unknown_subset)
    run = {
        "experiment": "E65", "config_sha256": sha256(CONFIG_PATH), "script_sha256": sha256(Path(__file__)),
        "outer_fold": outer, "inner_fold": inner, "classes": enrolled,
        "roles": {key: np.asarray(value).tolist() for key, value in roles.items() if key != "pairwise_disjoint"},
        "known_true": subjects[known_subset], "known_indices": known_subset, "unknown_indices": unknown_subset,
        "id_known": id_known, "id_unknown": id_unknown, "variants": {},
        "identity_fit_accuracy": float(artifact["info"]["fit_accuracy"]),
        "identity_best_epoch": int(artifact["info"]["best_epoch"]),
        "identity_artifact_sha256": sha256(E54_CACHE / f"oe_outer{outer}_inner{inner}.pt"),
    }
    models = {}
    for variant in VARIANTS:
        prototype = WCCNPrototype(features[variant][torch.as_tensor(fit_np, device=device)],
                                  subjects[fit_np], enrolled, config)
        models[variant] = prototype
        log_known = score_features(prototype, features[variant], known_subset,
                                   int(config["features"]["prediction_batch_size"]))
        log_unknown = score_features(prototype, features[variant], unknown_subset,
                                     int(config["features"]["prediction_batch_size"]))
        tau_low, tau_high = thresholds(log_known, "linear")
        dr_blocks = {}
        for block in ("fit", "calibration", "sanity"):
            indices = np.flatnonzero(np.isin(subjects, enrolled) & masks[block])
            block_scores = score_features(prototype, features[variant], indices,
                                          int(config["features"]["prediction_batch_size"]))
            dr_blocks[block] = distance_ratio(block_scores)
        run["variants"][variant] = {"log_known": log_known, "log_unknown": log_unknown,
                                    "tau_low": tau_low, "tau_high": tau_high,
                                    "dr_blocks": dr_blocks, "attacks": {}}
    raw_known = raw[torch.as_tensor(known_subset, device=device)]
    true = run["known_true"]
    lookup = {int(label): index for index, label in enumerate(enrolled)}
    labels = torch.as_tensor([lookup[int(value)] for value in true], dtype=torch.long, device=device)
    target = np.asarray([enrolled[(lookup[int(value)] + 1) % len(enrolled)] for value in true], dtype=int)
    target_indices = torch.as_tensor([lookup[int(value)] for value in target], dtype=torch.long, device=device)
    for attack in config["attacks"]["order"]:
        base, snr_text = attack.rsplit("_snr", 1)
        snr = int(snr_text)
        seed = E54.attack_seed(e54_config, 1, outer, inner, base, snr)
        identity_scores, diagnostics, waves = E54.raw_attack(
            model, raw_known, labels, snr, base, int(config["attacks"]["steps_as_e58_raw_attack_calls"]),
            seed, target=target_indices if base == "targeted_pgd" else None,
            batch_size=int(config["attacks"]["batch_size"]), return_waveforms=True)
        identity_np = identity_scores.cpu().numpy()
        for variant in VARIANTS:
            run["variants"][variant]["attacks"][attack] = {
                "id": identity_np, "log": score_raw(models[variant], waves.float(), variant, e36, config),
                "diagnostics": diagnostics,
            }
        print(f"  o{outer}i{inner} {attack} N={len(true)} done", flush=True)
        del waves, identity_scores
    torch.save(run, CACHE / f"inner_o{outer}_i{inner}.pt")
    del model, models, raw_known
    torch.cuda.empty_cache()
    return run


def gates_for(summary: dict, config: dict) -> dict:
    clean = summary["e65_correlation"]["clean_outer_mean"]
    gate = config["test_gate"]
    values = {
        "clean_direct_C": (float(clean["direct_C"]), float(gate["clean_direct_C_min"])),
        "clean_identity_macro_F1_direct": (float(clean["identity_macro_F1_direct"]),
                                             float(gate["clean_identity_macro_F1_direct_min"])),
        "clean_AUROC": (float(clean["AUROC"]), float(gate["clean_AUROC_min"])),
        "clean_U_at_95": (float(clean["U_at_95"]), float(gate["clean_U_at_95_min"])),
    }
    return {key: {"value": value, "minimum": minimum, "passed": bool(value >= minimum)}
            for key, (value, minimum) in values.items()}


def run_inner(config: dict, e36: dict, e54_config: dict, device: torch.device,
              resume: bool = True) -> dict:
    started = time.time()
    metadata = train_metadata(e36)
    feature_report = build_features(config, e36, device)
    raw = E54.load_all_raw(metadata, e54_config, device)
    baseline = np.load(E36_ZBAND, allow_pickle=False)
    correlation = np.load(CORR_PATH, allow_pickle=False)
    if baseline.shape != (len(metadata), 855) or not np.isfinite(baseline).all():
        raise ValueError("E36 baseline特征缓存无效")
    features = {name: torch.as_tensor(array, dtype=torch.float32, device=device)
                for name, array in zip(VARIANTS, (baseline, correlation))}
    subjects = metadata["subject"].to_numpy(dtype=int)
    all_subjects = np.unique(subjects)
    outer_groups = np.array_split(np.random.default_rng(42).permutation(all_subjects), 5)
    runs = []
    for outer, unknown_subjects in enumerate(outer_groups, start=1):
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        inner_order = np.random.default_rng(100 + outer).permutation(known_subjects)
        for inner, inner_unknown in enumerate(np.array_split(inner_order, 2), start=1):
            checkpoint = CACHE / f"inner_o{outer}_i{inner}.pt"
            run = None
            if resume and checkpoint.exists():
                candidate = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if candidate.get("config_sha256") == sha256(CONFIG_PATH) and candidate.get("script_sha256") == sha256(Path(__file__)):
                    run = candidate
                    print(f"复用E65断点 o{outer}i{inner}", flush=True)
            if run is None:
                print(f"E65 inner o{outer}i{inner}开始", flush=True)
                run = make_inner_run(raw, features, metadata, outer, inner, unknown_subjects,
                                     known_subjects, inner_unknown, config, e36, e54_config)
            runs.append(run)
            clean = clean_run_metrics(run, "e65_correlation")
            print(f"E65 o{outer}i{inner}: directC={clean['direct_C']:.4f} C={clean['threshold_C']:.4f} "
                  f"F1={clean['identity_macro_F1_threshold']:.4f} AUROC={clean['AUROC']:.4f} "
                  f"U95={clean['U_at_95']:.4f} U={clean['working_U']:.4f}", flush=True)
        partial = summarize_runs(runs, config)
        write_json(RESULTS_PATH, {"status": "running_inner", "completed_inner_runs": len(runs), **partial})
    summary = summarize_runs(runs, config)
    gates = gates_for(summary["summary"], config)
    all_clean_pass = all(row["passed"] for row in gates.values())
    attack = summary["summary"]["e65_correlation"]["attack_outer_mean"]
    result = {
        "status": "inner_complete_test_authorized" if all_clean_pass else "inner_complete_test_not_authorized",
        "experiment": "E65", "date": config["date_preregistered"],
        "main_variable": config["features"]["main_variable"],
        "feature_report": feature_report,
        "run_roles_and_identity": [{"outer_fold": run["outer_fold"], "inner_fold": run["inner_fold"],
                                    "classes": run["classes"].tolist(), "roles": run["roles"],
                                    "identity_fit_accuracy": run["identity_fit_accuracy"],
                                    "identity_best_epoch": run["identity_best_epoch"],
                                    "identity_artifact_sha256": run["identity_artifact_sha256"]} for run in runs],
        **summary,
        "test_gates": {"checks": gates, "all_clean_passed": all_clean_pass,
                       "RA_and_ASafeR_not_clean_gate": {"RA": attack["RA"], "ASafeR": attack["ASafeR"]},
                       "all_train_gates_passed_claim": False,
                       "reason": "CNN-only attack is not full-chain adaptive evidence even if clean gates pass"},
        "test": {"status": "pending_authorized_single_audit" if all_clean_pass else "N/A_clean_gate_failed",
                 "test_accessed": False, "metrics": TEST_METRICS},
        "evidence_boundary": config["reporting"]["evidence_boundary"],
        "attack_boundary": config["attacks"]["scope_warning"],
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "sklearn": sklearn.__version__,
                        "gpu": torch.cuda.get_device_name(0), "compute": "raw/features/WCCN/attacks on CUDA; MAT I/O and metrics on CPU"},
        "script_sha256": sha256(Path(__file__)), "config_sha256": sha256(CONFIG_PATH),
        "inner_elapsed_seconds": time.time() - started,
    }
    write_json(RESULTS_PATH, result)
    write_report(result, config)
    print(f"E65 INNER SUMMARY {json.dumps(summary['summary']['e65_correlation']['clean_outer_mean'], ensure_ascii=False)}", flush=True)
    print(f"E65 TEST GATES {json.dumps(gates, ensure_ascii=False)}", flush=True)
    del raw, features
    torch.cuda.empty_cache()
    return result


def final_blocks(metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    trial = metadata["trial"].to_numpy()
    maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    return {
        "fit": np.flatnonzero(trial <= 0.6 * maximum),
        "calibration": np.flatnonzero((trial > 0.6 * maximum) & (trial <= 0.7 * maximum)),
        "sanity": np.flatnonzero((trial > 0.7 * maximum) & (trial <= 0.8 * maximum)),
    }


def load_e64_ensemble(config: dict, e54_config: dict, device: torch.device):
    model_path = ROOT / config["identity"]["e64_frozen_model"]
    manifest_path = ROOT / config["identity"]["e64_frozen_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "E64" or sha256(model_path) != manifest["model_sha256"]:
        raise ValueError("E64冻结ensemble指纹校验失败")
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if not np.array_equal(np.asarray(payload["classes"], dtype=int), CLASSES) or len(payload["backbones"]) != 10:
        raise ValueError("E64冻结模型必须是十backbone/20类ensemble")
    models = []
    for backbone in payload["backbones"]:
        model = E54.EEGNetEmbedding(20, e54_config, task_adversarial=False).to(device)
        incompatible = model.load_state_dict(backbone["encoder_state"], strict=False)
        if set(incompatible.missing_keys) != {"prototypes", "radius_raw"} or incompatible.unexpected_keys:
            raise ValueError(f"E64 backbone加载异常：{incompatible}")
        with torch.no_grad():
            model.prototypes.copy_(backbone["new_prototypes"].to(device))
        model.eval()
        model.requires_grad_(False)
        models.append(model)
    return models, payload


def ensemble_scores(models: list[torch.nn.Module], raw: torch.Tensor,
                    indices: np.ndarray | None = None, batch_size: int = 256) -> np.ndarray:
    source = raw if indices is None else raw[torch.as_tensor(indices, dtype=torch.long, device=raw.device)]
    output = np.zeros((len(source), 20), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(source), batch_size):
            batch = source[start:start + batch_size]
            output[start:start + len(batch)] = np.mean([
                model.scores(batch).cpu().numpy().astype(np.float64) for model in models], axis=0)
    return output


def final_known_metrics(true: np.ndarray, eeg: np.ndarray, log: np.ndarray,
                        tau_low: float, tau_high: float) -> dict:
    pred, direct, dr = decide(eeg, log, CLASSES, tau_low, tau_high)
    return {
        "N": int(len(true)), "direct_C": float(np.mean(direct == true)),
        "threshold_C": float(np.mean(pred == true)),
        "identity_20class_macro_F1_direct": float(f1_score(true, direct, labels=CLASSES, average="macro", zero_division=0)),
        "identity_20class_macro_F1_threshold": float(f1_score(true, pred, labels=CLASSES, average="macro", zero_division=0)),
        "known_acceptance": float(np.mean(pred != 0)), "rejection_rate": float(np.mean(pred == 0)),
        "logSPD_route_rate": float(np.mean(dr >= tau_high)),
        "EEGNet_route_rate": float(np.mean((dr >= tau_low) & (dr < tau_high))),
        "AUC": "N/A（final20无pseudo-unknown）",
    }


def combined_raw_scores(raw: torch.Tensor, models: list[torch.nn.Module], logspd: WCCNPrototype,
                        e36: dict, config: dict) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        features = gpu_correlation_logspd(raw, e36, config)
        log = logspd(features).cpu().numpy().astype(np.float64)
        eeg = np.mean([model.scores(raw).cpu().numpy().astype(np.float64) for model in models], axis=0)
    return eeg, log


def source_paths(config: dict) -> dict[str, Path]:
    return {
        "e65_config": CONFIG_PATH, "e65_code": Path(__file__).resolve(),
        "e36_config": ROOT / "research" / "E36_pure_eeg" / "config.json",
        "e36_code": ROOT / "research" / "E36_pure_eeg" / "run_e36.py",
        "e54_config": E54.OUT / "config.json", "e54_code": E54.OUT / "run_e54.py",
        "e64_model": ROOT / config["identity"]["e64_frozen_model"],
        "e64_manifest": ROOT / config["identity"]["e64_frozen_manifest"],
        "train_metadata": ROOT / config["data"]["train_metadata"], "correlation_features": CORR_PATH,
    }


def run_final20_and_test(config: dict, e36: dict, e54_config: dict, device: torch.device) -> dict:
    started = time.time()
    result = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    if not result.get("test_gates", {}).get("all_clean_passed", False):
        result["test"] = {"status": "N/A_clean_gate_failed", "test_accessed": False, "metrics": TEST_METRICS,
                          "reason": "至少一项预注册inner clean门禁失败；未读取Test，未生成预测"}
        write_json(RESULTS_PATH, result)
        write_report(result, config)
        print("E65 Test=N/A：inner clean门禁未全部通过", flush=True)
        return result
    if PREDICTIONS_PATH.exists():
        raise RuntimeError("E65一次性Test预测已存在，拒绝重跑或尝试变体")
    metadata = train_metadata(e36)
    blocks = final_blocks(metadata)
    subjects = metadata["subject"].to_numpy(dtype=int)
    raw = E54.load_all_raw(metadata, e54_config, device)
    correlation = torch.as_tensor(np.load(CORR_PATH, allow_pickle=False), dtype=torch.float32, device=device)
    models, e64_payload = load_e64_ensemble(config, e54_config, device)
    fit_np = blocks["fit"]
    logspd = WCCNPrototype(correlation[torch.as_tensor(fit_np, device=device)],
                           subjects[fit_np], CLASSES, config)
    scores = {"eeg": {}, "log": {}}
    for block, indices in blocks.items():
        scores["eeg"][block] = ensemble_scores(models, raw, indices, 256)
        scores["log"][block] = score_features(logspd, correlation, indices, 512)
    tau_low, tau_high = thresholds(scores["log"]["calibration"], "higher")
    final_metrics = {block: final_known_metrics(subjects[indices], scores["eeg"][block], scores["log"][block],
                                                tau_low, tau_high)
                     for block, indices in blocks.items()}
    final_dr = {block: dr_summary(distance_ratio(scores["log"][block])) for block in blocks}
    for block in ("calibration", "sanity"):
        final_dr[block]["drift_vs_fit"] = {key: float(final_dr[block][key] - final_dr["fit"][key])
                                           for key in ("mean", "q02", "q05", "median", "q95")}
    selected = blocks["sanity"][:8]
    batch_raw = raw[torch.as_tensor(selected, device=device)]
    batch_eeg, batch_log = combined_raw_scores(batch_raw, models, logspd, e36, config)
    singles = [combined_raw_scores(batch_raw[i:i + 1], models, logspd, e36, config) for i in range(8)]
    single_eeg = np.concatenate([row[0] for row in singles])
    single_log = np.concatenate([row[1] for row in singles])
    permutation = np.random.default_rng(int(config["seed"])).permutation(8)
    inverse = np.argsort(permutation)
    perm_eeg, perm_log = combined_raw_scores(batch_raw[permutation], models, logspd, e36, config)
    batch_pred = decide(batch_eeg, batch_log, CLASSES, tau_low, tau_high)[0]
    single_pred = decide(single_eeg, single_log, CLASSES, tau_low, tau_high)[0]
    perm_pred = decide(perm_eeg, perm_log, CLASSES, tau_low, tau_high)[0][inverse]
    numerical = {
        "N": 8, "single_vs_batch_EEGNet_max_abs": float(np.max(np.abs(batch_eeg - single_eeg))),
        "single_vs_batch_logSPD_max_abs": float(np.max(np.abs(batch_log - single_log))),
        "permutation_EEGNet_max_abs": float(np.max(np.abs(batch_eeg - perm_eeg[inverse]))),
        "permutation_logSPD_max_abs": float(np.max(np.abs(batch_log - perm_log[inverse]))),
        "single_vs_batch_labels_equal": bool(np.array_equal(batch_pred, single_pred)),
        "permutation_labels_equal": bool(np.array_equal(batch_pred, perm_pred)),
        "cross_sample_statistics_used": False,
    }
    numerical["passed"] = bool(np.allclose(batch_eeg, single_eeg, atol=1e-5, rtol=1e-5)
                               and np.allclose(batch_log, single_log, atol=1e-5, rtol=1e-5)
                               and np.allclose(batch_eeg, perm_eeg[inverse], atol=1e-5, rtol=1e-5)
                               and np.allclose(batch_log, perm_log[inverse], atol=1e-5, rtol=1e-5)
                               and numerical["single_vs_batch_labels_equal"] and numerical["permutation_labels_equal"])
    if not numerical["passed"]:
        raise RuntimeError(f"E65 final20逐样本数值检查失败：{numerical}")
    frozen = {"format": "E65 = unchanged E64 ten-backbone/20-class CNN ensemble + correlation logSPD/WCCN",
              "classes": CLASSES, "backbones": e64_payload["backbones"],
              "logspd": {"classes": logspd.classes, "whitening": logspd.whitening.cpu(),
                         "prototypes": logspd.prototypes.cpu()}}
    torch.save(frozen, MODEL_PATH)
    manifest = {
        "frozen": True, "experiment": "E65", "tau_low": tau_low, "tau_high": tau_high,
        "threshold_quantiles": {"low": 0.02, "high": 0.05, "method": "higher"},
        "decision": config["decision"]["rule"], "ensemble_weights": config["identity"]["ensemble_weights"],
        "inner_clean_gates": result["test_gates"], "model_sha256": sha256(MODEL_PATH),
        "source_sha256": {name: sha256(path) for name, path in source_paths(config).items()},
        "test_session_used_for_freeze": False, "post_audit_changes_forbidden": True,
    }
    write_json(FREEZE_MANIFEST, manifest)
    result["final20_train"] = {
        "blocks": final_metrics, "dr": final_dr, "numerical_invariance": numerical,
        "fit": "all 20 Train subjects <=60% only; E64 CNN encoders/prototypes unchanged",
        "calibration": "60-70% clean known correlation dr only", "sanity": "70-80% known continuous block",
        "tau_low": tau_low, "tau_high": tau_high,
    }
    result["freeze"] = {"model": str(MODEL_PATH), "manifest": str(FREEZE_MANIFEST),
                        "model_sha256": manifest["model_sha256"],
                        "completed_before_any_test_index_or_waveform_read": True}
    result["status"] = "final20_frozen_test_in_progress"
    write_json(RESULTS_PATH, result)

    # 冻结及全部指纹完成后才读取Test索引；此前没有Test访问。
    for name, path in source_paths(config).items():
        if sha256(path) != manifest["source_sha256"][name]:
            raise ValueError(f"E65冻结源指纹变化：{name} ({path})")
    sample_order = pd.read_csv(ROOT / config["data"]["test_csv"], usecols=["Sample"])
    paths = pd.read_csv(ROOT / config["data"]["test_metadata"], usecols=["sample", "path"])
    expected = int(config["data"]["expected_test_rows"])
    if len(sample_order) != expected or len(paths) != expected:
        raise ValueError(f"Test必须为{expected}行")
    if not np.array_equal(sample_order["Sample"].astype(str).to_numpy(), paths["sample"].astype(str).to_numpy()):
        raise ValueError("Test Sample与路径顺序不一致")
    if not paths["path"].astype(str).str.startswith("Test/").all():
        raise ValueError("Test索引含非Test路径")
    labels = np.empty(expected, dtype=np.int64)
    all_dr = np.empty(expected, dtype=np.float64)
    batch_size = int(config["identity"]["inference_batch_size"])
    test_root = ROOT / "Brainprint_Competition"
    with torch.no_grad():
        for start in range(0, expected, batch_size):
            stop = min(start + batch_size, expected)
            arrays = [load_train_matrix(test_root / str(relative))
                      for relative in paths["path"].iloc[start:stop]]
            batch = torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=device)
            eeg, log = combined_raw_scores(batch, models, logspd, e36, config)
            pred, _, dr = decide(eeg, log, CLASSES, tau_low, tau_high)
            labels[start:stop] = pred
            all_dr[start:stop] = dr
            if stop % 1024 < batch_size or stop == expected:
                print(f"E65 Test独立推理 {stop}/{expected}", flush=True)
    if not np.isin(labels, np.arange(21)).all() or not np.isfinite(all_dr).all():
        raise RuntimeError("E65 Test预测无效")
    submission = pd.DataFrame({"Sample": sample_order["Sample"].to_numpy(), "Label": labels})
    submission.to_csv(PREDICTIONS_PATH, index=False)
    if len(pd.read_csv(PREDICTIONS_PATH)) != expected:
        raise RuntimeError("E65预测CSV行数校验失败")

    # 全部预测与CSV完成后，才首次读取Session做描述性审计。
    sessions = pd.read_csv(ROOT / config["data"]["test_metadata"], usecols=["sample", "session"])
    if not np.array_equal(sample_order["Sample"].astype(str).to_numpy(), sessions["sample"].astype(str).to_numpy()):
        raise ValueError("Session审计顺序不一致")
    def session_code(value: object) -> str:
        text = str(value).strip().casefold().replace("_", "").replace("-", "")
        mapping = {"1": "S1", "s1": "S1", "session1": "S1", "2": "S2", "s2": "S2", "session2": "S2"}
        if text not in mapping:
            raise ValueError(f"未知Test Session：{value}")
        return mapping[text]
    codes = sessions["session"].map(session_code).to_numpy()
    audit = {"N": expected, "overall_rejection_rate": float(np.mean(labels == 0)),
             "session_used_by_inference": False, "audit_after_predictions_and_csv": True,
             "S1_rejection_target": 0.95, "S2_acceptance_target": 0.98}
    for code in ("S1", "S2"):
        selected_mask = codes == code
        rejection = float(np.mean(labels[selected_mask] == 0))
        audit[f"{code}_N"] = int(selected_mask.sum())
        audit[f"{code}_rejection_rate"] = rejection
        audit[f"{code}_acceptance_rate"] = 1.0 - rejection
    audit["S1_rejection_ge_0.95"] = bool(audit["S1_rejection_rate"] >= 0.95)
    audit["S2_acceptance_ge_0.98"] = bool(audit["S2_acceptance_rate"] >= 0.98)
    result["status"] = "complete_one_shot_test_audit"
    result["test"] = {
        "status": "complete_one_shot_frozen_engineering_audit", "test_accessed": True,
        "predictions_csv": str(PREDICTIONS_PATH), "prediction_rows": expected,
        "single_trial_independent": True, "batch_statistics_or_test_calibration": False,
        "test_labels_read_or_used": False, "metrics": TEST_METRICS,
        "offline_session_audit_after_prediction": audit, "post_audit_threshold_or_variant_change": False,
        "test_elapsed_seconds": time.time() - started,
    }
    write_json(RESULTS_PATH, result)
    write_report(result, config)
    print(f"E65 Test审计：S1Rej={audit['S1_rejection_rate']:.4f} "
          f"S2Accept={audit['S2_acceptance_rate']:.4f}", flush=True)
    del raw, correlation, models
    torch.cuda.empty_cache()
    return result


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_report(result: dict, config: dict) -> None:
    if "summary" not in result:
        return
    summary = result["summary"]
    corr = summary["e65_correlation"]
    clean = corr["clean_outer_mean"]
    attack = corr["attack_outer_mean"]
    gates = result["test_gates"]
    lines = [
        "# E65：逐通道相关矩阵 logSPD / WCCN（E64 拒识表征修正）",
        "",
        "日期：2026-11-13。E64 的 S1 拒识/S2 接受失败只用于确定表征问题；未用于任何 Test 阈值或变体选择。",
        "",
        "## 结论与当前 AUC / F1",
        "",
        f"Correlation inner 五外层折均值：**direct C={fmt(clean['direct_C'])}，threshold C={fmt(clean['threshold_C'])}，direct身份F1={fmt(clean['identity_macro_F1_direct'])}，含拒识身份F1={fmt(clean['identity_macro_F1_threshold'])}，clean AUROC={fmt(clean['AUROC'])}，unknown AUPR={fmt(clean['unknown_AUPR'])}，U@95={fmt(clean['U_at_95'])}，工作点U={fmt(clean['working_U'])}**。",
        "",
        f"CNN-only 攻击宏均值：RA={fmt(attack['RA'])}，ASafeR={fmt(attack['ASafeR'])}。这不是全链路自适应攻击证据，不能据此宣称全部门禁通过。",
        "",
        "## 冻结方法",
        "",
        "- 唯一主变量：每个 trial、每个 FFT 频带内，18 个通道分别去均值并除以自身样本标准差（ddof=1），再计算 18×18 相关矩阵。禁止跨 trial、跨 Test 批次统计。",
        "- 后续严格沿用 E36：0.05 固定 SPD 收缩、特征值 floor=1e-12、logm、上三角（非对角乘 sqrt(2)）、detrend+zband；WCCN shrinkage=0.05、eigenvalue floor=1e-6。",
        "- 对照为同折 E36 原始协方差 logSPD/WCCN；两方案共享 E54 OE-EEGNet 身份分数、样本、攻击波形和角色，只各自重拟合 WCCN 并从 clean known dr 计算 q02/q05。",
        "- 十个 inner run 为 5 outer groups × 2 inner folds；每 run 的 outer/enrolled/OE-only/calibration 为 4/8/4/4 互斥被试。known 为 60–80% 连续块，每被试/任务排序取前8条；unknown 仅 calibration 四人。",
        "- 三段式：dr≥tau_high 用 logSPD 身份；中间区用 OE-EEGNet；dr<tau_low 拒识。inner 分位数 method=linear；U@95 是 dr<clean-known-q05 的 unknown 比例，工作点U来自实际 q02 拒识。",
        "- 攻击复现 E58 的 CNN-only 白盒 `raw_attack(..., steps=20)` 调用（含保留历史名称的六个攻击）；只攻击 OE-EEGNet，再冻结波形送入两种 logSPD。**不是对 logSPD+路由+EEGNet 的全链路自适应攻击**；E65 未新增对抗训练。",
        "",
        "## Train inner 五外层折均值",
        "",
        "| 表征 | direct C | threshold C | direct身份F1 | 含拒识身份F1 | 连续块 | AUROC | unknown AUPR | U@95 | 工作点U | unknown F1 | joint F1 | RA | ASafeR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row = summary[variant]["clean_outer_mean"]
        adv = summary[variant]["attack_outer_mean"]
        lines.append(f"| {variant} | {fmt(row['direct_C'])} | {fmt(row['threshold_C'])} | "
                     f"{fmt(row['identity_macro_F1_direct'])} | {fmt(row['identity_macro_F1_threshold'])} | "
                     f"{fmt(row['continuous_block_accuracy'])} | {fmt(row['AUROC'])} | {fmt(row['unknown_AUPR'])} | "
                     f"{fmt(row['U_at_95'])} | {fmt(row['working_U'])} | {fmt(row['unknown_F1'])} | "
                     f"{fmt(row['joint_F1'])} | {fmt(adv['RA'])} | {fmt(adv['ASafeR'])} |")
    lines.extend([
        "",
        "Pooled clean 结果（跨 run 固定1–20/0–20标签；每 run 阈值不变）见 results.json。逐折比较：",
        "",
        "| outer | 表征 | direct C | threshold C | direct F1 | 含拒识F1 | AUROC | AUPR | U@95 | U | RA | ASafeR |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for fold in result["outer_folds"]:
        for variant in VARIANTS:
            row = fold["variants"][variant]["clean"]
            adv = fold["variants"][variant]["attack_macro"]
            lines.append(f"| {fold['outer_fold']} | {variant} | {fmt(row['direct_C'])} | {fmt(row['threshold_C'])} | "
                         f"{fmt(row['identity_macro_F1_direct'])} | {fmt(row['identity_macro_F1_threshold'])} | "
                         f"{fmt(row['AUROC'])} | {fmt(row['unknown_AUPR'])} | {fmt(row['U_at_95'])} | "
                         f"{fmt(row['working_U'])} | {fmt(adv['RA'])} | {fmt(adv['ASafeR'])} |")
    lines.extend([
        "",
        "### 分攻击（五外层折均值）",
        "",
        "| 表征 | 攻击 | RA | ASafeR | direct RA | 拒识率 | N/run |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for variant in VARIANTS:
        for name, row in summary[variant]["per_attack_outer_mean"].items():
            lines.append(f"| {variant} | {name} | {fmt(row['RA'])} | {fmt(row['ASafeR'])} | "
                         f"{fmt(row['direct_RA'])} | {fmt(row['rejection_rate'])} | {row['N']:.0f} |")
    lines.extend([
        "",
        "## 连续块 dr 漂移（十个 inner run 均值的分布统计）",
        "",
        "漂移为当前块统计减去同 run 的 fit≤60% 统计；不是跨 Session 漂移。完整 q95/std/N 见 JSON。",
        "",
        "| 表征 | 块 | dr mean | q02 | q05 | median | mean漂移 | q02漂移 | q05漂移 | median漂移 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for variant in VARIANTS:
        for block, title in (("fit", "fit≤60%"), ("calibration", "60–70%"), ("sanity", "70–80%")):
            item = summary[variant]["dr_drift"][block]
            row, delta = item["run_mean"], item["drift_vs_fit_run_mean"]
            lines.append(f"| {variant} | {title} | {fmt(row['mean'])} | {fmt(row['q02'])} | {fmt(row['q05'])} | "
                         f"{fmt(row['median'])} | {fmt(delta['mean'])} | {fmt(delta['q02'])} | "
                         f"{fmt(delta['q05'])} | {fmt(delta['median'])} |")
    lines.extend([
        "",
        "## 预注册 Test 门禁",
        "",
        "| 门禁（correlation五折clean均值） | 实测 | 最低值 | 结果 |",
        "|---|---:|---:|---|",
    ])
    for name, row in gates["checks"].items():
        lines.append(f"| {name} | {fmt(row['value'])} | {fmt(row['minimum'])} | {'通过' if row['passed'] else '未通过'} |")
    lines.extend([
        "",
        f"四项 clean 门禁全部通过：**{gates['all_clean_passed']}**。RA/ASafeR 不阻止工程审计，但由于攻击不是全链路自适应，不宣称 Train 全部门禁通过。",
        "",
    ])
    if "final20_train" in result:
        final = result["final20_train"]
        lines.extend([
            "## 获准后的 final20 Train 冻结",
            "",
            f"保持 E64 十 backbone/20 类 CNN ensemble 不变，仅替换 correlation-logSPD/WCCN 与 dr；fit≤60% 拟合，60–70% clean known 以 method=higher 得到 tau_low={final['tau_low']:.9f}、tau_high={final['tau_high']:.9f}，70–80% sanity。",
            "",
            "| 块 | N | direct C | threshold C | direct F1 | 含拒识F1 | 拒识率 | dr median | median相对fit漂移 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for block in ("fit", "calibration", "sanity"):
            row = final["blocks"][block]
            dr = final["dr"][block]
            shift = dr.get("drift_vs_fit", {}).get("median", 0.0)
            lines.append(f"| {block} | {row['N']} | {fmt(row['direct_C'])} | {fmt(row['threshold_C'])} | "
                         f"{fmt(row['identity_20class_macro_F1_direct'])} | {fmt(row['identity_20class_macro_F1_threshold'])} | "
                         f"{fmt(row['rejection_rate'])} | {fmt(dr['median'])} | {fmt(shift)} |")
        lines.append("")
    lines.extend(["## Test 执行状态与工程审计", ""])
    test = result["test"]
    if test["status"] == "complete_one_shot_frozen_engineering_audit":
        audit = test["offline_session_audit_after_prediction"]
        lines.extend([
            f"已按门禁执行一次冻结 Test；`{PREDICTIONS_PATH}` 共 {test['prediction_rows']} 行。Session 仅在全部预测和 CSV 落盘后读取。",
            "",
            "| Session | N | 拒识率 | 接受率 | 目标 | 判定 |",
            "|---|---:|---:|---:|---|---|",
            f"| S1 | {audit['S1_N']} | {fmt(audit['S1_rejection_rate'])} | {fmt(audit['S1_acceptance_rate'])} | 拒识≥95% | {'达到' if audit['S1_rejection_ge_0.95'] else '未达到'} |",
            f"| S2 | {audit['S2_N']} | {fmt(audit['S2_rejection_rate'])} | {fmt(audit['S2_acceptance_rate'])} | 接受≥98% | {'达到' if audit['S2_acceptance_ge_0.98'] else '未达到'} |",
            "",
            "审计后未调整阈值、模型或变体。",
        ])
    elif gates["all_clean_passed"]:
        lines.append("四项 clean 门禁通过，已授权一次冻结 Test 工程审计；当前报告生成时预测尚未完成。")
    else:
        failed = "、".join(name for name, row in gates["checks"].items() if not row["passed"])
        lines.append(f"**Test=N/A，未读取 Test，未生成 e65_predictions.csv**。原因：{failed} 未达到预注册门禁；没有尝试任何 Test 变体。")
    lines.extend([
        "",
        "**Test AUC/F1/C/RA/ASafeR/U/Score=N/A（无真实标签）**。S1拒识/S2接受若存在仅为预测工程审计，不是官方真值指标。",
        "",
        "## 证据边界与复现",
        "",
        "- 所有 AUC/F1/C/U/RA/ASafeR 数值均来自 Train Session1 的 pseudo-unknown 与连续块代理，不是跨 Session 或官方新被试证据。fit 块 dr 为拟合内诊断，calibration/sanity 为连续块迁移诊断。",
        "- Correlation 与 covariance 的差值是同折配对描述，不涉及 Test 调参。E64 的历史 Test 结果只用于提出表征问题。",
        "- 特征与 final20（若执行）均检查单条/批次/置换一致性；GPU batch 仅加速，不使用批次统计。",
        "",
        "```powershell",
        f"& \"{ROOT / '.venv' / 'Scripts' / 'python.exe'}\" \"{OUT / 'run_e65.py'}\" --stage inner",
        "# 仅当上述结果四项 clean 门禁全部通过：",
        f"& \"{ROOT / '.venv' / 'Scripts' / 'python.exe'}\" \"{OUT / 'run_e65.py'}\" --stage test",
        "```",
        "",
        f"配置：`{CONFIG_PATH}`；结果：`{RESULTS_PATH}`；缓存：`{CACHE}`。断点绑定配置和代码 SHA256。",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("features", "inner", "test", "all"), default="inner")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("E65要求CUDA GPU")
    config, e36, e54_config = load_config(), load_e36_config(), E54.load_config()
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    device = torch.device("cuda")
    if args.stage == "features":
        build_features(config, e36, device)
        return
    result = None
    if args.stage in ("inner", "all"):
        result = run_inner(config, e36, e54_config, device, not args.no_resume)
    if args.stage == "test" or (args.stage == "all" and result["test_gates"]["all_clean_passed"]):
        run_final20_and_test(config, e36, e54_config, device)
    print(f"结果：{RESULTS_PATH}\n报告：{REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
