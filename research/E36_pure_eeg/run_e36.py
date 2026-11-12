"""E36A/B：仅从 Train MAT 构建五频带 logSPD，并执行严格嵌套开放集评估。"""
from __future__ import annotations

import argparse
import hashlib
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
from scipy.io import loadmat
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CONFIG_PATH = OUT / "config.json"
CACHE = ROOT / "BrainPrint_cache" / "e36_pure_eeg"
RAW_PATH = CACHE / "train_raw_cov5.npy"
ZBAND_PATH = CACHE / "train_detrend_zband_cov5.npy"
FEATURE_REPORT_PATH = OUT / "feature_build.json"
RESULTS_PATH = OUT / "results.json"
CANDIDATES = (
    "raw_cov5_ridge",
    "detrend_zband_cov5_ridge",
    "detrend_zband_cov5_prototype",
    "detrend_zband_cov5_wccn_prototype",
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def train_metadata(config: dict) -> pd.DataFrame:
    metadata = pd.read_csv(ROOT / config["data"]["train_metadata"])
    expected = config["data"]
    if len(metadata) != expected["expected_trials"]:
        raise ValueError(f"Train 数量 {len(metadata)} != {expected['expected_trials']}")
    if metadata["subject"].nunique() != expected["expected_subjects"]:
        raise ValueError("Train 被试数不符合冻结配置")
    if set(metadata["session"]) != {expected["session"]}:
        raise ValueError("本实验只允许 Train session1")
    if set(metadata["fs"]) != {expected["sampling_rate_hz"]}:
        raise ValueError("采样率不符合冻结配置")
    if not metadata["path"].str.startswith("Train/").all():
        raise ValueError("元数据含非 Train 路径")
    return metadata


def load_train_matrix(path: Path) -> np.ndarray:
    data = np.asarray(loadmat(path, variable_names=["data"])["data"], dtype=np.float32)
    if data.shape != (1200, 18):
        raise ValueError(f"{path}: 预期 (1200,18)，实际 {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: 含非有限值")
    return data.T


def gpu_logspd(raw: torch.Tensor, config: dict) -> torch.Tensor:
    """raw: B×18×1200；返回 B×(5×171)，全部信号计算在 GPU。"""
    cfg = config["features"]
    n_channels = config["data"]["channels"]
    n_samples = config["data"]["samples_per_trial"]
    fs = config["data"]["sampling_rate_hz"]
    frequencies = torch.fft.rfftfreq(n_samples, d=1.0 / fs, device=raw.device)
    masks = []
    for band_index, (low, high) in enumerate(cfg["bands_hz"]):
        upper = frequencies <= high if band_index == len(cfg["bands_hz"]) - 1 else frequencies < high
        masks.append((frequencies >= low) & upper)
    mask = torch.stack(masks).to(dtype=raw.dtype)
    spectrum = torch.fft.rfft(raw, dim=-1)
    filtered = torch.fft.irfft(
        spectrum[:, None, :, :] * mask[None, :, None, :], n=n_samples, dim=-1
    )
    filtered = filtered - filtered.mean(dim=-1, keepdim=True)
    covariance = filtered @ filtered.transpose(-1, -2)
    covariance = covariance / (n_samples - 1)
    shrinkage = float(cfg["covariance_shrinkage"])
    trace = torch.diagonal(covariance, dim1=-2, dim2=-1).sum(dim=-1)
    eye = torch.eye(n_channels, dtype=raw.dtype, device=raw.device)
    covariance = (
        (1.0 - shrinkage) * covariance
        + shrinkage * trace[..., None, None] / n_channels * eye
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(float(cfg["eigenvalue_floor"]))
    logspd = (eigenvectors * eigenvalues.log().unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)
    rows, columns = torch.triu_indices(n_channels, n_channels, device=raw.device)
    vectors = logspd[..., rows, columns]
    off_diagonal = rows != columns
    vectors[..., off_diagonal] *= float(cfg["off_diagonal_multiplier"])
    return vectors.flatten(1)


def detrend_zband(raw: torch.Tensor, config: dict) -> torch.Tensor:
    n_channels = config["data"]["channels"]
    n_bands = len(config["features"]["bands_hz"])
    bands = raw.reshape(*raw.shape[:-1], n_bands, -1).clone()
    rows, columns = np.triu_indices(n_channels)
    diagonal = torch.as_tensor(rows == columns, device=raw.device)
    diagonal_mean = bands[..., diagonal].mean(dim=-1, keepdim=True)
    bands[..., diagonal] = bands[..., diagonal] - diagonal_mean
    mean = bands.mean(dim=-1, keepdim=True)
    scale = bands.std(dim=-1, keepdim=True, unbiased=False).clamp_min(
        float(config["features"]["zband_epsilon"])
    )
    return ((bands - mean) / scale).flatten(-2)


def extract_metadata_batch(
    paths: list[Path], config: dict, device: torch.device
) -> torch.Tensor:
    arrays = [load_train_matrix(path) for path in paths]
    raw = torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=device)
    return gpu_logspd(raw, config)


def feature_invariance(
    metadata: pd.DataFrame, raw_cached: np.ndarray, z_cached: np.ndarray,
    config: dict, device: torch.device,
) -> dict:
    rng = np.random.default_rng(config["invariance"]["seed"])
    selected = np.sort(rng.choice(len(metadata), size=16, replace=False))
    paths = [ROOT / config["data"]["train_root"] / metadata.iloc[index]["path"] for index in selected]
    batch = extract_metadata_batch(paths, config, device)
    singles = torch.cat([
        extract_metadata_batch([path], config, device) for path in paths
    ])
    permutation = rng.permutation(len(selected))
    permuted = extract_metadata_batch([paths[index] for index in permutation], config, device)
    inverse = np.argsort(permutation)
    rebuilt = batch.cpu().numpy()
    cached = raw_cached[selected]
    z_rebuilt = detrend_zband(batch, config).cpu().numpy()
    return {
        "n_trials": len(selected),
        "cache_vs_rebuilt_max_abs": float(np.max(np.abs(cached - rebuilt))),
        "single_vs_batch_max_abs": float((batch - singles).abs().max().item()),
        "permutation_vs_batch_max_abs": float((batch - permuted[inverse]).abs().max().item()),
        "zband_vs_rebuilt_max_abs": float(np.max(np.abs(z_cached[selected] - z_rebuilt))),
        "cache_allclose_atol_1e-5": bool(np.allclose(cached, rebuilt, atol=1e-5, rtol=1e-5)),
    }


def build_features(config: dict, device: torch.device, rebuild: bool = False) -> dict:
    metadata = train_metadata(config)
    if not rebuild and RAW_PATH.exists() and ZBAND_PATH.exists() and FEATURE_REPORT_PATH.exists():
        print(f"复用 E36 五频带缓存：{RAW_PATH}", flush=True)
        return json.loads(FEATURE_REPORT_PATH.read_text(encoding="utf-8"))
    started = time.time()
    paths = [ROOT / config["data"]["train_root"] / path for path in metadata["path"]]
    batch_size = int(config["features"]["io_batch_size"])
    raw_output = np.lib.format.open_memmap(
        RAW_PATH, mode="w+", dtype=np.float32,
        shape=(len(metadata), 5 * 171),
    )
    with ThreadPoolExecutor(max_workers=int(config["features"]["io_workers"])) as pool:
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            arrays = list(pool.map(load_train_matrix, batch_paths))
            raw = torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=device)
            with torch.no_grad():
                features = gpu_logspd(raw, config)
            raw_output[start : start + len(arrays)] = features.cpu().numpy()
            completed = min(start + batch_size, len(paths))
            if completed % 1024 == 0 or completed == len(paths):
                print(
                    f"features {completed}/{len(paths)} | {time.time() - started:.1f}s",
                    flush=True,
                )
    raw_output.flush()
    del raw_output
    raw_array = np.load(RAW_PATH, mmap_mode="r")
    if not np.isfinite(raw_array).all():
        raise RuntimeError("五频带 raw 特征含非有限值")
    raw_gpu = torch.as_tensor(np.asarray(raw_array), device=device)
    z_gpu = detrend_zband(raw_gpu, config)
    np.save(ZBAND_PATH, z_gpu.cpu().numpy())
    z_array = np.load(ZBAND_PATH, mmap_mode="r")
    z_bands = z_array.reshape(len(z_array), 5, 171)
    invariance = feature_invariance(metadata, raw_array, z_array, config, device)
    report = {
        "data": "Train session1 only; Test not accessed",
        "n_trials": len(metadata),
        "shape": list(raw_array.shape),
        "dtype": str(raw_array.dtype),
        "bands_hz": config["features"]["bands_hz"],
        "raw_cache": str(RAW_PATH),
        "zband_cache": str(ZBAND_PATH),
        "raw_finite": bool(np.isfinite(raw_array).all()),
        "zband_finite": bool(np.isfinite(z_array).all()),
        "zband_max_abs_mean": float(np.abs(z_bands.mean(axis=-1)).max()),
        "zband_max_abs_std_error": float(np.abs(z_bands.std(axis=-1) - 1.0).max()),
        "invariance": invariance,
        "elapsed_seconds": time.time() - started,
        "raw_sha256": sha256(RAW_PATH),
        "zband_sha256": sha256(ZBAND_PATH),
        "metadata_sha256": sha256(ROOT / config["data"]["train_metadata"]),
        "config_sha256": sha256(CONFIG_PATH),
        "gpu": torch.cuda.get_device_name(0),
    }
    write_json(FEATURE_REPORT_PATH, report)
    print(json.dumps(report["invariance"], ensure_ascii=False, indent=2), flush=True)
    return report


class TorchRidge:
    def __init__(self, x: torch.Tensor, y: np.ndarray, classes: np.ndarray, alpha: float):
        self.classes = np.asarray(classes, dtype=int)
        self.mean = x.mean(dim=0)
        self.scale = x.std(dim=0, unbiased=False).clamp_min(1e-6)
        scaled = (x - self.mean) / self.scale
        class_to_index = {label: index for index, label in enumerate(self.classes)}
        target_indices = torch.as_tensor(
            [class_to_index[int(label)] for label in y], dtype=torch.long, device=x.device
        )
        targets = torch.full(
            (len(y), len(classes)), -1.0, dtype=torch.float32, device=x.device
        )
        targets[torch.arange(len(y), device=x.device), target_indices] = 1.0
        self.target_mean = targets.mean(dim=0)
        centered = targets - self.target_mean
        if len(y) < scaled.shape[1]:
            kernel = scaled @ scaled.T
            kernel.diagonal().add_(alpha)
            self.coef = scaled.T @ torch.linalg.solve(kernel, centered)
        else:
            gram = scaled.T @ scaled
            gram.diagonal().add_(alpha)
            self.coef = torch.linalg.solve(gram, scaled.T @ centered)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / self.scale) @ self.coef + self.target_mean


class TorchPrototype:
    def __init__(
        self, x: torch.Tensor, y: np.ndarray, classes: np.ndarray,
        wccn: bool, shrinkage: float, eigenvalue_floor: float,
    ):
        self.classes = np.asarray(classes, dtype=int)
        class_to_index = {label: index for index, label in enumerate(self.classes)}
        target_indices = torch.as_tensor(
            [class_to_index[int(label)] for label in y], dtype=torch.long, device=x.device
        )
        means = torch.stack([x[target_indices == index].mean(dim=0) for index in range(len(classes))])
        self.whitening = None
        if wccn:
            residual = x - means[target_indices]
            covariance = residual.T @ residual / max(len(x) - len(classes), 1)
            trace = torch.diagonal(covariance).sum()
            eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
            covariance = (
                (1.0 - shrinkage) * covariance
                + shrinkage * trace / x.shape[1] * eye
            )
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            inverse_root = eigenvalues.clamp_min(eigenvalue_floor).rsqrt()
            self.whitening = (eigenvectors * inverse_root.unsqueeze(0)) @ eigenvectors.T
            means = means @ self.whitening
        self.prototypes = means / means.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.whitening is not None:
            x = x @ self.whitening
        normalized = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return normalized @ self.prototypes.T


def fit_models(
    raw: torch.Tensor, zband: torch.Tensor, y: np.ndarray,
    classes: np.ndarray, config: dict,
) -> dict:
    return {
        "raw_cov5_ridge": TorchRidge(raw, y, classes, config["models"]["ridge_alpha"]),
        "detrend_zband_cov5_ridge": TorchRidge(zband, y, classes, config["models"]["ridge_alpha"]),
        "detrend_zband_cov5_prototype": TorchPrototype(
            zband, y, classes, False,
            config["models"]["wccn_shrinkage"], config["models"]["wccn_eigenvalue_floor"],
        ),
        "detrend_zband_cov5_wccn_prototype": TorchPrototype(
            zband, y, classes, True,
            config["models"]["wccn_shrinkage"], config["models"]["wccn_eigenvalue_floor"],
        ),
    }


def predict_scores(
    models: dict, raw: torch.Tensor, zband: torch.Tensor,
    indices: torch.Tensor, config: dict,
) -> dict[str, np.ndarray]:
    classes = models[CANDIDATES[0]].classes
    outputs = {
        name: np.empty((len(indices), len(classes)), dtype=np.float32)
        for name in CANDIDATES
    }
    batch_size = int(config["models"]["prediction_batch_size"])
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            batches = {
                "raw_cov5_ridge": raw[selected],
                "detrend_zband_cov5_ridge": zband[selected],
                "detrend_zband_cov5_prototype": zband[selected],
                "detrend_zband_cov5_wccn_prototype": zband[selected],
            }
            for name, model in models.items():
                outputs[name][start : start + len(selected)] = model(batches[name]).cpu().numpy()
    return outputs


def confidence_values(scores: np.ndarray, candidate: str) -> dict[str, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    ordered = np.sort(scores, axis=1)
    maximum = ordered[:, -1]
    second = ordered[:, -2]
    margin = maximum - second
    if candidate.endswith("prototype"):
        distance_ratio = (1.0 - second) / np.maximum(1.0 - maximum, 1e-6)
        return {"max_cosine": maximum, "margin": margin, "distance_ratio": distance_ratio}
    centered = scores - maximum[:, None]
    energy = maximum + np.log(np.exp(centered).sum(axis=1))
    return {"max": maximum, "margin": margin, "energy": energy}


def threshold_metrics(
    known_confidence: np.ndarray, unknown_confidence: np.ndarray,
    known_direct_correct: np.ndarray, threshold: float,
) -> dict:
    accepted = known_confidence >= threshold
    rejected_unknown = unknown_confidence < threshold
    c = float(np.mean(known_direct_correct & accepted))
    u = float(np.mean(rejected_unknown))
    detection_true = np.concatenate([
        np.ones(len(known_confidence), dtype=int),
        np.zeros(len(unknown_confidence), dtype=int),
    ])
    accepted_all = np.concatenate([accepted, ~rejected_unknown])
    return {
        "threshold": float(threshold),
        "C": c,
        "U": u,
        "known_acceptance": float(np.mean(accepted)),
        "unknown_f1": float(f1_score(
            detection_true, accepted_all, pos_label=0, zero_division=0
        )),
        "C_U_proxy_not_official": float(100 * c**0.40 * u**0.25) if c > 0 and u > 0 else 0.0,
    }


def select_operating_point(inner_runs: list[dict], candidate: str, config: dict) -> dict:
    known_scores = [run["known_scores"][candidate] for run in inner_runs]
    unknown_scores = [run["unknown_scores"][candidate] for run in inner_runs]
    known_direct_correct = np.concatenate([
        run["classes"][run["known_scores"][candidate].argmax(axis=1)] == run["known_true"]
        for run in inner_runs
    ])
    score_rows = []
    grid_size = int(config["nested_protocol"]["threshold_grid_points"])
    quantiles = np.linspace(0.0, 0.05, grid_size)
    for score_name in config["models"]["candidates"][candidate]:
        known = np.concatenate([
            confidence_values(scores, candidate)[score_name] for scores in known_scores
        ])
        unknown = np.concatenate([
            confidence_values(scores, candidate)[score_name] for scores in unknown_scores
        ])
        detection_true = np.concatenate([np.ones(len(known)), np.zeros(len(unknown))])
        detection_confidence = np.concatenate([known, unknown])
        auroc = float(roc_auc_score(detection_true, detection_confidence))
        unknown_aupr = float(average_precision_score(1 - detection_true, -detection_confidence))
        work_rows = []
        for threshold in np.quantile(known, quantiles):
            row = threshold_metrics(known, unknown, known_direct_correct, float(threshold))
            if row["known_acceptance"] >= 0.95 - 1e-12:
                work_rows.append(row)
        work = max(
            work_rows,
            key=lambda row: (
                row["C_U_proxy_not_official"], row["C"], row["U"], auroc,
            ),
        )
        q05 = threshold_metrics(
            known, unknown, known_direct_correct, float(np.quantile(known, 0.05))
        )
        score_rows.append({
            "score_name": score_name,
            "inner_auroc": auroc,
            "inner_unknown_aupr": unknown_aupr,
            "work": work,
            "q05": q05,
        })
    selected = max(
        score_rows,
        key=lambda row: (
            row["work"]["C_U_proxy_not_official"], row["work"]["C"],
            row["work"]["U"], row["inner_auroc"],
        ),
    )
    return {
        "score_name": selected["score_name"],
        "threshold": selected["work"]["threshold"],
        "q05_threshold": selected["q05"]["threshold"],
        "inner_work": selected["work"],
        "inner_q05": selected["q05"],
        "inner_auroc": selected["inner_auroc"],
        "inner_unknown_aupr": selected["inner_unknown_aupr"],
        "all_score_candidates": score_rows,
    }


def detection_summary(known: np.ndarray, unknown: np.ndarray) -> dict:
    labels = np.concatenate([np.ones(len(known), dtype=int), np.zeros(len(unknown), dtype=int)])
    confidence = np.concatenate([known, unknown])
    fpr, tpr, _ = roc_curve(labels, confidence)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return {
        "unknown_auroc": float(roc_auc_score(labels, confidence)),
        "unknown_aupr": float(average_precision_score(1 - labels, -confidence)),
        "known_aupr": float(average_precision_score(labels, confidence)),
        "eer": float((fpr[index] + fnr[index]) / 2),
    }


def evaluate_outer(
    known_true: np.ndarray, known_scores: np.ndarray, unknown_scores: np.ndarray,
    classes: np.ndarray, candidate: str, selection: dict,
) -> tuple[dict, dict]:
    score_name = selection["score_name"]
    known_confidence = confidence_values(known_scores, candidate)[score_name]
    unknown_confidence = confidence_values(unknown_scores, candidate)[score_name]
    direct_known = classes[known_scores.argmax(axis=1)]
    direct_unknown = classes[unknown_scores.argmax(axis=1)]
    report = {
        "n_known": int(len(known_true)),
        "n_unknown": int(len(unknown_scores)),
        "direct_C": float(np.mean(direct_known == known_true)),
        "continuous_block_accuracy": float(np.mean(direct_known == known_true)),
        "identity_macro_f1_direct": float(f1_score(
            known_true, direct_known, labels=classes, average="macro", zero_division=0
        )),
        **detection_summary(known_confidence, unknown_confidence),
    }
    records = {
        "known_true": known_true,
        "direct_known": direct_known,
        "direct_unknown": direct_unknown,
        "known_confidence": known_confidence,
        "unknown_confidence": unknown_confidence,
    }
    for prefix, threshold_key in (("", "threshold"), ("q05_", "q05_threshold")):
        threshold = selection[threshold_key]
        known_pred = direct_known.copy()
        unknown_pred = direct_unknown.copy()
        known_pred[known_confidence < threshold] = 0
        unknown_pred[unknown_confidence < threshold] = 0
        detection_true = np.concatenate([
            np.ones(len(known_true), dtype=int), np.zeros(len(unknown_pred), dtype=int)
        ])
        accepted = np.concatenate([
            known_confidence >= threshold, unknown_confidence >= threshold
        ])
        c = float(np.mean(known_pred == known_true))
        u = float(np.mean(unknown_pred == 0))
        report.update({
            f"{prefix}C": c,
            f"{prefix}U": u,
            f"{prefix}known_acceptance": float(np.mean(known_confidence >= threshold)),
            f"{prefix}identity_macro_f1": float(f1_score(
                known_true, known_pred, labels=classes, average="macro", zero_division=0
            )),
            f"{prefix}unknown_f1": float(f1_score(
                detection_true, accepted, pos_label=0, zero_division=0
            )),
            f"{prefix}joint_fold_macro_f1": float(f1_score(
                np.concatenate([known_true, np.zeros(len(unknown_pred), dtype=int)]),
                np.concatenate([known_pred, unknown_pred]),
                labels=np.concatenate([[0], classes]), average="macro", zero_division=0,
            )),
            f"{prefix}C_U_proxy_not_official": float(100 * c**0.40 * u**0.25) if c > 0 and u > 0 else 0.0,
        })
        records[f"{prefix}known_pred"] = known_pred
        records[f"{prefix}unknown_pred"] = unknown_pred
    return report, records


FOLD_METRICS = (
    "train_fit_accuracy", "train_fit_identity_macro_f1", "direct_C",
    "continuous_block_accuracy", "identity_macro_f1_direct", "unknown_auroc",
    "unknown_aupr", "known_aupr", "eer", "C", "U", "known_acceptance",
    "identity_macro_f1", "unknown_f1", "joint_fold_macro_f1",
    "C_U_proxy_not_official", "q05_C", "q05_U", "q05_known_acceptance",
    "q05_identity_macro_f1", "q05_unknown_f1", "q05_joint_fold_macro_f1",
)


def pooled_metrics(records: list[dict]) -> dict:
    known_true = np.concatenate([row["known_true"] for row in records])
    direct_known = np.concatenate([row["direct_known"] for row in records])
    direct_unknown = np.concatenate([row["direct_unknown"] for row in records])
    known_confidence = np.concatenate([row["known_confidence"] for row in records])
    unknown_confidence = np.concatenate([row["unknown_confidence"] for row in records])
    n_unknown = sum(len(row["direct_unknown"]) for row in records)
    output = {
        "n_known": int(len(known_true)),
        "n_unknown": int(n_unknown),
        "direct_C": float(np.mean(direct_known == known_true)),
        "identity_20class_macro_f1_direct": float(f1_score(
            known_true, direct_known, labels=np.arange(1, 21), average="macro", zero_division=0
        )),
        **detection_summary(known_confidence, unknown_confidence),
    }
    for prefix in ("", "q05_"):
        known_pred = np.concatenate([row[f"{prefix}known_pred"] for row in records])
        unknown_pred = np.concatenate([row[f"{prefix}unknown_pred"] for row in records])
        true = np.concatenate([known_true, np.zeros(n_unknown, dtype=int)])
        pred = np.concatenate([known_pred, unknown_pred])
        is_known = np.concatenate([
            np.ones(len(known_true), dtype=int), np.zeros(n_unknown, dtype=int)
        ])
        c = float(np.mean(known_pred == known_true))
        u = float(np.mean(unknown_pred == 0))
        output.update({
            f"{prefix}C": c,
            f"{prefix}U": u,
            f"{prefix}known_acceptance": float(np.mean(known_pred != 0)),
            f"{prefix}identity_20class_macro_f1": float(f1_score(
                known_true, known_pred, labels=np.arange(1, 21), average="macro", zero_division=0
            )),
            f"{prefix}unknown_f1": float(f1_score(
                is_known, pred != 0, pos_label=0, zero_division=0
            )),
            f"{prefix}joint_21class_macro_f1": float(f1_score(
                true, pred, labels=np.arange(21), average="macro", zero_division=0
            )),
            f"{prefix}C_U_proxy_not_official": float(100 * c**0.40 * u**0.25) if c > 0 and u > 0 else 0.0,
        })
    output["direct_unknown_all_misaccepted"] = int(np.sum(direct_unknown != 0))
    return output


def run_nested(config: dict, device: torch.device, feature_report: dict) -> dict:
    started = time.time()
    metadata = train_metadata(config)
    subjects = metadata["subject"].to_numpy(dtype=int)
    trial = metadata["trial"].to_numpy()
    maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    forward = trial > 0.8 * maximum
    inner_block = (trial > 0.6 * maximum) & (trial <= 0.8 * maximum)
    raw = torch.as_tensor(np.load(RAW_PATH), dtype=torch.float32, device=device)
    zband = torch.as_tensor(np.load(ZBAND_PATH), dtype=torch.float32, device=device)
    all_subjects = np.unique(subjects)
    outer_groups = np.array_split(
        np.random.default_rng(config["nested_protocol"]["outer_seed"]).permutation(all_subjects),
        config["nested_protocol"]["outer_folds"],
    )
    folds = []
    records_by_candidate = {name: [] for name in CANDIDATES}
    for outer_index, unknown_subjects in enumerate(outer_groups):
        outer_fold = outer_index + 1
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        outer_known = np.isin(subjects, known_subjects)
        inner_runs = []
        inner_order = np.random.default_rng(100 + outer_fold).permutation(known_subjects)
        for inner_unknown in np.array_split(inner_order, config["nested_protocol"]["inner_folds"]):
            inner_enrolled = np.setdiff1d(known_subjects, inner_unknown)
            enrolled_mask = np.isin(subjects, inner_enrolled)
            fit_np = np.flatnonzero(enrolled_mask & ~forward & ~inner_block)
            known_np = np.flatnonzero(enrolled_mask & inner_block)
            unknown_np = np.flatnonzero(np.isin(subjects, inner_unknown) & ~forward)
            fit_indices = torch.as_tensor(fit_np, device=device)
            models = fit_models(
                raw[fit_indices], zband[fit_indices], subjects[fit_np],
                inner_enrolled, config,
            )
            eval_np = np.concatenate([known_np, unknown_np])
            scores = predict_scores(
                models, raw, zband, torch.as_tensor(eval_np, device=device), config
            )
            inner_runs.append({
                "known_scores": {name: value[: len(known_np)] for name, value in scores.items()},
                "unknown_scores": {name: value[len(known_np) :] for name, value in scores.items()},
                "known_true": subjects[known_np],
                "classes": inner_enrolled,
            })
            print(
                f"outer={outer_fold} inner complete: fit={len(fit_np)} "
                f"known={len(known_np)} unknown={len(unknown_np)}", flush=True,
            )
            del models, scores
            torch.cuda.empty_cache()
        selections = {
            name: select_operating_point(inner_runs, name, config) for name in CANDIDATES
        }
        outer_fit_np = np.flatnonzero(outer_known & ~forward)
        outer_known_np = np.flatnonzero(outer_known & forward)
        outer_unknown_np = np.flatnonzero(np.isin(subjects, unknown_subjects))
        fit_indices = torch.as_tensor(outer_fit_np, device=device)
        models = fit_models(
            raw[fit_indices], zband[fit_indices], subjects[outer_fit_np],
            known_subjects, config,
        )
        eval_np = np.concatenate([outer_known_np, outer_unknown_np])
        scores = predict_scores(
            models, raw, zband, torch.as_tensor(eval_np, device=device), config
        )
        fit_scores = predict_scores(models, raw, zband, fit_indices, config)
        fold_report = {
            "outer_fold": outer_fold,
            "unknown_subjects": unknown_subjects.tolist(),
            "registered_subjects": known_subjects.tolist(),
            "n_outer_fit": len(outer_fit_np),
            "candidates": {},
        }
        for name in CANDIDATES:
            fit_pred = known_subjects[fit_scores[name].argmax(axis=1)]
            report, records = evaluate_outer(
                subjects[outer_known_np], scores[name][: len(outer_known_np)],
                scores[name][len(outer_known_np) :], known_subjects,
                name, selections[name],
            )
            report["train_fit_accuracy"] = float(np.mean(fit_pred == subjects[outer_fit_np]))
            report["train_fit_identity_macro_f1"] = float(f1_score(
                subjects[outer_fit_np], fit_pred, labels=known_subjects,
                average="macro", zero_division=0,
            ))
            fold_report["candidates"][name] = {
                "selection": selections[name], "metrics": report,
            }
            records_by_candidate[name].append(records)
            print(
                f"outer={outer_fold} {name}: directC={report['direct_C']:.4f} "
                f"C={report['C']:.4f} U={report['U']:.4f} "
                f"AUROC={report['unknown_auroc']:.4f} F1={report['identity_macro_f1']:.4f}",
                flush=True,
            )
        folds.append(fold_report)
        del models, scores, fit_scores
        torch.cuda.empty_cache()
        write_json(RESULTS_PATH, {
            "status": "partial", "completed_outer_folds": len(folds), "folds": folds
        })
    summaries = {}
    for name in CANDIDATES:
        candidate_folds = [fold["candidates"][name]["metrics"] for fold in folds]
        summaries[name] = {
            "fold_mean": {
                metric: float(np.mean([row[metric] for row in candidate_folds]))
                for metric in FOLD_METRICS
            },
            "fold_std": {
                metric: float(np.std([row[metric] for row in candidate_folds]))
                for metric in FOLD_METRICS
            },
            "pooled": pooled_metrics(records_by_candidate[name]),
        }
    full_indices = torch.arange(len(subjects), device=device)
    full_models = fit_models(raw, zband, subjects, all_subjects, config)
    full_scores = predict_scores(full_models, raw, zband, full_indices, config)
    train_fit = {}
    for name in CANDIDATES:
        prediction = all_subjects[full_scores[name].argmax(axis=1)]
        train_fit[name] = {
            "accuracy": float(np.mean(prediction == subjects)),
            "identity_20class_macro_f1": float(f1_score(
                subjects, prediction, labels=all_subjects, average="macro", zero_division=0
            )),
        }
    invariance = model_invariance(full_models, raw, zband, config)
    result = {
        "status": "complete",
        "experiment": "E36A/E36B",
        "date": "2026-11-12",
        "data_evidence": "Train session1 only; pseudo-unknown and continuous file-block proxy; not cross-session evidence; Test not accessed",
        "config_sha256": sha256(CONFIG_PATH),
        "script_sha256": sha256(Path(__file__).resolve()),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "feature_report": feature_report,
        "protocol": config["nested_protocol"],
        "folds": folds,
        "summaries": summaries,
        "full_train_fit": train_fit,
        "model_invariance": invariance,
        "adversarial_metrics": "N/A: no adversarial attack suite was run in E36A/B",
        "test_metrics": "N/A: no true labels; Test was not read or used",
        "elapsed_seconds_nested_and_checks": time.time() - started,
    }
    write_json(RESULTS_PATH, result)
    return result


def model_invariance(
    models: dict, raw: torch.Tensor, zband: torch.Tensor, config: dict,
) -> dict:
    rng = np.random.default_rng(config["invariance"]["seed"])
    selected_np = rng.choice(len(raw), size=config["invariance"]["trials"], replace=False)
    selected = torch.as_tensor(selected_np, device=raw.device)
    batch = predict_scores(models, raw, zband, selected, config)
    permutation = rng.permutation(len(selected_np))
    permuted = predict_scores(models, raw, zband, selected[permutation], config)
    inverse = np.argsort(permutation)
    output = {
        "n_trials": len(selected_np),
        "score_atol": config["invariance"]["score_atol"],
        "score_rtol": config["invariance"]["score_rtol"],
        "candidates": {},
    }
    for name in CANDIDATES:
        singles = np.concatenate([
            predict_scores(models, raw, zband, selected[index : index + 1], config)[name]
            for index in range(len(selected_np))
        ])
        single_difference = float(np.max(np.abs(batch[name] - singles)))
        permutation_difference = float(np.max(np.abs(batch[name] - permuted[name][inverse])))
        output["candidates"][name] = {
            "single_vs_batch_max_abs_score": single_difference,
            "permutation_vs_batch_max_abs_score": permutation_difference,
            "single_vs_batch_labels_equal": bool(np.array_equal(
                batch[name].argmax(axis=1), singles.argmax(axis=1)
            )),
            "permutation_vs_batch_labels_equal": bool(np.array_equal(
                batch[name].argmax(axis=1), permuted[name][inverse].argmax(axis=1)
            )),
            "single_vs_batch_scores_allclose": bool(np.allclose(
                batch[name], singles,
                atol=config["invariance"]["score_atol"],
                rtol=config["invariance"]["score_rtol"],
            )),
            "permutation_vs_batch_scores_allclose": bool(np.allclose(
                batch[name], permuted[name][inverse],
                atol=config["invariance"]["score_atol"],
                rtol=config["invariance"]["score_rtol"],
            )),
        }
    return output


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_report(result: dict) -> None:
    lines = [
        "# E36A/B 纯 EEG 五频带基线与嵌套开放集结果",
        "",
        "日期：2026-11-12。仅使用 Train Session1；未读取、推理或调参使用 Test。",
        "",
        "## 冻结方法与复现",
        "",
        "- 五个频带：1–8、8–13、13–20、20–30、30–40 Hz；逐 Trial FFT 隔离频带，GPU 协方差、0.05 固定收缩、logSPD、上三角向量化（非对角乘 sqrt(2)）。",
        "- 四候选：raw Ridge、detrend+zband Ridge、detrend+zband 原型、detrend+zband WCCN 原型。Ridge alpha=100；WCCN 仅用训练折。",
        "- 外层 RNG42 五折留被试，每折 4 人 unknown；内层 RNG=100+外层折号（折号从 1），16 人两折。known 折外评估为 subject/task 内后 20% 连续文件块代理。",
        "- 内层对每种分数搜索 101 个阈值，约束 known acceptance≥0.95，最大化 `100*C^0.40*U^0.25`。这是缺少 A 项的选择代理，不是官方分数；同时冻结 q05 工作点。",
        "- AUC 指 unknown-detection AUROC，较大 confidence 表示 known；unknown AUPR 使用 `-confidence`。身份 F1 与 unknown 类 F1 分开报告。",
        "",
        "实际命令：",
        "",
        "```powershell",
        "& \"d:\\WORK\\BCIC2026\\BrainPrint\\.venv\\Scripts\\python.exe\" \"d:\\WORK\\BCIC2026\\BrainPrint\\research\\E36_pure_eeg\\run_e36.py\" --stage features --rebuild-features",
        "& \"d:\\WORK\\BCIC2026\\BrainPrint\\.venv\\Scripts\\python.exe\" \"d:\\WORK\\BCIC2026\\BrainPrint\\research\\E36_pure_eeg\\run_e36.py\" --stage nested",
        "```",
        "",
        f"配置：`{CONFIG_PATH}`；完整逐折数值：`{RESULTS_PATH}`；特征证据：`{FEATURE_REPORT_PATH}`。",
        "",
        "## 特征与逐样本不变性",
        "",
    ]
    feature = result["feature_report"]
    lines.extend([
        f"- 特征形状：{feature['shape']}，raw/zband 均 finite={feature['raw_finite'] and feature['zband_finite']}。",
        f"- 缓存重建最大绝对差：{feature['invariance']['cache_vs_rebuilt_max_abs']:.3g}；单条/批次：{feature['invariance']['single_vs_batch_max_abs']:.3g}；置换/批次：{feature['invariance']['permutation_vs_batch_max_abs']:.3g}。",
        "",
        "| 候选 | 单条/批次最大分数差 | 置换/批次最大分数差 | 标签/分数容差检查 |",
        "|---|---:|---:|---|",
    ])
    for name in CANDIDATES:
        check = result["model_invariance"]["candidates"][name]
        passed = all(check[key] for key in (
            "single_vs_batch_labels_equal", "permutation_vs_batch_labels_equal",
            "single_vs_batch_scores_allclose", "permutation_vs_batch_scores_allclose",
        ))
        lines.append(
            f"| {name} | {check['single_vs_batch_max_abs_score']:.3g} | "
            f"{check['permutation_vs_batch_max_abs_score']:.3g} | {'通过' if passed else '未通过'} |"
        )
    lines.extend([
        "",
        "## 外层五折均值与 pooled 指标",
        "",
        "direct C 是不拒识的连续块身份准确率；C/U/unknown F1 是内层所选工作点；U@inner95 使用内层已知置信度 q05。joint 21 类 macro-F1 在跨折 pooled 的固定标签 0–20 上计算。",
        "",
        "| 候选 | Train fit | direct C | C | 身份F1 direct | 身份F1+拒识 | AUROC | unknown AUPR | U@inner95 | 工作点U | unknown F1 | joint 21类F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in CANDIDATES:
        mean = result["summaries"][name]["fold_mean"]
        pooled = result["summaries"][name]["pooled"]
        lines.append(
            f"| {name} | {fmt(result['full_train_fit'][name]['accuracy'])} | "
            f"{fmt(mean['direct_C'])} | {fmt(mean['C'])} | {fmt(mean['identity_macro_f1_direct'])} | "
            f"{fmt(mean['identity_macro_f1'])} | {fmt(mean['unknown_auroc'])} | "
            f"{fmt(mean['unknown_aupr'])} | {fmt(mean['q05_U'])} | {fmt(mean['U'])} | "
            f"{fmt(mean['unknown_f1'])} | {fmt(pooled['joint_21class_macro_f1'])} |"
        )
    lines.extend([
        "",
        "Pooled AUROC/AUPR 与逐折均值可能不同；完整 JSON 同时保存两者、fold std、EER、known AUPR、known acceptance 和逐折阈值。",
        "",
        "## 每折工作点结果",
        "",
        "| 候选 | 折 | unknown subjects | 分数/阈值来源 | direct C | C | U | AUROC | unknown F1 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ])
    for fold in result["folds"]:
        for name in CANDIDATES:
            item = fold["candidates"][name]
            metrics = item["metrics"]
            selection = item["selection"]
            lines.append(
                f"| {name} | {fold['outer_fold']} | {','.join(map(str, fold['unknown_subjects']))} | "
                f"{selection['score_name']} / inner | {fmt(metrics['direct_C'])} | {fmt(metrics['C'])} | "
                f"{fmt(metrics['U'])} | {fmt(metrics['unknown_auroc'])} | {fmt(metrics['unknown_f1'])} |"
            )
    lines.extend([
        "",
        "## 量化门禁（描述性比较，不据外层重新选模）",
        "",
        "| 候选 | Train fit=100% | direct C≥.95 | 身份F1 direct≥.92 | 连续块≥.90 | AUROC≥.98 | U@inner95≥.95 |",
        "|---|---|---|---|---|---|---|",
    ])
    for name in CANDIDATES:
        mean = result["summaries"][name]["fold_mean"]
        gates = [
            abs(result["full_train_fit"][name]["accuracy"] - 1.0) < 1e-12,
            mean["direct_C"] >= 0.95,
            mean["identity_macro_f1_direct"] >= 0.92,
            mean["continuous_block_accuracy"] >= 0.90,
            mean["unknown_auroc"] >= 0.98,
            mean["q05_U"] >= 0.95,
        ]
        lines.append(f"| {name} | " + " | ".join("通过" if gate else "未通过" for gate in gates) + " |")
    lines.extend([
        "",
        "## 证据边界与 N/A",
        "",
        "- 所有 unknown 都是 Train 留被试 pseudo-unknown，不等于官方 5 名新被试；连续文件块顺序只是代理，不是真实跨天或跨 Session 证据。",
        "- 内层与外层注册类别数不同，置信度尺度迁移仍是限制；四候选的外层结果只作冻结协议报告，不用于回头调参。",
        "- 对抗 RA、ASafeR、A 及对抗 F1：N/A（E36A/B 未运行攻击套件）。因此不能计算完整官方 Score。",
        "- Test AUC、F1、C、RA、ASafeR、U：N/A（无真实标签）；Test 未读取，未生成候选提交。",
        "- Train fit 100% 若出现，仅是容量/流程检查，不是泛化证据。",
        "",
    ])
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("features", "nested", "all"), default="all")
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("E36 要求 CUDA GPU")
    torch.manual_seed(20261112)
    np.random.seed(20261112)
    device = torch.device("cuda")
    config = load_config()
    if args.stage in ("features", "all"):
        feature_report = build_features(config, device, args.rebuild_features)
    else:
        if not (RAW_PATH.exists() and ZBAND_PATH.exists() and FEATURE_REPORT_PATH.exists()):
            raise FileNotFoundError("先运行 --stage features")
        feature_report = json.loads(FEATURE_REPORT_PATH.read_text(encoding="utf-8"))
        if feature_report["config_sha256"] != sha256(CONFIG_PATH):
            raise ValueError("配置已变化，须显式 --rebuild-features")
    if args.stage in ("nested", "all"):
        result = run_nested(config, device, feature_report)
        write_report(result)
        print(f"完成：{RESULTS_PATH}", flush=True)
        print(f"报告：{OUT / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
