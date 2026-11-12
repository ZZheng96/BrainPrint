"""E36E：Train-only WCCN distance_ratio ECDF 校准与 K 迁移检查。"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CONFIG_PATH = OUT / "config.json"
ZBAND_PATH = ROOT / "BrainPrint_cache" / "e36_pure_eeg" / "train_detrend_zband_cov5.npy"
RESULTS_PATH = OUT / "calibration_results.json"
SEED = 20261112


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_train_metadata(config: dict) -> pd.DataFrame:
    metadata = pd.read_csv(ROOT / config["data"]["train_metadata"])
    expected = config["data"]
    if len(metadata) != expected["expected_trials"]:
        raise ValueError("Train trial 数不符合冻结配置")
    if metadata["subject"].nunique() != expected["expected_subjects"]:
        raise ValueError("Train 被试数不符合冻结配置")
    if set(metadata["session"]) != {expected["session"]}:
        raise ValueError("只允许 Train session1")
    if not metadata["path"].str.startswith("Train/").all():
        raise ValueError("元数据含非 Train 路径")
    return metadata


class TorchWCCNPrototype:
    def __init__(
        self, x: torch.Tensor, y: np.ndarray, classes: np.ndarray,
        shrinkage: float, eigenvalue_floor: float,
    ):
        self.classes = np.asarray(classes, dtype=int)
        class_to_index = {label: index for index, label in enumerate(self.classes)}
        target = torch.as_tensor(
            [class_to_index[int(label)] for label in y], dtype=torch.long, device=x.device
        )
        means = torch.stack([x[target == index].mean(dim=0) for index in range(len(classes))])
        residual = x - means[target]
        covariance = residual.T @ residual / max(len(x) - len(classes), 1)
        trace = torch.diagonal(covariance).sum()
        eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
        covariance = (1.0 - shrinkage) * covariance + shrinkage * trace / x.shape[1] * eye
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        inverse_root = eigenvalues.clamp_min(eigenvalue_floor).rsqrt()
        self.whitening = (eigenvectors * inverse_root.unsqueeze(0)) @ eigenvectors.T
        means = means @ self.whitening
        self.prototypes = means / means.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.whitening
        normalized = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return normalized @ self.prototypes.T


def predict_scores(
    model: TorchWCCNPrototype, x: torch.Tensor, indices: np.ndarray | torch.Tensor,
    device: torch.device, batch_size: int,
) -> np.ndarray:
    if isinstance(indices, np.ndarray):
        indices = torch.as_tensor(indices, device=device)
    output = np.empty((len(indices), len(model.classes)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            output[start : start + len(selected)] = model(x[selected]).cpu().numpy()
    return output


def distance_ratio(scores: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(scores, dtype=np.float64), axis=1)
    maximum = ordered[:, -1]
    second = ordered[:, -2]
    return (1.0 - second) / np.maximum(1.0 - maximum, 1e-6)


def ecdf_map(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    sorted_reference = np.sort(np.asarray(reference, dtype=np.float64))
    return np.searchsorted(
        sorted_reference, np.asarray(values, dtype=np.float64), side="right"
    ) / len(sorted_reference)


def detection_summary(known: np.ndarray, unknown: np.ndarray) -> dict:
    labels = np.concatenate([np.ones(len(known), dtype=int), np.zeros(len(unknown), dtype=int)])
    confidence = np.concatenate([known, unknown])
    fpr, tpr, _ = roc_curve(labels, confidence)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return {
        "AUROC": float(roc_auc_score(labels, confidence)),
        "unknown_AUPR": float(average_precision_score(1 - labels, -confidence)),
        "known_AUPR": float(average_precision_score(labels, confidence)),
        "EER": float((fpr[index] + fnr[index]) / 2),
    }


def proxy_value(c: float, u: float) -> float:
    return float(100 * c**0.40 * u**0.25) if c > 0 and u > 0 else 0.0


def concatenate_records(records: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([record[f"known_{key}"] for record in records]),
        np.concatenate([record[f"unknown_{key}"] for record in records]),
    )


def metrics_at_threshold(
    records: list[dict], key: str, threshold: float, fixed_20_labels: bool,
) -> dict:
    known_true = np.concatenate([record["known_true"] for record in records])
    known_direct = np.concatenate([record["known_direct"] for record in records])
    unknown_direct = np.concatenate([record["unknown_direct"] for record in records])
    known, unknown = concatenate_records(records, key)
    known_accepted = known >= threshold
    unknown_accepted = unknown >= threshold
    known_pred = known_direct.copy()
    unknown_pred = unknown_direct.copy()
    known_pred[~known_accepted] = 0
    unknown_pred[~unknown_accepted] = 0
    detection_true = np.concatenate([
        np.ones(len(known_true), dtype=int), np.zeros(len(unknown_direct), dtype=int)
    ])
    accepted = np.concatenate([known_accepted, unknown_accepted])
    identity_labels = (
        np.arange(1, 21) if fixed_20_labels
        else np.unique(np.concatenate([record["classes"] for record in records]))
    )
    joint_labels = (
        np.arange(21) if fixed_20_labels
        else np.concatenate([[0], identity_labels])
    )
    c = float(np.mean(known_pred == known_true))
    u = float(np.mean(unknown_pred == 0))
    return {
        "threshold": float(threshold),
        "C": c,
        "U": u,
        "known_acceptance": float(np.mean(known_accepted)),
        "unknown_F1": float(f1_score(
            detection_true, accepted, pos_label=0, zero_division=0
        )),
        "identity_macro_F1": float(f1_score(
            known_true, known_pred, labels=identity_labels, average="macro", zero_division=0
        )),
        "joint_macro_F1": float(f1_score(
            np.concatenate([known_true, np.zeros(len(unknown_pred), dtype=int)]),
            np.concatenate([known_pred, unknown_pred]),
            labels=joint_labels, average="macro", zero_division=0,
        )),
        "C_U_proxy_not_official": proxy_value(c, u),
    }


def q05_threshold(known: np.ndarray) -> float:
    return float(np.quantile(known, 0.05, method="lower"))


def select_inner_threshold(records: list[dict], key: str, grid_points: int) -> dict:
    known, unknown = concatenate_records(records, key)
    known_true = np.concatenate([record["known_true"] for record in records])
    known_direct = np.concatenate([record["known_direct"] for record in records])
    known_correct = known_direct == known_true
    detection = detection_summary(known, unknown)
    candidates = []
    for threshold in np.quantile(known, np.linspace(0.0, 0.05, grid_points)):
        known_accepted = known >= threshold
        c = float(np.mean(known_correct & known_accepted))
        u = float(np.mean(unknown < threshold))
        acceptance = float(np.mean(known_accepted))
        if acceptance >= 0.95 - 1e-12:
            candidates.append({
                "threshold": float(threshold), "C": c, "U": u,
                "known_acceptance": acceptance,
                "C_U_proxy_not_official": proxy_value(c, u),
                "AUROC": detection["AUROC"],
            })
    best = max(candidates, key=lambda row: (
        row["C_U_proxy_not_official"], row["C"], row["U"], row["AUROC"],
        -row["threshold"],
    ))
    return {
        "score": key,
        "selection_pool": "inner_K8_pooled_only",
        "known_acceptance_constraint": 0.95,
        "proxy_threshold": best["threshold"],
        "q05_threshold": q05_threshold(known),
        "selection_detection": detection,
        "proxy_selection_preview": best,
        "warning": "仅C/U代理，无攻击A项，不是完整官方Score；未使用outer或Test选择",
    }


def summarize_records(
    records: list[dict], selections: dict, fixed_20_labels: bool,
) -> dict:
    known_true = np.concatenate([record["known_true"] for record in records])
    known_direct = np.concatenate([record["known_direct"] for record in records])
    identity_labels = (
        np.arange(1, 21) if fixed_20_labels
        else np.unique(np.concatenate([record["classes"] for record in records]))
    )
    output = {
        "n_known": int(len(known_true)),
        "n_unknown": int(sum(len(record["unknown_raw"]) for record in records)),
        "direct_C": float(np.mean(known_direct == known_true)),
        "identity_macro_F1_direct": float(f1_score(
            known_true, known_direct, labels=identity_labels, average="macro", zero_division=0
        )),
    }
    for key in ("raw", "u"):
        known, unknown = concatenate_records(records, key)
        local_q05 = q05_threshold(known)
        output[key] = {
            **detection_summary(known, unknown),
            "proxy_work_inner_threshold": metrics_at_threshold(
                records, key, selections[key]["proxy_threshold"], fixed_20_labels
            ),
            "q05_work_inner_threshold": metrics_at_threshold(
                records, key, selections[key]["q05_threshold"], fixed_20_labels
            ),
            "local_q05_diagnostic_not_for_selection": metrics_at_threshold(
                records, key, local_q05, fixed_20_labels
            ),
            "known_quantiles": dict(zip(
                ["q00", "q05", "q50", "q95", "q100"],
                map(float, np.quantile(known, [0, 0.05, 0.5, 0.95, 1])),
            )),
            "unknown_quantiles": dict(zip(
                ["q00", "q05", "q50", "q95", "q100"],
                map(float, np.quantile(unknown, [0, 0.05, 0.5, 0.95, 1])),
            )),
        }
    return output


def make_fold_record(
    fold_name: str, unknown_subjects: np.ndarray, enrolled_subjects: np.ndarray,
    fit_np: np.ndarray, known_np: np.ndarray, unknown_np: np.ndarray,
    subjects: np.ndarray, zband: torch.Tensor, config: dict, device: torch.device,
    shrinkage: float,
) -> dict:
    fit_indices = torch.as_tensor(fit_np, device=device)
    model = TorchWCCNPrototype(
        zband[fit_indices], subjects[fit_np], enrolled_subjects, shrinkage,
        float(config["models"]["wccn_eigenvalue_floor"]),
    )
    batch_size = int(config["models"]["prediction_batch_size"])
    known_scores = predict_scores(model, zband, known_np, device, batch_size)
    unknown_scores = predict_scores(model, zband, unknown_np, device, batch_size)
    known_raw = distance_ratio(known_scores)
    unknown_raw = distance_ratio(unknown_scores)
    reference = np.sort(known_raw)
    record = {
        "fold_name": fold_name,
        "K": int(len(enrolled_subjects)),
        "classes": enrolled_subjects,
        "unknown_subjects": unknown_subjects.tolist(),
        "registered_subjects": enrolled_subjects.tolist(),
        "n_fit": int(len(fit_np)),
        "known_true": subjects[known_np],
        "known_direct": enrolled_subjects[known_scores.argmax(axis=1)],
        "unknown_direct": enrolled_subjects[unknown_scores.argmax(axis=1)],
        "known_raw": known_raw,
        "unknown_raw": unknown_raw,
        "known_u": ecdf_map(reference, known_raw),
        "unknown_u": ecdf_map(reference, unknown_raw),
    }
    del model, known_scores, unknown_scores, fit_indices
    torch.cuda.empty_cache()
    return record


def known_only_metrics(
    true: np.ndarray, direct: np.ndarray, confidence: np.ndarray, threshold: float,
) -> dict:
    accepted = confidence >= threshold
    pred = direct.copy()
    pred[~accepted] = 0
    return {
        "threshold": float(threshold),
        "C": float(np.mean(pred == true)),
        "known_acceptance": float(np.mean(accepted)),
        "identity_20class_macro_F1": float(f1_score(
            true, pred, labels=np.arange(1, 21), average="macro", zero_division=0
        )),
    }


def transfer_delta(inner: dict, outer: dict, key: str, work: str) -> dict:
    source = inner[key][work]
    target = outer[key][work]
    return {
        metric: float(target[metric] - source[metric])
        for metric in (
            "C", "U", "known_acceptance", "unknown_F1", "identity_macro_F1",
            "joint_macro_F1", "C_U_proxy_not_official",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wccn-shrinkage", type=float, default=0.05)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("E36E 要求 CUDA GPU")
    if not 0.0 <= args.wccn_shrinkage <= 1.0:
        parser.error("--wccn-shrinkage 必须在 [0,1]")
    if not ZBAND_PATH.exists():
        raise FileNotFoundError(f"缺少已迁移的 Train 特征缓存：{ZBAND_PATH}")
    started = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    metadata = load_train_metadata(config)
    cached = np.load(ZBAND_PATH, mmap_mode="r")
    if cached.shape != (len(metadata), 855) or not np.isfinite(cached).all():
        raise ValueError("Train zband 缓存形状或数值无效")
    zband = torch.as_tensor(np.asarray(cached), dtype=torch.float32, device=device)
    subjects = metadata["subject"].to_numpy(dtype=int)
    trial = metadata["trial"].to_numpy()
    maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    forward = trial > 0.8 * maximum
    inner_block = (trial > 0.6 * maximum) & (trial <= 0.8 * maximum)
    all_subjects = np.unique(subjects)
    outer_groups = np.array_split(
        np.random.default_rng(config["nested_protocol"]["outer_seed"]).permutation(all_subjects),
        config["nested_protocol"]["outer_folds"],
    )
    inner_records = []
    outer_records = []
    for outer_index, unknown_subjects in enumerate(outer_groups):
        outer_fold = outer_index + 1
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        outer_known = np.isin(subjects, known_subjects)
        inner_order = np.random.default_rng(100 + outer_fold).permutation(known_subjects)
        for inner_index, inner_unknown in enumerate(np.array_split(
            inner_order, config["nested_protocol"]["inner_folds"]
        )):
            inner_enrolled = np.setdiff1d(known_subjects, inner_unknown)
            enrolled_mask = np.isin(subjects, inner_enrolled)
            fit_np = np.flatnonzero(enrolled_mask & ~forward & ~inner_block)
            known_np = np.flatnonzero(enrolled_mask & inner_block)
            unknown_np = np.flatnonzero(np.isin(subjects, inner_unknown) & ~forward)
            record = make_fold_record(
                f"outer{outer_fold}_inner{inner_index + 1}", inner_unknown, inner_enrolled,
                fit_np, known_np, unknown_np, subjects, zband, config, device,
                args.wccn_shrinkage,
            )
            inner_records.append(record)
            print(
                f"{record['fold_name']} K=8 fit={len(fit_np)} known={len(known_np)} "
                f"unknown={len(unknown_np)}", flush=True,
            )
        fit_np = np.flatnonzero(outer_known & ~forward)
        known_np = np.flatnonzero(outer_known & forward)
        unknown_np = np.flatnonzero(np.isin(subjects, unknown_subjects))
        record = make_fold_record(
            f"outer{outer_fold}", unknown_subjects, known_subjects,
            fit_np, known_np, unknown_np, subjects, zband, config, device,
            args.wccn_shrinkage,
        )
        outer_records.append(record)
        print(
            f"outer{outer_fold} K=16 fit={len(fit_np)} known={len(known_np)} "
            f"unknown={len(unknown_np)}", flush=True,
        )

    grid_points = int(config["nested_protocol"]["threshold_grid_points"])
    selections = {
        key: select_inner_threshold(inner_records, key, grid_points) for key in ("raw", "u")
    }
    pooled = {
        "inner_K8_calibration": summarize_records(inner_records, selections, True),
        "outer_K16_transfer_confirmation": summarize_records(outer_records, selections, True),
    }
    inner_summary = pooled["inner_K8_calibration"]
    outer_summary = pooled["outer_K16_transfer_confirmation"]
    fold_metrics = {}
    for group_name, records in (("inner_K8", inner_records), ("outer_K16", outer_records)):
        rows = []
        for record in records:
            summary = summarize_records([record], selections, False)
            rows.append({
                key: record[key] for key in (
                    "fold_name", "K", "unknown_subjects", "registered_subjects", "n_fit"
                )
            } | {"metrics": summary})
        fold_metrics[group_name] = rows

    drift = {
        "comparison": "K8内层与K16外层；负/正差均为描述性迁移结果，不用于调参",
        "raw_proxy_work_K16_minus_K8": transfer_delta(
            inner_summary, outer_summary, "raw", "proxy_work_inner_threshold"
        ),
        "ecdf_proxy_work_K16_minus_K8": transfer_delta(
            inner_summary, outer_summary, "u", "proxy_work_inner_threshold"
        ),
        "raw_q05_work_K16_minus_K8": transfer_delta(
            inner_summary, outer_summary, "raw", "q05_work_inner_threshold"
        ),
        "ecdf_q05_work_K16_minus_K8": transfer_delta(
            inner_summary, outer_summary, "u", "q05_work_inner_threshold"
        ),
        "AUROC_K16_minus_K8": {
            key: float(outer_summary[key]["AUROC"] - inner_summary[key]["AUROC"])
            for key in ("raw", "u")
        },
        "local_U_at_known95_K16_minus_K8": {
            key: float(
                outer_summary[key]["local_q05_diagnostic_not_for_selection"]["U"]
                - inner_summary[key]["local_q05_diagnostic_not_for_selection"]["U"]
            ) for key in ("raw", "u")
        },
    }
    for key in ("raw", "u"):
        local_thresholds = [
            row["metrics"][key]["local_q05_diagnostic_not_for_selection"]["threshold"]
            for rows in fold_metrics.values() for row in rows
        ]
        unknown_medians = [
            float(np.median(record[f"unknown_{key}"]))
            for record in inner_records + outer_records
        ]
        drift[f"{key}_fold_q05_threshold_std"] = float(np.std(local_thresholds))
        drift[f"{key}_fold_q05_threshold_cv"] = float(
            np.std(local_thresholds) / max(abs(np.mean(local_thresholds)), 1e-12)
        )
        drift[f"{key}_fold_unknown_median_std"] = float(np.std(unknown_medians))
    drift["ecdf_reduces_fold_q05_threshold_cv_vs_raw"] = bool(
        drift["u_fold_q05_threshold_cv"] < drift["raw_fold_q05_threshold_cv"]
    )

    fit_np = np.flatnonzero(~forward)
    oof_np = np.flatnonzero(forward)
    model20 = TorchWCCNPrototype(
        zband[torch.as_tensor(fit_np, device=device)], subjects[fit_np], all_subjects,
        args.wccn_shrinkage, float(config["models"]["wccn_eigenvalue_floor"]),
    )
    scores20 = predict_scores(
        model20, zband, oof_np, device, int(config["models"]["prediction_batch_size"])
    )
    raw20 = distance_ratio(scores20)
    reference20 = np.sort(raw20)
    u20 = ecdf_map(reference20, raw20)
    direct20 = all_subjects[scores20.argmax(axis=1)]
    true20 = subjects[oof_np]
    final20 = {
        "protocol": "全部20类均enrolled；各subject/task前80%连续块fit，后20%连续块OOF；F20仅用这些已知OOF distance_ratio",
        "n_fit": int(len(fit_np)),
        "n_known_oof": int(len(oof_np)),
        "direct_C": float(np.mean(direct20 == true20)),
        "identity_20class_macro_F1_direct": float(f1_score(
            true20, direct20, labels=all_subjects, average="macro", zero_division=0
        )),
        "F20_reference_sha256": hashlib.sha256(
            np.ascontiguousarray(reference20, dtype=np.float64).tobytes()
        ).hexdigest(),
        "raw_reference_quantiles": dict(zip(
            ["q00", "q05", "q50", "q95", "q100"],
            map(float, np.quantile(raw20, [0, 0.05, 0.5, 0.95, 1])),
        )),
        "known_u_quantiles": dict(zip(
            ["q00", "q05", "q50", "q95", "q100"],
            map(float, np.quantile(u20, [0, 0.05, 0.5, 0.95, 1])),
        )),
        "inner_u_proxy_threshold_transfer": known_only_metrics(
            true20, direct20, u20, selections["u"]["proxy_threshold"]
        ),
        "inner_u_q05_threshold_transfer": known_only_metrics(
            true20, direct20, u20, selections["u"]["q05_threshold"]
        ),
        "unknown_metrics": "N/A：20类OOF没有held-out unknown；unknown评估仅见K8/K16 pseudo-unknown折",
        "test_inference": "N/A：未读取或推理Test",
    }
    del model20, scores20

    all_indices = torch.arange(len(subjects), device=device)
    full_model = TorchWCCNPrototype(
        zband, subjects, all_subjects, args.wccn_shrinkage,
        float(config["models"]["wccn_eigenvalue_floor"]),
    )
    full_scores = predict_scores(
        full_model, zband, all_indices, device, int(config["models"]["prediction_batch_size"])
    )
    full_direct = all_subjects[full_scores.argmax(axis=1)]
    capacity = {
        "purpose": "仅容量/训练流程检查，不作为泛化证据，不用于Test推理",
        "accuracy": float(np.mean(full_direct == subjects)),
        "identity_20class_macro_F1": float(f1_score(
            subjects, full_direct, labels=all_subjects, average="macro", zero_division=0
        )),
    }
    rng = np.random.default_rng(config["invariance"]["seed"])
    selected_np = rng.choice(
        len(subjects), size=int(config["invariance"]["trials"]), replace=False
    )
    batch = predict_scores(
        full_model, zband, selected_np, device, int(config["models"]["prediction_batch_size"])
    )
    singles = np.concatenate([
        predict_scores(full_model, zband, selected_np[index:index + 1], device, 1)
        for index in range(len(selected_np))
    ])
    invariance = {
        "n_trials": int(len(selected_np)),
        "single_vs_batch_max_abs_score": float(np.max(np.abs(batch - singles))),
        "single_vs_batch_labels_equal": bool(np.array_equal(batch.argmax(axis=1), singles.argmax(axis=1))),
        "batch_statistics_used": False,
    }
    del full_model, full_scores, batch, singles

    u_outer = outer_summary["u"]
    gates = {
        "scope": "C/F1/AUROC/U使用outer K16 pooled独立确认；local U@95仅已知q05诊断；容量仅全Train拟合",
        "C_direct_ge_0.95": bool(outer_summary["direct_C"] >= 0.95),
        "identity_F1_direct_ge_0.92": bool(outer_summary["identity_macro_F1_direct"] >= 0.92),
        "continuous_block_C_ge_0.90": bool(outer_summary["direct_C"] >= 0.90),
        "ECDF_AUROC_ge_0.98": bool(u_outer["AUROC"] >= 0.98),
        "ECDF_U_at_local_known95_ge_0.95": bool(
            u_outer["local_q05_diagnostic_not_for_selection"]["U"] >= 0.95
        ),
        "full_train_capacity_accuracy_eq_1": bool(abs(capacity["accuracy"] - 1.0) < 1e-12),
    }
    gates["all_C_F1_AUROC_U"] = bool(all(gates[key] for key in (
        "C_direct_ge_0.95", "identity_F1_direct_ge_0.92",
        "continuous_block_C_ge_0.90", "ECDF_AUROC_ge_0.98",
        "ECDF_U_at_local_known95_ge_0.95",
    )))
    result = {
        "status": "complete",
        "experiment": "E36E final-20-class Train-only ECDF calibration",
        "date": "2026-11-12",
        "data_evidence": "仅Train session1；pseudo-unknown与连续文件块均为代理，不是真实跨Session证据；Test未读取",
        "score_reuse": "既有results.json无完整分数缓存，按冻结划分重算K8/K16",
        "wccn_shrinkage": float(args.wccn_shrinkage),
        "shrinkage_selection": "命令行固定参数，默认0.05；未根据outer或Test选择",
        "seed": SEED,
        "ecdf_definition": "每折F_K仅fit该折known OOF distance_ratio；u=right-continuous ECDF(r)=count(reference<=r)/n；known与unknown共用该折F_K",
        "threshold_protocol": "只在inner K8 pooled OOF选择raw/u阈值；101个known quantile候选覆盖接受率[.95,1]；outer K16只迁移确认；q05使用lower order statistic保证已知接受率>=.95",
        "proxy_warning": "C/U代理=100*C^0.40*U^0.25，仅用于阈值选择，不含攻击A项，不是完整官方Score；未运行或使用对抗指标",
        "config_sha256": sha256(CONFIG_PATH),
        "script_sha256": sha256(Path(__file__).resolve()),
        "zband_cache": str(ZBAND_PATH),
        "zband_cache_sha256": sha256(ZBAND_PATH),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "model_fit_and_inference": "CUDA GPU；仅指标/ECDF在CPU；无批次统计",
        },
        "inner_threshold_selection": selections,
        "pooled": pooled,
        "K_drift": drift,
        "folds": fold_metrics,
        "final_20class_continuous_block_oof": final20,
        "full_train_20class_capacity_check": capacity,
        "single_sample_invariance": invariance,
        "gates": gates,
        "adversarial_metrics": "N/A：攻击由独立实验负责，本实验不做对抗阈值选择",
        "test_metrics": "N/A：Test无真实标签且未读取",
        "elapsed_seconds": time.time() - started,
    }
    RESULTS_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    for name, summary in ((("K8-inner"), inner_summary), (("K16-outer"), outer_summary)):
        work = summary["u"]["proxy_work_inner_threshold"]
        q05 = summary["u"]["local_q05_diagnostic_not_for_selection"]
        print(
            f"{name} ECDF: C={work['C']:.4f} U={work['U']:.4f} "
            f"U@local95={q05['U']:.4f} AUROC={summary['u']['AUROC']:.4f} "
            f"AUPR={summary['u']['unknown_AUPR']:.4f} unknownF1={work['unknown_F1']:.4f} "
            f"identityF1={work['identity_macro_F1']:.4f} jointF1={work['joint_macro_F1']:.4f} "
            f"proxy={work['C_U_proxy_not_official']:.4f}", flush=True,
        )
    print(
        f"K漂移 U@local95(K16-K8): raw={drift['local_U_at_known95_K16_minus_K8']['raw']:+.4f} "
        f"ECDF={drift['local_U_at_known95_K16_minus_K8']['u']:+.4f}; "
        f"AUROC差: raw={drift['AUROC_K16_minus_K8']['raw']:+.4f} "
        f"ECDF={drift['AUROC_K16_minus_K8']['u']:+.4f}", flush=True,
    )
    print(
        f"20类OOF directC={final20['direct_C']:.4f} "
        f"directF1={final20['identity_20class_macro_F1_direct']:.4f}; "
        f"容量accuracy={capacity['accuracy']:.6f}", flush=True,
    )
    print("门禁: " + json.dumps(gates, ensure_ascii=False), flush=True)
    print("C/U代理不是完整Score；未使用攻击、outer或Test选择阈值。", flush=True)
    print(f"文件：{RESULTS_PATH}", flush=True)
    print(
        f"命令：& \"{Path(sys.executable)}\" \"{Path(__file__).resolve()}\" "
        f"--wccn-shrinkage {args.wccn_shrinkage}", flush=True,
    )


if __name__ == "__main__":
    main()
