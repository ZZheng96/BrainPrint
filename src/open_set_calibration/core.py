"""纯 EEG 开集识别的无监督目标域特征对齐。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

from src.open_set_improvement.core import TaskSpecificModel, row_zscore

CALIBRATION_VARIANTS = ("full", "base")
ALIGNMENTS = ("none", "target_zscore")
SCORE_NORMALIZATIONS = ("none", "row_zscore")


def config_name(variant: str, alignment: str, score_normalization: str = "none") -> str:
    suffix = "" if score_normalization == "none" else f"_{score_normalization}"
    return f"{variant}_{alignment}{suffix}"


CONFIGS = tuple(
    config_name(variant, alignment)
    for variant in CALIBRATION_VARIANTS
    for alignment in ALIGNMENTS
)
ALL_CONFIGS = CONFIGS + tuple(
    config_name(variant, alignment, "row_zscore")
    for variant in CALIBRATION_VARIANTS
    for alignment in ALIGNMENTS
)


def _target_zscore(
    target: torch.Tensor,
    source_mean: torch.Tensor,
    source_scale: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """仅用目标特征估计统计量，并映射到源模型训练特征尺度。"""
    target_mean = target.mean(dim=0)
    target_scale = target.std(dim=0, unbiased=False).clamp_min(1e-6)
    raw_ratio = source_scale / target_scale
    ratio = raw_ratio.clamp(0.5, 2.0)
    aligned = (target - target_mean) * ratio + source_mean
    clip_fraction = float((raw_ratio != ratio).float().mean().item())
    return aligned, clip_fraction


def align_features_by_task(
    model: TaskSpecificModel,
    x: np.ndarray,
    window: np.ndarray,
    tasks: np.ndarray,
    alignment: str,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """按任务对齐完整目标批次；不接收身份、Session 或已知/未知标签。"""
    if alignment not in ALIGNMENTS:
        raise ValueError(alignment)
    x_tensor = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=model.device)
    window_tensor = torch.as_tensor(
        np.asarray(window), dtype=torch.float32, device=model.device
    )
    if alignment == "none":
        return x_tensor, window_tensor, {}

    aligned_x = torch.empty_like(x_tensor)
    aligned_window = torch.empty_like(window_tensor)
    diagnostics = {}
    for task in sorted(np.unique(tasks)):
        indices = np.flatnonzero(tasks == task)
        if not len(indices):
            continue
        task_models = model.models[task]
        task_x = x_tensor[indices]
        task_window = window_tensor[indices]
        aligned_x[indices], x_clip = _target_zscore(
            task_x, task_models.full.mean, task_models.full.scale
        )
        flat_window = task_window.reshape(-1, task_window.shape[-1])
        aligned_flat, window_clip = _target_zscore(
            flat_window, task_models.window.mean, task_models.window.scale
        )
        aligned_window[indices] = aligned_flat.reshape(task_window.shape)
        diagnostics[str(task)] = {
            "n_target_trials": int(len(indices)),
            "full_scale_ratio_clip_fraction": x_clip,
            "window_scale_ratio_clip_fraction": window_clip,
        }
    return aligned_x, aligned_window, diagnostics


def predict_calibration_scores(
    model: TaskSpecificModel,
    x: np.ndarray,
    window: np.ndarray,
    tasks: np.ndarray,
    alignment: str,
    batch_size: int = 256,
    score_normalization: str = "none",
) -> tuple[dict[str, np.ndarray], dict]:
    """只计算 full/base；对齐统计在推理前由整个无标签目标批次估计。"""
    if score_normalization not in SCORE_NORMALIZATIONS:
        raise ValueError(score_normalization)
    aligned_x, aligned_window, diagnostics = align_features_by_task(
        model, x, window, tasks, alignment
    )
    outputs = {
        variant: np.full((len(x), len(model.classes)), np.nan, dtype=np.float32)
        for variant in CALIBRATION_VARIANTS
    }
    for task in sorted(np.unique(tasks)):
        indices = np.flatnonzero(tasks == task)
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            with torch.no_grad():
                for variant in CALIBRATION_VARIANTS:
                    scores = model.scores(
                        aligned_x[selected], aligned_window[selected], task, variant
                    )
                    if score_normalization == "row_zscore":
                        scores = row_zscore(scores)
                    outputs[variant][selected] = scores.cpu().numpy()
    if any(np.isnan(values).any() for values in outputs.values()):
        raise RuntimeError("部分 Trial 没有有效分数")
    scores_by_config = {
        config_name(variant, alignment, score_normalization): values
        for variant, values in outputs.items()
    }
    return scores_by_config, diagnostics


def aggregate_reports(records: dict[str, np.ndarray], fold_reports: list[dict]) -> dict:
    """C/U/F1 逐 Trial 汇总；AUROC/AUPR/EER 使用外层折宏平均。"""
    known = records["is_known"]
    true = records["true"]
    pred = records["pred"]
    confidence = records["confidence"]
    fpr, tpr, _ = roc_curve(known, confidence)
    fnr = 1 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    fold_metric = lambda name: float(np.mean([row[name] for row in fold_reports]))
    return {
        "n_known": int(known.sum()),
        "n_unknown": int((~known).sum()),
        "C": float(np.mean(pred[known] == true[known])),
        "direct_C_without_rejection": float(np.average(
            [row["direct_C_without_rejection"] for row in fold_reports],
            weights=[row["n_known"] for row in fold_reports],
        )),
        "U": float(np.mean(pred[~known] == 0)),
        "known_acceptance": float(np.average(
            [row["known_acceptance"] for row in fold_reports],
            weights=[row["n_known"] for row in fold_reports],
        )),
        "unknown_auroc": fold_metric("unknown_auroc"),
        "known_aupr": fold_metric("known_aupr"),
        "unknown_aupr": fold_metric("unknown_aupr"),
        "eer": fold_metric("eer"),
        "unknown_auroc_fold_min": float(min(row["unknown_auroc"] for row in fold_reports)),
        "unknown_auroc_fold_max": float(max(row["unknown_auroc"] for row in fold_reports)),
        "pooled_unknown_auroc_diagnostic": float(roc_auc_score(known, confidence)),
        "pooled_known_aupr_diagnostic": float(average_precision_score(known, confidence)),
        "pooled_eer_diagnostic": float((fpr[index] + fnr[index]) / 2),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "unknown_f1": float(f1_score(
            known.astype(int), pred != 0, pos_label=0, zero_division=0
        )),
        "detection_macro_f1": float(f1_score(
            known.astype(int), pred != 0, average="macro", zero_division=0
        )),
        "score_selection": [row["score_name"] for row in fold_reports],
        "thresholds": [row["threshold"] for row in fold_reports],
    }


def build_submission(
    template: pd.DataFrame,
    test_metadata: pd.DataFrame,
    predicted: np.ndarray,
) -> pd.DataFrame:
    """按 Sample 映射预测，保持 test.csv 原始行顺序。"""
    if len(template) != len(test_metadata) or len(predicted) != len(test_metadata):
        raise ValueError("提交模板、测试元数据和预测数量必须一致")
    if test_metadata["sample"].duplicated().any() or template["Sample"].duplicated().any():
        raise ValueError("Sample 必须唯一")
    sample_to_row = dict(zip(test_metadata["sample"], range(len(test_metadata))))
    rows = template["Sample"].map(sample_to_row)
    if rows.isna().any():
        raise ValueError("提交模板包含测试元数据中不存在的 Sample")
    predicted = np.asarray(predicted, dtype=int)
    if np.any((predicted < 0) | (predicted > 20)):
        raise ValueError("预测标签必须位于 0–20")
    submission = template.copy()
    submission["Label"] = predicted[rows.to_numpy()]
    return submission
