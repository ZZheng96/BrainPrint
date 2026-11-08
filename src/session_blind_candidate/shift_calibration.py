"""Session-blind training-domain robustification and score calibration."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.open_set_improvement.core import confidence_values


@dataclass(frozen=True)
class TrainingStatistics:
    """Robust per-feature statistics fitted only on the training partition."""

    median: np.ndarray
    mad: np.ndarray
    clip: float = 5.0

    @classmethod
    def fit(cls, x: np.ndarray, clip: float = 5.0) -> "TrainingStatistics":
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or not len(values):
            raise ValueError("x 必须是非空二维数组")
        if clip <= 0:
            raise ValueError("clip 必须为正")
        median = np.nanmedian(values, axis=0)
        mad = np.nanmedian(np.abs(values - median), axis=0) * 1.4826
        mad = np.where(np.isfinite(mad) & (mad > 1e-8), mad, 1e-6)
        return cls(median.astype(np.float32), mad.astype(np.float32), float(clip))

    def transform(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.median):
            raise ValueError("输入特征维度与训练统计量不一致")
        residual = np.nan_to_num(values - self.median, nan=0.0, posinf=0.0, neginf=0.0)
        return self.median + np.clip(residual / self.mad, -self.clip, self.clip) * self.mad


def fit_training_statistics(x: np.ndarray, window: np.ndarray | None = None) -> tuple[TrainingStatistics, TrainingStatistics | None]:
    """Fit statistics from training arrays; never estimates target-batch statistics."""
    feature_stats = TrainingStatistics.fit(x)
    window_stats = None
    if window is not None:
        flat = np.asarray(window).reshape(-1, np.asarray(window).shape[-1])
        window_stats = TrainingStatistics.fit(flat)
    return feature_stats, window_stats


def apply_robust_normalization(
    x: np.ndarray,
    window: np.ndarray,
    feature_stats: TrainingStatistics,
    window_stats: TrainingStatistics,
) -> tuple[np.ndarray, np.ndarray]:
    """Winsorize each sample using frozen training-domain robust statistics."""
    values = np.asarray(window)
    if values.ndim != 3:
        raise ValueError("window 必须是三维数组")
    normalized_window = window_stats.transform(values.reshape(-1, values.shape[-1])).reshape(values.shape)
    return feature_stats.transform(x), normalized_window


def _row_consistency(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first)
    second = np.asarray(second)
    if first.shape != second.shape:
        raise ValueError("比较的分数形状必须一致")
    return (first.argmax(axis=1) == second.argmax(axis=1)).astype(np.float32)


def score_diagnostics(
    direct_full: np.ndarray,
    direct_base: np.ndarray,
    robust_full: np.ndarray | None = None,
    robust_base: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return per-sample shift/quality signals without labels or session metadata."""
    direct_full = np.asarray(direct_full)
    direct_base = np.asarray(direct_base)
    if direct_full.ndim != 2 or direct_base.shape != direct_full.shape:
        raise ValueError("full/base 分数必须是相同形状的二维数组")
    direct_conf = confidence_values(direct_base)
    result = {
        "margin": direct_conf["margin"].astype(np.float32),
        "energy": direct_conf["energy"].astype(np.float32),
        "full_base_consistency": _row_consistency(direct_full, direct_base),
        "direct_quality": direct_conf["max_logit"].astype(np.float32),
    }
    if robust_full is not None or robust_base is not None:
        if robust_full is None or robust_base is None:
            raise ValueError("robust_full 与 robust_base 必须同时提供")
        robust_conf = confidence_values(robust_base)
        result.update({
            "robust_full_base_consistency": _row_consistency(robust_full, robust_base),
            "direct_robust_consistency": _row_consistency(direct_base, robust_base),
            "robust_quality": robust_conf["max_logit"].astype(np.float32),
            "shift": np.abs(direct_conf["max_logit"] - robust_conf["max_logit"]).astype(np.float32),
        })
    else:
        result["shift"] = np.zeros(len(direct_full), dtype=np.float32)
    result["quality"] = (
        result["full_base_consistency"]
        * (result.get("robust_full_base_consistency", result["full_base_consistency"]))
        * (1.0 / (1.0 + result["shift"]))
    ).astype(np.float32)
    return result


@dataclass(frozen=True)
class ConditionalThresholds:
    score_name: str
    edges: np.ndarray
    thresholds: np.ndarray
    default: float

    def threshold(self, quality: np.ndarray) -> np.ndarray:
        quality = np.asarray(quality, dtype=float)
        bins = np.clip(np.digitize(quality, self.edges[1:-1], right=False), 0, len(self.thresholds) - 1)
        return self.thresholds[bins]


def fit_conditional_thresholds(
    scores: np.ndarray,
    diagnostics: dict[str, np.ndarray],
    score_name: str = "max_logit",
    n_bins: int = 3,
    quantile: float = 0.05,
) -> ConditionalThresholds:
    """Fit thresholds from known OOF/training scores only, stratified by quality."""
    if not 0 < quantile < 1 or n_bins < 1:
        raise ValueError("quantile 必须在 0 和 1 之间，n_bins 必须为正")
    confidence = confidence_values(scores)[score_name]
    quality = np.asarray(diagnostics["quality"], dtype=float)
    if len(confidence) != len(quality):
        raise ValueError("分数和诊断长度不一致")
    edges = np.quantile(quality, np.linspace(0, 1, n_bins + 1))
    edges = np.maximum.accumulate(edges)
    thresholds = np.empty(n_bins, dtype=float)
    global_threshold = float(np.quantile(confidence, quantile))
    for index in range(n_bins):
        mask = (quality >= edges[index]) & ((quality <= edges[index + 1]) if index == n_bins - 1 else (quality < edges[index + 1]))
        thresholds[index] = float(np.quantile(confidence[mask], quantile)) if mask.any() else global_threshold
    return ConditionalThresholds(score_name, edges.astype(np.float32), thresholds.astype(np.float32), global_threshold)


def apply_conditional_rejection(
    scores: np.ndarray,
    classes: np.ndarray,
    diagnostics: dict[str, np.ndarray],
    calibration: ConditionalThresholds,
) -> np.ndarray:
    """Apply frozen conditional thresholds; no target labels or target statistics are used."""
    confidence = confidence_values(scores)[calibration.score_name]
    threshold = calibration.threshold(diagnostics["quality"])
    predicted = np.asarray(classes)[np.asarray(scores).argmax(axis=1)].copy()
    predicted[confidence < threshold] = 0
    return predicted
