"""E21A：Train 与 Test 协方差特征的偏移性质量化（纯诊断，不做任何决策）。

只读取无标签特征，判断跨会话偏移是「全局幅度缩放」还是「方向性漂移」。
不训练分类器，不产生候选。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eeg_features import COV_BANDS, N_CHANNELS, _DIAGONAL, _OFF_DIAGONAL
from src.open_set_improvement.core import COV_COLUMNS

CACHE = ROOT / "BrainPrint_cache"
OUT = Path(__file__).resolve().parent
N_BANDS = len(COV_BANDS)
N_TRIU = N_CHANNELS * (N_CHANNELS + 1) // 2


def quantiles(values: np.ndarray) -> dict:
    return {f"p{int(q * 100):02d}": float(np.quantile(values, q))
            for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}


def detrend_bands(features: np.ndarray) -> np.ndarray:
    """逐频带扣掉对角线均值（与 eeg_features.normalize_log_cov_determinant 等价）。"""
    blocks = features.reshape(len(features), N_BANDS, N_TRIU)
    diagonal_mean = blocks[:, :, _DIAGONAL].mean(axis=-1, keepdims=True)
    return (blocks - diagonal_mean).reshape(len(features), -1)


def describe(features: np.ndarray, name: str) -> dict:
    blocks = features.reshape(len(features), N_BANDS, N_TRIU)
    diagonal = blocks[:, :, _DIAGONAL].mean(axis=-1)  # (N, 6)
    offdiagonal = blocks[:, :, _OFF_DIAGONAL]
    detrended = detrend_bands(features)
    return {
        "name": name,
        "rows": int(len(features)),
        "norm": quantiles(np.linalg.norm(features, axis=1)),
        "detrended_norm": quantiles(np.linalg.norm(detrended, axis=1)),
        "band_log_variance_mean": [float(x) for x in diagonal.mean(axis=0)],
        "offdiagonal_mean": float(offdiagonal.mean()),
        "offdiagonal_std": float(offdiagonal.std()),
        "mean_vector": features.mean(axis=0),
        "std_vector": features.std(axis=0),
        "detrended_mean_vector": detrended.mean(axis=0),
        "detrended_std_vector": detrended.std(axis=0),
    }


def compare(reference: dict, target: dict, label: str) -> dict:
    delta = target["mean_vector"] - reference["mean_vector"]
    reference_mean = reference["mean_vector"]
    target_mean = target["mean_vector"]
    cosine = float(
        np.dot(reference_mean, target_mean)
        / (np.linalg.norm(reference_mean) * np.linalg.norm(target_mean) + 1e-12)
    )
    scalar = float(np.dot(reference_mean, target_mean) / (np.dot(reference_mean, reference_mean) + 1e-12))
    residual = target_mean - scalar * reference_mean
    standardised = delta / np.maximum(reference["std_vector"], 1e-6)
    drift_correlation = float(
        np.corrcoef(delta, reference_mean)[0, 1]
    )

    detrended_reference = reference["detrended_mean_vector"]
    detrended_target = target["detrended_mean_vector"]
    detrended_delta = detrended_target - detrended_reference
    detrended_cosine = float(
        np.dot(detrended_reference, detrended_target)
        / (np.linalg.norm(detrended_reference) * np.linalg.norm(detrended_target) + 1e-12)
    )
    detrended_scalar = float(
        np.dot(detrended_reference, detrended_target)
        / (np.dot(detrended_reference, detrended_reference) + 1e-12)
    )
    detrended_residual = detrended_target - detrended_scalar * detrended_reference

    diagonal_delta = np.abs(delta.reshape(N_BANDS, N_TRIU)[:, _DIAGONAL]).mean()
    offdiagonal_delta = np.abs(delta.reshape(N_BANDS, N_TRIU)[:, _OFF_DIAGONAL]).mean()

    return {
        "comparison": label,
        "mean_cosine": cosine,
        "best_scalar": scalar,
        "scalar_residual_ratio": float(np.linalg.norm(residual) / (np.linalg.norm(target_mean) + 1e-12)),
        "drift_vs_mean_correlation": drift_correlation,
        "standardised_delta": quantiles(standardised),
        "fraction_abs_standardised_delta_gt1": float(np.mean(np.abs(standardised) > 1.0)),
        "diagonal_abs_delta": float(diagonal_delta),
        "offdiagonal_abs_delta": float(offdiagonal_delta),
        "detrended_mean_cosine": detrended_cosine,
        "detrended_best_scalar": detrended_scalar,
        "detrended_scalar_residual_ratio": float(
            np.linalg.norm(detrended_residual) / (np.linalg.norm(detrended_target) + 1e-12)
        ),
        "detrended_standardised_delta": quantiles(
            detrended_delta / np.maximum(reference["detrended_std_vector"], 1e-6)
        ),
        "detrended_fraction_abs_standardised_delta_gt1": float(
            np.mean(np.abs(detrended_delta / np.maximum(reference["detrended_std_vector"], 1e-6)) > 1.0)
        ),
    }


def main() -> None:
    train = np.asarray(np.load(CACHE / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32)
    test = np.asarray(np.load(CACHE / "test_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32)
    sessions = pd.read_csv(CACHE / "test_metadata.csv", usecols=["session"])["session"].to_numpy()

    blocks = {
        "train": describe(train, "train"),
        "test_session1": describe(test[sessions == "session1"], "test_session1"),
        "test_session2": describe(test[sessions == "session2"], "test_session2"),
    }
    comparisons = [
        compare(blocks["train"], blocks["test_session1"], "train_vs_test_session1"),
        compare(blocks["train"], blocks["test_session2"], "train_vs_test_session2"),
        compare(blocks["test_session1"], blocks["test_session2"], "session1_vs_session2"),
    ]

    report = {
        "rows": {name: block["rows"] for name, block in blocks.items()},
        "distributions": {
            name: {key: block[key] for key in (
                "norm", "detrended_norm", "band_log_variance_mean",
                "offdiagonal_mean", "offdiagonal_std",
            )}
            for name, block in blocks.items()
        },
        "comparisons": comparisons,
    }
    (OUT / "e21a_shift_stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for name, block in blocks.items():
        dist = report["distributions"][name]
        print(f"\n{name} (n={block['rows']})")
        print(f"  norm              {dist['norm']}")
        print(f"  detrended norm    {dist['detrended_norm']}")
        print(f"  band log-var mean {[round(v, 3) for v in dist['band_log_variance_mean']]}")
        print(f"  offdiag mean/std  {dist['offdiagonal_mean']:.3f} / {dist['offdiagonal_std']:.3f}")
    for item in comparisons:
        print(f"\n{item['comparison']}")
        for key in (
            "mean_cosine", "best_scalar", "scalar_residual_ratio",
            "drift_vs_mean_correlation", "fraction_abs_standardised_delta_gt1",
            "diagonal_abs_delta", "offdiagonal_abs_delta",
            "detrended_mean_cosine", "detrended_best_scalar",
            "detrended_scalar_residual_ratio",
            "detrended_fraction_abs_standardised_delta_gt1",
        ):
            print(f"  {key:46s} {item[key]}")


if __name__ == "__main__":
    main()
