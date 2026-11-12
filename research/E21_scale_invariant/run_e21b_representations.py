"""E21B：构造逐样本可归纳的尺度不变表示，并用严格嵌套协议评估。

所有表示都由单条 Trial 的协方差特征独立计算，不使用任何批次/会话统计。

候选表示：
  cov_raw            log-covariance 原样（E20D 对照）
  cov_detrend        逐频带扣对角线均值（去除频带功率尺度）
  cov_offdiag        仅保留非对角元（去掉漂移最大的对角线）
  cov_detrend_zband  逐频带在 171 维内 z-score（逐样本）
  cov_detrend_unit   detrend 后整向量单位化（逐样本去全局幅度）
  corr_fisher        协方差转相关矩阵后 Fisher-z
  corr_fisher_zband  corr_fisher 再逐频带 z-score
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cross_session.features import (
    corr_fisher,
    detrend,
    unit_norm,
    zscore_bands,
)
from src.eeg_features import N_CHANNELS
from src.open_set_improvement.core import COV_COLUMNS
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

CACHE = ROOT / "BrainPrint_cache"
FEATURE_CACHE = CACHE / "e21_scale_invariant"
OUT = Path(__file__).resolve().parent
N_BANDS = 6
N_TRIU = N_CHANNELS * (N_CHANNELS + 1) // 2


def build(vectors: np.ndarray) -> dict[str, np.ndarray]:
    detrended = detrend(vectors)
    corr_vectors = corr_fisher(vectors)
    off_diagonal = np.triu_indices(N_CHANNELS)[0] != np.triu_indices(N_CHANNELS)[1]
    return {
        "cov_raw": vectors,
        "cov_detrend": detrended,
        "cov_offdiag": detrended.reshape(len(vectors), N_BANDS, N_TRIU)[..., off_diagonal].reshape(len(vectors), -1),
        "cov_detrend_zband": zscore_bands(detrended),
        "cov_detrend_unit": unit_norm(detrended),
        "corr_fisher": corr_vectors,
        "corr_fisher_zband": zscore_bands(corr_vectors),
    }


def margin(scores: np.ndarray) -> np.ndarray:
    top = np.partition(scores, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def fit_model(features: np.ndarray, subjects: np.ndarray, mask: np.ndarray):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300))
    model.fit(features[mask], subjects[mask])
    return model


def evaluate(features: np.ndarray, subjects: np.ndarray, metadata: pd.DataFrame) -> dict:
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    all_subjects = np.unique(subjects)
    groups = np.array_split(np.random.default_rng(42).permutation(all_subjects), 5)
    rows = []
    for fold, unknown_subjects in enumerate(groups, 1):
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        known = np.isin(subjects, known_subjects)
        outer_fit = known & ~forward

        inner_order = np.random.default_rng(100 + fold).permutation(known_subjects)
        inner_known_conf = []
        for inner_unknown_subjects in np.array_split(inner_order, 2):
            inner_enrolled = np.setdiff1d(known_subjects, inner_unknown_subjects)
            inner_known_mask = np.isin(subjects, inner_enrolled)
            inner_fit = inner_known_mask & ~forward & ~inner
            inner_eval_known = np.flatnonzero(inner_known_mask & inner)
            model = fit_model(features, subjects, inner_fit)
            inner_known_conf.append(margin(model.decision_function(features[inner_eval_known])))
        threshold = float(np.quantile(np.concatenate(inner_known_conf), 0.05))

        outer_known_eval = np.flatnonzero(known & forward)
        outer_unknown_eval = np.flatnonzero(np.isin(subjects, unknown_subjects))
        model = fit_model(features, subjects, outer_fit)
        classes = model.classes_
        known_scores = model.decision_function(features[outer_known_eval])
        unknown_scores = model.decision_function(features[outer_unknown_eval])
        known_conf = margin(known_scores)
        unknown_conf = margin(unknown_scores)
        known_pred = classes[known_scores.argmax(axis=1)].copy()
        known_pred[known_conf < threshold] = 0
        unknown_pred = classes[unknown_scores.argmax(axis=1)].copy()
        unknown_pred[unknown_conf < threshold] = 0

        detection_true = np.r_[np.ones(len(outer_known_eval)), np.zeros(len(outer_unknown_eval))]
        detection_conf = np.r_[known_conf, unknown_conf]
        rows.append({
            "fold": fold,
            "C": float(np.mean(known_pred == subjects[outer_known_eval])),
            "identity_macro_f1": float(f1_score(
                subjects[outer_known_eval], known_pred, average="macro", zero_division=0
            )),
            "U": float(np.mean(unknown_pred == 0)),
            "unknown_auroc": float(roc_auc_score(detection_true, detection_conf)),
            "unknown_f1": float(f1_score(detection_true, detection_conf >= threshold, pos_label=0, zero_division=0)),
            "closed_set_accuracy": float(np.mean(
                classes[known_scores.argmax(axis=1)] == subjects[outer_known_eval]
            )),
            "closed_set_errors": int(np.sum(
                classes[known_scores.argmax(axis=1)] != subjects[outer_known_eval]
            )),
            "known_acceptance": float(np.mean(known_conf >= threshold)),
            "threshold": threshold,
        })
    metrics = ("C", "identity_macro_f1", "U", "unknown_auroc", "unknown_f1",
               "closed_set_accuracy", "known_acceptance")
    summary = {key: float(np.mean([row[key] for row in rows])) for key in metrics}
    summary["closed_set_errors"] = int(sum(row["closed_set_errors"] for row in rows))
    summary["worst_fold_auroc"] = float(min(row["unknown_auroc"] for row in rows))
    summary["folds"] = rows
    return summary


def main() -> None:
    FEATURE_CACHE.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial"])
    subjects = metadata["subject"].to_numpy()
    raw_train = np.asarray(
        np.load(CACHE / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32
    )
    raw_test = np.asarray(
        np.load(CACHE / "test_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32
    )

    train_sets = build(raw_train)
    test_sets = build(raw_test)
    for name in train_sets:
        np.save(FEATURE_CACHE / f"train_{name}.npy", train_sets[name])
        np.save(FEATURE_CACHE / f"test_{name}.npy", test_sets[name])

    results = {}
    for name, features in train_sets.items():
        started = time.time()
        summary = evaluate(np.asarray(features, dtype=np.float32), subjects, metadata)
        results[name] = {"dim": int(features.shape[1]), **summary}
        print(
            f"{name:20s} dim={features.shape[1]:5d} closed={summary['closed_set_accuracy']:.4f} "
            f"err={summary['closed_set_errors']:3d} C={summary['C']:.4f} "
            f"F1id={summary['identity_macro_f1']:.4f} U={summary['U']:.4f} "
            f"AUROC={summary['unknown_auroc']:.4f} acc={summary['known_acceptance']:.4f} "
            f"({time.time()-started:.0f}s)",
            flush=True,
        )
    (OUT / "e21b_representations.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
