"""E24B：加性噪声抵消（noise-floor subtraction）。

E24A 证据：Train 注入 SNR 30 dB 白噪声后，log 协方差范数从 59.7 降到 31.7，
与 Test S2 低范数组（33.9）几乎重合。若扰动确实近似加性白噪声 σ²I，
则从每个频带协方差中减去 σ²I 就能还原干净协方差：

  ŝ² = 逐样本在高频带（70–90 Hz，脑电功率最低）的对角均值
  C_b' = C_b - ŝ² I，再投影回正定并取 log

全程逐样本、归纳式；不用 Test 标签，不使用 Session。
评估：Train 严格嵌套（必须不退化）+ Test 审计（低/高范数组的范数与分数变化）。
"""
from __future__ import annotations

import json
import sys
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
    N_BANDS,
    N_CHANNELS,
    cov_from_log,
    log_from_cov,
    vectorize,
)
from src.open_set_improvement.core import COV_COLUMNS
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

CACHE = ROOT / "BrainPrint_cache"
OUT = Path(__file__).resolve().parent
CHUNK = 3000
LOW_GROUP_NORM = 45.0


def denoise(vectors: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """按高频带噪声底逐样本减去 σ²I。"""
    covariances = cov_from_log(vectors)
    noise = covariances[:, N_BANDS - 1].diagonal(axis1=-2, axis2=-1).mean(axis=1)
    noise = np.clip(noise, 0.0, None) * scale
    identity = np.eye(N_CHANNELS)[None, None]
    corrected = covariances - noise[:, None, None, None] * identity
    corrected = (corrected + np.swapaxes(corrected, -1, -2)) / 2.0
    return vectorize(log_from_cov(corrected))


def apply_chunked(vectors: np.ndarray, scale: float) -> np.ndarray:
    return np.concatenate([
        denoise(vectors[start : start + CHUNK], scale) for start in range(0, len(vectors), CHUNK)
    ])


def margin(scores: np.ndarray) -> np.ndarray:
    top = np.partition(scores, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def fit(features: np.ndarray, labels: np.ndarray):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300))
    model.fit(features, labels)
    return model


def nested(features, subjects, metadata) -> dict:
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    groups = np.array_split(np.random.default_rng(42).permutation(np.unique(subjects)), 5)
    rows = []
    for fold, unknown in enumerate(groups, 1):
        known_subjects = np.setdiff1d(np.unique(subjects), unknown)
        is_known = np.isin(subjects, known_subjects)
        fit_mask = is_known & ~forward
        inner_conf = []
        for inner_unknown in np.array_split(np.random.default_rng(100 + fold).permutation(known_subjects), 2):
            enrolled = np.setdiff1d(known_subjects, inner_unknown)
            mask_known = np.isin(subjects, enrolled)
            model = fit(features[mask_known & ~forward & ~inner], subjects[mask_known & ~forward & ~inner])
            inner_conf.append(margin(model.decision_function(features[np.flatnonzero(mask_known & inner)])))
        threshold = float(np.quantile(np.concatenate(inner_conf), 0.05))
        evaluate = np.flatnonzero(is_known & forward)
        unknown_eval = np.flatnonzero(np.isin(subjects, unknown))
        model = fit(features[fit_mask], subjects[fit_mask])
        classes = model.classes_
        scores = model.decision_function(features[evaluate])
        unknown_scores = model.decision_function(features[unknown_eval])
        conf, unknown_conf = margin(scores), margin(unknown_scores)
        pred = classes[scores.argmax(1)].copy()
        pred[conf < threshold] = 0
        unknown_pred = classes[unknown_scores.argmax(1)].copy()
        unknown_pred[unknown_conf < threshold] = 0
        rows.append({
            "closed_acc": float(np.mean(classes[scores.argmax(1)] == subjects[evaluate])),
            "errors": int(np.sum(classes[scores.argmax(1)] != subjects[evaluate])),
            "C": float(np.mean(pred == subjects[evaluate])),
            "identity_f1": float(f1_score(subjects[evaluate], pred, average="macro", zero_division=0)),
            "U": float(np.mean(unknown_pred == 0)),
            "auroc": float(roc_auc_score(np.r_[np.ones(len(evaluate)), np.zeros(len(unknown_eval))], np.r_[conf, unknown_conf])),
        })
    summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    summary["errors"] = int(sum(row["errors"] for row in rows))
    return summary


def main() -> None:
    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial"])
    subjects = metadata["subject"].to_numpy()
    inner = inner_holdout(metadata)
    test_meta = pd.read_csv(CACHE / "test_metadata.csv", usecols=["sample", "session"])
    known = (test_meta["session"] == "session2").to_numpy()

    train_raw = np.asarray(np.load(CACHE / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32)
    test_raw = np.asarray(np.load(CACHE / "test_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32)
    test_norm = np.linalg.norm(test_raw, axis=1)
    low_group = known & (test_norm < LOW_GROUP_NORM)
    high_group = known & (test_norm >= LOW_GROUP_NORM)
    print(f"低范数组 {int(low_group.sum())} 条，高范数组 {int(high_group.sum())} 条")

    report = {}
    for scale in (0.0, 0.5, 1.0, 1.5):
        train_features = apply_chunked(train_raw, scale) if scale else train_raw
        test_features = apply_chunked(test_raw, scale) if scale else test_raw
        summary = nested(np.asarray(train_features, np.float32), subjects, metadata)
        model = fit(np.asarray(train_features, np.float32), subjects)
        inner_scores = margin(model.decision_function(np.asarray(train_features, np.float32)[inner]))
        test_scores = model.decision_function(np.asarray(test_features, np.float32))
        test_margin = margin(test_scores)
        threshold = float(np.quantile(inner_scores, 0.05))
        new_norm = np.linalg.norm(test_features, axis=1)
        report[f"scale_{scale}"] = {
            "nested": summary,
            "test_norm_low_group_p50": float(np.median(new_norm[low_group])),
            "test_norm_high_group_p50": float(np.median(new_norm[high_group])),
            "test_norm_session1_p50": float(np.median(new_norm[~known])),
            "s2_accept_q05": float(np.mean(test_margin[known] >= threshold)),
            "low_group_accept": float(np.mean(test_margin[low_group] >= threshold)),
            "high_group_accept": float(np.mean(test_margin[high_group] >= threshold)),
            "auroc_s2_vs_s1": float(roc_auc_score(known, np.sort(test_scores, axis=1)[:, -1])),
        }
        row = report[f"scale_{scale}"]
        print(f"scale={scale:.1f} | closed={summary['closed_acc']:.4f} err={summary['errors']:3d} "
              f"C={summary['C']:.4f} U={summary['U']:.4f} AUROC={summary['auroc']:.4f} | "
              f"norm low={row['test_norm_low_group_p50']:.1f} high={row['test_norm_high_group_p50']:.1f} | "
              f"accept S2={row['s2_accept_q05']:.3f} low={row['low_group_accept']:.3f} "
              f"high={row['high_group_accept']:.3f}", flush=True)

    (OUT / "e24b_noise_subtraction.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
