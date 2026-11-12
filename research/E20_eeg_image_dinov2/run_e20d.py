"""E20D：严格嵌套纯 EEG 开集评估（内层阈值标定，全程不使用 Session）。

与 E02/E06 口径一致：
  外层 5 折留被试，每折 4 人作未知、16 人作注册；
  注册者后 20% trial 作外层已知测试；
  内层 2 折伪未知只用于选择分数与阈值，不接触外层数据。

对比四组纯 EEG 特征（无 Session、无批次统计、无配对）：
  covariance_only   1,026 维 log-covariance（纯 EEG 对照基线）
  covariance5_image DINOv2 从 5 频带协方差图像提取的 768 维
  covariance5_fused 上两者拼接
  topomap5_image    DINOv2 从 5 频带拓扑图提取的 768 维
"""
from __future__ import annotations

import argparse
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

from src.open_set_improvement.core import COV_COLUMNS
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

CACHE = ROOT / "BrainPrint_cache"
OUT = ROOT / "research" / "E20_eeg_image_dinov2"
METRICS = ("C", "identity_macro_f1", "U", "unknown_auroc", "unknown_f1", "macro_f1_21")


def margin(scores: np.ndarray) -> np.ndarray:
    top = np.partition(scores, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def fit_model(features: np.ndarray, subjects: np.ndarray, mask: np.ndarray):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    model.fit(features[mask], subjects[mask])
    return model


def evaluate(features: np.ndarray, subjects: np.ndarray, metadata: pd.DataFrame) -> dict:
    """严格三层：outer 留被试、内层伪未知选阈值、外层一次性评估。"""
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    all_subjects = np.unique(subjects)
    groups = np.array_split(np.random.default_rng(42).permutation(all_subjects), 5)
    rows = []
    for fold, unknown_subjects in enumerate(groups, 1):
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        known = np.isin(subjects, known_subjects)
        outer_fit = known & ~forward

        # 内层：2 折伪未知，仅用于阈值
        inner_order = np.random.default_rng(100 + fold).permutation(known_subjects)
        inner_known_conf, inner_unknown_conf = [], []
        for inner_unknown_subjects in np.array_split(inner_order, 2):
            inner_enrolled = np.setdiff1d(known_subjects, inner_unknown_subjects)
            inner_known_mask = np.isin(subjects, inner_enrolled)
            inner_fit = inner_known_mask & ~forward & ~inner
            inner_eval_known = np.flatnonzero(inner_known_mask & inner)
            inner_eval_unknown = np.flatnonzero(
                np.isin(subjects, inner_unknown_subjects) & ~forward
            )
            model = fit_model(features, subjects, inner_fit)
            inner_known_conf.append(margin(model.decision_function(features[inner_eval_known])))
            inner_unknown_conf.append(margin(model.decision_function(features[inner_eval_unknown])))
        threshold = float(np.quantile(np.concatenate(inner_known_conf), 0.05))

        # 外层：一次性评估
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

        true_all = np.r_[subjects[outer_known_eval], np.zeros(len(outer_unknown_eval), dtype=int)]
        pred_all = np.r_[known_pred, unknown_pred]
        detection_true = np.r_[
            np.ones(len(outer_known_eval)), np.zeros(len(outer_unknown_eval))
        ]
        detection_conf = np.r_[known_conf, unknown_conf]
        rows.append({
            "fold": fold,
            "C": float(np.mean(known_pred == subjects[outer_known_eval])),
            "identity_macro_f1": float(f1_score(
                subjects[outer_known_eval], known_pred, average="macro", zero_division=0
            )),
            "U": float(np.mean(unknown_pred == 0)),
            "unknown_auroc": float(roc_auc_score(detection_true, detection_conf)),
            "unknown_f1": float(f1_score(
                detection_true, detection_conf >= threshold, pos_label=0, zero_division=0
            )),
            "macro_f1_21": float(f1_score(true_all, pred_all, average="macro", zero_division=0)),
            "known_acceptance": float(np.mean(known_conf >= threshold)),
            "threshold": threshold,
            "n_known": int(len(outer_known_eval)),
            "n_unknown": int(len(outer_unknown_eval)),
        })
        print(
            f"  fold {fold}: C={rows[-1]['C']:.4f} U={rows[-1]['U']:.4f} "
            f"AUROC={rows[-1]['unknown_auroc']:.4f} thr={threshold:.4f}",
            flush=True,
        )
    summary = {key: float(np.mean([row[key] for row in rows])) for key in METRICS}
    summary["worst_fold_unknown_auroc"] = float(min(row["unknown_auroc"] for row in rows))
    return {"folds": rows, "summary": summary}


def load_features(name: str) -> np.ndarray | None:
    directory = CACHE / "eeg_image_dinov2"
    if name == "covariance_only":
        return np.asarray(
            np.load(CACHE / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS],
            dtype=np.float32,
        )
    if name == "covariance5_image":
        return np.load(directory / "train_covariance5_dinov2.npy").astype(np.float32)
    if name == "topomap5_image":
        return np.load(directory / "train_topomap5_dinov2.npy").astype(np.float32)
    if name == "covariance5_fused":
        cov = np.asarray(
            np.load(CACHE / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS],
            dtype=np.float32,
        )
        image = np.load(directory / "train_covariance5_dinov2.npy").astype(np.float32)
        return np.concatenate([image, cov], axis=1)
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        nargs="+",
        default=[
            "covariance_only", "covariance5_image", "covariance5_fused", "topomap5_image",
        ],
    )
    parser.add_argument("--output", type=Path, default=OUT / "e20d_nested_pure_eeg.json")
    args = parser.parse_args()

    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial"])
    subjects = metadata["subject"].to_numpy()
    print(
        f"train={len(metadata)} subjects={len(np.unique(subjects))} "
        f"外层已知测试={int(forward_holdout(metadata).sum())}",
        flush=True,
    )

    results = {}
    for name in args.features:
        print(f"\n=== {name} ===", flush=True)
        features = load_features(name)
        started = time.time()
        results[name] = evaluate(features, subjects, metadata)
        print(
            f"{name} summary: {results[name]['summary']} "
            f"({time.time()-started:.0f}s)",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "protocol": {
            "data": "Train Session1 only; 纯 EEG，无 Session 参与",
            "outer": "5-fold subject holdout; 4 unknown, 16 registered",
            "outer_known_test": "last 20% trials per registered subject/task",
            "inner": "2-fold pseudo-unknown within registered subjects; threshold only",
            "threshold": "5th percentile of inner known margin; never outer data",
            "classifier": "StandardScaler + LogisticRegression(C=1.0)",
            "session_used": False,
            "batch_statistics_used": False,
            "pairing_used": False,
        },
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved {args.output}", flush=True)


if __name__ == "__main__":
    main()
