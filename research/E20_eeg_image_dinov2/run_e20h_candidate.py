"""E20H：用 two_stage 判据生成 Test 候选，并做无标签分布审计。

纯 EEG：推理只读 EEG 图像特征与 task；不读 session、不读标签。
Session 仅在候选写出后用于离线审计，不参与任何决策。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.open_set_improvement.core import COV_COLUMNS
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

CACHE = ROOT / "BrainPrint_cache"
OUT = Path(__file__).resolve().parent
DATA = ROOT / "Brainprint_Competition"


def scores_of(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(scores, axis=1)
    return ordered[:, -1], ordered[:, -1] - ordered[:, -2]


def fit_model(features, subjects, mask):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    model.fit(features[mask], subjects[mask])
    return model


def calibrate_thresholds(features, subjects, metadata) -> dict:
    """用全部训练数据的内层伪未知标定两个阈值（与 E20G 同口径）。"""
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    all_subjects = np.unique(subjects)
    inner_max, inner_margin = [], []
    # 复用外层折的内层划分，保证与 E20G 一致
    groups = np.array_split(np.random.default_rng(42).permutation(all_subjects), 5)
    for fold, unknown_subjects in enumerate(groups, 1):
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        inner_order = np.random.default_rng(100 + fold).permutation(known_subjects)
        for inner_unknown_subjects in np.array_split(inner_order, 2):
            inner_enrolled = np.setdiff1d(known_subjects, inner_unknown_subjects)
            inner_known_mask = np.isin(subjects, inner_enrolled)
            inner_fit = inner_known_mask & ~forward & ~inner
            inner_eval_known = np.flatnonzero(inner_known_mask & inner)
            model = fit_model(features, subjects, inner_fit)
            mx, mg = scores_of(model.decision_function(features[inner_eval_known]))
            inner_max.append(mx)
            inner_margin.append(mg)
    inner_max = np.concatenate(inner_max)
    inner_margin = np.concatenate(inner_margin)
    return {
        "margin_threshold": float(np.quantile(inner_margin, 0.001)),
        "max_threshold": float(np.quantile(inner_max, 0.05)),
        "n_inner_known": int(len(inner_max)),
    }


def main():
    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial"])
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()

    cov = np.asarray(
        np.load(CACHE / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32
    )
    image = np.load(CACHE / "eeg_image_dinov2" / "train_covariance5_dinov2.npy").astype(np.float32)
    fused = np.concatenate([image, cov], axis=1)

    print("标定阈值（内层伪未知）...", flush=True)
    thresholds = calibrate_thresholds(fused, subjects, metadata)
    print(f"  {thresholds}", flush=True)

    print("全量训练 + Test 推理...", flush=True)
    model = fit_model(fused, subjects, np.ones(len(subjects), dtype=bool))
    classes = model.classes_

    test_meta = pd.read_csv(CACHE / "test_metadata.csv", usecols=["sample", "task"])
    test_cov = np.asarray(
        np.load(CACHE / "test_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32
    )
    test_image = np.load(
        CACHE / "eeg_image_dinov2" / "test_covariance5_dinov2.npy"
    ).astype(np.float32)
    test_fused = np.concatenate([test_image, test_cov], axis=1)

    scores = model.decision_function(test_fused)
    max_score, margin = scores_of(scores)
    predicted = classes[scores.argmax(axis=1)].astype(int)
    accept = (margin >= thresholds["margin_threshold"]) & (max_score >= thresholds["max_threshold"])
    predicted[~accept] = 0

    template = pd.read_csv(DATA / "test.csv", usecols=["Sample"])
    sample_to_row = dict(zip(test_meta["sample"], range(len(test_meta))))
    submission = template.copy()
    submission["Label"] = predicted[template["Sample"].map(sample_to_row).to_numpy()]
    path = OUT / "test_candidate_two_stage.csv"
    submission.to_csv(path, index=False)

    # 候选写出后才读 session，仅作离线审计
    audit_meta = pd.read_csv(CACHE / "test_metadata.csv", usecols=["sample", "session", "task"])
    audit = audit_meta.copy()
    audit["predicted"] = predicted
    audit["rejected"] = predicted == 0
    by_session = audit.groupby("session").agg(
        n=("rejected", "size"), rejected=("rejected", "sum"), reject_rate=("rejected", "mean")
    ).reset_index()
    by_task = audit.groupby("task").agg(
        n=("rejected", "size"), rejected=("rejected", "sum"), reject_rate=("rejected", "mean")
    ).reset_index()
    by_session_task = audit.groupby(["session", "task"]).agg(
        n=("rejected", "size"), rejected=("rejected", "sum"), reject_rate=("rejected", "mean")
    ).reset_index()

    result = {
        "protocol": {
            "note": "纯 EEG；推理只读 EEG 特征与 task；session 仅候选写出后审计",
            "features": "covariance5 DINOv2 (768) + log-covariance (1026)",
            "rejection": "two_stage: margin>=q0.001 AND max>=q0.05, 内层伪未知标定",
            "session_used_for_inference": False,
        },
        "thresholds": thresholds,
        "candidate": {
            "path": str(path),
            "rows": int(len(submission)),
            "rejected": int((predicted == 0).sum()),
            "reject_rate": float((predicted == 0).mean()),
            "label_counts": {str(k): int(v) for k, v in zip(*np.unique(predicted, return_counts=True))},
            "sample_order_matches": bool(submission["Sample"].equals(template["Sample"])),
            "labels_valid": bool(submission["Label"].between(0, 20).all()),
        },
        "audit_by_session": by_session.to_dict("records"),
        "audit_by_task": by_task.to_dict("records"),
        "audit_by_session_task": by_session_task.to_dict("records"),
    }
    (OUT / "e20h_test_candidate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "candidate": result["candidate"],
        "by_session": result["audit_by_session"],
        "by_task": result["audit_by_task"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
