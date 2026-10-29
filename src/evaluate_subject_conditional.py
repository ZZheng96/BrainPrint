"""评估仅在训练折内拟合的类条件原型与对角协方差模型。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from balanced_sinkhorn import sinkhorn_labels
from run_baseline import FEATURE_SETS

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "BrainPrint_cache"
VARIANCE_SHRINKAGE = (0.5, 0.9, 0.99)


def class_conditional_scores(
    x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray
) -> dict[str, np.ndarray]:
    """返回类原型、共享对角马氏距离和类条件对角高斯分数。"""
    scaler = StandardScaler().fit(x_train)
    train = scaler.transform(x_train).astype(np.float64)
    validation = scaler.transform(x_validation).astype(np.float64)
    classes = np.arange(1, 21)
    means = np.stack([train[y_train == label].mean(axis=0) for label in classes])
    residuals = train - means[y_train - 1]
    pooled_variance = np.mean(residuals**2, axis=0) + 1e-6
    class_variances = np.stack(
        [np.mean(residuals[y_train == label] ** 2, axis=0) for label in classes]
    )

    validation_norm = np.sum(validation**2, axis=1)[:, None]
    mean_norm = np.sum(means**2, axis=1)[None, :]
    cross = validation @ means.T
    squared_distance = np.maximum(validation_norm + mean_norm - 2 * cross, 0)
    cosine = cross / np.maximum(
        np.sqrt(validation_norm * mean_norm), 1e-12
    )
    scores = {
        "centroid_cosine": cosine,
        "centroid_euclidean": -0.5 * squared_distance / train.shape[1],
    }

    shared_precision = 1.0 / pooled_variance
    shared_distance = (
        np.sum(validation**2 * shared_precision, axis=1)[:, None]
        + np.sum(means**2 * shared_precision, axis=1)[None, :]
        - 2 * (validation * shared_precision) @ means.T
    )
    scores["diag_lda"] = -0.5 * np.maximum(shared_distance, 0) / train.shape[1]

    for shrinkage in VARIANCE_SHRINKAGE:
        variances = (
            (1 - shrinkage) * class_variances
            + shrinkage * pooled_variance[None, :]
            + 1e-6
        )
        gaussian = np.empty((len(validation), len(classes)), dtype=np.float64)
        scaled_distance = np.empty_like(gaussian)
        for class_index in range(len(classes)):
            difference = validation - means[class_index]
            scaled_square = difference**2 / variances[class_index]
            gaussian[:, class_index] = -0.5 * np.mean(
                scaled_square + np.log(variances[class_index]), axis=1
            )
            scaled_distance[:, class_index] = -0.5 * np.mean(scaled_square, axis=1)
        scores[f"class_gaussian_{shrinkage:g}"] = gaussian
        scores[f"class_distance_{shrinkage:g}"] = scaled_distance
    return scores


def main() -> None:
    x = np.load(CACHE / "train_features.npy", mmap_mode="r")
    columns = FEATURE_SETS["cov"]
    metadata = pd.read_csv(CACHE / "train_metadata.csv")
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    trials = metadata["trial"].to_numpy()
    folds = ((trials - 1) // 10) % 5
    max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    report: dict[str, dict] = {}

    for protocol in ("blocked_oof", "forward"):
        selected = (
            np.ones(len(metadata), dtype=bool)
            if protocol == "blocked_oof"
            else trials > 0.8 * max_trial
        )
        split_values = range(5) if protocol == "blocked_oof" else [-1]
        protocol_scores: dict[str, np.ndarray] = {}
        for split in split_values:
            validation = folds == split if protocol == "blocked_oof" else selected
            fit = ~validation
            for task in sorted(metadata["task"].unique()):
                fit_indices = np.flatnonzero(fit & (tasks == task))
                val_indices = np.flatnonzero(validation & (tasks == task))
                scores = class_conditional_scores(
                    np.asarray(x[fit_indices][:, columns]),
                    subjects[fit_indices],
                    np.asarray(x[val_indices][:, columns]),
                )
                for name, values in scores.items():
                    if name not in protocol_scores:
                        protocol_scores[name] = np.full(
                            (len(metadata), 20), np.nan, dtype=np.float32
                        )
                    protocol_scores[name][val_indices] = values
                print(
                    f"{protocol} split={split} {task}: fit={len(fit_indices)} val={len(val_indices)}",
                    flush=True,
                )

        selected_indices = np.flatnonzero(selected)
        protocol_report = {}
        for name, scores in protocol_scores.items():
            np.save(CACHE / f"train_{protocol}_scores_subject_{name}.npy", scores)
            for use_sinkhorn in (False, True):
                key = name + ("_sinkhorn" if use_sinkhorn else "")
                predictions = scores[selected_indices].argmax(axis=1) + 1
                if use_sinkhorn:
                    for split in split_values:
                        validation = folds == split if protocol == "blocked_oof" else selected
                        fit = ~validation
                        for task in sorted(metadata["task"].unique()):
                            val_indices = np.flatnonzero(validation & (tasks == task))
                            fit_indices = np.flatnonzero(fit & (tasks == task))
                            positions = np.searchsorted(selected_indices, val_indices)
                            prior = pd.Series(subjects[fit_indices]).value_counts().reindex(
                                range(1, 21), fill_value=0
                            ).to_numpy()
                            predictions[positions] = sinkhorn_labels(
                                scores[val_indices], prior, temperature=1.0, iterations=100
                            )
                correct = predictions == subjects[selected_indices]
                errors = selected_indices[~correct]
                protocol_report[key] = {
                    "errors": len(errors),
                    "accuracy": float(correct.mean()),
                    "confusion": [
                        [
                            int(subjects[index]),
                            int(predictions[np.searchsorted(selected_indices, index)]),
                            metadata.at[index, "task"],
                            int(trials[index]),
                        ]
                        for index in errors
                    ],
                }
                print(protocol, key, protocol_report[key], flush=True)
        report[protocol] = protocol_report

    output = ROOT / "research" / "train_subject_conditional_comparison.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
