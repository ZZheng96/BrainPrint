"""评估类内局部近邻：每个身份独立聚合训练折内最相似 Trial。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from balanced_sinkhorn import sinkhorn_labels
from run_baseline import FEATURE_SETS

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "BrainPrint_cache"
NEIGHBOR_COUNTS = (1, 3, 5)


def neighbor_scores(
    x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray
) -> dict[str, np.ndarray]:
    """以类内 top-k 余弦相似度表示非线性的个体身份流形。"""
    scaler = StandardScaler().fit(x_train)
    train = scaler.transform(x_train).astype(np.float64)
    validation = scaler.transform(x_validation).astype(np.float64)
    train /= np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    validation /= np.maximum(np.linalg.norm(validation, axis=1, keepdims=True), 1e-12)
    similarities = validation @ train.T
    scores = {
        f"neighbor_top{k}": np.empty((len(validation), 20), dtype=np.float64)
        for k in NEIGHBOR_COUNTS
    }
    for label in range(1, 21):
        class_similarities = similarities[:, y_train == label]
        for k in NEIGHBOR_COUNTS:
            top = np.partition(class_similarities, -k, axis=1)[:, -k:]
            scores[f"neighbor_top{k}"][:, label - 1] = top.mean(axis=1)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=list(FEATURE_SETS), default="cov")
    args = parser.parse_args()
    x = np.load(CACHE / "train_features.npy", mmap_mode="r")
    columns = FEATURE_SETS[args.features]
    suffix = "" if args.features == "cov" else f"_{args.features}"
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
        selected_indices = np.flatnonzero(selected)
        split_values = range(5) if protocol == "blocked_oof" else [-1]
        protocol_scores: dict[str, np.ndarray] = {}
        for split in split_values:
            validation = folds == split if protocol == "blocked_oof" else selected
            fit = ~validation
            for task in sorted(metadata["task"].unique()):
                fit_indices = np.flatnonzero(fit & (tasks == task))
                val_indices = np.flatnonzero(validation & (tasks == task))
                scores = neighbor_scores(
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

        protocol_report = {}
        for name, scores in protocol_scores.items():
            np.save(CACHE / f"train_{protocol}_scores_subject_{name}{suffix}.npy", scores)
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

    output = ROOT / "research" / f"train_subject_neighbor_comparison{suffix}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
