"""严格嵌套的逐身份低频近邻校准与全局 Ridge 集成。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from balanced_sinkhorn import sinkhorn_labels
from evaluate_subject_neighbors import neighbor_scores
from run_baseline import FEATURE_SETS

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "BrainPrint_cache"


def row_zscore(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.mean(axis=1, keepdims=True)) / np.maximum(
        scores.std(axis=1, keepdims=True), 1e-6
    )


def top3_scores(
    x: np.ndarray,
    columns: np.ndarray,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    subjects: np.ndarray,
) -> np.ndarray:
    return neighbor_scores(
        np.asarray(x[fit_indices][:, columns]),
        subjects[fit_indices],
        np.asarray(x[validation_indices][:, columns]),
    )["neighbor_top3"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocols", nargs="+", choices=["blocked_oof", "forward"], default=["blocked_oof", "forward"])
    args = parser.parse_args()
    x = np.load(CACHE / "train_features.npy", mmap_mode="r")
    metadata = pd.read_csv(CACHE / "train_metadata.csv")
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    trials = metadata["trial"].to_numpy()
    folds = ((trials - 1) // 10) % 5
    max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    report: dict[str, dict] = {}

    for protocol in args.protocols:
        selected = (
            np.ones(len(metadata), dtype=bool)
            if protocol == "blocked_oof"
            else trials > 0.8 * max_trial
        )
        selected_indices = np.flatnonzero(selected)
        split_values = range(5) if protocol == "blocked_oof" else [-1]
        calibrated = np.full((len(metadata), 20), np.nan, dtype=np.float32)
        for split in split_values:
            validation = folds == split if protocol == "blocked_oof" else selected
            fit = ~validation
            for task in sorted(metadata["task"].unique()):
                fit_indices = np.flatnonzero(fit & (tasks == task))
                val_indices = np.flatnonzero(validation & (tasks == task))
                inner_bands = [
                    np.full((len(metadata), 20), np.nan, dtype=np.float64)
                    for _ in range(2)
                ]
                inner_splits = sorted(np.unique(folds[fit])) if protocol == "blocked_oof" else [-1]
                for inner_split in inner_splits:
                    if protocol == "blocked_oof":
                        inner_validation = fit & (folds == inner_split)
                    else:
                        inner_validation = fit & (trials > 0.6 * max_trial)
                    inner_fit = fit & ~inner_validation
                    inner_fit_indices = np.flatnonzero(inner_fit & (tasks == task))
                    inner_val_indices = np.flatnonzero(inner_validation & (tasks == task))
                    for band_index, feature_name in enumerate(("cov_1_8", "cov_1_13")):
                        inner_bands[band_index][inner_val_indices] = top3_scores(
                            x,
                            FEATURE_SETS[feature_name],
                            inner_fit_indices,
                            inner_val_indices,
                            subjects,
                        )
                outer_bands = [
                    top3_scores(x, FEATURE_SETS[feature_name], fit_indices, val_indices, subjects)
                    for feature_name in ("cov_1_8", "cov_1_13")
                ]
                calibration_indices = (
                    fit_indices
                    if protocol == "blocked_oof"
                    else np.flatnonzero(inner_validation & (tasks == task))
                )
                for class_index in range(20):
                    inner_features = np.column_stack(
                        [scores[calibration_indices, class_index] for scores in inner_bands]
                    )
                    outer_features = np.column_stack(
                        [scores[:, class_index] for scores in outer_bands]
                    )
                    if not np.isfinite(inner_features).all():
                        raise RuntimeError("Incomplete inner OOF neighbor scores")
                    calibrator = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000),
                    )
                    calibrator.fit(inner_features, subjects[calibration_indices] == class_index + 1)
                    calibrated[val_indices, class_index] = calibrator.decision_function(
                        outer_features
                    )
                print(
                    f"{protocol} split={split} {task}: "
                    f"inner_oof={len(fit_indices)} outer={len(val_indices)}",
                    flush=True,
                )

        np.save(CACHE / f"train_{protocol}_scores_subject_calibrated_neighbors.npy", calibrated)
        full = np.asarray(np.load(CACHE / f"train_{protocol}_scores_cov_alpha100.npy"))
        window = np.asarray(np.load(CACHE / f"train_{protocol}_scores_window2.npy"))
        base = 0.5 * full + 0.5 * window
        combinations = {
            "calibrated_neighbors": calibrated,
            "base_calibrated0.05": base + 0.05 * row_zscore(calibrated),
            "base_calibrated0.1": base + 0.1 * row_zscore(calibrated),
            "base_calibrated0.25": base + 0.25 * row_zscore(calibrated),
            "zbase_calibrated0.25": row_zscore(base) + 0.25 * row_zscore(calibrated),
            "zbase_calibrated0.5": row_zscore(base) + 0.5 * row_zscore(calibrated),
            "zbase_calibrated1": row_zscore(base) + row_zscore(calibrated),
        }
        protocol_report = {}
        for name, scores in combinations.items():
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

    suffix = "" if len(args.protocols) == 2 else f"_{args.protocols[0]}"
    output = ROOT / "research" / f"train_calibrated_neighbor_comparison{suffix}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
