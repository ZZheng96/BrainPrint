"""检验全局 Ridge、双窗口与低频个体局部模型的固定折外集成。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from balanced_sinkhorn import sinkhorn_labels

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "BrainPrint_cache"


def row_zscore(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.mean(axis=1, keepdims=True)) / np.maximum(
        scores.std(axis=1, keepdims=True), 1e-6
    )


def main() -> None:
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
        full = np.asarray(np.load(CACHE / f"train_{protocol}_scores_cov_alpha100.npy"))
        window = np.asarray(np.load(CACHE / f"train_{protocol}_scores_window2.npy"))
        low = np.asarray(np.load(CACHE / f"train_{protocol}_scores_cov_1_13_alpha100.npy"))
        neighbor = np.asarray(
            np.load(CACHE / f"train_{protocol}_scores_subject_neighbor_top3_cov_1_13.npy")
        )
        delta_neighbor = np.asarray(
            np.load(CACHE / f"train_{protocol}_scores_subject_neighbor_top3_cov_1_8.npy")
        )
        base = 0.5 * full + 0.5 * window
        combinations = {
            "window_full": base,
            "window_full_low0.25": base + 0.25 * low,
            "window_full_low0.5": base + 0.5 * low,
            "window_full_low1": base + low,
            "window_full_neighbor0.25": row_zscore(base) + 0.25 * row_zscore(neighbor),
            "window_full_neighbor0.5": row_zscore(base) + 0.5 * row_zscore(neighbor),
            "window_full_neighbor1": row_zscore(base) + row_zscore(neighbor),
            "window_full_low0.5_neighbor0.5": (
                row_zscore(base) + 0.5 * row_zscore(low) + 0.5 * row_zscore(neighbor)
            ),
            "window_full_delta0.25": row_zscore(base) + 0.25 * row_zscore(delta_neighbor),
            "window_full_delta0.5": row_zscore(base) + 0.5 * row_zscore(delta_neighbor),
            "window_full_delta1": row_zscore(base) + row_zscore(delta_neighbor),
            "window_full_lowfreq_neighbors": (
                row_zscore(base) + 0.5 * row_zscore(neighbor) + 0.5 * row_zscore(delta_neighbor)
            ),
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
                output = metadata.loc[selected, ["path", "subject", "task", "trial"]].copy()
                output["predicted_label"] = predictions
                output["correct"] = correct
                output.to_csv(CACHE / f"train_{protocol}_{key}_predictions.csv", index=False)
        report[protocol] = protocol_report

    output = ROOT / "research" / "train_subject_ensemble_comparison.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
