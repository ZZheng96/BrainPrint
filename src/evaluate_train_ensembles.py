"""公平比较完整频带与1-45Hz协方差模型的折外线性集成。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from balanced_sinkhorn import sinkhorn_labels

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "BrainPrint_cache"


def main() -> None:
    metadata = pd.read_csv(CACHE / "train_metadata.csv")
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    trials = metadata["trial"].to_numpy()
    folds = ((trials - 1) // 10) % 5
    max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    report = {}

    for protocol in ("blocked_oof", "forward"):
        full = np.load(CACHE / f"train_{protocol}_scores_cov_alpha100.npy", mmap_mode="r")
        low = np.load(CACHE / f"train_{protocol}_scores_cov_1_45_alpha100.npy", mmap_mode="r")
        selected = np.ones(len(metadata), dtype=bool) if protocol == "blocked_oof" else trials > 0.8 * max_trial
        selected_indices = np.flatnonzero(selected)
        protocol_report = {}
        for weight_full in (0.5, 0.75, 0.9):
            combined = weight_full * np.asarray(full) + (1 - weight_full) * np.asarray(low)
            for use_sinkhorn in (False, True):
                key = f"full{weight_full:g}_low{1-weight_full:g}{'_sinkhorn' if use_sinkhorn else ''}"
                predictions = combined[selected_indices].argmax(axis=1) + 1
                if use_sinkhorn:
                    split_values = range(5) if protocol == "blocked_oof" else [-1]
                    for split in split_values:
                        validation = folds == split if protocol == "blocked_oof" else selected
                        fit = ~validation
                        for task in sorted(metadata["task"].unique()):
                            val_indices = np.flatnonzero(validation & (tasks == task))
                            fit_indices = np.flatnonzero(fit & (tasks == task))
                            positions = np.searchsorted(selected_indices, val_indices)
                            prior = pd.Series(subjects[fit_indices]).value_counts().reindex(range(1, 21), fill_value=0).to_numpy()
                            predictions[positions] = sinkhorn_labels(combined[val_indices], prior, 1.0, 100)
                correct = predictions == subjects[selected_indices]
                error_indices = selected_indices[~correct]
                protocol_report[key] = {
                    "errors": len(error_indices),
                    "accuracy": float(correct.mean()),
                    "error_subjects": {
                        str(int(subject)): int(count)
                        for subject, count in pd.Series(subjects[error_indices]).value_counts().sort_index().items()
                    },
                    "confusion": [
                        [int(subjects[index]), int(predictions[np.searchsorted(selected_indices, index)])]
                        for index in error_indices
                    ],
                }
                output = metadata.loc[selected, ["path", "subject", "task", "trial"]].copy()
                output["predicted_label"] = predictions
                output["correct"] = correct
                output.to_csv(CACHE / f"train_{protocol}_{key}_predictions.csv", index=False)
                print(protocol, key, protocol_report[key], flush=True)
        report[protocol] = protocol_report

    path = ROOT / "research" / "train_ensemble_comparison.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
