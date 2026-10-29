"""用训练折类别先验容量执行无标签匈牙利硬分配。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "BrainPrint_cache"


def capacities_from_prior(prior: np.ndarray, n_samples: int) -> np.ndarray:
    expected = prior / prior.sum() * n_samples
    capacities = np.floor(expected).astype(int)
    remainder = n_samples - capacities.sum()
    order = np.argsort(expected - capacities)[::-1]
    capacities[order[:remainder]] += 1
    return capacities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["blocked_oof", "forward"], default="forward")
    parser.add_argument("--scores", choices=["cov", "ensemble"], default="ensemble")
    args = parser.parse_args()

    metadata = pd.read_csv(CACHE / "train_metadata.csv")
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    trials = metadata["trial"].to_numpy()
    folds = ((trials - 1) // 10) % 5
    max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    selected = np.ones(len(metadata), dtype=bool) if args.protocol == "blocked_oof" else trials > 0.8 * max_trial
    if args.scores == "cov":
        scores = np.load(CACHE / f"train_{args.protocol}_scores_cov_alpha100.npy", mmap_mode="r")
    else:
        full = np.load(CACHE / f"train_{args.protocol}_scores_cov_alpha100.npy", mmap_mode="r")
        low = np.load(CACHE / f"train_{args.protocol}_scores_cov_1_45_alpha100.npy", mmap_mode="r")
        scores = 0.5 * np.asarray(full) + 0.5 * np.asarray(low)

    selected_indices = np.flatnonzero(selected)
    predictions = np.zeros(len(selected_indices), dtype=int)
    split_values = range(5) if args.protocol == "blocked_oof" else [-1]
    for split in split_values:
        validation = folds == split if args.protocol == "blocked_oof" else selected
        fit = ~validation
        for task in sorted(metadata["task"].unique()):
            val_indices = np.flatnonzero(validation & (tasks == task))
            fit_indices = np.flatnonzero(fit & (tasks == task))
            positions = np.searchsorted(selected_indices, val_indices)
            prior = pd.Series(subjects[fit_indices]).value_counts().reindex(range(1, 21), fill_value=0).to_numpy()
            capacities = capacities_from_prior(prior, len(val_indices))
            class_slots = np.repeat(np.arange(1, 21), capacities)
            task_scores = np.asarray(scores[val_indices])
            task_scores = (task_scores - task_scores.mean(axis=0)) / np.maximum(task_scores.std(axis=0), 1e-6)
            rows, columns = linear_sum_assignment(task_scores[:, class_slots - 1], maximize=True)
            predictions[positions[rows]] = class_slots[columns]
            print(f"split={split} {task}: n={len(val_indices)} capacities={capacities.tolist()}", flush=True)

    correct = predictions == subjects[selected_indices]
    errors = selected_indices[~correct]
    report = {
        "protocol": args.protocol,
        "scores": args.scores,
        "n": len(selected_indices),
        "errors": len(errors),
        "accuracy": float(correct.mean()),
        "error_subjects": {str(int(k)): int(v) for k, v in pd.Series(subjects[errors]).value_counts().sort_index().items()},
        "confusion": [
            [int(subjects[index]), int(predictions[np.searchsorted(selected_indices, index)])]
            for index in errors
        ],
    }
    output = metadata.loc[selected, ["path", "subject", "task", "trial"]].copy()
    output["predicted_label"] = predictions
    output["correct"] = correct
    output.to_csv(CACHE / f"train_{args.protocol}_hard_{args.scores}_predictions.csv", index=False)
    path = ROOT / "research" / f"train_hard_assignment_{args.scores}_{args.protocol}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
