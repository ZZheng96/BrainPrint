"""比较2秒窗口与4秒模型的固定等权折外集成。"""
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
        window = np.load(CACHE / f"train_{protocol}_scores_window2.npy", mmap_mode="r")
        full = np.load(CACHE / f"train_{protocol}_scores_cov_alpha100.npy", mmap_mode="r")
        low = np.load(CACHE / f"train_{protocol}_scores_cov_1_45_alpha100.npy", mmap_mode="r")
        aligned = np.load(CACHE / f"train_{protocol}_scores_cov_alpha100_zscore.npy", mmap_mode="r")
        combinations = {
            "window_full": 0.5 * np.asarray(window) + 0.5 * np.asarray(full),
            "window_full_low": 0.5 * np.asarray(window) + 0.25 * np.asarray(full) + 0.25 * np.asarray(low),
            "window_aligned": 0.5 * np.asarray(window) + 0.5 * np.asarray(aligned),
        }
        selected = np.ones(len(metadata), dtype=bool) if protocol == "blocked_oof" else trials > 0.8 * max_trial
        selected_indices = np.flatnonzero(selected)
        protocol_report = {}
        for name, scores in combinations.items():
            for use_sinkhorn in (False, True):
                key = name + ("_sinkhorn" if use_sinkhorn else "")
                predictions = scores[selected_indices].argmax(axis=1) + 1
                if use_sinkhorn:
                    for split in (range(5) if protocol == "blocked_oof" else [-1]):
                        validation = folds == split if protocol == "blocked_oof" else selected
                        fit = ~validation
                        for task in sorted(metadata["task"].unique()):
                            val_indices = np.flatnonzero(validation & (tasks == task))
                            fit_indices = np.flatnonzero(fit & (tasks == task))
                            positions = np.searchsorted(selected_indices, val_indices)
                            prior = pd.Series(subjects[fit_indices]).value_counts().reindex(range(1, 21), fill_value=0).to_numpy()
                            predictions[positions] = sinkhorn_labels(scores[val_indices], prior, 1.0, 100)
                correct = predictions == subjects[selected_indices]
                errors = selected_indices[~correct]
                protocol_report[key] = {
                    "errors": len(errors),
                    "accuracy": float(correct.mean()),
                    "error_subjects": {str(int(k)): int(v) for k, v in pd.Series(subjects[errors]).value_counts().sort_index().items()},
                    "confusion": [[int(subjects[i]), int(predictions[np.searchsorted(selected_indices, i)])] for i in errors],
                }
                output = metadata.loc[selected, ["path", "subject", "task", "trial"]].copy()
                output["predicted_label"] = predictions
                output["correct"] = correct
                output.to_csv(CACHE / f"train_{protocol}_{key}_predictions.csv", index=False)
                print(protocol, key, protocol_report[key], flush=True)
        report[protocol] = protocol_report

    path = ROOT / "research" / "train_window_ensemble_comparison.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
