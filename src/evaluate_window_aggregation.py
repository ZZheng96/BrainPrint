"""比较两个2秒窗口的预定义决策聚合规则。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

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
        window_scores = np.load(CACHE / f"train_{protocol}_window_scores_window2.npy", mmap_mode="r")
        selected = np.ones(len(metadata), dtype=bool) if protocol == "blocked_oof" else trials > 0.8 * max_trial
        selected_indices = np.flatnonzero(selected)
        pairs = np.stack([window_scores[2 * selected_indices], window_scores[2 * selected_indices + 1]], axis=1)
        margins = np.sort(pairs, axis=2)[:, :, -1] - np.sort(pairs, axis=2)[:, :, -2]
        best_window = margins.argmax(axis=1)
        mean_probability = softmax(pairs, axis=2).mean(axis=1)
        aggregations = {
            "mean_logits": pairs.mean(axis=1),
            "mean_probability": np.log(np.maximum(mean_probability, 1e-12)),
            "first": pairs[:, 0],
            "second": pairs[:, 1],
            "high_margin": pairs[np.arange(len(pairs)), best_window],
        }
        protocol_report = {}
        for name, selected_scores in aggregations.items():
            for use_sinkhorn in (False, True):
                key = name + ("_sinkhorn" if use_sinkhorn else "")
                predictions = selected_scores.argmax(axis=1) + 1
                if use_sinkhorn:
                    for split in (range(5) if protocol == "blocked_oof" else [-1]):
                        validation_parents = np.flatnonzero(folds == split) if protocol == "blocked_oof" else selected_indices
                        fit_parents = np.flatnonzero(folds != split) if protocol == "blocked_oof" else np.flatnonzero(~selected)
                        for task in sorted(metadata["task"].unique()):
                            val_parents = validation_parents[tasks[validation_parents] == task]
                            fit_parents_task = fit_parents[tasks[fit_parents] == task]
                            positions = np.searchsorted(selected_indices, val_parents)
                            prior = pd.Series(subjects[fit_parents_task]).value_counts().reindex(range(1, 21), fill_value=0).to_numpy()
                            predictions[positions] = sinkhorn_labels(selected_scores[positions], prior, 1.0, 100)
                correct = predictions == subjects[selected_indices]
                errors = selected_indices[~correct]
                protocol_report[key] = {
                    "errors": len(errors),
                    "accuracy": float(correct.mean()),
                    "confusion": [[int(subjects[i]), int(predictions[np.searchsorted(selected_indices, i)])] for i in errors],
                }
                print(protocol, key, protocol_report[key], flush=True)
        report[protocol] = protocol_report

    path = ROOT / "research" / "train_window_aggregation_comparison.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
