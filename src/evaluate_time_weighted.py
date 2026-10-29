"""嵌套时间前推：仅用内部时段选择近期Trial加权强度。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_train_errors import summarize_predictions
from run_baseline import FEATURE_SETS, make_model

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "BrainPrint_cache"


def time_weights(positions: np.ndarray, strength: float) -> np.ndarray:
    weights = np.exp(strength * (positions - 0.5))
    return weights / weights.mean()


def main() -> None:
    x = np.load(CACHE / "train_features.npy", mmap_mode="r")
    columns = FEATURE_SETS["cov"]
    metadata = pd.read_csv(CACHE / "train_metadata.csv")
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    trials = metadata["trial"].to_numpy()
    max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    positions = trials / max_trial
    inner_fit = positions <= 0.6
    inner_validation = (positions > 0.6) & (positions <= 0.8)
    outer_fit = positions <= 0.8
    outer_validation = positions > 0.8
    strengths = (0.0, 1.0, 2.0, 4.0)
    scores = np.full((len(metadata), 20), np.nan, dtype=np.float32)
    task_selection = {}

    for task in sorted(metadata["task"].unique()):
        inner_results = {}
        for strength in strengths:
            fit_indices = np.flatnonzero(inner_fit & (tasks == task))
            val_indices = np.flatnonzero(inner_validation & (tasks == task))
            model = make_model("ridge", seed=42, alpha=100.0)
            model.fit(
                x[fit_indices][:, columns],
                subjects[fit_indices],
                ridgeclassifier__sample_weight=time_weights(positions[fit_indices], strength),
            )
            predicted = model.predict(x[val_indices][:, columns])
            inner_results[strength] = float(np.mean(predicted == subjects[val_indices]))
        best_strength = min(strengths, key=lambda value: (-inner_results[value], value))
        fit_indices = np.flatnonzero(outer_fit & (tasks == task))
        val_indices = np.flatnonzero(outer_validation & (tasks == task))
        model = make_model("ridge", seed=42, alpha=100.0)
        model.fit(
            x[fit_indices][:, columns],
            subjects[fit_indices],
            ridgeclassifier__sample_weight=time_weights(positions[fit_indices], best_strength),
        )
        scores[np.ix_(val_indices, model.classes_.astype(int) - 1)] = model.decision_function(x[val_indices][:, columns])
        task_selection[task] = {
            "inner_accuracy": {str(k): v for k, v in inner_results.items()},
            "selected_strength": best_strength,
        }
        print(task, task_selection[task], flush=True)

    indices = np.flatnonzero(outer_validation)
    labels = scores[indices].argmax(axis=1) + 1
    output = metadata.loc[outer_validation, ["path", "subject", "task", "task_folder", "trial"]].copy()
    output["predicted_label"] = labels
    output["correct"] = labels == subjects[indices]
    output.to_csv(CACHE / "train_forward_predictions_time_weighted.csv", index=False)
    np.save(CACHE / "train_forward_scores_time_weighted.npy", scores)
    report = summarize_predictions(output, "nested forward time weighting")
    report["task_selection"] = task_selection
    path = ROOT / "research" / "train_error_summary_time_weighted_forward.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"accuracy": report["accuracy"], "errors": report["errors"], "confusion": report["confusion_pairs"], "tasks": task_selection}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
