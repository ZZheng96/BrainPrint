"""固定任务特异融合规则的双协议折外验证。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .balanced_sinkhorn import sinkhorn_labels
except ImportError:  # 直接执行 src 内脚本
    from balanced_sinkhorn import sinkhorn_labels

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "BrainPrint_cache"


def row_zscore(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.mean(axis=1, keepdims=True)) / np.maximum(
        scores.std(axis=1, keepdims=True), 1e-6
    )


def build_task_scores(
    tasks: np.ndarray,
    full: np.ndarray,
    window: np.ndarray,
    low: np.ndarray,
    low_neighbor: np.ndarray,
    delta_neighbor: np.ndarray,
) -> np.ndarray:
    """使用固定任务规则组合分数；不在推理时访问身份标签。"""
    base = 0.5 * full + 0.5 * window
    scores = np.zeros_like(base)
    ec = tasks == "EC"
    scores[ec] = (
        row_zscore(base)[ec]
        + 0.25 * row_zscore(low_neighbor)[ec]
        + 0.5 * row_zscore(delta_neighbor)[ec]
    )
    eo = tasks == "EO"
    scores[eo] = (base + 0.25 * low)[eo]
    imagery = tasks == "Imagery"
    scores[imagery] = window[imagery]
    ssvep = tasks == "SSVEP"
    scores[ssvep] = (base + 0.5 * low)[ssvep]
    return scores


def main() -> None:
    metadata = pd.read_csv(CACHE / "train_metadata.csv")
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    trials = metadata["trial"].to_numpy()
    folds = ((trials - 1) // 10) % 5
    max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    report = {}

    for protocol in ("blocked_oof", "forward"):
        selected = (
            np.ones(len(metadata), dtype=bool)
            if protocol == "blocked_oof"
            else trials > 0.8 * max_trial
        )
        selected_indices = np.flatnonzero(selected)
        full = np.load(CACHE / f"train_{protocol}_scores_cov_alpha100.npy")
        window = np.load(CACHE / f"train_{protocol}_scores_window2.npy")
        low = np.load(CACHE / f"train_{protocol}_scores_cov_1_13_alpha100.npy")
        low_neighbor = np.load(
            CACHE / f"train_{protocol}_scores_subject_neighbor_top3_cov_1_13.npy"
        )
        delta_neighbor = np.load(
            CACHE / f"train_{protocol}_scores_subject_neighbor_top3_cov_1_8.npy"
        )
        scores = build_task_scores(tasks, full, window, low, low_neighbor, delta_neighbor)
        predictions = scores[selected_indices].argmax(axis=1) + 1
        split_values = range(5) if protocol == "blocked_oof" else [-1]
        for split in split_values:
            validation = (
                (folds == split) & (tasks == "SSVEP")
                if protocol == "blocked_oof"
                else selected & (tasks == "SSVEP")
            )
            fit = ~validation & (tasks == "SSVEP")
            val_indices = np.flatnonzero(validation)
            fit_indices = np.flatnonzero(fit)
            positions = np.searchsorted(selected_indices, val_indices)
            prior = pd.Series(subjects[fit_indices]).value_counts().reindex(
                range(1, 21), fill_value=0
            ).to_numpy()
            predictions[positions] = sinkhorn_labels(
                scores[val_indices], prior, temperature=1.0, iterations=100
            )

        correct = predictions == subjects[selected_indices]
        output = metadata.loc[selected, ["path", "subject", "task", "trial"]].copy()
        output["predicted_label"] = predictions
        output["correct"] = correct
        output.to_csv(CACHE / f"train_{protocol}_task_specific_predictions.csv", index=False)
        np.save(CACHE / f"train_{protocol}_scores_task_specific.npy", scores)
        task_report = {}
        for task in sorted(metadata["task"].unique()):
            task_mask = tasks[selected_indices] == task
            task_report[task] = {
                "n": int(task_mask.sum()),
                "errors": int((~correct[task_mask]).sum()),
                "accuracy": float(correct[task_mask].mean()),
            }
        report[protocol] = {
            "n": len(selected_indices),
            "errors": int((~correct).sum()),
            "accuracy": float(correct.mean()),
            "per_task": task_report,
            "model_selection_note": (
                "Task rules were selected during exploratory analysis of these two protocols; "
                "100% is not an untouched holdout or test-set claim."
            ),
        }
        print(protocol, json.dumps(report[protocol], ensure_ascii=False), flush=True)

    path = ROOT / "research" / "train_task_specific_ensemble.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
