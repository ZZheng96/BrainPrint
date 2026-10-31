"""固定任务特异融合规则的双协议折外验证。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .balanced_sinkhorn import sinkhorn_labels
except ImportError:  # 直接执行 src 内脚本
    from balanced_sinkhorn import sinkhorn_labels

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "BrainPrint_cache"


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


def predict_with_task_postprocessing(
    scores: np.ndarray,
    subjects: np.ndarray,
    tasks: np.ndarray,
    folds: np.ndarray,
    selected: np.ndarray,
    protocol: str,
    balanced_ec_imagery: bool,
    guarded_ec: bool = False,
) -> np.ndarray:
    """逐训练折类别先验做转导分配，不读取验证身份标签。"""
    selected_indices = np.flatnonzero(selected)
    predictions = scores[selected_indices].argmax(axis=1) + 1
    split_values = range(5) if protocol == "blocked_oof" else [-1]
    sinkhorn_tasks = ("EC", "Imagery", "SSVEP") if balanced_ec_imagery else ("SSVEP",)
    for split in split_values:
        validation = folds == split if protocol == "blocked_oof" else selected
        fit = ~validation
        for task in sinkhorn_tasks:
            val_indices = np.flatnonzero(validation & (tasks == task))
            fit_indices = np.flatnonzero(fit & (tasks == task))
            positions = np.searchsorted(selected_indices, val_indices)
            prior = pd.Series(subjects[fit_indices]).value_counts().reindex(
                range(1, 21), fill_value=0
            ).to_numpy()
            balanced = sinkhorn_labels(
                scores[val_indices], prior, temperature=1.0, iterations=100
            )
            if task == "EC" and guarded_ec:
                direct_scores = scores[val_indices]
                top_two = np.partition(direct_scores, -2, axis=1)[:, -2:]
                low_confidence = (top_two[:, 1] - top_two[:, 0]) < 1.0
                balanced = np.where(
                    low_confidence, balanced, direct_scores.argmax(axis=1) + 1
                )
            predictions[positions] = balanced
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced-ec-imagery", action="store_true")
    parser.add_argument("--guarded-ec", action="store_true")
    args = parser.parse_args()
    if args.guarded_ec and not args.balanced_ec_imagery:
        parser.error("--guarded-ec requires --balanced-ec-imagery")
    metadata = pd.read_csv(CACHE / "train_metadata.csv")
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    trials = metadata["trial"].to_numpy()
    folds = ((trials - 1) // 10) % 5
    max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    report = {}
    suffix = (
        "_guarded"
        if args.balanced_ec_imagery and args.guarded_ec
        else ("_balanced" if args.balanced_ec_imagery else "")
    )

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
        predictions = predict_with_task_postprocessing(
            scores,
            subjects,
            tasks,
            folds,
            selected,
            protocol,
            args.balanced_ec_imagery,
            args.guarded_ec,
        )
        correct = predictions == subjects[selected_indices]
        output = metadata.loc[selected, ["path", "subject", "task", "trial"]].copy()
        output["predicted_label"] = predictions
        output["correct"] = correct
        output.to_csv(
            CACHE / f"train_{protocol}_task_specific{suffix}_predictions.csv", index=False
        )
        np.save(CACHE / f"train_{protocol}_scores_task_specific{suffix}.npy", scores)
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
            "postprocessing": (
                "EC guarded Sinkhorn (margin<1); Imagery/SSVEP Sinkhorn T=1; EO direct"
                if args.balanced_ec_imagery and args.guarded_ec
                else (
                    "EC/Imagery/SSVEP Sinkhorn T=1; EO direct"
                    if args.balanced_ec_imagery
                    else "SSVEP Sinkhorn T=1; other tasks direct"
                )
            ),
            "model_selection_note": (
                "Task rules and postprocessing were selected during exploratory analysis; "
                "100% is not an untouched holdout or test-set claim."
            ),
        }
        print(protocol, json.dumps(report[protocol], ensure_ascii=False), flush=True)

    path = ROOT / "research" / f"train_task_specific_ensemble{suffix}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
