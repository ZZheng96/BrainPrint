"""用首20%反向或中间20%留段确认任务特异融合。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_subject_neighbors import neighbor_scores
from evaluate_task_specific_ensemble import (
    build_task_scores,
    predict_with_task_postprocessing,
)
from run_baseline import FEATURE_SETS, make_model

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "BrainPrint_cache"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", choices=["reverse", "middle"], default="reverse")
    parser.add_argument("--balanced-ec-imagery", action="store_true")
    parser.add_argument("--guarded-ec", action="store_true")
    args = parser.parse_args()
    if args.guarded_ec and not args.balanced_ec_imagery:
        parser.error("--guarded-ec requires --balanced-ec-imagery")
    x = np.load(CACHE / "train_features.npy", mmap_mode="r")
    window_x = np.load(CACHE / "train_window2_features.npy", mmap_mode="r")
    metadata = pd.read_csv(CACHE / "train_metadata.csv")
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    trials = metadata["trial"].to_numpy()
    folds = ((trials - 1) // 10) % 5
    max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    selected = (
        trials <= 0.2 * max_trial
        if args.holdout == "reverse"
        else (trials > 0.4 * max_trial) & (trials <= 0.6 * max_trial)
    )
    fit = ~selected
    selected_indices = np.flatnonzero(selected)
    parent = np.repeat(np.arange(len(metadata)), 2)
    score_sets = {
        name: np.full((len(metadata), 20), np.nan, dtype=np.float32)
        for name in ("full", "window", "low", "low_neighbor", "delta_neighbor")
    }

    for task in sorted(metadata["task"].unique()):
        fit_indices = np.flatnonzero(fit & (tasks == task))
        val_indices = np.flatnonzero(selected & (tasks == task))
        for name, feature_name in (("full", "cov"), ("low", "cov_1_13")):
            model = make_model("ridge", seed=42, alpha=100.0)
            model.fit(x[fit_indices][:, FEATURE_SETS[feature_name]], subjects[fit_indices])
            score_sets[name][val_indices] = model.decision_function(
                x[val_indices][:, FEATURE_SETS[feature_name]]
            )
        fit_rows = np.isin(parent, fit_indices)
        val_rows = np.isin(parent, val_indices)
        window_model = make_model("ridge", seed=42, alpha=100.0)
        window_model.fit(
            window_x[fit_rows][:, FEATURE_SETS["cov"]], subjects[parent[fit_rows]]
        )
        window_decision = window_model.decision_function(
            window_x[val_rows][:, FEATURE_SETS["cov"]]
        )
        window_parent_scores = np.zeros((len(val_indices), 20))
        for class_index in range(20):
            sums = np.bincount(
                parent[val_rows], weights=window_decision[:, class_index], minlength=len(metadata)
            )
            window_parent_scores[:, class_index] = sums[val_indices] / 2
        score_sets["window"][val_indices] = window_parent_scores

        for name, feature_name in (
            ("low_neighbor", "cov_1_13"),
            ("delta_neighbor", "cov_1_8"),
        ):
            score_sets[name][val_indices] = neighbor_scores(
                np.asarray(x[fit_indices][:, FEATURE_SETS[feature_name]]),
                subjects[fit_indices],
                np.asarray(x[val_indices][:, FEATURE_SETS[feature_name]]),
            )["neighbor_top3"]
        print(
            f"{args.holdout} {task}: fit={len(fit_indices)} val={len(val_indices)}",
            flush=True,
        )

    scores = build_task_scores(tasks, **score_sets)
    predictions = predict_with_task_postprocessing(
        scores,
        subjects,
        tasks,
        folds,
        selected,
        args.holdout,
        args.balanced_ec_imagery,
        args.guarded_ec,
    )
    correct = predictions == subjects[selected_indices]
    errors = selected_indices[~correct]
    holdout_description = (
        "first 20% validation, last 80% training"
        if args.holdout == "reverse"
        else "middle 40%-60% validation, remaining trials training"
    )
    report = {
        "protocol": f"{args.holdout} holdout: {holdout_description}",
        "n": len(selected_indices),
        "errors": len(errors),
        "accuracy": float(correct.mean()),
        "postprocessing": (
            "EC guarded Sinkhorn (margin<1); Imagery/SSVEP Sinkhorn T=1; EO direct"
            if args.balanced_ec_imagery and args.guarded_ec
            else (
                "EC/Imagery/SSVEP Sinkhorn T=1; EO direct"
                if args.balanced_ec_imagery
                else "SSVEP Sinkhorn T=1; other tasks direct"
            )
        ),
        "confusion": [
            [
                int(subjects[index]),
                int(predictions[np.searchsorted(selected_indices, index)]),
                metadata.at[index, "task"],
                int(trials[index]),
            ]
            for index in errors
        ],
        "per_task": {},
    }
    for task in sorted(metadata["task"].unique()):
        task_mask = tasks[selected_indices] == task
        report["per_task"][task] = {
            "n": int(task_mask.sum()),
            "errors": int((~correct[task_mask]).sum()),
            "accuracy": float(correct[task_mask].mean()),
        }
    suffix = (
        "_guarded"
        if args.balanced_ec_imagery and args.guarded_ec
        else ("_balanced" if args.balanced_ec_imagery else "")
    )
    output = metadata.loc[selected, ["path", "subject", "task", "trial"]].copy()
    output["predicted_label"] = predictions
    output["correct"] = correct
    output.to_csv(
        CACHE / f"train_{args.holdout}_task_specific{suffix}_predictions.csv", index=False
    )
    for name, values in score_sets.items():
        np.save(CACHE / f"train_{args.holdout}_scores_task_{name}.npy", values)
    path = ROOT / "research" / f"train_task_specific_{args.holdout}{suffix}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
