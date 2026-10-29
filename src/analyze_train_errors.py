"""生成全部训练被试的折外预测与逐被试/逐 Trial 错误明细。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_features import normalize_log_cov_determinant
from metrics import decision_scores
from predict_ridge import align_features
from run_baseline import FEATURE_SETS, make_model

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "BrainPrint_cache"


def summarize_predictions(predictions: pd.DataFrame, protocol: str) -> dict:
    subject_rows = []
    for subject, group in predictions.groupby("subject"):
        row = {
            "subject": int(subject),
            "n": len(group),
            "errors": int((~group["correct"]).sum()),
            "accuracy": float(group["correct"].mean()),
        }
        for task in sorted(group["task"].unique()):
            task_group = group[group["task"] == task]
            row[f"{task}_n"] = len(task_group)
            row[f"{task}_errors"] = int((~task_group["correct"]).sum())
        subject_rows.append(row)
    errors = predictions[~predictions["correct"]]
    return {
        "protocol": protocol,
        "n": len(predictions),
        "errors": len(errors),
        "accuracy": float(predictions["correct"].mean()),
        "per_subject": subject_rows,
        "confusion_pairs": [
            {"true": int(true), "predicted": int(predicted), "count": int(count)}
            for (true, predicted), count in errors.groupby(["subject", "predicted_label"]).size().items()
        ],
        "per_task": {
            task: {
                "n": len(group),
                "errors": int((~group["correct"]).sum()),
                "accuracy": float(group["correct"].mean()),
            }
            for task, group in predictions.groupby("task")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ridge", "logistic"], default="ridge")
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--features", choices=list(FEATURE_SETS), default="cov")
    parser.add_argument("--det-normalize", action="store_true")
    parser.add_argument("--alignment", choices=["none", "mean", "zscore"], default="none")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    x = np.load(CACHE / "train_features.npy", mmap_mode="r")
    metadata = pd.read_csv(CACHE / "train_metadata.csv")
    columns = FEATURE_SETS[args.features]
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    trials = metadata["trial"].to_numpy()
    folds = ((trials - 1) // 10) % 5
    max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    if args.det_normalize and args.features != "cov":
        raise ValueError("--det-normalize currently requires --features cov")

    def select_rows(indices: np.ndarray) -> np.ndarray:
        values = np.asarray(x[indices][:, columns])
        return normalize_log_cov_determinant(values) if args.det_normalize else values

    configurations = {
        "blocked_oof": np.zeros(len(metadata), dtype=bool),
        "forward": trials > 0.8 * max_trial,
    }
    all_reports = {}
    model_prefix = "" if args.model == "ridge" else f"{args.model}_"
    alignment_suffix = "" if args.alignment == "none" else f"_{args.alignment}"
    tag = args.tag or f"{model_prefix}{args.features}_alpha{args.alpha:g}{'_detnorm' if args.det_normalize else ''}{alignment_suffix}"
    for protocol, selection in configurations.items():
        scores = np.full((len(metadata), 20), np.nan, dtype=np.float32)
        if protocol == "blocked_oof":
            split_values = range(5)
            selection_for_fold = lambda fold: folds == fold
        else:
            split_values = [0]
            selection_for_fold = lambda _: selection

        for split_value in split_values:
            validation = selection_for_fold(split_value)
            fit = ~validation
            for task in sorted(metadata["task"].unique()):
                fit_indices = np.flatnonzero(fit & (tasks == task))
                val_indices = np.flatnonzero(validation & (tasks == task))
                model = make_model(args.model, seed=42, alpha=args.alpha)
                x_train = select_rows(fit_indices)
                x_validation = select_rows(val_indices)
                x_validation = align_features(x_train, x_validation, x_validation, args.alignment)
                model.fit(x_train, subjects[fit_indices])
                decision = model.decision_function(x_validation)
                scores[np.ix_(val_indices, model.classes_.astype(int) - 1)] = decision
                print(
                    f"{protocol} split={split_value} {task}: fit={len(fit_indices)} val={len(val_indices)}",
                    flush=True,
                )

        selected = selection if protocol == "forward" else np.ones(len(metadata), dtype=bool)
        selected_indices = np.flatnonzero(selected)
        selected_scores = scores[selected_indices]
        labels = selected_scores.argmax(axis=1) + 1
        score_features = decision_scores(selected_scores)
        output = metadata.loc[
            selected,
            ["path", "subject", "task", "task_folder", "trial", "raw_std_mean", "filtered_std_mean", "clip_fraction_mean", "clip_fraction_max", "nonfinite_fraction"],
        ].copy()
        output["protocol"] = protocol
        output["fold"] = folds[selected_indices] if protocol == "blocked_oof" else -1
        output["predicted_label"] = labels
        output["correct"] = labels == subjects[selected_indices]
        output["max_logit"] = score_features["max_logit"]
        output["margin"] = score_features["margin"]
        output["energy"] = score_features["energy"]
        output["true_score"] = selected_scores[np.arange(len(selected_indices)), subjects[selected_indices] - 1]
        output["predicted_score"] = selected_scores[np.arange(len(selected_indices)), labels - 1]
        output["score_gap"] = output["predicted_score"] - output["true_score"]
        output["trial_position"] = trials[selected_indices] / max_trial[selected_indices]
        output.to_csv(CACHE / f"train_{protocol}_predictions_{tag}.csv", index=False)
        np.save(CACHE / f"train_{protocol}_scores_{tag}.npy", scores)
        report = summarize_predictions(output, protocol)
        report["config"] = vars(args)
        all_reports[protocol] = report
        print(
            f"{protocol}: accuracy={report['accuracy']:.6f}, errors={report['errors']}",
            flush=True,
        )

    output_path = ROOT / "research" / f"train_error_summary_{tag}.json"
    output_path.write_text(json.dumps(all_reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary: {output_path}", flush=True)


if __name__ == "__main__":
    main()
