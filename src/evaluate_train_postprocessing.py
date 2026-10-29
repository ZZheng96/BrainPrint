"""在全部20名训练被试上公平比较折外 Sinkhorn 后处理。"""
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
    temperatures = (0.5, 1.0, 2.0)
    report = {}

    for protocol in ("blocked_oof", "forward"):
        scores = np.load(CACHE / f"train_{protocol}_scores_cov_alpha100.npy", mmap_mode="r")
        selected = np.ones(len(metadata), dtype=bool) if protocol == "blocked_oof" else trials > 0.8 * max_trial
        selected_indices = np.flatnonzero(selected)
        direct = np.asarray(scores[selected_indices]).argmax(axis=1) + 1
        protocol_report = {
            "n": len(selected_indices),
            "direct_errors": int(np.sum(direct != subjects[selected_indices])),
            "direct_accuracy": float(np.mean(direct == subjects[selected_indices])),
        }
        for temperature in temperatures:
            predictions = direct.copy()
            split_values = range(5) if protocol == "blocked_oof" else [-1]
            for split in split_values:
                validation = folds == split if protocol == "blocked_oof" else selected
                fit = ~validation
                for task in sorted(metadata["task"].unique()):
                    val_indices = np.flatnonzero(validation & (tasks == task))
                    fit_indices = np.flatnonzero(fit & (tasks == task))
                    positions = np.searchsorted(selected_indices, val_indices)
                    prior = pd.Series(subjects[fit_indices]).value_counts().reindex(range(1, 21), fill_value=0).to_numpy()
                    predictions[positions] = sinkhorn_labels(
                        np.asarray(scores[val_indices]), prior, temperature=temperature, iterations=100
                    )
            correct = predictions == subjects[selected_indices]
            errors = selected_indices[~correct]
            output = metadata.loc[selected, ["path", "subject", "task", "trial"]].copy()
            output["predicted_label"] = predictions
            output["correct"] = correct
            output.to_csv(CACHE / f"train_{protocol}_sinkhorn_{temperature:g}_predictions.csv", index=False)
            protocol_report[f"T{temperature:g}"] = {
                "errors": len(errors),
                "accuracy": float(correct.mean()),
                "error_subjects": {
                    str(int(subject)): int(count)
                    for subject, count in pd.Series(subjects[errors]).value_counts().sort_index().items()
                },
                "confusion_pairs": [
                    {"true": int(subjects[index]), "predicted": int(predictions[np.flatnonzero(selected_indices == index)[0]])}
                    for index in errors
                ],
            }
            print(protocol, temperature, protocol_report[f"T{temperature:g}"], flush=True)
        report[protocol] = protocol_report

    output_path = ROOT / "research" / "train_postprocessing_comparison.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
