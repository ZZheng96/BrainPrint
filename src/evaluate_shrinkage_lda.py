"""评估任务特异 StandardScaler + Ledoit-Wolf shrinkage LDA。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from balanced_sinkhorn import sinkhorn_labels
from run_baseline import FEATURE_SETS

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "BrainPrint_cache"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=list(FEATURE_SETS), default="cov")
    args = parser.parse_args()
    x = np.load(CACHE / "train_features.npy", mmap_mode="r")
    columns = FEATURE_SETS[args.features]
    metadata = pd.read_csv(CACHE / "train_metadata.csv")
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    trials = metadata["trial"].to_numpy()
    folds = ((trials - 1) // 10) % 5
    max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    report: dict[str, dict] = {}

    for protocol in ("blocked_oof", "forward"):
        selected = (
            np.ones(len(metadata), dtype=bool)
            if protocol == "blocked_oof"
            else trials > 0.8 * max_trial
        )
        selected_indices = np.flatnonzero(selected)
        split_values = range(5) if protocol == "blocked_oof" else [-1]
        scores = np.full((len(metadata), 20), np.nan, dtype=np.float32)
        for split in split_values:
            validation = folds == split if protocol == "blocked_oof" else selected
            fit = ~validation
            for task in sorted(metadata["task"].unique()):
                fit_indices = np.flatnonzero(fit & (tasks == task))
                val_indices = np.flatnonzero(validation & (tasks == task))
                model = make_pipeline(
                    StandardScaler(),
                    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                )
                model.fit(x[fit_indices][:, columns], subjects[fit_indices])
                scores[val_indices] = model.decision_function(x[val_indices][:, columns])
                print(
                    f"{protocol} split={split} {task}: fit={len(fit_indices)} val={len(val_indices)}",
                    flush=True,
                )

        np.save(CACHE / f"train_{protocol}_scores_lda_auto_{args.features}.npy", scores)
        protocol_report = {}
        for use_sinkhorn in (False, True):
            key = "lda_auto" + ("_sinkhorn" if use_sinkhorn else "")
            predictions = scores[selected_indices].argmax(axis=1) + 1
            if use_sinkhorn:
                for split in split_values:
                    validation = folds == split if protocol == "blocked_oof" else selected
                    fit = ~validation
                    for task in sorted(metadata["task"].unique()):
                        val_indices = np.flatnonzero(validation & (tasks == task))
                        fit_indices = np.flatnonzero(fit & (tasks == task))
                        positions = np.searchsorted(selected_indices, val_indices)
                        prior = pd.Series(subjects[fit_indices]).value_counts().reindex(
                            range(1, 21), fill_value=0
                        ).to_numpy()
                        predictions[positions] = sinkhorn_labels(
                            scores[val_indices], prior, temperature=1.0, iterations=100
                        )
            correct = predictions == subjects[selected_indices]
            errors = selected_indices[~correct]
            protocol_report[key] = {
                "errors": len(errors),
                "accuracy": float(correct.mean()),
                "confusion": [
                    [
                        int(subjects[index]),
                        int(predictions[np.searchsorted(selected_indices, index)]),
                        metadata.at[index, "task"],
                        int(trials[index]),
                    ]
                    for index in errors
                ],
            }
            print(protocol, key, protocol_report[key], flush=True)
        report[protocol] = protocol_report

    output = ROOT / "research" / f"train_shrinkage_lda_{args.features}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
