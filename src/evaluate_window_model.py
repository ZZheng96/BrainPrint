"""按父Trial严格折外评估2秒窗口训练与双窗口分数平均。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_train_errors import summarize_predictions
from run_baseline import FEATURE_SETS, make_model

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "BrainPrint_cache"


def main() -> None:
    x = np.load(CACHE / "train_window2_features.npy", mmap_mode="r")
    columns = FEATURE_SETS["cov"]
    metadata = pd.read_csv(CACHE / "train_metadata.csv")
    n_parents = len(metadata)
    parent = np.repeat(np.arange(n_parents), 2)
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    trials = metadata["trial"].to_numpy()
    folds = ((trials - 1) // 10) % 5
    max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()

    for protocol in ("blocked_oof", "forward"):
        selected = np.ones(n_parents, dtype=bool) if protocol == "blocked_oof" else trials > 0.8 * max_trial
        split_values = range(5) if protocol == "blocked_oof" else [-1]
        scores = np.full((n_parents, 20), np.nan, dtype=np.float32)
        window_scores = np.full((n_parents * 2, 20), np.nan, dtype=np.float32)
        for split in split_values:
            validation = folds == split if protocol == "blocked_oof" else selected
            fit = ~validation
            for task in sorted(metadata["task"].unique()):
                fit_parents = np.flatnonzero(fit & (tasks == task))
                val_parents = np.flatnonzero(validation & (tasks == task))
                fit_rows = np.isin(parent, fit_parents)
                val_rows = np.isin(parent, val_parents)
                model = make_model("ridge", seed=42, alpha=100.0)
                model.fit(x[fit_rows][:, columns], subjects[parent[fit_rows]])
                decision = model.decision_function(x[val_rows][:, columns])
                window_scores[np.ix_(val_rows, model.classes_.astype(int) - 1)] = decision
                parent_decision = np.zeros((len(val_parents), 20))
                for class_index in range(20):
                    sums = np.bincount(
                        parent[val_rows], weights=decision[:, class_index], minlength=n_parents
                    )
                    parent_decision[:, class_index] = sums[val_parents] / 2
                scores[val_parents] = parent_decision
                print(f"{protocol} split={split} {task}: parents={len(fit_parents)}/{len(val_parents)}", flush=True)

        indices = np.flatnonzero(selected)
        labels = scores[indices].argmax(axis=1) + 1
        output = metadata.loc[selected, ["path", "subject", "task", "task_folder", "trial"]].copy()
        output["predicted_label"] = labels
        output["correct"] = labels == subjects[indices]
        output.to_csv(CACHE / f"train_{protocol}_predictions_window2.csv", index=False)
        np.save(CACHE / f"train_{protocol}_scores_window2.npy", scores)
        np.save(CACHE / f"train_{protocol}_window_scores_window2.npy", window_scores)
        report = summarize_predictions(output, protocol)
        path = ROOT / "research" / f"train_error_summary_window2_{protocol}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps({"protocol": protocol, "accuracy": report["accuracy"], "errors": report["errors"], "confusion": report["confusion_pairs"]}, ensure_ascii=False),
            flush=True,
        )


if __name__ == "__main__":
    main()
