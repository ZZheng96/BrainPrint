"""在训练集前推保留数据上验证类别均衡 Sinkhorn。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from balanced_sinkhorn import sinkhorn_labels
from run_baseline import FEATURE_SETS, make_model

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "BrainPrint_cache"


def main() -> None:
    x = np.load(CACHE / "train_features.npy", mmap_mode="r")
    metadata = pd.read_csv(CACHE / "train_metadata.csv")
    columns = FEATURE_SETS["cov"]
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    trials = metadata["trial"].to_numpy()
    max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    known = subjects < 17
    validation = known & (trials > 0.8 * max_trial)
    fit = known & (trials <= 0.8 * max_trial)
    temperatures = (0.1, 0.5, 1.0)
    predictions = {temperature: np.zeros(validation.sum(), dtype=int) for temperature in temperatures}
    direct_all = np.zeros(validation.sum(), dtype=int)
    true_all = subjects[validation]
    val_indices = np.flatnonzero(validation)
    position = np.full(len(metadata), -1)
    position[val_indices] = np.arange(len(val_indices))
    task_reports = {}

    for task in sorted(metadata["task"].unique()):
        fit_indices = np.flatnonzero(fit & (tasks == task))
        indices = np.flatnonzero(validation & (tasks == task))
        model = make_model("ridge", seed=42)
        model.fit(x[fit_indices][:, columns], subjects[fit_indices])
        scores = model.decision_function(x[indices][:, columns])
        direct = model.classes_[scores.argmax(axis=1)]
        positions = position[indices]
        direct_all[positions] = direct
        prior = pd.Series(subjects[fit_indices]).value_counts().reindex(range(1, 17), fill_value=0).to_numpy()
        task_reports[task] = {"direct": float(np.mean(direct == subjects[indices]))}
        for temperature in temperatures:
            labels = sinkhorn_labels(scores, prior, temperature, 100)
            labels = model.classes_[labels - 1]
            predictions[temperature][positions] = labels
            task_reports[task][str(temperature)] = float(np.mean(labels == subjects[indices]))
        print(task, task_reports[task], flush=True)

    report = {
        "protocol": "subjects 1-16; fit first 80%, validate last 20% per subject/task",
        "direct_accuracy": float(np.mean(direct_all == true_all)),
        "sinkhorn_accuracy": {
            str(temperature): float(np.mean(predictions[temperature] == true_all))
            for temperature in temperatures
        },
        "tasks": task_reports,
    }
    output = ROOT / "research" / "validation_sinkhorn.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
