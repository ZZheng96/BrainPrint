"""训练双2秒窗口模型并对测试Trial输出平均决策分数。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from predict_ridge import align_features
from run_baseline import FEATURE_SETS, make_model

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "BrainPrint_cache"


def main() -> None:
    train_x = np.load(CACHE / "train_window2_features.npy", mmap_mode="r")
    test_x = np.load(CACHE / "test_window2_features.npy", mmap_mode="r")
    columns = FEATURE_SETS["cov"]
    train_meta = pd.read_csv(CACHE / "train_metadata.csv")
    test_meta = pd.read_csv(CACHE / "test_metadata.csv")
    train_parent = np.repeat(np.arange(len(train_meta)), 2)
    test_parent = np.repeat(np.arange(len(test_meta)), 2)
    train_subjects = train_meta["subject"].to_numpy()
    train_tasks = train_meta["task"].to_numpy()
    test_tasks = test_meta["task"].to_numpy()
    test_sessions = test_meta["session"].to_numpy()
    scores = np.full((len(test_meta), 20), -1e6, dtype=np.float32)

    for task in sorted(train_meta["task"].unique()):
        train_parents = np.flatnonzero(train_tasks == task)
        test_parents = np.flatnonzero(test_tasks == task)
        fit_rows = np.isin(train_parent, train_parents)
        test_rows = np.isin(test_parent, test_parents)
        reference_rows = test_rows & (test_sessions[test_parent] == "session2")
        x_train = np.asarray(train_x[fit_rows][:, columns])
        x_test = np.asarray(test_x[test_rows][:, columns])
        reference = np.asarray(test_x[reference_rows][:, columns])
        aligned = align_features(x_train, x_test, reference, "zscore")
        model = make_model("ridge", seed=42, alpha=100.0)
        model.fit(x_train, train_subjects[train_parent[fit_rows]])
        decision = model.decision_function(aligned)
        parent_decision = np.zeros((len(test_parents), 20))
        for class_index in range(20):
            sums = np.bincount(test_parent[test_rows], weights=decision[:, class_index], minlength=len(test_meta))
            parent_decision[:, class_index] = sums[test_parents] / 2
        scores[test_parents] = parent_decision
        print(f"{task}: train windows={fit_rows.sum()} test windows={test_rows.sum()}", flush=True)

    if not np.isfinite(scores).all():
        raise RuntimeError("Non-finite window scores")
    output = CACHE / "scores_window2_zscore.npy"
    np.save(output, scores)
    print(f"saved {scores.shape} -> {output}", flush=True)


if __name__ == "__main__":
    main()
