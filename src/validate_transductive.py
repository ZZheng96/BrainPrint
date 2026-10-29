"""在训练集前推保留数据上验证测试聚类 + 匈牙利身份映射流程。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from run_baseline import FEATURE_SETS, make_model

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "BrainPrint_cache"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=list(FEATURE_SETS), default="cov")
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--whiten", action="store_true")
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--unknown-start", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    x = np.load(CACHE / "train_features.npy", mmap_mode="r")
    metadata = pd.read_csv(CACHE / "train_metadata.csv")
    columns = FEATURE_SETS[args.features]
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    trials = metadata["trial"].to_numpy()
    max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    known = subjects < args.unknown_start
    validation = known & (trials > 0.8 * max_trial)
    fit = known & (trials <= 0.8 * max_trial)
    n_classes = args.unknown_start - 1
    direct_all = np.zeros(validation.sum(), dtype=int)
    cluster_all = np.zeros(validation.sum(), dtype=int)
    true_all = subjects[validation]
    validation_indices = np.flatnonzero(validation)
    position = np.full(len(metadata), -1)
    position[validation_indices] = np.arange(len(validation_indices))
    task_reports = {}

    for task in sorted(metadata["task"].unique()):
        fit_indices = np.flatnonzero(fit & (tasks == task))
        val_indices = np.flatnonzero(validation & (tasks == task))
        model = make_model("ridge", seed=42)
        model.fit(x[fit_indices][:, columns], subjects[fit_indices])
        scores = model.decision_function(x[val_indices][:, columns])
        direct = model.classes_[scores.argmax(axis=1)]
        target = StandardScaler().fit_transform(x[val_indices][:, columns])
        embedding = PCA(
            n_components=min(args.components, len(val_indices) - 1),
            whiten=args.whiten,
            svd_solver="randomized",
            random_state=42,
        ).fit_transform(target)
        cluster = KMeans(n_clusters=n_classes, n_init=args.n_init, random_state=42).fit_predict(embedding)
        score_matrix = np.zeros((n_classes, n_classes))
        for cluster_id in range(n_classes):
            score_matrix[cluster_id] = scores[cluster == cluster_id].mean(axis=0)
        score_matrix = (score_matrix - score_matrix.mean(axis=0)) / np.maximum(score_matrix.std(axis=0), 1e-6)
        rows, cols = linear_sum_assignment(score_matrix, maximize=True)
        mapping = dict(zip(rows, model.classes_[cols]))
        clustered = np.asarray([mapping[c] for c in cluster])
        positions = position[val_indices]
        direct_all[positions] = direct
        cluster_all[positions] = clustered
        task_reports[task] = {
            "n": len(val_indices),
            "direct_accuracy": float(np.mean(direct == subjects[val_indices])),
            "cluster_accuracy": float(np.mean(clustered == subjects[val_indices])),
            "cluster_sizes": np.bincount(cluster, minlength=n_classes).tolist(),
        }
        print(task, task_reports[task], flush=True)

    report = {
        "config": vars(args),
        "protocol": "subjects 1-16; fit first 80%, transductive cluster last 20% per subject/task",
        "direct_accuracy": float(np.mean(direct_all == true_all)),
        "cluster_accuracy": float(np.mean(cluster_all == true_all)),
        "tasks": task_reports,
    }
    output = args.output or ROOT / "research" / f"validation_cluster_{args.features}_{args.components}{'_whiten' if args.whiten else ''}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
