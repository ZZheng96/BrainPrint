"""测试 Session 内无监督聚类，并将簇与训练身份进行一对一匹配。"""
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

from run_baseline import FEATURE_SETS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Brainprint_Competition"
CACHE = ROOT.parent / "BrainPrint_cache"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=list(FEATURE_SETS), default="cov")
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--whiten", action="store_true")
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--scores", default="scores_ridge_cov_none.npy")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    test_x = np.load(CACHE / "test_features.npy", mmap_mode="r")
    test_meta = pd.read_csv(CACHE / "test_metadata.csv")
    direct_scores = np.load(CACHE / args.scores, mmap_mode="r")
    columns = FEATURE_SETS[args.features]
    sessions = test_meta["session"].to_numpy()
    tasks = test_meta["task"].to_numpy()
    predicted = np.zeros(len(test_meta), dtype=int)
    direct_labels = direct_scores.argmax(axis=1) + 1
    task_reports = {}

    for task in sorted(test_meta["task"].unique()):
        indices = np.flatnonzero((tasks == task) & (sessions == "session2"))
        x = np.asarray(test_x[indices][:, columns])
        x = StandardScaler().fit_transform(x)
        n_components = min(args.components, len(indices) - 1, x.shape[1])
        embedding = PCA(
            n_components=n_components,
            whiten=args.whiten,
            svd_solver="randomized",
            random_state=42,
        ).fit_transform(x)
        cluster = KMeans(n_clusters=20, n_init=args.n_init, random_state=42).fit_predict(embedding)
        score_matrix = np.zeros((20, 20), dtype=np.float64)
        for cluster_id in range(20):
            cluster_scores = np.asarray(direct_scores[indices[cluster == cluster_id]])
            score_matrix[cluster_id] = cluster_scores.mean(axis=0)
        score_matrix = (score_matrix - score_matrix.mean(axis=0)) / np.maximum(score_matrix.std(axis=0), 1e-6)
        cluster_rows, subject_columns = linear_sum_assignment(score_matrix, maximize=True)
        mapping = dict(zip(cluster_rows, subject_columns + 1))
        predicted[indices] = np.asarray([mapping[c] for c in cluster])
        sizes = np.bincount(cluster, minlength=20)
        direct_purity = []
        for cluster_id in range(20):
            labels = direct_labels[indices[cluster == cluster_id]]
            direct_purity.append(float(np.mean(labels == np.bincount(labels, minlength=21).argmax())))
        task_reports[task] = {
            "n": len(indices),
            "cluster_size_min": int(sizes.min()),
            "cluster_size_max": int(sizes.max()),
            "cluster_sizes": sizes.tolist(),
            "cluster_to_subject": {str(k + 1): int(v) for k, v in mapping.items()},
            "direct_majority_purity_mean": float(np.mean(direct_purity)),
            "direct_label_agreement": float(np.mean(predicted[indices] == direct_labels[indices])),
        }
        print(f"{task}: direct agreement={task_reports[task]['direct_label_agreement']:.4f}", flush=True)

    tag = args.tag or f"cluster_{args.features}_pca{args.components}{'_whiten' if args.whiten else ''}"
    submission = pd.read_csv(DATA / "test.csv")
    sample_to_row = dict(zip(test_meta["sample"], range(len(test_meta))))
    submission["Label"] = predicted[submission["Sample"].map(sample_to_row).to_numpy()]
    submission_path = ROOT / "research" / f"submission_{tag}.csv"
    submission.to_csv(submission_path, index=False)

    pair_report = {}
    pair_path = CACHE / "waveform_pairs.csv"
    if pair_path.exists():
        pairs = pd.read_csv(pair_path)
        valid = pairs[pairs["reciprocal"] & (pairs["similarity"] > 0.99)]
        left = valid["sample"].map(sample_to_row).to_numpy()
        right = valid["nearest_sample"].map(sample_to_row).to_numpy()
        pair_report = {
            "high_similarity_pair_agreement": float(np.mean(predicted[left] == predicted[right])),
            "direct_high_similarity_pair_agreement": float(np.mean(direct_labels[left] == direct_labels[right])),
        }
    report = {
        "config": vars(args),
        "tasks": task_reports,
        "pair": pair_report,
        "predicted_class_counts": {str(k): int(v) for k, v in zip(*np.unique(predicted, return_counts=True))},
        "submission_path": str(submission_path),
    }
    (ROOT / "research" / f"diagnostics_{tag}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
