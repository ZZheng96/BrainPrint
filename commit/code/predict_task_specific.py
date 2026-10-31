"""用固定任务特异融合生成独立测试候选。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from balanced_sinkhorn import sinkhorn_labels
from evaluate_subject_neighbors import neighbor_scores
from evaluate_task_specific_ensemble import build_task_scores
from predict_ridge import align_features, pair_agreement
from run_baseline import FEATURE_SETS, make_model

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Brainprint_Competition"
CACHE = ROOT / "BrainPrint_cache"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-fusion", action="store_true")
    parser.add_argument("--pair-similarity", type=float, default=0.99)
    parser.add_argument("--balanced-ec-imagery", action="store_true")
    parser.add_argument("--guarded-ec", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.guarded_ec and not args.balanced_ec_imagery:
        parser.error("--guarded-ec requires --balanced-ec-imagery")
    train_x = np.load(CACHE / "train_features.npy", mmap_mode="r")
    test_x = np.load(CACHE / "test_features.npy", mmap_mode="r")
    train_meta = pd.read_csv(CACHE / "train_metadata.csv")
    test_meta = pd.read_csv(CACHE / "test_metadata.csv")
    train_subjects = train_meta["subject"].to_numpy()
    train_tasks = train_meta["task"].to_numpy()
    test_tasks = test_meta["task"].to_numpy()
    test_sessions = test_meta["session"].to_numpy()
    full = np.load(CACHE / "scores_ridge_cov_zscore.npy")
    window = np.load(CACHE / "scores_window2_zscore.npy")
    low = np.full((len(test_meta), 20), np.nan, dtype=np.float32)
    low_neighbor = np.full_like(low, np.nan)
    delta_neighbor = np.full_like(low, np.nan)

    for task in sorted(train_meta["task"].unique()):
        train_indices = np.flatnonzero(train_tasks == task)
        test_indices = np.flatnonzero(test_tasks == task)
        reference_indices = test_indices[test_sessions[test_indices] == "session2"]
        for feature_name in ("cov_1_13", "cov_1_8"):
            columns = FEATURE_SETS[feature_name]
            x_train = np.asarray(train_x[train_indices][:, columns])
            x_test = np.asarray(test_x[test_indices][:, columns])
            reference = np.asarray(test_x[reference_indices][:, columns])
            aligned = align_features(x_train, x_test, reference, "zscore")
            if feature_name == "cov_1_13":
                model = make_model("ridge", seed=42, alpha=100.0)
                model.fit(x_train, train_subjects[train_indices])
                low[test_indices] = model.decision_function(aligned)
                low_neighbor[test_indices] = neighbor_scores(
                    x_train, train_subjects[train_indices], aligned
                )["neighbor_top3"]
            else:
                delta_neighbor[test_indices] = neighbor_scores(
                    x_train, train_subjects[train_indices], aligned
                )["neighbor_top3"]
        print(f"{task}: train={len(train_indices)} test={len(test_indices)}", flush=True)

    scores = build_task_scores(test_tasks, full, window, low, low_neighbor, delta_neighbor)
    fused_groups = 0
    fused_rows = 0
    sample_to_row = dict(zip(test_meta["sample"], range(len(test_meta))))
    if args.pair_fusion:
        pairs = pd.read_csv(CACHE / "waveform_pairs.csv")
        valid_pairs = pairs[pairs["similarity"] > args.pair_similarity]
        left = valid_pairs["sample"].map(sample_to_row).to_numpy()
        right = valid_pairs["nearest_sample"].map(sample_to_row).to_numpy()
        graph = coo_matrix(
            (np.ones(len(left)), (left, right)),
            shape=(len(test_meta), len(test_meta)),
        )
        _, components = connected_components(graph, directed=False)
        for component in np.flatnonzero(np.bincount(components) > 1):
            indices = np.flatnonzero(components == component)
            if len(set(test_tasks[indices])) != 1:
                raise RuntimeError("High-similarity component crosses tasks")
            scores[indices] = scores[indices].mean(axis=0)
            fused_groups += 1
            fused_rows += len(indices)
    direct = scores.argmax(axis=1) + 1
    predicted = direct.copy()
    sinkhorn_tasks = (
        ("EC", "Imagery", "SSVEP") if args.balanced_ec_imagery else ("SSVEP",)
    )
    sinkhorn_agreements = {}
    ssvep_indices = np.flatnonzero(
        (test_tasks == "SSVEP") & (test_sessions == "session2")
    )
    for task in sinkhorn_tasks:
        indices = np.flatnonzero((test_tasks == task) & (test_sessions == "session2"))
        prior = train_meta.loc[train_meta["task"] == task, "subject"].value_counts().reindex(
            range(1, 21), fill_value=0
        ).to_numpy()
        balanced = sinkhorn_labels(scores[indices], prior, temperature=1.0, iterations=100)
        if task == "EC" and args.guarded_ec:
            top_two = np.partition(scores[indices], -2, axis=1)[:, -2:]
            low_confidence = (top_two[:, 1] - top_two[:, 0]) < 1.0
            balanced = np.where(low_confidence, balanced, direct[indices])
        predicted[indices] = balanced
        sinkhorn_agreements[task] = float(np.mean(balanced == direct[indices]))
    predicted[test_sessions == "session1"] = 0

    tag = args.tag or (
        f"task_specific_zscore{'_grouped' if args.pair_fusion else ''}"
        f"{'_guarded' if args.guarded_ec else ('_balanced' if args.balanced_ec_imagery else '')}"
    )
    np.save(CACHE / f"scores_{tag}.npy", scores)
    np.save(CACHE / "scores_task_low13_ridge_zscore.npy", low)
    np.save(CACHE / "scores_task_low13_neighbor_zscore.npy", low_neighbor)
    np.save(CACHE / "scores_task_delta_neighbor_zscore.npy", delta_neighbor)
    submission = pd.read_csv(DATA / "test.csv")
    submission["Label"] = predicted[submission["Sample"].map(sample_to_row).to_numpy()]
    submission_path = ROOT / "research" / f"submission_{tag}.csv"
    submission.to_csv(submission_path, index=False)

    pairs = pd.read_csv(CACHE / "waveform_pairs.csv")
    valid_pairs = pairs[pairs["similarity"] > args.pair_similarity]
    previous_path = ROOT / "research" / "submission_sinkhorn_window_cov_zscore_grouped.csv"
    previous = pd.read_csv(previous_path)
    previous_labels = previous.set_index("Sample").loc[submission["Sample"], "Label"].to_numpy()
    diagnostics = {
        "config": {
            "EC": (
                "row-z base + low-frequency neighbors; guarded Sinkhorn T=1 (margin<1)"
                if args.guarded_ec
                else "row-z base + low-frequency neighbors, direct"
            ),
            "EO": "base + 0.25 1-13Hz Ridge, direct",
            "Imagery": (
                "two-window Ridge, Sinkhorn T=1"
                if args.balanced_ec_imagery
                else "two-window Ridge, direct"
            ),
            "SSVEP": "base + 0.5 1-13Hz Ridge, Sinkhorn T=1",
            "alignment": "session2 zscore",
            "pair_fusion": args.pair_fusion,
            "pair_similarity": args.pair_similarity if args.pair_fusion else None,
            "metadata_rejection": "test session1 -> 0",
        },
        "fused_groups": fused_groups,
        "fused_rows": fused_rows,
        "submission_rows": len(submission),
        "sample_order_matches": bool(submission["Sample"].equals(pd.read_csv(DATA / "test.csv")["Sample"])),
        "labels_valid": bool(submission["Label"].between(0, 20).all() and submission["Label"].notna().all()),
        "predicted_class_counts": {
            str(k): int(v) for k, v in zip(*np.unique(predicted, return_counts=True))
        },
        "sinkhorn_agreement_by_task": sinkhorn_agreements,
        "ssvep_sinkhorn_agreement": float(
            np.mean(predicted[ssvep_indices] == direct[ssvep_indices])
        ),
        "high_similarity_pair_agreement": pair_agreement(
            predicted, valid_pairs, sample_to_row
        ),
        "previous_candidate_changed_rows": int(np.sum(submission["Label"].to_numpy() != previous_labels)),
        "previous_candidate_agreement": float(np.mean(submission["Label"].to_numpy() == previous_labels)),
        "submission_path": str(submission_path),
        "test_accuracy_claim": "unknown; no test ground truth or official score",
    }
    (ROOT / "research" / f"diagnostics_{tag}.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
