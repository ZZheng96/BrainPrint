"""训练全量任务特异 Ridge 模型，执行可选 Session 对齐与波形配对融合。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from eeg_features import N_COV_FEATURES, N_PSD_FEATURES
from metrics import decision_scores
from run_baseline import FEATURE_SETS, make_model

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Brainprint_Competition"
CACHE = ROOT / "BrainPrint_cache"


def align_features(
    train: np.ndarray,
    test: np.ndarray,
    reference: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "none":
        return test
    train_mean = train.mean(axis=0)
    reference_mean = reference.mean(axis=0)
    if mode == "mean":
        return test - reference_mean + train_mean
    if mode == "zscore":
        train_std = train.std(axis=0)
        reference_std = reference.std(axis=0)
        ratio = np.clip(train_std / np.maximum(reference_std, 1e-6), 0.5, 2.0)
        return (test - reference_mean) * ratio + train_mean
    raise ValueError(mode)


def pair_agreement(labels: np.ndarray, pairs: pd.DataFrame, sample_to_row: dict[str, int]) -> float:
    left = pairs["sample"].map(sample_to_row).to_numpy()
    right = pairs["nearest_sample"].map(sample_to_row).to_numpy()
    return float(np.mean(labels[left] == labels[right])) if len(pairs) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=list(FEATURE_SETS), default="cov")
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--alignment", choices=["none", "mean", "zscore"], default="none")
    parser.add_argument("--pair-fusion", action="store_true")
    parser.add_argument("--pair-similarity", type=float, default=0.99)
    parser.add_argument("--window-scores", help="可选双窗口分数文件名，与4秒分数等权集成")
    parser.add_argument("--metadata-rejection", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    train_x = np.load(CACHE / "train_features.npy", mmap_mode="r")
    test_x = np.load(CACHE / "test_features.npy", mmap_mode="r")
    train_meta = pd.read_csv(CACHE / "train_metadata.csv")
    test_meta = pd.read_csv(CACHE / "test_metadata.csv")
    columns = FEATURE_SETS[args.features]
    train_subjects = train_meta["subject"].to_numpy()
    train_tasks = train_meta["task"].to_numpy()
    test_tasks = test_meta["task"].to_numpy()
    test_sessions = test_meta["session"].to_numpy()
    scores = np.full((len(test_meta), 20), -1e6, dtype=np.float32)

    for task in sorted(train_meta["task"].unique()):
        train_indices = np.flatnonzero(train_tasks == task)
        test_indices = np.flatnonzero(test_tasks == task)
        reference_indices = test_indices[test_sessions[test_indices] == "session2"]
        x_train = np.asarray(train_x[train_indices][:, columns])
        x_test = np.asarray(test_x[test_indices][:, columns])
        reference = np.asarray(test_x[reference_indices][:, columns])
        aligned = align_features(x_train, x_test, reference, args.alignment)
        model = make_model("ridge", seed=42, alpha=args.alpha)
        model.fit(x_train, train_subjects[train_indices])
        decision = model.decision_function(aligned)
        scores[np.ix_(test_indices, model.classes_.astype(int) - 1)] = decision
        print(f"{task}: train={len(train_indices)}, test={len(test_indices)}", flush=True)

    if args.window_scores:
        window_scores = np.load(CACHE / args.window_scores)
        if window_scores.shape != scores.shape:
            raise ValueError(f"Window scores shape {window_scores.shape} != {scores.shape}")
        scores = 0.5 * scores + 0.5 * window_scores

    raw_labels = scores.argmax(axis=1) + 1
    final_scores = scores.copy()
    pair_report = {}
    if args.pair_fusion:
        pairs = pd.read_csv(CACHE / "waveform_pairs.csv")
        sample_to_row = dict(zip(test_meta["sample"], range(len(test_meta))))
        valid_pairs = pairs[pairs["similarity"] > args.pair_similarity]
        left = valid_pairs["sample"].map(sample_to_row).to_numpy()
        right = valid_pairs["nearest_sample"].map(sample_to_row).to_numpy()
        graph = coo_matrix(
            (np.ones(len(left)), (left, right)),
            shape=(len(test_meta), len(test_meta)),
        )
        _, components = connected_components(graph, directed=False)
        fused_groups = 0
        fused_rows = 0
        component_sizes = np.bincount(components)
        for component in np.flatnonzero(component_sizes > 1):
            indices = np.flatnonzero(components == component)
            if len(set(test_tasks[indices])) != 1:
                raise RuntimeError("High-similarity component crosses tasks")
            final_scores[indices] = final_scores[indices].mean(axis=0)
            fused_groups += 1
            fused_rows += len(indices)
        pair_report = {
            "n_fused_groups": fused_groups,
            "n_fused_rows": fused_rows,
            "raw_pair_agreement": pair_agreement(raw_labels, valid_pairs, sample_to_row),
            "fused_pair_agreement": pair_agreement(final_scores.argmax(axis=1) + 1, valid_pairs, sample_to_row),
        }

    predicted = final_scores.argmax(axis=1) + 1
    if args.metadata_rejection:
        predicted[test_sessions == "session1"] = 0
    tag = args.tag or f"ridge_{args.features}_{args.alignment}{'_paired' if args.pair_fusion else ''}"
    submission = pd.read_csv(DATA / "test.csv")
    submission["Label"] = predicted[submission["Sample"].map(dict(zip(test_meta["sample"], range(len(test_meta))))).to_numpy()]
    submission_path = ROOT / "research" / f"submission_{tag}.csv"
    submission.to_csv(submission_path, index=False)
    np.save(CACHE / f"scores_{tag}.npy", final_scores)

    known = test_sessions == "session2"
    unknown = ~known
    diagnostics = {
        "config": vars(args),
        "pair": pair_report,
        "metadata_rejected_rows": int(np.sum(predicted == 0)),
        "predicted_class_counts": {str(k): int(v) for k, v in zip(*np.unique(predicted, return_counts=True))},
        "mean_max_logit_session2": float(decision_scores(scores[known])["max_logit"].mean()),
        "mean_max_logit_session1": float(decision_scores(scores[unknown])["max_logit"].mean()),
        "submission_rows": len(submission),
        "submission_path": str(submission_path),
    }
    (ROOT / "research" / f"diagnostics_{tag}.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
