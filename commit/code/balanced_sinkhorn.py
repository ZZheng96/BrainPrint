"""用带类别先验的 Sinkhorn 最优传输修正测试 Session 的类别偏移。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Brainprint_Competition"
CACHE = ROOT.parent / "BrainPrint_cache"


def sinkhorn_labels(scores: np.ndarray, prior: np.ndarray, temperature: float, iterations: int) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if np.any(~np.isfinite(prior)) or np.any(prior < 0) or prior.sum() <= 0:
        raise ValueError("prior must be finite, nonnegative, and have positive mass")
    scores = (scores - scores.mean(axis=0)) / np.maximum(scores.std(axis=0), 1e-6)
    log_kernel = scores / temperature
    log_row = -np.log(len(scores))
    log_col = np.log(prior / prior.sum())
    u = np.zeros(len(scores))
    v = np.zeros(len(prior))
    for _ in range(iterations):
        u = log_row - logsumexp(log_kernel + v[None, :], axis=1)
        v = log_col - logsumexp(log_kernel + u[:, None], axis=0)
    transport = log_kernel + u[:, None] + v[None, :]
    return transport.argmax(axis=1) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="scores_ridge_cov_zscore.npy")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    scores = np.load(CACHE / args.scores, mmap_mode="r")
    train_meta = pd.read_csv(CACHE / "train_metadata.csv")
    test_meta = pd.read_csv(CACHE / "test_metadata.csv")
    sessions = test_meta["session"].to_numpy()
    tasks = test_meta["task"].to_numpy()
    predicted = np.zeros(len(test_meta), dtype=int)
    direct = scores.argmax(axis=1) + 1
    report_tasks = {}

    for task in sorted(test_meta["task"].unique()):
        indices = np.flatnonzero((tasks == task) & (sessions == "session2"))
        prior = train_meta.loc[train_meta["task"] == task, "subject"].value_counts().reindex(range(1, 21), fill_value=0).to_numpy()
        labels = sinkhorn_labels(np.asarray(scores[indices]), prior, args.temperature, args.iterations)
        predicted[indices] = labels
        report_tasks[task] = {
            "n": len(indices),
            "direct_agreement": float(np.mean(labels == direct[indices])),
            "counts": np.bincount(labels, minlength=21)[1:].tolist(),
            "train_prior": prior.tolist(),
        }
        print(task, report_tasks[task]["direct_agreement"], flush=True)

    tag = args.tag or f"sinkhorn_{Path(args.scores).stem}_{args.temperature}"
    submission = pd.read_csv(DATA / "test.csv")
    sample_to_row = dict(zip(test_meta["sample"], range(len(test_meta))))
    submission["Label"] = predicted[submission["Sample"].map(sample_to_row).to_numpy()]
    submission_path = ROOT / "research" / f"submission_{tag}.csv"
    submission.to_csv(submission_path, index=False)
    report = {
        "config": vars(args),
        "tasks": report_tasks,
        "class_counts": {str(k): int(v) for k, v in zip(*np.unique(predicted, return_counts=True))},
        "submission_path": str(submission_path),
    }
    (ROOT / "research" / f"diagnostics_{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
