"""E34B：在 covariance_hr 特征上复跑 E31 的任务条件化 WCCN 开集评估。

用法：python run_e34_eval.py covariance_hr
对比基线：covariance5（E31 最优，Test AUROC 0.7327；q01 S2 0.917 / S1 0.509）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

CACHE = ROOT / "BrainPrint_cache"
DINO = CACHE / "eeg_image_dinov2"
OUT = Path(__file__).resolve().parent
TASKS = ("EC", "EO", "Imagery", "SSVEP")
QUANTILES = (0.01, 0.02, 0.05, 0.10, 0.20)


def dist_matrix(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(data**2, axis=1, keepdims=True)
        - 2 * data @ centroids.T
        + np.sum(centroids**2, axis=1)[None, :]
    )
    return np.sqrt(np.maximum(squared, 0.0))


def fit(train_x, labels):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300))
    model.fit(train_x, labels)
    return model


def whitening_from(z: np.ndarray, labels: np.ndarray, shrinkage: float = 0.2) -> np.ndarray:
    centered = np.zeros_like(z)
    classes = np.unique(labels)
    for label in classes:
        mask = labels == label
        centered[mask] = z[mask] - z[mask].mean(0)
    scatter = centered.T @ centered / max(len(z) - len(classes), 1)
    scatter = (1 - shrinkage) * scatter + shrinkage * np.trace(scatter) / scatter.shape[0] * np.eye(scatter.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    return (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T


def task_wccn_distance(train_x, test_x, subjects, train_task, test_task, fit_mask):
    """任务特异模型 + 任务内 WCCN，返回训练/测试的最近中心距离与预测。"""
    train_out = np.zeros(len(train_x), dtype=np.float64)
    test_out = np.zeros(len(test_x), dtype=np.float64)
    predicted = np.zeros(len(test_x), dtype=int)
    for task in TASKS:
        task_fit = fit_mask & (train_task == task)
        if task_fit.sum() < 100:
            continue
        model = fit(train_x[task_fit], subjects[task_fit])
        scaler = model.named_steps["standardscaler"]
        classifier = model.named_steps["logisticregression"]
        z_train = scaler.transform(train_x).astype(np.float32)
        z_test = scaler.transform(test_x).astype(np.float32)
        whitening = whitening_from(z_train[task_fit], subjects[task_fit])
        z_train_w = (z_train @ whitening).astype(np.float32)
        z_test_w = (z_test @ whitening).astype(np.float32)
        centers = np.stack([z_train_w[task_fit & (subjects == c)].mean(0) for c in classifier.classes_])
        rows = np.flatnonzero(train_task == task)
        test_rows = np.flatnonzero(test_task == task)
        train_out[rows] = dist_matrix(z_train_w[rows], centers).min(1)
        if len(test_rows):
            test_out[test_rows] = dist_matrix(z_test_w[test_rows], centers).min(1)
            predicted[test_rows] = classifier.classes_[
                model.decision_function(test_x[test_rows]).argmax(1)
            ]
    return train_out, test_out, predicted


def main() -> None:
    method = sys.argv[1] if len(sys.argv) > 1 else "covariance_hr"
    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial"])
    subjects = metadata["subject"].to_numpy()
    train_task = metadata["task"].to_numpy()
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    test_meta = pd.read_csv(CACHE / "test_metadata.csv", usecols=["sample", "session", "task"])
    known = (test_meta["session"] == "session2").to_numpy()
    test_task = test_meta["task"].to_numpy()

    train_x = np.load(DINO / f"train_{method}_dinov2.npy").astype(np.float32)
    test_x = np.load(DINO / f"test_{method}_dinov2.npy").astype(np.float32)

    inner_aurocs = []
    for fold, unknown in enumerate(np.array_split(np.random.default_rng(42).permutation(np.unique(subjects)), 5), 1):
        known_subjects = np.setdiff1d(np.unique(subjects), unknown)
        for inner_unknown in np.array_split(np.random.default_rng(100 + fold).permutation(known_subjects), 2):
            enrolled = np.setdiff1d(known_subjects, inner_unknown)
            mask_enrolled = np.isin(subjects, enrolled)
            inner_fit = mask_enrolled & ~forward & ~inner
            eval_known = np.flatnonzero(mask_enrolled & inner)
            eval_unknown = np.flatnonzero(np.isin(subjects, inner_unknown) & ~forward)
            train_signal, _, _ = task_wccn_distance(
                train_x, test_x, subjects, train_task, test_task, inner_fit
            )
            labels = np.r_[np.ones(len(eval_known)), np.zeros(len(eval_unknown))]
            inner_aurocs.append(float(roc_auc_score(
                labels, -train_signal[np.r_[eval_known, eval_unknown]]
            )))
    inner_auroc = float(np.mean(inner_aurocs))
    print(f"{method}: 内层 AUROC = {inner_auroc:.4f}", flush=True)

    train_signal, test_signal, predicted = task_wccn_distance(
        train_x, test_x, subjects, train_task, test_task, np.ones(len(subjects), dtype=bool)
    )
    test_auroc = float(roc_auc_score(known, -test_signal))
    print(f"{method}: Test AUROC = {test_auroc:.4f} | train {np.median(train_signal[inner]):.2f} "
          f"S2 {np.median(test_signal[known]):.2f} S1 {np.median(test_signal[~known]):.2f}", flush=True)

    report = {
        "method": method,
        "inner_auroc": inner_auroc,
        "test_auroc": test_auroc,
        "points": {},
    }
    for q in QUANTILES:
        threshold = float(np.quantile(train_signal[inner], 1 - q))
        keep = test_signal <= threshold
        candidate = np.where(keep, predicted, 0)
        report["points"][f"q{int(q*100):02d}"] = {
            "s2_accept": float(keep[known].mean()),
            "s1_reject": float((~keep[~known]).mean()),
            "zero_count": int((candidate == 0).sum()),
        }
        pd.DataFrame({"Sample": test_meta["sample"], "Label": candidate}).to_csv(
            OUT / f"candidate_{method}_q{int(q*100):02d}.csv", index=False
        )
        print(f"  q{int(q*100):02d}: S2 accept={report['points'][f'q{int(q*100):02d}']['s2_accept']:.3f} "
              f"S1 reject={report['points'][f'q{int(q*100):02d}']['s1_reject']:.3f}", flush=True)

    (OUT / f"e34_{method}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
