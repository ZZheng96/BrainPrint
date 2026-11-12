"""E31：DINO 任务条件化类中心与距离。

动机：不同任务的身份可分性差异很大（EC/EO 各约 1,070 条，Imagery 5,504，SSVEP 3,658），
全局类中心把四种任务的特征混在一起，可能模糊了任务内的身份结构。

三个变体（全部只用训练集估计，逐样本推断）：
  A. 全局模型 + 全局中心（E30 基线）
  B. 全局模型 + 同任务中心（只和本任务的类中心比距离）
  C. 任务特异模型 + 同任务中心 + 任务内 WCCN

评估：内层伪未知 AUROC（选型用）与 Test AUROC（审计）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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


def nearest_distance(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(data**2, axis=1, keepdims=True)
        - 2 * data @ centroids.T
        + np.sum(centroids**2, axis=1)[None, :]
    )
    return np.sqrt(np.maximum(squared.min(axis=1), 0.0))


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


def distance_signals(train_x, test_x, subjects, train_task, test_task, fit_mask, mode):
    """返回 (train_signal, test_signal, predicted)。"""
    if mode == "global_global":
        model = fit(train_x[fit_mask], subjects[fit_mask])
        scaler = model.named_steps["standardscaler"]
        classifier = model.named_steps["logisticregression"]
        z_train = scaler.transform(train_x).astype(np.float32)
        z_test = scaler.transform(test_x).astype(np.float32)
        centers = np.stack([z_train[fit_mask & (subjects == c)].mean(0) for c in classifier.classes_])
        return nearest_distance(z_train, centers), nearest_distance(z_test, centers), \
            classifier.classes_[model.decision_function(test_x).argmax(1)]

    if mode == "global_taskcenter":
        model = fit(train_x[fit_mask], subjects[fit_mask])
        scaler = model.named_steps["standardscaler"]
        classifier = model.named_steps["logisticregression"]
        z_train = scaler.transform(train_x).astype(np.float32)
        z_test = scaler.transform(test_x).astype(np.float32)
        train_signal = np.zeros(len(train_x), dtype=np.float32)
        test_signal = np.zeros(len(test_x), dtype=np.float32)
        for task in TASKS:
            task_fit = fit_mask & (train_task == task)
            centers = np.stack([z_train[task_fit & (subjects == c)].mean(0) for c in classifier.classes_])
            rows = train_task == task
            train_signal[rows] = nearest_distance(z_train[rows], centers)
            rows_test = test_task == task
            if rows_test.any():
                test_signal[rows_test] = nearest_distance(z_test[rows_test], centers)
        return train_signal, test_signal, classifier.classes_[model.decision_function(test_x).argmax(1)]

    if mode == "task_model_taskcenter":
        train_signal = np.zeros(len(train_x), dtype=np.float32)
        test_signal = np.zeros(len(test_x), dtype=np.float32)
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
            centers = np.stack([z_train[task_fit & (subjects == c)].mean(0) for c in classifier.classes_])
            rows = train_task == task
            train_signal[rows] = nearest_distance(z_train[rows], centers)
            rows_test = test_task == task
            if rows_test.any():
                test_signal[rows_test] = nearest_distance(z_test[rows_test], centers)
                predicted[rows_test] = classifier.classes_[model.decision_function(test_x[rows_test]).argmax(1)]
        return train_signal, test_signal, predicted

    if mode == "task_model_wccn":
        train_signal = np.zeros(len(train_x), dtype=np.float32)
        test_signal = np.zeros(len(test_x), dtype=np.float32)
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
            rows = train_task == task
            train_signal[rows] = nearest_distance(z_train_w[rows], centers)
            rows_test = test_task == task
            if rows_test.any():
                test_signal[rows_test] = nearest_distance(z_test_w[rows_test], centers)
                predicted[rows_test] = classifier.classes_[model.decision_function(test_x[rows_test]).argmax(1)]
        return train_signal, test_signal, predicted

    raise ValueError(mode)


def main() -> None:
    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial"])
    subjects = metadata["subject"].to_numpy()
    train_task = metadata["task"].to_numpy()
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    test_meta = pd.read_csv(CACHE / "test_metadata.csv", usecols=["sample", "session", "task"])
    known = (test_meta["session"] == "session2").to_numpy()
    test_task = test_meta["task"].to_numpy()

    train_x = np.load(DINO / "train_covariance5_dinov2.npy").astype(np.float32)
    test_x = np.load(DINO / "test_covariance5_dinov2.npy").astype(np.float32)

    modes = ("global_global", "global_taskcenter", "task_model_taskcenter", "task_model_wccn")
    report = {}

    # 内层评估（选型）：5 折留被试，内层再留一半
    inner_scores = {mode: [] for mode in modes}
    for fold, unknown in enumerate(np.array_split(np.random.default_rng(42).permutation(np.unique(subjects)), 5), 1):
        known_subjects = np.setdiff1d(np.unique(subjects), unknown)
        for inner_unknown in np.array_split(np.random.default_rng(100 + fold).permutation(known_subjects), 2):
            enrolled = np.setdiff1d(known_subjects, inner_unknown)
            enrolled_mask = np.isin(subjects, enrolled)
            inner_fit = enrolled_mask & ~forward & ~inner
            eval_known = np.flatnonzero(enrolled_mask & inner)
            eval_unknown = np.flatnonzero(np.isin(subjects, inner_unknown) & ~forward)
            labels = np.r_[np.ones(len(eval_known)), np.zeros(len(eval_unknown))]
            for mode in modes:
                train_signal, _, _ = distance_signals(
                    train_x, test_x, subjects, train_task, test_task, inner_fit, mode
                )
                column = train_signal[np.r_[eval_known, eval_unknown]]
                inner_scores[mode].append(float(roc_auc_score(labels, -column)))

    for mode in modes:
        report[mode] = {"inner_auroc": float(np.mean(inner_scores[mode]))}
        print(f"{mode:24s} 内层 AUROC = {report[mode]['inner_auroc']:.4f}", flush=True)

    # 全量：Test 审计 + 工作点
    fit_all = np.ones(len(subjects), dtype=bool)
    for mode in modes:
        train_signal, test_signal, predicted = distance_signals(
            train_x, test_x, subjects, train_task, test_task, fit_all, mode
        )
        auroc = float(roc_auc_score(known, -test_signal))
        entry = {
            "inner_auroc": report[mode]["inner_auroc"],
            "test_auroc": auroc,
            "train_p50": float(np.median(train_signal[inner])),
            "s2_p50": float(np.median(test_signal[known])),
            "s1_p50": float(np.median(test_signal[~known])),
            "points": {},
        }
        for q in QUANTILES:
            threshold = float(np.quantile(train_signal[inner], 1 - q))
            keep = test_signal <= threshold
            candidate = np.where(keep, predicted, 0)
            entry["points"][f"q{int(q*100):02d}"] = {
                "s2_accept": float(keep[known].mean()),
                "s1_reject": float((~keep[~known]).mean()),
                "zero_count": int((candidate == 0).sum()),
            }
            pd.DataFrame({"Sample": test_meta["sample"], "Label": candidate}).to_csv(
                OUT / f"candidate_{mode}_q{int(q*100):02d}.csv", index=False
            )
        report[mode] = entry
        print(f"{mode:24s} Test AUROC={auroc:.4f} | train {entry['train_p50']:.2f} "
              f"S2 {entry['s2_p50']:.2f} S1 {entry['s1_p50']:.2f} | " + " ".join(
                  f"{k}:{v['s2_accept']:.3f}/{v['s1_reject']:.3f}" for k, v in entry["points"].items()
              ), flush=True)

    (OUT / "e31_task_conditional.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
