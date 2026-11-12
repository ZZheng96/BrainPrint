"""E35：按任务拆解身份证据，并做任务内分位阈值的拒识。

E31 只用「全局分位」设阈值，但四种任务的距离分布差异很大（EC/EO/Imagery/SSVEP）。
本实验：

1. 逐任务测量开集可分性（内层伪未知 AUROC、Test 审计 AUROC）；
2. 逐任务测量闭集身份准确率（留被试）；
3. 对比两种拒识策略：
   - global：一个全局阈值（E31 做法）
   - per_task：每个任务各自取「内层已知距离的 q 分位」

阈值全部来自 Train 内层伪未知，Test 只做审计与工作点报告。
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
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
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


def task_distances(train_x, test_x, subjects, train_task, test_task, fit_mask):
    """任务特异模型 + 任务内 WCCN，返回距离、预测与逐任务闭集准确率。"""
    train_out = np.full(len(train_x), np.nan)
    test_out = np.full(len(test_x), np.nan)
    predicted = np.zeros(len(test_x), dtype=int)
    closed = {}
    for task in TASKS:
        task_fit = fit_mask & (train_task == task)
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
        predicted[test_rows] = classifier.classes_[
            model.decision_function(test_x[test_rows]).argmax(1)
        ]
        test_out[test_rows] = dist_matrix(z_test_w[test_rows], centers).min(1)
        closed[task] = float(np.mean(
            classifier.classes_[model.decision_function(train_x[rows]).argmax(1)] == subjects[rows]
        ))
    return train_out, test_out, predicted, closed


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

    # 内层逐任务标定
    inner_known = {task: [] for task in TASKS}
    inner_unknown = {task: [] for task in TASKS}
    inner_task_auroc = {task: [] for task in TASKS}
    for fold, unknown in enumerate(np.array_split(np.random.default_rng(42).permutation(np.unique(subjects)), 5), 1):
        known_subjects = np.setdiff1d(np.unique(subjects), unknown)
        for inner_unknown_subjects in np.array_split(
            np.random.default_rng(100 + fold).permutation(known_subjects), 2
        ):
            enrolled = np.setdiff1d(known_subjects, inner_unknown_subjects)
            mask_enrolled = np.isin(subjects, enrolled)
            inner_fit = mask_enrolled & ~forward & ~inner
            train_signal, _, _, _ = task_distances(
                train_x, test_x, subjects, train_task, test_task, inner_fit
            )
            known_rows = np.flatnonzero(mask_enrolled & inner)
            unknown_rows = np.flatnonzero(np.isin(subjects, inner_unknown_subjects) & ~forward)
            for task in TASKS:
                k = known_rows[train_task[known_rows] == task]
                u = unknown_rows[train_task[unknown_rows] == task]
                if len(k) < 20 or len(u) < 20:
                    continue
                inner_known[task].append(train_signal[k])
                inner_unknown[task].append(train_signal[u])
                inner_task_auroc[task].append(float(roc_auc_score(
                    np.r_[np.ones(len(k)), np.zeros(len(u))], -np.r_[train_signal[k], train_signal[u]]
                )))

    inner_known = {t: np.concatenate(v) for t, v in inner_known.items()}
    inner_unknown = {t: np.concatenate(v) for t, v in inner_unknown.items()}
    inner_auroc = {t: float(np.mean(v)) for t, v in inner_task_auroc.items()}

    # 全量
    train_signal, test_signal, predicted, closed = task_distances(
        train_x, test_x, subjects, train_task, test_task, np.ones(len(subjects), dtype=bool)
    )

    report = {"per_task": {}, "strategies": {}}
    for task in TASKS:
        test_rows = known & (test_task == task)
        test_unknown = (~known) & (test_task == task)
        report["per_task"][task] = {
            "closed_acc_insample": closed[task],
            "inner_auroc": inner_auroc[task],
            "test_auroc": float(roc_auc_score(
                np.r_[np.ones(test_rows.sum()), np.zeros(test_unknown.sum())],
                -np.r_[test_signal[test_rows], test_signal[test_unknown]],
            )),
            "train_p50": float(np.median(inner_known[task])),
            "s2_p50": float(np.median(test_signal[test_rows])),
            "s1_p50": float(np.median(test_signal[test_unknown])),
        }
        print(f"{task:8s} closed={closed[task]:.4f} 内层AUROC={inner_auroc[task]:.4f} "
              f"TestAUROC={report['per_task'][task]['test_auroc']:.4f} | "
              f"train {np.median(inner_known[task]):.2f} "
              f"S2 {np.median(test_signal[test_rows]):.2f} S1 {np.median(test_signal[test_unknown]):.2f}",
              flush=True)

    # 两种阈值策略
    for name in ("global", "per_task"):
        entry = {"points": {}}
        for q in QUANTILES:
            if name == "global":
                threshold = float(np.quantile(train_signal[inner], 1 - q))
                keep = test_signal <= threshold
            else:
                keep = np.ones(len(test_signal), dtype=bool)
                for task in TASKS:
                    threshold = float(np.quantile(inner_known[task], 1 - q))
                    rows = test_task == task
                    keep[rows] = test_signal[rows] <= threshold
            candidate = np.where(keep, predicted, 0)
            entry["points"][f"q{int(q*100):02d}"] = {
                "s2_accept": float(keep[known].mean()),
                "s1_reject": float((~keep[~known]).mean()),
                "zero_count": int((candidate == 0).sum()),
            }
            pd.DataFrame({"Sample": test_meta["sample"], "Label": candidate}).to_csv(
                OUT / f"candidate_{name}_q{int(q*100):02d}.csv", index=False
            )
        report["strategies"][name] = entry
        print(f"{name:10s} " + " ".join(
            f"{k}:{v['s2_accept']:.3f}/{v['s1_reject']:.3f}" for k, v in entry["points"].items()
        ), flush=True)

    (OUT / "e35_per_task.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
