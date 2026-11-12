"""E32：距离比率等开集判据 + 任务条件化 WCCN。

E31 结果：task_model_wccn 的 Test AUROC 0.7327 最好，但工作点与 E30 持平。
本实验在任务条件化 WCCN 空间里试三种更细的判据：

  dist        最近类中心距离（基线）
  ratio       d1/d2（最近 / 次近），未知样本两类距离都大且接近，比率趋近 1
  gap         d2 - d1（绝对间隔）
  zratio      (d1 - mean_train) / std_train 的逐任务标准化

以及两者的分位平均组合。选择仍只用 Train 内层伪未知 AUROC。
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


def task_signals(train_x, test_x, subjects, train_task, test_task, fit_mask):
    """任务特异模型 + 任务内 WCCN，返回四个新颖度信号与预测。"""
    names = ("dist", "ratio", "gap")
    train_out = {name: np.zeros(len(train_x), dtype=np.float64) for name in names}
    test_out = {name: np.zeros(len(test_x), dtype=np.float64) for name in names}
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

        train_rows = np.flatnonzero(train_task == task)
        test_rows = np.flatnonzero(test_task == task)

        train_matrix = np.sort(dist_matrix(z_train_w[train_rows], centers), axis=1)
        test_matrix = np.sort(dist_matrix(z_test_w[test_rows], centers), axis=1)

        def pack(matrix):
            d1, d2 = matrix[:, 0], matrix[:, 1]
            return {
                "dist": d1,
                "ratio": d1 / np.maximum(d2, 1e-6),
                "gap": d2 - d1,
            }

        train_pack = pack(train_matrix)
        test_pack = pack(test_matrix)
        for name in names:
            train_out[name][train_rows] = train_pack[name]
            test_out[name][test_rows] = test_pack[name]

        if len(test_rows):
            predicted[test_rows] = classifier.classes_[
                model.decision_function(test_x[test_rows]).argmax(1)
            ]

    return train_out, test_out, predicted


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

    # 内层评估
    inner_result = {}
    for fold, unknown in enumerate(np.array_split(np.random.default_rng(42).permutation(np.unique(subjects)), 5), 1):
        known_subjects = np.setdiff1d(np.unique(subjects), unknown)
        for inner_unknown in np.array_split(np.random.default_rng(100 + fold).permutation(known_subjects), 2):
            enrolled = np.setdiff1d(known_subjects, inner_unknown)
            enrolled_mask = np.isin(subjects, enrolled)
            inner_fit = enrolled_mask & ~forward & ~inner
            eval_known = np.flatnonzero(enrolled_mask & inner)
            eval_unknown = np.flatnonzero(np.isin(subjects, inner_unknown) & ~forward)
            labels = np.r_[np.ones(len(eval_known)), np.zeros(len(eval_unknown))]
            train_signals, _, _ = task_signals(train_x, test_x, subjects, train_task, test_task, inner_fit)
            for name, values in train_signals.items():
                column = values[np.r_[eval_known, eval_unknown]]
                inner_result.setdefault(name, []).append(float(roc_auc_score(labels, -column)))

    inner_mean = {name: float(np.mean(values)) for name, values in inner_result.items()}
    print("内层 AUROC:", {k: round(v, 4) for k, v in inner_mean.items()}, flush=True)

    # 全量
    train_signals, test_signals, predicted = task_signals(
        train_x, test_x, subjects, train_task, test_task, np.ones(len(subjects), dtype=bool)
    )

    report = {"inner_auroc": inner_mean, "signals": {}}
    combos = {
        "dist": ["dist"],
        "ratio": ["ratio"],
        "gap": ["gap"],
        "dist+ratio": ["dist", "ratio"],
        "dist+gap": ["dist", "gap"],
        "ratio+gap": ["ratio", "gap"],
        "dist+ratio+gap": ["dist", "ratio", "gap"],
    }
    for combo_name, members in combos.items():
        def to_quantile(values, name):
            reference = np.sort(train_signals[name][inner])
            return np.searchsorted(reference, values) / len(reference)

        test_fused = np.mean([to_quantile(test_signals[m], m) for m in members], axis=0)
        train_fused = np.mean([to_quantile(train_signals[m], m) for m in members], axis=0)
        auroc = float(roc_auc_score(known, -test_fused))
        entry = {"test_auroc": auroc, "points": {}}
        for q in QUANTILES:
            threshold = float(np.quantile(train_fused[inner], 1 - q))
            keep = test_fused <= threshold
            candidate = np.where(keep, predicted, 0)
            entry["points"][f"q{int(q*100):02d}"] = {
                "s2_accept": float(keep[known].mean()),
                "s1_reject": float((~keep[~known]).mean()),
                "zero_count": int((candidate == 0).sum()),
            }
            pd.DataFrame({"Sample": test_meta["sample"], "Label": candidate}).to_csv(
                OUT / f"candidate_{combo_name}_q{int(q*100):02d}.csv", index=False
            )
        report["signals"][combo_name] = entry
        print(f"{combo_name:20s} Test AUROC={auroc:.4f} | " + " ".join(
            f"{k}:{v['s2_accept']:.3f}/{v['s1_reject']:.3f}" for k, v in entry["points"].items()
        ), flush=True)

    (OUT / "e32_ratio.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
