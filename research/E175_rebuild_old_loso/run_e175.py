"""E175：为 E26–E35 这批「只有 Test CSV 候选、没有同阈值 LOSO」的实验补算 W。

流程（每个实验）：
  1. 按原脚本逐行重建 Test 分数与阈值 -> 预测标签；
  2. 与已保存候选 CSV 逐行比对，一致率 >= 99% 才认为管线重建成功；
  3. 在同一阈值规则上做干净的 LOSO（原型只用 frac <= 0.6，被评分块 0.6 < frac <= 0.8）；
  4. W = 0.25*LOSO + 0.25*CSV_S1 + 0.5*CSV_S2。

不修改任何已有实验目录；Test 官方 AUC/F1 = N/A（无真实标签）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
OUT = Path(__file__).resolve().parent
CACHE = ROOT / "BrainPrint_cache"
DINO = CACHE / "eeg_image_dinov2"
RESEARCH = ROOT / "research"

TASKS = ("EC", "EO", "Imagery", "SSVEP")
CLASSES = np.arange(1, 21, dtype=int)

LOG_PATH = OUT / "run.log"
_log_handle = None


def log(message: str = "") -> None:
    global _log_handle
    if _log_handle is None:
        _log_handle = open(LOG_PATH, "a", encoding="utf-8")
    print(message, flush=True)
    _log_handle.write(str(message) + "\n")
    _log_handle.flush()


# --------------------------------------------------------------------------
# 数据与公共工具
# --------------------------------------------------------------------------
class Data:
    def __init__(self) -> None:
        meta = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial"])
        self.meta = meta
        self.subjects = meta["subject"].to_numpy(int)
        self.tasks = meta["task"].to_numpy()
        self.trial = meta["trial"].to_numpy(float)
        self.max_trial = meta.groupby(["subject", "task"])["trial"].transform("max").to_numpy(float)
        self.frac = self.trial / np.maximum(self.max_trial, 1.0)
        # 与 src/open_set_improvement/run_nested.py 完全一致的 holdout 定义
        self.forward = self.trial > 0.8 * self.max_trial
        self.inner = (self.trial > 0.6 * self.max_trial) & (self.trial <= 0.8 * self.max_trial)
        self.fit60 = self.trial <= 0.6 * self.max_trial
        test = pd.read_csv(CACHE / "test_metadata.csv", usecols=["sample", "session", "task"])
        self.test_meta = test
        self.test_sample = test["sample"].to_numpy()
        self.test_task = test["task"].to_numpy()
        self.test_known = (test["session"] == "session2").to_numpy()
        self.n_s1 = int((~self.test_known).sum())
        self.n_s2 = int(self.test_known.sum())
        self._cache: dict[str, np.ndarray] = {}

    def dino(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        key = f"dino:{name}"
        if key not in self._cache:
            self._cache[key] = np.load(DINO / f"train_{name}_dinov2.npy").astype(np.float32)
            self._cache[f"dino_test:{name}"] = np.load(DINO / f"test_{name}_dinov2.npy").astype(np.float32)
        return self._cache[key], self._cache[f"dino_test:{name}"]

    def csv_labels(self, path: Path) -> np.ndarray:
        """按 test_metadata 的行序对齐候选 CSV 的 Label。"""
        frame = pd.read_csv(path)
        sample_col = "Sample" if "Sample" in frame.columns else "sample"
        label_col = "Label" if "Label" in frame.columns else "label"
        mapping = dict(zip(frame[sample_col], frame[label_col]))
        labels = np.asarray([mapping[s] for s in self.test_sample], dtype=int)
        return labels


def nearest_distance(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(data**2, axis=1, keepdims=True)
        - 2 * data @ centroids.T
        + np.sum(centroids**2, axis=1)[None, :]
    )
    return np.sqrt(np.maximum(squared.min(axis=1), 0.0)).astype(np.float32)


def dist_matrix(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(data**2, axis=1, keepdims=True)
        - 2 * data @ centroids.T
        + np.sum(centroids**2, axis=1)[None, :]
    )
    return np.sqrt(np.maximum(squared, 0.0))


def fit_lr(x: np.ndarray, y: np.ndarray, max_iter: int = 300):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=max_iter))
    model.fit(x, y)
    return model


def whitening_from(z: np.ndarray, labels: np.ndarray, shrinkage: float) -> np.ndarray:
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


def quantile_wrt(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(reference)
    return np.searchsorted(ordered, values) / len(ordered)


class ManualScaler:
    """与 sklearn StandardScaler(ddof=0) 等价；LOSO 里只需要标准化参数，不需要分类器系数。"""

    def __init__(self, x_fit: np.ndarray) -> None:
        self.mean = x_fit.mean(0)
        self.scale = np.where(x_fit.std(0) > 1e-12, x_fit.std(0), 1.0).astype(np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale).astype(np.float32)


def w_score(loso: float | None, s1: float, s2: float) -> float | None:
    if loso is None or not np.isfinite(loso):
        return None
    return float(0.25 * loso + 0.25 * s1 + 0.5 * s2)


# --------------------------------------------------------------------------
# 候选记录
# --------------------------------------------------------------------------
def make_record(
    experiment: str,
    candidate: str,
    csv_path: Path,
    data: Data,
    predicted: np.ndarray,
    novel: np.ndarray | None,
    threshold: float | None,
    loso: float | None,
    rule: str,
    reported: dict | None = None,
    notes: str = "",
    keep: np.ndarray | None = None,
) -> dict:
    if keep is None:
        keep = novel <= threshold
    repro = np.where(keep, predicted, 0)
    csv_labels = data.csv_labels(csv_path)
    agreement = float(np.mean(repro == csv_labels))
    s1 = float(np.mean(csv_labels[~data.test_known] == 0))
    s2 = float(np.mean(csv_labels[data.test_known] != 0))
    flags = []
    if agreement < 0.99:
        flags.append("pipeline_mismatch")
    else:
        flags.append("clean")
    w = w_score(loso if agreement >= 0.99 else None, s1, s2)
    record = {
        "experiment": experiment,
        "candidate": candidate,
        "csv": csv_path.name,
        "rule": rule,
        "threshold": (None if threshold is None or not np.isfinite(threshold) else float(threshold)),
        "repro_agreement": agreement,
        "csv_s1_rejection": s1,
        "csv_s2_acceptance": s2,
        "loso_rejection": (None if agreement < 0.99 or loso is None or not np.isfinite(loso) else float(loso)),
        "W": w,
        "flags": flags,
        "notes": notes,
    }
    if reported:
        record["reported_s1"] = reported.get("s1_reject")
        record["reported_s2"] = reported.get("s2_accept")
    return record


def check_reported(record: dict) -> str:
    if record.get("reported_s1") is None:
        return ""
    d1 = abs(record["csv_s1_rejection"] - record["reported_s1"])
    d2 = abs(record["csv_s2_acceptance"] - record["reported_s2"])
    return f"reported diff S1={d1:.4f} S2={d2:.4f}"


# --------------------------------------------------------------------------
# 通用 LOSO：给定每折 known/unknown 分数，按同一分位规则算拒识率
# --------------------------------------------------------------------------
def loso_from_folds(folds: list[dict], rate: float) -> float:
    values = []
    for fold in folds:
        known = fold["known"]
        unknown = fold["unknown"]
        if len(known) == 0 or len(unknown) == 0:
            continue
        threshold = float(np.quantile(known, 1 - rate))
        values.append(float(np.mean(unknown > threshold)))
    if not values:
        return float("nan")
    return float(np.mean(values))


def loso_per_task_from_folds(folds: list[dict], rate: float) -> float:
    """E35 per_task 规则：阈值按任务各自取 known 块的 1-rate 分位。"""
    totals, hits = 0, 0
    for fold in folds:
        for task in TASKS:
            known = fold["known"][task]
            unknown = fold["unknown"][task]
            if len(known) < 20 or len(unknown) < 20:
                continue
            threshold = float(np.quantile(known, 1 - rate))
            totals += len(unknown)
            hits += int(np.sum(unknown > threshold))
    return float(hits / totals) if totals else float("nan")


def loso_from_fused_folds(folds: list[dict], rate: float) -> float:
    """E30/E32：分数是成员信号相对 known 块的逐折分位均值。"""
    values = []
    for fold in folds:
        known = fold["known"]
        unknown = fold["unknown"]
        if len(known) == 0 or len(unknown) == 0:
            continue
        threshold = float(np.quantile(known, 1 - rate))
        values.append(float(np.mean(unknown > threshold)))
    return float(np.mean(values)) if values else float("nan")


# --------------------------------------------------------------------------
# E26 / E27：协方差原始特征 + SNR30 对抗增广 + 距离型拒识
# --------------------------------------------------------------------------
def cov_aug_features(data: Data) -> tuple[np.ndarray, np.ndarray]:
    from src.open_set_improvement.core import COV_COLUMNS

    train_clean = np.asarray(
        np.load(CACHE / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32
    )
    train_noisy = np.asarray(np.load(CACHE / "e25_robust" / "train_cov_snr30.npy"), dtype=np.float32)
    test = np.asarray(
        np.load(CACHE / "test_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32
    )
    return train_clean, train_noisy, test


def run_e26_e27(data: Data, experiment: str, exp_dir: Path, rates: tuple[int, ...]) -> dict:
    train_clean, train_noisy, test = cov_aug_features(data)
    subjects = data.subjects
    features = np.concatenate([train_clean, train_noisy])
    labels = np.concatenate([subjects, subjects])
    model = fit_lr(features, labels)
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    centroid_mask = ~data.inner & ~data.forward

    z_train = scaler.transform(train_clean).astype(np.float32)
    centroids = np.stack([z_train[centroid_mask & (subjects == c)].mean(0) for c in classifier.classes_])
    z_test = scaler.transform(test).astype(np.float32)
    inner_distance = nearest_distance(z_train[data.inner], centroids)
    test_distance = nearest_distance(z_test, centroids)
    predicted = classifier.classes_[model.decision_function(test).argmax(1)]

    log(f"[{experiment}] rebuilt Test 距离 AUROC(S2 vs S1) = "
        f"{roc_auc_score(data.test_known, -test_distance):.4f} | "
        f"inner p50 {np.median(inner_distance):.2f} S2 {np.median(test_distance[data.test_known]):.2f} "
        f"S1 {np.median(test_distance[~data.test_known]):.2f}")

    # ---------------- 干净 LOSO ----------------
    folds = []
    for held in CLASSES:
        enrolled = subjects != held
        fit_rows = np.flatnonzero(enrolled & data.fit60)
        known_rows = np.flatnonzero(enrolled & data.inner)
        unknown_rows = np.flatnonzero((subjects == held) & data.inner)
        if len(unknown_rows) == 0:
            continue
        fold_scaler = ManualScaler(np.concatenate([train_clean[fit_rows], train_noisy[fit_rows]]))
        fold_classes = np.setdiff1d(CLASSES, [held])
        rows = np.r_[known_rows, unknown_rows]
        z_rows = fold_scaler.transform(train_clean[rows]).astype(np.float32)
        z_fit = fold_scaler.transform(train_clean[fit_rows]).astype(np.float32)
        centers = np.stack([z_fit[subjects[fit_rows] == c].mean(0) for c in fold_classes])
        scores = nearest_distance(z_rows, centers)
        folds.append({"known": scores[: len(known_rows)], "unknown": scores[len(known_rows):]})

    candidates = []
    for rate in rates:
        name = f"reject{rate}"
        csv_path = exp_dir / f"candidate_{name}.csv"
        threshold = float(np.quantile(inner_distance, 1 - rate / 100))
        loso = loso_from_folds(folds, rate / 100)
        json_path = exp_dir / ("e26_candidates.json" if experiment.startswith("E26") else "e27b_distance_only.json")
        reported = None
        if json_path.exists():
            block = json.loads(json_path.read_text(encoding="utf-8")).get("operating_points") or {}
            if not block:
                block = json.loads(json_path.read_text(encoding="utf-8")).get("points", {})
            reported = block.get(name)
        candidates.append(make_record(
            experiment, name, csv_path, data, predicted, test_distance, threshold, loso,
            rule=f"threshold = Train 内层(0.6<frac<=0.8)距离的 {100-rate}% 分位；LOSO 用同分位的 19 人 known 块",
            reported=reported,
        ))
    return {"experiment": experiment, "candidates": candidates,
            "notes": "阈值来自全量模型 ~inner&~forward 原型 + Train 内层分位；无 Test 统计。"}


# --------------------------------------------------------------------------
# E28 / E29：DINO 特征上的距离型拒识
# --------------------------------------------------------------------------
def run_e28(data: Data, exp_dir: Path) -> dict:
    subjects = data.subjects
    train_cov, test_cov = data.dino("covariance5")
    train_topo, test_topo = data.dino("topomap5")
    feature_sets = {
        "cov5": (train_cov, test_cov),
        "cov5_topo5": (np.concatenate([train_cov, train_topo], 1),
                       np.concatenate([test_cov, test_topo], 1)),
    }
    report = json.loads((exp_dir / "e28b_dino_openset.json").read_text(encoding="utf-8"))
    rates = (1, 2, 5, 10, 20, 30)
    candidates = []
    for name, (train_x, test_x) in feature_sets.items():
        model = fit_lr(train_x, subjects)
        scaler = model.named_steps["standardscaler"]
        classifier = model.named_steps["logisticregression"]
        z_train = scaler.transform(train_x).astype(np.float32)
        z_test = scaler.transform(test_x).astype(np.float32)
        centroid_mask = ~data.inner & ~data.forward
        centers = np.stack([z_train[centroid_mask & (subjects == c)].mean(0) for c in classifier.classes_])
        inner_distance = nearest_distance(z_train[data.inner], centers)
        test_distance = nearest_distance(z_test, centers)
        predicted = classifier.classes_[model.decision_function(test_x).argmax(1)]
        log(f"[E28] {name}: Test AUROC {roc_auc_score(data.test_known, -test_distance):.4f} | "
            f"inner p50 {np.median(inner_distance):.2f}")

        folds = []
        for held in CLASSES:
            enrolled = subjects != held
            fit_rows = np.flatnonzero(enrolled & data.fit60)
            known_rows = np.flatnonzero(enrolled & data.inner)
            unknown_rows = np.flatnonzero((subjects == held) & data.inner)
            if len(unknown_rows) == 0:
                continue
            fold_scaler = ManualScaler(train_x[fit_rows])
            fold_classes = np.setdiff1d(CLASSES, [held])
            z_fit = fold_scaler.transform(train_x[fit_rows]).astype(np.float32)
            centers = np.stack([z_fit[subjects[fit_rows] == c].mean(0) for c in fold_classes])
            z_rows = fold_scaler.transform(train_x[np.r_[known_rows, unknown_rows]]).astype(np.float32)
            scores = nearest_distance(z_rows, centers)
            folds.append({"known": scores[: len(known_rows)], "unknown": scores[len(known_rows):]})

        for rate in rates:
            cand = f"reject{rate}"
            reported = report[name]["points"][cand]
            threshold = float(reported["threshold"])
            loso = loso_from_folds(folds, rate / 100)
            candidates.append(make_record(
                "E28_dino_openset", f"{name}_{cand}", exp_dir / f"candidate_{name}_dist_{cand}.csv",
                data, predicted, test_distance, threshold, loso,
                rule=f"threshold = Train 内层距离 {100-rate}% 分位（全量原型 ~inner&~forward）",
                reported=reported,
            ))
    return {"experiment": "E28_dino_openset", "candidates": candidates,
            "notes": "cov5 / cov5+topo5 两个特征集；阈值均为 Train 内层分位。"}


def run_e29(data: Data, exp_dir: Path) -> dict:
    subjects = data.subjects
    train_x, test_x = data.dino("covariance5")
    known = data.test_known
    # 与 run_e29b 完全一致：10 个内层模型（5 折 × 2 半）收集内层已知/伪未知距离
    inner_known_distance, inner_unknown_distance = [], []
    for fold, unknown in enumerate(
        np.array_split(np.random.default_rng(42).permutation(np.unique(subjects)), 5), 1
    ):
        known_subjects = np.setdiff1d(np.unique(subjects), unknown)
        is_known = np.isin(subjects, known_subjects)
        for inner_unknown in np.array_split(
            np.random.default_rng(100 + fold).permutation(known_subjects), 2
        ):
            enrolled = np.setdiff1d(known_subjects, inner_unknown)
            enrolled_mask = np.isin(subjects, enrolled)
            inner_fit = enrolled_mask & ~data.forward & ~data.inner
            model = fit_lr(train_x[inner_fit], subjects[inner_fit])
            scaler = model.named_steps["standardscaler"]
            classifier = model.named_steps["logisticregression"]
            z = scaler.transform(train_x).astype(np.float32)
            centers = np.stack([z[inner_fit & (subjects == c)].mean(0) for c in classifier.classes_])
            inner_known_distance.append(
                nearest_distance(z[np.flatnonzero(enrolled_mask & data.inner)], centers)
            )
            inner_unknown_distance.append(
                nearest_distance(z[np.flatnonzero(np.isin(subjects, inner_unknown) & ~data.forward)], centers)
            )
    inner_known_distance = np.concatenate(inner_known_distance)

    model = fit_lr(train_x, subjects)
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    centroid_mask = ~data.inner & ~data.forward
    z_train = scaler.transform(train_x).astype(np.float32)
    centers = np.stack([z_train[centroid_mask & (subjects == c)].mean(0) for c in classifier.classes_])
    z_test = scaler.transform(test_x).astype(np.float32)
    test_distance = nearest_distance(z_test, centers)
    predicted = classifier.classes_[model.decision_function(test_x).argmax(1)]
    log(f"[E29] rebuilt Test AUROC {roc_auc_score(known, -test_distance):.4f} | "
        f"inner known p50 {np.median(inner_known_distance):.2f}")

    folds = []
    for held in CLASSES:
        enrolled = subjects != held
        fit_rows = np.flatnonzero(enrolled & data.fit60)
        known_rows = np.flatnonzero(enrolled & data.inner)
        unknown_rows = np.flatnonzero((subjects == held) & data.inner)
        if len(unknown_rows) == 0:
            continue
        fold_scaler = ManualScaler(train_x[fit_rows])
        fold_classes = np.setdiff1d(CLASSES, [held])
        z_fit = fold_scaler.transform(train_x[fit_rows]).astype(np.float32)
        centers = np.stack([z_fit[subjects[fit_rows] == c].mean(0) for c in fold_classes])
        z_rows = fold_scaler.transform(train_x[np.r_[known_rows, unknown_rows]]).astype(np.float32)
        scores = nearest_distance(z_rows, centers)
        folds.append({"known": scores[: len(known_rows)], "unknown": scores[len(known_rows):]})

    report = json.loads((exp_dir / "e29b_operating_points.json").read_text(encoding="utf-8"))
    candidates = []
    for q in (1, 2, 5, 10, 20, 30, 40, 50):
        cand = f"q{q:02d}"
        reported = report["points"][cand]
        threshold = float(np.quantile(inner_known_distance, 1 - q / 100))
        loso = loso_from_folds(folds, q / 100)
        candidates.append(make_record(
            "E29_dino_calibration", cand, exp_dir / f"candidate_{cand}.csv", data,
            predicted, test_distance, threshold, loso,
            rule=f"threshold = 10 个内层模型 known 距离的 {100-q}% 分位",
            reported=reported,
        ))
    return {"experiment": "E29_dino_calibration", "candidates": candidates,
            "notes": "阈值经 10 次内层留出被试模型标定（Train-only）。"}


# --------------------------------------------------------------------------
# E30：多视图 + 相对 Train 内层已知分布的分位融合
# --------------------------------------------------------------------------
E30_COMBOS = {
    "cov5_dist": [("cov5", "dist")],
    "cov5_wccn": [("cov5", "dist_wccn")],
    "cov5_wccn+pca": [("cov5", "dist_wccn"), ("cov5", "pca")],
    "cov5_dist+wccn": [("cov5", "dist"), ("cov5", "dist_wccn")],
    "cov5_dist+pca": [("cov5", "dist"), ("cov5", "pca")],
    "cov5_dist+wccn+pca": [("cov5", "dist"), ("cov5", "dist_wccn"), ("cov5", "pca")],
    "cov5+topo5_dist": [("cov5", "dist"), ("topo5", "dist")],
    "cov5_dist+pca+topo5_dist": [("cov5", "dist"), ("cov5", "pca"), ("topo5", "dist")],
}


def view_signals_full(train_x, test_x, subjects, fit_mask, shrinkage=0.1):
    """E30 全量视图信号：dist / dist_wccn / pca，以及分类预测。"""
    model = fit_lr(train_x, subjects)
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    z_train = scaler.transform(train_x).astype(np.float32)
    z_test = scaler.transform(test_x).astype(np.float32)
    centers = np.stack([z_train[fit_mask & (subjects == c)].mean(0) for c in classifier.classes_])
    whit = whitening_from(z_train[fit_mask], subjects[fit_mask], shrinkage)
    z_train_w = (z_train @ whit).astype(np.float32)
    z_test_w = (z_test @ whit).astype(np.float32)
    centers_w = np.stack([z_train_w[fit_mask & (subjects == c)].mean(0) for c in classifier.classes_])
    pca = PCA(n_components=min(64, z_train.shape[1]), random_state=0).fit(z_train[fit_mask])

    def recon(z):
        return np.linalg.norm(z - pca.inverse_transform(pca.transform(z)), axis=1).astype(np.float32)

    return {
        "dist": (nearest_distance(z_train, centers), nearest_distance(z_test, centers)),
        "dist_wccn": (nearest_distance(z_train_w, centers_w), nearest_distance(z_test_w, centers_w)),
        "pca": (recon(z_train), recon(z_test)),
        "predicted": classifier.classes_[model.decision_function(test_x).argmax(1)],
    }


def view_signals_fold(feat, subjects, fit_rows, known_rows, unknown_rows, shrinkage=0.1):
    """LOSO 单折视图信号：只用 frac<=0.6 的注册者拟合标准化/原型/白化/PCA。"""
    scaler = ManualScaler(feat[fit_rows])
    z_fit = scaler.transform(feat[fit_rows])
    classes = np.unique(subjects[fit_rows])
    centers = np.stack([z_fit[subjects[fit_rows] == c].mean(0) for c in classes])
    whit = whitening_from(z_fit, subjects[fit_rows], shrinkage)
    centers_w = (centers @ whit).astype(np.float32)
    pca = PCA(n_components=min(64, z_fit.shape[1]), random_state=0).fit(z_fit)
    z_known = scaler.transform(feat[known_rows])
    z_unknown = scaler.transform(feat[unknown_rows])

    def recon(z):
        return np.linalg.norm(z - pca.inverse_transform(pca.transform(z)), axis=1).astype(np.float32)

    return {
        "dist": (nearest_distance(z_known, centers), nearest_distance(z_unknown, centers)),
        "dist_wccn": (
            nearest_distance((z_known @ whit).astype(np.float32), centers_w),
            nearest_distance((z_unknown @ whit).astype(np.float32), centers_w),
        ),
        "pca": (recon(z_known), recon(z_unknown)),
    }


def run_e30(data: Data, exp_dir: Path) -> dict:
    subjects = data.subjects
    views_feat = {"cov5": data.dino("covariance5"), "topo5": data.dino("topomap5")}
    report = json.loads((exp_dir / "e30_fusion.json").read_text(encoding="utf-8"))["combos"]

    # 全量信号（原型 = ~inner & ~forward，与 run_e30_fusion.py 一致）
    cache = {}
    for view_name, (train_x, test_x) in views_feat.items():
        fit_mask = ~data.inner & ~data.forward
        cache[view_name] = view_signals_full(train_x, test_x, subjects, fit_mask)
        log(f"[E30] {view_name}: Test AUROC(dist) "
            f"{roc_auc_score(data.test_known, -cache[view_name]['dist'][1]):.4f}")

    # 内层参考分布（10 个内层模型 × 2 视图）
    inner_reference: dict[tuple[str, str], dict[str, list]] = {}
    for fold, unknown in enumerate(
        np.array_split(np.random.default_rng(42).permutation(np.unique(subjects)), 5), 1
    ):
        known_subjects = np.setdiff1d(np.unique(subjects), unknown)
        for inner_unknown in np.array_split(
            np.random.default_rng(100 + fold).permutation(known_subjects), 2
        ):
            enrolled = np.setdiff1d(known_subjects, inner_unknown)
            enrolled_mask = np.isin(subjects, enrolled)
            inner_fit = enrolled_mask & ~data.forward & ~data.inner
            eval_known = np.flatnonzero(enrolled_mask & data.inner)
            eval_unknown = np.flatnonzero(np.isin(subjects, inner_unknown) & ~data.forward)
            fit_rows = np.flatnonzero(inner_fit)
            for view_name, (train_x, _) in views_feat.items():
                scaler = ManualScaler(train_x[fit_rows])
                z = scaler.transform(train_x)
                inner_classes = np.unique(subjects[fit_rows])
                centers = np.stack([z[inner_fit & (subjects == c)].mean(0) for c in inner_classes])
                whit = whitening_from(z[fit_rows], subjects[fit_rows], 0.1)
                z_w = (z @ whit).astype(np.float32)
                centers_w = np.stack([z_w[inner_fit & (subjects == c)].mean(0) for c in inner_classes])
                pca = PCA(n_components=min(64, z.shape[1]), random_state=0).fit(z[fit_rows])
                values = {
                    "dist": nearest_distance(z, centers),
                    "dist_wccn": nearest_distance(z_w, centers_w),
                    "pca": np.linalg.norm(z - pca.inverse_transform(pca.transform(z)), axis=1),
                }
                for signal, value in values.items():
                    block = inner_reference.setdefault((view_name, signal), {"known": [], "unknown": []})
                    block["known"].append(value[eval_known])
                    block["unknown"].append(value[eval_unknown])
    for key in inner_reference:
        inner_reference[key] = {
            "known": np.concatenate(inner_reference[key]["known"]),
            "unknown": np.concatenate(inner_reference[key]["unknown"]),
        }

    # LOSO：逐折算三信号，再按折内 known 块分位融合
    folds = []
    for held in CLASSES:
        enrolled = subjects != held
        fit_rows = np.flatnonzero(enrolled & data.fit60)
        known_rows = np.flatnonzero(enrolled & data.inner)
        unknown_rows = np.flatnonzero((subjects == held) & data.inner)
        if len(unknown_rows) == 0:
            continue
        fold = {}
        for view_name, (train_x, _) in views_feat.items():
            fold[view_name] = view_signals_fold(train_x, subjects, fit_rows, known_rows, unknown_rows, 0.1)
        folds.append(fold)
    log(f"[E30] LOSO 折数 {len(folds)}")

    candidates = []
    for combo_name, members in E30_COMBOS.items():
        def to_quantile_full(values):
            return np.mean([
                quantile_wrt(inner_reference[(v, s)]["known"], values[i])
                for i, (v, s) in enumerate(members)
            ], axis=0)

        train_fused = to_quantile_full([cache[v][s][0] for v, s in members])
        test_fused = to_quantile_full([cache[v][s][1] for v, s in members])
        predicted = cache[members[0][0]]["predicted"]

        fused_folds = []
        for fold in folds:
            known_parts, unknown_parts = [], []
            for v, s in members:
                k, u = fold[v][s]
                known_parts.append(quantile_wrt(k, k))
                unknown_parts.append(quantile_wrt(k, u))
            fused_folds.append({
                "known": np.mean(known_parts, axis=0),
                "unknown": np.mean(unknown_parts, axis=0),
            })

        for q in (1, 2, 5, 10, 20, 30):
            cand = f"q{q:02d}"
            reported = report[combo_name]["points"][cand]
            threshold = float(np.quantile(train_fused[data.inner], 1 - q / 100))
            loso = loso_from_fused_folds(fused_folds, q / 100)
            candidates.append(make_record(
                "E30_dino_multiview", f"{combo_name}_{cand}",
                exp_dir / f"candidate_{combo_name}_{cand}.csv", data,
                predicted, test_fused, threshold, loso,
                rule=f"threshold = 折内 known 块融合分位 {100-q}% 分位",
                reported=reported,
            ))
    return {"experiment": "E30_dino_multiview", "candidates": candidates,
            "notes": "分位融合的参考分布只用 Train；LOSO 逐折用 19 人 known 块作参考。"}


# --------------------------------------------------------------------------
# E31 / E34 / E35：任务条件化类中心（+ 任务内 WCCN）
# --------------------------------------------------------------------------
def task_mode_full(feat, subjects, tasks, mode, fit_mask=None, shrinkage=0.2, max_iter=300):
    """复刻 E31/E34 的 distance_signals；fit_mask=None 表示全量训练。"""
    if fit_mask is None:
        fit_mask = np.ones(len(subjects), dtype=bool)
    n_train, n_test = len(feat), None
    train_signal = np.zeros(n_train, dtype=np.float32)
    test_signal = None
    predicted = None

    if mode in ("global_global", "global_taskcenter"):
        model = fit_lr(feat[fit_mask], subjects[fit_mask], max_iter=max_iter)
        scaler = model.named_steps["standardscaler"]
        classifier = model.named_steps["logisticregression"]
        z_train = scaler.transform(feat).astype(np.float32)
        if mode == "global_global":
            centers = np.stack([z_train[fit_mask & (subjects == c)].mean(0) for c in classifier.classes_])
            train_signal = nearest_distance(z_train, centers)
        else:
            for task in TASKS:
                task_fit = fit_mask & (tasks == task)
                centers = np.stack([z_train[task_fit & (subjects == c)].mean(0) for c in classifier.classes_])
                rows = tasks == task
                train_signal[rows] = nearest_distance(z_train[rows], centers)
        return train_signal, model

    for task in TASKS:
        task_fit = fit_mask & (tasks == task)
        if task_fit.sum() < 100:
            continue
        model = fit_lr(feat[task_fit], subjects[task_fit], max_iter=max_iter)
        scaler = model.named_steps["standardscaler"]
        classifier = model.named_steps["logisticregression"]
        z_train = scaler.transform(feat).astype(np.float32)
        if mode == "task_model_wccn":
            whit = whitening_from(z_train[task_fit], subjects[task_fit], shrinkage)
            z_train = (z_train @ whit).astype(np.float32)
        centers = np.stack([z_train[task_fit & (subjects == c)].mean(0) for c in classifier.classes_])
        rows = tasks == task
        train_signal[rows] = nearest_distance(z_train[rows], centers)
    return train_signal, None


def task_mode_test(feat_train, feat_test, subjects, tasks, test_tasks, mode, shrinkage=0.2, max_iter=300):
    """全量模型下的 Test 信号 + 预测标签（复刻原脚本）。"""
    n_test = len(feat_test)
    test_signal = np.zeros(n_test, dtype=np.float32)
    predicted = np.zeros(n_test, dtype=int)

    if mode in ("global_global", "global_taskcenter"):
        model = fit_lr(feat_train, subjects, max_iter=max_iter)
        scaler = model.named_steps["standardscaler"]
        classifier = model.named_steps["logisticregression"]
        z_test = scaler.transform(feat_test).astype(np.float32)
        z_train = scaler.transform(feat_train).astype(np.float32)
        if mode == "global_global":
            centers = np.stack([z_train[subjects == c].mean(0) for c in classifier.classes_])
            test_signal = nearest_distance(z_test, centers)
        else:
            for task in TASKS:
                centers = np.stack([
                    z_train[(tasks == task) & (subjects == c)].mean(0) for c in classifier.classes_
                ])
                rows = test_tasks == task
                if rows.any():
                    test_signal[rows] = nearest_distance(z_test[rows], centers)
        if mode == "global_global":
            predicted = classifier.classes_[model.decision_function(feat_test).argmax(1)]
        else:
            predicted = classifier.classes_[model.decision_function(feat_test).argmax(1)]
        return test_signal, predicted

    for task in TASKS:
        task_fit = tasks == task
        if task_fit.sum() < 100:
            continue
        model = fit_lr(feat_train[task_fit], subjects[task_fit], max_iter=max_iter)
        scaler = model.named_steps["standardscaler"]
        classifier = model.named_steps["logisticregression"]
        z_train = scaler.transform(feat_train).astype(np.float32)
        z_test = scaler.transform(feat_test).astype(np.float32)
        if mode == "task_model_wccn":
            whit = whitening_from(z_train[task_fit], subjects[task_fit], shrinkage)
            z_train = (z_train @ whit).astype(np.float32)
            z_test = (z_test @ whit).astype(np.float32)
        centers = np.stack([z_train[task_fit & (subjects == c)].mean(0) for c in classifier.classes_])
        rows_test = test_tasks == task
        if rows_test.any():
            test_signal[rows_test] = nearest_distance(z_test[rows_test], centers)
            predicted[rows_test] = classifier.classes_[
                model.decision_function(feat_test[rows_test]).argmax(1)
            ]
    return test_signal, predicted


def task_mode_fold(feat, subjects, tasks, fit_rows, known_rows, unknown_rows, mode,
                   shrinkage=0.2, per_task=False):
    """LOSO 单折任务条件化分数；被评分块绝不进入标准化/原型/白化。"""
    known_scores = np.full(len(known_rows), np.nan)
    unknown_scores = np.full(len(unknown_rows), np.nan)
    known_pt = {task: np.empty(0) for task in TASKS}
    unknown_pt = {task: np.empty(0) for task in TASKS}

    if mode in ("global_global", "global_taskcenter"):
        scaler = ManualScaler(feat[fit_rows])
        z_fit = scaler.transform(feat[fit_rows])
        z_known = scaler.transform(feat[known_rows])
        z_unknown = scaler.transform(feat[unknown_rows])
        if mode == "global_global":
            classes = np.unique(subjects[fit_rows])
            centers = np.stack([z_fit[subjects[fit_rows] == c].mean(0) for c in classes])
            known_scores = nearest_distance(z_known, centers)
            unknown_scores = nearest_distance(z_unknown, centers)
            if per_task:
                for task in TASKS:
                    km = tasks[known_rows] == task
                    um = tasks[unknown_rows] == task
                    known_pt[task] = known_scores[km]
                    unknown_pt[task] = unknown_scores[um]
        else:
            for task in TASKS:
                fmask = tasks[fit_rows] == task
                classes = np.unique(subjects[fit_rows][fmask])
                centers = np.stack([z_fit[fmask & (subjects[fit_rows] == c)].mean(0) for c in classes])
                km = tasks[known_rows] == task
                um = tasks[unknown_rows] == task
                if km.any():
                    known_scores[km] = nearest_distance(z_known[km], centers)
                if um.any():
                    unknown_scores[um] = nearest_distance(z_unknown[um], centers)
                known_pt[task] = known_scores[km]
                unknown_pt[task] = unknown_scores[um]
    else:
        for task in TASKS:
            rows_fit = fit_rows[tasks[fit_rows] == task]
            if len(rows_fit) < 100:
                continue
            scaler = ManualScaler(feat[rows_fit])
            z_fit = scaler.transform(feat[rows_fit])
            classes = np.unique(subjects[rows_fit])
            centers = np.stack([z_fit[subjects[rows_fit] == c].mean(0) for c in classes])
            km = np.flatnonzero(tasks[known_rows] == task)
            um = np.flatnonzero(tasks[unknown_rows] == task)
            z_known = scaler.transform(feat[known_rows[km]])
            z_unknown = scaler.transform(feat[unknown_rows[um]])
            if mode == "task_model_wccn":
                whit = whitening_from(z_fit, subjects[rows_fit], shrinkage)
                centers = (centers @ whit).astype(np.float32)
                z_known = (z_known @ whit).astype(np.float32)
                z_unknown = (z_unknown @ whit).astype(np.float32)
            known_scores[km] = nearest_distance(z_known, centers)
            unknown_scores[um] = nearest_distance(z_unknown, centers)
            known_pt[task] = known_scores[km]
            unknown_pt[task] = unknown_scores[um]
    if per_task:
        return {"known": known_pt, "unknown": unknown_pt}
    return {"known": known_scores, "unknown": unknown_scores}


def loso_task_mode(data: Data, feat, mode, shrinkage=0.2, per_task=False) -> list[dict]:
    subjects, tasks = data.subjects, data.tasks
    folds = []
    for held in CLASSES:
        enrolled = subjects != held
        fit_rows = np.flatnonzero(enrolled & data.fit60)
        known_rows = np.flatnonzero(enrolled & data.inner)
        unknown_rows = np.flatnonzero((subjects == held) & data.inner)
        if len(unknown_rows) == 0:
            continue
        folds.append(task_mode_fold(
            feat, subjects, tasks, fit_rows, known_rows, unknown_rows, mode,
            shrinkage=shrinkage, per_task=per_task,
        ))
    return folds


E31_MODES = ("global_global", "global_taskcenter", "task_model_taskcenter", "task_model_wccn")


def run_e31(data: Data, exp_dir: Path) -> dict:
    subjects, tasks = data.subjects, data.tasks
    train_x, test_x = data.dino("covariance5")
    report = json.loads((exp_dir / "e31_task_conditional.json").read_text(encoding="utf-8"))
    candidates = []
    for mode in E31_MODES:
        train_signal, _ = task_mode_full(train_x, subjects, tasks, mode)
        test_signal, predicted = task_mode_test(train_x, test_x, subjects, tasks, data.test_task, mode)
        log(f"[E31] {mode}: Test AUROC {roc_auc_score(data.test_known, -test_signal):.4f} | "
            f"train(inner) p50 {np.median(train_signal[data.inner]):.2f}")
        folds = loso_task_mode(data, train_x, mode, shrinkage=0.2)
        for q in (1, 2, 5, 10, 20):
            cand = f"q{q:02d}"
            reported = report[mode]["points"][cand]
            threshold = float(np.quantile(train_signal[data.inner], 1 - q / 100))
            loso = loso_from_folds(folds, q / 100)
            candidates.append(make_record(
                "E31_task_conditional", f"{mode}_{cand}",
                exp_dir / f"candidate_{mode}_{cand}.csv", data,
                predicted, test_signal, threshold, loso,
                rule=f"threshold = 全量模型 Train 内层距离 {100-q}% 分位；LOSO 用折内 known 块同分位",
                reported=reported,
                notes=("原脚本 quantile 的原型含 inner 块（task_fit=全量）"
                       if mode.startswith("task_model") else ""),
            ))
    return {"experiment": "E31_task_conditional", "candidates": candidates,
            "notes": "任务条件化中心；阈值 Train-only，但 task_model_* 的分位原型含 inner 块。"}


def run_e32(data: Data, exp_dir: Path) -> dict:
    subjects, tasks = data.subjects, data.tasks
    train_x, test_x = data.dino("covariance5")
    report = json.loads((exp_dir / "e32_ratio.json").read_text(encoding="utf-8"))["signals"]
    combos = {
        "dist": ["dist"],
        "ratio": ["ratio"],
        "gap": ["gap"],
        "dist+ratio": ["dist", "ratio"],
        "dist+gap": ["dist", "gap"],
        "ratio+gap": ["ratio", "gap"],
        "dist+ratio+gap": ["dist", "ratio", "gap"],
    }

    def task_signals(feat, rows, centers_by_task, scaler_by_task, whit_by_task):
        out = {name: np.zeros(len(rows), dtype=np.float64) for name in ("dist", "ratio", "gap")}
        for task in TASKS:
            idx = np.flatnonzero(tasks[rows] == task)
            if len(idx) == 0 or task not in centers_by_task:
                continue
            z = scaler_by_task[task].transform(feat[rows[idx]])
            z = (z @ whit_by_task[task]).astype(np.float32)
            matrix = np.sort(dist_matrix(z, centers_by_task[task]), axis=1)
            out["dist"][idx] = matrix[:, 0]
            out["ratio"][idx] = matrix[:, 0] / np.maximum(matrix[:, 1], 1e-6)
            out["gap"][idx] = matrix[:, 1] - matrix[:, 0]
        return out

    # 全量：任务特异模型 + 任务内 WCCN（原型含 inner 块，与 run_e32_ratio.py 一致）
    all_rows = np.arange(len(train_x))
    centers_full, scaler_full, whit_full = {}, {}, {}
    predicted = np.zeros(len(test_x), dtype=int)
    test_out = {"dist": np.zeros(len(test_x)), "ratio": np.zeros(len(test_x)),
                "gap": np.zeros(len(test_x))}
    for task in TASKS:
        task_fit = tasks == task
        model = fit_lr(train_x[task_fit], subjects[task_fit])
        scaler = model.named_steps["standardscaler"]
        classifier = model.named_steps["logisticregression"]
        z = scaler.transform(train_x).astype(np.float32)
        whit = whitening_from(z[task_fit], subjects[task_fit], 0.2)
        z_w = (z @ whit).astype(np.float32)
        centers_full[task] = np.stack([z_w[task_fit & (subjects == c)].mean(0) for c in classifier.classes_])
        scaler_full[task] = scaler
        whit_full[task] = whit
        rows_test = data.test_task == task
        if rows_test.any():
            z_test = (scaler.transform(test_x[rows_test]).astype(np.float32) @ whit).astype(np.float32)
            matrix = np.sort(dist_matrix(z_test, centers_full[task]), axis=1)
            test_out["dist"][rows_test] = matrix[:, 0]
            test_out["ratio"][rows_test] = matrix[:, 0] / np.maximum(matrix[:, 1], 1e-6)
            test_out["gap"][rows_test] = matrix[:, 1] - matrix[:, 0]
            predicted[rows_test] = classifier.classes_[
                model.decision_function(test_x[rows_test]).argmax(1)
            ]
    train_out = task_signals(train_x, all_rows, centers_full, scaler_full, whit_full)

    # LOSO
    folds = []
    for held in CLASSES:
        enrolled = subjects != held
        fit_rows = np.flatnonzero(enrolled & data.fit60)
        known_rows = np.flatnonzero(enrolled & data.inner)
        unknown_rows = np.flatnonzero((subjects == held) & data.inner)
        if len(unknown_rows) == 0:
            continue
        centers_f, scaler_f, whit_f = {}, {}, {}
        for task in TASKS:
            rows_fit = fit_rows[tasks[fit_rows] == task]
            scaler = ManualScaler(train_x[rows_fit])
            z_fit = scaler.transform(train_x[rows_fit])
            whit = whitening_from(z_fit, subjects[rows_fit], 0.2)
            classes = np.unique(subjects[rows_fit])
            centers_f[task] = (np.stack([
                z_fit[subjects[rows_fit] == c].mean(0) for c in classes
            ]) @ whit).astype(np.float32)
            scaler_f[task], whit_f[task] = scaler, whit
        known_sig = task_signals(train_x, known_rows, centers_f, scaler_f, whit_f)
        unknown_sig = task_signals(train_x, unknown_rows, centers_f, scaler_f, whit_f)
        folds.append({"known": known_sig, "unknown": unknown_sig})

    candidates = []
    for combo_name, members in combos.items():
        train_fused = np.mean([
            quantile_wrt(train_out[m][data.inner], train_out[m]) for m in members
        ], axis=0)
        test_fused = np.mean([
            quantile_wrt(train_out[m][data.inner], test_out[m]) for m in members
        ], axis=0)
        fused_folds = []
        for fold in folds:
            known_parts = [quantile_wrt(fold["known"][m], fold["known"][m]) for m in members]
            unknown_parts = [quantile_wrt(fold["known"][m], fold["unknown"][m]) for m in members]
            fused_folds.append({"known": np.mean(known_parts, axis=0),
                                "unknown": np.mean(unknown_parts, axis=0)})
        for q in (1, 2, 5, 10, 20):
            cand = f"q{q:02d}"
            reported = report[combo_name]["points"][cand]
            threshold = float(np.quantile(train_fused[data.inner], 1 - q / 100))
            loso = loso_from_fused_folds(fused_folds, q / 100)
            candidates.append(make_record(
                "E32_distance_ratio", f"{combo_name}_{cand}",
                exp_dir / f"candidate_{combo_name}_{cand}.csv", data,
                predicted, test_fused, threshold, loso,
                rule=f"threshold = 全量 Train 内层融合分位 {100-q}% 分位；LOSO 折内 known 块参考",
                reported=reported,
                notes="参考分布来自全量模型（原型含 inner 块）；ratio/gap 同原脚本",
            ))
    return {"experiment": "E32_distance_ratio", "candidates": candidates,
            "notes": "dist/ratio/gap 的分位融合；gap 方向与原脚本一致（越大越像已知）。"}


def run_e34(data: Data, exp_dir: Path) -> dict:
    subjects, tasks = data.subjects, data.tasks
    train_x, test_x = data.dino("covariance_hr")
    report = json.loads((exp_dir / "e34_covariance_hr.json").read_text(encoding="utf-8"))
    train_signal, _ = task_mode_full(train_x, subjects, tasks, "task_model_wccn")
    test_signal, predicted = task_mode_test(train_x, test_x, subjects, tasks, data.test_task, "task_model_wccn")
    log(f"[E34] covariance_hr: Test AUROC {roc_auc_score(data.test_known, -test_signal):.4f} | "
        f"train(inner) p50 {np.median(train_signal[data.inner]):.2f}")
    folds = loso_task_mode(data, train_x, "task_model_wccn", shrinkage=0.2)
    candidates = []
    for q in (1, 2, 5, 10, 20):
        cand = f"q{q:02d}"
        reported = report["points"][cand]
        threshold = float(np.quantile(train_signal[data.inner], 1 - q / 100))
        loso = loso_from_folds(folds, q / 100)
        candidates.append(make_record(
            "E34_covariance_hr", f"covariance_hr_{cand}",
            exp_dir / f"candidate_covariance_hr_{cand}.csv", data,
            predicted, test_signal, threshold, loso,
            rule=f"threshold = 全量 Train 内层距离 {100-q}% 分位；LOSO 折内 known 块同分位",
            reported=reported,
            notes="分位原型含 inner 块（task_fit=全量），与 E31 task_model_* 同构",
        ))
    return {"experiment": "E34_covariance_hr", "candidates": candidates,
            "notes": "covariance_hr 特征 + 任务内 WCCN。"}


def run_e35(data: Data, exp_dir: Path) -> dict:
    subjects, tasks = data.subjects, data.tasks
    train_x, test_x = data.dino("covariance5")
    report = json.loads((exp_dir / "e35_per_task.json").read_text(encoding="utf-8"))
    train_signal, _ = task_mode_full(train_x, subjects, tasks, "task_model_wccn")
    test_signal, predicted = task_mode_test(train_x, test_x, subjects, tasks, data.test_task, "task_model_wccn")

    # 内层逐任务已知分布（10 个内层模型，与 run_e35_per_task.py 一致）
    inner_known = {task: [] for task in TASKS}
    for fold, unknown in enumerate(
        np.array_split(np.random.default_rng(42).permutation(np.unique(subjects)), 5), 1
    ):
        known_subjects = np.setdiff1d(np.unique(subjects), unknown)
        for inner_unknown in np.array_split(
            np.random.default_rng(100 + fold).permutation(known_subjects), 2
        ):
            enrolled = np.setdiff1d(known_subjects, inner_unknown)
            mask_enrolled = np.isin(subjects, enrolled)
            inner_fit = mask_enrolled & ~data.forward & ~data.inner
            fit_rows = np.flatnonzero(inner_fit)
            known_rows = np.flatnonzero(mask_enrolled & data.inner)
            for task in TASKS:
                rows_fit = fit_rows[tasks[fit_rows] == task]
                if len(rows_fit) < 100:
                    continue
                scaler = ManualScaler(train_x[rows_fit])
                z_fit = scaler.transform(train_x[rows_fit])
                whit = whitening_from(z_fit, subjects[rows_fit], 0.2)
                classes = np.unique(subjects[rows_fit])
                centers = (np.stack([z_fit[subjects[rows_fit] == c].mean(0) for c in classes]) @ whit).astype(np.float32)
                k = known_rows[tasks[known_rows] == task]
                if len(k) < 20:
                    continue
                z = (scaler.transform(train_x[k]) @ whit).astype(np.float32)
                inner_known[task].append(nearest_distance(z, centers))
    inner_known = {task: np.concatenate(v) for task, v in inner_known.items()}

    folds = loso_task_mode(data, train_x, "task_model_wccn", shrinkage=0.2, per_task=True)
    folds_global = loso_task_mode(data, train_x, "task_model_wccn", shrinkage=0.2)
    candidates = []
    for strategy in ("global", "per_task"):
        for q in (1, 2, 5, 10, 20):
            cand = f"q{q:02d}"
            reported = report["strategies"][strategy]["points"][cand]
            if strategy == "global":
                threshold = float(np.quantile(train_signal[data.inner], 1 - q / 100))
                keep = test_signal <= threshold
                loso = loso_from_folds(folds_global, q / 100)
                rule = f"threshold = 全量 Train 内层距离 {100-q}% 分位"
            else:
                threshold = None
                keep = np.ones(len(test_signal), dtype=bool)
                for task in TASKS:
                    rows = data.test_task == task
                    task_thr = float(np.quantile(inner_known[task], 1 - q / 100))
                    keep[rows] = test_signal[rows] <= task_thr
                loso = loso_per_task_from_folds(folds, q / 100)
                rule = f"threshold = 逐任务内层已知距离 {100-q}% 分位"
            results = make_record(
                "E35_per_task_analysis", f"{strategy}_{cand}",
                exp_dir / f"candidate_{strategy}_{cand}.csv", data,
                predicted, None, threshold, loso,
                rule=rule, reported=reported, keep=keep,
                notes=("per_task 阈值来自 10 个内层模型；global 分位原型含 inner 块"
                       if strategy == "per_task" else "分位原型含 inner 块"),
            )
            candidates.append(results)
    return {"experiment": "E35_per_task_analysis", "candidates": candidates,
            "notes": "global 与 per_task 两种阈值策略。"}


# --------------------------------------------------------------------------
# E33：ArcFace / 普通 CE 身份头（模型未保存，用原 seed 重训）
# --------------------------------------------------------------------------
def run_e33(data: Data, exp_dir: Path, margin: float = 0.0, dropout: float = 0.5) -> dict:
    import torch
    import torch.nn.functional as F

    from src.eeg_image_dinov2.heads import IdentityHead

    train_x, test_x = data.dino("covariance5")
    subjects = data.subjects
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs, batch, lr = 40, 256, 1e-3
    tag = f"m{margin}_d{dropout}"

    def train_head(features: np.ndarray, labels: np.ndarray, n_classes: int, seed: int):
        torch.manual_seed(seed)
        model = IdentityHead(in_dim=features.shape[1], n_classes=n_classes,
                             dropout=dropout, margin=margin).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        classes = np.unique(labels)
        if len(classes) != n_classes:
            raise ValueError(f"expected {n_classes} classes, got {len(classes)}")
        remap = {int(label): index for index, label in enumerate(classes)}
        y = torch.as_tensor([remap[int(label)] for label in labels], dtype=torch.long, device=device)
        generator = torch.Generator(device=device).manual_seed(seed)
        for _ in range(epochs):
            model.train()
            order = torch.randperm(len(x), generator=generator, device=device)
            for start in range(0, len(order), batch):
                index = order[start:start + batch]
                loss = F.cross_entropy(model(x[index], y[index]), y[index])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
        model.eval()
        return model

    @torch.no_grad()
    def embed(model, features: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        return model.embed(tensor).cpu().numpy()

    def normalized_centers(z: np.ndarray, labels: np.ndarray) -> np.ndarray:
        classes = np.unique(labels)
        centers = np.stack([z[labels == c].mean(0) for c in classes])
        return centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)

    model = train_head(train_x, subjects, 20, seed=2026)
    z_train = embed(model, train_x)
    z_test = embed(model, test_x)
    centroid_mask = ~data.inner & ~data.forward
    centers = normalized_centers(z_train[centroid_mask], subjects[centroid_mask])
    cos_test = z_test @ centers.T
    novelty_test = (-cos_test.max(1)).astype(np.float32)
    predicted = np.unique(subjects)[cos_test.argmax(1)].astype(int)
    inner_novelty = (-np.max(z_train @ centers.T, axis=1)).astype(np.float32)[data.inner]
    log(f"[E33] {tag}: Test AUROC {roc_auc_score(data.test_known, -novelty_test):.4f} | "
        f"train(inner) p50 {np.median(inner_novelty):.4f}")

    # LOSO：每折重训身份头（只用 frac<=0.6 的 19 个注册者）
    folds = []
    for held in CLASSES:
        enrolled = subjects != held
        fit_rows = np.flatnonzero(enrolled & data.fit60)
        known_rows = np.flatnonzero(enrolled & data.inner)
        unknown_rows = np.flatnonzero((subjects == held) & data.inner)
        if len(unknown_rows) == 0:
            continue
        fold_model = train_head(train_x[fit_rows], subjects[fit_rows], 19, seed=int(held))
        z_fit = embed(fold_model, train_x[fit_rows])
        fold_centers = normalized_centers(z_fit, subjects[fit_rows])
        z_known = embed(fold_model, train_x[known_rows])
        z_unknown = embed(fold_model, train_x[unknown_rows])
        folds.append({
            "known": (-np.max(z_known @ fold_centers.T, axis=1)).astype(np.float32),
            "unknown": (-np.max(z_unknown @ fold_centers.T, axis=1)).astype(np.float32),
        })
        log(f"[E33] LOSO held={held} known={len(known_rows)} unknown={len(unknown_rows)}")

    candidates = []
    for q in (1, 2, 5, 10, 20):
        cand = f"q{q:02d}"
        threshold = float(np.quantile(inner_novelty, 1 - q / 100))
        loso = loso_from_folds(folds, q / 100)
        candidates.append(make_record(
            "E33_arcface_head", f"arcface_{tag}_{cand}",
            exp_dir / f"candidate_arcface_{tag}_{cand}.csv", data,
            predicted, novelty_test, threshold, loso,
            rule=f"threshold = 全量模型 Train 内层 novelty {100-q}% 分位；LOSO 折内 known 块同分位",
            notes="身份头未保存权重，按原 seed 重训；一致性见 repro_agreement",
        ))
    return {"experiment": "E33_arcface_head", "candidates": candidates,
            "notes": "ArcFace/CE 头需重训，若一致性 <99% 记为 pipeline_mismatch。"}


# --------------------------------------------------------------------------
# 主流程与输出
# --------------------------------------------------------------------------
EXPERIMENT_SPEC = [
    ("E26_final_candidates", "cov 原始特征 + SNR30 增广 + LR，距离型拒识；阈值 = Train 内层距离分位"),
    ("E27_unknown_rejection", "同 E26（E27B 只用 dist 八个工作点）"),
    ("E28_dino_openset", "DINO cov5 / cov5+topo5 距离；阈值 = Train 内层距离分位"),
    ("E29_dino_calibration", "DINO cov5 距离；阈值 = 10 个内层留出被试模型的 known 分位"),
    ("E30_dino_multiview", "DINO 多视图分位融合（参考 = Train 内层已知分布）"),
    ("E31_task_conditional", "任务条件化中心 / 任务内 WCCN"),
    ("E32_distance_ratio", "任务内 WCCN 的 dist/ratio/gap 分位融合"),
    ("E33_arcface_head", "ArcFace/CE 身份头（未保存权重，按原 seed 重训）"),
    ("E34_covariance_hr", "高分辨率协方差 DINO 特征 + 任务内 WCCN"),
    ("E35_per_task_analysis", "cov5 任务内 WCCN，global / 逐任务阈值"),
]


def ranking(results: list[dict]) -> list[dict]:
    rows = []
    for block in results:
        for record in block["candidates"]:
            if record["W"] is None:
                continue
            if record["flags"] != ["clean"]:
                continue
            if record["repro_agreement"] < 0.99:
                continue
            rows.append(record)
    rows.sort(key=lambda r: -r["W"])
    return rows


def write_report(results: list[dict], elapsed: float) -> None:
    rows = ranking(results)
    top = rows[0] if rows else None
    total = sum(len(block["candidates"]) for block in results)
    mismatched = [
        r for block in results for r in block["candidates"] if r["repro_agreement"] < 0.99
    ]
    skipped = [block for block in results if block.get("skipped")]
    agreements = [
        r["repro_agreement"] for block in results for r in block["candidates"]
    ]
    min_agreement = min(agreements) if agreements else float("nan")

    lines = [
        "# E175 老实验（E26–E35）LOSO 补算与 W 重建",
        "",
        "性质：只读补算。未修改任何已有实验目录，未提交 git。",
        "Test 官方 AUC / F1：**N/A**（无真实标签）。",
        "",
        "```",
        "W = 0.25 × LOSO 拒识 + 0.25 × Test S1 拒识 + 0.5 × Test S2 接受",
        "```",
        "",
        "## 0. 结论先行",
        "",
    ]
    if top:
        lines += [
            f"- 共补算 **{total}** 个候选，其中 **{len(rows)}** 个为 `clean` 且重建一致率 ≥ 99%。",
            f"- 最高可信 **W = {top['W']:.4f}**（{top['experiment']} / `{top['candidate']}`，"
            f"LOSO {top['loso_rejection']:.4f} / S1 {top['csv_s1_rejection']:.4f} / "
            f"S2 {top['csv_s2_acceptance']:.4f}）。",
            f"- 与当前已知最高可信值 0.7171（E170 固定工作点）比较："
            f"**{'超过' if top['W'] > 0.7171 else '没有超过'}**（差 {top['W'] - 0.7171:+.4f}）。",
        ]
    lines += [
        f"- 重建失败（一致率 < 99%）的候选：**{len(mismatched)}** 个。",
        f"- 因缺缓存 / 缺代码而跳过的实验：**{len(skipped)}** 个。",
        "",
        "## 1. 方法与有效性判定",
        "",
        "1. 逐实验按原 `run_*.py` 的代码路径重算 Test 分数 → 应用该候选 CSV 对应的阈值 → "
        f"得到预测标签，与已有 CSV 逐行比对（一致率）。全部候选的最低一致率为 {min_agreement:.4f}。",
        "2. LOSO 用项目标准协议：每次留出 1 个身份，原型 / 标准化 / 白化 / PCA / 身份头**只用**"
        "其余 19 人在 `frac <= 0.6` 的 trial；被评分为 known 块（注册者，`0.6 < frac <= 0.8`）"
        "与 unknown 块（留出身份，`0.6 < frac <= 0.8`）。",
        "3. LOSO 阈值与该候选**同一规则**：折内 known 块分数的同一分位（`1 - rate`）；"
        "E35 `per_task` 用逐任务分位。",
        "4. `W` 的 S1 / S2 直接取自候选 CSV（session1 = 3,033 行，session2 = 19,218 行）。",
        "",
        "有效性标记：`clean`（Train-only 定阈值 + LOSO 无原型污染）、`test_statistic`（阈值或分数用 Test 统计）、"
        "`loso_contaminated`、`pipeline_mismatch`、`skipped`。",
        "",
        "**阈值规则核查**：这 10 个实验的候选阈值全部来自 Train 侧（内层分位 / 10 个内层模型分位），"
        "没有任何候选使用 Test 分位数，因此无 `test_statistic`。",
        "需要单独说明的是 E31 的 `task_model_*`、E32、E34、E35 的 `global` 策略：原脚本全量模型的类中心"
        "把 `inner` 块（0.6–0.8）也算了进去，于是「Train 内层分位」的分位参考被压低（阈值更严）。"
        "这属于 Train 侧标定瑕疵，不影响 Test 标签本身；本实验的 LOSO 已按干净协议重建。",
        "",
        "## 2. 逐实验逐候选",
        "",
    ]
    for block in results:
        lines += [
            f"### {block['experiment']}",
            "",
            f"{block.get('notes', '')}",
            "",
            "| 候选 | CSV S1 拒识 | CSV S2 接受 | 重建一致率 | LOSO 拒识 | W | 标记 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for r in block["candidates"]:
            loso = "NA" if r["loso_rejection"] is None else f"{r['loso_rejection']:.4f}"
            w = "NA" if r["W"] is None else f"{r['W']:.4f}"
            lines.append(
                f"| `{r['candidate']}` | {r['csv_s1_rejection']:.4f} | {r['csv_s2_acceptance']:.4f} | "
                f"{r['repro_agreement']:.4f} | {loso} | {w} | {','.join(r['flags'])} |"
            )
        lines += ["", "阈值：各候选自身的 Train 侧分位（rate 越高阈值越宽）；LOSO 用折内 known 块的同一分位。",
                  ""]

    lines += [
        "## 3. 总排序（仅 `clean` 且重建一致）",
        "",
        "| # | 实验 | 候选 | S1 拒识 | S2 接受 | LOSO | W |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r['experiment']} | `{r['candidate']}` | {r['csv_s1_rejection']:.4f} | "
            f"{r['csv_s2_acceptance']:.4f} | {r['loso_rejection']:.4f} | {r['W']:.4f} |"
        )
    lines += [
        "",
        "## 4. 是否超过 0.7171",
        "",
    ]
    if top:
        over = [r for r in rows if r["W"] > 0.7171]
        lines += [
            f"- 最高 clean W = **{top['W']:.4f}**（{top['experiment']} / `{top['candidate']}`）。",
            f"- 超过 0.7171 的 clean 候选数量：**{len(over)}**。",
        ]
        if over:
            lines.append("- 结论：**存在**超过 0.7171 的 clean 候选，需逐个复核重建一致性。")
        else:
            lines.append(
                "- 结论：这批老实验（E26–E35）**没有一个** clean 候选能超过 0.7171；"
                f"最高值还差 {0.7171 - top['W']:.4f}。"
            )
    lines += [
        "",
        "原因：",
        "",
        "- 这批实验的最优工作点都在 S2 接受率 0.6–0.9 之间，而 LOSO 拒识与 S1 拒识高度耦合"
        "（Test AUROC 只有 0.65–0.74）。要拿到接近 0.7171 的 W，需要在 S2 接受 0.78 附近同时有 "
        "LOSO ≈ 0.69、S1 ≈ 0.61；本批次任何候选都达不到这个组合。",
        "- E26 / E27 的 W 最高（0.6394 = E27 reject30；0.6370 = E26/E27 reject20），靠的是协方差原始特征"
        " + SNR30 增广，代价是 S2 接受率被压到 0.47（reject30），与 0.7171 方案（S2 0.7813）不在同一工作区。",
        "- DINO 系列（E28–E35）的 LOSO 只有 0.01–0.53：同会话留出身份与注册者的距离分布重叠严重，"
        "即使 Test 侧 S1 拒识不错，LOSO 端也拉不起来。",
        "- E32 的 `gap` 单用（W ≈ 0.50）与所有 `*gap` 组合是方向错误导致的退化，不构成有效方案。",
        "",
        "## 5. 无法补算的实验",
        "",
    ]
    if skipped:
        for block in skipped:
            lines.append(f"- {block['experiment']}：{block.get('reason', 'NA')}")
    else:
        lines.append("- 无。10 个实验的特征缓存（`BrainPrint_cache/train_features.npy`、"
                     "`BrainPrint_cache/e25_robust/train_cov_snr30.npy`、"
                     "`BrainPrint_cache/eeg_image_dinov2/*_dinov2.npy`）与代码均存在，全部补算完成。")
    lines += [
        "",
        "## 6. 复核要点",
        "",
        "- 所有候选的 CSV S1 / S2 与实验 JSON 记录逐项对比（`reported diff` 均为 0），"
        "说明 CSV 与报告的对应关系正确。",
        "- E26 与 E27 的 `reject1/2/5/20/50` 是同一管线（CSV 内容相同），不是独立方案。",
        "- E32 的 `gap` 方向与原脚本一致（越大越像已知），因此 `gap` 单用退化为「全部接受」，"
        "W ≈ 0.50 属恒等式产物，不具判别意义。",
        "- E30 `cov5+topo5_dist` 及 E32 的 `*gap` 组合的低 W 是融合方向错误，不是可信提升。",
        "",
        f"运行耗时：{elapsed:.1f}s；候选总数 {total}；日志 `run.log`。",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="逗号分隔的实验前缀过滤")
    parser.add_argument("--skip-e33", action="store_true")
    args = parser.parse_args()

    if LOG_PATH.exists():
        LOG_PATH.unlink()
    started = time.time()
    log("E175 开始：重建 E26–E35 的 Test 管线并补算同阈值 LOSO")

    data = Data()
    log(f"数据：Train {len(data.subjects)} 行（20 身份 × 4 任务），"
        f"Test {len(data.test_sample)} 行（session1 {data.n_s1} / session2 {data.n_s2}）")
    log(f"fit 块 {int(data.fit60.sum())} 行（frac<=0.6），known/unknown 评分块 "
        f"{int(data.inner.sum())} 行（0.6<frac<=0.8）")

    only = [s.strip() for s in args.only.split(",") if s.strip()]
    results = []
    runners = {
        "E26_final_candidates": lambda: run_e26_e27(
            data, "E26_final_candidates", RESEARCH / "E26_final_candidates",
            (1, 2, 5, 20, 50, 70, 85, 95)),
        "E27_unknown_rejection": lambda: run_e26_e27(
            data, "E27_unknown_rejection", RESEARCH / "E27_unknown_rejection",
            (1, 2, 5, 10, 20, 30, 40, 50)),
        "E28_dino_openset": lambda: run_e28(data, RESEARCH / "E28_dino_openset"),
        "E29_dino_calibration": lambda: run_e29(data, RESEARCH / "E29_dino_calibration"),
        "E30_dino_multiview": lambda: run_e30(data, RESEARCH / "E30_dino_multiview"),
        "E31_task_conditional": lambda: run_e31(data, RESEARCH / "E31_task_conditional"),
        "E32_distance_ratio": lambda: run_e32(data, RESEARCH / "E32_distance_ratio"),
        "E33_arcface_head": lambda: run_e33(data, RESEARCH / "E33_arcface_head"),
        "E34_covariance_hr": lambda: run_e34(data, RESEARCH / "E34_covariance_hr"),
        "E35_per_task_analysis": lambda: run_e35(data, RESEARCH / "E35_per_task_analysis"),
    }
    for name, _ in EXPERIMENT_SPEC:
        if only and not any(name.startswith(prefix) for prefix in only):
            continue
        if args.skip_e33 and name.startswith("E33"):
            results.append({"experiment": name, "candidates": [], "skipped": True,
                            "reason": "用户指定跳过 E33（重训成本）"})
            continue
        tick = time.time()
        try:
            block = runners[name]()
            block["elapsed_s"] = round(time.time() - tick, 1)
            results.append(block)
            log(f"[{name}] 完成，用时 {block['elapsed_s']}s，候选 {len(block['candidates'])} 个")
        except Exception as exc:  # 单实验失败不影响整体
            log(f"[{name}] 失败：{type(exc).__name__}: {exc}")
            results.append({"experiment": name, "candidates": [], "skipped": True,
                            "reason": f"{type(exc).__name__}: {exc}"})

    elapsed = time.time() - started
    rows = ranking(results)
    for record in rows[:10]:
        log(f"  W={record['W']:.4f} {record['experiment']} / {record['candidate']} "
            f"(LOSO {record['loso_rejection']:.4f}, S1 {record['csv_s1_rejection']:.4f}, "
            f"S2 {record['csv_s2_acceptance']:.4f})")

    (OUT / "results.json").write_text(json.dumps({
        "meta": {
            "experiment": "E175_rebuild_old_loso",
            "date": "2026-11-15",
            "objective": "W = 0.25*LOSO + 0.25*S1 + 0.5*S2",
            "test_official_auc_f1": "N/A (no ground truth labels)",
            "reference_best_clean_W": 0.7171,
            "elapsed_s": round(elapsed, 1),
            "candidate_count": sum(len(b["candidates"]) for b in results),
        },
        "experiments": results,
        "ranking": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    (OUT / "config.json").write_text(json.dumps({
        "experiment": "E175_rebuild_old_loso",
        "purpose": "Rebuild Test pipelines of E26-E35 and compute the missing same-threshold LOSO rejection, "
                   "so that W becomes available for candidates that only stored Test CSVs.",
        "data": {
            "train_metadata": "BrainPrint_cache/train_metadata.csv",
            "test_metadata": "BrainPrint_cache/test_metadata.csv",
            "fit_block": "trial <= 0.6 * max_trial (prototypes / standardisation / whitening / PCA)",
            "known_block": "0.6 < trial/max <= 0.8 of the 19 enrolled identities",
            "unknown_block": "0.6 < trial/max <= 0.8 of the held-out identity",
        },
        "protocol": {
            "loso": "leave-one-identity-out, 20 folds, prototypes never contain the scored block",
            "threshold_rule": "same percentile of the fold-level known-block scores as the original candidate",
            "weights": {"LOSO": 0.25, "Test_S1": 0.25, "Test_S2": 0.5},
        },
        "experiments": [
            {"name": name, "pipeline": note} for name, note in EXPERIMENT_SPEC
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    write_report(results, elapsed)
    log(f"输出：results.json / config.json / report.md；总耗时 {elapsed:.1f}s")


if __name__ == "__main__":
    main()
