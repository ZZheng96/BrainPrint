"""E30：DINO 多视图距离融合，提高 S1/S2 可分性（AUROC）。

E29 结论：cov5 的 dist AUROC=0.7065，dist_wccn 0.7248，topo5 单独 0.5892。
本实验检验视图间的互补性：

  cov5_dist        到最近类中心的欧氏距离
  cov5_dist_wccn   cov5 类内白化后的距离
  cov5_pca         cov5 的 PCA 重构残差
  topo5_dist       topomap5 的距离

融合方式：对每个信号的 Train 内层已知/伪未知分布做经验 CDF 变换（逐信号转成「分位数」），
再等权平均。CDF 参数只用训练集估计，逐样本施加，不使用 Test 统计。

信号与权重选择只用 Train 内层伪未知的 AUROC，Test 只做审计。
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
QUANTILES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30)


def nearest_distance(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(data**2, axis=1, keepdims=True)
        - 2 * data @ centroids.T
        + np.sum(centroids**2, axis=1)[None, :]
    )
    return np.sqrt(np.maximum(squared.min(axis=1), 0.0))


def within_class_whitening(train_z: np.ndarray, labels: np.ndarray, shrinkage: float = 0.1):
    classes = np.unique(labels)
    centered = np.zeros_like(train_z)
    for label in classes:
        mask = labels == label
        centered[mask] = train_z[mask] - train_z[mask].mean(0)
    scatter = centered.T @ centered / max(len(train_z) - len(classes), 1)
    scatter = (1 - shrinkage) * scatter + shrinkage * np.trace(scatter) / scatter.shape[0] * np.eye(scatter.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    return (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T


def fit(train_x, labels):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300))
    model.fit(train_x, labels)
    return model


def view_signals(train_x, test_x, subjects, fit_mask, eval_known, eval_unknown):
    """在同一视图上算四种新颖度信号，返回 (train_all, test_all) 及内层 AUROC。"""
    model = fit(train_x[fit_mask], subjects[fit_mask])
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    z_train = scaler.transform(train_x).astype(np.float32)
    z_test = scaler.transform(test_x).astype(np.float32)
    centers = np.stack([z_train[fit_mask & (subjects == c)].mean(0) for c in classifier.classes_])

    whitening = within_class_whitening(z_train[fit_mask], subjects[fit_mask])
    z_train_w = (z_train @ whitening).astype(np.float32)
    z_test_w = (z_test @ whitening).astype(np.float32)
    centers_w = np.stack([z_train_w[fit_mask & (subjects == c)].mean(0) for c in classifier.classes_])

    pca = PCA(n_components=min(64, z_train.shape[1]), random_state=0).fit(z_train[fit_mask])

    def recon(z):
        return np.linalg.norm(z - pca.inverse_transform(pca.transform(z)), axis=1)

    signals = {
        "dist": (nearest_distance(z_train, centers), nearest_distance(z_test, centers)),
        "dist_wccn": (nearest_distance(z_train_w, centers_w), nearest_distance(z_test_w, centers_w)),
        "pca": (recon(z_train), recon(z_test)),
    }
    aurocs = {}
    labels = np.r_[np.ones(len(eval_known)), np.zeros(len(eval_unknown))]
    for name, (train_value, _) in signals.items():
        column = train_value[np.r_[eval_known, eval_unknown]]
        aurocs[name] = float(roc_auc_score(labels, -column))
    return signals, aurocs


def main() -> None:
    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial"])
    subjects = metadata["subject"].to_numpy()
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    test_meta = pd.read_csv(CACHE / "test_metadata.csv", usecols=["sample", "session"])
    known = (test_meta["session"] == "session2").to_numpy()

    views = {
        "cov5": (
            np.load(DINO / "train_covariance5_dinov2.npy").astype(np.float32),
            np.load(DINO / "test_covariance5_dinov2.npy").astype(np.float32),
        ),
        "topo5": (
            np.load(DINO / "train_topomap5_dinov2.npy").astype(np.float32),
            np.load(DINO / "test_topomap5_dinov2.npy").astype(np.float32),
        ),
    }

    # 逐视图、逐信号：内层伪未知 AUROC + Test AUROC
    summary = {}
    inner_known_idx = np.flatnonzero(inner)
    for view_name, (train_x, test_x) in views.items():
        for fold, unknown in enumerate(np.array_split(np.random.default_rng(42).permutation(np.unique(subjects)), 5), 1):
            known_subjects = np.setdiff1d(np.unique(subjects), unknown)
            is_known = np.isin(subjects, known_subjects)
            fit_mask = is_known & ~forward
            for inner_unknown in np.array_split(np.random.default_rng(100 + fold).permutation(known_subjects), 2):
                enrolled = np.setdiff1d(known_subjects, inner_unknown)
                enrolled_mask = np.isin(subjects, enrolled)
                inner_fit = enrolled_mask & ~forward & ~inner
                eval_known = np.flatnonzero(enrolled_mask & inner)
                eval_unknown = np.flatnonzero(np.isin(subjects, inner_unknown) & ~forward)
                _, aurocs = view_signals(train_x, test_x, subjects, inner_fit, eval_known, eval_unknown)
                summary.setdefault(view_name, {}).setdefault("inner", []).append(aurocs)

    # 汇总内层 AUROC
    inner_mean = {}
    for view_name, block in summary.items():
        inner_mean[view_name] = {
            signal: float(np.mean([row[signal] for row in block["inner"]]))
            for signal in block["inner"][0]
        }
        print(f"{view_name}: 内层 AUROC {({k: round(v,4) for k,v in inner_mean[view_name].items()})}", flush=True)

    # 全量模型：内层选择信号组合
    combos = {
        "cov5_dist": [("cov5", "dist")],
        "cov5_wccn": [("cov5", "dist_wccn")],
        "cov5_wccn+pca": [("cov5", "dist_wccn"), ("cov5", "pca")],
        "cov5_dist+wccn": [("cov5", "dist"), ("cov5", "dist_wccn")],
        "cov5_dist+pca": [("cov5", "dist"), ("cov5", "pca")],
        "cov5_dist+wccn+pca": [("cov5", "dist"), ("cov5", "dist_wccn"), ("cov5", "pca")],
        "cov5+topo5_dist": [("cov5", "dist"), ("topo5", "dist")],
        "cov5_dist+pca+topo5_dist": [("cov5", "dist"), ("cov5", "pca"), ("topo5", "dist")],
    }

    # 全量拟合，缓存每个视图的 train/test 信号
    cache = {}
    for view_name, (train_x, test_x) in views.items():
        full = fit(train_x, subjects)
        scaler = full.named_steps["standardscaler"]
        classifier = full.named_steps["logisticregression"]
        centroid_mask = ~inner & ~forward
        z_train = scaler.transform(train_x).astype(np.float32)
        z_test = scaler.transform(test_x).astype(np.float32)
        centers = np.stack([z_train[centroid_mask & (subjects == c)].mean(0) for c in classifier.classes_])
        whitening = within_class_whitening(z_train[centroid_mask], subjects[centroid_mask])
        z_train_w = (z_train @ whitening).astype(np.float32)
        z_test_w = (z_test @ whitening).astype(np.float32)
        centers_w = np.stack([z_train_w[centroid_mask & (subjects == c)].mean(0) for c in classifier.classes_])
        pca = PCA(n_components=min(64, z_train.shape[1]), random_state=0).fit(z_train[centroid_mask])
        cache[view_name] = {
            "dist": (nearest_distance(z_train, centers), nearest_distance(z_test, centers)),
            "dist_wccn": (nearest_distance(z_train_w, centers_w), nearest_distance(z_test_w, centers_w)),
            "pca": (
                np.linalg.norm(z_train - pca.inverse_transform(pca.transform(z_train)), axis=1),
                np.linalg.norm(z_test - pca.inverse_transform(pca.transform(z_test)), axis=1),
            ),
            "predicted": classifier.classes_[full.decision_function(test_x).argmax(1)],
        }

    # 内层伪未知距离（用于 CDF 与阈值），按视图信号
    inner_reference = {}
    for fold, unknown in enumerate(np.array_split(np.random.default_rng(42).permutation(np.unique(subjects)), 5), 1):
        known_subjects = np.setdiff1d(np.unique(subjects), unknown)
        for inner_unknown in np.array_split(np.random.default_rng(100 + fold).permutation(known_subjects), 2):
            enrolled = np.setdiff1d(known_subjects, inner_unknown)
            enrolled_mask = np.isin(subjects, enrolled)
            inner_fit = enrolled_mask & ~forward & ~inner
            eval_known = np.flatnonzero(enrolled_mask & inner)
            eval_unknown = np.flatnonzero(np.isin(subjects, inner_unknown) & ~forward)
            for view_name, (train_x, test_x) in views.items():
                model = fit(train_x[inner_fit], subjects[inner_fit])
                scaler = model.named_steps["standardscaler"]
                classifier = model.named_steps["logisticregression"]
                z = scaler.transform(train_x).astype(np.float32)
                centers = np.stack([z[inner_fit & (subjects == c)].mean(0) for c in classifier.classes_])
                whitening = within_class_whitening(z[inner_fit], subjects[inner_fit])
                z_w = (z @ whitening).astype(np.float32)
                centers_w = np.stack([z_w[inner_fit & (subjects == c)].mean(0) for c in classifier.classes_])
                pca = PCA(n_components=min(64, z.shape[1]), random_state=0).fit(z[inner_fit])
                values = {
                    "dist": nearest_distance(z, centers),
                    "dist_wccn": nearest_distance(z_w, centers_w),
                    "pca": np.linalg.norm(z - pca.inverse_transform(pca.transform(z)), axis=1),
                }
                for signal, value in values.items():
                    inner_reference.setdefault((view_name, signal), {"known": [], "unknown": []})
                    inner_reference[(view_name, signal)]["known"].append(value[eval_known])
                    inner_reference[(view_name, signal)]["unknown"].append(value[eval_unknown])

    for key in inner_reference:
        inner_reference[key] = {
            "known": np.concatenate(inner_reference[key]["known"]),
            "unknown": np.concatenate(inner_reference[key]["unknown"]),
        }

    report = {}
    for combo_name, members in combos.items():
        # 每个信号转成「相对内层已知分布的分位」
        def to_quantile(values, key):
            reference = np.sort(inner_reference[key]["known"])
            return np.searchsorted(reference, values) / len(reference)

        train_fused = np.mean([to_quantile(cache[v][s][0], (v, s)) for v, s in members], axis=0)
        test_fused = np.mean([to_quantile(cache[v][s][1], (v, s)) for v, s in members], axis=0)
        auroc = float(roc_auc_score(known, -test_fused))
        entry = {"test_auroc": auroc, "points": {}}
        predicted = cache[members[0][0]]["predicted"]
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
        report[combo_name] = entry
        print(f"{combo_name:28s} Test AUROC={auroc:.4f} | " + " ".join(
            f"{k}:S2={v['s2_accept']:.3f}/S1={v['s1_reject']:.3f}" for k, v in entry["points"].items()
        ), flush=True)

    (OUT / "e30_fusion.json").write_text(
        json.dumps({"inner_auroc": inner_mean, "combos": report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
