"""E22A/B：会话偏移模拟 + 域随机化训练，检验跨会话身份是否可恢复。

训练侧可用标签，因此用「会话偏移后的精度保持率」作为跨会话鲁棒性代理；
测试侧只算无标签诊断量（分数比、AUROC 审计、接受率），不参与任何决策。

偏移算子（特征空间，物理含义对应采集差异）：
  gain  D C D       电极阻抗/增益变化，逐通道（跨频带共享）
  band  s * C       放大器或滤波造成的逐频带功率差异
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cross_session.features import (
    corr_fisher,
    cov_from_log,
    detrend,
    log_from_cov,
    vectorize,
    zscore_bands,
)
from src.eeg_features import N_CHANNELS
from src.open_set_improvement.core import COV_COLUMNS
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

CACHE = ROOT / "BrainPrint_cache"
OUT = Path(__file__).resolve().parent
N_BANDS = 6


def augment(vectors: np.ndarray, rng: np.random.Generator,
            gain_sigma: float = 0.0, band_sigma: float = 0.0) -> np.ndarray:
    covariances = cov_from_log(vectors)
    if gain_sigma > 0:
        gains = np.exp(rng.normal(0.0, gain_sigma, (len(vectors), 1, N_CHANNELS)))
        covariances = covariances * gains[..., :, None] * gains[..., None, :]
    if band_sigma > 0:
        scales = np.exp(rng.normal(0.0, band_sigma, (len(vectors), N_BANDS, 1, 1)))
        covariances = covariances * scales
    return vectorize(log_from_cov(covariances))


REPRESENTATIONS = {
    "cov_raw": lambda v: v,
    "cov_detrend_zband": lambda v: zscore_bands(detrend(v)),
    "corr_fisher": corr_fisher,
    "corr_fisher_zband": lambda v: zscore_bands(corr_fisher(v)),
}


def fit(features: np.ndarray, labels: np.ndarray):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300))
    model.fit(features, labels)
    return model


def main() -> None:
    started = time.time()
    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial"])
    subjects = metadata["subject"].to_numpy()
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    test_meta = pd.read_csv(CACHE / "test_metadata.csv", usecols=["session"])
    known = (test_meta["session"] == "session2").to_numpy()

    train_vectors = np.asarray(
        np.load(CACHE / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32
    )
    test_vectors = np.asarray(
        np.load(CACHE / "test_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32
    )

    rng = np.random.default_rng(2026)
    shifts = {
        "clean": {},
        "gain0.25": {"gain_sigma": 0.25},
        "gain0.50": {"gain_sigma": 0.50},
        "band0.20": {"band_sigma": 0.20},
        "combo": {"gain_sigma": 0.35, "band_sigma": 0.20},
    }
    shifted_train = {name: augment(train_vectors, rng, **kwargs) for name, kwargs in shifts.items() if kwargs}

    fold_groups = np.array_split(np.random.default_rng(42).permutation(np.unique(subjects)), 5)
    report = {}
    for rep_name, transform in REPRESENTATIONS.items():
        features = {name: np.asarray(transform(vectors), dtype=np.float32)
                    for name, vectors in {"clean": train_vectors, **shifted_train}.items()}
        clean_accuracy, shifted_accuracy, augmented_accuracy = [], [], []
        for unknown in fold_groups:
            known_subjects = np.setdiff1d(np.unique(subjects), unknown)
            in_known = np.isin(subjects, known_subjects)
            fit_mask = in_known & ~forward
            evaluate = np.flatnonzero(in_known & forward)
            labels = subjects[evaluate]

            model = fit(features["clean"][fit_mask], subjects[fit_mask])
            clean_accuracy.append(float(np.mean(model.predict(features["clean"][evaluate]) == labels)))
            shifted_accuracy.append({
                name: float(np.mean(model.predict(features[name][evaluate]) == labels))
                for name in shifted_train
            })

            augmented_mask = fit_mask
            augmented_x = [features["clean"][fit_mask], features["combo"][fit_mask], features["gain0.25"][fit_mask]]
            augmented_y = [subjects[fit_mask]] * len(augmented_x)
            model_aug = fit(np.concatenate(augmented_x), np.concatenate(augmented_y))
            augmented_accuracy.append({
                name: float(np.mean(model_aug.predict(features[name][evaluate]) == labels))
                for name in ["clean", *shifted_train]
            })

        report[rep_name] = {
            "clean_accuracy": float(np.mean(clean_accuracy)),
            "shifted_accuracy": {
                name: float(np.mean([row[name] for row in shifted_accuracy]))
                for name in shifted_train
            },
            "retention": {
                name: float(np.mean([row[name] for row in shifted_accuracy]) / np.mean(clean_accuracy))
                for name in shifted_train
            },
            "augmented_accuracy": {
                name: float(np.mean([row[name] for row in augmented_accuracy]))
                for name in ["clean", *shifted_train]
            },
        }
        block = report[rep_name]
        print(f"{rep_name:20s} clean={block['clean_accuracy']:.4f} | " + " ".join(
            f"{name}:{value:.4f}(ret {block['retention'][name]:.3f})"
            for name, value in block["shifted_accuracy"].items()
        ) + " | aug " + " ".join(
            f"{name}:{value:.4f}" for name, value in block["augmented_accuracy"].items()
        ), flush=True)

    # 测试侧诊断：clean 训练 vs 增强训练
    diagnostics = {}
    for rep_name, transform in REPRESENTATIONS.items():
        train_x = np.asarray(transform(train_vectors), dtype=np.float32)
        test_x = np.asarray(transform(test_vectors), dtype=np.float32)
        clean_model = fit(train_x, subjects)
        augmented_model = fit(
            np.concatenate([train_x, np.asarray(transform(shifted_train["combo"]), np.float32),
                            np.asarray(transform(shifted_train["gain0.25"]), np.float32)]),
            np.concatenate([subjects, subjects, subjects]),
        )
        entry = {}
        for tag, model in (("clean_model", clean_model), ("augmented_model", augmented_model)):
            inner_scores = model.decision_function(train_x[inner])
            ordered = np.sort(inner_scores, axis=1)
            inner_max, inner_margin = ordered[:, -1], ordered[:, -1] - ordered[:, -2]
            test_scores = model.decision_function(test_x)
            ordered_t = np.sort(test_scores, axis=1)
            test_max, test_margin = ordered_t[:, -1], ordered_t[:, -1] - ordered_t[:, -2]
            threshold = float(np.quantile(inner_margin, 0.05))
            entry[tag] = {
                "train_max_p50": float(np.median(inner_max)),
                "test_s2_max_p50": float(np.median(test_max[known])),
                "max_ratio": float(np.median(test_max[known]) / np.median(inner_max)),
                "margin_ratio": float(np.median(test_margin[known]) / np.median(inner_margin)),
                "auroc_s2_vs_s1_max": float(roc_auc_score(known, test_max)),
                "auroc_s2_vs_s1_margin": float(roc_auc_score(known, test_margin)),
                "s2_accept_q05": float(np.mean(test_margin[known] >= threshold)),
                "s1_accept_q05": float(np.mean(test_margin[~known] >= threshold)),
            }
        diagnostics[rep_name] = entry

    (OUT / "e22a_shift_robustness.json").write_text(
        json.dumps({"shift_robustness": report, "test_diagnostics": diagnostics},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n测试侧诊断（clean / augmented）")
    for rep_name, entry in diagnostics.items():
        for tag, row in entry.items():
            print(f"{rep_name:20s} {tag:16s} max_ratio={row['max_ratio']:.2f} "
                  f"margin_ratio={row['margin_ratio']:.2f} S2accept={row['s2_accept_q05']:.3f} "
                  f"AUROC(S2vsS1)={row['auroc_s2_vs_s1_max']:.3f}")
    print(f"\n{time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
