"""E21E：拒识阈值与官方评分的权衡曲线（决策分析，不产生提交）。

已知事实：Test 的 Session1（未注册）与 Session2（注册）在分类分数上几乎不可区分，
因此任何统一阈值都会同时拒识两类样本。本脚本量化这条权衡：

  - 用 Train 内层伪未知的分位数 q 作为阈值（严格内层标定）；
  - 统计 Test 上 S2 接受率与 S1 拒识率；
  - 用官方公式做代理估计：C ≈ C0 × S2接受率（C0 为接受样本上的闭集准确率，未知量，
    由第 1 次提交 69.17 反推约 0.40，仅作敏感性示意）；U = S1拒绝率；
    RA ≈ C、ASafeR ≈ C + (1-C)×拒识率（无真实对抗标签，只能给上界近似）。

同时给出距离型新颖度（最近类马氏距离）的同一权衡，作为「唯一方向正确」的对照。
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
FEATURE_CACHE = CACHE / "e21_scale_invariant"
OUT = Path(__file__).resolve().parent
PSD_DIM = 18 * 90
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 40))
QS = (0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01, 0.001)


def band_relative(psd: np.ndarray) -> np.ndarray:
    block = psd.reshape(len(psd), 18, 90)
    band_power = np.stack(
        [block[:, :, low - 1 : high - 1].mean(axis=2) for low, high in BANDS], axis=2
    )
    total = band_power.mean(axis=2, keepdims=True)
    return np.log(np.maximum(band_power, 1e-8) / np.maximum(total, 1e-8)).reshape(len(psd), -1)


def margin(scores: np.ndarray) -> np.ndarray:
    top = np.partition(scores, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def proxy_score(c: float, u: float) -> float:
    """RA/ASafeR 无真实对抗标签，按 RA=C、ASafeR=C+(1-C)(1-C) 保守近似。"""
    ra = c
    asafer = c + (1 - c) * (1 - c)
    a = 0.7 * ra + 0.3 * asafer
    return float(100 * c**0.40 * a**0.35 * u**0.25) if c > 0 and u > 0 else 0.0


def main() -> None:
    train_psd = np.asarray(np.load(CACHE / "train_features.npy", mmap_mode="r")[:, :PSD_DIM], np.float32)
    test_psd = np.asarray(np.load(CACHE / "test_features.npy", mmap_mode="r")[:, :PSD_DIM], np.float32)
    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial"])
    subjects = metadata["subject"].to_numpy()
    test_meta = pd.read_csv(CACHE / "test_metadata.csv", usecols=["session"])
    known = (test_meta["session"] == "session2").to_numpy()

    train = np.concatenate([band_relative(train_psd), np.load(FEATURE_CACHE / "train_cov_detrend_zband.npy")], axis=1)
    test = np.concatenate([band_relative(test_psd), np.load(FEATURE_CACHE / "test_cov_detrend_zband.npy")], axis=1)

    scaler = StandardScaler().fit(train)
    ztr, zte = scaler.transform(train).astype(np.float32), scaler.transform(test).astype(np.float32)
    model = LogisticRegression(C=1.0, max_iter=300).fit(ztr, subjects)

    inner = inner_holdout(metadata)
    forward = forward_holdout(metadata)
    inner_scores = model.decision_function(ztr[inner])
    inner_margin = margin(inner_scores)
    outer_margin = margin(model.decision_function(ztr[forward]))

    test_scores = model.decision_function(zte)
    test_margin = margin(test_scores)

    # 距离型新颖度：到最近已注册类中心的欧氏距离（标准化空间）
    # 中心只用既不在内层也不在外层的训练样本估计，避免标定泄漏
    centroid_mask = ~inner & ~forward
    centroids = np.stack([ztr[centroid_mask & (subjects == c)].mean(0) for c in model.classes_])

    def nearest_distance(data: np.ndarray) -> np.ndarray:
        squared = (
            np.sum(data**2, axis=1, keepdims=True)
            - 2 * data @ centroids.T
            + np.sum(centroids**2, axis=1)[None, :]
        )
        return np.sqrt(np.maximum(squared.min(axis=1), 0.0))

    test_distance = nearest_distance(zte)
    distance = test_distance
    inner_distance = nearest_distance(ztr[inner])
    outer_distance = nearest_distance(ztr[forward])

    curves = {"margin": [], "distance": []}
    for q in QS:
        threshold = float(np.quantile(inner_margin, q))
        keep = test_margin >= threshold
        s2_accept, s1_accept = float(keep[known].mean()), float(keep[~known].mean())
        curves["margin"].append({
            "q": q,
            "train_inner_acceptance": float(np.mean(inner_margin >= threshold)),
            "train_outer_acceptance": float(np.mean(outer_margin >= threshold)),
            "test_s2_accept": s2_accept,
            "test_s1_reject": 1 - s1_accept,
            "proxy_C_c0_0.40": 0.40 * s2_accept,
            "proxy_U": 1 - s1_accept,
            "proxy_score_c0_0.40": proxy_score(0.40 * s2_accept, 1 - s1_accept),
        })
    for q in (0.5, 0.4, 0.3, 0.2, 0.1, 0.05):
        threshold = float(np.quantile(inner_distance, 1 - q))
        keep = test_distance <= threshold
        s2_accept, s1_accept = float(keep[known].mean()), float(keep[~known].mean())
        curves["distance"].append({
            "reject_rate_on_train": 1 - q,
            "train_outer_acceptance": float(np.mean(outer_distance <= threshold)),
            "test_s2_accept": s2_accept,
            "test_s1_reject": 1 - s1_accept,
            "proxy_C_c0_0.40": 0.40 * s2_accept,
            "proxy_U": 1 - s1_accept,
            "proxy_score_c0_0.40": proxy_score(0.40 * s2_accept, 1 - s1_accept),
        })

    report = {
        "auroc_s2_vs_s1_margin": float(roc_auc_score(known, test_margin)),
        "auroc_s2_vs_s1_distance": float(roc_auc_score(known, -distance)),
        "test_s2_margin_p50": float(np.median(test_margin[known])),
        "test_s1_margin_p50": float(np.median(test_margin[~known])),
        "train_inner_margin_p50": float(np.median(inner_margin)),
        "curves": curves,
    }
    (OUT / "e21e_tradeoff.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"AUROC(S2 vs S1) margin={report['auroc_s2_vs_s1_margin']:.4f} "
          f"distance={report['auroc_s2_vs_s1_distance']:.4f}")
    print("\nmargin 曲线（q 为内层已知分位阈值）")
    print("     q  train_acc  S2_acc  S1_rej  proxy_C  proxy_U  proxy_score")
    for row in curves["margin"]:
        print(f"{row['q']:6.3f} {row['train_inner_acceptance']:10.3f} {row['test_s2_accept']:7.3f} "
              f"{row['test_s1_reject']:7.3f} {row['proxy_C_c0_0.40']:8.3f} {row['proxy_U']:8.3f} "
              f"{row['proxy_score_c0_0.40']:10.2f}")
    print("\ndistance 曲线（拒绝距离最大的那部分）")
    print(" train_rej  S2_acc  S1_rej  proxy_C  proxy_U  proxy_score")
    for row in curves["distance"]:
        print(f"{row['reject_rate_on_train']:9.2f} {row['test_s2_accept']:7.3f} {row['test_s1_reject']:7.3f} "
              f"{row['proxy_C_c0_0.40']:8.3f} {row['proxy_U']:8.3f} {row['proxy_score_c0_0.40']:10.2f}")


if __name__ == "__main__":
    main()
