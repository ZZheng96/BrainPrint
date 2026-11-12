"""诊断：训练内层 vs Test 的分数分布偏移。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.open_set_improvement.core import COV_COLUMNS
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

CACHE = ROOT / "BrainPrint_cache"
OUT = Path(__file__).resolve().parent


def scores_of(scores):
    ordered = np.sort(scores, axis=1)
    return ordered[:, -1], ordered[:, -1] - ordered[:, -2]


def quantiles(values):
    return {f"p{int(q*100):02d}": float(np.quantile(values, q))
            for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}


def main():
    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial"])
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)

    cov = np.asarray(
        np.load(CACHE / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32
    )
    image = np.load(CACHE / "eeg_image_dinov2" / "train_covariance5_dinov2.npy").astype(np.float32)
    fused = np.concatenate([image, cov], axis=1)

    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    model.fit(fused, subjects)

    # 训练内层已知
    inner_idx = np.flatnonzero(inner)
    k_max, k_margin = scores_of(model.decision_function(fused[inner_idx]))
    # 训练外层已知（时间前推）
    outer_idx = np.flatnonzero(forward)
    o_max, o_margin = scores_of(model.decision_function(fused[outer_idx]))

    # Test
    test_meta = pd.read_csv(CACHE / "test_metadata.csv", usecols=["sample", "task", "session"])
    test_cov = np.asarray(
        np.load(CACHE / "test_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32
    )
    test_image = np.load(CACHE / "eeg_image_dinov2" / "test_covariance5_dinov2.npy").astype(np.float32)
    test_fused = np.concatenate([test_image, test_cov], axis=1)
    t_max, t_margin = scores_of(model.decision_function(test_fused))

    sessions = test_meta["session"].to_numpy()
    report = {
        "train_inner_known": {"max": quantiles(k_max), "margin": quantiles(k_margin)},
        "train_outer_known": {"max": quantiles(o_max), "margin": quantiles(o_margin)},
        "test_session1": {
            "max": quantiles(t_max[sessions == "session1"]),
            "margin": quantiles(t_margin[sessions == "session1"]),
        },
        "test_session2": {
            "max": quantiles(t_max[sessions == "session2"]),
            "margin": quantiles(t_margin[sessions == "session2"]),
        },
        "thresholds": {
            "margin_q0001": float(np.quantile(k_margin, 0.001)),
            "max_q005": float(np.quantile(k_max, 0.05)),
        },
    }
    (OUT / "e20i_shift_diagnosis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, block in report.items():
        if name == "thresholds":
            print(f"\nthresholds: {block}")
            continue
        print(f"\n{name}:")
        print(f"  max    {block['max']}")
        print(f"  margin {block['margin']}")


if __name__ == "__main__":
    main()
