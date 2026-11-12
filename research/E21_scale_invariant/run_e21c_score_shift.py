"""E21C：检验尺度不变表示能否修复 Test 分数偏移，并审计 Session2 拒识率。

流程：
  1. 对每个表示，用全量 Train 拟合 20 类模型；
  2. 内层（train 60%-80% trial）标定 margin / max 阈值（q=0.001 / q=0.05，与 E20G 一致）；
  3. 对 Test 打分，统计 max/margin 分位数与接受率；
  4. 写出后才读 session 做无标签审计。

必须说明：Test 只用于审计与报告，不参与任何表示或阈值选择。
"""
from __future__ import annotations

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

from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

CACHE = ROOT / "BrainPrint_cache"
FEATURE_CACHE = CACHE / "e21_scale_invariant"
OUT = Path(__file__).resolve().parent
REPRESENTATIONS = (
    "cov_raw",
    "cov_detrend",
    "cov_offdiag",
    "cov_detrend_zband",
    "cov_detrend_unit",
    "corr_fisher",
    "corr_fisher_zband",
)


def top_two(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(scores, axis=1)
    return ordered[:, -1], ordered[:, -1] - ordered[:, -2]


def quantiles(values: np.ndarray) -> dict:
    return {f"p{int(q * 100):02d}": float(np.quantile(values, q))
            for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}


def main() -> None:
    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial"])
    subjects = metadata["subject"].to_numpy()
    inner = inner_holdout(metadata)
    forward = forward_holdout(metadata)
    inner_known = np.flatnonzero(inner)
    outer_known = np.flatnonzero(forward)

    test_meta = pd.read_csv(CACHE / "test_metadata.csv", usecols=["sample", "task", "session"])
    sessions = test_meta["session"].to_numpy()

    report = {}
    for name in REPRESENTATIONS:
        train = np.load(FEATURE_CACHE / f"train_{name}.npy").astype(np.float32)
        test = np.load(FEATURE_CACHE / f"test_{name}.npy").astype(np.float32)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300))
        model.fit(train, subjects)

        i_max, i_margin = top_two(model.decision_function(train[inner_known]))
        o_max, o_margin = top_two(model.decision_function(train[outer_known]))
        t_scores = model.decision_function(test)
        t_max, t_margin = top_two(t_scores)
        t_pred = model.classes_[t_scores.argmax(axis=1)]

        margin_threshold = float(np.quantile(i_margin, 0.001))
        max_threshold = float(np.quantile(i_max, 0.05))
        accept = (t_margin >= margin_threshold) & (t_max >= max_threshold)
        accept_margin = t_margin >= float(np.quantile(i_margin, 0.05))
        norm_train = np.linalg.norm(train[inner_known], axis=1)
        norm_test = np.linalg.norm(test, axis=1)

        report[name] = {
            "inner_known": {"max": quantiles(i_max), "margin": quantiles(i_margin)},
            "outer_known": {"max": quantiles(o_max), "margin": quantiles(o_margin)},
            "test_all": {"max": quantiles(t_max), "margin": quantiles(t_margin)},
            "test_session1": {
                "max": quantiles(t_max[sessions == "session1"]),
                "margin": quantiles(t_margin[sessions == "session1"]),
                "accept_rate_two_stage": float(accept[sessions == "session1"].mean()),
                "accept_rate_margin_q05": float(accept_margin[sessions == "session1"].mean()),
            },
            "test_session2": {
                "max": quantiles(t_max[sessions == "session2"]),
                "margin": quantiles(t_margin[sessions == "session2"]),
                "accept_rate_two_stage": float(accept[sessions == "session2"].mean()),
                "accept_rate_margin_q05": float(accept_margin[sessions == "session2"].mean()),
            },
            "test_accept_rate_two_stage_all": float(accept.mean()),
            "thresholds": {"margin_q0001": margin_threshold, "max_q005": max_threshold},
            "score_norm_correlation": {
                "train_inner_known": float(np.corrcoef(i_margin, norm_train)[0, 1]),
                "test_session2": float(np.corrcoef(
                    t_margin[sessions == "session2"], norm_test[sessions == "session2"]
                )[0, 1]),
            },
        }
        block = report[name]
        print(
            f"{name:20s} S2 accept(two_stage)={block['test_session2']['accept_rate_two_stage']:.4f} "
            f"S2 max_med={block['test_session2']['max']['p50']:.2f} "
            f"train max_med={block['inner_known']['max']['p50']:.2f} "
            f"S2 margin_med={block['test_session2']['margin']['p50']:.2f} "
            f"train margin_med={block['inner_known']['margin']['p50']:.2f} "
            f"r(margin,norm)={block['score_norm_correlation']['test_session2']:.3f}",
            flush=True,
        )

    (OUT / "e21c_score_shift.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
