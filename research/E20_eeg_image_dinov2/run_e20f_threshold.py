"""E20F：拒识阈值权衡曲线。

目标：在保持伪未知拒识能力的前提下，把已知接受率推到接近 100%，
使模型在全是注册者的 Session2 上自然产生接近 0 的拒识率（不依赖 Session 规则）。

对每个外层折，扫描内层已知置信度分位 q ∈ {0.05, 0.02, 0.01, 0.005, 0.001, 0}，
报告外层已知接受率、C、U、unknown AUROC、unknown F1。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.open_set_improvement.core import COV_COLUMNS
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

CACHE = ROOT / "BrainPrint_cache"
OUT = Path(__file__).resolve().parent
QUANTILES = (0.05, 0.02, 0.01, 0.005, 0.001, 0.0)


def margin(scores: np.ndarray) -> np.ndarray:
    top = np.partition(scores, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def fit_model(features, subjects, mask):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    model.fit(features[mask], subjects[mask])
    return model


def evaluate(features, subjects, metadata):
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    all_subjects = np.unique(subjects)
    groups = np.array_split(np.random.default_rng(42).permutation(all_subjects), 5)
    per_quantile = {q: [] for q in QUANTILES}

    for fold, unknown_subjects in enumerate(groups, 1):
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        known = np.isin(subjects, known_subjects)
        outer_fit = known & ~forward

        inner_order = np.random.default_rng(100 + fold).permutation(known_subjects)
        inner_known_conf = []
        for inner_unknown_subjects in np.array_split(inner_order, 2):
            inner_enrolled = np.setdiff1d(known_subjects, inner_unknown_subjects)
            inner_known_mask = np.isin(subjects, inner_enrolled)
            inner_fit = inner_known_mask & ~forward & ~inner
            inner_eval_known = np.flatnonzero(inner_known_mask & inner)
            model = fit_model(features, subjects, inner_fit)
            inner_known_conf.append(margin(model.decision_function(features[inner_eval_known])))
        inner_conf = np.concatenate(inner_known_conf)

        outer_known_eval = np.flatnonzero(known & forward)
        outer_unknown_eval = np.flatnonzero(np.isin(subjects, unknown_subjects))
        model = fit_model(features, subjects, outer_fit)
        classes = model.classes_
        known_scores = model.decision_function(features[outer_known_eval])
        unknown_scores = model.decision_function(features[outer_unknown_eval])
        known_conf = margin(known_scores)
        unknown_conf = margin(unknown_scores)
        known_true = subjects[outer_known_eval]
        detection_true = np.r_[np.ones(len(outer_known_eval)), np.zeros(len(outer_unknown_eval))]
        detection_conf = np.r_[known_conf, unknown_conf]
        auroc = float(roc_auc_score(detection_true, detection_conf))

        for q in QUANTILES:
            threshold = float(np.quantile(inner_conf, q))
            known_pred = classes[known_scores.argmax(axis=1)].copy()
            known_pred[known_conf < threshold] = 0
            unknown_pred = classes[unknown_scores.argmax(axis=1)].copy()
            unknown_pred[unknown_conf < threshold] = 0
            per_quantile[q].append({
                "fold": fold,
                "threshold": threshold,
                "known_acceptance": float(np.mean(known_conf >= threshold)),
                "C": float(np.mean(known_pred == known_true)),
                "U": float(np.mean(unknown_pred == 0)),
                "unknown_auroc": auroc,
                "unknown_f1": float(f1_score(
                    detection_true, detection_conf >= threshold, pos_label=0, zero_division=0
                )),
            })

    summary = {}
    for q, rows in per_quantile.items():
        summary[str(q)] = {
            key: float(np.mean([r[key] for r in rows]))
            for key in ("known_acceptance", "C", "U", "unknown_auroc", "unknown_f1")
        }
    return {"per_quantile": {str(k): v for k, v in per_quantile.items()}, "summary": summary}


def main():
    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial"])
    subjects = metadata["subject"].to_numpy()
    cov = np.asarray(
        np.load(CACHE / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32
    )
    image = np.load(CACHE / "eeg_image_dinov2" / "train_covariance5_dinov2.npy").astype(np.float32)
    fused = np.concatenate([image, cov], axis=1)

    results = {}
    for name, features in (("covariance_only", cov), ("covariance5_fused", fused)):
        print(f"\n=== {name} ===", flush=True)
        results[name] = evaluate(features, subjects, metadata)
        for q, s in results[name]["summary"].items():
            print(f"  q={q:>6s} accept={s['known_acceptance']:.4f} C={s['C']:.4f} "
                  f"U={s['U']:.4f} AUROC={s['unknown_auroc']:.4f} F1={s['unknown_f1']:.4f}",
                  flush=True)

    (OUT / "e20f_threshold_tradeoff.json").write_text(
        json.dumps({
            "protocol": {
                "note": "纯 EEG；阈值只由内层已知置信度分位产生；无 Session 参与",
                "quantiles": list(QUANTILES),
            },
            "results": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved {OUT / 'e20f_threshold_tradeoff.json'}")


if __name__ == "__main__":
    main()
