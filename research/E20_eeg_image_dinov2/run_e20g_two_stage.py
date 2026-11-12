"""E20G：双判据拒识——在保持已知接受率≈100% 的同时维持未知拒识。

动机：E20F 显示单一 margin 阈值无法兼顾（接受率 99.9% 时 U 崩到 0.12），
但 unknown AUROC 始终 0.993，说明排序能力足够，问题在判据设计。

本实验比较三种判据（全部只用内层已知分数标定，纯 EEG）：
  margin_only    top1 - top2
  max_only       top1 绝对分数
  two_stage      margin 高接受率门 + max 分数二次拒识
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


def scores_of(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(scores, axis=1)
    return ordered[:, -1], ordered[:, -1] - ordered[:, -2]


def fit_model(features, subjects, mask):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    model.fit(features[mask], subjects[mask])
    return model


def evaluate(features, subjects, metadata):
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    all_subjects = np.unique(subjects)
    groups = np.array_split(np.random.default_rng(42).permutation(all_subjects), 5)
    rows = []

    for fold, unknown_subjects in enumerate(groups, 1):
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        known = np.isin(subjects, known_subjects)
        outer_fit = known & ~forward

        inner_order = np.random.default_rng(100 + fold).permutation(known_subjects)
        inner_max, inner_margin = [], []
        for inner_unknown_subjects in np.array_split(inner_order, 2):
            inner_enrolled = np.setdiff1d(known_subjects, inner_unknown_subjects)
            inner_known_mask = np.isin(subjects, inner_enrolled)
            inner_fit = inner_known_mask & ~forward & ~inner
            inner_eval_known = np.flatnonzero(inner_known_mask & inner)
            model = fit_model(features, subjects, inner_fit)
            mx, mg = scores_of(model.decision_function(features[inner_eval_known]))
            inner_max.append(mx)
            inner_margin.append(mg)
        inner_max = np.concatenate(inner_max)
        inner_margin = np.concatenate(inner_margin)

        outer_known_eval = np.flatnonzero(known & forward)
        outer_unknown_eval = np.flatnonzero(np.isin(subjects, unknown_subjects))
        model = fit_model(features, subjects, outer_fit)
        classes = model.classes_
        k_max, k_margin = scores_of(model.decision_function(features[outer_known_eval]))
        u_max, u_margin = scores_of(model.decision_function(features[outer_unknown_eval]))
        known_true = subjects[outer_known_eval]
        detection_true = np.r_[np.ones(len(outer_known_eval)), np.zeros(len(outer_unknown_eval))]

        # 判据 1：仅 margin，q=0.05（原基线）
        t_margin = float(np.quantile(inner_margin, 0.05))
        # 判据 2：仅 max，q=0.05
        t_max = float(np.quantile(inner_max, 0.05))
        # 判据 3：两阶段——margin 门放到 q=0.001（接受率≈99.8%），再用 max 二次拒识
        t_margin_loose = float(np.quantile(inner_margin, 0.001))
        t_max_strict = float(np.quantile(inner_max, 0.05))

        variants = {
            "margin_only": (k_margin >= t_margin, u_margin >= t_margin),
            "max_only": (k_max >= t_max, u_max >= t_max),
            "two_stage": (
                (k_margin >= t_margin_loose) & (k_max >= t_max_strict),
                (u_margin >= t_margin_loose) & (u_max >= t_max_strict),
            ),
        }
        for name, (k_acc, u_acc) in variants.items():
            known_pred = classes[model.decision_function(features[outer_known_eval]).argmax(axis=1)].copy()
            known_pred[~k_acc] = 0
            unknown_pred = classes[model.decision_function(features[outer_unknown_eval]).argmax(axis=1)].copy()
            unknown_pred[~u_acc] = 0
            detection_conf = np.r_[k_margin, u_margin]
            rows.append({
                "fold": fold, "variant": name,
                "known_acceptance": float(np.mean(k_acc)),
                "C": float(np.mean(known_pred == known_true)),
                "U": float(np.mean(unknown_pred == 0)),
                "unknown_auroc": float(roc_auc_score(detection_true, detection_conf)),
                "unknown_f1": float(f1_score(
                    detection_true, np.r_[k_acc, u_acc], pos_label=0, zero_division=0
                )),
            })

    summary = {}
    for name in ("margin_only", "max_only", "two_stage"):
        subset = [r for r in rows if r["variant"] == name]
        summary[name] = {
            key: float(np.mean([r[key] for r in subset]))
            for key in ("known_acceptance", "C", "U", "unknown_auroc", "unknown_f1")
        }
    return {"folds": rows, "summary": summary}


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
        for variant, s in results[name]["summary"].items():
            print(f"  {variant:12s} accept={s['known_acceptance']:.4f} C={s['C']:.4f} "
                  f"U={s['U']:.4f} AUROC={s['unknown_auroc']:.4f} F1={s['unknown_f1']:.4f}",
                  flush=True)

    (OUT / "e20g_two_stage.json").write_text(
        json.dumps({
            "protocol": {
                "note": "纯 EEG；所有阈值只由内层已知分数标定；无 Session 参与",
                "variants": {
                    "margin_only": "top1-top2, q=0.05",
                    "max_only": "top1, q=0.05",
                    "two_stage": "margin q=0.001 AND max q=0.05",
                },
            },
            "results": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved {OUT / 'e20g_two_stage.json'}")


if __name__ == "__main__":
    main()
