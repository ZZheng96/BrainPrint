"""Evaluate session-blind identity scores with distance/margin/window-consistency rejection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "BrainPrint_cache"
OUT = Path(__file__).resolve().parent


def robust_z(x: np.ndarray) -> np.ndarray:
    median = np.nanmedian(x)
    scale = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    return (x - median) / max(float(scale), 1e-6)


def evaluate(protocol: str, metadata: pd.DataFrame) -> dict:
    y = metadata.subject.to_numpy() - 1
    trials = metadata.trial.to_numpy()
    max_trial = metadata.groupby(["subject", "task"]).trial.transform("max").to_numpy()
    selected = np.ones(len(metadata), dtype=bool) if protocol == "blocked_oof" else trials > 0.8 * max_trial
    idx = np.flatnonzero(selected)
    y = y[idx]

    full = np.load(CACHE / f"train_{protocol}_scores_cov_alpha100.npy")[idx]
    window = np.load(CACHE / f"train_{protocol}_scores_window2.npy")[idx]
    neighbor = np.load(CACHE / f"train_{protocol}_scores_subject_calibrated_neighbors.npy")[idx]
    neighbor_z = (neighbor - neighbor.mean(1, keepdims=True)) / np.maximum(neighbor.std(1, keepdims=True), 1e-6)
    fused = 0.5 * full + 0.5 * window + 0.05 * neighbor_z

    full_pred = full.argmax(1)
    window_pred = window.argmax(1)
    fused_pred = fused.argmax(1)
    order = np.sort(fused, axis=1)
    max_score = fused.max(1)
    margin = order[:, -1] - order[:, -2]
    consistency = (full_pred == window_pred).astype(float)
    quality = consistency + 0.25 * robust_z(max_score) + 0.25 * robust_z(margin)

    # OOF/forward thresholds are derived only from the selected known validation scores.
    thresholds = {
        "max_score_q05": float(np.quantile(max_score, 0.05)),
        "margin_q05": float(np.quantile(margin, 0.05)),
        "quality_q05": float(np.quantile(quality, 0.05)),
    }
    decisions = {
        "max_score": max_score >= thresholds["max_score_q05"],
        "margin": margin >= thresholds["margin_q05"],
        "joint_quality": quality >= thresholds["quality_q05"],
        "strict_joint": (max_score >= np.quantile(max_score, 0.10)) & (margin >= np.quantile(margin, 0.10)) & (consistency > 0),
    }
    results = {}
    for name, accepted in decisions.items():
        pred = np.where(accepted, fused_pred, -1)
        results[name] = {
            "accept_rate": float(accepted.mean()),
            "reject_rate": float(1.0 - accepted.mean()),
            "accepted_accuracy": float(np.mean(pred[accepted] == y[accepted])) if accepted.any() else float("nan"),
            "overall_accuracy_with_reject": float(np.mean(pred == y)),
            "macro_f1_with_reject": float(f1_score(y, pred, labels=np.arange(21), average="macro", zero_division=0)),
            "window_agreement": float(consistency[accepted].mean()) if accepted.any() else float("nan"),
        }
    results["continuous"] = {
        "macro_auc": float(roc_auc_score(y, softmax(fused, axis=1), multi_class="ovr", average="macro")),
        "macro_f1": float(f1_score(y, fused_pred, average="macro", zero_division=0)),
        "accuracy": float(np.mean(fused_pred == y)),
    }
    return {"n": int(len(idx)), "thresholds": thresholds, "results": results}


def main() -> None:
    metadata = pd.read_csv(CACHE / "train_metadata.csv")
    report = {protocol: evaluate(protocol, metadata) for protocol in ("blocked_oof", "forward")}
    output = OUT / "joint_rejection_metrics.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for protocol, value in report.items():
        print(protocol, json.dumps(value, ensure_ascii=False))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
