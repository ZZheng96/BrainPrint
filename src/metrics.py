"""赛题指标与开放集评估工具。"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def score_predictions(
    clean_true: np.ndarray,
    clean_pred: np.ndarray,
    adversarial_true: np.ndarray,
    adversarial_pred: np.ndarray,
    unknown_pred: np.ndarray,
) -> dict[str, float]:
    clean_true = np.asarray(clean_true)
    clean_pred = np.asarray(clean_pred)
    adversarial_true = np.asarray(adversarial_true)
    adversarial_pred = np.asarray(adversarial_pred)
    unknown_pred = np.asarray(unknown_pred)
    c = float(np.mean(clean_pred == clean_true)) if len(clean_true) else float("nan")
    ra = float(np.mean(adversarial_pred == adversarial_true)) if len(adversarial_true) else float("nan")
    asafer = (
        float(np.mean((adversarial_pred == adversarial_true) | (adversarial_pred == 0)))
        if len(adversarial_true)
        else float("nan")
    )
    a = 0.7 * ra + 0.3 * asafer
    u = float(np.mean(unknown_pred == 0)) if len(unknown_pred) else float("nan")
    components = np.asarray([c, a, u])
    proxy = 100.0 * float(np.prod(components) ** (1 / 3)) if np.all(components > 0) else 0.0
    official_score = (
        100.0 * c**0.40 * a**0.35 * u**0.25
        if np.all(np.isfinite(components)) and np.all(components > 0)
        else 0.0
    )
    return {
        "C": c,
        "RA": ra,
        "ASafeR": asafer,
        "A": a,
        "U": u,
        "official_score": official_score,
        "official_score_rounded": round(official_score, 2),
        "internal_equal_weight_proxy": proxy,
    }


def decision_scores(decision: np.ndarray) -> dict[str, np.ndarray]:
    decision = np.asarray(decision)
    ordered = np.sort(decision, axis=1)
    max_logit = decision.max(axis=1)
    centered = decision - max_logit[:, None]
    energy = max_logit + np.log(np.exp(centered).sum(axis=1))
    return {
        "max_logit": max_logit,
        "margin": ordered[:, -1] - ordered[:, -2],
        "energy": energy,
    }


def open_set_metrics(known_scores: np.ndarray, unknown_scores: np.ndarray) -> dict[str, float]:
    known_scores = np.asarray(known_scores)
    unknown_scores = np.asarray(unknown_scores)
    labels = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])
    scores = np.concatenate([known_scores, unknown_scores])
    threshold = np.quantile(known_scores, 0.05)
    return {
        "unknown_auroc": float(roc_auc_score(labels, scores)),
        "unknown_rejection_at_95pct_known_acceptance": float(np.mean(unknown_scores < threshold)),
    }
