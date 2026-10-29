"""任务特异的线性脑纹基线：块保留验证 + 留被试开放集评估。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg_features import N_COV_FEATURES, N_PSD_FEATURES
from metrics import decision_scores, open_set_metrics

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "BrainPrint_cache"
FEATURE_SETS = {
    "psd": np.arange(N_PSD_FEATURES),
    "cov": np.arange(N_PSD_FEATURES, N_PSD_FEATURES + N_COV_FEATURES),
    "cov_1_8": np.arange(N_PSD_FEATURES, N_PSD_FEATURES + 171),
    "cov_8_13": np.arange(N_PSD_FEATURES + 171, N_PSD_FEATURES + 2 * 171),
    "cov_1_13": np.arange(N_PSD_FEATURES, N_PSD_FEATURES + 2 * 171),
    "cov_1_30": np.arange(N_PSD_FEATURES, N_PSD_FEATURES + 3 * 171),
    "cov_1_45": np.arange(N_PSD_FEATURES, N_PSD_FEATURES + 4 * 171),
    "cov_1_70": np.arange(N_PSD_FEATURES, N_PSD_FEATURES + 5 * 171),
    "all": np.arange(N_PSD_FEATURES + N_COV_FEATURES),
}


def make_model(name: str, seed: int, alpha: float = 100.0):
    if name == "ridge":
        return make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha, solver="lsqr"))
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=300, tol=1e-4, solver="lbfgs"),
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(name)


def get_decision(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    probabilities = model.predict_proba(x)
    return np.log(np.maximum(probabilities, 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ridge", "logistic", "extra_trees"], default="ridge")
    parser.add_argument("--features", choices=list(FEATURE_SETS), default="all")
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--validation-fold", type=int, default=4)
    parser.add_argument("--split-mode", choices=["blocked", "forward"], default="blocked")
    parser.add_argument("--unknown-start", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    x = np.load(CACHE / "train_features.npy", mmap_mode="r")
    columns = FEATURE_SETS[args.features]
    metadata = pd.read_csv(CACHE / "train_metadata.csv")
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    folds = ((metadata["trial"].to_numpy() - 1) // 10) % 5
    known_mask = subjects < args.unknown_start
    unknown_mask = ~known_mask
    if args.split_mode == "blocked":
        selected_validation = folds == args.validation_fold
    else:
        max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
        selected_validation = metadata["trial"].to_numpy() > 0.8 * max_trial
    validation_mask = known_mask & selected_validation
    fit_mask = known_mask & ~selected_validation

    all_known_indices = np.flatnonzero(validation_mask)
    all_unknown_indices = np.flatnonzero(unknown_mask)
    predictions = np.zeros(len(all_known_indices), dtype=int)
    known_confidences = {name: np.zeros(len(all_known_indices)) for name in ("max_logit", "margin", "energy")}
    unknown_confidences = {name: np.zeros(len(all_unknown_indices)) for name in known_confidences}
    per_task = {}
    position_lookup = np.full(len(metadata), -1, dtype=int)
    position_lookup[all_known_indices] = np.arange(len(all_known_indices))
    position_lookup[all_unknown_indices] = np.arange(len(all_unknown_indices))

    for task in sorted(metadata["task"].unique()):
        fit_indices = np.flatnonzero(fit_mask & (tasks == task))
        known_indices = np.flatnonzero(validation_mask & (tasks == task))
        unknown_indices = np.flatnonzero(unknown_mask & (tasks == task))
        print(f"{task}: fit={len(fit_indices)}, known={len(known_indices)}, unknown={len(unknown_indices)}", flush=True)
        model = make_model(args.model, seed=42, alpha=args.alpha)
        model.fit(x[fit_indices][:, columns], subjects[fit_indices])
        for indices, confidence_target in ((known_indices, known_confidences), (unknown_indices, unknown_confidences)):
            if not len(indices):
                continue
            decision = get_decision(model, x[indices][:, columns])
            predicted = model.classes_[np.argmax(decision, axis=1)]
            scores = decision_scores(decision)
            positions = position_lookup[indices]
            if len(known_indices) and indices is known_indices:
                predictions[positions] = predicted
            for name in confidence_target:
                confidence_target[name][positions] = scores[name]
        if len(known_indices):
            known_positions = position_lookup[known_indices]
            per_task[task] = {
                "n": len(known_indices),
                "clean_accuracy": float(np.mean(predictions[known_positions] == subjects[known_indices])),
            }

    known_true = subjects[all_known_indices]
    report = {
        "protocol": {
            "model": args.model,
            "features": args.features,
            "alpha": args.alpha,
            "validation": (
                "within-session contiguous blocks of 10 trials, blocked modulo 5"
                if args.split_mode == "blocked"
                else "within-subject/task forward holdout: last 20 percent of trial IDs"
            ),
            "split_mode": args.split_mode,
            "validation_fold": args.validation_fold,
            "known_subjects": list(range(1, args.unknown_start)),
            "held_out_unknown_subjects": list(range(args.unknown_start, 21)),
            "cross_session_claim": False,
        },
        "clean_accuracy": float(np.mean(predictions == known_true)),
        "per_task": per_task,
        "open_set": {
            name: open_set_metrics(known_confidences[name], unknown_confidences[name])
            for name in known_confidences
        },
    }
    output = args.output or ROOT / "research" / f"baseline_{args.model}_{args.features}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
