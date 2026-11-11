"""嵌套留被试开集评估、对抗消融与可选纯 EEG 候选生成。"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

from src.balanced_sinkhorn import sinkhorn_labels
from src.open_set_improvement.core import (
    COV_COLUMNS,
    SCORE_NAMES,
    VARIANTS,
    ROOT,
    TaskSpecificModel,
    attack_batch,
    competition_report,
    confidence_values,
    open_set_report,
    resolve_cache,
    select_operating_point,
    subject_bootstrap,
)

OUTPUT_DIR = ROOT / "research" / "E02_open_set_improvement"


def forward_holdout(metadata: pd.DataFrame) -> np.ndarray:
    max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    return metadata["trial"].to_numpy() > 0.8 * max_trial


def inner_holdout(metadata: pd.DataFrame) -> np.ndarray:
    max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    trial = metadata["trial"].to_numpy()
    return (trial > 0.6 * max_trial) & (trial <= 0.8 * max_trial)


def pooled_report(records: dict[str, np.ndarray], fold_reports: list[dict]) -> dict:
    known = records["is_known"]
    true = records["true"]
    pred = records["pred"]
    confidence = records["confidence"]
    fpr, tpr, _ = roc_curve(known, confidence)
    fnr = 1 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    n_known = sum(row["n_known"] for row in fold_reports)
    return {
        "n_known": int(known.sum()),
        "n_unknown": int((~known).sum()),
        "C": float(np.mean(pred[known] == true[known])),
        "direct_C_without_rejection": float(np.average(
            [row["direct_C_without_rejection"] for row in fold_reports],
            weights=[row["n_known"] for row in fold_reports],
        )),
        "U": float(np.mean(pred[~known] == 0)),
        "known_acceptance": float(np.average(
            [row["known_acceptance"] for row in fold_reports],
            weights=[row["n_known"] for row in fold_reports],
        )),
        "unknown_auroc": float(roc_auc_score(known, confidence)),
        "known_aupr": float(average_precision_score(known, confidence)),
        "unknown_aupr": float(average_precision_score(~known, -confidence)),
        "eer": float((fpr[index] + fnr[index]) / 2),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "unknown_f1": float(f1_score(
            known.astype(int), pred != 0, pos_label=0, zero_division=0
        )),
        "detection_macro_f1": float(np.average(
            [row["detection_macro_f1"] for row in fold_reports],
            weights=[row["n_known"] + row["n_unknown"] for row in fold_reports],
        )),
        "score_selection": [row["score_name"] for row in fold_reports],
        "thresholds": [row["threshold"] for row in fold_reports],
        "n_known_threshold_denominator": n_known,
    }


def aggregate_attack_reports(reports: list[dict]) -> dict:
    known_weights = [row["n_known"] for row in reports]
    unknown_weights = [row["n_unknown"] for row in reports]
    aggregated = {
        "n_known": int(sum(known_weights)),
        "n_unknown": int(sum(unknown_weights)),
    }
    for name in ("C", "RA", "ASafeR", "clean_macro_f1_with_reject", "adversarial_macro_f1_with_reject"):
        aggregated[name] = float(np.average(
            [row[name] for row in reports], weights=known_weights
        ))
    for name in ("U", "unknown_f1"):
        aggregated[name] = float(np.average(
            [row[name] for row in reports], weights=unknown_weights
        ))
    aggregated["A"] = 0.7 * aggregated["RA"] + 0.3 * aggregated["ASafeR"]
    c, a, u = aggregated["C"], aggregated["A"], aggregated["U"]
    aggregated["official_score"] = 100 * c**0.40 * a**0.35 * u**0.25 if min(c, a, u) > 0 else 0.0
    aggregated["official_score_rounded"] = round(aggregated["official_score"], 2)
    aggregated["n_folds"] = len(reports)
    return aggregated


def label_overrides(
    clean_scores: np.ndarray,
    adversarial_scores: np.ndarray,
    known_tasks: np.ndarray,
    subjects: np.ndarray,
    tasks: np.ndarray,
    fit_mask: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    clean_labels = classes[clean_scores.argmax(axis=1)]
    adversarial_labels = classes[adversarial_scores.argmax(axis=1)]
    for task in sorted(np.unique(known_tasks)):
        clean_indices = np.flatnonzero(known_tasks == task)
        fit_indices = np.flatnonzero(fit_mask & (tasks == task))
        prior = np.asarray([
            np.sum(subjects[fit_indices] == label) for label in classes
        ])
        combined = np.concatenate([
            clean_scores[clean_indices], adversarial_scores[clean_indices]
        ])
        balanced = classes[sinkhorn_labels(combined, prior, 1.0, 100) - 1]
        clean_labels[clean_indices] = balanced[: len(clean_indices)]
        adversarial_labels[clean_indices] = balanced[len(clean_indices) :]
    return clean_labels, adversarial_labels


def evaluate_attacks(
    model: TaskSpecificModel,
    clean_indices: np.ndarray,
    unknown_scores: np.ndarray,
    x: np.ndarray,
    window: np.ndarray,
    subjects: np.ndarray,
    tasks: np.ndarray,
    fit_mask: np.ndarray,
    clean_scores: np.ndarray,
    operating_point: dict,
    epsilons: list[float],
    pgd_steps: int,
    batch_size: int,
) -> dict:
    clean_true = subjects[clean_indices]
    known_tasks = tasks[clean_indices]
    results = {}
    for epsilon in epsilons:
        for method in ("fgsm", "pgd"):
            adversarial_scores = np.empty_like(clean_scores)
            observed_epsilon = []
            for task in sorted(np.unique(known_tasks)):
                positions = np.flatnonzero(known_tasks == task)
                for start in range(0, len(positions), batch_size):
                    selected = positions[start : start + batch_size]
                    indices = clean_indices[selected]
                    scores, max_norm = attack_batch(
                        model,
                        np.asarray(x[indices]),
                        np.asarray(window[indices]),
                        task,
                        clean_true[selected],
                        epsilon,
                        method,
                        pgd_steps,
                    )
                    adversarial_scores[selected] = scores
                    observed_epsilon.append(float(max_norm.max()))
            fused_scores = (clean_scores + adversarial_scores) / 2
            postprocessors = {
                "direct": (clean_scores, adversarial_scores, None),
                "graph_pair": (fused_scores, fused_scores, None),
            }
            direct_override = label_overrides(
                clean_scores, adversarial_scores, known_tasks,
                subjects, tasks, fit_mask, model.classes,
            )
            graph_override = label_overrides(
                fused_scores, fused_scores, known_tasks,
                subjects, tasks, fit_mask, model.classes,
            )
            postprocessors["sinkhorn"] = (clean_scores, adversarial_scores, direct_override)
            postprocessors["graph_pair_sinkhorn"] = (fused_scores, fused_scores, graph_override)
            for postprocessor, (clean, adversarial, override) in postprocessors.items():
                report = competition_report(
                    clean_true,
                    clean,
                    clean_true,
                    adversarial,
                    unknown_scores,
                    model.classes,
                    operating_point["score_name"],
                    operating_point["threshold"],
                    override,
                )
                report["n_known"] = len(clean_true)
                results.setdefault(f"{method}_eps{epsilon:g}", {})[postprocessor] = report
            results[f"{method}_eps{epsilon:g}"]["observed_max_linf_standardized"] = max(observed_epsilon)
    return results


def generate_candidate(
    cache: Path,
    x: np.ndarray,
    window: np.ndarray,
    subjects: np.ndarray,
    tasks: np.ndarray,
    selections: dict[str, list[dict]],
    device: torch.device,
) -> dict:
    variant = max(
        VARIANTS,
        key=lambda name: np.mean([
            row["inner_detection_macro_f1"] for row in selections[name]
        ]),
    )
    variant_selections = selections[variant]
    counts = Counter(row["score_name"] for row in variant_selections)
    score_name = min(
        SCORE_NAMES,
        key=lambda name: (
            -counts[name],
            -np.mean([
                row["inner_detection_macro_f1"] for row in variant_selections
                if row["score_name"] == name
            ] or [0]),
            SCORE_NAMES.index(name),
        ),
    )
    full_oof = np.load(cache / "train_blocked_oof_scores_cov_alpha100.npy")
    if variant == "full":
        oof = full_oof
    elif variant == "base":
        oof = 0.5 * full_oof + 0.5 * np.load(cache / "train_blocked_oof_scores_window2.npy")
    else:
        oof = np.load(cache / "train_blocked_oof_scores_task_specific.npy")
    threshold = float(np.quantile(confidence_values(oof)[score_name], 0.05))
    model = TaskSpecificModel(device).fit(
        x, window, subjects, tasks, np.ones(len(subjects), dtype=bool)
    )
    test_metadata = pd.read_csv(cache / "test_metadata.csv")
    test_x = np.asarray(np.load(cache / "test_features.npy", mmap_mode="r")[:, COV_COLUMNS])
    test_window = np.asarray(
        np.load(cache / "test_window2_features.npy", mmap_mode="r")[:, COV_COLUMNS]
    ).reshape(len(test_metadata), 2, -1)
    scores = model.predict_scores(test_x, test_window, test_metadata["task"].to_numpy())[variant]
    confidence = confidence_values(scores)[score_name]
    predicted = model.classes[scores.argmax(axis=1)].copy()
    predicted[confidence < threshold] = 0
    sample_to_row = dict(zip(test_metadata["sample"], range(len(test_metadata))))
    submission = pd.read_csv(ROOT / "Brainprint_Competition" / "test.csv")
    submission["Label"] = predicted[submission["Sample"].map(sample_to_row).to_numpy()]
    path = OUTPUT_DIR / "submission_pure_eeg.csv"
    submission.to_csv(path, index=False)
    np.save(cache / "scores_open_set_pure_eeg.npy", scores)
    sessions = test_metadata["session"].to_numpy()
    return {
        "path": str(path),
        "variant": variant,
        "variant_selection_source": "highest mean inner detection macro-F1 across outer folds",
        "score_name": score_name,
        "threshold_source": "5th percentile of existing all-subject blocked-OOF known scores; no test labels",
        "threshold": threshold,
        "metadata_rejection": False,
        "batch_postprocessing": False,
        "rejected_rows": int(np.sum(predicted == 0)),
        "rejected_by_session_diagnostic_only": {
            session: int(np.sum(predicted[sessions == session] == 0))
            for session in sorted(np.unique(sessions))
        },
        "class_counts": {
            str(label): int(count) for label, count in zip(*np.unique(predicted, return_counts=True))
        },
        "test_accuracy_claim": "unknown; experimental candidate, formal answer untouched",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--skip-attacks", action="store_true")
    parser.add_argument("--epsilons", default="0.03,0.1")
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--generate-candidate", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "nested_results.json")
    args = parser.parse_args()
    if args.outer_folds < 2 or not 2 <= args.inner_folds <= 3:
        parser.error("outer-folds>=2 且 inner-folds 范围为 2–3")
    if not torch.cuda.is_available():
        parser.error("本实验要求 CUDA GPU")
    device = torch.device("cuda")
    cache = resolve_cache()
    metadata = pd.read_csv(cache / "train_metadata.csv")
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    x = np.asarray(np.load(cache / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS])
    window = np.asarray(
        np.load(cache / "train_window2_features.npy", mmap_mode="r")[:, COV_COLUMNS]
    ).reshape(len(metadata), 2, -1)
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    all_subjects = np.unique(subjects)
    shuffled = np.random.default_rng(42).permutation(all_subjects)
    outer_groups = np.array_split(shuffled, args.outer_folds)
    fold_outputs = []
    records_by_variant = {variant: [] for variant in VARIANTS}
    subjects_by_variant = {variant: [] for variant in VARIANTS}

    for outer_index, unknown_subjects in enumerate(outer_groups):
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        outer_known = np.isin(subjects, known_subjects)
        outer_fit = outer_known & ~forward
        inner_known_scores = {variant: [] for variant in VARIANTS}
        inner_unknown_scores = {variant: [] for variant in VARIANTS}
        inner_order = np.random.default_rng(100 + outer_index).permutation(known_subjects)
        inner_groups = np.array_split(inner_order, args.inner_folds)
        for inner_index, inner_unknown in enumerate(inner_groups):
            inner_enrolled = np.setdiff1d(known_subjects, inner_unknown)
            inner_known_mask = np.isin(subjects, inner_enrolled)
            inner_fit = inner_known_mask & ~forward & ~inner
            inner_eval_known = np.flatnonzero(inner_known_mask & inner)
            inner_eval_unknown = np.flatnonzero(np.isin(subjects, inner_unknown) & ~forward)
            inner_eval = np.concatenate([inner_eval_known, inner_eval_unknown])
            model = TaskSpecificModel(device).fit(x, window, subjects, tasks, inner_fit)
            scores = model.predict_scores(x[inner_eval], window[inner_eval], tasks[inner_eval])
            for variant in VARIANTS:
                inner_known_scores[variant].append(scores[variant][: len(inner_eval_known)])
                inner_unknown_scores[variant].append(scores[variant][len(inner_eval_known) :])
            print(
                f"outer={outer_index + 1} inner={inner_index + 1} "
                f"known={len(inner_eval_known)} unknown={len(inner_eval_unknown)}",
                flush=True,
            )
            del model

        selections = {
            variant: select_operating_point(inner_known_scores[variant], inner_unknown_scores[variant])
            for variant in VARIANTS
        }
        outer_eval_known = np.flatnonzero(outer_known & forward)
        outer_eval_unknown = np.flatnonzero(np.isin(subjects, unknown_subjects))
        outer_eval = np.concatenate([outer_eval_known, outer_eval_unknown])
        model = TaskSpecificModel(device).fit(x, window, subjects, tasks, outer_fit)
        scores = model.predict_scores(x[outer_eval], window[outer_eval], tasks[outer_eval])
        fold_report = {
            "outer_fold": outer_index + 1,
            "unknown_subjects": unknown_subjects.tolist(),
            "registered_subjects": known_subjects.tolist(),
            "selections": selections,
            "variants": {},
        }
        for variant in VARIANTS:
            known_scores = scores[variant][: len(outer_eval_known)]
            unknown_scores = scores[variant][len(outer_eval_known) :]
            report, records = open_set_report(
                subjects[outer_eval_known], known_scores, unknown_scores,
                model.classes, selections[variant]["score_name"], selections[variant]["threshold"],
            )
            fold_report["variants"][variant] = report
            records_by_variant[variant].append(records)
            subject_record = subjects[outer_eval]
            if len(subject_record) != len(records["is_known"]):
                raise RuntimeError(
                    f"逐折被试索引 {len(subject_record)} 与评估记录 {len(records['is_known'])} 不一致"
                )
            subjects_by_variant[variant].append(subject_record)
            print(f"outer={outer_index + 1} {variant}: {report}", flush=True)

        if not args.skip_attacks:
            fold_report["attacks"] = evaluate_attacks(
                model,
                outer_eval_known,
                scores["fused"][len(outer_eval_known) :],
                x,
                window,
                subjects,
                tasks,
                outer_fit,
                scores["fused"][: len(outer_eval_known)],
                selections["fused"],
                [float(value) for value in args.epsilons.split(",")],
                args.pgd_steps,
                args.batch_size,
            )
            print(f"outer={outer_index + 1} attacks complete", flush=True)
        fold_outputs.append(fold_report)
        del model
        torch.cuda.empty_cache()

    variants_report = {}
    for variant in VARIANTS:
        records = {
            key: np.concatenate([fold[key] for fold in records_by_variant[variant]])
            for key in records_by_variant[variant][0]
        }
        record_subjects = np.concatenate(subjects_by_variant[variant])
        variants_report[variant] = {
            "pooled": pooled_report(
                records, [fold["variants"][variant] for fold in fold_outputs]
            ),
            "subject_bootstrap_95pct": subject_bootstrap(
                records, record_subjects, args.bootstrap
            ) if args.bootstrap else {},
        }
    attacks_report = {}
    if not args.skip_attacks:
        attack_names = [
            name for name in fold_outputs[0]["attacks"]
            if not name.startswith("observed_")
        ]
        for attack in attack_names:
            attacks_report[attack] = {
                postprocessor: aggregate_attack_reports([
                    fold["attacks"][attack][postprocessor] for fold in fold_outputs
                ])
                for postprocessor in ("direct", "graph_pair", "sinkhorn", "graph_pair_sinkhorn")
            }
            attacks_report[attack]["observed_max_linf_standardized"] = max(
                fold["attacks"][attack]["observed_max_linf_standardized"] for fold in fold_outputs
            )
    result = {
        "protocol": {
            "data": "Train Session1 only; no cross-session claim",
            "outer": f"{args.outer_folds}-fold subject holdout; 4 subjects/fold at default, all held-out trials unknown",
            "known_outer_test": "last 20% trials within each registered subject/task",
            "inner": f"{args.inner_folds} pseudo-unknown subject groups; registered 60–80% trials and pseudo-unknown first 80% trials select score/threshold; disjoint from outer last 20%",
            "variants": {
                "full": "4-second covariance Ridge",
                "base": "equal 4-second + two-window Ridge",
                "fused": "fixed task-specific full/window/low-Ridge/neighbor fusion; no batch operations",
            },
            "attack": "feature-space L∞ FGSM/PGD against fused scores; epsilon in per-feature training SD units; not raw-EEG attack",
            "graph_pair": "oracle clean/adversarial correspondence; non-adaptive ideal-pair upper bound",
            "sinkhorn": "oracle enrolled-only clean+adversarial batch diagnostic; not a deployable open-set rule",
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
        "folds": fold_outputs,
        "variants": variants_report,
        "attacks": attacks_report,
    }
    if args.generate_candidate:
        result["candidate"] = generate_candidate(
            cache, x, window, subjects, tasks,
            {
                variant: [fold["selections"][variant] for fold in fold_outputs]
                for variant in VARIANTS
            },
            device,
        )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "variants": {name: value["pooled"] for name, value in variants_report.items()},
        "candidate": result.get("candidate"),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
