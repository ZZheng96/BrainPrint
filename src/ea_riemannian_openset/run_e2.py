"""E2：固定 Riemannian 重居中，仅替换轻量分类器的正反时间嵌套评估。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.ea_riemannian_openset.core import SPDRecenter
from src.ea_riemannian_openset.e2_models import E2TaskSpecificModel, MODEL_NAMES
from src.ea_riemannian_openset.run import (
    OUTPUT_DIR,
    TIME_PROTOCOLS,
    macro_mean,
    time_holdouts,
    worst_fold,
)
from src.open_set_calibration.core import aggregate_reports
from src.open_set_improvement.core import (
    COV_COLUMNS,
    open_set_report,
    resolve_cache,
    select_operating_point,
    subject_bootstrap,
)


def nested_loso_groups(
    subjects: np.ndarray, outer_folds: int = 5, inner_folds: int = 2
) -> list[tuple[np.ndarray, np.ndarray, list[np.ndarray]]]:
    """与 E1 相同的随机种子和被试划分；返回 unknown/known/inner unknown groups。"""
    all_subjects = np.unique(subjects)
    shuffled = np.random.default_rng(42).permutation(all_subjects)
    groups = []
    for outer_index, unknown_subjects in enumerate(np.array_split(shuffled, outer_folds)):
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        inner_order = np.random.default_rng(100 + outer_index).permutation(known_subjects)
        inner_groups = list(np.array_split(inner_order, inner_folds))
        groups.append((unknown_subjects, known_subjects, inner_groups))
    return groups


def fit_predict_e2_split(
    x: np.ndarray,
    subjects: np.ndarray,
    tasks: np.ndarray,
    fit_mask: np.ndarray,
    eval_indices: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 256,
    riemannian_max_iter: int = 50,
    riemannian_tol: float = 1e-7,
) -> tuple[dict[str, np.ndarray], dict, np.ndarray]:
    """每个 split 的源/目标参考与变换各计算一次，三种模型共享结果。"""
    source_aligner = SPDRecenter(
        "riemannian_recenter",
        device,
        batch_size=batch_size,
        max_iter=riemannian_max_iter,
        tol=riemannian_tol,
    ).fit(x, tasks, fit_mask)
    fit_indices = np.flatnonzero(fit_mask)
    source_x = source_aligner.transform(x[fit_indices], tasks[fit_indices])
    model = E2TaskSpecificModel(device).fit(
        source_x,
        np.asarray(subjects)[fit_indices],
        np.asarray(tasks)[fit_indices],
        np.ones(len(fit_indices), dtype=bool),
    )
    target_aligner = SPDRecenter(
        "riemannian_recenter",
        device,
        batch_size=batch_size,
        max_iter=riemannian_max_iter,
        tol=riemannian_tol,
    ).fit(x[eval_indices], tasks[eval_indices])
    target_x = target_aligner.transform(x[eval_indices], tasks[eval_indices])
    outputs = model.predict_scores(target_x, tasks[eval_indices], batch_size)
    diagnostics = {
        "source": source_aligner.diagnostics,
        "target": target_aligner.diagnostics,
    }
    return outputs, diagnostics, model.classes.copy()


def evaluate_e2(args: argparse.Namespace, device: torch.device) -> dict:
    cache = resolve_cache()
    metadata = pd.read_csv(cache / "train_metadata.csv")
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    x = np.asarray(
        np.load(cache / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS]
    )
    outer_time, inner_time = time_holdouts(metadata, args.time_protocol)
    fold_outputs = []
    records_by_model = {name: [] for name in MODEL_NAMES}
    subjects_by_model = {name: [] for name in MODEL_NAMES}

    for outer_index, (unknown_subjects, known_subjects, inner_groups) in enumerate(
        nested_loso_groups(subjects, args.outer_folds, args.inner_folds)
    ):
        outer_known = np.isin(subjects, known_subjects)
        outer_fit = outer_known & ~outer_time
        inner_scores = {
            name: {"known": [], "unknown": []} for name in MODEL_NAMES
        }
        inner_diagnostics = []
        for inner_index, inner_unknown in enumerate(inner_groups):
            inner_enrolled = np.setdiff1d(known_subjects, inner_unknown)
            inner_known_mask = np.isin(subjects, inner_enrolled)
            inner_fit = inner_known_mask & ~outer_time & ~inner_time
            inner_eval_known = np.flatnonzero(inner_known_mask & inner_time)
            inner_eval_unknown = np.flatnonzero(
                np.isin(subjects, inner_unknown) & ~outer_time
            )
            inner_eval = np.concatenate([inner_eval_known, inner_eval_unknown])
            scores, diagnostics, _ = fit_predict_e2_split(
                x,
                subjects,
                tasks,
                inner_fit,
                inner_eval,
                device,
                batch_size=args.batch_size,
                riemannian_max_iter=args.riemannian_max_iter,
                riemannian_tol=args.riemannian_tol,
            )
            inner_diagnostics.append(diagnostics)
            for name in MODEL_NAMES:
                inner_scores[name]["known"].append(
                    scores[name][: len(inner_eval_known)]
                )
                inner_scores[name]["unknown"].append(
                    scores[name][len(inner_eval_known) :]
                )
            print(
                f"protocol={args.time_protocol} outer={outer_index + 1} "
                f"inner={inner_index + 1} known={len(inner_eval_known)} "
                f"unknown={len(inner_eval_unknown)}",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

        selections = {
            name: select_operating_point(
                inner_scores[name]["known"], inner_scores[name]["unknown"]
            )
            for name in MODEL_NAMES
        }
        outer_eval_known = np.flatnonzero(outer_known & outer_time)
        outer_eval_unknown = np.flatnonzero(np.isin(subjects, unknown_subjects))
        outer_eval = np.concatenate([outer_eval_known, outer_eval_unknown])
        scores, diagnostics, classes = fit_predict_e2_split(
            x,
            subjects,
            tasks,
            outer_fit,
            outer_eval,
            device,
            batch_size=args.batch_size,
            riemannian_max_iter=args.riemannian_max_iter,
            riemannian_tol=args.riemannian_tol,
        )
        fold_report = {
            "outer_fold": outer_index + 1,
            "unknown_subjects": unknown_subjects.tolist(),
            "registered_subjects": known_subjects.tolist(),
            "selections": selections,
            "inner_alignment_diagnostics": inner_diagnostics,
            "outer_alignment_diagnostics": diagnostics,
            "results": {},
        }
        for name in MODEL_NAMES:
            report, records = open_set_report(
                subjects[outer_eval_known],
                scores[name][: len(outer_eval_known)],
                scores[name][len(outer_eval_known) :],
                classes,
                selections[name]["score_name"],
                selections[name]["threshold"],
            )
            fold_report["results"][name] = report
            records_by_model[name].append(records)
            subjects_by_model[name].append(subjects[outer_eval])
            print(
                f"protocol={args.time_protocol} outer={outer_index + 1} {name}: "
                f"AUROC={report['unknown_auroc']:.6f} "
                f"macro-F1={report['macro_f1']:.6f}",
                flush=True,
            )
        fold_outputs.append(fold_report)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    configs_report = {}
    for name in MODEL_NAMES:
        records = {
            key: np.concatenate([fold[key] for fold in records_by_model[name]])
            for key in records_by_model[name][0]
        }
        record_subjects = np.concatenate(subjects_by_model[name])
        fold_reports = [fold["results"][name] for fold in fold_outputs]
        configs_report[name] = {
            "aggregate": aggregate_reports(records, fold_reports),
            "macro_mean": macro_mean(fold_reports),
            "worst_fold": worst_fold(fold_outputs, name),
            "subject_bootstrap_95pct": subject_bootstrap(
                records, record_subjects, args.bootstrap
            ) if args.bootstrap else {},
        }
    return {
        "protocol": {
            "data": "Train Session1 only; temporal/subject pseudo-target, not cross-session evidence",
            "time_protocol": args.time_protocol,
            "outer": f"{args.outer_folds}-fold LOSO; same seed/groups as E1; held-out subjects are unknown",
            "known_outer_test": (
                "last 20% trials within each registered subject/task"
                if args.time_protocol == "forward"
                else "first 20% trials within each registered subject/task"
            ),
            "inner": (
                f"{args.inner_folds} pseudo-unknown groups; source first 60%, registered 60–80%, pseudo-unknown first 80%"
                if args.time_protocol == "forward"
                else f"{args.inner_folds} pseudo-unknown groups; source last 60%, registered 20–40%, pseudo-unknown last 80%"
            ),
            "features": "full 4-second log-cov only; same SPDRecenter riemannian_recenter as E1",
            "shared_alignment": "source/target SPD reference and log-vectorized transforms computed exactly once per split for all three models",
            "models": list(MODEL_NAMES),
            "ridge": "existing E1 TorchRidge alpha=100, float32",
            "shrinkage_lda": "StandardScaler + sklearn LDA lsqr/auto semantics: per-class Ledoit-Wolf covariance weighted by empirical priors; CUDA float64",
            "cosine_prototype": "source-fit StandardScaler class-mean prototypes; cosine similarity, higher is more similar",
            "threshold": "unchanged E1 select_operating_point; inner known confidence 5th percentile",
            "aggregation": "aggregate_reports plus equal-weight outer-fold macro means; worst fold selected by detection-macro-F1",
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
        "folds": fold_outputs,
        "configs": configs_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--riemannian-max-iter", type=int, default=50)
    parser.add_argument("--riemannian-tol", type=float, default=1e-7)
    parser.add_argument("--time-protocol", choices=TIME_PROTOCOLS, default="forward")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = OUTPUT_DIR / f"e2_{args.time_protocol}_results.json"
    if args.outer_folds < 2 or not 2 <= args.inner_folds <= 3:
        parser.error("outer-folds>=2 且 inner-folds 范围为 2–3")
    if args.batch_size < 1 or args.riemannian_max_iter < 1 or args.riemannian_tol <= 0:
        parser.error("batch-size/max-iter 必须为正，riemannian-tol 必须大于 0")
    if not torch.cuda.is_available():
        parser.error("本实验要求 CUDA GPU")
    result = evaluate_e2(args, torch.device("cuda"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "configs": {
            name: {
                "aggregate": value["aggregate"],
                "macro_mean": value["macro_mean"],
                "worst_fold": value["worst_fold"],
            }
            for name, value in result["configs"].items()
        },
    }, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
