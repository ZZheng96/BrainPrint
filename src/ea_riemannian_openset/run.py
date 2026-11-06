"""E1：full Ridge 下的 EA / 仿射不变 Riemannian 重居中嵌套开集评估。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.ea_riemannian_openset.core import (
    RECENTER_METHODS,
    fit_recentered_model,
    predict_recentered_full_scores,
)
from src.open_set_calibration.core import aggregate_reports, predict_calibration_scores
from src.open_set_improvement.core import (
    COV_COLUMNS,
    ROOT,
    TaskSpecificModel,
    open_set_report,
    resolve_cache,
    select_operating_point,
    subject_bootstrap,
)
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

OUTPUT_DIR = ROOT / "research" / "ea_riemannian_openset"
BASELINE_ALIGNMENTS = ("none", "target_zscore")
ALIGNMENTS = BASELINE_ALIGNMENTS + RECENTER_METHODS
CONFIGS = tuple(f"full_{alignment}" for alignment in ALIGNMENTS)
TIME_PROTOCOLS = ("forward", "reverse")
MACRO_METRICS = (
    "C",
    "direct_C_without_rejection",
    "U",
    "known_acceptance",
    "unknown_auroc",
    "known_aupr",
    "unknown_aupr",
    "eer",
    "macro_f1",
    "unknown_f1",
    "detection_macro_f1",
)


def time_holdouts(metadata: pd.DataFrame, protocol: str = "forward") -> tuple[np.ndarray, np.ndarray]:
    """返回注册者 outer/inner 时间掩码；reverse 是 forward 的严格时间镜像。"""
    if protocol == "forward":
        return forward_holdout(metadata), inner_holdout(metadata)
    if protocol != "reverse":
        raise ValueError(protocol)
    max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    trial = metadata["trial"].to_numpy()
    outer = trial <= 0.2 * max_trial
    inner = (trial > 0.2 * max_trial) & (trial <= 0.4 * max_trial)
    return outer, inner


def fit_predict_split(
    x: np.ndarray,
    window: np.ndarray,
    subjects: np.ndarray,
    tasks: np.ndarray,
    fit_mask: np.ndarray,
    eval_indices: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 256,
    riemannian_max_iter: int = 50,
    riemannian_tol: float = 1e-7,
) -> tuple[dict[str, np.ndarray], dict[str, dict], np.ndarray]:
    """固定评估四个 full 配置；eval_indices 的已知/未知拼接顺序原样保留。"""
    eval_x = np.asarray(x[eval_indices])
    eval_window = np.asarray(window[eval_indices])
    eval_tasks = np.asarray(tasks)[eval_indices]
    outputs: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict] = {}

    baseline_model = TaskSpecificModel(device).fit(
        x, window, subjects, tasks, fit_mask
    )
    classes = baseline_model.classes.copy()
    for alignment in BASELINE_ALIGNMENTS:
        scores, alignment_diagnostics = predict_calibration_scores(
            baseline_model,
            eval_x,
            eval_window,
            eval_tasks,
            alignment,
            batch_size=batch_size,
        )
        outputs[f"full_{alignment}"] = scores[f"full_{alignment}"]
        diagnostics[f"full_{alignment}"] = alignment_diagnostics
    del baseline_model

    for method in RECENTER_METHODS:
        model, source_aligner = fit_recentered_model(
            x,
            window,
            subjects,
            tasks,
            fit_mask,
            method,
            device,
            batch_size=batch_size,
            max_iter=riemannian_max_iter,
            tol=riemannian_tol,
        )
        if not np.array_equal(model.classes, classes):
            raise RuntimeError("对齐分支与基线的注册类别不一致")
        values, alignment_diagnostics = predict_recentered_full_scores(
            model,
            source_aligner,
            eval_x,
            eval_window,
            eval_tasks,
            batch_size=batch_size,
        )
        outputs[f"full_{method}"] = values
        diagnostics[f"full_{method}"] = alignment_diagnostics
        del model, source_aligner
    return outputs, diagnostics, classes


def macro_mean(fold_reports: list[dict]) -> dict:
    """对所有外层折等权计算宏均值，与逐 Trial pooled aggregate 分开报告。"""
    return {
        metric: float(np.mean([report[metric] for report in fold_reports]))
        for metric in MACRO_METRICS
    }


def worst_fold(fold_outputs: list[dict], config: str) -> dict:
    """以开集 detection-macro-F1 最低者定义最差折，并列阈值作稳定次序。"""
    selected = min(
        fold_outputs,
        key=lambda fold: (
            fold["results"][config]["detection_macro_f1"],
            fold["outer_fold"],
        ),
    )
    return {
        "selection_metric": "minimum detection_macro_f1",
        "outer_fold": int(selected["outer_fold"]),
        "metrics": selected["results"][config],
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
        args.output = OUTPUT_DIR / (
            "e1_results.json" if args.time_protocol == "forward" else "e1_reverse_results.json"
        )
    if args.outer_folds < 2 or not 2 <= args.inner_folds <= 3:
        parser.error("outer-folds>=2 且 inner-folds 范围为 2–3")
    if args.batch_size < 1 or args.riemannian_max_iter < 1 or args.riemannian_tol <= 0:
        parser.error("batch-size/max-iter 必须为正，riemannian-tol 必须大于 0")
    if not torch.cuda.is_available():
        parser.error("本实验要求 CUDA GPU")
    device = torch.device("cuda")
    cache = resolve_cache()
    metadata = pd.read_csv(cache / "train_metadata.csv")
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    x = np.asarray(
        np.load(cache / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS]
    )
    window = np.asarray(
        np.load(cache / "train_window2_features.npy", mmap_mode="r")[:, COV_COLUMNS]
    ).reshape(len(metadata), 2, -1)
    outer_time, inner_time = time_holdouts(metadata, args.time_protocol)
    all_subjects = np.unique(subjects)
    shuffled = np.random.default_rng(42).permutation(all_subjects)
    outer_groups = np.array_split(shuffled, args.outer_folds)
    fold_outputs = []
    records_by_config = {config: [] for config in CONFIGS}
    subjects_by_config = {config: [] for config in CONFIGS}

    for outer_index, unknown_subjects in enumerate(outer_groups):
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        outer_known = np.isin(subjects, known_subjects)
        outer_fit = outer_known & ~outer_time
        inner_scores = {
            config: {"known": [], "unknown": []} for config in CONFIGS
        }
        inner_diagnostics = {config: [] for config in CONFIGS}
        inner_order = np.random.default_rng(100 + outer_index).permutation(known_subjects)
        inner_groups = np.array_split(inner_order, args.inner_folds)
        for inner_index, inner_unknown in enumerate(inner_groups):
            inner_enrolled = np.setdiff1d(known_subjects, inner_unknown)
            inner_known_mask = np.isin(subjects, inner_enrolled)
            inner_fit = inner_known_mask & ~outer_time & ~inner_time
            inner_eval_known = np.flatnonzero(inner_known_mask & inner_time)
            inner_eval_unknown = np.flatnonzero(
                np.isin(subjects, inner_unknown) & ~outer_time
            )
            inner_eval = np.concatenate([inner_eval_known, inner_eval_unknown])
            scores, diagnostics, _ = fit_predict_split(
                x,
                window,
                subjects,
                tasks,
                inner_fit,
                inner_eval,
                device,
                batch_size=args.batch_size,
                riemannian_max_iter=args.riemannian_max_iter,
                riemannian_tol=args.riemannian_tol,
            )
            for config in CONFIGS:
                inner_scores[config]["known"].append(
                    scores[config][: len(inner_eval_known)]
                )
                inner_scores[config]["unknown"].append(
                    scores[config][len(inner_eval_known) :]
                )
                inner_diagnostics[config].append(diagnostics[config])
            print(
                f"outer={outer_index + 1} inner={inner_index + 1} "
                f"known={len(inner_eval_known)} unknown={len(inner_eval_unknown)}",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

        selections = {
            config: select_operating_point(
                inner_scores[config]["known"], inner_scores[config]["unknown"]
            )
            for config in CONFIGS
        }
        outer_eval_known = np.flatnonzero(outer_known & outer_time)
        outer_eval_unknown = np.flatnonzero(np.isin(subjects, unknown_subjects))
        outer_eval = np.concatenate([outer_eval_known, outer_eval_unknown])
        scores, diagnostics, classes = fit_predict_split(
            x,
            window,
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
        for config in CONFIGS:
            known_scores = scores[config][: len(outer_eval_known)]
            unknown_scores = scores[config][len(outer_eval_known) :]
            report, records = open_set_report(
                subjects[outer_eval_known],
                known_scores,
                unknown_scores,
                classes,
                selections[config]["score_name"],
                selections[config]["threshold"],
            )
            fold_report["results"][config] = report
            records_by_config[config].append(records)
            subject_record = subjects[outer_eval]
            if len(subject_record) != len(records["is_known"]):
                raise RuntimeError("逐折被试索引与评估记录长度不一致")
            subjects_by_config[config].append(subject_record)
            print(
                f"outer={outer_index + 1} {config}: "
                f"AUROC={report['unknown_auroc']:.6f} "
                f"macro-F1={report['macro_f1']:.6f}",
                flush=True,
            )
        fold_outputs.append(fold_report)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    configs_report = {}
    for config in CONFIGS:
        records = {
            key: np.concatenate([fold[key] for fold in records_by_config[config]])
            for key in records_by_config[config][0]
        }
        record_subjects = np.concatenate(subjects_by_config[config])
        fold_reports = [fold["results"][config] for fold in fold_outputs]
        configs_report[config] = {
            "aggregate": aggregate_reports(records, fold_reports),
            "macro_mean": macro_mean(fold_reports),
            "worst_fold": worst_fold(fold_outputs, config),
            "subject_bootstrap_95pct": subject_bootstrap(
                records, record_subjects, args.bootstrap
            ) if args.bootstrap else {},
        }

    result = {
        "protocol": {
            "data": "Train Session1 only; temporal/subject pseudo-target, not cross-session evidence",
            "time_protocol": args.time_protocol,
            "outer": f"{args.outer_folds}-fold LOSO; held-out subjects are unknown",
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
            "model": "full 4-second covariance Ridge only; window features passed unchanged but window/base scores are not evaluated",
            "configs": list(CONFIGS),
            "none_target_zscore": "existing TaskSpecificModel and predict_calibration_scores",
            "ea": "per-task/band arithmetic SPD reference; source and target transformed by their own references",
            "riemannian_recenter": "per-task/band affine-invariant geometric mean by whitened tangent fixed-point iterations; not log-Euclidean",
            "source_reference": "fit_mask trials only, separately by task/band",
            "target_reference": "entire unlabeled eval batch, separately by task/band; no subject/session/known-unknown labels",
            "features": "restore shrinkage SPD from weighted log-cov upper triangles; recenter with R^-1/2 C R^-1/2; log-vectorize both source and target",
            "threshold": "inner select_operating_point; known confidence 5th percentile",
            "aggregation": "aggregate_reports plus equal-weight outer-fold macro means; worst fold selected by detection-macro-F1",
            "precision": "SPD eigensolvers and references use batched CUDA float64; Ridge uses existing float32 implementation",
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
        "folds": fold_outputs,
        "configs": configs_report,
    }
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
            for name, value in configs_report.items()
        },
    }, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
