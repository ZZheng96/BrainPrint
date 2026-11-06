"""full/base × 无对齐/无监督目标域 z-score 的嵌套单因素实验。"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.open_set_calibration.core import (
    ALIGNMENTS,
    ALL_CONFIGS,
    CALIBRATION_VARIANTS,
    SCORE_NORMALIZATIONS,
    aggregate_reports,
    build_submission,
    config_name,
    predict_calibration_scores,
)
from src.open_set_improvement.core import (
    COV_COLUMNS,
    ROOT,
    SCORE_NAMES,
    TaskSpecificModel,
    confidence_values,
    open_set_report,
    resolve_cache,
    row_zscore,
    select_operating_point,
    subject_bootstrap,
)
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

OUTPUT_DIR = ROOT / "research" / "open_set_calibration"


def choose_score_name(selections: list[dict]) -> str:
    """只用内层选择结果确定最终分数，固定顺序处理并列。"""
    counts = Counter(row["score_name"] for row in selections)
    return min(
        SCORE_NAMES,
        key=lambda name: (
            -counts[name],
            -np.mean([
                row["inner_detection_macro_f1"] for row in selections
                if row["score_name"] == name
            ] or [0]),
            SCORE_NAMES.index(name),
        ),
    )


def source_oof_scores(cache: Path, variant: str) -> np.ndarray:
    full = np.load(cache / "train_blocked_oof_scores_cov_alpha100.npy")
    if variant == "full":
        return full
    if variant == "base":
        window = np.load(cache / "train_blocked_oof_scores_window2.npy")
        return 0.5 * full + 0.5 * window
    raise ValueError(variant)


def generate_candidate(
    config: str,
    cache: Path,
    x: np.ndarray,
    window: np.ndarray,
    subjects: np.ndarray,
    tasks: np.ndarray,
    fold_outputs: list[dict],
    device: torch.device,
    output_dir: Path,
    batch_size: int,
) -> dict:
    """显式指定配置才生成实验候选；正式提交始终不动。"""
    if config not in ALL_CONFIGS:
        raise ValueError(config)
    variant = next(name for name in CALIBRATION_VARIANTS if config.startswith(f"{name}_"))
    alignment = next(
        name for name in ALIGNMENTS if config.startswith(f"{variant}_{name}")
    )
    score_normalization = (
        "row_zscore" if config.endswith("_row_zscore") else "none"
    )
    selections = [fold["selections"][config] for fold in fold_outputs]
    score_name = choose_score_name(selections)
    oof = source_oof_scores(cache, variant)
    if score_normalization == "row_zscore":
        oof = row_zscore(torch.as_tensor(oof, dtype=torch.float32)).numpy()
    threshold = float(np.quantile(confidence_values(oof)[score_name], 0.05))

    model = TaskSpecificModel(device).fit(
        x, window, subjects, tasks, np.ones(len(subjects), dtype=bool)
    )
    test_metadata = pd.read_csv(cache / "test_metadata.csv")
    test_x = np.asarray(np.load(cache / "test_features.npy", mmap_mode="r")[:, COV_COLUMNS])
    test_window = np.asarray(
        np.load(cache / "test_window2_features.npy", mmap_mode="r")[:, COV_COLUMNS]
    ).reshape(len(test_metadata), 2, -1)
    scores, alignment_diagnostics = predict_calibration_scores(
        model, test_x, test_window, test_metadata["task"].to_numpy(), alignment,
        batch_size, score_normalization,
    )
    selected_scores = scores[config]
    confidence = confidence_values(selected_scores)[score_name]
    predicted = model.classes[selected_scores.argmax(axis=1)].copy()
    predicted[confidence < threshold] = 0
    template = pd.read_csv(ROOT / "Brainprint_Competition" / "test.csv")
    submission = build_submission(template, test_metadata, predicted)
    path = output_dir / f"submission_{config}.csv"
    submission.to_csv(path, index=False)
    return {
        "path": str(path),
        "config": config,
        "score_normalization": score_normalization,
        "score_name": score_name,
        "score_selection_source": "inner folds only",
        "threshold_source": "5th percentile of all-subject source blocked-OOF known scores",
        "threshold": threshold,
        "target_alignment": "all unlabeled test trials within each task; no session split",
        "alignment_diagnostics": alignment_diagnostics,
        "metadata_rejection": False,
        "rejected_rows": int(np.sum(predicted == 0)),
        "total_rows": int(len(predicted)),
        "rejected_fraction": float(np.mean(predicted == 0)),
        "confidence_quantiles": {
            str(q): float(np.quantile(confidence, q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "class_counts": {
            str(label): int(count)
            for label, count in zip(*np.unique(predicted, return_counts=True))
        },
        "test_accuracy_claim": "unknown; experimental candidate, formal answer untouched",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--alignments", choices=ALIGNMENTS, nargs="+", default=list(ALIGNMENTS))
    parser.add_argument("--score-normalization", choices=SCORE_NORMALIZATIONS, default="none")
    parser.add_argument("--generate-candidate", choices=ALL_CONFIGS)
    parser.add_argument("--from-results", type=Path, help="复用已完成的嵌套结果，仅生成候选")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "calibration_results.json")
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
    if args.from_results:
        if not args.generate_candidate:
            parser.error("--from-results 必须同时指定 --generate-candidate")
        result = json.loads(args.from_results.read_text(encoding="utf-8"))
        result["candidate"] = generate_candidate(
            args.generate_candidate, cache, x, window, subjects, tasks,
            result["folds"], device, args.output.parent, args.batch_size,
        )
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result["candidate"], ensure_ascii=False, indent=2), flush=True)
        return

    active_configs = tuple(
        config_name(variant, alignment, args.score_normalization)
        for variant in CALIBRATION_VARIANTS
        for alignment in args.alignments
    )
    forward = forward_holdout(metadata)
    inner = inner_holdout(metadata)
    all_subjects = np.unique(subjects)
    shuffled = np.random.default_rng(42).permutation(all_subjects)
    outer_groups = np.array_split(shuffled, args.outer_folds)
    fold_outputs = []
    records_by_config = {config: [] for config in active_configs}
    subjects_by_config = {config: [] for config in active_configs}

    for outer_index, unknown_subjects in enumerate(outer_groups):
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        outer_known = np.isin(subjects, known_subjects)
        outer_fit = outer_known & ~forward
        inner_scores = {
            config: {"known": [], "unknown": []} for config in active_configs
        }
        inner_alignment_diagnostics = {config: [] for config in active_configs}
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
            for alignment in args.alignments:
                scores, diagnostics = predict_calibration_scores(
                    model, x[inner_eval], window[inner_eval], tasks[inner_eval],
                    alignment, args.batch_size, args.score_normalization,
                )
                for config, values in scores.items():
                    inner_scores[config]["known"].append(values[: len(inner_eval_known)])
                    inner_scores[config]["unknown"].append(values[len(inner_eval_known) :])
                    inner_alignment_diagnostics[config].append(diagnostics)
            print(
                f"outer={outer_index + 1} inner={inner_index + 1} "
                f"known={len(inner_eval_known)} unknown={len(inner_eval_unknown)}",
                flush=True,
            )
            del model

        selections = {
            config: select_operating_point(
                inner_scores[config]["known"], inner_scores[config]["unknown"]
            )
            for config in active_configs
        }
        outer_eval_known = np.flatnonzero(outer_known & forward)
        outer_eval_unknown = np.flatnonzero(np.isin(subjects, unknown_subjects))
        outer_eval = np.concatenate([outer_eval_known, outer_eval_unknown])
        model = TaskSpecificModel(device).fit(x, window, subjects, tasks, outer_fit)
        fold_report = {
            "outer_fold": outer_index + 1,
            "unknown_subjects": unknown_subjects.tolist(),
            "registered_subjects": known_subjects.tolist(),
            "selections": selections,
            "inner_alignment_diagnostics": inner_alignment_diagnostics,
            "outer_alignment_diagnostics": {},
            "results": {},
        }
        for alignment in args.alignments:
            scores, diagnostics = predict_calibration_scores(
                model, x[outer_eval], window[outer_eval], tasks[outer_eval],
                alignment, args.batch_size, args.score_normalization,
            )
            for config, values in scores.items():
                fold_report["outer_alignment_diagnostics"][config] = diagnostics
                known_scores = values[: len(outer_eval_known)]
                unknown_scores = values[len(outer_eval_known) :]
                report, records = open_set_report(
                    subjects[outer_eval_known], known_scores, unknown_scores,
                    model.classes, selections[config]["score_name"],
                    selections[config]["threshold"],
                )
                fold_report["results"][config] = report
                records_by_config[config].append(records)
                subject_record = subjects[outer_eval]
                if len(subject_record) != len(records["is_known"]):
                    raise RuntimeError("逐折被试索引与评估记录长度不一致")
                subjects_by_config[config].append(subject_record)
                print(f"outer={outer_index + 1} {config}: {report}", flush=True)
        fold_outputs.append(fold_report)
        del model
        torch.cuda.empty_cache()

    configs_report = {}
    for config in active_configs:
        records = {
            key: np.concatenate([fold[key] for fold in records_by_config[config]])
            for key in records_by_config[config][0]
        }
        record_subjects = np.concatenate(subjects_by_config[config])
        configs_report[config] = {
            "aggregate": aggregate_reports(
                records, [fold["results"][config] for fold in fold_outputs]
            ),
            "subject_bootstrap_95pct": subject_bootstrap(
                records, record_subjects, args.bootstrap
            ) if args.bootstrap else {},
        }

    result = {
        "protocol": {
            "data": "Train Session1 only; pseudo-target shift is temporal/subject holdout, not cross-session evidence",
            "outer": f"{args.outer_folds}-fold subject holdout; held-out subjects are unknown",
            "known_outer_test": "last 20% trials within each registered subject/task",
            "inner": f"{args.inner_folds} pseudo-unknown groups; source first 60%, target registered 60–80% and pseudo-unknown first 80%",
            "variants": {
                "full": "4-second covariance Ridge",
                "base": "equal 4-second + two-window Ridge",
            },
            "none": "raw target features; exactly the full/base branches of the prior nested protocol",
            "target_zscore": "per-task unsupervised target feature mean/std mapped to source training mean/std; scale ratio clipped to [0.5, 2]",
            "score_normalization": args.score_normalization,
            "row_zscore": "per-trial class-score mean/std normalization before confidence; same transform applied to source OOF threshold",
            "target_statistics": "entire unlabeled inner/outer evaluation batch, including known and unknown trials; no identity/session labels",
            "threshold": "inner known confidence 5th percentile; outer labels never used for selection or calibration",
            "aggregation": "C/U/F1 pooled by trial; AUROC/AUPR/EER macro-averaged over outer folds",
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
        "folds": fold_outputs,
        "configs": configs_report,
    }
    if args.generate_candidate:
        result["candidate"] = generate_candidate(
            args.generate_candidate, cache, x, window, subjects, tasks,
            fold_outputs, device, args.output.parent, args.batch_size,
        )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "configs": {name: value["aggregate"] for name, value in configs_report.items()},
        "candidate": result.get("candidate"),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
