"""训练折内 raw-PGD 对抗 Ridge 的嵌套开集验收。"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.open_set_improvement.core import (
    COV_COLUMNS,
    ROOT,
    TaskSpecificModel,
    confidence_values,
    resolve_cache,
    select_operating_point,
)
from src.open_set_calibration.core import build_submission
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout
from src.raw_eeg_anchor.run import grouped_sample
from src.raw_eeg_defense.gpu_attack import GPUCovExtractor
from src.raw_eeg_defense.run import (
    ATTACKS,
    aggregate,
    competition_metrics,
    load_raw_and_fingerprints,
    open_set_predictions,
    predict_subset,
    smoke_indices,
)
from src.raw_eeg_robust.adversarial import (
    adversarial_original_features,
    fit_adversarial_ridge,
)

OUTPUT_DIR = ROOT / "research" / "raw_eeg_robust"
CACHE_VERSION = "adversarial-ridge-stratified-random-target-v2"
METHODS = ("direct", "adversarial_ridge")
TARGETED_PER_GROUP = 2


def mixed_training_features(
    cache_dir: Path,
    cache_name: str,
    model: TaskSpecificModel,
    extractor: GPUCovExtractor,
    metadata: pd.DataFrame,
    selected: np.ndarray,
    raw: np.ndarray,
    tasks: np.ndarray,
    targets: np.ndarray,
    pool: ProcessPoolExecutor,
    steps: int,
    seed: int,
    batch_size: int,
    targeted_per_group: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    targeted_positions = []
    for _, group in metadata.iloc[selected].groupby(["subject", "task"], sort=False):
        group_indices = group.index.to_numpy()
        positions = np.flatnonzero(np.isin(selected, group_indices))
        rng = np.random.default_rng(seed + int(group_indices.sum()))
        targeted_positions.extend(rng.choice(positions, size=targeted_per_group, replace=False))
    targeted_positions = np.asarray(sorted(targeted_positions), dtype=int)
    untargeted_positions = np.setdiff1d(np.arange(len(selected)), targeted_positions)
    outputs = []
    for label, positions, targeted in (
        ("untargeted", untargeted_positions, False),
        ("targeted", targeted_positions, True),
    ):
        full, window, diagnostics = cached_adversarial_features(
            cache_dir, f"{cache_name}_{label}", model, extractor,
            raw[positions], tasks[positions], targets[positions], selected[positions],
            20.0, targeted, steps, seed + int(targeted), pool, batch_size,
        )
        outputs.append((label, positions, full, window, diagnostics))
    full = np.empty((len(selected), outputs[0][2].shape[1]), dtype=np.float32)
    window = np.empty((len(selected), 2, outputs[0][3].shape[2]), dtype=np.float32)
    diagnostics = {"targeted_per_group": targeted_per_group, "parts": {}}
    for label, positions, adv_full, adv_window, part_diagnostics in outputs:
        full[positions] = adv_full
        window[positions] = adv_window
        diagnostics["parts"][label] = part_diagnostics
    return full, window, diagnostics


def cached_adversarial_features(
    cache_dir: Path,
    cache_name: str,
    model: TaskSpecificModel,
    extractor: GPUCovExtractor,
    raw: np.ndarray,
    tasks: np.ndarray,
    targets: np.ndarray,
    source_indices: np.ndarray,
    snr_db: float,
    targeted: bool,
    steps: int,
    seed: int,
    pool: ProcessPoolExecutor,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    path = cache_dir / f"{cache_name}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            diagnostics = json.loads(str(saved["diagnostics"]))
            valid = (
                str(saved["version"]) == CACHE_VERSION
                and int(saved["steps"]) == steps
                and float(saved["snr_db"]) == snr_db
                and bool(saved["targeted"]) == targeted
                and np.array_equal(saved["targets"], targets)
                and np.array_equal(saved["source_indices"], source_indices)
                and saved["full"].shape[0] == len(raw)
            )
            if valid:
                print(f"attack cache: reuse {path.name}", flush=True)
                return saved["full"], saved["window"], diagnostics
    full, window, diagnostics = adversarial_original_features(
        model, extractor, raw, tasks, targets, snr_db, targeted,
        steps, seed, pool, batch_size,
    )
    temporary = path.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        version=np.asarray(CACHE_VERSION),
        steps=np.asarray(steps),
        snr_db=np.asarray(snr_db),
        targeted=np.asarray(targeted),
        targets=targets,
        source_indices=source_indices,
        full=full,
        window=window,
        diagnostics=np.asarray(json.dumps(diagnostics)),
    )
    temporary.replace(path)
    return full, window, diagnostics


def report_from_scores(
    true: np.ndarray,
    clean_scores: np.ndarray,
    adversarial_scores: np.ndarray,
    unknown_scores: np.ndarray,
    classes: np.ndarray,
    operating_point: dict,
) -> dict:
    clean_pred, clean_confidence = open_set_predictions(
        clean_scores, classes, operating_point
    )
    adversarial_pred, _ = open_set_predictions(
        adversarial_scores, classes, operating_point
    )
    unknown_pred, unknown_confidence = open_set_predictions(
        unknown_scores, classes, operating_point
    )
    return competition_metrics(
        true, clean_pred, adversarial_pred, unknown_pred,
        clean_confidence, unknown_confidence,
    )


def acceptance(results: dict) -> dict:
    candidate = "adversarial_ridge"
    baseline = "direct"
    primary = results["attacks"]["untargeted_snr20"]
    row = primary[candidate]
    base = primary[baseline]
    fold_scores = np.asarray([
        fold["attacks"]["untargeted_snr20"]["methods"][candidate]["official_score"]
        - fold["attacks"]["untargeted_snr20"]["methods"][baseline]["official_score"]
        for fold in results["folds"]
    ])
    checks = {
        "score_gain_at_least_2": row["official_score"] - base["official_score"] >= 2.0,
        "A_gain_at_least_0.05": row["A"] - base["A"] >= 0.05,
        "RA_at_least_0.60_and_gain_0.03": row["RA"] >= 0.60 and row["RA"] - base["RA"] >= 0.03,
        "ASafeR_at_least_0.90": row["ASafeR"] >= 0.90,
        "C_at_least_0.90_and_drop_at_most_0.02": row["C"] >= 0.90 and row["C"] - base["C"] >= -0.02,
        "U_at_least_0.85_and_drop_at_most_0.02": row["U"] >= 0.85 and row["U"] - base["U"] >= -0.02,
        "unknown_f1_drop_at_most_0.01": row["unknown_f1"] - base["unknown_f1"] >= -0.01,
        "AUROC_drop_at_most_0.005": row["unknown_auroc"] - base["unknown_auroc"] >= -0.005,
        "four_of_five_folds_not_worse": int(np.sum(fold_scores >= 0)) >= 4,
        "worst_fold_score_drop_at_most_1": float(fold_scores.min()) >= -1.0,
    }
    for attack_name in ("untargeted_snr30", "targeted_snr20"):
        attack = results["attacks"][attack_name]
        checks[f"{attack_name}_ASafeR_at_least_0.90"] = attack[candidate]["ASafeR"] >= 0.90
        checks[f"{attack_name}_A_drop_at_most_0.01"] = (
            attack[candidate]["A"] - attack[baseline]["A"] >= -0.01
        )
    return {
        "candidate": candidate,
        "passed": bool(all(checks.values())),
        "score_delta": row["official_score"] - base["official_score"],
        "A_delta": row["A"] - base["A"],
        "RA_delta": row["RA"] - base["RA"],
        "fold_score_deltas": fold_scores.tolist(),
        "checks": checks,
        "adaptive_attack_required_before_submission": bool(all(checks.values())),
    }


def predict_test(
    cache: Path,
    cache_dir: Path,
    output: Path,
    workers: int,
    batch_size: int,
) -> None:
    device = torch.device("cuda")
    train_metadata = pd.read_csv(cache / "train_metadata.csv")
    train_x = np.load(cache / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS]
    train_window = np.load(cache / "train_window2_features.npy", mmap_mode="r").reshape(
        len(train_metadata), 2, -1
    )[:, :, COV_COLUMNS]
    subjects = train_metadata["subject"].to_numpy()
    tasks = train_metadata["task"].to_numpy()
    fit_mask = np.ones(len(train_metadata), dtype=bool)
    test_metadata = pd.read_csv(
        cache / "test_metadata.csv", usecols=["sample", "task", "path"]
    )
    test_x = np.load(cache / "test_features.npy", mmap_mode="r")[:, COV_COLUMNS]
    test_window = np.load(cache / "test_window2_features.npy", mmap_mode="r").reshape(
        len(test_metadata), 2, -1
    )[:, :, COV_COLUMNS]
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        extractor = GPUCovExtractor(device)
        baseline = TaskSpecificModel(device).fit(
            train_x, train_window, subjects, tasks, fit_mask
        )
        selected = grouped_sample(train_metadata, np.flatnonzero(fit_mask), 8)
        raw, _ = load_raw_and_fingerprints(train_metadata, selected, keep_raw=True)
        adv_full, adv_window, attack_diagnostics = mixed_training_features(
            cache_dir, "final_train", baseline, extractor, train_metadata, selected,
            raw, tasks[selected], subjects[selected], pool, 10,
            20291107 + 9000000, batch_size, 2,
        )
        robust, fit_diagnostics = fit_adversarial_ridge(
            baseline, train_x, train_window, subjects, tasks, fit_mask,
            selected, adv_full, adv_window,
        )
        scores = robust.predict_scores(
            test_x, test_window, test_metadata["task"].to_numpy()
        )["fused"]

    results_path = OUTPUT_DIR / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    points = [fold["operating_points"]["adversarial_ridge"] for fold in results["folds"]]
    score_name = max(set(point["score_name"] for point in points), key=lambda name: sum(
        point["score_name"] == name for point in points
    ))
    thresholds = [point["threshold"] for point in points if point["score_name"] == score_name]
    threshold = float(np.median(thresholds))
    predicted, _ = open_set_predictions(
        scores, robust.classes, {"score_name": score_name, "threshold": threshold}
    )
    template = pd.read_csv(ROOT / "Brainprint_Competition" / "test.csv")
    submission = build_submission(template, test_metadata, predicted)
    submission.to_csv(output, index=False)
    print(json.dumps({
        "output": str(output),
        "rows": len(submission),
        "score_name": score_name,
        "threshold": threshold,
        "label_counts": pd.Series(predicted).value_counts().sort_index().to_dict(),
        "attack": attack_diagnostics,
        "fit": fit_diagnostics,
    }, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.predict_test:
        if not torch.cuda.is_available():
            parser.error("raw-PGD Test 推理要求 CUDA GPU")
        cache = resolve_cache()
        output = args.output or ROOT / "research" / "submission_raw_pgd_adversarial_ridge.csv"
        predict_test(
            cache, cache / "raw_eeg_robust" / "test_final",
            output, args.workers, args.batch_size,
        )
        return
    if not torch.cuda.is_available():
        parser.error("本实验要求 CUDA GPU")
    device = torch.device("cuda")
    cache = resolve_cache()
    cache_dir = cache / "raw_eeg_robust" / ("smoke" if args.smoke else "full")
    cache_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(cache / "train_metadata.csv")
    x = np.load(cache / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS]
    window = np.load(cache / "train_window2_features.npy", mmap_mode="r").reshape(
        len(metadata), 2, -1
    )[:, :, COV_COLUMNS]
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    output = args.output or (
        cache_dir / "smoke_results.json" if args.smoke else OUTPUT_DIR / "results.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = output.with_name(f"{output.stem}.partial.json")
    steps = 2 if args.smoke else 10
    per_group = 2 if args.smoke else 8
    targeted_per_group = min(TARGETED_PER_GROUP, per_group - 1)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        extractor = GPUCovExtractor(device)
        forward = forward_holdout(metadata)
        inner = inner_holdout(metadata)
        all_subjects = np.unique(subjects)
        outer_groups = np.array_split(np.random.default_rng(42).permutation(all_subjects), 5)
        if args.smoke:
            outer_groups = outer_groups[:1]
        fold_outputs = []
        if not args.smoke and partial_output.exists():
            fold_outputs = json.loads(partial_output.read_text(encoding="utf-8"))["folds"]
            for index, fold in enumerate(fold_outputs):
                if fold["outer_fold"] != index + 1 or fold["unknown_subjects"] != outer_groups[index].tolist():
                    raise RuntimeError("断点文件与固定折分不一致")
            print(f"resume: {len(fold_outputs)} fold(s)", flush=True)

        def save_checkpoint() -> None:
            if not args.smoke:
                partial_output.write_text(json.dumps({
                    "complete": False, "folds": fold_outputs,
                }, ensure_ascii=False, indent=2), encoding="utf-8")

        for outer_index, unknown_subjects in enumerate(outer_groups):
            if outer_index < len(fold_outputs) and set(fold_outputs[outer_index]["attacks"]) == {row[0] for row in ATTACKS}:
                print(f"outer={outer_index + 1}: reuse complete checkpoint", flush=True)
                continue
            known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
            outer_known = np.isin(subjects, known_subjects)
            outer_fit = outer_known & ~forward
            baseline_inner_known = []
            baseline_inner_unknown = []
            robust_inner_known = []
            robust_inner_unknown = []
            inner_training = []
            inner_order = np.random.default_rng(100 + outer_index).permutation(known_subjects)
            for inner_index, inner_unknown_subjects in enumerate(np.array_split(inner_order, 2)):
                inner_enrolled = np.setdiff1d(known_subjects, inner_unknown_subjects)
                inner_known_mask = np.isin(subjects, inner_enrolled)
                inner_fit = inner_known_mask & ~forward & ~inner
                inner_eval_known = np.flatnonzero(inner_known_mask & inner)
                inner_eval_unknown = np.flatnonzero(
                    np.isin(subjects, inner_unknown_subjects) & ~forward
                )
                baseline = TaskSpecificModel(device).fit(
                    x, window, subjects, tasks, inner_fit
                )
                baseline_inner_known.append(predict_subset(
                    baseline, x, window, tasks, inner_eval_known
                ))
                baseline_inner_unknown.append(predict_subset(
                    baseline, x, window, tasks, inner_eval_unknown
                ))
                fit_indices = np.flatnonzero(inner_fit)
                selected = grouped_sample(metadata, fit_indices, per_group)
                raw, _ = load_raw_and_fingerprints(metadata, selected, keep_raw=True)
                adv_full, adv_window, attack_diagnostics = mixed_training_features(
                    cache_dir, f"outer{outer_index + 1}_inner{inner_index + 1}_train",
                    baseline, extractor, metadata, selected, raw, tasks[selected],
                    subjects[selected], pool, steps,
                    20291107 + outer_index * 1000000 + inner_index * 300000,
                    args.batch_size, targeted_per_group,
                )
                robust, fit_diagnostics = fit_adversarial_ridge(
                    baseline, x, window, subjects, tasks, inner_fit,
                    selected, adv_full, adv_window,
                )
                robust_inner_known.append(predict_subset(
                    robust, x, window, tasks, inner_eval_known
                ))
                robust_inner_unknown.append(predict_subset(
                    robust, x, window, tasks, inner_eval_unknown
                ))
                inner_training.append({
                    "inner_fold": inner_index + 1,
                    "n_adversarial": int(len(selected)),
                    "fit": fit_diagnostics,
                    "attack": attack_diagnostics,
                })
                print(f"outer={outer_index + 1} inner={inner_index + 1} robust fit complete", flush=True)
                del baseline, robust, raw, adv_full, adv_window
                torch.cuda.empty_cache()

            baseline_point = select_operating_point(
                baseline_inner_known, baseline_inner_unknown
            )
            robust_point = select_operating_point(
                robust_inner_known, robust_inner_unknown
            )
            outer_eval_known = np.flatnonzero(outer_known & forward)
            outer_eval_unknown = np.flatnonzero(np.isin(subjects, unknown_subjects))
            if args.smoke:
                outer_eval_known = smoke_indices(metadata, outer_eval_known)
                outer_eval_unknown = smoke_indices(metadata, outer_eval_unknown)

            baseline = TaskSpecificModel(device).fit(x, window, subjects, tasks, outer_fit)
            fit_indices = np.flatnonzero(outer_fit)
            selected = grouped_sample(metadata, fit_indices, per_group)
            train_raw, _ = load_raw_and_fingerprints(metadata, selected, keep_raw=True)
            train_full, train_window, train_attack = mixed_training_features(
                cache_dir, f"outer{outer_index + 1}_train",
                baseline, extractor, metadata, selected, train_raw, tasks[selected],
                subjects[selected], pool, steps,
                20291107 + outer_index * 1000000 + 900000,
                args.batch_size, targeted_per_group,
            )
            robust, outer_fit_diagnostics = fit_adversarial_ridge(
                baseline, x, window, subjects, tasks, outer_fit,
                selected, train_full, train_window,
            )
            del train_raw, train_full, train_window

            baseline_clean = predict_subset(baseline, x, window, tasks, outer_eval_known)
            baseline_unknown = predict_subset(baseline, x, window, tasks, outer_eval_unknown)
            robust_clean = predict_subset(robust, x, window, tasks, outer_eval_known)
            robust_unknown = predict_subset(robust, x, window, tasks, outer_eval_unknown)
            clean_raw, _ = load_raw_and_fingerprints(metadata, outer_eval_known, keep_raw=True)
            if outer_index < len(fold_outputs):
                fold = fold_outputs[outer_index]
                if fold["operating_points"] != {
                    "direct": baseline_point, "adversarial_ridge": robust_point
                }:
                    raise RuntimeError("断点阈值与当前内层选择不一致")
            else:
                fold = {
                    "outer_fold": outer_index + 1,
                    "unknown_subjects": unknown_subjects.tolist(),
                    "registered_subjects": known_subjects.tolist(),
                    "operating_points": {
                        "direct": baseline_point,
                        "adversarial_ridge": robust_point,
                    },
                    "inner_training": inner_training,
                    "outer_training": {
                        "n_adversarial": int(len(selected)),
                        "fit": outer_fit_diagnostics,
                        "attack": train_attack,
                    },
                    "attacks": {},
                }
                fold_outputs.append(fold)
                save_checkpoint()

            attacks = ATTACKS[:1] if args.smoke else ATTACKS
            for attack_index, (attack_name, snr_db, targeted) in enumerate(attacks):
                if attack_name in fold["attacks"]:
                    print(f"outer={outer_index + 1} {attack_name}: reuse checkpoint", flush=True)
                    continue
                adv_full, adv_window, attack_diagnostics = cached_adversarial_features(
                    cache_dir, f"outer{outer_index + 1}_eval_{attack_name}",
                    baseline, extractor, clean_raw, tasks[outer_eval_known],
                    subjects[outer_eval_known], outer_eval_known,
                    snr_db, targeted, steps,
                    20261107 + outer_index * 1000000 + attack_index * 200000,
                    pool, args.batch_size,
                )
                evaluation_tasks = tasks[outer_eval_known]
                baseline_adversarial = baseline.predict_scores(
                    adv_full, adv_window, evaluation_tasks
                )["fused"]
                robust_adversarial = robust.predict_scores(
                    adv_full, adv_window, evaluation_tasks
                )["fused"]
                reports = {
                    "direct": report_from_scores(
                        subjects[outer_eval_known], baseline_clean, baseline_adversarial,
                        baseline_unknown, baseline.classes, baseline_point,
                    ),
                    "adversarial_ridge": report_from_scores(
                        subjects[outer_eval_known], robust_clean, robust_adversarial,
                        robust_unknown, robust.classes, robust_point,
                    ),
                }
                fold["attacks"][attack_name] = {
                    "methods": reports,
                    "attack": attack_diagnostics,
                    "fixed_surrogate": "outer clean direct model; identical attacked features for both methods",
                }
                save_checkpoint()
                print(f"outer={outer_index + 1} {attack_name} complete", flush=True)
                del adv_full, adv_window
            del baseline, robust, clean_raw
            torch.cuda.empty_cache()

    result = {
        "protocol": {
            "data": "Train Session1 only; 5-fold subject holdout; nested threshold selection",
            "candidate": "one frozen candidate: clean all + within-fit raw-PGD augmentation; refit Ridge components only",
            "training_attack": {
                "snr_db": 20.0, "targeted": "mixed",
                "steps": steps,
                "per_subject_task": per_group,
                "targeted_per_subject_task": targeted_per_group,
                "selection": "last targeted trials and remaining first trials in each subject/task within current fit mask",
            },
            "evaluation_attacks": [
                {"name": name, "snr_db": snr, "targeted": targeted, "steps": steps}
                for name, snr, targeted in (ATTACKS[:1] if args.smoke else ATTACKS)
            ],
            "evaluation_surrogate": "clean direct model; candidate receives identical exact SciPy features",
            "neighbor_components": "retain clean fit; only full/window/low Ridge are refit",
            "model_selection": "no outer-fold retuning; no test labels or clean/adv oracle",
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "smoke": args.smoke,
        },
        "folds": fold_outputs,
    }
    result["attacks"] = {
        attack_name: {
            method: aggregate([
                fold["attacks"][attack_name]["methods"][method] for fold in fold_outputs
            ])
            for method in METHODS
        }
        for attack_name in fold_outputs[0]["attacks"]
    }
    if not args.smoke:
        result["acceptance"] = acceptance(result)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.smoke:
        partial_output.unlink(missing_ok=True)
    print(json.dumps({
        "output": str(output),
        "attacks": result["attacks"],
        "acceptance": result.get("acceptance"),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
