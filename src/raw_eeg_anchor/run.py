"""训练折内学习干净锚点，并审计配对传播方向。"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.open_set_improvement.core import (
    COV_COLUMNS,
    ROOT,
    TaskSpecificModel,
    confidence_values,
    resolve_cache,
    select_operating_point,
)
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout
from src.raw_eeg_anchor.anchor import (
    CleanAnchorModel,
    anchor_features,
    learned_propagation,
    propagation_audit,
)
from src.raw_eeg_defense.gpu_attack import GPUCovExtractor
from src.raw_eeg_defense.run import (
    ATTACKS,
    DATA,
    VIEW_NAMES,
    aggregate,
    asymmetric_propagation,
    attack_known,
    competition_metrics,
    evaluate_attack_methods,
    load_raw_and_fingerprints,
    open_set_predictions,
    pair_positions,
    predict_subset,
    prepare_clean_view_cache,
    smoke_indices,
    view_arrays,
    waveform_neighbors,
)

OUTPUT_DIR = ROOT / "research" / "raw_eeg_anchor"
CACHE_VERSION = "learned-anchor-v1"


def grouped_sample(metadata: pd.DataFrame, indices: np.ndarray, per_group: int) -> np.ndarray:
    selected = []
    for _, group in metadata.iloc[indices].groupby(["subject", "task"], sort=False):
        selected.extend(group.index.to_numpy()[:per_group])
    return np.asarray(selected, dtype=int)


def cached_attack(
    cache_dir: Path,
    cache_name: str,
    model: TaskSpecificModel,
    extractor: GPUCovExtractor,
    raw: np.ndarray,
    tasks: np.ndarray,
    targets: np.ndarray,
    attack_name: str,
    snr_db: float,
    targeted: bool,
    steps: int,
    seed: int,
    pool: ProcessPoolExecutor,
    batch_size: int,
) -> tuple[list[np.ndarray], dict, np.ndarray]:
    path = cache_dir / f"{cache_name}_{attack_name}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            valid = (
                str(saved["version"]) == CACHE_VERSION
                and int(saved["steps"]) == steps
                and np.array_equal(saved["targets"], targets)
                and saved["fingerprints"].shape[0] == len(raw)
            )
            if valid:
                scores = [saved[f"scores_{view}"] for view in range(3)]
                diagnostics = json.loads(str(saved["diagnostics"]))
                return scores, diagnostics, saved["fingerprints"]
    scores, diagnostics, fingerprints = attack_known(
        model, extractor, raw, tasks, targets, snr_db, targeted,
        steps, seed, pool, batch_size,
    )
    np.savez(
        path,
        version=np.asarray(CACHE_VERSION),
        steps=np.asarray(steps),
        targets=targets,
        fingerprints=fingerprints,
        diagnostics=np.asarray(json.dumps(diagnostics)),
        **{f"scores_{view}": scores[view] for view in range(3)},
    )
    return scores, diagnostics, fingerprints


def fit_anchor(
    clean_blocks: list[list[np.ndarray]],
    adversarial_blocks: dict[str, list[list[np.ndarray]]],
    inner_classes: list[np.ndarray],
    operating_points: list[dict],
    device: torch.device,
) -> tuple[CleanAnchorModel, dict]:
    features = []
    labels = []
    for inner_index, scores in enumerate(clean_blocks):
        features.append(anchor_features(scores, inner_classes[inner_index], operating_points))
        labels.append(np.ones(len(scores[0]), dtype=np.float32))
    for blocks in adversarial_blocks.values():
        for inner_index, scores in enumerate(blocks):
            features.append(anchor_features(scores, inner_classes[inner_index], operating_points))
            labels.append(np.zeros(len(scores[0]), dtype=np.float32))
    x = np.concatenate(features)
    y = np.concatenate(labels)
    model = CleanAnchorModel(device).fit(x, y)
    probability = model.predict_proba(x)
    diagnostics = {
        "n_clean": int(np.sum(y == 1)),
        "n_adversarial": int(np.sum(y == 0)),
        "fit_auroc_not_validation": float(roc_auc_score(y, probability)),
        "clean_probability_mean": float(probability[y == 1].mean()),
        "adversarial_probability_mean": float(probability[y == 0].mean()),
    }
    return model, diagnostics


def learned_report(
    anchor: CleanAnchorModel,
    clean_true: np.ndarray,
    clean_scores: list[np.ndarray],
    adversarial_scores: list[np.ndarray],
    unknown_scores: list[np.ndarray],
    classes: np.ndarray,
    operating_points: list[dict],
    neighbors: dict[str, np.ndarray],
) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    n_known = len(clean_true)
    n_unknown = len(unknown_scores[0])
    joined_scores = [
        np.concatenate([clean_scores[view], adversarial_scores[view], unknown_scores[view]])
        for view in range(3)
    ]
    probability = anchor.predict_proba(anchor_features(joined_scores, classes, operating_points))
    direct, direct_confidence = open_set_predictions(joined_scores[0], classes, operating_points[0])
    learned = learned_propagation(direct, probability, neighbors, pair_positions)
    report = competition_metrics(
        clean_true,
        learned[:n_known],
        learned[n_known : 2 * n_known],
        learned[2 * n_known :],
        direct_confidence[:n_known],
        direct_confidence[2 * n_known :],
    )
    detection_true = np.concatenate([np.ones(n_known), np.zeros(n_known)])
    diagnostics = {
        "changed": int(np.sum(learned != direct)),
        "clean_probability_mean": float(probability[:n_known].mean()),
        "adversarial_probability_mean": float(probability[n_known : 2 * n_known].mean()),
        "unknown_probability_mean": float(probability[2 * n_known :].mean()),
        "clean_vs_adversarial_auroc": float(roc_auc_score(
            detection_true, probability[: 2 * n_known]
        )),
        "audit": propagation_audit(direct, learned, clean_true, n_known),
    }
    return report, diagnostics, direct, learned


def acceptance(results: dict) -> dict:
    candidate = "learned_anchor"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("本实验要求 CUDA GPU")
    device = torch.device("cuda")
    cache = resolve_cache()
    cache_dir = cache / "raw_eeg_anchor"
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

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        additional_full, additional_window = prepare_clean_view_cache(metadata, cache, pool)
        full_views, window_views = view_arrays(x, window, additional_full, additional_window)
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
            inner_known = [[] for _ in VIEW_NAMES]
            inner_unknown = [[] for _ in VIEW_NAMES]
            anchor_clean_blocks = []
            anchor_adv_blocks = {name: [] for name, _, _ in ATTACKS}
            inner_classes = []
            inner_order = np.random.default_rng(100 + outer_index).permutation(known_subjects)
            steps = 2 if args.smoke else 10
            for inner_index, inner_unknown_subjects in enumerate(np.array_split(inner_order, 2)):
                inner_enrolled = np.setdiff1d(known_subjects, inner_unknown_subjects)
                inner_known_mask = np.isin(subjects, inner_enrolled)
                inner_fit = inner_known_mask & ~forward & ~inner
                inner_eval_known = np.flatnonzero(inner_known_mask & inner)
                inner_eval_unknown = np.flatnonzero(np.isin(subjects, inner_unknown_subjects) & ~forward)
                model = TaskSpecificModel(device).fit(x, window, subjects, tasks, inner_fit)
                for view in range(3):
                    inner_known[view].append(predict_subset(
                        model, full_views[view], window_views[view], tasks, inner_eval_known
                    ))
                    inner_unknown[view].append(predict_subset(
                        model, full_views[view], window_views[view], tasks, inner_eval_unknown
                    ))
                selected = grouped_sample(metadata, inner_eval_known, 2 if args.smoke else 8)
                anchor_clean_blocks.append([
                    predict_subset(model, full_views[view], window_views[view], tasks, selected)
                    for view in range(3)
                ])
                inner_classes.append(model.classes.copy())
                raw, _ = load_raw_and_fingerprints(metadata, selected, keep_raw=True)
                for attack_index, (attack_name, snr_db, targeted) in enumerate(ATTACKS):
                    scores, _, _ = cached_attack(
                        cache_dir, f"outer{outer_index + 1}_inner{inner_index + 1}",
                        model, extractor, raw, tasks[selected], subjects[selected],
                        attack_name, snr_db, targeted, steps,
                        20271107 + outer_index * 1000000 + inner_index * 300000 + attack_index * 50000,
                        pool, args.batch_size,
                    )
                    anchor_adv_blocks[attack_name].append(scores)
                print(f"outer={outer_index + 1} inner={inner_index + 1} anchor data complete", flush=True)
                del model
            operating_points = [
                select_operating_point(inner_known[view], inner_unknown[view])
                for view in range(3)
            ]
            anchor, anchor_training = fit_anchor(
                anchor_clean_blocks, anchor_adv_blocks, inner_classes, operating_points, device
            )
            anchor_values = np.concatenate([
                confidence_values(scores)[operating_points[0]["score_name"]]
                for scores in inner_known[0]
            ])
            anchor_minimum = float(np.median(anchor_values))
            outer_eval_known = np.flatnonzero(outer_known & forward)
            outer_eval_unknown = np.flatnonzero(np.isin(subjects, unknown_subjects))
            if args.smoke:
                outer_eval_known = smoke_indices(metadata, outer_eval_known)
                outer_eval_unknown = smoke_indices(metadata, outer_eval_unknown)
            model = TaskSpecificModel(device).fit(x, window, subjects, tasks, outer_fit)
            clean_scores = [
                predict_subset(model, full_views[view], window_views[view], tasks, outer_eval_known)
                for view in range(3)
            ]
            unknown_scores = [
                predict_subset(model, full_views[view], window_views[view], tasks, outer_eval_unknown)
                for view in range(3)
            ]
            clean_raw, clean_fingerprints = load_raw_and_fingerprints(
                metadata, outer_eval_known, keep_raw=True
            )
            _, unknown_fingerprints = load_raw_and_fingerprints(
                metadata, outer_eval_unknown, keep_raw=False
            )
            if outer_index < len(fold_outputs):
                fold = fold_outputs[outer_index]
                if fold["operating_points"] != operating_points:
                    raise RuntimeError("断点阈值与当前内层选择不一致")
            else:
                fold = {
                    "outer_fold": outer_index + 1,
                    "unknown_subjects": unknown_subjects.tolist(),
                    "operating_points": operating_points,
                    "anchor_minimum": anchor_minimum,
                    "anchor_training": anchor_training,
                    "attacks": {},
                }
                fold_outputs.append(fold)
            attacks = ATTACKS[:1] if args.smoke else ATTACKS
            for attack_index, (attack_name, snr_db, targeted) in enumerate(attacks):
                if attack_name in fold["attacks"]:
                    print(f"outer={outer_index + 1} {attack_name}: reuse checkpoint", flush=True)
                    continue
                adv_scores, attack_diagnostics, adv_fingerprints = cached_attack(
                    cache_dir, f"outer{outer_index + 1}_eval", model, extractor,
                    clean_raw, tasks[outer_eval_known], subjects[outer_eval_known],
                    attack_name, snr_db, targeted, steps,
                    20261107 + outer_index * 1000000 + attack_index * 200000,
                    pool, args.batch_size,
                )
                fingerprints = np.concatenate([clean_fingerprints, adv_fingerprints, unknown_fingerprints])
                target_tasks = np.concatenate([
                    tasks[outer_eval_known], tasks[outer_eval_known], tasks[outer_eval_unknown]
                ])
                neighbors = waveform_neighbors(fingerprints, target_tasks, device)
                reports, pair_diagnostics = evaluate_attack_methods(
                    subjects[outer_eval_known], clean_scores, adv_scores, unknown_scores,
                    model.classes, operating_points, anchor_minimum, neighbors,
                )
                learned, learned_diagnostics, direct, learned_pred = learned_report(
                    anchor, subjects[outer_eval_known], clean_scores, adv_scores, unknown_scores,
                    model.classes, operating_points, neighbors,
                )
                direct_confidence = confidence_values(np.concatenate([
                    clean_scores[0], adv_scores[0], unknown_scores[0]
                ]))[operating_points[0]["score_name"]]
                asymmetric = asymmetric_propagation(
                    direct, direct_confidence, neighbors, anchor_minimum
                )
                reports["learned_anchor"] = learned
                pair_diagnostics["learned_anchor"] = learned_diagnostics
                pair_diagnostics["original_asymmetric_audit"] = propagation_audit(
                    direct, asymmetric, subjects[outer_eval_known], len(outer_eval_known)
                )
                fold["attacks"][attack_name] = {
                    "methods": {
                        name: reports[name]
                        for name in ("direct", "asymmetric_pair", "combined", "learned_anchor")
                    },
                    "attack": attack_diagnostics,
                    "pairs": pair_diagnostics,
                }
                save_checkpoint()
                print(f"outer={outer_index + 1} {attack_name} complete", flush=True)
            del model, anchor
            torch.cuda.empty_cache()

    result = {
        "protocol": {
            "data": "Train Session1 only; 5-fold subject holdout; nested threshold selection",
            "candidate": "one frozen candidate: learned clean-anchor propagation",
            "anchor_training": "inner held-out known trials; clean plus all three generated attacks; no outer data",
            "propagation": "higher clean probability anchor >=0.5; receiver <0.5; existing reciprocal pairs",
            "attacks": [
                {"name": name, "snr_db": snr, "targeted": targeted, "steps": 2 if args.smoke else 10}
                for name, snr, targeted in (ATTACKS[:1] if args.smoke else ATTACKS)
            ],
            "gpu": torch.cuda.get_device_name(0),
            "smoke": args.smoke,
        },
        "folds": fold_outputs,
    }
    result["attacks"] = {
        attack_name: {
            method: aggregate([
                fold["attacks"][attack_name]["methods"][method] for fold in fold_outputs
            ])
            for method in ("direct", "asymmetric_pair", "combined", "learned_anchor")
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
