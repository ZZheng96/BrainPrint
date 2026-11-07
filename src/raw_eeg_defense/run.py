"""原始 EEG PGD 下的非对称配对与多视图防御评测。"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

from src.eeg_features import extract_features, load_eeg
from src.metrics import score_predictions
from src.open_set_improvement.core import (
    COV_COLUMNS,
    ROOT,
    TaskSpecificModel,
    confidence_values,
    resolve_cache,
    select_operating_point,
)
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout
from src.raw_eeg_defense.gpu_attack import GPUCovExtractor, raw_pgd

DATA = ROOT / "Brainprint_Competition"
OUTPUT_DIR = ROOT / "research" / "raw_eeg_defense"
VIEW_NAMES = ("original", "lowpass40", "resample150")
ATTACKS = (
    ("untargeted_snr20", 20.0, False),
    ("untargeted_snr30", 30.0, False),
    ("targeted_snr20", 20.0, True),
)
METHODS = ("direct", "graph_mean", "asymmetric_pair", "multiview", "combined")
METRIC_NAMES = (
    "C", "RA", "ASafeR", "A", "U", "official_score", "unknown_auroc",
    "known_aupr", "unknown_aupr", "eer", "clean_macro_f1_with_reject",
    "adversarial_macro_f1_with_reject", "unknown_f1",
)
_LOWPASS_SOS = butter(4, 40, btype="lowpass", fs=300, output="sos")


def transform_view(data: np.ndarray, view: str) -> np.ndarray:
    if view == "original":
        return data
    if view == "lowpass40":
        return sosfiltfilt(_LOWPASS_SOS, data, axis=0)
    if view == "resample150":
        downsampled = resample_poly(data, up=1, down=2, axis=0)
        restored = resample_poly(downsampled, up=2, down=1, axis=0)
        if restored.shape[0] != data.shape[0]:
            raise ValueError(f"重采样长度错误: {restored.shape}")
        return restored
    raise ValueError(view)


def extract_view(data: np.ndarray, view: str) -> tuple[np.ndarray, np.ndarray]:
    transformed = transform_view(data, view)
    full = extract_features(transformed)[0][COV_COLUMNS]
    windows = np.stack([
        extract_features(transformed[:600])[0][COV_COLUMNS],
        extract_features(transformed[600:])[0][COV_COLUMNS],
    ])
    return full, windows


def extract_all_views(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    extracted = [extract_view(data, view) for view in VIEW_NAMES]
    return (
        np.stack([row[0] for row in extracted]).astype(np.float32),
        np.stack([row[1] for row in extracted]).astype(np.float32),
    )


def extract_additional_path(path_string: str) -> tuple[np.ndarray, np.ndarray]:
    data, _ = load_eeg(path_string)
    extracted = [extract_view(data, view) for view in VIEW_NAMES[1:]]
    return (
        np.stack([row[0] for row in extracted]).astype(np.float32),
        np.stack([row[1] for row in extracted]).astype(np.float32),
    )


def load_raw_path(path_string: str) -> np.ndarray:
    return load_eeg(path_string)[0].astype(np.float32)


def cache_valid(path: Path, shape: tuple[int, ...]) -> bool:
    if not path.exists():
        return False
    try:
        return np.load(path, mmap_mode="r").shape == shape
    except (OSError, ValueError):
        return False


def prepare_clean_view_cache(
    metadata: pd.DataFrame, cache: Path, pool: ProcessPoolExecutor
) -> tuple[np.ndarray, np.ndarray]:
    """缓存两个新增视图的干净特征；原始视图沿用既有缓存。"""
    directory = cache / "raw_eeg_defense"
    directory.mkdir(parents=True, exist_ok=True)
    full_path = directory / "train_additional_full.npy"
    window_path = directory / "train_additional_window.npy"
    full_shape = (len(metadata), 2, len(COV_COLUMNS))
    window_shape = (len(metadata), 2, 2, len(COV_COLUMNS))
    if cache_valid(full_path, full_shape) and cache_valid(window_path, window_shape):
        print("clean view cache: reuse", flush=True)
        return np.load(full_path, mmap_mode="r"), np.load(window_path, mmap_mode="r")

    full_tmp = directory / "train_additional_full.tmp.npy"
    window_tmp = directory / "train_additional_window.tmp.npy"
    full = np.lib.format.open_memmap(full_tmp, mode="w+", dtype=np.float32, shape=full_shape)
    window = np.lib.format.open_memmap(window_tmp, mode="w+", dtype=np.float32, shape=window_shape)
    paths = metadata["path"].map(lambda value: str(DATA / value)).tolist()
    for index, (full_row, window_row) in enumerate(
        pool.map(extract_additional_path, paths, chunksize=8), 1
    ):
        full[index - 1] = full_row
        window[index - 1] = window_row
        if index % 250 == 0 or index == len(paths):
            print(f"clean view cache: {index}/{len(paths)}", flush=True)
    full.flush()
    window.flush()
    del full, window
    full_tmp.replace(full_path)
    window_tmp.replace(window_path)
    return np.load(full_path, mmap_mode="r"), np.load(window_path, mmap_mode="r")


def view_arrays(
    x: np.ndarray, window: np.ndarray, additional_full: np.ndarray, additional_window: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    full_views = [x, additional_full[:, 0], additional_full[:, 1]]
    window_views = [window, additional_window[:, 0], additional_window[:, 1]]
    return full_views, window_views


def predict_subset(
    model: TaskSpecificModel,
    full: np.ndarray,
    window: np.ndarray,
    tasks: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    return model.predict_scores(
        np.asarray(full[indices]), np.asarray(window[indices]), tasks[indices]
    )["fused"]


def waveform_fingerprint(data: np.ndarray, stride: int = 20) -> np.ndarray:
    centered = data - data.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True).clip(min=1e-12)
    normalized = centered / scale
    vector = normalized[::stride].reshape(-1)
    return (vector / max(float(np.linalg.norm(vector)), 1e-12)).astype(np.float32)


def load_raw_and_fingerprints(
    metadata: pd.DataFrame, indices: np.ndarray, keep_raw: bool
) -> tuple[np.ndarray | None, np.ndarray]:
    raw = np.empty((len(indices), 1200, 18), dtype=np.float32) if keep_raw else None
    fingerprints = np.empty((len(indices), 1080), dtype=np.float32)
    for position, index in enumerate(indices):
        data = load_raw_path(str(DATA / metadata.iloc[index]["path"]))
        if keep_raw:
            raw[position] = data
        fingerprints[position] = waveform_fingerprint(data)
        if (position + 1) % 500 == 0 or position + 1 == len(indices):
            print(f"raw load: {position + 1}/{len(indices)}", flush=True)
    return raw, fingerprints


def waveform_neighbors(
    fingerprints: np.ndarray, tasks: np.ndarray, device: torch.device
) -> dict[str, np.ndarray]:
    """同任务内互为近邻；不读取 Session 或真实 clean/adv 对应关系。"""
    nearest = np.full(len(fingerprints), -1, dtype=np.int32)
    similarity = np.zeros(len(fingerprints), dtype=np.float32)
    second_similarity = np.zeros(len(fingerprints), dtype=np.float32)
    for task in sorted(np.unique(tasks)):
        indices = np.flatnonzero(tasks == task)
        values = torch.as_tensor(fingerprints[indices], device=device)
        similarities = values @ values.T
        similarities.fill_diagonal_(-float("inf"))
        top = similarities.topk(k=2, dim=1)
        nearest[indices] = indices[top.indices[:, 0].cpu().numpy()]
        similarity[indices] = top.values[:, 0].cpu().numpy()
        second_similarity[indices] = top.values[:, 1].cpu().numpy()
        del values, similarities, top
    reciprocal = nearest[nearest] == np.arange(len(fingerprints))
    return {
        "nearest": nearest,
        "similarity": similarity,
        "second_similarity": second_similarity,
        "reciprocal": reciprocal,
    }


def pair_positions(neighbors: dict[str, np.ndarray], gap: float) -> np.ndarray:
    valid = (
        neighbors["reciprocal"]
        & (neighbors["similarity"] > 0.99)
        & ((neighbors["similarity"] - neighbors["second_similarity"]) > gap)
    )
    return np.flatnonzero(valid & (np.arange(len(valid)) < neighbors["nearest"]))


def graph_average_scores(scores: np.ndarray, neighbors: dict[str, np.ndarray]) -> np.ndarray:
    """复现现有图组件均值，仅使用相似度阈值，不增加间隔约束。"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    left = np.flatnonzero(neighbors["similarity"] > 0.99)
    if not len(left):
        return scores.copy()
    right = neighbors["nearest"][left]
    graph = coo_matrix(
        (np.ones(len(left)), (left, right)), shape=(len(scores), len(scores))
    )
    _, components = connected_components(graph, directed=False)
    output = scores.copy()
    for component in np.unique(components):
        indices = np.flatnonzero(components == component)
        if len(indices) > 1:
            output[indices] = scores[indices].mean(axis=0)
    return output


def open_set_predictions(
    scores: np.ndarray, classes: np.ndarray, operating_point: dict
) -> tuple[np.ndarray, np.ndarray]:
    confidence = confidence_values(scores)[operating_point["score_name"]]
    predicted = classes[scores.argmax(axis=1)].copy()
    predicted[confidence < operating_point["threshold"]] = 0
    return predicted, confidence


def multiview_predictions(predictions: list[np.ndarray]) -> np.ndarray:
    output = np.zeros(len(predictions[0]), dtype=int)
    for left, right in ((0, 1), (0, 2), (1, 2)):
        agreement = (
            (predictions[left] == predictions[right])
            & (predictions[left] > 0)
            & (output == 0)
        )
        output[agreement] = predictions[left][agreement]
    return output


def asymmetric_propagation(
    predictions: np.ndarray,
    confidence: np.ndarray,
    neighbors: dict[str, np.ndarray],
    anchor_minimum: float,
    anchor_eligible: np.ndarray | None = None,
) -> np.ndarray:
    output = predictions.copy()
    if anchor_eligible is None:
        anchor_eligible = predictions > 0
    for left in pair_positions(neighbors, gap=0.005):
        right = neighbors["nearest"][left]
        anchor, receiver = (left, right) if confidence[left] >= confidence[right] else (right, left)
        if (
            anchor_eligible[anchor]
            and predictions[anchor] > 0
            and confidence[anchor] >= anchor_minimum
            and confidence[receiver] < confidence[anchor]
        ):
            output[receiver] = predictions[anchor]
    return output


def competition_metrics(
    clean_true: np.ndarray,
    clean_pred: np.ndarray,
    adversarial_pred: np.ndarray,
    unknown_pred: np.ndarray,
    clean_confidence: np.ndarray,
    unknown_confidence: np.ndarray,
) -> dict:
    report = score_predictions(
        clean_true, clean_pred, clean_true, adversarial_pred, unknown_pred
    )
    detection_true = np.concatenate([
        np.ones(len(clean_pred), dtype=int), np.zeros(len(unknown_pred), dtype=int)
    ])
    detection_scores = np.concatenate([clean_confidence, unknown_confidence])
    detection_pred = np.concatenate([clean_pred != 0, unknown_pred != 0])
    fpr, tpr, _ = roc_curve(detection_true, detection_scores)
    eer_index = int(np.argmin(np.abs(fpr - (1 - tpr))))
    report.update({
        "n_known": int(len(clean_true)),
        "n_unknown": int(len(unknown_pred)),
        "unknown_auroc": float(roc_auc_score(detection_true, detection_scores)),
        "known_aupr": float(average_precision_score(detection_true, detection_scores)),
        "unknown_aupr": float(average_precision_score(1 - detection_true, -detection_scores)),
        "eer": float((fpr[eer_index] + 1 - tpr[eer_index]) / 2),
        "clean_macro_f1_with_reject": float(f1_score(
            clean_true, clean_pred, average="macro", zero_division=0
        )),
        "adversarial_macro_f1_with_reject": float(f1_score(
            clean_true, adversarial_pred, average="macro", zero_division=0
        )),
        "unknown_f1": float(f1_score(
            detection_true, detection_pred, pos_label=0, zero_division=0
        )),
    })
    return report


def evaluate_attack_methods(
    clean_true: np.ndarray,
    clean_view_scores: list[np.ndarray],
    adversarial_view_scores: list[np.ndarray],
    unknown_view_scores: list[np.ndarray],
    classes: np.ndarray,
    operating_points: list[dict],
    anchor_minimum: float,
    neighbors: dict[str, np.ndarray],
) -> tuple[dict, dict]:
    n_known = len(clean_true)
    n_unknown = len(unknown_view_scores[0])
    view_predictions = []
    for view in range(3):
        clean_pred, _ = open_set_predictions(clean_view_scores[view], classes, operating_points[view])
        adv_pred, _ = open_set_predictions(adversarial_view_scores[view], classes, operating_points[view])
        unknown_pred, _ = open_set_predictions(unknown_view_scores[view], classes, operating_points[view])
        view_predictions.append(np.concatenate([clean_pred, adv_pred, unknown_pred]))

    original_scores = np.concatenate([
        clean_view_scores[0], adversarial_view_scores[0], unknown_view_scores[0]
    ])
    direct_pred = view_predictions[0]
    direct_confidence = np.concatenate([
        confidence_values(clean_view_scores[0])[operating_points[0]["score_name"]],
        confidence_values(adversarial_view_scores[0])[operating_points[0]["score_name"]],
        confidence_values(unknown_view_scores[0])[operating_points[0]["score_name"]],
    ])
    averaged = graph_average_scores(original_scores, neighbors)
    graph_pred, graph_confidence = open_set_predictions(averaged, classes, operating_points[0])
    asymmetric = asymmetric_propagation(
        direct_pred, direct_confidence, neighbors, anchor_minimum
    )
    multiview = multiview_predictions(view_predictions)
    combined = asymmetric_propagation(
        multiview,
        direct_confidence,
        neighbors,
        anchor_minimum,
        anchor_eligible=multiview > 0,
    )
    multiview_confidence = np.mean(np.stack([
        np.concatenate([
            confidence_values(clean_view_scores[view])[operating_points[view]["score_name"]]
            - operating_points[view]["threshold"],
            confidence_values(adversarial_view_scores[view])[operating_points[view]["score_name"]]
            - operating_points[view]["threshold"],
            confidence_values(unknown_view_scores[view])[operating_points[view]["score_name"]]
            - operating_points[view]["threshold"],
        ])
        for view in range(3)
    ]), axis=0)
    predictions = {
        "direct": direct_pred,
        "graph_mean": graph_pred,
        "asymmetric_pair": asymmetric,
        "multiview": multiview,
        "combined": combined,
    }
    confidences = {
        "direct": direct_confidence,
        "graph_mean": graph_confidence,
        "asymmetric_pair": direct_confidence,
        "multiview": multiview_confidence,
        "combined": multiview_confidence,
    }
    reports = {}
    for name, predicted in predictions.items():
        confidence = confidences[name]
        reports[name] = competition_metrics(
            clean_true,
            predicted[:n_known],
            predicted[n_known : 2 * n_known],
            predicted[2 * n_known :],
            confidence[:n_known],
            confidence[2 * n_known :],
        )
    diagnostics = {
        "reciprocal_pairs": int(np.sum(neighbors["reciprocal"] & (
            np.arange(len(neighbors["reciprocal"])) < neighbors["nearest"]
        ))),
        "pairs_similarity_0.99": int(len(pair_positions(neighbors, gap=-1.0))),
        "pairs_with_gap_0.005": int(len(pair_positions(neighbors, gap=0.005))),
        "direct_pair_agreement": float(np.mean(
            direct_pred[pair_positions(neighbors, gap=-1.0)]
            == direct_pred[neighbors["nearest"][pair_positions(neighbors, gap=-1.0)]]
        )) if len(pair_positions(neighbors, gap=-1.0)) else float("nan"),
        "asymmetric_changed": int(np.sum(asymmetric != direct_pred)),
        "multiview_changed": int(np.sum(multiview != direct_pred)),
        "combined_changed": int(np.sum(combined != direct_pred)),
    }
    return reports, diagnostics


def attack_known(
    model: TaskSpecificModel,
    extractor: GPUCovExtractor,
    raw: np.ndarray,
    tasks: np.ndarray,
    targets: np.ndarray,
    snr_db: float,
    targeted: bool,
    steps: int,
    seed: int,
    pool: ProcessPoolExecutor,
    batch_size: int,
) -> tuple[list[np.ndarray], dict, np.ndarray]:
    n = len(raw)
    adversarial = np.empty_like(raw)
    surrogate_success = []
    for task in sorted(np.unique(tasks)):
        positions = np.flatnonzero(tasks == task)
        for start in range(0, len(positions), batch_size):
            selected = positions[start : start + batch_size]
            data = torch.as_tensor(raw[selected], device=model.device)
            attack_targets = targets[selected]
            if targeted:
                target_indices = (np.searchsorted(model.classes, attack_targets) + 1) % len(model.classes)
                attack_targets = model.classes[target_indices]
            adv, diagnostics = raw_pgd(
                model, extractor, data, task, attack_targets, snr_db,
                targeted=targeted, steps=steps,
                seed=seed + start + int(snr_db) * 1000 + int(targeted) * 100000,
            )
            adversarial[selected] = adv.cpu().numpy()
            surrogate_scores = diagnostics["final_scores"]
            if isinstance(surrogate_scores, torch.Tensor):
                surrogate_scores = surrogate_scores.cpu().numpy()
            predicted = model.classes[np.asarray(surrogate_scores).argmax(axis=1)]
            surrogate_success.append(predicted == targets[selected])
            print(
                f"attack snr={snr_db:g} targeted={targeted} task={task}: "
                f"{min(start + batch_size, len(positions))}/{len(positions)}",
                flush=True,
            )
    centered = raw - raw.mean(axis=1, keepdims=True)
    delta = adversarial - raw
    signal_norm = np.linalg.norm(centered.reshape(n, -1), axis=1)
    delta_norm = np.linalg.norm(delta.reshape(n, -1), axis=1)
    actual_snr = 20 * np.log10(signal_norm / np.maximum(delta_norm, 1e-12))

    adv_full = np.empty((n, 3, len(COV_COLUMNS)), dtype=np.float32)
    adv_window = np.empty((n, 3, 2, len(COV_COLUMNS)), dtype=np.float32)
    for index, (full, window) in enumerate(
        pool.map(extract_all_views, adversarial, chunksize=4), 1
    ):
        adv_full[index - 1] = full
        adv_window[index - 1] = window
        if index % 250 == 0 or index == n:
            print(f"exact adversarial features: {index}/{n}", flush=True)
    scores = []
    for view in range(3):
        scores.append(model.predict_scores(
            adv_full[:, view], adv_window[:, view], tasks
        )["fused"])
    diagnostics = {
        "actual_snr_min": float(actual_snr.min()),
        "actual_snr_median": float(np.median(actual_snr)),
        "actual_snr_max": float(actual_snr.max()),
        "surrogate_identity_retention": float(np.mean(np.concatenate(surrogate_success))),
        "evaluation_features": "existing SciPy extractor re-applied to attacked raw EEG",
    }
    adversarial_fingerprints = np.stack([waveform_fingerprint(row) for row in adversarial])
    return scores, diagnostics, adversarial_fingerprints


def aggregate(reports: list[dict]) -> dict:
    output = {name: float(np.mean([row[name] for row in reports])) for name in METRIC_NAMES}
    output["official_score_rounded"] = round(output["official_score"], 2)
    output["worst_fold_official_score"] = float(min(row["official_score"] for row in reports))
    output["n_folds"] = len(reports)
    output["n_known"] = int(sum(row["n_known"] for row in reports))
    output["n_unknown"] = int(sum(row["n_unknown"] for row in reports))
    return output


def acceptance(results: dict) -> dict:
    primary = results["attacks"]["untargeted_snr20"]
    baseline_name = max(("direct", "graph_mean"), key=lambda name: primary[name]["official_score"])
    baseline = primary[baseline_name]
    output = {"primary_baseline": baseline_name, "candidates": {}}
    for candidate in ("asymmetric_pair", "multiview", "combined"):
        if candidate not in primary:
            output["candidates"][candidate] = {"considered": False, "passed": False}
            continue
        row = primary[candidate]
        fold_scores = np.asarray([
            fold["attacks"]["untargeted_snr20"]["methods"][candidate]["official_score"]
            - fold["attacks"]["untargeted_snr20"]["methods"][baseline_name]["official_score"]
            for fold in results["folds"]
        ])
        checks = {
            "score_gain_at_least_2": row["official_score"] - baseline["official_score"] >= 2.0,
            "A_gain_at_least_0.05": row["A"] - baseline["A"] >= 0.05,
            "RA_at_least_0.60_and_gain_0.03": (
                row["RA"] >= 0.60 and row["RA"] - baseline["RA"] >= 0.03
            ),
            "ASafeR_at_least_0.90": row["ASafeR"] >= 0.90,
            "C_at_least_0.90_and_drop_at_most_0.02": (
                row["C"] >= 0.90 and row["C"] - baseline["C"] >= -0.02
            ),
            "U_at_least_0.85_and_drop_at_most_0.02": (
                row["U"] >= 0.85 and row["U"] - baseline["U"] >= -0.02
            ),
            "unknown_f1_drop_at_most_0.01": row["unknown_f1"] - baseline["unknown_f1"] >= -0.01,
            "AUROC_drop_at_most_0.005": row["unknown_auroc"] - baseline["unknown_auroc"] >= -0.005,
            "four_of_five_folds_not_worse": int(np.sum(fold_scores >= 0)) >= 4,
            "worst_fold_score_drop_at_most_1": float(fold_scores.min()) >= -1.0,
        }
        auxiliary = {}
        for attack_name in ("untargeted_snr30", "targeted_snr20"):
            attack = results["attacks"][attack_name]
            aux_baseline = attack[baseline_name]
            aux_row = attack[candidate]
            auxiliary[attack_name] = {
                "ASafeR_at_least_0.90": aux_row["ASafeR"] >= 0.90,
                "A_drop_at_most_0.01": aux_row["A"] - aux_baseline["A"] >= -0.01,
            }
            checks.update({f"{attack_name}_{key}": value for key, value in auxiliary[attack_name].items()})
        output["candidates"][candidate] = {
            "considered": True,
            "passed": bool(all(checks.values())),
            "score_delta": row["official_score"] - baseline["official_score"],
            "A_delta": row["A"] - baseline["A"],
            "RA_delta": row["RA"] - baseline["RA"],
            "fold_score_deltas": fold_scores.tolist(),
            "checks": checks,
        }
    early_stop = all(
        output["candidates"][name]["considered"]
        and output["candidates"][name]["score_delta"] < 1.0
        for name in ("asymmetric_pair", "multiview")
    )
    if early_stop and output["candidates"]["combined"]["considered"]:
        output["candidates"]["combined"]["passed"] = False
        output["candidates"]["combined"]["cancelled_by_frozen_early_stop"] = True
    output["any_passed"] = any(
        row["passed"] for row in output["candidates"].values()
    )
    return output


def smoke_indices(metadata: pd.DataFrame, indices: np.ndarray, per_group: int = 2) -> np.ndarray:
    selected = []
    for _, group in metadata.iloc[indices].groupby(["subject", "task"], sort=False):
        selected.extend(group.index.to_numpy()[:per_group])
    return np.asarray(selected, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
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
    output = args.output
    if output is None:
        output = cache / "raw_eeg_defense" / "smoke_results.json" if args.smoke else OUTPUT_DIR / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        additional_full, additional_window = prepare_clean_view_cache(metadata, cache, pool)
        if args.prepare_only:
            print("clean view cache ready", flush=True)
            return
        full_views, window_views = view_arrays(x, window, additional_full, additional_window)
        extractor = GPUCovExtractor(device)
        forward = forward_holdout(metadata)
        inner = inner_holdout(metadata)
        all_subjects = np.unique(subjects)
        shuffled = np.random.default_rng(42).permutation(all_subjects)
        outer_groups = np.array_split(shuffled, 5)
        if args.smoke:
            outer_groups = outer_groups[:1]
        fold_outputs = []
        partial_output = output.with_name(f"{output.stem}.partial.json")
        if not args.smoke and partial_output.exists():
            saved = json.loads(partial_output.read_text(encoding="utf-8"))
            fold_outputs = saved["folds"]
            for index, saved_fold in enumerate(fold_outputs):
                if saved_fold["outer_fold"] != index + 1 or saved_fold["unknown_subjects"] != outer_groups[index].tolist():
                    raise RuntimeError("断点文件与当前固定折分不一致")
                if not set(saved_fold["attacks"]).issubset({row[0] for row in ATTACKS}):
                    raise RuntimeError("断点文件包含未知攻击")
            print(f"resume: reuse {len(fold_outputs)} fold(s)", flush=True)

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
            inner_order = np.random.default_rng(100 + outer_index).permutation(known_subjects)
            for inner_index, inner_unknown_subjects in enumerate(np.array_split(inner_order, 2)):
                inner_enrolled = np.setdiff1d(known_subjects, inner_unknown_subjects)
                inner_known_mask = np.isin(subjects, inner_enrolled)
                inner_fit = inner_known_mask & ~forward & ~inner
                inner_eval_known = np.flatnonzero(inner_known_mask & inner)
                inner_eval_unknown = np.flatnonzero(np.isin(subjects, inner_unknown_subjects) & ~forward)
                model = TaskSpecificModel(device).fit(x, window, subjects, tasks, inner_fit)
                for view in range(3):
                    known_scores = predict_subset(
                        model, full_views[view], window_views[view], tasks, inner_eval_known
                    )
                    unknown_scores = predict_subset(
                        model, full_views[view], window_views[view], tasks, inner_eval_unknown
                    )
                    inner_known[view].append(known_scores)
                    inner_unknown[view].append(unknown_scores)
                print(f"outer={outer_index + 1} inner={inner_index + 1} complete", flush=True)
                del model
            operating_points = [
                select_operating_point(inner_known[view], inner_unknown[view])
                for view in range(3)
            ]
            anchor_score_name = operating_points[0]["score_name"]
            anchor_values = np.concatenate([
                confidence_values(scores)[anchor_score_name] for scores in inner_known[0]
            ])
            anchor_minimum = float(np.median(anchor_values))

            outer_eval_known = np.flatnonzero(outer_known & forward)
            outer_eval_unknown = np.flatnonzero(np.isin(subjects, unknown_subjects))
            if args.smoke:
                outer_eval_known = smoke_indices(metadata, outer_eval_known)
                outer_eval_unknown = smoke_indices(metadata, outer_eval_unknown)
            model = TaskSpecificModel(device).fit(x, window, subjects, tasks, outer_fit)
            clean_view_scores = [
                predict_subset(model, full_views[view], window_views[view], tasks, outer_eval_known)
                for view in range(3)
            ]
            unknown_view_scores = [
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
                    "registered_subjects": known_subjects.tolist(),
                    "operating_points": operating_points,
                    "anchor_minimum": anchor_minimum,
                    "attacks": {},
                }
                fold_outputs.append(fold)
            attacks = ATTACKS[:1] if args.smoke else ATTACKS
            for attack_index, (attack_name, snr_db, targeted) in enumerate(attacks):
                if attack_name in fold["attacks"]:
                    print(f"outer={outer_index + 1} {attack_name}: reuse checkpoint", flush=True)
                    continue
                adv_view_scores, attack_diagnostics, adv_fingerprints = attack_known(
                    model, extractor, clean_raw, tasks[outer_eval_known], subjects[outer_eval_known],
                    snr_db, targeted, 2 if args.smoke else 10,
                    20261107 + outer_index * 1000000 + attack_index * 200000,
                    pool, args.batch_size,
                )
                fingerprints = np.concatenate([
                    clean_fingerprints, adv_fingerprints, unknown_fingerprints
                ])
                target_tasks = np.concatenate([
                    tasks[outer_eval_known], tasks[outer_eval_known], tasks[outer_eval_unknown]
                ])
                neighbors = waveform_neighbors(fingerprints, target_tasks, device)
                reports, pair_diagnostics = evaluate_attack_methods(
                    subjects[outer_eval_known],
                    clean_view_scores,
                    adv_view_scores,
                    unknown_view_scores,
                    model.classes,
                    operating_points,
                    anchor_minimum,
                    neighbors,
                )
                fold["attacks"][attack_name] = {
                    "methods": reports,
                    "attack": attack_diagnostics,
                    "pairs": pair_diagnostics,
                }
                save_checkpoint()
                print(f"outer={outer_index + 1} {attack_name} complete", flush=True)
            del model
            torch.cuda.empty_cache()

    result = {
        "protocol": {
            "data": "Train Session1 only; no cross-session claim",
            "outer": "5-fold subject holdout; 4 unknown and 16 registered subjects per fold",
            "known_outer_test": "last 20% trials within each registered subject/task",
            "inner": "2-fold pseudo-unknown; thresholds selected once per view",
            "attacks": [
                {"name": name, "snr_db": snr, "targeted": targeted, "steps": 2 if args.smoke else 10}
                for name, snr, targeted in (ATTACKS[:1] if args.smoke else ATTACKS)
            ],
            "pair_rule": "same task, reciprocal nearest, similarity>0.99; asymmetric defense also requires gap>0.005",
            "targeted_attack": "deterministic next class in the enrolled class order",
            "views": list(VIEW_NAMES),
            "model_selection": "no outer-fold retuning; at most three candidates",
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "smoke": args.smoke,
        },
        "folds": fold_outputs,
    }
    attack_names = list(fold_outputs[0]["attacks"])
    result["attacks"] = {
        attack_name: {
            method: aggregate([
                fold["attacks"][attack_name]["methods"][method] for fold in fold_outputs
            ])
            for method in METHODS
        }
        for attack_name in attack_names
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
