"""Generate and validate a session-blind EEG-only submission candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.open_set_calibration.core import (
    build_submission,
    predict_calibration_scores,
)
from src.session_blind_candidate.shift_calibration import (
    apply_conditional_rejection,
    apply_robust_normalization,
    fit_conditional_thresholds,
    fit_training_statistics,
    score_diagnostics,
)
from src.open_set_improvement.core import (
    COV_COLUMNS,
    TaskSpecificModel,
    confidence_values,
    resolve_cache,
)

ROOT = Path(__file__).resolve().parents[2]
TRAIN_COLUMNS = ["subject", "task"]
TEST_COLUMNS = ["sample", "task"]
FORBIDDEN_COLUMNS = ["session", "trial", "path", "fs"]
VARIANTS = ("full", "base")
ALIGNMENTS = ("none", "target_zscore")


def read_metadata(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, usecols=columns)[columns]


def load_inputs(cache: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    train_meta = read_metadata(cache / "train_metadata.csv", TRAIN_COLUMNS)
    test_meta = read_metadata(cache / "test_metadata.csv", TEST_COLUMNS)
    train_features = np.load(cache / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS]
    test_features = np.load(cache / "test_features.npy", mmap_mode="r")[:, COV_COLUMNS]
    train_windows_raw = np.load(cache / "train_window2_features.npy", mmap_mode="r")[:, COV_COLUMNS]
    test_windows_raw = np.load(cache / "test_window2_features.npy", mmap_mode="r")[:, COV_COLUMNS]
    if train_windows_raw.shape[0] != 2 * len(train_meta) or test_windows_raw.shape[0] != 2 * len(test_meta):
        raise ValueError("窗口缓存应包含每条 Trial 的两个窗口")
    train_windows = np.asarray(train_windows_raw).reshape(len(train_meta), 2, -1)
    test_windows = np.asarray(test_windows_raw).reshape(len(test_meta), 2, -1)
    if len(train_meta) != len(train_features) or len(test_meta) != len(test_features):
        raise ValueError("元数据和特征数量不一致")
    if train_windows.shape[0] != len(train_meta) or test_windows.shape[0] != len(test_meta):
        raise ValueError("窗口特征和元数据数量不一致")
    if train_meta["task"].isna().any() or test_meta["task"].isna().any():
        raise ValueError("task 不能缺失")
    if test_meta["sample"].duplicated().any():
        raise ValueError("sample 必须唯一")
    return train_features, train_windows, train_meta, test_features, test_windows, test_meta


def fit_and_score(cache: Path, device: torch.device) -> tuple[dict[str, np.ndarray], dict]:
    train_x, train_window, train_meta, test_x, test_window, test_meta = load_inputs(cache)
    model = TaskSpecificModel(device=device, alpha=100.0).fit(
        train_x, train_window, train_meta["subject"].to_numpy(),
        train_meta["task"].to_numpy(), np.ones(len(train_meta), dtype=bool),
    )
    scores: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict] = {}
    for alignment in ALIGNMENTS:
        current, diag = predict_calibration_scores(
            model, test_x, test_window, test_meta["task"].to_numpy(), alignment,
        )
        scores.update(current)
        diagnostics[alignment] = diag
    return scores, {
        "test_meta": test_meta,
        "classes": model.classes,
        "diagnostics": diagnostics,
        "metadata_columns_read": {"train": TRAIN_COLUMNS, "test": TEST_COLUMNS},
        "forbidden_columns": FORBIDDEN_COLUMNS,
    }


def fit_internal_calibration(cache: Path, device: torch.device) -> dict:
    """Fit frozen conditional rejection thresholds from training predictions only."""
    train_meta = read_metadata(cache / "train_metadata.csv", TRAIN_COLUMNS)
    train_x = np.load(cache / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS]
    train_windows_raw = np.load(cache / "train_window2_features.npy", mmap_mode="r")[:, COV_COLUMNS]
    if train_windows_raw.shape[0] != 2 * len(train_meta):
        raise ValueError("窗口缓存应包含每条 Trial 的两个窗口")
    train_window = np.asarray(train_windows_raw).reshape(len(train_meta), 2, -1)
    if len(train_meta) != len(train_x):
        raise ValueError("元数据和特征数量不一致")
    model = TaskSpecificModel(device=device, alpha=100.0).fit(
        train_x, train_window, train_meta["subject"].to_numpy(),
        train_meta["task"].to_numpy(), np.ones(len(train_meta), dtype=bool),
    )
    scores = model.predict_scores(train_x, train_window, train_meta["task"].to_numpy())
    diagnostics = score_diagnostics(scores["full"], scores["base"])
    calibration = fit_conditional_thresholds(scores["base"], diagnostics, "max_logit")
    return {"calibration": calibration, "diagnostics": diagnostics, "classes": model.classes}


def threshold_from_oof(cache: Path, variant: str, score_name: str) -> float:
    path = cache / "train_blocked_oof_scores_cov_alpha100.npy"
    scores = np.load(path, mmap_mode="r")
    if scores.ndim != 2:
        raise ValueError("OOF score shape 不正确")
    return float(np.quantile(confidence_values(scores)[score_name], 0.05))


def make_candidates(output_dir: Path, cache: Path, device: torch.device, use_internal_calibration: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores, info = fit_and_score(cache, device)
    internal = fit_internal_calibration(cache, device) if use_internal_calibration else None
    template = pd.read_csv(ROOT / "Brainprint_Competition" / "test.csv", usecols=["Sample"])
    result = {"session_used": False, "session_columns_read": False, "candidates": {}}
    for alignment in ALIGNMENTS:
        for variant in VARIANTS:
            key = f"{variant}_{alignment}"
            score_key = key
            if score_key not in scores:
                raise KeyError(score_key)
            values = scores[score_key]
            if internal is not None and variant == "base" and alignment == "none":
                predicted = apply_conditional_rejection(
                    values, info["classes"],
                    score_diagnostics(scores["full_none"], scores["base_none"]),
                    internal["calibration"],
                )
                threshold = internal["calibration"].default
            else:
                threshold = threshold_from_oof(cache, variant, "max_logit")
                confidence = confidence_values(values)["max_logit"]
                predicted = info["classes"][values.argmax(axis=1)].astype(int)
                predicted[confidence < threshold] = 0
            metadata = info["test_meta"].rename(columns={"sample": "sample"})
            submission = build_submission(template, metadata, predicted)
            path = output_dir / f"candidate_{key}.csv"
            submission.to_csv(path, index=False)
            result["candidates"][key] = {
                "path": str(path), "threshold": threshold,
                "n_rows": int(len(submission)),
                "n_reject": int(np.sum(predicted == 0)),
                "reject_rate": float(np.mean(predicted == 0)),
                "test_auc": None, "test_f1": None,
            }
    (output_dir.parent / "candidate_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "research/E11_session_blind_candidate/candidates")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--internal-calibration", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；按实验要求不静默退回 CPU")
    cache = resolve_cache()
    result = make_candidates(
        args.output_dir, cache, torch.device(args.device), args.internal_calibration
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
