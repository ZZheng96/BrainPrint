"""Generate five session-blind candidates with method-matched blocked OOF thresholds."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ea_riemannian_openset.core import (
    fit_recentered_model,
    predict_recentered_full_scores,
)
from src.open_set_calibration.core import build_submission, predict_calibration_scores
from src.open_set_improvement.core import TaskSpecificModel, confidence_values, resolve_cache
from src.session_blind_candidate.run import load_inputs

OUTPUT = Path(__file__).resolve().parent / "five_candidates"
METHODS = (
    "full_none", "base_none", "full_target_zscore", "base_target_zscore",
    "full_riemannian_recenter",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_batch(x, window, subjects, tasks, fit, target_x, target_window, target_tasks, device):
    model = TaskSpecificModel(device, alpha=100.0).fit(x, window, subjects, tasks, fit)
    classes = model.classes.copy()
    scores = {}
    diagnostics = {}
    for alignment in ("none", "target_zscore"):
        current, diagnostics[alignment] = predict_calibration_scores(
            model, target_x, target_window, target_tasks, alignment,
        )
        scores.update(current)
    del model
    model, source = fit_recentered_model(
        x, window, subjects, tasks, fit, "riemannian_recenter", device,
    )
    if not np.array_equal(classes, model.classes):
        raise ValueError("Inconsistent class order")
    scores[METHODS[-1]], diagnostics["riemannian_recenter"] = predict_recentered_full_scores(
        model, source, target_x, target_window, target_tasks,
    )
    if any(not np.isfinite(value).all() for value in scores.values()):
        raise ValueError("Nonfinite scores")
    return scores, classes, diagnostics


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no silent CPU fallback")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    cache = resolve_cache()
    protected = sorted((ROOT / "commit/answer").glob("赞美诗_第*次.csv"))
    if len(protected) != 2:
        raise ValueError("Expected two formal submissions")
    before = {str(path.relative_to(ROOT)): sha256(path) for path in protected}
    x, window, metadata, test_x, test_window, test_metadata = load_inputs(cache)
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    classes = np.unique(subjects)
    fold_path = cache / "train_blocked_oof_predictions_cov_alpha100.csv"
    # Reuse only the established split assignments, never the old model's scores.
    split_table = pd.read_csv(fold_path, usecols=["subject", "task", "fold"])
    if not split_table[["subject", "task"]].equals(metadata[["subject", "task"]]):
        raise ValueError("Cached fold table is not aligned with training metadata")
    folds = split_table["fold"].to_numpy()
    if set(np.unique(folds)) != set(range(5)):
        raise ValueError("Expected five blocked folds")
    oof = {name: np.full((len(x), len(classes)), np.nan, np.float32) for name in METHODS}
    fold_diagnostics = {}
    for fold in range(5):
        print(f"Blocked OOF fold {fold + 1}/5", flush=True)
        validation = folds == fold
        scores, current_classes, diagnostics = score_batch(
            x, window, subjects, tasks, ~validation,
            x[validation], window[validation], tasks[validation], device,
        )
        if not np.array_equal(classes, current_classes):
            raise ValueError("Fold lacks training identities")
        for name in METHODS:
            oof[name][validation] = scores[name]
        fold_diagnostics[str(fold)] = diagnostics
    thresholds = {}
    for name, values in oof.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Incomplete OOF: {name}")
        thresholds[name] = float(np.quantile(confidence_values(values)["max_logit"], 0.05))
    print("Full training and unlabeled Test scoring", flush=True)
    test_scores, current_classes, diagnostics = score_batch(
        x, window, subjects, tasks, np.ones(len(x), dtype=bool),
        test_x, test_window, test_metadata["task"].to_numpy(), device,
    )
    template = pd.read_csv(ROOT / "Brainprint_Competition/test.csv", usecols=["Sample"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / "blocked_oof_scores.npz", fold=folds, classes=classes, **oof)
    manifest = {
        "methods": METHODS, "seed": 42, "device": torch.cuda.get_device_name(0),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "cache": str(cache), "n_train": len(x),
        "metadata_columns_read": {"train": ["subject", "task"], "test": ["sample", "task"],
                                  "split_table": ["subject", "task", "fold"]},
        "forbidden_columns_read": [], "session_used": False,
        "fold_source": str(fold_path), "fold_source_sha256": sha256(fold_path),
        "fold_definition": "Existing contiguous blocks of 10 trials modulo 5; read precomputed fold only",
        "threshold_source": "Each method's freshly recomputed all-subject blocked-OOF max_logit, quantile=0.05 (linear)",
        "target_reference": "Unlabeled held-out fold or full Test batch, grouped only by task",
        "oof_diagnostics": fold_diagnostics, "test_diagnostics": diagnostics,
        "candidates": {},
    }
    for name in METHODS:
        values = test_scores[name]
        confidence = confidence_values(values)["max_logit"]
        predicted = current_classes[values.argmax(axis=1)].astype(int)
        predicted[confidence < thresholds[name]] = 0
        submission = build_submission(template, test_metadata, predicted)
        path = OUTPUT / f"candidate_{name}.csv"
        submission.to_csv(path, index=False)
        saved = pd.read_csv(path)
        if (len(saved) != 22251 or not saved["Sample"].is_unique
                or not saved["Sample"].equals(template["Sample"])
                or not saved["Label"].isin(range(21)).all()
                or not pd.api.types.is_integer_dtype(saved["Label"])):
            raise ValueError(f"Invalid candidate: {name}")
        manifest["candidates"][name] = {
            "file": path.name, "sha256": sha256(path), "threshold": thresholds[name],
            "n_rows": len(saved), "sample_unique": True, "official_sample_order_matches": True,
            "labels_valid": True, "n_reject": int((predicted == 0).sum()),
            "reject_rate": float((predicted == 0).mean()),
            "label_counts": {str(k): int(v) for k, v in saved["Label"].value_counts().sort_index().items()},
            "test_auc": None, "test_f1": None,
        }
    after = {str(path.relative_to(ROOT)): sha256(path) for path in protected}
    if before != after:
        raise RuntimeError("Formal submission changed during generation")
    manifest["formal_submissions_sha256"] = after
    manifest["formal_submissions_unchanged"] = True
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["candidates"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
