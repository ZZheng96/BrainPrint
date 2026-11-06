"""三注册身份、其余身份未知的逐 Trial 决赛场景模拟。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.open_set_improvement.core import (
    COV_COLUMNS,
    ROOT,
    TorchRidge,
    confidence_values,
    open_set_report,
    resolve_cache,
    select_operating_point,
    subject_bootstrap,
)
from src.open_set_improvement.run_nested import pooled_report

OUTPUT_DIR = ROOT / "research" / "open_set_improvement"


class TorchPrototype:
    """训练折标准化空间中的类均值欧氏原型。"""

    def __init__(self, device: torch.device):
        self.device = device

    def fit(self, x: np.ndarray, y: np.ndarray, classes: np.ndarray) -> "TorchPrototype":
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        self.mean = x_tensor.mean(dim=0)
        self.scale = x_tensor.std(dim=0, unbiased=False).clamp_min(1e-6)
        standardized = (x_tensor - self.mean) / self.scale
        self.classes = np.asarray(classes, dtype=int)
        targets = torch.as_tensor(y, dtype=torch.long, device=self.device)
        self.centroids = torch.stack([
            standardized[targets == label].mean(dim=0) for label in self.classes
        ])
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        standardized = (x - self.mean) / self.scale
        return -((standardized[:, None, :] - self.centroids[None, :, :]) ** 2).mean(dim=2)


def make_model(name: str, device: torch.device):
    return TorchRidge(device) if name == "ridge" else TorchPrototype(device)


def predict_batches(model, x: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    outputs = []
    for start in range(0, len(x), batch_size):
        batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
        with torch.no_grad():
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs)


def role_masks(metadata: pd.DataFrame, registration_trials: int) -> dict[str, np.ndarray]:
    calibration_trials = max(2, registration_trials // 2)
    registration = np.zeros(len(metadata), dtype=bool)
    calibration = np.zeros(len(metadata), dtype=bool)
    for _, indices in metadata.groupby(["subject", "task"]).groups.items():
        ordered = metadata.loc[indices].sort_values("trial").index.to_numpy()
        registration[ordered[:registration_trials]] = True
        calibration[ordered[registration_trials : registration_trials + calibration_trials]] = True
    test = ~registration & ~calibration
    return {"registration": registration, "calibration": calibration, "test": test}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--registration-trials", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "final_simulation.json")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20 or args.registration_trials < 4:
        parser.error("repeats 范围为 1–20，registration-trials>=4")
    if not torch.cuda.is_available():
        parser.error("本实验要求 CUDA GPU")
    device = torch.device("cuda")
    cache = resolve_cache()
    metadata = pd.read_csv(cache / "train_metadata.csv")
    x = np.asarray(np.load(cache / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS])
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    all_subjects = np.unique(subjects)
    masks = role_masks(metadata, args.registration_trials)
    fold_reports = []
    records = {name: [] for name in ("ridge", "prototype")}
    record_subjects = {name: [] for name in records}

    for repeat in range(args.repeats):
        registered = np.asarray([all_subjects[(repeat + offset) % 20] for offset in range(3)])
        unknown = np.setdiff1d(all_subjects, registered)
        for task in sorted(np.unique(tasks)):
            task_mask = tasks == task
            for model_name in records:
                inner_known_scores = []
                inner_unknown_scores = []
                for pseudo_unknown in registered:
                    enrolled = np.setdiff1d(registered, [pseudo_unknown])
                    fit_mask = task_mask & masks["registration"] & np.isin(subjects, enrolled)
                    classes = np.unique(subjects[fit_mask])
                    model = make_model(model_name, device).fit(x[fit_mask], subjects[fit_mask], classes)
                    known_indices = np.flatnonzero(
                        task_mask & masks["calibration"] & np.isin(subjects, enrolled)
                    )
                    unknown_indices = np.flatnonzero(
                        task_mask & (masks["registration"] | masks["calibration"])
                        & (subjects == pseudo_unknown)
                    )
                    inner_known_scores.append(predict_batches(model, x[known_indices], device))
                    inner_unknown_scores.append(predict_batches(model, x[unknown_indices], device))
                selected = select_operating_point(inner_known_scores, inner_unknown_scores)

                fit_mask = task_mask & masks["registration"] & np.isin(subjects, registered)
                classes = np.unique(subjects[fit_mask])
                model = make_model(model_name, device).fit(x[fit_mask], subjects[fit_mask], classes)
                calibration_indices = np.flatnonzero(
                    task_mask & masks["calibration"] & np.isin(subjects, registered)
                )
                calibration_scores = predict_batches(model, x[calibration_indices], device)
                threshold = float(np.quantile(
                    confidence_values(calibration_scores)[selected["score_name"]], 0.05
                ))
                known_indices = np.flatnonzero(
                    task_mask & masks["test"] & np.isin(subjects, registered)
                )
                unknown_indices = np.flatnonzero(task_mask & np.isin(subjects, unknown))
                known_scores = predict_batches(model, x[known_indices], device)
                unknown_scores = predict_batches(model, x[unknown_indices], device)
                report, record = open_set_report(
                    subjects[known_indices], known_scores, unknown_scores, classes,
                    selected["score_name"], threshold,
                )
                report.update({
                    "repeat": repeat + 1,
                    "registered_subjects": registered.tolist(),
                    "task": task,
                    "model": model_name,
                    "inner_score_name": selected["score_name"],
                    "inner_detection_macro_f1": selected["inner_detection_macro_f1"],
                })
                fold_reports.append(report)
                records[model_name].append(record)
                record_subjects[model_name].append(np.concatenate([
                    subjects[known_indices], subjects[unknown_indices]
                ]))
            print(f"repeat={repeat + 1} task={task} complete", flush=True)

    summary = {}
    for model_name in records:
        model_folds = [row for row in fold_reports if row["model"] == model_name]
        model_records = {
            key: np.concatenate([fold[key] for fold in records[model_name]])
            for key in records[model_name][0]
        }
        model_subjects = np.concatenate(record_subjects[model_name])
        per_task = {}
        for task in sorted(np.unique(tasks)):
            task_folds = [row for row in model_folds if row["task"] == task]
            per_task[task] = {
                name: float(np.mean([row[name] for row in task_folds]))
                for name in (
                    "C", "U", "unknown_auroc", "known_aupr", "unknown_aupr", "eer",
                    "macro_f1", "unknown_f1", "detection_macro_f1", "known_acceptance",
                )
            }
        summary[model_name] = {
            "pooled": pooled_report(model_records, model_folds),
            "per_task_macro_mean": per_task,
            "subject_bootstrap_95pct": subject_bootstrap(
                model_records, model_subjects, args.bootstrap
            ) if args.bootstrap else {},
        }
    result = {
        "protocol": {
            "data": "Train Session1 only; pseudo-time holdout, not cross-session evidence",
            "registered_subjects_per_repeat": 3,
            "unknown_subjects_per_repeat": 17,
            "repeats": args.repeats,
            "registration_trials_per_subject_task": args.registration_trials,
            "calibration_trials_per_subject_task": max(2, args.registration_trials // 2),
            "inner": "leave one of three registered subjects out as pseudo-unknown; score selection only",
            "threshold": "5th percentile of disjoint calibration trials from all three registered subjects",
            "outer": "all remaining registered trials + all trials of 17 unknown subjects",
            "inference": "per-trial, no metadata rejection, batch alignment, graph fusion or Sinkhorn",
            "scenario_note": "trial counts and task-specific evaluation are simulation choices, not claimed final rules",
            "gpu": torch.cuda.get_device_name(0),
        },
        "summary": summary,
        "folds": fold_reports,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
