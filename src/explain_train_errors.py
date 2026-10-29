"""对训练折外误判给出可复核的质量、近邻和模型贡献证据。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from run_baseline import FEATURE_SETS, make_model

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "BrainPrint_cache"
CHANNELS = ("P3", "C3", "F3", "Fz", "F4", "C4", "P4", "Cz", "Fp1", "Fp2", "T3", "T5", "O1", "O2", "F7", "F8", "T6", "T4")
BANDS = ((1, 8), (8, 13), (13, 30), (30, 45), (45, 70), (70, 90))
TRIU = np.triu_indices(18)
N_COV_PER_BAND = 171


def percentile(values: np.ndarray, value: float) -> float:
    return float(np.mean(values <= value))


def main() -> None:
    x = np.load(CACHE / "train_features.npy", mmap_mode="r")
    metadata = pd.read_csv(CACHE / "train_metadata.csv")
    columns = FEATURE_SETS["cov"]
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    trials = metadata["trial"].to_numpy()
    folds = ((trials - 1) // 10) % 5
    max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    path_lookup = dict(zip(metadata["path"], range(len(metadata))))
    explanations = []

    for protocol in ("blocked_oof", "forward"):
        predictions = pd.read_csv(CACHE / f"train_{protocol}_predictions_cov_alpha100.csv")
        errors = predictions[~predictions["correct"]]
        for error in errors.itertuples(index=False):
            index = path_lookup[error.path]
            task = error.task
            if protocol == "blocked_oof":
                fit = folds != error.fold
            else:
                fit = trials <= 0.8 * max_trial
            fit_indices = np.flatnonzero(fit & (tasks == task))
            x_train = np.asarray(x[fit_indices][:, columns])
            scaler = StandardScaler().fit(x_train)
            scaled_train = scaler.transform(x_train)
            scaled_error = scaler.transform(np.asarray(x[index, columns])[None, :])[0]

            normalized_train = scaled_train / np.maximum(np.linalg.norm(scaled_train, axis=1, keepdims=True), 1e-12)
            normalized_error = scaled_error / max(float(np.linalg.norm(scaled_error)), 1e-12)
            similarities = normalized_train @ normalized_error
            nearest_position = int(np.argmax(similarities))
            nearest_index = fit_indices[nearest_position]
            true_mask = subjects[fit_indices] == error.subject
            predicted_mask = subjects[fit_indices] == error.predicted_label
            true_best_position = int(np.flatnonzero(true_mask)[np.argmax(similarities[true_mask])])
            predicted_best_position = int(np.flatnonzero(predicted_mask)[np.argmax(similarities[predicted_mask])])

            centroids = {}
            for label in (error.subject, error.predicted_label):
                class_vectors = scaled_train[subjects[fit_indices] == label]
                centroid = class_vectors.mean(axis=0)
                centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
                centroids[label] = float(centroid @ normalized_error)

            model = make_model("ridge", seed=42, alpha=100.0)
            model.fit(x_train, subjects[fit_indices])
            ridge = model.named_steps["ridgeclassifier"]
            classes = model.classes_
            true_position = int(np.flatnonzero(classes == error.subject)[0])
            predicted_position = int(np.flatnonzero(classes == error.predicted_label)[0])
            coefficient_difference = ridge.coef_[predicted_position] - ridge.coef_[true_position]
            contributions = scaled_error * coefficient_difference
            band_contributions = {
                f"{low}-{high}": float(contributions[band_index * N_COV_PER_BAND:(band_index + 1) * N_COV_PER_BAND].sum())
                for band_index, (low, high) in enumerate(BANDS)
            }
            top_feature_indices = np.argsort(contributions)[-3:][::-1]
            top_features = []
            for feature_index in top_feature_indices:
                band_index = int(feature_index // N_COV_PER_BAND)
                within_band = int(feature_index % N_COV_PER_BAND)
                channel_a = CHANNELS[TRIU[0][within_band]]
                channel_b = CHANNELS[TRIU[1][within_band]]
                top_features.append({
                    "band": f"{BANDS[band_index][0]}-{BANDS[band_index][1]}",
                    "channels": f"{channel_a}-{channel_b}",
                    "contribution_to_wrong_minus_true": float(contributions[feature_index]),
                })

            same_group = (subjects == error.subject) & (tasks == task)
            quality_evidence = {}
            for column in ("raw_std_mean", "filtered_std_mean", "clip_fraction_mean", "clip_fraction_max"):
                values = metadata.loc[same_group, column].to_numpy()
                value = float(getattr(error, column))
                quality_evidence[column] = {
                    "value": value,
                    "subject_task_percentile": percentile(values, value),
                    "median_ratio": float(value / max(float(np.median(values)), 1e-12)),
                }
            flags = []
            quality_extreme = False
            for column, evidence in quality_evidence.items():
                percentile_value = evidence["subject_task_percentile"]
                ratio = evidence["median_ratio"]
                if column in ("raw_std_mean", "filtered_std_mean"):
                    quality_extreme |= (
                        max(percentile_value, 1 - percentile_value) >= 0.95
                        and (ratio >= 1.5 or ratio <= 2 / 3)
                    )
                elif evidence["value"] > 0 and percentile_value >= 0.95:
                    quality_extreme = True
            if quality_extreme:
                flags.append("quality_extreme_within_subject_task")
            if subjects[nearest_index] == error.predicted_label:
                flags.append("nearest_training_trial_supports_predicted_identity")
            elif subjects[nearest_index] == error.subject:
                flags.append("nearest_training_trial_supports_true_identity_but_global_boundary_fails")
            if error.trial_position > 0.8:
                flags.append("late_recording_segment")
            if error.margin < 0.05:
                flags.append("low_decision_margin")
            if not flags:
                flags.append("no_simple_quality_or_neighbor_explanation")

            explanations.append({
                "protocol": protocol,
                "path": error.path,
                "subject": int(error.subject),
                "predicted_label": int(error.predicted_label),
                "task": task,
                "trial": int(error.trial),
                "fold": int(error.fold),
                "trial_position": float(error.trial_position),
                "margin": float(error.margin),
                "score_gap": float(error.score_gap),
                "quality": quality_evidence,
                "nearest_training": {
                    "path": metadata.at[nearest_index, "path"],
                    "subject": int(subjects[nearest_index]),
                    "cosine_similarity": float(similarities[nearest_position]),
                },
                "best_true_subject_similarity": float(similarities[true_best_position]),
                "best_predicted_subject_similarity": float(similarities[predicted_best_position]),
                "true_centroid_cosine": centroids[error.subject],
                "predicted_centroid_cosine": centroids[error.predicted_label],
                "band_contributions_wrong_minus_true": band_contributions,
                "top_features_supporting_wrong_identity": top_features,
                "evidence_flags": flags,
            })
            print(
                f"{protocol} S{error.subject:02d}->S{error.predicted_label:02d} {Path(error.path).name} "
                f"nearest=S{subjects[nearest_index]:02d} flags={','.join(flags)}",
                flush=True,
            )

    output = ROOT / "research" / "train_error_explanations.json"
    output.write_text(json.dumps(explanations, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(explanations)} explanations -> {output}", flush=True)


if __name__ == "__main__":
    main()
