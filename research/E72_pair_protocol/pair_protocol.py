"""E72 shared Train-only query/template pairing and nested open-set protocol."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "E65_correlation_spd"))
from src.metrics import open_set_metrics  # noqa: E402
from run_e65 import WCCNPrototype, distance_ratio  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_train_metadata(config: dict) -> pd.DataFrame:
    expected = config["data"]
    metadata = pd.read_csv(ROOT / expected["train_metadata"])
    if len(metadata) != int(expected["expected_trials"]):
        raise ValueError("Train trial count mismatch")
    if metadata["subject"].nunique() != int(expected["expected_subjects"]):
        raise ValueError("Train subject count mismatch")
    if set(metadata["subject"].astype(int)) != set(range(1, 21)):
        raise ValueError("Train subjects must be encoded 1..20")
    if set(metadata["session"]) != {expected["session"]}:
        raise ValueError("Only Train session1 is allowed")
    if set(metadata["fs"]) != {int(expected["sampling_rate_hz"])}:
        raise ValueError("Sampling rate mismatch")
    if set(metadata["task"]) != set(expected["tasks"]):
        raise ValueError("Four-task metadata mismatch")
    if not metadata["path"].str.startswith("Train/").all():
        raise ValueError("Non-Train path in metadata")
    if metadata["path"].duplicated().any() or metadata["path"].str.contains(r"\.\.").any():
        raise ValueError("Train paths must be unique and must not traverse directories")
    if (metadata["trial"].astype(int) <= 0).any():
        raise ValueError("Trial indices must be positive")
    return metadata


def continuous_masks(metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    trial = metadata["trial"].to_numpy(dtype=float)
    maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy(dtype=float)
    early = trial <= 0.6 * maximum
    inner_known = (trial > 0.6 * maximum) & (trial <= 0.8 * maximum)
    before_forward = trial <= 0.8 * maximum
    forward = trial > 0.8 * maximum
    if not np.all(early | inner_known | forward):
        raise ValueError("Continuous blocks do not cover Train")
    return {
        "inner_fit": early,
        "inner_known": inner_known,
        "inner_unknown": before_forward,
        "outer_fit": before_forward,
        "outer_known": forward,
    }


def outer_groups(subjects: np.ndarray, config: dict) -> list[np.ndarray]:
    groups = np.array_split(
        np.random.default_rng(int(config["protocol"]["outer_seed"])).permutation(subjects),
        int(config["protocol"]["outer_folds"]),
    )
    if [len(group) for group in groups] != [4] * 5:
        raise ValueError("Expected five outer groups of four subjects")
    return [np.asarray(group, dtype=int) for group in groups]


def inner_splits(known_subjects: np.ndarray, outer_fold: int, config: dict) -> list[np.ndarray]:
    order = np.random.default_rng(100 + outer_fold).permutation(known_subjects)
    splits = np.array_split(order, int(config["protocol"]["inner_folds"]))
    if [len(split) for split in splits] != [8, 8]:
        raise ValueError("Expected two inner unknown groups of eight subjects")
    return [np.asarray(split, dtype=int) for split in splits]


def subject_roles(all_subjects: np.ndarray, outer_unknown: np.ndarray,
                  inner_unknown: np.ndarray) -> dict[str, np.ndarray]:
    known = np.setdiff1d(all_subjects, outer_unknown)
    enrolled = np.setdiff1d(known, inner_unknown)
    roles = {
        "outer_unknown": np.asarray(outer_unknown, dtype=int),
        "enrolled": np.asarray(enrolled, dtype=int),
        "calibration_unknown": np.asarray(inner_unknown, dtype=int),
    }
    role_sets = [set(values.tolist()) for values in roles.values()]
    disjoint = all(
        role_sets[left].isdisjoint(role_sets[right])
        for left in range(len(role_sets)) for right in range(left + 1, len(role_sets))
    )
    if not disjoint or set.union(*role_sets) != set(all_subjects.tolist()):
        raise ValueError("E72 subject roles are not a disjoint 4/8/8 partition")
    if [len(values) for values in roles.values()] != [4, 8, 8]:
        raise ValueError("E72 role sizes must be outer=4, enrolled=8, calibration=8")
    return roles


class CosinePrototype:
    def __init__(self, x: torch.Tensor, y: np.ndarray, classes: np.ndarray):
        self.classes = np.asarray(classes, dtype=int)
        lookup = {int(label): index for index, label in enumerate(self.classes)}
        target = torch.as_tensor(
            [lookup[int(label)] for label in y], dtype=torch.long, device=x.device
        )
        means = torch.stack([x[target == index].mean(0) for index in range(len(classes))])
        self.prototypes = means / means.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.whitening = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return normalized @ self.prototypes.T


def fit_spaces(x: torch.Tensor, y: np.ndarray, classes: np.ndarray, config: dict) -> dict:
    return {
        "cosine": CosinePrototype(x, y, classes),
        "wccn": WCCNPrototype(x, y, classes, config),
    }


def predict_spaces(models: dict, features: torch.Tensor, indices: np.ndarray,
                   batch_size: int) -> dict[str, np.ndarray]:
    classes = models["cosine"].classes
    outputs = {
        space: np.empty((len(indices), len(classes)), dtype=np.float32)
        for space in models
    }
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected_np = indices[start:start + batch_size]
            selected = torch.as_tensor(selected_np, dtype=torch.long, device=features.device)
            for space, model in models.items():
                outputs[space][start:start + len(selected_np)] = model(features[selected]).cpu().numpy()
    return outputs


def confidence_values(scores: np.ndarray, score_name: str) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    ordered = np.sort(values, axis=1)
    maximum, second = ordered[:, -1], ordered[:, -2]
    if score_name == "max_cosine":
        return maximum
    if score_name == "margin":
        return maximum - second
    if score_name == "distance_ratio":
        return distance_ratio(values)
    raise ValueError(f"Unknown confidence: {score_name}")


def calibrate_q05(inner_runs: list[dict], space: str, score_name: str) -> float:
    known = np.concatenate([
        confidence_values(run["known_scores"][space], score_name) for run in inner_runs
    ])
    if not len(known) or not np.isfinite(known).all():
        raise ValueError("Inner known confidence must be nonempty and finite")
    return float(np.quantile(known, 0.05, method="linear"))


def make_template_pairs(scores: np.ndarray, classes: np.ndarray,
                        true_labels: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Materialize query-template pairs only; never materialize trial-by-trial N^2 pairs."""
    scores = np.asarray(scores)
    n_queries, n_templates = scores.shape
    if n_templates != len(classes):
        raise ValueError("Score columns and template classes differ")
    query_index = np.repeat(np.arange(n_queries, dtype=int), n_templates)
    template_index = np.tile(np.arange(n_templates, dtype=int), n_queries)
    if true_labels is None:
        same = np.zeros(len(query_index), dtype=np.int8)
    else:
        true_labels = np.asarray(true_labels, dtype=int)
        if len(true_labels) != n_queries:
            raise ValueError("Pair label length mismatch")
        same = (np.asarray(classes)[template_index] == true_labels[query_index]).astype(np.int8)
    return {"query_index": query_index, "template_index": template_index, "same": same}


def evaluate_queries(known_true: np.ndarray, known_scores: np.ndarray,
                     unknown_scores: np.ndarray, classes: np.ndarray,
                     score_name: str, threshold: float) -> tuple[dict, dict]:
    known_true = np.asarray(known_true, dtype=int)
    classes = np.asarray(classes, dtype=int)
    known_confidence = confidence_values(known_scores, score_name)
    unknown_confidence = confidence_values(unknown_scores, score_name)
    direct_known = classes[np.asarray(known_scores).argmax(axis=1)]
    direct_unknown = classes[np.asarray(unknown_scores).argmax(axis=1)]
    known_pred = np.where(known_confidence >= threshold, direct_known, 0).astype(int)
    unknown_pred = np.where(unknown_confidence >= threshold, direct_unknown, 0).astype(int)
    detection_true = np.concatenate([
        np.ones(len(known_true), dtype=int), np.zeros(len(unknown_pred), dtype=int)
    ])
    accepted = np.concatenate([known_pred != 0, unknown_pred != 0])
    detection_scores = np.concatenate([known_confidence, unknown_confidence])
    c = float(np.mean(known_pred == known_true))
    u = float(np.mean(unknown_pred == 0))
    direct_c = float(np.mean(direct_known == known_true))
    metrics = {
        "N_known": int(len(known_true)),
        "N_unknown": int(len(unknown_pred)),
        "N_same_pairs": int(len(known_true)),
        "N_different_pairs": int(len(known_true) * (len(classes) - 1) + len(unknown_pred) * len(classes)),
        "threshold": float(threshold),
        "direct_C": direct_c,
        "continuous_block_accuracy": direct_c,
        "continuous_block_accuracy_with_rejection": c,
        "identity_macro_F1_direct": float(f1_score(
            known_true, direct_known, labels=classes, average="macro", zero_division=0
        )),
        "C": c,
        "U": u,
        "known_acceptance": float(np.mean(known_pred != 0)),
        "identity_macro_F1": float(f1_score(
            known_true, known_pred, labels=classes, average="macro", zero_division=0
        )),
        "unknown_F1": float(f1_score(detection_true, accepted, pos_label=0, zero_division=0)),
        "joint_macro_F1": float(f1_score(
            np.concatenate([known_true, np.zeros(len(unknown_pred), dtype=int)]),
            np.concatenate([known_pred, unknown_pred]),
            labels=np.concatenate([[0], classes]), average="macro", zero_division=0,
        )),
        **open_set_metrics(known_confidence, unknown_confidence),
        "unknown_AUPR": float(average_precision_score(1 - detection_true, -detection_scores)),
        "C_U_proxy_not_official": float(100 * c**0.40 * u**0.25) if c > 0 and u > 0 else 0.0,
    }
    metrics["U_at_95"] = metrics.pop("unknown_rejection_at_95pct_known_acceptance")
    metrics["AUROC"] = metrics.pop("unknown_auroc")
    records = {
        "known_true": known_true,
        "direct_known": direct_known,
        "known_pred": known_pred,
        "unknown_pred": unknown_pred,
        "known_confidence": known_confidence,
        "unknown_confidence": unknown_confidence,
    }
    return metrics, records


def mean_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key, value in rows[0].items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
    }


def std_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {
        key: float(np.std([row[key] for row in rows]))
        for key, value in rows[0].items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
    }


def pooled_metrics(records: list[dict]) -> dict:
    known_true = np.concatenate([row["known_true"] for row in records])
    direct_known = np.concatenate([row["direct_known"] for row in records])
    known_pred = np.concatenate([row["known_pred"] for row in records])
    unknown_pred = np.concatenate([row["unknown_pred"] for row in records])
    known_confidence = np.concatenate([row["known_confidence"] for row in records])
    unknown_confidence = np.concatenate([row["unknown_confidence"] for row in records])
    classes = np.arange(1, 21, dtype=int)
    detection_true = np.concatenate([
        np.ones(len(known_true), dtype=int), np.zeros(len(unknown_pred), dtype=int)
    ])
    accepted = np.concatenate([known_pred != 0, unknown_pred != 0])
    detection_scores = np.concatenate([known_confidence, unknown_confidence])
    c = float(np.mean(known_pred == known_true))
    u = float(np.mean(unknown_pred == 0))
    direct_c = float(np.mean(direct_known == known_true))
    metrics = {
        "N_known": int(len(known_true)),
        "N_unknown": int(len(unknown_pred)),
        "direct_C": direct_c,
        "continuous_block_accuracy": direct_c,
        "continuous_block_accuracy_with_rejection": c,
        "identity_20class_macro_F1_direct": float(f1_score(
            known_true, direct_known, labels=classes, average="macro", zero_division=0
        )),
        "C": c,
        "U": u,
        "known_acceptance": float(np.mean(known_pred != 0)),
        "identity_20class_macro_F1": float(f1_score(
            known_true, known_pred, labels=classes, average="macro", zero_division=0
        )),
        "unknown_F1": float(f1_score(detection_true, accepted, pos_label=0, zero_division=0)),
        "joint_21class_macro_F1": float(f1_score(
            np.concatenate([known_true, np.zeros(len(unknown_pred), dtype=int)]),
            np.concatenate([known_pred, unknown_pred]),
            labels=np.arange(21), average="macro", zero_division=0,
        )),
        **open_set_metrics(known_confidence, unknown_confidence),
        "unknown_AUPR": float(average_precision_score(1 - detection_true, -detection_scores)),
        "C_U_proxy_not_official": float(100 * c**0.40 * u**0.25) if c > 0 and u > 0 else 0.0,
    }
    metrics["U_at_95"] = metrics.pop("unknown_rejection_at_95pct_known_acceptance")
    metrics["AUROC"] = metrics.pop("unknown_auroc")
    return metrics
