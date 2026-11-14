"""E86: shared outer-test prediction CSV schema, metrics and comparison.

External BCIC IV-2a experiments evaluate on the held-out outer queries (run 5).
Each experiment exports one row per evaluated sample so the outer metrics can be
recomputed from the CSV and compared across experiments.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

COLUMNS = [
    "experiment", "outer_fold", "direction", "method", "role",
    "sample_id", "session", "run", "task", "artifact",
    "true_identity", "pred_label", "accepted", "confidence", "threshold",
]
ROLES = {"enrolled", "outer_unknown"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_records(*, experiment: str, outer_fold: int, direction: str, method: str,
                  roles: np.ndarray, sample_id: np.ndarray, session: np.ndarray,
                  run: np.ndarray, task: np.ndarray, artifact: np.ndarray,
                  true_identity: np.ndarray, pred_label: np.ndarray,
                  accepted: np.ndarray, confidence: np.ndarray,
                  threshold: float) -> pd.DataFrame:
    frame = pd.DataFrame({
        "experiment": experiment,
        "outer_fold": int(outer_fold),
        "direction": direction,
        "method": method,
        "role": np.asarray(roles, dtype=str),
        "sample_id": np.asarray(sample_id, dtype=str),
        "session": np.asarray(session),
        "run": np.asarray(run, dtype=int),
        "task": np.asarray(task, dtype=int),
        "artifact": np.asarray(artifact, dtype=bool),
        "true_identity": np.asarray(true_identity, dtype=int),
        "pred_label": np.asarray(pred_label, dtype=int),
        "accepted": np.asarray(accepted, dtype=bool),
        "confidence": np.asarray(confidence, dtype=np.float64),
        "threshold": float(threshold),
    })
    return frame[COLUMNS]


def write_csv(frames: list[pd.DataFrame], path: Path) -> dict:
    frame = pd.concat(frames, ignore_index=True)
    validate_csv_frame(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    written = pd.read_csv(path)
    validate_csv_frame(written)
    manifest = {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "n_rows": int(len(written)),
        "n_groups": int(written.groupby(["outer_fold", "direction", "method"]).ngroups),
        "columns": list(written.columns),
    }
    return manifest


def validate_csv_frame(frame: pd.DataFrame) -> None:
    if list(frame.columns) != COLUMNS:
        raise ValueError(f"CSV 列必须严格为 {COLUMNS}")
    if frame.isna().any().any():
        raise ValueError("CSV 含缺失值")
    if not set(frame["role"]).issubset(ROLES):
        raise ValueError("role 只能是 enrolled 或 outer_unknown")
    if not (frame["pred_label"] >= 0).all():
        raise ValueError("pred_label 必须非负；0 表示拒识")
    if not np.isfinite(frame["confidence"].to_numpy()).all():
        raise ValueError("confidence 含非有限值")


def metrics_from_frame(frame: pd.DataFrame) -> dict:
    known = frame[frame["role"] == "enrolled"]
    unknown = frame[frame["role"] == "outer_unknown"]
    if known.empty or unknown.empty:
        raise ValueError("每个分组必须同时包含 enrolled 与 outer_unknown")
    classes = np.sort(known["true_identity"].unique())
    known_conf = known["confidence"].to_numpy()
    unknown_conf = unknown["confidence"].to_numpy()
    threshold = float(frame["threshold"].iloc[0])
    acceptance = known["accepted"].to_numpy(dtype=bool)
    unknown_accept = unknown["accepted"].to_numpy(dtype=bool)
    labels = np.r_[np.ones(len(known_conf)), np.zeros(len(unknown_conf))]
    detection = np.r_[acceptance, unknown_accept]
    direct = known["pred_label"].to_numpy()
    correct = direct == known["true_identity"].to_numpy()
    fpr, tpr, _ = roc_curve(labels, np.r_[known_conf, unknown_conf])
    dir_threshold = float(np.quantile(unknown_conf, 0.99, method="linear"))
    return {
        "N_known": int(len(known)),
        "N_unknown": int(len(unknown)),
        "threshold": threshold,
        "AUROC": float(roc_auc_score(labels, np.r_[known_conf, unknown_conf])),
        "detection_EER": float(np.min(np.maximum(fpr, 1.0 - tpr))),
        "C": float(np.mean(correct)),
        "identity_macro_F1": float(f1_score(known["true_identity"], direct, labels=classes, average="macro", zero_division=0)),
        "unknown_F1": float(f1_score(labels, detection, pos_label=0, zero_division=0)),
        "known_acceptance": float(np.mean(acceptance)),
        "unknown_false_acceptance": float(np.mean(unknown_accept)),
        "unknown_rejection": float(1.0 - np.mean(unknown_accept)),
        "U_at_95": float(np.mean(unknown_conf < np.quantile(known_conf, 0.05, method="linear"))),
        "DIR_at_FPIR_1": float(np.mean(correct & (known_conf >= dir_threshold))),
    }


def metrics_by_group(frame: pd.DataFrame) -> dict:
    return {
        f"fold{int(fold)}/{direction}/{method}": metrics_from_frame(group)
        for (fold, direction, method), group in frame.groupby(["outer_fold", "direction", "method"])
    }


def compare_frames(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    for frame, name in ((a, "A"), (b, "B")):
        validate_csv_frame(frame)
    keys = ["outer_fold", "direction", "sample_id"]
    left = a.set_index(keys)["pred_label"]
    right = b.set_index(keys)["pred_label"].reindex(left.index)
    if right.isna().any():
        raise ValueError("两个 CSV 的样本键不一致，B 缺少 A 中的样本")
    pa, pb = left.to_numpy(), right.to_numpy().astype(int)
    same = pa == pb
    transition = np.zeros((int(max(pa.max(), pb.max())) + 1,) * 2, dtype=int)
    np.add.at(transition, (pa, pb), 1)
    return {
        "n": int(len(pa)),
        "agreement_rate": float(same.mean()),
        "n_diff": int((~same).sum()),
        "a_reject_b_accept": int(((pa == 0) & (pb != 0)).sum()),
        "a_accept_b_reject": int(((pa != 0) & (pb == 0)).sum()),
        "both_accept_different_identity": int(((pa != 0) & (pb != 0) & (pa != pb)).sum()),
        "transition": transition.tolist(),
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
