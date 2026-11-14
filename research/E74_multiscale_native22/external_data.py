"""Load only E73-derived BCIC IV-2a arrays and construct subject/run-isolated folds."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

E73_DIR = Path(__file__).resolve().parents[1] / "E73_external_multisession"


@dataclass
class ExternalData:
    raw: torch.Tensor
    subject: np.ndarray
    session: np.ndarray
    run: np.ndarray
    task: np.ndarray
    artifact: np.ndarray
    source_file: np.ndarray
    channel_mask: np.ndarray

    def indices(self, subjects: np.ndarray, *, session: str | None = None,
                runs: list[int] | None = None) -> np.ndarray:
        mask = np.isin(self.subject, subjects)
        if session is not None:
            if session not in {"T", "E"}:
                raise ValueError(f"Invalid session {session}")
            mask &= self.session == (0 if session == "T" else 1)
        if runs is not None:
            mask &= np.isin(self.run, runs)
        return np.flatnonzero(mask)


def subject_folds(config: dict) -> list[dict[str, np.ndarray]]:
    subjects = np.asarray(config["data"]["subjects"], dtype=int)
    protocol = config["protocol"]
    groups = np.array_split(
        np.random.default_rng(int(protocol["outer_seed"])).permutation(subjects),
        int(protocol["outer_folds"]),
    )
    folds = []
    for outer_index, outer in enumerate(groups, start=1):
        known = np.setdiff1d(subjects, outer)
        order = np.random.default_rng(100 + outer_index).permutation(known)
        n_calibration = int(protocol["calibration_unknown_subjects"])
        calibration = np.sort(order[:n_calibration])
        enrolled = np.sort(order[n_calibration:])
        roles = {
            "enrolled": np.asarray(enrolled, dtype=int),
            "calibration_unknown": np.asarray(calibration, dtype=int),
            "outer_unknown": np.asarray(outer, dtype=int),
        }
        expected = [
            int(protocol["enrolled_subjects"]),
            n_calibration,
            int(protocol["outer_unknown_subjects"]),
        ]
        if [len(values) for values in roles.values()] != expected:
            raise ValueError(f"Unexpected E73 role sizes: {roles}")
        role_sets = [set(values.tolist()) for values in roles.values()]
        if any(role_sets[left] & role_sets[right]
               for left in range(3) for right in range(left + 1, 3)):
            raise ValueError("E73 subject roles overlap")
        if set.union(*role_sets) != set(subjects.tolist()):
            raise ValueError("E73 subject roles do not cover all external subjects")
        folds.append(roles)
    return folds


def load_external_data(config: dict, device: torch.device) -> ExternalData:
    cfg = config["data"]
    root = E73_DIR / cfg["derived_root"]
    raw_parts, subject_parts, session_parts = [], [], []
    run_parts, task_parts, artifact_parts, file_parts = [], [], [], []
    channel_mask = None
    for subject in cfg["subjects"]:
        for session_index, session in enumerate(cfg["sessions"]):
            path = root / f"subject_{int(subject):02d}" / f"A{int(subject):02d}{session}_300Hz_1-40Hz_4s.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing E73 derived session: {path}")
            with np.load(path, allow_pickle=False) as item:
                raw = np.asarray(item[cfg["feature_key"]], dtype=np.float32)
                expected = (int(cfg["expected_trials_per_subject_session"]),
                            int(cfg["channels"]), int(cfg["samples_per_trial"]))
                if raw.shape != expected or not np.isfinite(raw).all():
                    raise ValueError(f"Invalid external array {path}: {raw.shape}, expected {expected}")
                if float(np.asarray(item["sfreq"])) != float(cfg["sampling_rate_hz"]):
                    raise ValueError(f"Sampling rate mismatch: {path}")
                if int(np.asarray(item["subject"])) != int(subject) or str(np.asarray(item["session"])) != session:
                    raise ValueError(f"Subject/session metadata mismatch: {path}")
                run = np.asarray(item["run_index"], dtype=int)
                task = np.asarray(item["task_label"], dtype=int)
                artifact = np.asarray(item["artifact"], dtype=bool)
                if not (len(run) == len(task) == len(artifact) == len(raw)):
                    raise ValueError(f"Trial metadata is not row-aligned: {path}")
                if set(run) != set(range(int(cfg["expected_runs_per_session"]))):
                    raise ValueError(f"Expected six source runs: {path}")
                mask = np.asarray(item["target_mask"], dtype=bool)
                target_channels = np.asarray(item["target_channels"])
                if mask.shape != target_channels.shape:
                    raise ValueError(f"Target mask mismatch: {path}")
                if cfg["feature_key"] == "target_data":
                    if int(mask.sum()) != int(cfg["expected_observed_target_channels"]):
                        raise ValueError(f"Observed target channel count mismatch: {path}")
                    if np.any(raw[:, ~mask]):
                        raise ValueError(f"Unobserved channels contain nonzero values: {path}")
                if channel_mask is None:
                    channel_mask = mask
                elif not np.array_equal(channel_mask, mask):
                    raise ValueError("Channel mask differs across external sessions")
                raw_parts.append(raw)
                subject_parts.append(np.full(len(raw), subject, dtype=np.int16))
                session_parts.append(np.full(len(raw), session_index, dtype=np.int8))
                run_parts.append(run.astype(np.int16))
                task_parts.append(task.astype(np.int16))
                artifact_parts.append(artifact)
                file_parts.append(np.full(len(raw), path.relative_to(E73_DIR).as_posix()))
    raw_all = np.concatenate(raw_parts)
    data = ExternalData(
        raw=torch.as_tensor(raw_all, dtype=torch.float32, device=device),
        subject=np.concatenate(subject_parts),
        session=np.concatenate(session_parts),
        run=np.concatenate(run_parts),
        task=np.concatenate(task_parts),
        artifact=np.concatenate(artifact_parts),
        source_file=np.concatenate(file_parts),
        channel_mask=channel_mask,
    )
    if len(raw_all) != len(cfg["subjects"]) * len(cfg["sessions"]) * int(cfg["expected_trials_per_subject_session"]):
        raise ValueError("External trial total mismatch")
    return data
