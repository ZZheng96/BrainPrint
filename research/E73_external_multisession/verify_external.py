"""Verify and preprocess all requested E73 BCIC IV-2a external MAT sessions.

The script reads only E73/cache/bciciv2a/raw/subject_XX/Axx[T/E].mat and the
public AxxE_labels.mat files. It never discovers or accesses BrainPrint
competition Train/Test files. Derived arrays retain all trials and artifact
flags; no model is trained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt

E73_DIR = Path(__file__).resolve().parent
RAW_DIR = E73_DIR / "cache" / "bciciv2a" / "raw"
DERIVED_DIR = E73_DIR / "cache" / "bciciv2a" / "derived"
VERIFICATION_PATH = E73_DIR / "verification.json"
MANIFEST_PATH = E73_DIR / "manifest.json"

SOURCE_CHANNELS = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
)
TARGET_CHANNELS = (
    "P3", "C3", "F3", "Fz", "F4", "C4", "P4", "Cz", "Fp1", "Fp2",
    "T3", "T5", "O1", "O2", "F7", "F8", "T6", "T4",
)
TASK_NAMES = {1: "left_hand", 2: "right_hand", 3: "feet", 4: "tongue"}
N_VALID_RUNS = 6
N_TRIALS_PER_RUN = 48
N_TRIALS_PER_SESSION = 288
N_TARGET_SAMPLES = 1200
ALL_SUBJECTS = "1,2,3,4,5,6,7,8,9"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_subjects(value: str) -> list[int]:
    subjects = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not subjects or any(subject < 1 or subject > 9 for subject in subjects):
        raise ValueError("Subjects must be comma-separated integers in 1..9")
    if len(set(subjects)) != len(subjects):
        raise ValueError("Subjects must not contain duplicates")
    return subjects


def target_mapping() -> dict:
    source_lookup = {name: index for index, name in enumerate(SOURCE_CHANNELS)}
    indices = [source_lookup.get(name, -1) for name in TARGET_CHANNELS]
    return {
        "target_channels": list(TARGET_CHANNELS),
        "source_channels": list(SOURCE_CHANNELS),
        "target_source_index": indices,
        "observed_target_mask": [index >= 0 for index in indices],
        "directly_observed": [name for name, index in zip(TARGET_CHANNELS, indices) if index >= 0],
        "unobserved": [name for name, index in zip(TARGET_CHANNELS, indices) if index < 0],
        "policy": (
            "Exact name matches only. Missing target channels are zero-filled placeholders and "
            "are marked False in target_mask; they must never be used as measured signals. "
            "No spatial interpolation or fabricated renaming is used."
        ),
    }


def hash_against_manifest(path: Path) -> dict:
    relative = path.relative_to(E73_DIR).as_posix()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    record = manifest.get("files", {}).get(relative)
    if not record or not record.get("sha256"):
        raise ValueError(f"No downloaded-source SHA-256 in manifest for {relative}")
    actual_size = path.stat().st_size
    actual_sha256 = sha256_file(path)
    expected_bytes = record.get("bytes")
    matches = actual_sha256 == record["sha256"] and (
        expected_bytes is None or actual_size == expected_bytes
    )
    result = {
        "bytes": actual_size,
        "sha256": actual_sha256,
        "manifest_bytes": expected_bytes,
        "manifest_sha256": record["sha256"],
        "matches_manifest": matches,
        "hash_provenance": "SHA-256 computed locally at download; not an upstream-signed checksum",
    }
    if not matches:
        raise ValueError(f"Hash/size mismatch against manifest for {relative}: {result}")
    return result


def coerce_labels(values, *, expected_count: int, context: str) -> np.ndarray:
    raw = np.asarray(values).reshape(-1)
    if len(raw) != expected_count:
        raise ValueError(f"{context}: expected {expected_count} labels, got {len(raw)}")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: labels are not numeric") from exc
    if not np.isfinite(numeric).all():
        raise ValueError(f"{context}: labels contain NaN or infinity")
    labels = numeric.astype(np.int64)
    if not np.array_equal(numeric, labels):
        raise ValueError(f"{context}: labels are not integers")
    if not set(labels).issubset(TASK_NAMES):
        raise ValueError(f"{context}: unexpected task labels {sorted(set(labels))}")
    return labels


def load_public_e_labels(subject: int) -> tuple[np.ndarray, dict, str]:
    path = RAW_DIR / f"subject_{subject:02d}" / f"A{subject:02d}E_labels.mat"
    if not path.is_file():
        raise FileNotFoundError(f"Required public evaluation labels are missing: {path}")
    hash_check = hash_against_manifest(path)
    mat = loadmat(path, struct_as_record=False, squeeze_me=True)
    if "classlabel" not in mat:
        raise ValueError(f"{path.name} does not contain the public 'classlabel' variable")
    labels = coerce_labels(
        mat["classlabel"],
        expected_count=N_TRIALS_PER_SESSION,
        context=f"{path.name} classlabel",
    )
    return labels, hash_check, path.relative_to(E73_DIR).as_posix()


def embedded_e_label_status(run, public_labels: np.ndarray, context: str) -> str:
    if not hasattr(run, "y"):
        return "absent_used_public_true_labels"
    raw = np.asarray(run.y).reshape(-1)
    if raw.size != len(public_labels):
        raise ValueError(f"{context}: embedded/public trial count mismatch")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: embedded labels are not numeric") from exc
    finite = np.isfinite(numeric)
    if finite.all():
        embedded = numeric.astype(np.int64)
        if np.array_equal(numeric, embedded) and set(embedded).issubset(TASK_NAMES):
            if not np.array_equal(embedded, public_labels):
                raise ValueError(f"{context}: embedded labels disagree with public true_labels")
            return "matches_public_true_labels"
    if np.all(~finite | (numeric == 0)):
        return "missing_or_placeholder_used_public_true_labels"
    raise ValueError(f"{context}: embedded labels are partially invalid or ambiguous")


def _trial_window(
    run,
    data: np.ndarray,
    labels: np.ndarray,
    *,
    offset_seconds: float,
    duration_seconds: float,
) -> tuple[np.ndarray, dict]:
    fs = int(np.asarray(run.fs).reshape(-1)[0])
    if fs != 250:
        raise ValueError(f"Expected 250 Hz source, got {fs}")
    trials = np.asarray(run.trial).reshape(-1).astype(np.int64)
    if len(trials) != N_TRIALS_PER_RUN:
        raise ValueError(f"Expected {N_TRIALS_PER_RUN} trials per run, got {len(trials)}")
    if len(labels) != len(trials):
        raise ValueError(f"Trial/label mismatch: {len(trials)} vs {len(labels)}")
    artifacts = (
        np.asarray(run.artifacts).reshape(-1).astype(bool)
        if hasattr(run, "artifacts") and np.asarray(run.artifacts).size
        else np.zeros(len(trials), dtype=bool)
    )
    if len(artifacts) != len(trials):
        raise ValueError("Artifact flags do not match trial count")

    start_offset = int(round(offset_seconds * fs))
    n_source_samples = int(round(duration_seconds * fs))
    starts = trials - 1 + start_offset
    stops = starts + n_source_samples
    if np.any(starts < 0) or np.any(stops > len(data)):
        raise ValueError("A requested 2--6 s trial window exceeds its source run")

    sos = butter(4, (1.0, 40.0), btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, data.astype(np.float64), axis=0)
    epochs = np.stack([filtered[start:stop] for start, stop in zip(starts, stops)])
    epochs = resample_poly(epochs, up=6, down=5, axis=1).astype(np.float32)
    expected_shape = (len(trials), N_TARGET_SAMPLES, len(SOURCE_CHANNELS))
    if epochs.shape != expected_shape:
        raise ValueError(f"Unexpected preprocessed shape {epochs.shape}; expected {expected_shape}")
    if not np.isfinite(epochs).all():
        raise ValueError("Non-finite values after preprocessing")
    metadata = {
        "fs_source": fs,
        "trials": trials,
        "labels": labels,
        "artifacts": artifacts,
        "source_window_start_samples": starts,
        "source_window_stop_samples": stops,
    }
    return epochs, metadata


def validate_arrays(arrays: dict[str, np.ndarray], subject: int, session: str) -> None:
    n = N_TRIALS_PER_SESSION
    if arrays["data"].shape != (n, len(SOURCE_CHANNELS), N_TARGET_SAMPLES):
        raise ValueError(f"Invalid native data shape: {arrays['data'].shape}")
    if arrays["target_data"].shape != (n, len(TARGET_CHANNELS), N_TARGET_SAMPLES):
        raise ValueError(f"Invalid target data shape: {arrays['target_data'].shape}")
    if arrays["data"].dtype != np.float32 or arrays["target_data"].dtype != np.float32:
        raise ValueError("Derived signals must both be float32")
    if not np.isfinite(arrays["data"]).all() or not np.isfinite(arrays["target_data"]).all():
        raise ValueError("Derived signals contain non-finite values")

    for key in (
        "task_label", "task_name", "artifact", "source_run_index", "run_index",
        "trial_in_run", "source_trial",
    ):
        if arrays[key].shape != (n,):
            raise ValueError(f"{key} has shape {arrays[key].shape}; expected {(n,)}")
    if arrays["source_channels"].tolist() != list(SOURCE_CHANNELS):
        raise ValueError("Source channel order changed")
    if arrays["target_channels"].tolist() != list(TARGET_CHANNELS):
        raise ValueError("Target channel order changed")
    mapping = target_mapping()
    if not np.array_equal(arrays["target_source_index"], mapping["target_source_index"]):
        raise ValueError("Target/source index map changed")
    if not np.array_equal(arrays["target_mask"], mapping["observed_target_mask"]):
        raise ValueError("Target observation mask changed")

    labels = arrays["task_label"]
    if not set(int(value) for value in labels).issubset(TASK_NAMES):
        raise ValueError("Derived task labels are invalid")
    expected_names = np.asarray([TASK_NAMES[int(label)] for label in labels])
    if not np.array_equal(arrays["task_name"], expected_names):
        raise ValueError("task_name is not aligned with task_label")
    if arrays["artifact"].dtype != np.bool_:
        raise ValueError("artifact must be boolean")
    if int(np.asarray(arrays["subject"]).item()) != subject:
        raise ValueError("Subject metadata mismatch")
    if str(np.asarray(arrays["session"]).item()) != session:
        raise ValueError("Session metadata mismatch")
    if float(np.asarray(arrays["sfreq"]).item()) != 300.0:
        raise ValueError("Derived sampling frequency is not 300 Hz")

    expected_run_index = np.repeat(np.arange(N_VALID_RUNS), N_TRIALS_PER_RUN)
    expected_trial_in_run = np.tile(np.arange(N_TRIALS_PER_RUN), N_VALID_RUNS)
    if not np.array_equal(arrays["run_index"], expected_run_index):
        raise ValueError("run_index is not aligned as six consecutive 48-trial runs")
    if not np.array_equal(arrays["trial_in_run"], expected_trial_in_run):
        raise ValueError("trial_in_run is not aligned within runs")
    source_runs = []
    for run_index in range(N_VALID_RUNS):
        selected = arrays["source_run_index"][arrays["run_index"] == run_index]
        if len(np.unique(selected)) != 1:
            raise ValueError("A derived run maps to multiple source runs")
        source_runs.append(int(selected[0]))
    if len(set(source_runs)) != N_VALID_RUNS:
        raise ValueError("Source run indices are not unique")
    if np.any(arrays["source_trial"] <= 0):
        raise ValueError("source_trial must retain 1-based positive source positions")

    observed = arrays["target_mask"]
    if np.any(arrays["target_data"][:, ~observed] != 0):
        raise ValueError("Unobserved target channels contain nonzero placeholder data")
    for target_index, source_index in enumerate(arrays["target_source_index"]):
        if source_index >= 0 and not np.array_equal(
            arrays["target_data"][:, target_index], arrays["data"][:, source_index]
        ):
            raise ValueError("An observed target channel differs from its measured source channel")


def process_session(
    subject: int,
    session: str,
    *,
    offset_seconds: float,
    duration_seconds: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    filename = f"A{subject:02d}{session}.mat"
    path = RAW_DIR / f"subject_{subject:02d}" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required external session is missing: {path}")
    hash_check = hash_against_manifest(path)
    public_e_labels = None
    public_label_hash = None
    public_label_file = None
    if session == "E":
        public_e_labels, public_label_hash, public_label_file = load_public_e_labels(subject)

    mat = loadmat(path, struct_as_record=False, squeeze_me=True)
    run_array = np.atleast_1d(mat["data"])
    native_epochs = []
    all_labels = []
    all_artifacts = []
    all_source_run = []
    all_valid_run = []
    all_trial_in_run = []
    all_source_trial = []
    run_records = []
    class_counts = Counter()
    public_label_offset = 0

    for source_run_index, run in enumerate(run_array):
        if not hasattr(run, "trial") or np.asarray(run.trial).size == 0:
            run_records.append({"source_run_index": source_run_index, "status": "skipped_no_trials"})
            continue
        data = np.asarray(run.X)
        if data.ndim != 2 or data.shape[1] != 25:
            raise ValueError(f"Run {source_run_index}: expected samples x 25, got {data.shape}")
        n_trials = len(np.asarray(run.trial).reshape(-1))
        context = f"{filename} source run {source_run_index}"
        if session == "E":
            stop = public_label_offset + n_trials
            labels = public_e_labels[public_label_offset:stop]
            if len(labels) != n_trials:
                raise ValueError(f"{context}: public true_labels ended early")
            label_status = embedded_e_label_status(run, labels, context)
            public_label_offset = stop
        else:
            labels = coerce_labels(run.y, expected_count=n_trials, context=context)
            label_status = "embedded_training_labels"
        epochs, meta = _trial_window(
            run,
            data[:, : len(SOURCE_CHANNELS)],
            labels,
            offset_seconds=offset_seconds,
            duration_seconds=duration_seconds,
        )
        valid_run_index = len(native_epochs)
        native_epochs.append(epochs)
        all_labels.append(meta["labels"])
        all_artifacts.append(meta["artifacts"])
        all_source_run.append(np.full(n_trials, source_run_index, dtype=np.int16))
        all_valid_run.append(np.full(n_trials, valid_run_index, dtype=np.int16))
        all_trial_in_run.append(np.arange(n_trials, dtype=np.int16))
        all_source_trial.append(meta["trials"].astype(np.int32))
        counts = Counter(int(label) for label in meta["labels"])
        class_counts.update(counts)
        run_records.append(
            {
                "source_run_index": source_run_index,
                "valid_run_index": valid_run_index,
                "status": "used",
                "source_shape_samples_by_channels": list(data.shape),
                "n_trials": n_trials,
                "n_artifacts": int(meta["artifacts"].sum()),
                "task_counts_original_labels": {str(key): value for key, value in sorted(counts.items())},
                "label_validation": label_status,
                "first_trial_source_sample_1indexed": int(meta["trials"][0]),
                "last_trial_source_sample_1indexed": int(meta["trials"][-1]),
            }
        )

    if len(native_epochs) != N_VALID_RUNS:
        raise ValueError(f"{filename}: expected {N_VALID_RUNS} valid runs, got {len(native_epochs)}")
    if session == "E" and public_label_offset != N_TRIALS_PER_SESSION:
        raise ValueError(
            f"{filename}: consumed {public_label_offset} public labels, expected {N_TRIALS_PER_SESSION}"
        )
    native = np.concatenate(native_epochs).transpose(0, 2, 1).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.int16)
    artifacts = np.concatenate(all_artifacts).astype(bool)
    if len(labels) != N_TRIALS_PER_SESSION:
        raise ValueError(f"{filename}: expected 288 trials, got {len(labels)}")
    if class_counts != Counter({label: 72 for label in TASK_NAMES}):
        raise ValueError(f"{filename}: expected 72 trials per class, got {dict(class_counts)}")

    mapping = target_mapping()
    target = np.zeros((len(native), len(TARGET_CHANNELS), N_TARGET_SAMPLES), dtype=np.float32)
    for target_index, source_index in enumerate(mapping["target_source_index"]):
        if source_index >= 0:
            target[:, target_index] = native[:, source_index]

    label_source = "public_true_labels" if session == "E" else "embedded_run_y"
    arrays = {
        "data": native,
        "target_data": target,
        "target_mask": np.asarray(mapping["observed_target_mask"], dtype=bool),
        "target_source_index": np.asarray(mapping["target_source_index"], dtype=np.int16),
        "source_channels": np.asarray(SOURCE_CHANNELS),
        "target_channels": np.asarray(TARGET_CHANNELS),
        "task_label": labels,
        "task_name": np.asarray([TASK_NAMES[int(label)] for label in labels]),
        "task_label_source": np.asarray(label_source, dtype="U32"),
        "artifact": artifacts,
        "source_run_index": np.concatenate(all_source_run),
        "run_index": np.concatenate(all_valid_run),
        "trial_in_run": np.concatenate(all_trial_in_run),
        "source_trial": np.concatenate(all_source_trial),
        "subject": np.asarray(subject, dtype=np.int16),
        "session": np.asarray(session),
        "sfreq": np.asarray(300.0, dtype=np.float32),
        "unit": np.asarray("uV"),
        "source_sha256": np.asarray(hash_check["sha256"], dtype="U64"),
        "public_e_labels_sha256": np.asarray(
            public_label_hash["sha256"] if public_label_hash else "", dtype="U64"
        ),
        "filter_band_hz": np.asarray([1.0, 40.0], dtype=np.float32),
        "trial_window_seconds": np.asarray([offset_seconds, offset_seconds + duration_seconds], dtype=np.float32),
    }
    validate_arrays(arrays, subject, session)
    label_validation = {
        "source": label_source,
        "n_trials": len(labels),
        "all_valid_runs_checked": True,
    }
    if session == "E":
        label_validation.update(
            public_file=public_label_file,
            public_hash_check=public_label_hash,
            public_labels_consumed_in_run_order=public_label_offset,
        )
    summary = {
        "file": path.relative_to(E73_DIR).as_posix(),
        "subject": subject,
        "session": session,
        "hash_check": hash_check,
        "label_validation": label_validation,
        "n_source_runs_including_empty": len(run_array),
        "n_valid_runs": len(native_epochs),
        "n_trials": len(labels),
        "n_artifact_trials": int(artifacts.sum()),
        "task_counts": {TASK_NAMES[key]: value for key, value in sorted(class_counts.items())},
        "runs": run_records,
        "derived_native_shape": list(native.shape),
        "derived_target_shape": list(target.shape),
        "derived_dtype": str(native.dtype),
        "derived_unit": "uV",
        "derived_finite": True,
        "task_run_session_alignment_verified": True,
        "observed_target_channels_verified_against_source": True,
    }
    return summary, arrays


def update_manifest(verification: dict) -> None:
    if not MANIFEST_PATH.exists():
        return
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for session in verification["sessions"]:
        if "derived_file" not in session:
            continue
        path = E73_DIR / session["derived_file"]
        record = {
            "path": session["derived_file"],
            "kind": "local_preprocessed_derivative_not_for_redistribution",
            "status": "verified",
            "subject": session["subject"],
            "session": session["session"],
            "n_trials": session["n_trials"],
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "source_sha256": session["hash_check"]["sha256"],
            "task_label_source": session["label_validation"]["source"],
            "license": manifest["license"],
        }
        if session["session"] == "E":
            record["public_e_labels_sha256"] = session["label_validation"]["public_hash_check"]["sha256"]
        manifest["files"][session["derived_file"]] = record
    manifest["last_verification"] = {
        "verified_at": verification["verified_at"],
        "status": verification["status"],
        "verification_file": VERIFICATION_PATH.name,
        "totals": verification["totals"],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", default=ALL_SUBJECTS, help="Comma-separated IDs; default: 1..9")
    parser.add_argument("--sessions", default="T,E")
    parser.add_argument("--write-derived", action="store_true")
    parser.add_argument("--offset-seconds", type=float, default=2.0)
    parser.add_argument("--duration-seconds", type=float, default=4.0)
    args = parser.parse_args()
    subjects = parse_subjects(args.subjects)
    if args.duration_seconds != 4.0 or args.offset_seconds != 2.0:
        raise ValueError("The frozen E73 preprocessing check uses the MOABB 2--6 s (4 s) window")
    sessions = [item.strip().upper() for item in args.sessions.split(",") if item.strip()]
    if not sessions or any(session not in {"T", "E"} for session in sessions):
        raise ValueError("Sessions must contain only T and/or E")
    if len(set(sessions)) != len(sessions):
        raise ValueError("Sessions must not contain duplicates")

    verification = {
        "schema_version": 2,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "BNCI2014_001 (BCIC IV-2a)",
        "status": "running",
        "subjects": subjects,
        "requested_sessions": sessions,
        "access_policy": {
            "reads_limited_to": str(RAW_DIR),
            "competition_test_accessed": False,
            "competition_train_accessed": False,
            "model_training_started": False,
        },
        "preprocessing": {
            "source_sfreq_hz": 250,
            "target_sfreq_hz": 300,
            "resampling": "scipy.signal.resample_poly(up=6, down=5), 1000 -> 1200 samples",
            "filter": "4th-order Butterworth 1--40 Hz SOS, zero-phase sosfiltfilt on each continuous run",
            "trial_window_seconds_relative_to_source_trial": [2.0, 6.0],
            "window_samples_at_target": N_TARGET_SAMPLES,
            "artifact_policy": "retain every trial and store artifact boolean; no silent trial rejection",
            "evaluation_label_policy": (
                "Use public true_labels/AxxE.mat classlabel in valid-run/trial order; compare "
                "with embedded AxxE run.y whenever the embedded labels are valid."
            ),
            "channel_policy": target_mapping(),
        },
        "expected_per_session": {
            "valid_runs": N_VALID_RUNS,
            "trials_per_run": N_TRIALS_PER_RUN,
            "trials": N_TRIALS_PER_SESSION,
            "trials_per_class": 72,
        },
        "npz_fields": None,
        "sessions": [],
    }
    try:
        for subject in subjects:
            for session in sessions:
                summary, arrays = process_session(
                    subject,
                    session,
                    offset_seconds=args.offset_seconds,
                    duration_seconds=args.duration_seconds,
                )
                fields = {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in arrays.items()
                }
                if verification["npz_fields"] is None:
                    verification["npz_fields"] = fields
                elif fields != verification["npz_fields"]:
                    raise ValueError("NPZ schema differs between sessions")
                if args.write_derived:
                    output_dir = DERIVED_DIR / f"subject_{subject:02d}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"A{subject:02d}{session}_300Hz_1-40Hz_4s.npz"
                    np.savez_compressed(output_path, **arrays)
                    with np.load(output_path, allow_pickle=False) as check:
                        loaded = {key: check[key] for key in check.files}
                    if set(loaded) != set(arrays):
                        raise ValueError("Derived NPZ field round-trip mismatch")
                    validate_arrays(loaded, subject, session)
                    for key in arrays:
                        if not np.array_equal(loaded[key], arrays[key]):
                            raise ValueError(f"Derived NPZ round-trip mismatch for {key}")
                    summary["derived_file"] = output_path.relative_to(E73_DIR).as_posix()
                    summary["derived_bytes"] = output_path.stat().st_size
                    summary["derived_sha256"] = sha256_file(output_path)
                    summary["derived_round_trip_verified"] = True
                verification["sessions"].append(summary)
                print(json.dumps(summary, ensure_ascii=False), flush=True)
        expected_session_count = len(subjects) * len(sessions)
        if len(verification["sessions"]) != expected_session_count:
            raise ValueError("Not every requested subject/session was verified")
        session_counts = {
            session: sum(item["n_trials"] for item in verification["sessions"] if item["session"] == session)
            for session in sessions
        }
        subject_counts = {
            str(subject): sum(
                item["n_trials"] for item in verification["sessions"] if item["subject"] == subject
            )
            for subject in subjects
        }
        verification["totals"] = {
            "subjects": len(subjects),
            "sessions": len(verification["sessions"]),
            "trials": sum(item["n_trials"] for item in verification["sessions"]),
            "trials_by_session": session_counts,
            "trials_by_subject": subject_counts,
            "artifact_trials_retained": sum(item["n_artifact_trials"] for item in verification["sessions"]),
            "derived_npz_files": sum("derived_file" in item for item in verification["sessions"]),
        }
        verification["status"] = "success"
        verification["auc"] = "N/A (data preparation only; no model trained)"
        verification["f1"] = "N/A (data preparation only; no model trained)"
    except Exception as exc:
        verification["status"] = "failed"
        verification["error_type"] = type(exc).__name__
        verification["error"] = str(exc)
        VERIFICATION_PATH.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    VERIFICATION_PATH.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_manifest(verification)
    print(f"Verification: {VERIFICATION_PATH}")


if __name__ == "__main__":
    main()
