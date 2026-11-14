from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parent
DEFAULT_DERIVED = ROOT / "cache" / "derived"


@dataclass(frozen=True)
class BIData:
    raw: torch.Tensor
    subject: int
    session: int
    run: torch.Tensor
    trial_index: torch.Tensor
    source_file: Path


def _as_int_vector(value, name: str, expected: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (expected,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be integer shape ({expected},), got {array.shape} {array.dtype}")
    return array


def _validate(subject: int, session: int, path: Path, archive) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"raw", "run", "trial_index"}
    if set(archive.files) != required:
        raise ValueError(f"{path}: expected NPZ keys {sorted(required)}, got {archive.files}")
    raw = np.asarray(archive["raw"])
    if raw.ndim != 3 or raw.shape[1:] != (32, 300) or raw.dtype != np.float32:
        raise ValueError(f"{path}: raw must be float32 (n, 32, 300), got {raw.dtype} {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError(f"{path}: raw contains non-finite values")
    n = raw.shape[0]
    run = _as_int_vector(archive["run"], "run", n)
    trial_index = _as_int_vector(archive["trial_index"], "trial_index", n)
    if not np.array_equal(np.unique(run), np.arange(6)) or np.any(np.bincount(run, minlength=6) < 1):
        raise ValueError(f"{path}: invalid six-run partition")
    if not np.array_equal(trial_index, np.arange(n)):
        raise ValueError(f"{path}: trial_index must be 0..{n - 1}")
    return raw, run, trial_index


def load_bi_data(subject: int, session: int, device: str = "cuda", derived_root: Path | str = DEFAULT_DERIVED) -> BIData:
    if subject not in range(1, 44) or session not in (1, 2, 3):
        raise ValueError("subject must be 1..43 and session must be 1..3")
    path = Path(derived_root) / f"subject_{subject:02d}" / f"session_{session}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        raw, run, trial_index = _validate(subject, session, path, archive)
        if torch is None or not hasattr(torch, "as_tensor"):
            raise RuntimeError("PyTorch with tensor support is required to load derived EEG data")
        raw_tensor = torch.as_tensor(raw)
        run_tensor = torch.as_tensor(run)
        trial_tensor = torch.as_tensor(trial_index)
    if device == "cuda" and torch.cuda.is_available():
        raw_tensor = raw_tensor.to("cuda")
        run_tensor = run_tensor.to("cuda")
        trial_tensor = trial_tensor.to("cuda")
    elif device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cuda' or 'cpu'")
    return BIData(raw_tensor, subject, session, run_tensor, trial_tensor, path)


def indices(subjects: Iterable[int], session: int, runs: Iterable[int]) -> list[tuple[int, int, int, int]]:
    subjects = tuple(subjects)
    runs = tuple(runs)
    if any(subject not in range(1, 44) for subject in subjects):
        raise ValueError("subjects must be in 1..43")
    if session not in (1, 2, 3) or any(run not in range(6) for run in runs):
        raise ValueError("session must be 1..3 and runs must be 0..5")
    result = []
    for subject in subjects:
        data = load_bi_data(subject, session, device="cpu")
        for run in runs:
            result.extend((subject, session, run, int(trial)) for trial in torch.where(data.run == run)[0].tolist())
    return result
