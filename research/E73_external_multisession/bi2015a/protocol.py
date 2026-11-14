from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
DIRECTIONS = tuple(tuple(pair) for pair in CONFIG["session_directions"])


def subject_folds(config: dict = CONFIG) -> list[dict[str, tuple[int, ...]]]:
    subjects = list(config["subjects"])
    if len(subjects) != 43 or config["outer_folds"] != 3 or config["outer_seed"] != 42:
        raise ValueError("BI2015a requires the fixed 43-subject, 3-fold seed-42 protocol")
    # Fixed rotation, independent of trial order and data availability.
    groups = [subjects[:14], subjects[14:28], subjects[28:42]]
    remainder = subjects[42:]
    folds = []
    for index in range(3):
        unknown = tuple(groups[index])
        calibration = tuple(groups[(index + 1) % 3][:5])
        enrolled = tuple(subject for group in groups for subject in group if subject not in unknown and subject not in calibration) + tuple(remainder)
        if len(enrolled) != 24 or len(calibration) != 5 or len(unknown) != 14:
            raise AssertionError("invalid fixed fold sizes")
        folds.append({"enrolled": enrolled, "calibration_unknown": calibration, "outer_unknown": unknown})
    return folds


def session_directions(config: dict = CONFIG) -> tuple[tuple[int, int], ...]:
    expected = ((1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2))
    actual = tuple(tuple(pair) for pair in config["session_directions"])
    if actual != expected:
        raise ValueError("session directions are not fixed")
    return actual


def training_plan(enrolled: Iterable[int], source_session: int, target_session: int, config: dict = CONFIG) -> list[tuple[int, int, int]]:
    if source_session == target_session or (source_session, target_session) not in session_directions(config):
        raise ValueError("source and target sessions must be a fixed distinct direction")
    if tuple(config["train_runs"]) != (0, 1, 2):
        raise ValueError("training runs must be exactly 0, 1, 2")
    return [(int(subject), source_session, run) for subject in enrolled for run in config["train_runs"]] + [
        (int(subject), target_session, run) for subject in enrolled for run in config["train_runs"]
    ]


def _role_indices(subjects: Iterable[int], session: int, run: int, role: str, manifest: dict | None = None) -> list[tuple[int, int, int, int, str]]:
    if run not in (3, 4, 5):
        raise ValueError("role indices may only use template/calibration/evaluation runs")
    counts = (manifest or {}).get("run_counts", {})
    result = []
    for subject in subjects:
        run_count = int(counts.get(f"{subject}:{session}", {}).get(str(run), 0))
        result.extend((int(subject), session, run, trial, role) for trial in range(run_count))
    return result


def template_indices(subjects: Iterable[int], session: int, manifest: dict | None = None) -> list[tuple[int, int, int, int, str]]:
    return _role_indices(subjects, session, CONFIG["template_run"], "template", manifest)


def calibration_indices(subjects: Iterable[int], session: int, manifest: dict | None = None) -> list[tuple[int, int, int, int, str]]:
    return _role_indices(subjects, session, CONFIG["calibration_known_run"], "calibration_known", manifest)


def evaluation_indices(subjects: Iterable[int], session: int, manifest: dict | None = None) -> list[tuple[int, int, int, int, str]]:
    return _role_indices(subjects, session, CONFIG["evaluation_run"], "evaluation", manifest)
