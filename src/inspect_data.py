"""审计 MAT 数据结构与元数据，不修改原始数据。"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from scipy.io import loadmat, whosmat

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Brainprint_Competition"


def scalar_text(value) -> str:
    return str(value.reshape(-1)[0])


def inspect_file(path: Path) -> dict:
    meta = loadmat(
        path,
        variable_names=["fs", "channel_names", "unit", "session_id", "task_label"],
    )
    variables = {name: shape for name, shape, _ in whosmat(path)}
    channels = tuple(str(x.reshape(-1)[0]) for x in meta["channel_names"].reshape(-1))
    return {
        "path": path.relative_to(DATA).as_posix(),
        "session": scalar_text(meta["session_id"]),
        "task": scalar_text(meta["task_label"]),
        "fs": int(meta["fs"].reshape(-1)[0]),
        "unit": scalar_text(meta["unit"]),
        "channels": channels,
        "data_shape": variables["data"],
    }


def summarize(split: str, files: list[Path]) -> dict:
    records = []
    for index, path in enumerate(files, 1):
        records.append(inspect_file(path))
        if index % 1000 == 0 or index == len(files):
            print(f"{split}: {index}/{len(files)}", flush=True)

    summary = {
        "n_files": len(files),
        "fs": dict(Counter(r["fs"] for r in records)),
        "units": dict(Counter(r["unit"] for r in records)),
        "data_shapes": {str(k): v for k, v in Counter(r["data_shape"] for r in records).items()},
        "channel_sets": {str(k): v for k, v in Counter(r["channels"] for r in records).items()},
        "sessions": dict(Counter(r["session"] for r in records)),
        "tasks": dict(Counter(r["task"] for r in records)),
        "session_task": {
            f"{session}|{task}": count
            for (session, task), count in sorted(Counter((r["session"], r["task"]) for r in records).items())
        },
    }
    if split == "train":
        per_subject = defaultdict(Counter)
        missing_trials = {}
        for record in records:
            match = re.match(r"Train/Subject_(\d+)/Task(\d+)/trial_(\d+)\.mat", record["path"])
            if not match:
                raise ValueError(f"Unexpected train path: {record['path']}")
            subject, task, trial = (int(x) for x in match.groups())
            per_subject[subject][(record["session"], task, record["task"])] += 1
        for subject, counts in per_subject.items():
            for (_, task, _), count in counts.items():
                expected = set(range(1, count + 1))
                actual = {
                    int(re.search(r"trial_(\d+)\.mat", r["path"]).group(1))
                    for r in records
                    if int(re.search(r"Subject_(\d+)", r["path"]).group(1)) == subject
                    and int(re.search(r"Task(\d+)", r["path"]).group(1)) == task
                }
                missing = sorted(expected - actual)
                if missing:
                    missing_trials[f"Subject_{subject:02d}/Task{task}"] = missing
        summary["per_subject"] = {
            f"Subject_{subject:02d}": {
                f"{session}|Task{task}|{task_name}": count
                for (session, task, task_name), count in sorted(counts.items())
            }
            for subject, counts in sorted(per_subject.items())
        }
        summary["missing_trials_assuming_contiguous_ranges"] = missing_trials
    else:
        summary["first_path"] = records[0]["path"]
        summary["last_path"] = records[-1]["path"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "data_audit.json")
    args = parser.parse_args()

    train_files = sorted((DATA / "Train").glob("Subject_*/Task*/trial_*.mat"))
    test_files = sorted(
        (DATA / "Test").glob("test_*.mat"),
        key=lambda p: int(re.search(r"test_(\d+)", p.stem).group(1)),
    )
    report = {
        "train": summarize("train", train_files),
        "test": summarize("test", test_files),
    }
    submission = pd.read_csv(DATA / "test.csv")
    expected_samples = [f"test_{i:03d}" for i in range(1, len(test_files) + 1)]
    report["submission"] = {
        "rows": len(submission),
        "columns": list(submission.columns),
        "sample_order_matches_numeric_file_order": submission["Sample"].tolist() == expected_samples,
        "empty_labels": int(submission["Label"].isna().sum()),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "train"}, ensure_ascii=False, indent=2))
    print(f"完整审计结果: {args.output}")


if __name__ == "__main__":
    main()
