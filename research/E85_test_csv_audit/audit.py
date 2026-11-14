from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = ROOT / "Brainprint_Competition" / "test.csv"
METADATA_PATH = ROOT / "BrainPrint_cache" / "test_metadata.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
N_LABELS = 21
EXPECTED_ROWS = 22251
NA_TEST_METRICS = {
    name: "N/A（无Test真实标签）"
    for name in ("AUC", "F1", "C", "RA", "ASafeR", "U", "Official_Score")
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_column(frame: pd.DataFrame) -> str:
    matches = [c for c in frame.columns if c.casefold() == "sample"]
    if len(matches) != 1:
        raise ValueError("必须存在且只能存在一个 Sample 字段")
    return matches[0]


def load_test_index() -> tuple[pd.DataFrame, pd.DataFrame]:
    template = pd.read_csv(TEMPLATE_PATH, usecols=["Sample"])
    metadata = pd.read_csv(METADATA_PATH)
    sample_template = _sample_column(template)
    sample_meta = _sample_column(metadata)
    if len(template) != EXPECTED_ROWS or len(metadata) != EXPECTED_ROWS:
        raise ValueError(f"Test 行数必须为 {EXPECTED_ROWS}")
    if template[sample_template].isna().any() or not template[sample_template].is_unique:
        raise ValueError("官方模板 Sample 缺失或不唯一")
    if metadata[sample_meta].isna().any() or not metadata[sample_meta].is_unique:
        raise ValueError("Test metadata Sample 缺失或不唯一")
    if not template[sample_template].astype(str).equals(metadata[sample_meta].astype(str)):
        raise ValueError("官方模板与 metadata 的 Sample 顺序不一致")
    path_col = next((c for c in metadata.columns if c.casefold() == "path"), None)
    if path_col is None or not metadata[path_col].astype(str).str.startswith("Test/").all():
        raise ValueError("Test metadata 含非 Test 路径")
    return template.rename(columns={sample_template: "Sample"}), metadata.rename(columns={sample_meta: "Sample"})


def _validate_labels(labels: Iterable[int], n: int) -> np.ndarray:
    values = np.asarray(list(labels))
    if values.ndim != 1 or len(values) != n or values.size == 0:
        raise ValueError("Label 长度与 Test 行数不一致")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.all(np.isfinite(values)) or not np.all(values == values.astype(np.int64)):
            raise ValueError("Label 必须为整数且非空")
        values = values.astype(np.int64)
    values = values.astype(np.int64, copy=False)
    if not np.isin(values, np.arange(N_LABELS)).all():
        raise ValueError("Label 必须在 0..20")
    return values


def validate_csv(path: Path, expected_samples: pd.Series) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["Sample", "Label"]:
        raise ValueError(f"{path} 字段必须严格为 Sample,Label")
    if len(frame) != len(expected_samples):
        raise ValueError(f"{path} 行数错误")
    if frame["Sample"].isna().any() or not frame["Sample"].is_unique:
        raise ValueError(f"{path} Sample 缺失或不唯一")
    if not frame["Sample"].astype(str).equals(expected_samples.astype(str)):
        raise ValueError(f"{path} Sample 顺序不一致")
    labels = _validate_labels(frame["Label"].to_numpy(), len(frame))
    frame["Label"] = labels
    return frame


def _rate(labels: np.ndarray) -> dict:
    rejected = labels == 0
    return {"n": int(len(labels)), "reject": int(rejected.sum()),
            "rejection_rate": float(rejected.mean()) if len(labels) else None,
            "acceptance_rate": float((~rejected).mean()) if len(labels) else None}


def _group_stats(frame: pd.DataFrame, metadata: pd.DataFrame, group: str) -> dict:
    values = metadata[group].astype(str).to_numpy()
    return {value: _rate(frame["Label"].to_numpy()[values == value])
            for value in sorted(pd.unique(values))}


def audit_csv(path: Path, template: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    frame = validate_csv(path, template["Sample"])
    labels = frame["Label"].to_numpy(dtype=np.int64)
    session_col = next(c for c in metadata.columns if c.casefold() == "session")
    task_col = next(c for c in metadata.columns if c.casefold() == "task")
    sessions = metadata[session_col].astype(str).str.strip().str.casefold()
    session_map = {"1": "S1", "s1": "S1", "session1": "S1",
                   "2": "S2", "s2": "S2", "session2": "S2"}
    codes = sessions.map(session_map)
    if codes.isna().any():
        raise ValueError("发现未知 Session 值")
    s1 = _rate(labels[codes.to_numpy() == "S1"])
    s2 = _rate(labels[codes.to_numpy() == "S2"])
    session_task = {}
    for session in ("S1", "S2"):
        for task in sorted(metadata[task_col].astype(str).unique()):
            selected = (codes.to_numpy() == session) & (metadata[task_col].astype(str).to_numpy() == task)
            session_task[f"{session}/{task}"] = _rate(labels[selected])
    return {
        "csv": str(path.resolve()),
        "csv_sha256": sha256(path),
        "n_rows": int(len(frame)),
        "integrity": {"sample_order_valid": True, "label_range_valid": True},
        "overall": _rate(labels),
        "S1": {**s1, "rejection_gate_ge_0.95": bool(s1["rejection_rate"] >= 0.95)},
        "S2": {**s2, "rejection_gate_le_0.02": bool(s2["rejection_rate"] <= 0.02),
               "acceptance_gate_ge_0.98": bool(s2["acceptance_rate"] >= 0.98)},
        "by_task": _group_stats(frame, metadata, task_col),
        "by_session_task": session_task,
        "label_distribution": {str(i): int((labels == i).sum()) for i in range(N_LABELS)},
        "session_used_by_inference": False,
        "test_labels_used": False,
        "test_metrics": NA_TEST_METRICS,
    }


def compare_csv(a_path: Path, b_path: Path, template: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    a = validate_csv(a_path, template["Sample"])
    b = validate_csv(b_path, template["Sample"])
    pa, pb = a["Label"].to_numpy(int), b["Label"].to_numpy(int)
    same = pa == pb
    transition = np.zeros((N_LABELS, N_LABELS), dtype=int)
    np.add.at(transition, (pa, pb), 1)
    session_col = next(c for c in metadata.columns if c.casefold() == "session")
    task_col = next(c for c in metadata.columns if c.casefold() == "task")
    sess = metadata[session_col].astype(str).to_numpy()
    task = metadata[task_col].astype(str).to_numpy()
    groups = {"overall": np.ones(len(pa), dtype=bool)}
    groups.update({f"session/{x}": sess == x for x in np.unique(sess)})
    groups.update({f"task/{x}": task == x for x in np.unique(task)})
    groups.update({f"session_task/{s}/{t}": (sess == s) & (task == t)
                   for s in np.unique(sess) for t in np.unique(task)})
    grouped = {name: {"n": int(mask.sum()), "agreement_rate": float(same[mask].mean()),
                      "a_rejection_rate": float((pa[mask] == 0).mean()),
                      "b_rejection_rate": float((pb[mask] == 0).mean())}
               for name, mask in groups.items() if mask.any()}
    return {
        "a": str(a_path.resolve()), "b": str(b_path.resolve()), "n": int(len(pa)),
        "agreement_rate": float(same.mean()), "n_diff": int((~same).sum()),
        "a_reject_b_accept": int(((pa == 0) & (pb != 0)).sum()),
        "a_accept_b_reject": int(((pa != 0) & (pb == 0)).sum()),
        "both_accept_different_identity": int(((pa != 0) & (pb != 0) & (pa != pb)).sum()),
        "transition": transition.tolist(),
        "grouped": grouped,
        "test_labels_used": False,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", type=Path, required=True)
    parser.add_argument("--compare", nargs=2, type=Path)
    args = parser.parse_args()
    template, metadata = load_test_index()
    results = {path.stem: audit_csv(path, template, metadata) for path in args.csv}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results.json").write_text(json.dumps({"candidates": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.compare:
        comparison = compare_csv(args.compare[0], args.compare[1], template, metadata)
        (OUTPUT_DIR / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidates": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
