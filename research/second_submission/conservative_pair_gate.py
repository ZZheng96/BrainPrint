"""生成第二次提交的保守配对门控候选；不修改正式提交。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "BrainPrint_cache"
OUT = ROOT / "research" / "second_submission"
MAX_CHANGE_RATIO = 0.05
MIN_COMPONENT_SIZE = 2
MIN_DIRECT_PURITY = 0.75


def main() -> None:
    template = pd.read_csv(ROOT / "Brainprint_Competition" / "test.csv")
    first = pd.read_csv(ROOT / "commit" / "answer" / "赞美诗_第1次.csv")
    baseline = pd.read_csv(ROOT / "research" / "submission_second_submission_baseline.csv")
    meta = pd.read_csv(CACHE / "test_metadata.csv")
    direct_scores = np.load(CACHE / "scores_task_specific_zscore.npy")
    grouped_scores = np.load(CACHE / "scores_task_specific_zscore_grouped_guarded.npy")
    pairs = pd.read_csv(CACHE / "waveform_pairs.csv")
    sample_to_row = dict(zip(meta["sample"], range(len(meta))))
    direct = direct_scores.argmax(axis=1) + 1
    grouped = grouped_scores.argmax(axis=1) + 1
    margin = np.diff(np.partition(direct_scores, -2, axis=1)[:, -2:], axis=1)[:, 0]

    valid = pairs[pairs["similarity"] > 0.99].copy()
    left = valid["sample"].map(sample_to_row).to_numpy()
    right = valid["nearest_sample"].map(sample_to_row).to_numpy()
    graph = coo_matrix(
        (np.ones(len(left)), (left, right)), shape=(len(meta), len(meta))
    )
    _, components = connected_components(graph, directed=False)
    component_sizes = np.bincount(components)
    component_direct_purity = np.ones(len(meta), dtype=float)
    component_support = np.ones(len(meta), dtype=int)
    for component in np.flatnonzero(component_sizes >= MIN_COMPONENT_SIZE):
        rows = np.flatnonzero(components == component)
        counts = np.bincount(direct[rows], minlength=21)[1:]
        component_direct_purity[rows] = counts.max() / len(rows)
        component_support[rows] = len(rows)

    # Start from the already validated first-submission-equivalent snapshot.
    candidate = baseline["Label"].to_numpy(dtype=int).copy()
    disagreement = direct != grouped
    session2 = meta["session"].to_numpy() == "session2"
    # Keep only low-margin, sufficiently supported components; revert weak pair effects.
    allow_pair = (
        disagreement
        & session2
        & (margin < 0.10)
        & (component_support >= MIN_COMPONENT_SIZE)
        & (component_direct_purity >= MIN_DIRECT_PURITY)
    )
    candidate[disagreement & session2 & ~allow_pair] = direct[disagreement & session2 & ~allow_pair]
    candidate[~session2] = 0

    output = template.copy()
    output["Label"] = candidate[np.asarray(template["Sample"].map(sample_to_row), dtype=int)]
    output_path = OUT / "candidate_conservative_pair_gate.csv"
    output.to_csv(output_path, index=False)

    changed = output["Label"].to_numpy() != first["Label"].to_numpy()
    session1_changed = changed & (meta["session"].to_numpy() == "session1")
    diagnostics = {
        "candidate": str(output_path),
        "base_official_score": 69.17,
        "test_ground_truth_available": False,
        "test_auc": None,
        "test_f1": None,
        "reference_internal_unknown_auc": 0.9803,
        "reference_internal_unknown_f1": 0.9472,
        "pair_similarity_threshold": 0.99,
        "min_component_size": MIN_COMPONENT_SIZE,
        "min_direct_purity": MIN_DIRECT_PURITY,
        "margin_threshold": 0.10,
        "changed_rows_vs_first": int(changed.sum()),
        "change_ratio_vs_first": float(changed.mean()),
        "session1_changed_rows": int(session1_changed.sum()),
        "allowed_pair_rows": int(allow_pair.sum()),
        "reverted_pair_rows": int((disagreement & session2 & ~allow_pair).sum()),
        "labels_valid": bool(output["Label"].between(0, 20).all()),
        "sample_order_matches": bool(output["Sample"].equals(template["Sample"])),
        "label_counts": {str(k): int(v) for k, v in zip(*np.unique(candidate, return_counts=True))},
        "test_auc_f1_note": "Test 无真实标签；AUC/F1 不可计算。参考值来自内部验证，不是官方 Test 成绩。",
    }
    (OUT / "candidate_conservative_pair_gate.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
