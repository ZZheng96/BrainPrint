"""在 raw-PGD 当前冻结协议下对齐验收两次提交方案。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.open_set_improvement.core import (  # noqa: E402
    COV_COLUMNS,
    ROOT as CORE_ROOT,
    TaskSpecificModel,
    confidence_values,
    resolve_cache,
)
from src.raw_eeg_defense.run import (  # noqa: E402
    ATTACKS,
    competition_metrics,
    load_raw_and_fingerprints,
    predict_subset,
    open_set_predictions,
)
from src.raw_eeg_robust.run import (  # noqa: E402
    acceptance as raw_acceptance,
    cached_adversarial_features,
)

OUT = ROOT / "research" / "second_submission"
CACHE_VERSION = "adversarial-ridge-stratified-random-target-v2"
SIMILARITY = 0.99
MIN_COMPONENT_SIZE = 2
MIN_DIRECT_PURITY = 0.75
MARGIN_THRESHOLD = 0.10


def folds(metadata: pd.DataFrame) -> list[np.ndarray]:
    subjects = np.unique(metadata["subject"].to_numpy())
    groups = np.array_split(np.random.default_rng(42).permutation(subjects), 5)
    return [np.asarray(x) for x in groups]


def pair_components(indices: np.ndarray, features: np.ndarray, tasks: np.ndarray) -> list[np.ndarray]:
    components = []
    for task in sorted(np.unique(tasks[indices])):
        local = indices[tasks[indices] == task]
        if len(local) < MIN_COMPONENT_SIZE:
            continue
        nn = NearestNeighbors(n_neighbors=2, metric="cosine", n_jobs=-1).fit(features[local])
        distance, neighbor = nn.kneighbors(features[local])
        edges = [[] for _ in local]
        for left, right, dist in zip(range(len(local)), neighbor[:, 1], distance[:, 1]):
            if 1.0 - dist > SIMILARITY:
                edges[left].append(int(right))
                edges[right].append(int(left))
        seen = set()
        for start in range(len(local)):
            if start in seen or not edges[start]:
                continue
            stack, group = [start], []
            seen.add(start)
            while stack:
                node = stack.pop()
                group.append(node)
                for nxt in edges[node]:
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            if len(group) >= MIN_COMPONENT_SIZE:
                components.append(local[np.asarray(group, dtype=int)])
    return components


def pair_gate(
    indices: np.ndarray,
    scores: np.ndarray,
    features: np.ndarray,
    tasks: np.ndarray,
) -> tuple[np.ndarray, dict]:
    direct = scores.argmax(axis=1)
    gated_scores = scores.copy()
    components = pair_components(indices, features, tasks)
    allowed = changed = 0
    for component in components:
        labels = direct[component]
        counts = np.bincount(labels, minlength=scores.shape[1])
        purity = float(counts.max() / len(component))
        margins = np.diff(np.partition(scores[component], -2, axis=1)[:, -2:], axis=1)[:, 0]
        if purity < MIN_DIRECT_PURITY:
            continue
        fused = scores[component].mean(axis=0)
        for row, margin in zip(component, margins):
            if margin < MARGIN_THRESHOLD and direct[row] != fused.argmax():
                gated_scores[row] = fused
                changed += 1
                allowed += 1
    return gated_scores, {
        "components": len(components),
        "component_rows": int(sum(len(x) for x in components)),
        "allowed_pair_rows": int(allowed),
        "changed_rows": int(changed),
        "similarity_threshold": SIMILARITY,
        "min_component_size": MIN_COMPONENT_SIZE,
        "min_direct_purity": MIN_DIRECT_PURITY,
        "margin_threshold": MARGIN_THRESHOLD,
    }


def report(true: np.ndarray, clean: np.ndarray, adv: np.ndarray, unknown: np.ndarray,
           classes: np.ndarray, point: dict, pair_features: np.ndarray, tasks: np.ndarray,
           indices: np.ndarray) -> tuple[dict, dict]:
    clean_gate, clean_diag = pair_gate(indices, clean, pair_features, tasks)
    adv_gate, adv_diag = pair_gate(indices, adv, pair_features, tasks)
    baseline = report_scores(true, clean, adv, unknown, classes, point)
    candidate = report_scores(true, clean_gate, adv_gate, unknown, classes, point)
    return baseline, {"clean": clean_diag, "adversarial": adv_diag, "candidate": candidate}


def report_scores(true: np.ndarray, clean: np.ndarray, adv: np.ndarray, unknown: np.ndarray,
                  classes: np.ndarray, point: dict) -> dict:
    clean_pred, clean_conf = open_set_predictions(clean, classes, point)
    adv_pred, _ = open_set_predictions(adv, classes, point)
    unknown_pred, unknown_conf = open_set_predictions(unknown, classes, point)
    return competition_metrics(true, clean_pred, adv_pred, unknown_pred, clean_conf, unknown_conf)


def aggregate(rows: list[dict]) -> dict:
    known = np.asarray([x["n_known"] for x in rows])
    unknown = np.asarray([x["n_unknown"] for x in rows])
    out = {}
    for name in ("C", "RA", "ASafeR", "clean_macro_f1_with_reject", "adversarial_macro_f1_with_reject"):
        out[name] = float(np.average([x[name] for x in rows], weights=known))
    for name in ("U", "unknown_f1"):
        out[name] = float(np.average([x[name] for x in rows], weights=unknown))
    out["A"] = 0.7 * out["RA"] + 0.3 * out["ASafeR"]
    out["official_score"] = float(100 * out["C"] ** 0.40 * out["A"] ** 0.35 * out["U"] ** 0.25)
    out["unknown_auroc"] = float(np.average([x["unknown_auroc"] for x in rows], weights=known + unknown))
    out["n_known"] = int(known.sum())
    out["n_unknown"] = int(unknown.sum())
    return out


def aligned_acceptance(result: dict) -> dict:
    attacks = result["attacks"]
    primary = attacks["untargeted_snr20"]
    base, candidate = primary["first_submission_equivalent"], primary["second_pair_gate"]
    fold_scores = np.asarray([
        f["attacks"]["untargeted_snr20"]["second_pair_gate"]["official_score"]
        - f["attacks"]["untargeted_snr20"]["first_submission_equivalent"]["official_score"]
        for f in result["folds"]
    ])
    checks = {
        "score_gain_at_least_2": candidate["official_score"] - base["official_score"] >= 2,
        "A_gain_at_least_0.05": candidate["A"] - base["A"] >= 0.05,
        "RA_at_least_0.60_and_gain_0.03": candidate["RA"] >= 0.60 and candidate["RA"] - base["RA"] >= 0.03,
        "ASafeR_at_least_0.90": candidate["ASafeR"] >= 0.90,
        "C_at_least_0.90_and_drop_at_most_0.02": candidate["C"] >= 0.90 and candidate["C"] - base["C"] >= -0.02,
        "U_at_least_0.85_and_drop_at_most_0.02": candidate["U"] >= 0.85 and candidate["U"] - base["U"] >= -0.02,
        "unknown_f1_drop_at_most_0.01": candidate["unknown_f1"] - base["unknown_f1"] >= -0.01,
        "AUROC_drop_at_most_0.005": candidate["unknown_auroc"] - base["unknown_auroc"] >= -0.005,
        "four_of_five_folds_not_worse": int(np.sum(fold_scores >= 0)) >= 4,
        "worst_fold_score_drop_at_most_1": float(fold_scores.min()) >= -1,
    }
    for attack_name in ("untargeted_snr30", "targeted_snr20"):
        row, ref = attacks[attack_name]["second_pair_gate"], attacks[attack_name]["first_submission_equivalent"]
        checks[f"{attack_name}_ASafeR_at_least_0.90"] = row["ASafeR"] >= 0.90
        checks[f"{attack_name}_A_drop_at_most_0.01"] = row["A"] - ref["A"] >= -0.01
    return {"candidate": "second_pair_gate", "passed": bool(all(checks.values())), "checks": checks,
            "score_delta": candidate["official_score"] - base["official_score"],
            "fold_score_deltas": fold_scores.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("本实验要求 CUDA GPU")
    cache = resolve_cache()
    metadata = pd.read_csv(cache / "train_metadata.csv")
    x = np.load(cache / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS]
    window = np.load(cache / "train_window2_features.npy", mmap_mode="r").reshape(len(metadata), 2, -1)[:, :, COV_COLUMNS]
    subjects, tasks = metadata["subject"].to_numpy(), metadata["task"].to_numpy()
    fold_groups = folds(metadata)
    attack_cache = cache / "raw_eeg_robust" / "full"
    result = {"protocol": {"outer_folds": "same five subject holdout as raw_eeg_robust/run.py", "model": "TaskSpecificModel; clean first_submission_equivalent scores are refit per fold", "attacks": [{"name": n, "snr_db": s, "targeted": t, "steps": 10} for n, s, t in ATTACKS], "pair_gate": "same-task cosine NN graph rebuilt per fold/evaluation subset; no test labels, attack labels, or Test waveform_pairs", "attack_cache": {"directory": str(attack_cache), "cache_version": CACHE_VERSION, "same_attack_samples": True}}, "folds": [], "attacks": {}}
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        device = torch.device("cuda")
        for fold_no, unknown_subjects in enumerate(fold_groups, 1):
            unknown = np.isin(subjects, unknown_subjects)
            known_subjects = np.setdiff1d(np.unique(subjects), unknown_subjects)
            forward = metadata["trial"].to_numpy() > 0.8 * metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
            fit_mask = np.isin(subjects, known_subjects) & ~forward
            eval_known = np.flatnonzero(np.isin(subjects, known_subjects) & forward)
            eval_unknown = np.flatnonzero(unknown)
            model = TaskSpecificModel(device).fit(x, window, subjects, tasks, fit_mask)
            clean_known = predict_subset(model, x, window, tasks, eval_known)
            clean_unknown = predict_subset(model, x, window, tasks, eval_unknown)
            point = json.loads((ROOT / "research" / "raw_eeg_robust" / "results.json").read_text(encoding="utf-8"))["folds"][fold_no - 1]["operating_points"]["direct"]
            raw, _ = load_raw_and_fingerprints(metadata, eval_known, keep_raw=True)
            fold = {"outer_fold": fold_no, "unknown_subjects": unknown_subjects.tolist(), "attacks": {}}
            for attack_index, (attack_name, snr, targeted) in enumerate(ATTACKS):
                path = attack_cache / f"outer{fold_no}_eval_{attack_name}.npz"
                with np.load(path, allow_pickle=False) as saved:
                    if not np.array_equal(saved["source_indices"], eval_known):
                        raise RuntimeError(f"fold {fold_no} attack cache source indices mismatch")
                    adv_full, adv_window = saved["full"], saved["window"]
                adv_known = model.predict_scores(adv_full, adv_window, tasks[eval_known])["fused"]
                base, details = report(subjects[eval_known], clean_known, adv_known, clean_unknown, model.classes, point, x[eval_known], tasks[eval_known], np.arange(len(eval_known)))
                gated_clean, gate_clean_diag = pair_gate(np.arange(len(eval_known)), clean_known, x[eval_known], tasks[eval_known])
                gated_adv, gate_adv_diag = pair_gate(np.arange(len(eval_known)), adv_known, x[eval_known], tasks[eval_known])
                candidate = report_scores(subjects[eval_known], gated_clean, gated_adv, clean_unknown, model.classes, point)
                fold["attacks"][attack_name] = {"first_submission_equivalent": base, "second_pair_gate": candidate, "pair_gate": {"clean": gate_clean_diag, "adversarial": gate_adv_diag}, "attack_cache": str(path)}
            result["folds"].append(fold)
            del model, raw
            torch.cuda.empty_cache()
    for attack_name, _, _ in ATTACKS:
        result["attacks"][attack_name] = {method: aggregate([f["attacks"][attack_name][method] for f in result["folds"]]) for method in ("first_submission_equivalent", "second_pair_gate")}
    result["acceptance"] = aligned_acceptance(result)
    output = args.output or OUT / "aligned_evaluation.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 前两次提交方案对齐 raw-PGD 冻结验收", "", "同一五折 subject holdout、同一 TaskSpecificModel、同一攻击缓存；未使用 Test 标签或 Test waveform_pairs。", "", "|攻击|方案|C|RA|ASafeR|U|官方分|unknown AUC|unknown F1|", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for attack_name, _, _ in ATTACKS:
        for method in ("first_submission_equivalent", "second_pair_gate"):
            row = result["attacks"][attack_name][method]
            lines.append(f"|{attack_name}|{method}|{row['C']:.6f}|{row['RA']:.6f}|{row['ASafeR']:.6f}|{row['U']:.6f}|{row['official_score']:.6f}|{row['unknown_auroc']:.6f}|{row['unknown_f1']:.6f}|")
    lines += ["", f"验收：{'通过' if result['acceptance']['passed'] else '未通过'}", "", "逐门禁结果："]
    lines.extend(f"- {key}: {'通过' if value else '未通过'}" for key, value in result["acceptance"]["checks"].items())
    lines += ["", f"JSON：`{output}`", f"攻击缓存完全同一：`{result['protocol']['attack_cache']['same_attack_samples']}`"]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "acceptance": result["acceptance"], "attacks": result["attacks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
