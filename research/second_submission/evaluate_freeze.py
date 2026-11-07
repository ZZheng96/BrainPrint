"""冻结验收：比较第一次任务特异预测与训练内保守 pair gate。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "BrainPrint_cache"
SIMILARITY = 0.99
MIN_COMPONENT_SIZE = 2
MIN_DIRECT_PURITY = 0.75
MARGIN_THRESHOLD = 0.10


def folds_for(meta: pd.DataFrame) -> np.ndarray:
    return ((meta["trial"].to_numpy() - 1) // 10) % 5


def pair_components(indices: np.ndarray, meta: pd.DataFrame, features: np.ndarray) -> list[np.ndarray]:
    """只在当前训练评估子集内、同 task 建立高相似图。"""
    tasks = meta["task"].to_numpy()
    components: list[np.ndarray] = []
    for task in sorted(np.unique(tasks[indices])):
        local = indices[tasks[indices] == task]
        if len(local) < 2:
            continue
        nn = NearestNeighbors(n_neighbors=2, metric="cosine", n_jobs=-1).fit(features[local])
        distances, neighbors = nn.kneighbors(features[local])
        left = np.arange(len(local))
        right = neighbors[:, 1]
        edges = [[] for _ in local]
        for a, b, distance in zip(left, right, distances[:, 1]):
            if 1.0 - distance > SIMILARITY:
                edges[a].append(int(b))
                edges[b].append(int(a))
        seen = set()
        for start in range(len(local)):
            if start in seen or not edges[start]:
                continue
            stack = [start]
            seen.add(start)
            group = []
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


def pair_gate(indices: np.ndarray, direct_scores: np.ndarray, meta: pd.DataFrame, features: np.ndarray) -> tuple[np.ndarray, dict]:
    direct = direct_scores.argmax(axis=1) + 1
    gated = direct.copy()
    components = pair_components(indices, meta, features)
    allowed = 0
    changed = 0
    for component in components:
        local_scores = direct_scores[component]
        fused = local_scores.mean(axis=0)
        fused_label = int(fused.argmax() + 1)
        component_direct = direct[component]
        purity = float(np.mean(component_direct == np.bincount(component_direct, minlength=21).argmax()))
        margins = np.partition(local_scores, -2, axis=1)[:, -1] - np.partition(local_scores, -2, axis=1)[:, -2]
        allow = (purity >= MIN_DIRECT_PURITY) & (margins < MARGIN_THRESHOLD)
        # 与第二次方案一致：只保留低 margin 且组件纯度足够的 pair 影响。
        for row, low_margin in zip(component, allow):
            if low_margin and direct[row] != fused_label:
                gated[row] = fused_label
                changed += 1
                allowed += 1
    diag = {
        "components": len(components),
        "component_rows": int(sum(len(x) for x in components)),
        "allowed_pair_rows": int(allowed),
        "changed_rows": int(changed),
        "similarity_threshold": SIMILARITY,
        "min_component_size": MIN_COMPONENT_SIZE,
        "min_direct_purity": MIN_DIRECT_PURITY,
        "margin_threshold": MARGIN_THRESHOLD,
    }
    return gated, diag


def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    return {
        "n": int(len(true)),
        "C_accuracy": float(np.mean(pred == true)),
        "F1_macro_known": float(f1_score(true, pred, labels=np.arange(1, 21), average="macro", zero_division=0)),
        "U": "N/A（未构造未知样本/拒识阈值）",
        "unknown_auc": "N/A（无内部未知标签）",
        "unknown_f1": "N/A（无内部未知标签）",
        "RA": "N/A（无 raw EEG attack 数据）",
        "ASafeR": "N/A（无 raw EEG attack 数据）",
    }


def evaluate(name: str, selected: np.ndarray, score_path: str, meta: pd.DataFrame, features: np.ndarray, foldwise: bool) -> dict:
    scores = np.asarray(np.load(CACHE / score_path, mmap_mode="r"))
    truth = meta["subject"].to_numpy()
    folds = folds_for(meta)
    blocks = range(5) if foldwise else [None]
    reports = []
    for block in blocks:
        block_selected = selected & ((folds == block) if block is not None else True)
        idx = np.flatnonzero(block_selected)
        direct_scores = scores[idx]
        direct = direct_scores.argmax(axis=1) + 1
        gated, gate_diag = pair_gate(idx, scores, meta, features)
        # pair_gate 返回全局索引对应的标签；取当前子集。
        gated_local = gated[idx]
        reports.append({
            "block": int(block) if block is not None else "all",
            "first_submission_equivalent": metrics(direct, truth[idx]),
            "second_pair_gate": metrics(gated_local, truth[idx]),
            "gate_diagnostics": gate_diag,
        })
    all_idx = np.flatnonzero(selected)
    direct = scores[all_idx].argmax(axis=1) + 1
    gated, gate_diag = pair_gate(all_idx, scores, meta, features)
    return {
        "protocol": name,
        "score_file": score_path,
        "pair_gate_training_protocol": "仅使用 BrainPrint_cache/train_features.npy 与 train_metadata.csv；同 task、当前评估子集内 cosine similarity>0.99 的近邻图；blocked OOF 每个 pair 只在同一验证 fold 内构造；不读取 Test 或标签来决定门控，组件 purity 仅由 direct prediction 计算。",
        "overall": {
            "first_submission_equivalent": metrics(direct, truth[all_idx]),
            "second_pair_gate": metrics(gated[all_idx], truth[all_idx]),
            "gate_diagnostics": gate_diag,
        },
        "blocks": reports,
    }


def main() -> None:
    meta = pd.read_csv(CACHE / "train_metadata.csv")
    features = np.load(CACHE / "train_features.npy", mmap_mode="r")
    folds = folds_for(meta)
    all_selected = np.ones(len(meta), dtype=bool)
    max_trial = meta.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    reverse = meta["trial"].to_numpy() <= 0.2 * max_trial
    middle = (meta["trial"].to_numpy() > 0.4 * max_trial) & (meta["trial"].to_numpy() <= 0.6 * max_trial)
    report = {
        "dataset": "BrainPrint train only; Test 标签未使用；正式 CSV 未修改",
        "definitions": "C=identity accuracy；F1=registered identities 1..20 macro-F1。U/unknown AUC/F1 仅在有未知标签时计算；RA/ASafeR 仅在 raw EEG attack 数据可复用时计算。",
        "blocked_oof": evaluate("blocked_oof (5 trial-block folds)", all_selected, "train_blocked_oof_scores_task_specific_guarded.npy", meta, features, True),
        "reverse_holdout": evaluate("reverse (first 20% holdout)", reverse, "train_forward_scores_task_specific_guarded.npy", meta, features, False),
        "middle_holdout": evaluate("middle (40%-60% holdout)", middle, "train_forward_scores_task_specific_guarded.npy", meta, features, False),
    }
    out_json = ROOT / "research" / "second_submission" / "freeze_evaluation.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 冻结验收实验报告", "", "- 仅使用训练 features/metadata 与已有任务特异分数；未使用 Test 标签，未修改正式 CSV。", "- pair gate 协议：同 task、session2、相似度 > 0.99；blocked OOF 在同 fold 内构图；组件 purity 与 margin 只来自 direct prediction。", "", "| 协议 | 方案 | C/accuracy | F1 macro | unknown AUC | unknown F1 |", "|---|---|---:|---:|---:|---:|"]
    for key in ("blocked_oof", "reverse_holdout", "middle_holdout"):
        for method in ("first_submission_equivalent", "second_pair_gate"):
            x = report[key]["overall"][method]
            lines.append(f"|{report[key]['protocol']}|{method}|{x['C_accuracy']:.6f}|{x['F1_macro_known']:.6f}|N/A|N/A|")
    lines += ["", "RA/ASafeR：N/A（本次缓存没有可复用的 raw EEG attack 数据）。", "", f"JSON：`{out_json}`"]
    (out_json.with_suffix(".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
