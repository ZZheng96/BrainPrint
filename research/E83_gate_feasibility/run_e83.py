"""E83: audit gate feasibility from frozen E76/E79/E82 outer summaries only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
E83_DIR = Path(__file__).resolve().parent
DIRECTIONS = ("T_to_E", "E_to_T")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def points_for(candidate: str, record: dict, direction: str) -> list[dict]:
    outer = record["directions"][direction]["outer"]
    points = []
    if candidate == "E76":
        points.append({"label": "reported_threshold", "threshold": outer.get("threshold"),
                       "tpr": outer["known_acceptance"], "fpr": 1.0 - outer["U"],
                       "auroc": outer.get("AUROC")})
    elif candidate == "E79":
        points.append({"label": "q05", "threshold": outer.get("threshold"),
                       "tpr": outer["known_acceptance"], "fpr": 1.0 - outer["U"],
                       "auroc": outer.get("AUROC")})
        if "working_98_threshold" in outer:
            points.append({"label": "q02", "threshold": outer["working_98_threshold"],
                           "tpr": outer["working_98_known_acceptance"],
                           "fpr": 1.0 - outer["working_98_unknown_rejection"],
                           "auroc": outer.get("AUROC")})
    elif candidate == "E82":
        for label, value in sorted(outer.items()):
            if label in {"0.01", "0.02"}:
                points.append({"label": f"target_fpir_{label}", "threshold": None,
                               "tpr": value["known_acceptance"],
                               "fpr": value["unknown_false_acceptance"],
                               "auroc": value.get("AUROC")})
    return points


def summarize(points: list[dict], tpr_min: float, fpr_max: float) -> dict:
    feasible = [p for p in points if p["tpr"] >= tpr_min and p["fpr"] <= fpr_max]
    fpr_limited = [p for p in points if p["fpr"] <= fpr_max]
    tpr_limited = [p for p in points if p["tpr"] >= tpr_min]
    max_tpr = max(p["tpr"] for p in points)
    min_fpr = min(p["fpr"] for p in points)
    return {
        "simultaneous_gate_feasible": bool(feasible),
        "max_reachable_known_acceptance": max_tpr,
        "fpr_at_max_known_acceptance": min(p["fpr"] for p in points if p["tpr"] == max_tpr),
        "min_reachable_unknown_false_acceptance": min_fpr,
        "known_acceptance_at_min_unknown_false_acceptance": max(p["tpr"] for p in points if p["fpr"] == min_fpr),
        "TPR_at_FPR_2pct": max((p["tpr"] for p in fpr_limited), default=None),
        "FPR_at_TPR_95pct": min((p["fpr"] for p in tpr_limited), default=None),
        "n_frontier_points": len(points),
        "frontier": points,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config = load_json(E83_DIR / args.config)
    candidates = {}
    for candidate, relative in config["source_results"].items():
        source = load_json((E83_DIR / relative).resolve())
        rows = []
        for record in source["runs"]:
            for direction in DIRECTIONS:
                points = points_for(candidate, record, direction)
                rows.append({"outer_fold": record["outer_fold"], "direction": direction,
                             **summarize(points, config["known_acceptance_min"], config["unknown_false_acceptance_max"])})
        macro = {}
        numeric_keys = ("max_reachable_known_acceptance", "fpr_at_max_known_acceptance",
                        "min_reachable_unknown_false_acceptance", "known_acceptance_at_min_unknown_false_acceptance",
                        "TPR_at_FPR_2pct", "FPR_at_TPR_95pct")
        for key in numeric_keys:
            values = [row[key] for row in rows if row[key] is not None]
            macro[key] = mean(values) if values else None
        macro["simultaneous_gate_feasible_rate"] = mean(float(row["simultaneous_gate_feasible"]) for row in rows)
        macro["n_fold_direction_cells"] = len(rows)
        candidates[candidate] = {"outer_fold_direction": rows, "macro_average": macro,
                                 "source_test": source.get("test", {"AUC": "N/A", "F1": "N/A"})}
    result = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "scope": config["scope"],
        "thresholds": {"known_acceptance_min": config["known_acceptance_min"],
                       "unknown_false_acceptance_max": config["unknown_false_acceptance_max"]},
        "candidates": candidates,
        "test": {"AUC": "N/A", "F1": "N/A", "accessed": False},
        "competition_test_accessed": False,
        "interpretation": "Frontier uses only operating points already present in the three frozen outer results.json files; it does not choose a model or alter a threshold.",
    }
    output = E83_DIR / "outputs"
    output.mkdir(exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# E83 gate feasibility 审计", "", "## 范围", "",
             "仅读取 E76/E79/E82 的 `outputs/results.json` 外层结果；不访问竞争 Test，不训练模型，不选择模型或修改阈值。",
             "Frontier 是三份结果中已落盘 operating points 的审计，不是新的阈值搜索。由于输入 results.json 未保存逐样本 score，无法从这些文件重建连续 ROC；因此 TPR@FPR=2% 与 FPR@TPR=95% 按已落盘 operating points 的可达值报告，AUROC 沿用来源结果。", "",
             "## 门禁定义", "", "- known acceptance/TPR ≥ 0.95", "- unknown false acceptance/FPR ≤ 0.02", "",
             "## 结果", "", "| 候选 | 宏平均最大 TPR | TPR@FPR≤2% | FPR@TPR≥95% | 同时满足比例 |", "|---|---:|---:|---:|---:|"]
    for candidate, value in candidates.items():
        macro = value["macro_average"]
        fmt = lambda x: "N/A" if x is None else f"{x:.6f}"
        lines.append(f"| {candidate} | {fmt(macro['max_reachable_known_acceptance'])} | {fmt(macro['TPR_at_FPR_2pct'])} | {fmt(macro['FPR_at_TPR_95pct'])} | {macro['simultaneous_gate_feasible_rate']:.3f} |")
    lines += ["", "## 逐折/方向审计", "", "每个单元报告：最大可达 known acceptance、达到该值时 FPR、最小可达 FPR、达到该值时 TPR、TPR@FPR=2%、FPR@TPR=95%，以及是否存在同一已报告 operating point 同时满足两侧约束。", ""]
    for candidate, value in candidates.items():
        lines += [f"### {candidate}", "", "| fold | direction | max TPR | FPR@max TPR | min FPR | TPR@min FPR | TPR@FPR2% | FPR@TPR95% | simultaneous |", "|---:|---|---:|---:|---:|---:|---:|---:|---|"]
        for row in value["outer_fold_direction"]:
            fmt = lambda x: "N/A" if x is None else f"{x:.6f}"
            lines.append(f"| {row['outer_fold']} | {row['direction']} | {fmt(row['max_reachable_known_acceptance'])} | {fmt(row['fpr_at_max_known_acceptance'])} | {fmt(row['min_reachable_unknown_false_acceptance'])} | {fmt(row['known_acceptance_at_min_unknown_false_acceptance'])} | {fmt(row['TPR_at_FPR_2pct'])} | {fmt(row['FPR_at_TPR_95pct'])} | {'是' if row['simultaneous_gate_feasible'] else '否'} |")
        lines.append("")
    lines += ["## 结论", "", "本审计只回答当前外层数据分布在已生成 operating points 上是否可分；不构成模型选择或阈值改进。", "", "Test AUC/F1=N/A（未访问竞争 Test）。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
