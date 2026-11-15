"""E183 single-working-point gate matrix.

Rebuilds the ten gates *at one operating point at a time* from the frozen E183
results (no retraining, no new Test pass, no threshold re-fitting): the adversarial
per-condition metrics were already stored at every a, so only the threshold is
re-applied.  No cross-working-point stitching is performed.

Outputs gate_matrix.json and appends the `gate_matrix` key to results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
RESULTS_PATH = OUT / "results.json"
MATRIX_PATH = OUT / "gate_matrix.json"
ACC = ("0.90", "0.92", "0.95", "0.97")

GATES = {
    "#1": {"name": "Train 准确率（拟合容量）", "target": "= 1.00", "kind": "eq",
           "target_value": 1.0, "source": "capacity.in_sample_logits_argmax"},
    "#2": {"name": "C（5 块 OOF 闭集准确率）", "target": "≥ 0.95", "kind": "ge",
           "target_value": 0.95, "source": "capacity.oof_accuracy_C"},
    "#3": {"name": "身份 macro-F1（5 块 OOF）", "target": "≥ 0.92", "kind": "ge",
           "target_value": 0.92, "source": "capacity.oof_macro_f1"},
    "#4": {"name": "块外准确率（frac>0.6）", "target": "≥ 0.90", "kind": "ge",
           "target_value": 0.90, "source": "capacity.oof_accuracy_forward_blocks"},
    "#5": {"name": "pseudo-unknown AUROC（20 折 LOSO）", "target": "≥ 0.98", "kind": "ge",
           "target_value": 0.98, "source": "fusion.selected.pooled_auc"},
    "#6": {"name": "U@95%（known 接受率 95%）", "target": "≥ 0.95", "kind": "ge",
           "target_value": 0.95, "source": "fusion.selected.u95_*"},
    "#7": {"name": "RA（自建 PGD，ε×目标取最差）", "target": "≥ 0.95", "kind": "ge",
           "target_value": 0.95, "source": "attack.conditions[].at_acceptance"},
    "#8": {"name": "ASafeR（同一攻击取最差）", "target": "≥ 0.98", "kind": "ge",
           "target_value": 0.98, "source": "attack.conditions[].at_acceptance"},
    "#9": {"name": "Test S2 接受率", "target": "≥ 0.75", "kind": "ge",
           "target_value": 0.75, "source": "test.grid[a].S2_acceptance"},
    "#10": {"name": "Test S1 拒识率", "target": "≥ 0.75", "kind": "ge",
            "target_value": 0.75, "source": "test.grid[a].S1_rejection"},
}


def margin(kind: str, target: float, value: float) -> float:
    return value - target


def judge(gate: dict, value: float) -> bool:
    return value >= gate["target_value"] if gate["kind"] == "ge" else \
        abs(value - gate["target_value"]) < 1e-12


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    attack = results["attack"]
    grid = results["test"]["grid"]
    selected = results["fusion"]["selected"]
    capacity = results["capacity"]

    # ---- a-independent gates (Train side)
    fixed = {
        "#1": {"value": capacity["in_sample_logits_argmax"],
               "extra": {"prototype_assisted_technical_max":
                         capacity["in_sample_best_technical"]}},
        "#2": {"value": capacity["oof_accuracy_C"], "extra": {}},
        "#3": {"value": capacity["oof_macro_f1"], "extra": {}},
        "#4": {"value": capacity["oof_accuracy_forward_blocks"], "extra": {}},
        "#5": {"value": selected["pooled_auc"],
               "extra": {"strict_known_pooled_auc": selected["strict_pooled_auc"],
                         "fold_auc_mean": selected["fold_auc_mean"],
                         "fold_auc_std": selected["fold_auc_std"]}},
        "#6": {"value": min(selected["u95_pooled_threshold"], selected["u95_mean"]),
               "extra": {"global_threshold": selected["u95_pooled_threshold"],
                         "per_fold_threshold": selected["u95_mean"],
                         "strict_known_per_fold": selected["strict_u95_mean"],
                         "judged_on": "worst of the two threshold readings"}},
    }

    # ---- per-working-point gates from the stored adversarial / Test metrics
    per_a_attack = {}
    for a in ACC:
        ra_rows = [(row["eps"], row["goal"], row["at_acceptance"][a]["RA"])
                   for row in attack["conditions"]]
        safe_rows = [(row["eps"], row["goal"], row["at_acceptance"][a]["ASafeR"])
                     for row in attack["conditions"]]
        accept_rows = [(row["eps"], row["goal"], row["at_acceptance"][a]["accept_rate"])
                       for row in attack["conditions"]]
        ra = min(ra_rows, key=lambda item: item[2])
        safe = min(safe_rows, key=lambda item: item[2])
        per_a_attack[a] = {
            "worst_RA": {"value": ra[2], "eps": ra[0], "goal": ra[1]},
            "worst_ASafeR": {"value": safe[2], "eps": safe[0], "goal": safe[1]},
            "min_accept_rate": min(row[2] for row in accept_rows),
            "clean_accept_rate": attack["clean"]["at_acceptance"][a]["accept_rate"],
            "clean_RA": attack["clean"]["at_acceptance"][a]["RA"],
            "clean_ASafeR": attack["clean"]["at_acceptance"][a]["ASafeR"],
            "tau": attack["tau_loso"][a],
        }

    matrix = {}
    for key, gate in GATES.items():
        values = {}
        passes = {}
        details = {}
        for a in ACC:
            if key in fixed:
                value = fixed[key]["value"]
                detail = dict(fixed[key]["extra"])
            elif key == "#7":
                value = per_a_attack[a]["worst_RA"]["value"]
                detail = {k: per_a_attack[a]["worst_RA"][k] for k in ("eps", "goal")}
                detail["clean_accept_rate"] = per_a_attack[a]["clean_accept_rate"]
            elif key == "#8":
                value = per_a_attack[a]["worst_ASafeR"]["value"]
                detail = {k: per_a_attack[a]["worst_ASafeR"][k] for k in ("eps", "goal")}
                detail["clean_accept_rate"] = per_a_attack[a]["clean_accept_rate"]
            elif key == "#9":
                value = grid[a]["S2_acceptance"]
                detail = {"S1_rejection_same_point": grid[a]["S1_rejection"]}
            else:
                value = grid[a]["S1_rejection"]
                detail = {"S2_acceptance_same_point": grid[a]["S2_acceptance"]}
            values[a] = float(value)
            passes[a] = bool(judge(gate, float(value)))
            details[a] = detail
        matrix[key] = {"name": gate["name"], "target": gate["target"],
                       "kind": gate["kind"], "target_value": gate["target_value"],
                       "source": gate["source"], "values": values,
                       "margins": {a: margin(gate["kind"], gate["target_value"],
                                             values[a]) for a in ACC},
                       "pass": passes, "details": details,
                       "a_independent": key in fixed,
                       "structural_fail": not any(passes.values())}

    per_point = {}
    for a in ACC:
        passed = [key for key in GATES if matrix[key]["pass"][a]]
        failed = [key for key in GATES if not matrix[key]["pass"][a]]
        per_point[a] = {
            "n_pass": len(passed), "passed": passed, "failed": failed,
            "shortfalls": {key: {"value": matrix[key]["values"][a],
                                 "target": matrix[key]["target_value"],
                                 "gap": -matrix[key]["margins"][a]}
                           for key in failed},
            "test_pair": {"S1_rejection": matrix["#10"]["values"][a],
                          "S2_acceptance": matrix["#9"]["values"][a],
                          "S1_plus_S2": matrix["#10"]["values"][a]
                          + matrix["#9"]["values"][a]},
            "attack_working_point": {
                "tau": per_a_attack[a]["tau"],
                "clean_accept_rate": per_a_attack[a]["clean_accept_rate"],
                "clean_RA": per_a_attack[a]["clean_RA"],
                "clean_ASafeR": per_a_attack[a]["clean_ASafeR"]},
        }

    best = max(per_point.values(), key=lambda row: row["n_pass"])["n_pass"]
    best_points = [a for a in ACC if per_point[a]["n_pass"] == best]
    # tie-break on the Test audit pair (the only axis that actually depends on a
    # besides #7/#8): larger S1+S2 means the gate is closer to satisfying #9 and #10
    tie = sorted(best_points, key=lambda a: -per_point[a]["test_pair"]["S1_plus_S2"])
    overall = {
        "max_pass_single_working_point": best,
        "working_points_reaching_max": best_points,
        "tie_break_rule": "larger S1_rejection + S2_acceptance (the pair #9/#10 needs "
                          "1.50 simultaneously); then smaller #10 shortfall",
        "recommended_single_working_point": tie[0],
        "recommended_evidence": {
            "S1_plus_S2": per_point[tie[0]]["test_pair"]["S1_plus_S2"],
            "S1_plus_S2_by_point": {a: per_point[a]["test_pair"]["S1_plus_S2"]
                                    for a in ACC},
            "gap10_by_point": {a: -matrix["#10"]["margins"][a] for a in ACC},
        },
        "structural_failures": [key for key in GATES if matrix[key]["structural_fail"]],
        "no_stitching": ("every column is one full system at one threshold; the gate "
                         "table of report §4 mixes working points and is superseded by "
                         "this matrix for pass counting"),
    }

    payload = {"experiment": results["experiment"],
               "note": ("single-working-point gate matrix rebuilt from the frozen E183 "
                        "results; adversarial metrics were stored per a, only the "
                        "threshold changes; no retraining, no Test re-scoring"),
               "working_points": list(ACC),
               "gates": matrix,
               "per_working_point": per_point,
               "overall": overall,
               "attack_per_working_point": per_a_attack}
    MATRIX_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                      default=float), encoding="utf-8")
    results["gate_matrix"] = {"path": MATRIX_PATH.name,
                              "max_pass_single_working_point": best,
                              "working_points_reaching_max": best_points,
                              "recommended_single_working_point": tie[0],
                              "structural_failures": overall["structural_failures"],
                              "per_working_point_n_pass": {a: per_point[a]["n_pass"]
                                                           for a in ACC},
                              "per_working_point_failed": {a: per_point[a]["failed"]
                                                           for a in ACC}}
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2,
                                       default=float), encoding="utf-8")

    print(f"{'#':<4}{'target':<10}" + "".join(f"a={a:<9}" for a in ACC) + " verdict")
    for key in GATES:
        row = matrix[key]
        cells = "".join(f"{row['values'][a]:<12.4f}" for a in ACC)
        verdict = "结构性未通过" if row["structural_fail"] else \
            ("全工作点通过" if all(row["pass"].values()) else
             "部分工作点通过")
        print(f"{key:<4}{GATES[key]['target']:<10}{cells}{verdict}")
    for a in ACC:
        row = per_point[a]
        print(f"a={a}: 通过 {row['n_pass']}/10  未通过 {row['failed']}  "
              f"S1+S2 {row['test_pair']['S1_plus_S2']:.4f}  "
              f"clean accept {row['attack_working_point']['clean_accept_rate']:.4f}")
    print(f"单工作点最多通过 {best} 项，达到该数的工作点 {best_points}；"
          f"推荐 {tie[0]}；结构性未通过 {overall['structural_failures']}")


if __name__ == "__main__":
    main()
