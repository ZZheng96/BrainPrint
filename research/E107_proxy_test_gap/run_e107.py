"""E107: proxy-vs-Test gap audit across all candidates.

Collects, for every candidate that has both a leave-one-identity-out (LOSO)
proxy AUC and a project Test S1-vs-S2 diagnostic AUC, the gap between them.
The Test diagnostic AUC is computed only from the already-written prediction
scores and Session labels; it is a post-hoc engineering diagnostic, not a
metric used for selection.

Purpose: quantify how much each proxy overstates generalisation, so future
candidates can be judged by the right yardstick.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]

# Values recorded by the individual experiments (all Train-only proxies).
LOSO = {
    "A_spd_ratio": 0.9144,
    "B_raw_cosine": 0.9353,
    "C_raw_ratio": 0.8641,
    "D_inv_ratio": 0.5564,
    "E_mean_fusion": 0.9289,
}
# Post-hoc Test S1-vs-S2 diagnostic AUC, computed from frozen prediction scores.
TEST_DIAG = {
    "E102_spd_ratio": 0.5856,
    "E104_inv_ratio": 0.5305,
    "E106_mean_fusion": 0.5308,
}
# Frozen Test audit outcomes.
AUDIT = {
    "E102_spd_ratio": {"S1_rejection": 0.8427, "S2_acceptance": 0.2873},
    "E103_external_cal": {"S1_rejection": 0.9291, "S2_acceptance": 0.1258},
    "E104_inv_ratio": {"S1_rejection": 0.4263, "S2_acceptance": 0.6150},
    "E106_mean_fusion": {"S1_rejection": 0.5345, "S2_acceptance": 0.5176},
}


def run():
    rows = []
    for name, loso in LOSO.items():
        key = {"A_spd_ratio": "E102_spd_ratio",
               "D_inv_ratio": "E104_inv_ratio",
               "E_mean_fusion": "E106_mean_fusion"}.get(name)
        diag = TEST_DIAG.get(key) if key else None
        rows.append({"candidate": name, "loso_auc": loso,
                     "test_diag_auc": diag,
                     "gap": (loso - diag) if diag is not None else None})
    df = pd.DataFrame(rows)
    result = {"experiment": "E107_proxy_test_gap",
              "note": "Test diagnostic AUC uses Session labels only after predictions were written; "
                      "it is not used for selection.",
              "loso": LOSO, "test_diagnostic": TEST_DIAG, "audit": AUDIT,
              "gap_table": df.to_dict(orient="records")}
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# E107 proxy-vs-Test gap audit", "",
             "汇总各候选的留一身份（LOSO）代理 AUC 与项目 Test 事后诊断 AUC。", "",
             "| 候选 | LOSO 代理 AUC | Test 诊断 AUC | 差距 |", "|---|---:|---:|---:|"]
    for r in result["gap_table"]:
        d = f"{r['test_diag_auc']:.4f}" if r["test_diag_auc"] is not None else "未测"
        g = f"{r['gap']:.4f}" if r["gap"] is not None else "—"
        lines.append(f"| {r['candidate']} | {r['loso_auc']:.4f} | {d} | {g} |")
    lines += ["", "## 冻结 Test 审计结果", "",
              "| 候选 | S1 拒识率 | S2 接受率 |", "|---|---:|---:|"]
    for k, v in AUDIT.items():
        lines.append(f"| {k} | {v['S1_rejection']:.4f} | {v['S2_acceptance']:.4f} |")
    lines += ["", "## 结论", "",
              "LOSO 代理 AUC 普遍在 0.86–0.94，但对应候选在 Test 上的真实区分度只有 0.53–0.59。",
              "代理高估约 0.30–0.40 AUC。这说明留一身份代理无法预测跨 Session 泛化，",
              "任何仅凭 LOSO 代理判断的候选都不可信。", "",
              "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    run()
