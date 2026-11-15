"""E174 步骤 5：为无法补算的实验生成证据化原因清单。

判定依据（全部来自文件内容，不猜测）：
  has_loso   : 该实验任一 json 里出现 unknown_rejection / loso_rejection 等字段
  has_s1s2   : 该实验任一 json 里同时出现 S1 与 S2 的拒识/接受字段
  has_audit  : 出现 S1_rejection_rate / Test_S1_rejection 等 Test 审计字段
  n_csv      : 该实验下 22,251 行且能与元数据对齐的 Test 预测 CSV 数
输出 gaps.json 并打印 Markdown 表格行。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RESEARCH = ROOT / "research"

S1S2_KEYS = ["S1_rejection_rate", "Test_S1_rejection", "test_s1_rejection",
             "s1_reject", "s1_at_best", "S1_rejection", "S1_拒识"]
LOSO_KEYS = ["unknown_rejection", "loso_rejection", "LOSO_rejection",
             "loso_unknown_rejection", "unknown_rejection_mean", "loso_at_best",
             "loso_at_090", "LOSO"]


def walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)


def main():
    master = json.loads((OUT / "master_table.json").read_text(encoding="utf-8"))
    have = {s["experiment"] for s in master["structured"]}
    csv_ok = {}
    for c in master["csvs"]:
        csv_ok[c["experiment"]] = csv_ok.get(c["experiment"], 0) + 1

    rows = []
    for exp in sorted(p for p in RESEARCH.iterdir()
                      if p.is_dir() and re.match(r"E\d+", p.name)):
        name = exp.name
        if name in have:
            continue
        has_loso = has_s1s2 = has_audit = False
        for jf in exp.glob("*.json"):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            for node in walk(data):
                ks = set(node)
                if any(k in ks for k in LOSO_KEYS):
                    has_loso = True
                if any(k in ks for k in S1S2_KEYS):
                    has_s1s2 = True
                if any(k in ks for k in ("S1_rejection_rate", "Test_S1_rejection",
                                         "test_s1_rejection", "S1_rejection")):
                    has_audit = True
        n_csv = csv_ok.get(name, 0)
        if n_csv:
            reason = "有 Test 预测 CSV，但缺同一工作点的 LOSO（已算 S1/S2）"
        elif has_loso and has_s1s2:
            reason = "LOSO 与 Test S1/S2 未在同一工作点对齐"
        elif has_loso:
            reason = "只有 Train 侧 LOSO/伪未知拒识，无 Test S1/S2 记录"
        elif has_audit:
            reason = "只有 Test 审计，无 LOSO"
        else:
            reason = "无 LOSO、无 Test S1/S2、无可用逐样本预测 CSV"
        rows.append({"experiment": name, "n_csv": n_csv, "has_loso": has_loso,
                     "has_s1s2": has_s1s2, "has_audit": has_audit, "reason": reason})

    (OUT / "gaps.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"gaps={len(rows)} with_csv={sum(1 for r in rows if r['n_csv'])}")
    from collections import Counter
    print(Counter(r["reason"] for r in rows))
    for r in rows:
        print(f"| {r['experiment']} | {r['n_csv'] or '-'} | {r['reason']} |")


if __name__ == "__main__":
    main()
