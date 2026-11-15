"""E174 步骤 6：生成 report.md 所需的 Markdown 表格（只打印，不写文件）。"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def load(n):
    return json.loads((OUT / n).read_text(encoding="utf-8"))


def main():
    m = load("master_table.json")
    S, C = m["structured"], m["csvs"]
    best = {}
    for s in S:
        cur = best.get(s["experiment"])
        if cur is None or s["W"] > cur["W"]:
            best[s["experiment"]] = s

    def short(e):
        return e.split("_")[0]

    print("### CLEAN")
    print("| # | 实验 | 方法/配置 | 接受率 | 口径 | S1 拒识 | S2 接受 | LOSO | W | 备注 |")
    print("|---:|---|---|---:|---|---:|---:|---:|---:|---|")
    rows = sorted((s for s in best.values() if s["validity"] == "clean"),
                  key=lambda s: -s["W"])
    for i, s in enumerate(rows, 1):
        if s["degenerate"]:
            note = "退化（S1=0/S2=1 的恒定分数）"
        elif "oracle" in str(s["path"]).lower():
            note = "oracle 阈值（Test 选点）"
        elif s["acceptance"] is not None and abs(s["acceptance"] - 0.9) < 1e-6:
            note = ""
        elif s["W_source"] in ("best", "md_table"):
            note = "网格最优（Test 选点）"
        elif not s["on_grid"]:
            note = "网格外工作点，仅记录"
        else:
            note = ""
        acc = "" if s["acceptance"] is None else f"{s['acceptance']:.2f}"
        cal = ("a=0.90" if s["acceptance"] is not None and abs(s["acceptance"] - 0.9) < 1e-6
               else ("grid-best" if s["W_source"] == "best" else "other"))
        cfg = s["config"] or s["experiment"].split("_", 1)[-1]
        print(f"| {i} | {short(s['experiment'])} | {cfg[:58]} | {acc} | {cal} | "
              f"{s['S1']:.4f} | {s['S2']:.4f} | {s['LOSO']:.4f} | {s['W']:.4f} | {note} |")

    print("\n### INVALID")
    print("| 实验 | 方法/配置 | a | S1 | S2 | LOSO | W | 标记 |")
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for s in sorted((s for s in best.values() if s["validity"] != "clean"),
                    key=lambda s: -s["W"]):
        acc = "" if s["acceptance"] is None else f"{s['acceptance']:.2f}"
        cfg = s["config"] or s["experiment"].split("_", 1)[-1]
        print(f"| {short(s['experiment'])} | {cfg[:55]} | {acc} | {s['S1']:.4f} | "
              f"{s['S2']:.4f} | {s['LOSO']:.4f} | {s['W']:.4f} | {s['validity']} |")

    print("\n### CSV ONLY")
    print("| 实验 | CSV 数 | 该 CSV 的 S1 | 该 CSV 的 S2 | W(LOSO=0) | 达 0.71 所需 LOSO | 备注 |")
    print("|---|---:|---:|---:|---:|---:|---|")
    by = defaultdict(list)
    for c in C:
        by[c["experiment"]].append(c)
    for k in sorted(by):
        if k in best:
            continue
        cs = by[k]
        bc = max(cs, key=lambda c: 0.25 * c["S1"] + 0.5 * c["S2"])
        w0 = 0.25 * bc["S1"] + 0.5 * bc["S2"]
        need = (0.71 - w0) / 0.25
        note = "含 Session 置 0 候选" if any(c["session_zeroed"] for c in cs) else ""
        name = k if k.startswith("E") else f"`{k}`"
        print(f"| {name} | {len(cs)} | {bc['S1']:.4f} | {bc['S2']:.4f} | {w0:.4f} | "
              f"{need:+.3f} | {note} |")

    print("\n### GAPS")
    gaps = load("gaps.json")
    grp = defaultdict(list)
    for g in gaps:
        if g["experiment"] == "E174_recompute_all_w":
            continue
        grp[g["reason"]].append(g["experiment"])
    for r, es in grp.items():
        print(f"\n**{r}（{len(es)}）**：{'、'.join(es)}")

    print(f"\ncounts: all=177, with_W={len(best)}, csv_only={len(set(by)-set(best))}, "
          f"gap_no_csv={sum(1 for g in gaps if not g['n_csv'] and g['experiment'] not in best and g['experiment'] != 'E174_recompute_all_w')}")

    # 口径统计
    def mx(rows):
        rows = list(rows)
        return max(rows, key=lambda s: s["W"]) if rows else None
    A = [s for s in S if s["acceptance"] is not None and abs(s["acceptance"] - 0.9) < 1e-6]
    print("\n### CALIBERS")
    for name, rows in [
        ("A all", A),
        ("A clean", [s for s in A if s["validity"] == "clean"]),
        ("A clean nondeg", [s for s in A if s["validity"] == "clean" and not s["degenerate"]]),
        ("B all", [s for s in S if s["on_grid"]]),
        ("B clean", [s for s in S if s["on_grid"] and s["validity"] == "clean"]),
        ("B clean nondeg", [s for s in S if s["on_grid"] and s["validity"] == "clean" and not s["degenerate"]]),
        ("C clean nondeg", [s for s in S if s["validity"] == "clean" and not s["degenerate"]]),
    ]:
        r = mx(rows)
        print(f"{name:16s} n={len(rows):5d} max={r['W']:.4f} {r['experiment']} "
              f"a={r['acceptance']} S1={r['S1']:.4f} S2={r['S2']:.4f} L={r['LOSO']:.4f}")

    zero = [c for c in C if c["session_zeroed"]]
    print(f"\nsession_zeroed CSVs: {len(zero)}")
    for c in zero:
        print(f"  {c['csv']}  S1={c['S1']:.4f} S2={c['S2']:.4f} zero={c['zero_count']}")


if __name__ == "__main__":
    main()
