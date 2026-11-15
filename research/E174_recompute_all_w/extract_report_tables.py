"""E174 步骤 3b：解析 report.md 中的 Markdown 表格，抽取同工作点的三元组。

只接受同时含 S1 拒识列、S2 接受列、LOSO 拒识列的表格行；
LOSO AUC 列不算。输出 report_tables.json。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def cells(line):
    s = line.strip()
    if not s.startswith("|"):
        return None
    s = s.strip("|")
    return [c.strip() for c in s.split("|")]


def isnum(c):
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?", c.replace(" ", "")))


def tofloat(c):
    c = c.replace(" ", "").replace("%", "")
    try:
        v = float(c)
    except ValueError:
        return None
    return v


def col_idx(hdr, pred):
    for i, h in enumerate(hdr):
        if pred(h):
            return i
    return None


def main():
    out = []
    for rep in sorted((ROOT / "research").rglob("report.md")):
        exp = rep.relative_to(ROOT / "research").parts[0]
        if not re.match(r"E\d+", exp):
            continue
        try:
            lines = rep.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        i = 0
        while i < len(lines):
            hdr = cells(lines[i])
            if hdr is None or i + 1 >= len(lines) or not set(lines[i + 1].replace("|", "").strip()) <= set("-: "):
                i += 1
                continue
            ih = [h.lower() for h in hdr]
            i_s1 = col_idx(ih, lambda h: ("s1" in h or "test s1" in h) and
                           ("拒识" in h or "reject" in h or "refus" in h))
            i_s2 = col_idx(ih, lambda h: "s2" in h and ("接受" in h or "accept" in h))
            i_l = col_idx(ih, lambda h: "loso" in h and ("拒识" in h or "reject" in h)
                          and "auc" not in h)
            i_a = col_idx(ih, lambda h: ("接受率" in h or "known acc" in h
                                         or "acceptance" in h) and "s2" not in h)
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                row = cells(lines[j])
                if row and i_s1 is not None and i_s2 is not None and i_l is not None \
                        and max(i_s1, i_s2, i_l) < len(row):
                    s1, s2, l = tofloat(row[i_s1]), tofloat(row[i_s2]), tofloat(row[i_l])
                    a = tofloat(row[i_a]) if i_a is not None and i_a < len(row) else None
                    if None not in (s1, s2, l) and all(0 <= v <= 1 for v in (s1, s2, l)):
                        out.append(dict(
                            experiment=exp, file=str(rep.relative_to(ROOT)).replace("\\", "/"),
                            path=f"md_table@L{i + 1}", config=row[0], acceptance=a,
                            LOSO=l, S1=s1, S2=s2,
                            W=0.25 * l + 0.25 * s1 + 0.5 * s2, W_source="md_table"))
                j += 1
            i = j if j > i else i + 1
    print(f"extracted {len(out)} report-table rows from {len({o['experiment'] for o in out})} experiments")
    (OUT / "report_tables.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    best = {}
    for r in out:
        if r["experiment"] not in best or r["W"] > best[r["experiment"]]["W"]:
            best[r["experiment"]] = r
    for k, r in sorted(best.items(), key=lambda x: -x[1]["W"]):
        print(f"{r['W']:.4f} {k:42s} a={str(r['acceptance'])[:5]:>5s} S1={r['S1']:.4f} "
              f"S2={r['S2']:.4f} L={r['LOSO']:.4f} {r['config'][:40]}")


if __name__ == "__main__":
    main()
