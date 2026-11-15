"""E174 步骤 1：用 Test 预测 CSV 补算 S1 拒识 / S2 接受。

只读：research/**、commit/** 下的预测 CSV 与 BrainPrint_cache/test_metadata.csv。
输出：csv_audit.json（每条 CSV 的审计结果）。

判定规则：
  S1 拒识率 = P(label == 0 | session1)
  S2 接受率 = P(label != 0 | session2)
  若 label==0 条数 == session1 条数，则该候选是按 Session 直接置 0（session 规则），
  标记 session_zeroed = True，不计入有效成绩。
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "BrainPrint_cache" / "test_metadata.csv"
OUT = Path(__file__).resolve().parent

NAME_HINTS = ("prediction", "submission", "candidate")
SIZE_MIN, SIZE_MAX = 150_000, 700_000
N_EXPECT = 22251


def load_meta():
    sess, tasks = {}, {}
    with META.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            s = row["sample"].strip()
            sess[s] = row["session"].strip()
            tasks[s] = row["task"].strip()
    return sess, tasks


def read_pred(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        fields = {n.strip().lower(): n for n in (rd.fieldnames or [])}
        sc = fields.get("sample")
        lc = fields.get("label")
        if sc is None or lc is None:
            return None, None, f"columns={rd.fieldnames}"
        rows = []
        for r in rd:
            rows.append((r[sc].strip(), r[lc].strip()))
    return rows, None, None


def main():
    sess, tasks = load_meta()
    print(f"metadata: {len(sess)} samples, "
          f"{Counter(sess.values())}")

    files = []
    for base in (ROOT / "research", ROOT / "commit"):
        for p in base.rglob("*.csv"):
            rel = str(p.relative_to(ROOT)).replace("\\", "/").lower()
            if "pseudo_answer" in rel or p.name.lower() == "test_template.csv":
                continue  # 伪真值/模板，不是候选预测
            try:
                sz = p.stat().st_size
            except OSError:
                continue
            if SIZE_MIN <= sz <= SIZE_MAX:
                files.append(p)

    results = []
    for p in sorted(files):
        rows, err, note = (None, None, None)
        try:
            rows, err, note = read_pred(p)
        except Exception as e:  # noqa: BLE001
            err = f"read-error: {e}"
        rec = {
            "csv": str(p.relative_to(ROOT)).replace("\\", "/"),
            "experiment": p.relative_to(ROOT).parts[1] if p.relative_to(ROOT).parts[0] == "research"
                          else p.relative_to(ROOT).parts[0],
            "size": p.stat().st_size,
            "n_rows": None, "S1": None, "S2": None,
            "zero_count": None, "n_s1": None, "n_s2": None,
            "status": "ok", "note": note or "",
        }
        if rows is None and not err:
            rec["status"] = "not_test_csv"
            rec["note"] = note or "no Sample/Label columns"
            results.append(rec)
            continue
        if err or rows is None:
            rec["status"] = "unreadable"
            rec["note"] = err or ""
            results.append(rec)
            continue
        n = len(rows)
        rec["n_rows"] = n
        if n != N_EXPECT:
            rec["status"] = "misaligned"
            rec["note"] = f"rows={n} != {N_EXPECT}"
            results.append(rec)
            continue
        samples = [s for s, _ in rows]
        if len(set(samples)) != n or set(samples) != set(sess):
            rec["status"] = "misaligned"
            missing = len(set(sess) - set(samples))
            extra = len(set(samples) - set(sess))
            rec["note"] = f"sample set mismatch: missing={missing}, extra={extra}"
            results.append(rec)
            continue
        labs = {}
        bad = 0
        for s, l in rows:
            try:
                labs[s] = int(float(l))
            except ValueError:
                bad += 1
        if bad:
            rec["status"] = "misaligned"
            rec["note"] = f"{bad} non-numeric labels"
            results.append(rec)
            continue
        s1 = [labs[s] for s in sess if sess[s] == "session1"]
        s2 = [labs[s] for s in sess if sess[s] == "session2"]
        zero = sum(1 for v in labs.values() if v == 0)
        rec.update({
            "S1": sum(1 for v in s1 if v == 0) / len(s1),
            "S2": sum(1 for v in s2 if v != 0) / len(s2),
            "zero_count": zero, "n_s1": len(s1), "n_s2": len(s2),
            "session_zeroed": zero == len(s1),
            "nonint_labels": bad,
        })
        results.append(rec)

    ok = [r for r in results if r["status"] == "ok"]
    print(f"files={len(files)} ok={len(ok)} "
          f"misaligned={sum(1 for r in results if r['status']=='misaligned')} "
          f"unreadable={sum(1 for r in results if r['status']=='unreadable')}")
    print(f"session_zeroed candidates: {sum(1 for r in ok if r.get('session_zeroed'))}")

    (OUT / "csv_audit.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
