"""E174 步骤 4：合并结构化三元组 + CSV 补算，打有效性标签，输出主表数据。

输入：reported_w.json、json_pairs.json、all_records.json、csv_audit.json
输出：master_table.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

LOSO_CONTAMINATED = {
    "E119_score_battery", "E133_mean_objective", "E136_clean_three_way",
    "E137_multi_view", "E138_scale_free_score", "E139_feature_fusion",
    "E140_metric_learning", "E141_weighted_image_route", "E142_cdf_fusion",
}
TEST_STATISTIC = {"E135_rank_calibration", "E142_cdf_fusion"}
WORKPOINT_MISMATCH = {"E132_submission_candidates"}
DEGENERATE_S1, DEGENERATE_S2 = 0.005, 0.995
GRID_LO, GRID_HI = 0.80, 0.96


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main():
    reported = load("reported_w.json")
    md_tables = load("report_tables.json")
    pairs = [p for p in load("json_pairs.json") if p["LOSO"] is not None]
    old = load("all_records.json")
    csvs = [c for c in load("csv_audit.json") if c["status"] == "ok"]

    struct = []
    for r in reported:
        struct.append(dict(experiment=r["experiment"], source="reported",
                           file=r["file"], path=r["path"], config=r["config"],
                           acceptance=r["acceptance"], LOSO=r["LOSO"],
                           S1=r["S1"], S2=r["S2"], W=r["W"],
                           W_source=r["W_source"]))
    for r in md_tables:
        struct.append(dict(experiment=r["experiment"], source="md_table",
                           file=r["file"], path=r["path"], config=r["config"],
                           acceptance=r["acceptance"], LOSO=r["LOSO"],
                           S1=r["S1"], S2=r["S2"], W=r["W"],
                           W_source="md_table"))
    for p in pairs:
        struct.append(dict(experiment=p["experiment"], source="json_pair",
                           file=p["file"], path=p["path"], config="",
                           acceptance=p["known_acceptance"], LOSO=p["LOSO"],
                           S1=p["S1"], S2=p["S2"],
                           W=0.25 * p["LOSO"] + 0.25 * p["S1"] + 0.5 * p["S2"],
                           W_source="pair"))
    keys = {(r["experiment"], round(r["S1"], 5), round(r["S2"], 5),
             round(r["LOSO"], 5)) for r in struct}
    for r in old:
        k = (r["experiment"], round(r["S1"], 5), round(r["S2"], 5), round(r["LOSO"], 5))
        if k in keys:
            continue
        keys.add(k)
        struct.append(dict(experiment=r["experiment"], source="old_scan",
                           file=r["file"], path=r["path"],
                           config=str(r.get("label") or ""),
                           acceptance=r.get("acceptance"), LOSO=r["LOSO"],
                           S1=r["S1"], S2=r["S2"], W=r["W"], W_source="old"))

    seen, uniq = set(), []
    for r in struct:
        k = (r["experiment"], round(r["S1"], 6), round(r["S2"], 6), round(r["LOSO"], 6))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    struct = uniq

    # 同一三元组的近似重复（四舍五入差异）：保留带 acceptance 的那条
    struct.sort(key=lambda r: (r["acceptance"] is None, -r["W"]))
    dedup = []
    for r in struct:
        dup = next((d for d in dedup
                    if d["experiment"] == r["experiment"]
                    and abs(d["S1"] - r["S1"]) < 2e-4
                    and abs(d["S2"] - r["S2"]) < 2e-4
                    and abs(d["LOSO"] - r["LOSO"]) < 2e-4), None)
        if dup is not None:
            if dup["acceptance"] is None and r["acceptance"] is not None:
                dup["acceptance"] = r["acceptance"]
            continue
        dedup.append(r)
    struct = dedup

    # CSV 与结构化记录比对：同一个 Test 工作点才认
    for c in csvs:
        cands = [s for s in struct
                 if s["experiment"] == c["experiment"]
                 and abs(s["S1"] - c["S1"]) <= 2e-3
                 and abs(s["S2"] - c["S2"]) <= 2e-3]
        c["matched"] = bool(cands)
        c["matched_loso"] = sorted({round(s["LOSO"], 6) for s in cands})
        c["matched_acc"] = sorted({round(s["acceptance"], 6) for s in cands
                                   if s["acceptance"] is not None})
        c["matched_path"] = cands[0]["path"] if cands else None
        if c["session_zeroed"]:
            c["validity"], c["reason"] = "test_statistic", "按 Session 规则将 session1 全部置 0"
        elif c["experiment"] in LOSO_CONTAMINATED:
            c["validity"], c["reason"] = "loso_contaminated", "原型集合曾包含被评分块"
        elif c["experiment"] in TEST_STATISTIC:
            c["validity"], c["reason"] = "test_statistic", "使用 Test 侧统计量"
        elif c["experiment"] in WORKPOINT_MISMATCH:
            c["validity"], c["reason"] = "no_loso", "只有一个全局 LOSO，未与各工作点对齐"
        elif not cands:
            c["validity"], c["reason"] = "no_loso", "同实验内没有同工作点的 LOSO 记录"
        else:
            c["validity"], c["reason"] = "clean", "CSV 的 S1/S2 与已有记录一致"
        c["W"] = (0.25 * cands[0]["LOSO"] + 0.25 * c["S1"] + 0.5 * c["S2"]
                  if cands and len({round(s["LOSO"], 6) for s in cands}) == 1 else None)

    for s in struct:
        if s["experiment"] in LOSO_CONTAMINATED:
            s["validity"] = "loso_contaminated"
        elif s["experiment"] in TEST_STATISTIC:
            s["validity"] = "test_statistic"
        elif s["experiment"] in WORKPOINT_MISMATCH:
            s["validity"] = "no_loso"
        else:
            s["validity"] = "clean"
        s["degenerate"] = (s["S1"] is not None and s["S2"] is not None
                           and (s["S1"] <= DEGENERATE_S1 or s["S2"] >= DEGENERATE_S2))
        s["on_grid"] = (s["acceptance"] is not None
                        and GRID_LO - 1e-9 <= s["acceptance"] <= GRID_HI + 1e-9)

    master = {"structured": struct, "csvs": csvs}
    (OUT / "master_table.json").write_text(
        json.dumps(master, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"structured={len(struct)} csv_ok={len(csvs)} "
          f"csv_matched={sum(1 for c in csvs if c['matched'])}")
    best = {}
    for s in struct:
        cur = best.get(s["experiment"])
        if cur is None or s["W"] > cur["W"]:
            best[s["experiment"]] = s
    print("\n=== 结构化记录：每个实验最大 W ===")
    for k, s in sorted(best.items(), key=lambda x: -x[1]["W"]):
        print(f"{s['W']:.4f} {k:44s} a={str(s['acceptance'])[:6]:>6s} "
              f"S1={s['S1']:.4f} S2={s['S2']:.4f} L={s['LOSO']:.4f} "
              f"{s['validity']:18s}{' DEGEN' if s['degenerate'] else ''}"
              f"{'' if s['on_grid'] else ' offgrid'}")

    print("\n=== CSV 覆盖、但没有同工作点 LOSO 的实验 ===")
    have = set(best)
    by = {}
    for c in csvs:
        by.setdefault(c["experiment"], []).append(c)
    for k in sorted(by):
        if k in have:
            continue
        cs = by[k]
        bc = max(cs, key=lambda c: 0.25 * c["S1"] + 0.5 * c["S2"])
        w0 = 0.25 * bc["S1"] + 0.5 * bc["S2"]
        need = (0.71 - w0) / 0.25
        print(f"{k:47s} n={len(cs):2d} S1={bc['S1']:.4f} S2={bc['S2']:.4f} "
              f"W(L=0)={w0:.4f} needLOSO@0.71={need:+.3f} zeroed={bc['session_zeroed']}")


if __name__ == "__main__":
    main()
