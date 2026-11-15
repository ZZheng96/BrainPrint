"""E174：全量 W 补算。

递归扫描 research/ 下所有 results.json / metrics.json / *.json，
把每条含有三项指标（LOSO 拒识、Test S1 拒识、Test S2 接受）的记录抽出来，
在同一工作点上计算 W = 0.25*LOSO + 0.25*S1 + 0.5*S2。

只做补算，不改动任何已有实验。
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research"
OUT = Path(__file__).resolve().parent

# ---- 键名归一化 -------------------------------------------------------------
LOSO_KEYS = [
    "LOSO_rejection", "loso_rejection", "loso_rej", "LOSO_rej",
    "loso_unknown_rejection", "LOSO_unknown_rejection",
    "unknown_rejection", "Unknown_rejection", "ur", "UR",
    "loso_at_best", "loso_at_090", "loso_at_090_any", "loso_at_ref",
    "LOSO_rejection_rate", "heldout_rejection",
]
S1_KEYS = [
    "Test_S1_rejection", "test_s1_rejection", "s1_rejection", "S1_rejection",
    "s1_rej", "S1_rej", "testS1_rejection", "s1_at_best", "s1_at_090",
    "S1_refusal", "test_s1_refusal",
]
S2_KEYS = [
    "Test_S2_acceptance", "test_s2_acceptance", "s2_acceptance", "S2_acceptance",
    "s2_acc", "S2_acc", "testS2_acceptance", "s2_at_best", "s2_at_090",
    "S2_acceptance_rate", "known_acceptance_test",
]
ACC_KEYS = ["acceptance_rate", "accept_rate", "a", "target_acceptance", "ka", "known_acceptance"]
W_KEYS = ["W", "w", "weighted", "W_total", "new_W"]

FORBIDDEN = {"E135_rank_calibration"}
CONTAMINATED_LOSO = {
    "E119_score_battery", "E137_multi_view", "E138_scale_free_score",
    "E139_feature_fusion", "E140_metric_learning", "E141_weighted_image_route",
    "E142_cdf_fusion", "E136_clean_three_way", "E133_mean_objective",
}


def num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def pick(d, keys):
    for k in keys:
        if k in d:
            v = num(d[k])
            if v is not None:
                return v
    return None


def walk(obj, path=""):
    """产生所有 dict 记录。"""
    if isinstance(obj, dict):
        yield path, obj
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")


def main():
    records = []
    scanned = 0
    for exp_dir in sorted(p for p in RESEARCH.iterdir() if p.is_dir() and re.match(r"E\d+", p.name)):
        for jf in sorted(exp_dir.glob("*.json")):
            if jf.name in {"feature_manifest.json", "pixel_manifest.json", "manifest.json",
                           "encoder_config.json", "config.json"}:
                continue
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            scanned += 1
            for path, rec in walk(data):
                loso = pick(rec, LOSO_KEYS)
                s1 = pick(rec, S1_KEYS)
                s2 = pick(rec, S2_KEYS)
                if loso is None or s1 is None or s2 is None:
                    continue
                if not (0 <= loso <= 1 and 0 <= s1 <= 1 and 0 <= s2 <= 1):
                    continue
                acc = pick(rec, ACC_KEYS)
                w = 0.25 * loso + 0.25 * s1 + 0.5 * s2
                records.append({
                    "experiment": exp_dir.name,
                    "file": jf.name,
                    "path": path,
                    "LOSO": loso,
                    "S1": s1,
                    "S2": s2,
                    "acceptance": acc,
                    "W": w,
                    "label": rec.get("config") or rec.get("method") or rec.get("name")
                             or rec.get("variant") or rec.get("set") or "",
                })

    print(f"scanned {scanned} json files, found {len(records)} complete three-way rows")

    # 每个实验的最佳 W
    best = {}
    for r in records:
        cur = best.get(r["experiment"])
        if cur is None or r["W"] > cur["W"]:
            best[r["experiment"]] = r
    ranked = sorted(best.values(), key=lambda r: -r["W"])

    print("\n=== 每个实验的最佳 W（同一工作点）===")
    print(f"{'实验':46s} {'W':>7s} {'LOSO':>7s} {'S1':>7s} {'S2':>7s} {'acc':>6s}  flag")
    for r in ranked:
        flag = ""
        if r["experiment"] in FORBIDDEN:
            flag = "违规(Test统计)"
        elif r["experiment"] in CONTAMINATED_LOSO:
            flag = "LOSO污染"
        acc = f"{r['acceptance']:.3f}" if r["acceptance"] is not None else "  -  "
        print(f"{r['experiment'][:46]:46s} {r['W']:7.4f} {r['LOSO']:7.4f} {r['S1']:7.4f} "
              f"{r['S2']:7.4f} {acc:>6s}  {flag}")

    (OUT / "all_records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")

    # 关键：哪些实验有 S1/S2 但缺 LOSO（无法补算）
    partial = Counter()
    for exp_dir in sorted(p for p in RESEARCH.iterdir() if p.is_dir() and re.match(r"E\d+", p.name)):
        has = any(r["experiment"] == exp_dir.name for r in records)
        if not has:
            partial[exp_dir.name] = 1
    print(f"\n没有任何完整三项记录的实验数：{len(partial)}")
    print("示例：", list(partial)[:20])


if __name__ == "__main__":
    main()
