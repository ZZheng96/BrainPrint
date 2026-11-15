"""E188：按新口径「S1 拒识与 S2 接受的均值 > 0.7」重排全部工作点。

同时报告 min(S1,S2) 以暴露偏科：均值达标但一侧极低的点不算真正兼顾。
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
REC = ROOT / "research" / "E174_recompute_all_w" / "all_records.json"

CONTAM = {
    "E119_score_battery", "E133_mean_objective", "E136_clean_three_way",
    "E137_multi_view", "E138_scale_free_score", "E139_feature_fusion",
    "E140_metric_learning", "E141_weighted_image_route", "E142_cdf_fusion",
}
FORBIDDEN = {"E135_rank_calibration"}

rows = json.loads(REC.read_text(encoding="utf-8"))
valid = [r for r in rows if r["experiment"] not in CONTAM and r["experiment"] not in FORBIDDEN]
for r in valid:
    r["mean"] = (r["S1"] + r["S2"]) / 2
    r["min"] = min(r["S1"], r["S2"])

passed = [r for r in valid if r["mean"] > 0.70]
print(f"有效工作点 {len(valid)} 条；均值 >0.70 的 {len(passed)} 条")

print("\n=== 按 min(S1,S2) 排序（真正兼顾，最不偏科）===")
print(f"{'min':>6} {'mean':>6} {'S1':>6} {'S2':>6} {'LOSO':>6}  实验 / 配置")
for r in sorted(passed, key=lambda r: -r["min"])[:12]:
    print(f"{r['min']:6.4f} {r['mean']:6.4f} {r['S1']:6.4f} {r['S2']:6.4f} {r['LOSO']:6.4f}  "
          f"{r['experiment']}  {str(r.get('label'))[:34]}")

print("\n=== 按 mean 排序（含偏科）===")
for r in sorted(passed, key=lambda r: -r["mean"])[:12]:
    print(f"{r['mean']:6.4f} {r['min']:6.4f} {r['S1']:6.4f} {r['S2']:6.4f} {r['LOSO']:6.4f}  "
          f"{r['experiment']}  {str(r.get('label'))[:34]}")

# 兼顾判定：均值 >0.70 且 min >= 0.60（避免一侧塌陷）
strict = [r for r in passed if r["min"] >= 0.60]
print(f"\n均值 >0.70 且 min>=0.60（不偏科）的工作点：{len(strict)} 条")
for r in sorted(strict, key=lambda r: -r["min"])[:10]:
    print(f"  min {r['min']:.4f} mean {r['mean']:.4f} S1 {r['S1']:.4f} S2 {r['S2']:.4f} "
          f"LOSO {r['LOSO']:.4f} W {r['W']:.4f}  {r['experiment']}  {str(r.get('label'))[:30]}")

json.dump(
    {"n_valid": len(valid), "n_mean_gt_070": len(passed),
     "n_mean_gt_070_and_min_ge_060": len(strict),
     "top_by_min": [{k: r[k] for k in ("experiment", "label", "S1", "S2", "LOSO", "W", "mean", "min")}
                    for r in sorted(passed, key=lambda r: -r["min"])[:20]],
     "top_by_mean": [{k: r[k] for k in ("experiment", "label", "S1", "S2", "LOSO", "W", "mean", "min")}
                     for r in sorted(passed, key=lambda r: -r["mean"])[:20]]},
    open(OUT / "results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
