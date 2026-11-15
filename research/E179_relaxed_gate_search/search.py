"""E179：在 E174 的全量记录里搜索「十项门禁」可行点。

放宽口径（按用户要求）：#9 → Test S2 接受率 ≥ 0.75；#10 → Test S1 拒识率 ≥ 0.75。
其余 8 项维持原阈值。
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
print(f"记录数 {len(rows)}")

valid = [r for r in rows
         if r["experiment"] not in CONTAM and r["experiment"] not in FORBIDDEN]
print(f"排除污染/违规后 {len(valid)}")

# 1) S1 + S2 之和的上界：决定放松后的 #9/#10 能否同时满足
sums = sorted(((r["S1"] + r["S2"], r) for r in valid), key=lambda t: -t[0])
print("\n=== S1 拒识 + S2 接受 最大者 ===")
for s, r in sums[:8]:
    print(f"  {s:.4f}  S1 {r['S1']:.4f}  S2 {r['S2']:.4f}  "
          f"LOSO {r['LOSO']:.4f}  W {r['W']:.4f}  {r['experiment']}")

# 2) 同时满足 S1>=0.75 且 S2>=0.75 的记录
both = [r for r in valid if r["S1"] >= 0.75 and r["S2"] >= 0.75]
print(f"\n同时满足 S1>=0.75 且 S2>=0.75 的记录：{len(both)}")
for r in sorted(both, key=lambda r: -r["W"])[:10]:
    print(f"  W {r['W']:.4f}  S1 {r['S1']:.4f}  S2 {r['S2']:.4f}  LOSO {r['LOSO']:.4f}  "
          f"{r['experiment']}  {str(r.get('label'))[:40]}")

# 3) 若 #9 仍理解为「拒识率 <= 0.75」（即接受 >= 0.25），条件更松，结果同上或更多
loose = [r for r in valid if r["S1"] >= 0.75 and r["S2"] >= 0.25]
print(f"\n若 #9 仅要求 S2 接受 >= 0.25：{len(loose)} 条")

# 4) 单独看 #5（AUROC>=0.98）与 #6（U@95>=0.95）在历史记录里的最好值
#    all_records 只含 LOSO 拒识（同一接受率），不含 U@95 与 AUROC，因此这里给可用的近似
best_loso = sorted(valid, key=lambda r: -r["LOSO"])[:5]
print("\n=== LOSO 拒识最高的记录（#6 的相关证据）===")
for r in best_loso:
    print(f"  LOSO {r['LOSO']:.4f}  S1 {r['S1']:.4f}  S2 {r['S2']:.4f}  "
          f"acc {r['acceptance']}  {r['experiment']}")

summary = {
    "n_records": len(rows),
    "n_valid": len(valid),
    "max_S1_plus_S2": sums[0][0] if sums else None,
    "max_sum_record": {k: sums[0][1][k] for k in ("experiment", "S1", "S2", "LOSO", "W")} if sums else None,
    "n_both_075": len(both),
    "both_075": [{k: r[k] for k in ("experiment", "S1", "S2", "LOSO", "W", "label")} for r in both[:20]],
    "best_loso": [{k: r[k] for k in ("experiment", "LOSO", "S1", "S2")} for r in best_loso],
}
(OUT / "results.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
