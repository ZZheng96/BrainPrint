"""E190：检验「S1>=0.70 且 S2>=0.70」与其余 8 项门禁能否同时成立。"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
rows = json.loads((ROOT / "research" / "E174_recompute_all_w" / "all_records.json")
                  .read_text(encoding="utf-8"))

CONTAM = {"E119_score_battery", "E133_mean_objective", "E136_clean_three_way",
          "E137_multi_view", "E138_scale_free_score", "E139_feature_fusion",
          "E140_metric_learning", "E141_weighted_image_route", "E142_cdf_fusion"}
valid = [r for r in rows
         if r["experiment"] not in CONTAM and r["experiment"] != "E135_rank_calibration"]
for r in valid:
    r["mn"] = min(r["S1"], r["S2"])

both = [r for r in valid if r["mn"] >= 0.70]
print("有效工作点", len(valid))
print("min(S1,S2) >= 0.70 的工作点数:", len(both))

both.sort(key=lambda r: -r["LOSO"])
print("\n其中 LOSO 最高的 10 个：")
for r in both[:10]:
    print(f"  LOSO {r['LOSO']:.4f}  S1 {r['S1']:.4f}  S2 {r['S2']:.4f}  min {r['mn']:.4f}  "
          f"{r['experiment']}  {str(r.get('label'))[:30]}")

both.sort(key=lambda r: -r["mn"])
print("\n其中 min 最高的 10 个：")
for r in both[:10]:
    print(f"  min {r['mn']:.4f}  S1 {r['S1']:.4f}  S2 {r['S2']:.4f}  LOSO {r['LOSO']:.4f}  "
          f"{r['experiment']}  {str(r.get('label'))[:30]}")

for th in (0.70, 0.79, 0.85, 0.90, 0.95, 0.98):
    n = len([r for r in both if r["LOSO"] >= th])
    print(f"  min>=0.70 且 LOSO>={th:.2f}: {n} 条")

loso_max = max((r["LOSO"] for r in both), default=0.0)
mn_max = max((r["mn"] for r in both), default=0.0)
print(f"\n同时满足双 0.70 时：LOSO 最大 {loso_max:.4f}（#5 需 0.98）；min 最大 {mn_max:.4f}")

json.dump({"n_valid": len(valid), "n_both_070": len(both),
           "max_loso_given_both": loso_max, "max_min_given_both": mn_max,
           "top_by_loso": [{k: r[k] for k in ("experiment", "label", "S1", "S2", "LOSO", "mn")}
                           for r in sorted(both, key=lambda r: -r["LOSO"])[:20]]},
          open(OUT / "results.json", "w", encoding="utf-8"), ensure_ascii=False,
          indent=2, default=str)
