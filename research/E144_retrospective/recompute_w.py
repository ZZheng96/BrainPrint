"""Recompute the weighted objective for every stored experiment.

New objective (user, 2026-11-15):
    W = 0.25 * LOSO_rejection + 0.25 * Test_S1_rejection + 0.5 * Test_S2_acceptance

For each research/E*/results.json we look for rows carrying all three rates,
compute W, and keep the best row per experiment.  Experiments that never stored
one of the three rates cannot be recomputed and are listed as such instead of
being guessed.

CM knowledge baked in:
  - E135 used Test-batch CDF calibration  -> forbidden, flagged
  - E137-E142 LOSO prototypes contained the scored known block -> inflated, flagged
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
KEYS = ("LOSO_rejection", "Test_S1_rejection", "Test_S2_acceptance")

# experiments whose LOSO is known to be measured on a contaminated prototype set
LOSO_CONTAMINATED = {"E137_multi_view", "E138_scale_free_score", "E139_feature_fusion",
                     "E140_metric_learning", "E141_weighted_image_route",
                     "E142_cdf_fusion"}
# experiments that violate the clean protocol in other ways
FORBIDDEN = {"E135_rank_calibration"}


def rows_of(obj):
    """Yield every dict that looks like an operating point."""
    if isinstance(obj, dict):
        if all(k in obj for k in KEYS):
            yield obj
        for v in obj.values():
            yield from rows_of(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from rows_of(v)


def main():
    recs, skipped = [], []
    for p in sorted(ROOT.glob("research/E*/results.json")):
        name = p.parent.name
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:                      # noqa: BLE001
            skipped.append((name, f"解析失败: {exc}"))
            continue
        cand = list(rows_of(j))
        if not cand:
            skipped.append((name, "未存 LOSO/S1/S2 三者，无法补算"))
            continue
        for r in cand:
            r = dict(r)
            r["W"] = (0.25 * float(r["LOSO_rejection"]) + 0.25 * float(r["Test_S1_rejection"])
                      + 0.5 * float(r["Test_S2_acceptance"]))
            r["experiment"] = name
            r["loso_contaminated"] = name in LOSO_CONTAMINATED
            r["forbidden"] = name in FORBIDDEN
            recs.append(r)

    df = pd.DataFrame(recs)
    df.to_csv(OUT / "recomputed_all_points.csv", index=False, encoding="utf-8")

    # best row per experiment, ignoring forbidden ones for the headline table
    best = (df.sort_values("W", ascending=False)
              .groupby("experiment", as_index=False).first()
              .sort_values("W", ascending=False))
    best.to_csv(OUT / "recomputed_best_per_experiment.csv", index=False, encoding="utf-8")

    def cfg(r):
        for k in ("space", "set", "label", "config", "variant", "form", "fusion", "candidate"):
            if k in r and isinstance(r[k], str) and r[k]:
                return r[k]
        return "-"

    lines = ["# 新加权目标重算：全部已存实验", "",
             "目标：`W = 0.25×LOSO 拒识 + 0.25×Test S1 拒识 + 0.5×Test S2 接受`。",
             "每行是该实验在所有已存工作点里 W 最高的一个。", "",
             "| 实验 | 配置 | 接受率 | LOSO | S1 拒识 | S2 接受 | W | 诊断 AUC | 备注 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for _, r in best.iterrows():
        note = []
        if r["forbidden"]:
            note.append("**犯规：用了 Test 侧统计**")
        if r["loso_contaminated"]:
            note.append("**LOSO 被污染，W 偏高**")
        lines.append(
            f"| {r['experiment']} | {cfg(r)} | "
            f"{r.get('acceptance_rate', float('nan')):.2f} | "
            f"{r['LOSO_rejection']:.4f} | {r['Test_S1_rejection']:.4f} | "
            f"{r['Test_S2_acceptance']:.4f} | **{r['W']:.4f}** | "
            f"{r.get('diagnostic_AUC', r.get('test_auc', float('nan'))):.4f} | "
            f"{'；'.join(note) if note else '干净'} |")

    lines += ["", "## 无法补算的实验", "",
              "这些实验当时没有在同一工作点同时记录 LOSO 拒识与 Test S1/S2，"
              "因此无法事后拼出 W（拼出来的数不是同一个阈值下的三个率，没有意义）。", "",
              "| 实验 | 原因 |", "|---|---|"]
    for name, why in skipped:
        lines.append(f"| {name} | {why} |")
    (OUT / "recomputed.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"\npoints={len(df)}  experiments_recomputed={best.shape[0]}  skipped={len(skipped)}")


if __name__ == "__main__":
    main()
