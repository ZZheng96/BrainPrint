"""E137 evaluation: greedy LOSO-driven view fusion, then the clean three-way metrics.

Everything here is Train-side except the final *scoring* of Test with a
Train-derived threshold. View/k selection uses LOSO AUC only.
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
TASKS = CFG["tasks"]


def z_with(values, values_tasks, ref, ref_tasks):
    out = np.full(len(values), np.nan, dtype=np.float32)
    for t in TASKS:
        m = values_tasks == t
        r = ref[ref_tasks == t]
        r = r[np.isfinite(r)]
        if len(r) < 10:
            continue
        out[m] = (values[m] - float(r.mean())) / (float(r.std()) + 1e-6)
    return out


def loso_parts(entry, train_tasks):
    """Per-fold z-scored known/unknown LOSO scores."""
    ref = entry["ref_raw"]
    parts = []
    for fd in entry["raw_loso"]:
        kn = z_with(fd["known"], fd["known_task"], ref, train_tasks)
        un = z_with(fd["unknown"], fd["unknown_task"], ref, train_tasks)
        parts.append((kn, un))
    return parts


def loso_metrics(parts_list, a):
    """parts_list: list of per-fold (known, unknown) lists, one entry per view."""
    rejs, aucs = [], []
    for f_i in range(len(parts_list[0])):
        kn = np.mean([p[f_i][0] for p in parts_list], axis=0)
        un = np.mean([p[f_i][1] for p in parts_list], axis=0)
        th = float(np.quantile(kn, 1 - a))
        rejs.append(float(np.mean(un < th)))
        if len(np.unique(un)) > 1:
            aucs.append(float(roc_auc_score(np.r_[np.ones(len(kn)), np.zeros(len(un))],
                                            np.r_[kn, un])))
    return float(np.mean(rejs)), float(np.mean(aucs))


def main():
    with open(OUT / "view_scores.pkl", "rb") as fh:
        d = pickle.load(fh)
    store, is_s1, tasks = d["store"], d["is_s1"], d["tasks"]
    train_tasks = d["tasks"]
    names = sorted(store.keys(), key=lambda x: -store[x]["auc"])
    y = np.r_[np.zeros(int(is_s1.sum())), np.ones(int((~is_s1).sum()))]

    for nm in names:
        print(f"{nm[0]:14s} k={nm[1]}  Test AUC {store[nm]['auc']:.4f}", flush=True)

    parts = {nm: loso_parts(store[nm], train_tasks) for nm in store}
    single = {}
    for nm in store:
        r, auc = loso_metrics([parts[nm]], 0.85)
        single[nm] = (r, auc)
        print(f"{nm[0]:14s} k={nm[1]}  LOSO AUC {auc:.4f}  LOSO rej@0.85 {r:.4f}", flush=True)

    # ---- greedy forward selection on LOSO AUC (Train only)
    chosen, best = [], -1.0
    remaining = list(store.keys())
    while remaining:
        cand, cand_auc = None, best
        for nm in remaining:
            _, auc = loso_metrics([parts[c] for c in chosen + [nm]], 0.85)
            if auc > cand_auc + 1e-5:
                cand, cand_auc = nm, auc
        if cand is None:
            break
        chosen.append(cand)
        remaining.remove(cand)
        best = cand_auc
        print(f"greedy += {cand[0]}_k{cand[1]} -> LOSO AUC {best:.4f}", flush=True)

    fusions = {"greedy": chosen, "all_equal": list(store.keys())}

    rows = []
    for label, sel in fusions.items():
        te = np.mean([store[nm]["te"] for nm in sel], axis=0)
        ref = np.mean([store[nm]["ref"] for nm in sel], axis=0)
        plist = [parts[nm] for nm in sel]
        auc = float(roc_auc_score(y, np.r_[te[is_s1], te[~is_s1]]))
        for a in CFG["acceptance_grid"]:
            th = float(np.quantile(ref[np.isfinite(ref)], 1 - a))
            s2 = float(np.mean(te[~is_s1] >= th))
            s1 = float(np.mean(te[is_s1] < th))
            loso, lauc = loso_metrics(plist, a)
            rows.append({"fusion": label, "n_views": len(sel), "acceptance_rate": a,
                         "diag_auc": auc, "LOSO_rejection": loso,
                         "Test_S1_rejection": s1, "Test_S2_acceptance": s2,
                         "three_way_mean": (loso + s1 + s2) / 3.0})
        # oracle Test threshold, diagnostic only (upper bound)
        qs = np.quantile(te, np.linspace(0.01, 0.995, 400))
        bests = max(((float(np.mean(te[is_s1] < t)), float(np.mean(te[~is_s1] >= t)))
                     for t in qs), key=lambda p: p[0] + p[1])
        print(f"{label}: n={len(sel)} Test AUC {auc:.4f} "
              f"oracle S1+S2 max {bests[0]:.4f}+{bests[1]:.4f}={bests[0]+bests[1]:.4f}", flush=True)

    rows.sort(key=lambda r: -r["three_way_mean"])
    out = {"experiment": CFG["experiment"], "clean": True,
           "selected": {"greedy": [f"{v}_k{k}" for v, k in chosen]},
           "single_view": {f"{v}_k{k}": {"test_auc": store[(v, k)]["auc"],
                                         "loso_auc": single[(v, k)][1],
                                         "loso_rej@0.85": single[(v, k)][0]}
                           for (v, k) in store},
           "best": rows[0], "curve": rows}
    (OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    lines = ["# E137 Multi-view score fusion (clean)", "",
             "目标：(LOSO 拒识 + Test S1 拒识 + Test S2 接受)/3 > 0.85。"
             "视图选择仅用 LOSO AUC；PCA/原型/分数标准化/阈值全部来自 Train。Test 只被评分。", "",
             "## 单视图", "",
             "| 视图 | k | Test AUC | LOSO AUC | LOSO 拒识 @0.85 |",
             "|---|---:|---:|---:|---:|"]
    for nm in names:
        lines.append(f"| {nm[0]} | {nm[1]} | {store[nm]['auc']:.4f} | "
                     f"{single[nm][1]:.4f} | {single[nm][0]:.4f} |")
    lines += ["", "贪婪选择（按 LOSO AUC）：" + " → ".join(f"{v}_k{k}" for v, k in chosen), "",
              "## 融合结果", "",
              "| 融合 | 视图数 | 接受率 | 诊断 AUC | LOSO 拒识 | Test S1 拒识 | Test S2 接受 | 三项均值 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows[:20]:
        lines.append(f"| {r['fusion']} | {r['n_views']} | {r['acceptance_rate']:.2f} | "
                     f"{r['diag_auc']:.4f} | {r['LOSO_rejection']:.4f} | "
                     f"{r['Test_S1_rejection']:.4f} | {r['Test_S2_acceptance']:.4f} | "
                     f"{r['three_way_mean']:.4f} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = "
              "N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out["best"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
