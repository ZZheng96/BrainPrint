"""E243 补充分析：排序能力 vs 工作点上限。

只读读取 results.json 与 scores/outer_scores.npz，不重新训练、不读取 Test。
输出：
  1) 每个候选的逐折 AUC 均值；
  2) TPR@FAR=1%/5%（FAR 以 known 分数下分位定义，接近实际部署的极低误接受要求）；
  3) 逐折最优阈值下的 unknown-F1（oracle 上限，仅用于诊断，不可作为可交付结果）；
  4) eeg_logbp 与 eeg_only 的配对差值。
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
CANDIDATES = ["eeg_only", "eeg_logbp", "eeg_dino", "eeg_logbp_dino"]


def tpr_at_far(known, unknown, far):
    """阈值 = known 分数的 far 分位，即 known 被误拒比例 = far。"""
    if not len(known) or not len(unknown):
        return float("nan")
    th = float(np.quantile(known, far, method="lower"))
    return float((unknown < th).mean())


def f1_at_known_reject(known, unknown, known_reject):
    """阈值取 known 分数的 known_reject 分位（即已知接受 1-known_reject）。"""
    th = float(np.quantile(known, known_reject, method="lower"))
    tp = int((unknown < th).sum())
    fp = int((known < th).sum())
    fn = int(len(unknown) - tp)
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def best_f1(known, unknown):
    s = np.r_[known, unknown]
    is_unknown = np.r_[np.zeros(len(known), bool), np.ones(len(unknown), bool)]
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    u_sorted = is_unknown[order]
    # 依次把阈值放在每个 distinct 分数之后：reject = s < t
    pos_total = int(is_unknown.sum())
    tp = 0
    fp = 0
    best = 0.0
    i = 0
    n = len(s_sorted)
    while i < n:
        j = i
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        blk = u_sorted[i:j]
        tp += int(blk.sum())
        fp += int((~blk).sum())
        fn = pos_total - tp
        if tp > 0:
            f1 = 2 * tp / (2 * tp + fp + fn)
            best = max(best, f1)
        i = j
    return float(best)


def main():
    res = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    z = np.load(OUT / "scores" / "outer_scores.npz")
    folds = list(range(1, 21))
    table = {}
    for name in CANDIDATES:
        aucs, t1, t5, t10, bf1, rej90, acc90, f1_90 = [], [], [], [], [], [], [], []
        for held in folds:
            row = next(r for r in res["results"][name]["folds"] if r["held_subject"] == held)
            known = z[f"{name}_held_{held:02d}_known"]
            unknown = z[f"{name}_held_{held:02d}_unknown"]
            aucs.append(row["AUC"])
            t1.append(tpr_at_far(known, unknown, 0.01))
            t5.append(tpr_at_far(known, unknown, 0.05))
            t10.append(tpr_at_far(known, unknown, 0.10))
            bf1.append(best_f1(known, unknown))
            f1_90.append(f1_at_known_reject(known, unknown, 0.10))
            # 实际使用的阈值（validation subject 90% 接受）
            th = row["threshold"]
            rej90.append(float((unknown < th).mean()))
            acc90.append(float((known >= th).mean()))
        table[name] = {
            "AUC_mean": float(np.mean(aucs)),
            "TPR_at_FAR1": float(np.mean(t1)),
            "TPR_at_FAR5": float(np.mean(t5)),
            "TPR_at_FAR10": float(np.mean(t10)),
            "oracle_threshold_F1_mean": float(np.mean(bf1)),
            "F1_at_known_acceptance90": float(np.mean(f1_90)),
            "used_threshold_unknown_rejection": float(np.mean(rej90)),
            "used_threshold_known_acceptance": float(np.mean(acc90)),
            "per_fold_AUC": aucs,
            "per_fold_oracle_F1": bf1,
        }

    base, de = np.array(table["eeg_only"]["per_fold_AUC"]), np.array(table["eeg_logbp"]["per_fold_AUC"])
    d = de - base
    paired = {
        "delta_mean": float(d.mean()),
        "delta_std": float(d.std(ddof=1)),
        "delta_se": float(d.std(ddof=1) / np.sqrt(len(d))),
        "folds_improved": int((d > 0).sum()),
        "folds_total": int(len(d)),
    }
    payload = {"protocol_note": "TPR@FAR 阈值取每折 known 分数的 far 分位（known 误拒率=far），TPR=unknown 被拒比例；oracle F1 只用该折标签扫描阈值，属诊断上限，非可交付结果。",
               "candidates": table, "paired_logbp_minus_only_AUC": paired}
    (OUT / "analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'candidate':18s} {'AUC':>7s} {'TPR@F1%':>8s} {'TPR@F5%':>8s} {'TPR@F10%':>8s} {'F1@Acc90':>9s} {'oracleF1':>9s} {'usedRej':>8s} {'usedAcc':>8s}")
    for name in CANDIDATES:
        t = table[name]
        print(f"{name:18s} {t['AUC_mean']:7.4f} {t['TPR_at_FAR1']:8.4f} {t['TPR_at_FAR5']:8.4f} "
              f"{t['TPR_at_FAR10']:8.4f} {t['F1_at_known_acceptance90']:9.4f} {t['oracle_threshold_F1_mean']:9.4f} "
              f"{t['used_threshold_unknown_rejection']:8.4f} {t['used_threshold_known_acceptance']:8.4f}")
    print(f"paired AUC delta (logbp - only): {paired['delta_mean']:+.4f} +- {paired['delta_se']:.4f} (SE), "
          f"improved {paired['folds_improved']}/{paired['folds_total']}")


if __name__ == "__main__":
    main()
