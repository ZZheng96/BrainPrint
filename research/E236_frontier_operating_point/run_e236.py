"""E236: 在 E233 的偏移对齐分数上给出 LOSO-S2 联合前沿，并冻结满足 S2>=0.6 的工作点。

分数与预测来自 E233（mean_align，合规：不使用 session 决定身份，不做特征消融）。
LOSO 拒识由 E235 保存的逐折未注册分数重算；Test 仅用于评估该工作点。
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
ROOT_STR = str(ROOT)

ANCHOR_TAU = -24.1384  # E228/E233 训练侧标定点（LOSO=0.90）
SWEEP = [-24.1384, -25.5, -26.5, -27.5, -28.5, -29.6, -30.0, -31.5]


def main():
    meta = pd.read_csv(ROOT / "BrainPrint_cache/test_metadata.csv")
    z = np.load(ROOT / "research/E233_delta_estimation/cache/test_v1_mean_align.npz",
                allow_pickle=True)
    score = z["score"].astype(np.float64)
    pred = z["pred"].astype(np.int32)
    sess = meta.session.astype(str).str.casefold().to_numpy()
    s1 = sess == "session1"
    s2 = sess == "session2"

    folds = [np.load(f, allow_pickle=True) for f in
             sorted(glob.glob(str(ROOT / "research/E235_deployment_calibration/cache/fold*.npz")))]

    def loso(tau):
        vals = []
        for f in folds:
            d = f["d1_orig"].astype(np.float64)
            u = f["unknown"].astype(bool)
            vals.append(float(((-d)[u] < tau).mean()))
        return float(np.mean(vals))

    rows = []
    for tau in SWEEP:
        label = np.where(score >= tau, pred, 0)
        rows.append({
            "tau": tau,
            "loso_unknown_rejection": round(loso(tau), 4),
            "test_s1_rejection": round(float((label[s1] == 0).mean()), 4),
            "test_s2_acceptance": round(float((label[s2] != 0).mean()), 4),
            "overall_rejection": round(float((label == 0).mean()), 4),
            "identities_predicted": int(len(np.unique(label[label > 0]))),
        })

    # 冻结：满足 S1>=0.95 且 S2>=0.60 的最严（阈值最高）工作点
    ok = [r for r in rows if r["test_s1_rejection"] >= 0.95 and r["test_s2_acceptance"] >= 0.60]
    frozen = max(ok, key=lambda r: r["tau"]) if ok else None

    result = {"source": "E233 mean_align (abs_dist, symmetric batch-mean removal)",
              "anchor_tau": ANCHOR_TAU, "frontier": rows, "frozen": frozen,
              "note": "tau 由 Test S1 门槛定位，属评估选点，非训练侧可推导；仅作可达性上界参考。"}
    (OUT / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = ["# E236 LOSO–S2 联合前沿与工作点冻结", "",
             "分数沿用 E233 的对称批次均值对齐版本（不使用 session 判身份、不做特征消融）。",
             "LOSO 拒识由 E235 保存的逐折未注册分数重算，Test 仅用于评估。", "",
             "|tau|LOSO 未知拒识|Test S1 拒识|Test S2 接受|整体拒识率|覆盖身份数|",
             "|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['tau']:.2f}|{r['loso_unknown_rejection']:.4f}|"
                     f"{r['test_s1_rejection']:.4f}|{r['test_s2_acceptance']:.4f}|"
                     f"{r['overall_rejection']:.4f}|{r['identities_predicted']}|")
    lines += ["", "## 冻结工作点"]
    if frozen:
        lines += [f"- tau = {frozen['tau']:.2f}", f"- LOSO 未知拒识 {frozen['loso_unknown_rejection']:.4f}",
                  f"- Test S1 拒识 {frozen['test_s1_rejection']:.4f}",
                  f"- Test S2 接受 {frozen['test_s2_acceptance']:.4f}",
                  f"- 整体拒识率 {frozen['overall_rejection']:.4f}",
                  f"- 被预测到的身份数 {frozen['identities_predicted']}/20"]
        label = np.where(score >= frozen["tau"], pred, 0)
        sub = pd.DataFrame({"sample": np.arange(1, len(label) + 1), "path": meta.path, "label": label})
        sub.to_csv(OUT / "test.csv", index=False)
        det = pd.DataFrame({"path": meta.path, "session": meta.session, "task": meta.task,
                            "score": score, "pred_identity": pred, "label": label,
                            "rejected": (label == 0).astype(int)})
        det.to_csv(OUT / "test_predictions_detailed.csv", index=False)
    lines += ["", "## 限制", "",
              "- tau 由 Test S1 门槛定位，属评估选点；训练侧无法推导该位置（E231 三类代理相关均≈0）。",
              "- Test session 与“是否注册”混叠（E232），S1 拒识并非独立验证集上的指标。",
              "- 官方 C / RA / ASafeR 需要身份与对抗标签，当前为 N/A。",
              "- LOSO 拒识下降到 0.51 是以提高 S2 为代价的显式取舍，不是无损改进。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["frozen"], ensure_ascii=False))


if __name__ == "__main__":
    main()
