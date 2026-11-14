"""E68 实验：多假设集成——是否存在可迁移的「注册者 vs 未注册者」共同成分？

E67 证明：单个「留 4 名被试」训练的二分类器迁移到 Test 只有 0.49。
本实验检验：多个不同留出组合训练的分类器，其方向是否有共同成分？

方法：
  1. 在 Train 上枚举 C(20,4)=4845 种留出组合中的 N 种（随机采样）
  2. 每种训练一个 L2 二分类器，得到方向向量 w_i
  3. 计算方向间的平均余弦相似度（共同成分强度）
  4. 集成：对 Test 用所有 w_i 的平均分数，看 AUROC
  5. 对照：单个 w_i 的 Test AUROC 分布

若集成 AUROC 显著高于单个（0.49），说明存在共同成分；
若仍接近 0.5，说明方向完全随机，不存在通用判据。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CACHE = ROOT / "BrainPrint_cache"


def main() -> None:
    train_meta = pd.read_csv(CACHE / "train_metadata.csv")
    test_meta = pd.read_csv(CACHE / "test_metadata.csv")
    subjects = train_meta["subject"].to_numpy()

    train_layers = dict(np.load(CACHE / "e63_root_cause" / "train_layers.npz"))
    test_layers = dict(np.load(CACHE / "e63_root_cause" / "test_layers.npz"))

    session = test_meta["session"].astype(str).str.strip().str.casefold()
    codes = session.map({"1": "S1", "s1": "S1", "session1": "S1",
                         "2": "S2", "s2": "S2", "session2": "S2"})
    is_s1 = (codes == "S1").to_numpy()
    is_s2 = (codes == "S2").to_numpy()
    test_labels = np.concatenate([np.ones(is_s2.sum()), np.zeros(is_s1.sum())])

    all_subjects = np.unique(subjects)
    rng = np.random.default_rng(42)
    n_hypotheses = 30

    report = {}
    for layer in ("L1_band_power", "L2_log_cov"):
        train_x = train_layers[layer]
        test_x = np.concatenate([test_layers[layer][is_s2], test_layers[layer][is_s1]])

        directions, test_scores, single_aurocs = [], [], []
        for h in range(n_hypotheses):
            unknown = rng.choice(all_subjects, size=4, replace=False)
            labels = (~np.isin(subjects, unknown)).astype(int)
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
            model.fit(train_x, labels)
            scaler = model.named_steps["standardscaler"]
            coef = model.named_steps["logisticregression"].coef_.ravel()
            # 还原到原始特征空间的方向
            w = coef / scaler.scale_
            directions.append(w / (np.linalg.norm(w) + 1e-12))
            scores = model.decision_function(test_x)
            test_scores.append(scores)
            single_aurocs.append(float(roc_auc_score(test_labels, scores)))

        directions = np.stack(directions)
        test_scores = np.stack(test_scores)

        # 方向间平均余弦相似度
        gram = directions @ directions.T
        off_diag = gram[~np.eye(n_hypotheses, dtype=bool)]
        mean_cos = float(off_diag.mean())

        # 集成：平均分数
        ensemble = test_scores.mean(0)
        ensemble_auroc = float(roc_auc_score(test_labels, ensemble))

        # 集成：多数投票方向（符号对齐后平均）
        signs = np.sign(test_scores.mean(1, keepdims=True))
        aligned = test_scores * signs
        aligned_ensemble = aligned.mean(0)
        aligned_auroc = float(roc_auc_score(test_labels, aligned_ensemble))

        report[layer] = {
            "n_hypotheses": n_hypotheses,
            "mean_pairwise_cosine": mean_cos,
            "single_auroc_mean": float(np.mean(single_aurocs)),
            "single_auroc_std": float(np.std(single_aurocs)),
            "single_auroc_min": float(np.min(single_aurocs)),
            "single_auroc_max": float(np.max(single_aurocs)),
            "ensemble_auroc": ensemble_auroc,
            "aligned_ensemble_auroc": aligned_auroc,
        }
        print(f"\n=== {layer} ===")
        print(f"  方向间平均余弦 = {mean_cos:.4f}")
        print(f"  单假设 Test AUROC: mean={np.mean(single_aurocs):.4f} "
              f"std={np.std(single_aurocs):.4f} "
              f"[{np.min(single_aurocs):.4f}, {np.max(single_aurocs):.4f}]")
        print(f"  集成（平均分数）Test AUROC = {ensemble_auroc:.4f}")
        print(f"  集成（符号对齐）Test AUROC = {aligned_auroc:.4f}")

    (OUT / "ensemble_probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved {OUT / 'ensemble_probe.json'}", flush=True)


if __name__ == "__main__":
    main()
