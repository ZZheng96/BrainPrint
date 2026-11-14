"""E67 诊断 3：合规性验证——Train 上训练的「注册者 vs 未注册者」判据能否迁移到 Test。

E67 诊断 1 发现：用 Test 标签训练的 L2 二分类器 AUROC 0.9971。
但那是作弊。本诊断检验合规版本：

  在 Train 上，用「留出被试 = 未注册者，其余 = 注册者」训练二分类器，
  然后直接应用到 Test，看 AUROC(S2 vs S1) 是多少。

若迁移后仍显著高于 0.7327，说明存在合规可达的判据；
若迁移后掉回 0.5–0.7，说明 Test 的 0.9971 依赖 Test 特有的分布信息，不可迁移。

同时对比多种特征层与分类器，找出最稳的组合。
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
    groups = np.array_split(np.random.default_rng(42).permutation(all_subjects), 5)

    report = {}
    print(f"{'layer':16s} {'fold':>4s} {'train AUROC':>12s} {'test AUROC':>11s}", flush=True)
    for layer in ("L1_band_power", "L2_log_cov"):
        train_x = train_layers[layer]
        test_x = np.concatenate([test_layers[layer][is_s2], test_layers[layer][is_s1]])
        fold_rows = []
        for fold, unknown_subjects in enumerate(groups, 1):
            labels = (~np.isin(subjects, unknown_subjects)).astype(int)
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
            model.fit(train_x, labels)
            # Train 内部：留出被试的分数
            held = np.isin(subjects, unknown_subjects)
            train_scores = model.decision_function(train_x[held])
            train_auroc = float(roc_auc_score(1 - labels[held], train_scores))
            # Test 迁移
            test_scores = model.decision_function(test_x)
            test_auroc = float(roc_auc_score(test_labels, test_scores))
            fold_rows.append({
                "fold": fold,
                "train_auroc": train_auroc,
                "test_auroc": test_auroc,
                "test_auroc_flipped": max(test_auroc, 1 - test_auroc),
            })
            print(f"{layer:16s} {fold:4d} {train_auroc:12.4f} {test_auroc:11.4f}", flush=True)
        report[layer] = {
            "folds": fold_rows,
            "mean_train_auroc": float(np.mean([r["train_auroc"] for r in fold_rows])),
            "mean_test_auroc": float(np.mean([r["test_auroc"] for r in fold_rows])),
            "mean_test_auroc_flipped": float(np.mean([r["test_auroc_flipped"] for r in fold_rows])),
        }
        print(f"{layer:16s} mean {report[layer]['mean_train_auroc']:12.4f} "
              f"{report[layer]['mean_test_auroc']:11.4f}\n", flush=True)

    (OUT / "transfer_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved {OUT / 'transfer_check.json'}", flush=True)


if __name__ == "__main__":
    main()
