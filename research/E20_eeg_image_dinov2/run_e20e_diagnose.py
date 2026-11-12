"""诊断：当前 E20 各特征在训练集闭集上的错误分布，以及拒识阈值对已知接受率的影响。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.open_set_improvement.core import COV_COLUMNS
from src.open_set_improvement.run_nested import forward_holdout

CACHE = ROOT / "BrainPrint_cache"
OUT = Path(__file__).resolve().parent


def margin(scores):
    top = np.partition(scores, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def main():
    metadata = pd.read_csv(CACHE / "train_metadata.csv", usecols=["subject", "task", "trial"])
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    forward = forward_holdout(metadata)
    all_subjects = np.unique(subjects)

    cov = np.asarray(
        np.load(CACHE / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS], dtype=np.float32
    )
    image = np.load(CACHE / "eeg_image_dinov2" / "train_covariance5_dinov2.npy").astype(np.float32)
    fused = np.concatenate([image, cov], axis=1)

    report = {}
    for name, features in (("covariance_only", cov), ("covariance5_fused", fused)):
        groups = np.array_split(np.random.default_rng(42).permutation(all_subjects), 5)
        rows = []
        for fold, unknown_subjects in enumerate(groups, 1):
            known = np.isin(subjects, np.setdiff1d(all_subjects, unknown_subjects))
            fit = known & ~forward
            eval_idx = np.flatnonzero(known & forward)
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
            model.fit(features[fit], subjects[fit])
            scores = model.decision_function(features[eval_idx])
            pred = model.classes_[scores.argmax(axis=1)]
            conf = margin(scores)
            correct = pred == subjects[eval_idx]
            # 错误按任务分布
            error_tasks = pd.Series(tasks[eval_idx][~correct]).value_counts().to_dict()
            # 正确样本的置信度分位
            rows.append({
                "fold": fold,
                "n": int(len(eval_idx)),
                "errors": int((~correct).sum()),
                "accuracy": float(correct.mean()),
                "error_by_task": {str(k): int(v) for k, v in error_tasks.items()},
                "conf_p05": float(np.quantile(conf, 0.05)),
                "conf_p01": float(np.quantile(conf, 0.01)),
                "conf_min": float(conf.min()),
                "conf_median": float(np.median(conf)),
            })
        report[name] = rows
        total_err = sum(r["errors"] for r in rows)
        total_n = sum(r["n"] for r in rows)
        print(f"\n{name}: 总错误 {total_err}/{total_n} = {total_err/total_n:.4f}")
        for r in rows:
            print(f"  fold{r['fold']}: acc={r['accuracy']:.4f} err={r['errors']} "
                  f"by_task={r['error_by_task']} conf_p05={r['conf_p05']:.3f}")

    (OUT / "e20e_error_diagnosis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved {OUT / 'e20e_error_diagnosis.json'}")


if __name__ == "__main__":
    main()
