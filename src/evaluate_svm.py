"""评估非线性 RBF-SVM 是否能解决线性边界的持续性误判。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from analyze_train_errors import summarize_predictions
from run_baseline import FEATURE_SETS

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "BrainPrint_cache"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["blocked_oof", "forward"], default="forward")
    parser.add_argument("--c", type=float, default=10.0)
    parser.add_argument("--features", choices=list(FEATURE_SETS), default="cov")
    args = parser.parse_args()

    x = np.load(CACHE / "train_features.npy", mmap_mode="r")
    columns = FEATURE_SETS[args.features]
    metadata = pd.read_csv(CACHE / "train_metadata.csv")
    subjects = metadata["subject"].to_numpy()
    tasks = metadata["task"].to_numpy()
    trials = metadata["trial"].to_numpy()
    folds = ((trials - 1) // 10) % 5
    max_trial = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    selected = np.ones(len(metadata), dtype=bool) if args.protocol == "blocked_oof" else trials > 0.8 * max_trial
    split_values = range(5) if args.protocol == "blocked_oof" else [-1]
    scores = np.full((len(metadata), 20), np.nan, dtype=np.float32)

    for split in split_values:
        validation = folds == split if args.protocol == "blocked_oof" else selected
        fit = ~validation
        for task in sorted(metadata["task"].unique()):
            fit_indices = np.flatnonzero(fit & (tasks == task))
            val_indices = np.flatnonzero(validation & (tasks == task))
            model = make_pipeline(
                StandardScaler(),
                SVC(C=args.c, kernel="rbf", gamma="scale", decision_function_shape="ovr", cache_size=2000),
            )
            model.fit(x[fit_indices][:, columns], subjects[fit_indices])
            decision = model.decision_function(x[val_indices][:, columns])
            scores[np.ix_(val_indices, model.classes_.astype(int) - 1)] = decision
            print(f"split={split} {task}: fit={len(fit_indices)} val={len(val_indices)}", flush=True)

    indices = np.flatnonzero(selected)
    labels = scores[indices].argmax(axis=1) + 1
    output = metadata.loc[selected, ["path", "subject", "task", "task_folder", "trial"]].copy()
    output["predicted_label"] = labels
    output["correct"] = labels == subjects[indices]
    tag = f"svm_{args.features}_c{args.c:g}"
    output.to_csv(CACHE / f"train_{args.protocol}_predictions_{tag}.csv", index=False)
    np.save(CACHE / f"train_{args.protocol}_scores_{tag}.npy", scores)
    report = summarize_predictions(output, args.protocol)
    report["config"] = vars(args)
    path = ROOT / "research" / f"train_error_summary_{tag}_{args.protocol}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"accuracy": report["accuracy"], "errors": report["errors"], "per_subject": report["per_subject"], "confusion_pairs": report["confusion_pairs"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
