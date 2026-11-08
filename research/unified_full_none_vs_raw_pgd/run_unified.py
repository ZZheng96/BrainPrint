from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.open_set_improvement.core import (
    COV_COLUMNS,
    ROOT,
    TaskSpecificModel,
    confidence_values,
    resolve_cache,
    select_operating_point,
)
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout
from src.raw_eeg_defense.run import competition_metrics, open_set_predictions

OUT = ROOT / "research" / "unified_full_none_vs_raw_pgd"
CACHE_VERSION = "unified-full-none-vs-raw-pgd-v1"
ATTACKS = (("untargeted_snr20", 20.0, False), ("untargeted_snr30", 30.0, False), ("targeted_snr20", 20.0, True))


def predict_full(model, x, window, tasks, indices):
    return model.predict_scores(np.asarray(x[indices]), np.asarray(window[indices]), tasks[indices])["full"]


def report(true, clean, adv, unknown, classes, point):
    clean_pred, clean_conf = open_set_predictions(clean, classes, point)
    adv_pred, _ = open_set_predictions(adv, classes, point)
    unknown_pred, unknown_conf = open_set_predictions(unknown, classes, point)
    result = competition_metrics(true, clean_pred, adv_pred, unknown_pred, clean_conf, unknown_conf)
    result["clean_f1"] = result["clean_macro_f1_with_reject"]
    result["n_known"] = int(len(true))
    result["n_unknown"] = int(len(unknown))
    result["score_name"] = point["score_name"]
    result["threshold"] = float(point["threshold"])
    return result


def weighted_aggregate(rows):
    known = sum(r["n_known"] for r in rows)
    unknown = sum(r["n_unknown"] for r in rows)
    out = {"n_known": known, "n_unknown": unknown, "n_folds": len(rows)}
    for name in ("C", "RA", "ASafeR", "clean_f1", "clean_macro_f1_with_reject", "adversarial_macro_f1_with_reject"):
        out[name] = float(np.average([r[name] for r in rows], weights=[r["n_known"] for r in rows]))
    for name in ("U", "unknown_f1"):
        out[name] = float(np.average([r[name] for r in rows], weights=[r["n_unknown"] for r in rows]))
    out["A"] = 0.7 * out["RA"] + 0.3 * out["ASafeR"]
    out["official_score"] = float(100 * out["C"] ** 0.40 * out["A"] ** 0.35 * out["U"] ** 0.25) if min(out["C"], out["A"], out["U"]) > 0 else 0.0
    out["official_score_rounded"] = round(out["official_score"], 2)
    for name in ("unknown_auroc",):
        out[name] = float(np.average([r[name] for r in rows], weights=[r["n_known"] + r["n_unknown"] for r in rows]))
    out["thresholds"] = [r["threshold"] for r in rows]
    out["score_names"] = [r["score_name"] for r in rows]
    return out


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA")
    device = torch.device("cuda")
    cache = resolve_cache()
    meta = pd.read_csv(cache / "train_metadata.csv")
    x = np.load(cache / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS]
    window = np.load(cache / "train_window2_features.npy", mmap_mode="r").reshape(len(meta), 2, -1)[:, :, COV_COLUMNS]
    subjects, tasks = meta["subject"].to_numpy(), meta["task"].to_numpy()
    outer_time, inner_time = forward_holdout(meta), inner_holdout(meta)
    all_subjects = np.unique(subjects)
    outer_groups = np.array_split(np.random.default_rng(42).permutation(all_subjects), 5)
    attack_dir = cache / "raw_eeg_robust" / "full"
    folds, attack_rows = [], {name: [] for name, _, _ in ATTACKS}
    for oi, unknown_subjects in enumerate(outer_groups):
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        outer_known = np.isin(subjects, known_subjects)
        outer_fit = outer_known & ~outer_time
        inner_known_scores, inner_unknown_scores = [], []
        for inner_unknown in np.array_split(np.random.default_rng(100 + oi).permutation(known_subjects), 2):
            enrolled = np.setdiff1d(known_subjects, inner_unknown)
            inner_known = np.isin(subjects, enrolled)
            inner_fit = inner_known & ~outer_time & ~inner_time
            eval_known = np.flatnonzero(inner_known & inner_time)
            eval_unknown = np.flatnonzero(np.isin(subjects, inner_unknown) & ~outer_time)
            model = TaskSpecificModel(device).fit(x, window, subjects, tasks, inner_fit)
            inner_known_scores.append(predict_full(model, x, window, tasks, eval_known))
            inner_unknown_scores.append(predict_full(model, x, window, tasks, eval_unknown))
            del model
            torch.cuda.empty_cache()
        point = select_operating_point(inner_known_scores, inner_unknown_scores)
        eval_known = np.flatnonzero(outer_known & outer_time)
        eval_unknown = np.flatnonzero(np.isin(subjects, unknown_subjects))
        model = TaskSpecificModel(device).fit(x, window, subjects, tasks, outer_fit)
        clean = predict_full(model, x, window, tasks, eval_known)
        unknown = predict_full(model, x, window, tasks, eval_unknown)
        fold = {"outer_fold": oi + 1, "unknown_subjects": unknown_subjects.tolist(), "operating_point": point, "attacks": {}}
        for name, snr, targeted in ATTACKS:
            path = attack_dir / f"outer{oi + 1}_eval_{name}.npz"
            with np.load(path, allow_pickle=False) as saved:
                adv_full, adv_window = saved["full"], saved["window"]
            adv = model.predict_scores(adv_full, adv_window, tasks[eval_known])["full"]
            row = report(subjects[eval_known], clean, adv, unknown, model.classes, point)
            row["attack"] = name
            row["model"] = "full_none_global"
            fold["attacks"][name] = row
            attack_rows[name].append(row)
        folds.append(fold)
        del model
        torch.cuda.empty_cache()
        print(f"fold {oi + 1}/5 complete", flush=True)
    full_aggregate = {name: weighted_aggregate(rows) for name, rows in attack_rows.items()}
    raw_path = ROOT / "research" / "raw_eeg_robust" / "results.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_aggregate = raw["attacks"]
    rows = []
    for method, source in (("full_none_global", full_aggregate), ("adversarial_ridge", raw_aggregate)):
        for attack, values in source.items():
            if method == "adversarial_ridge":
                values = values[method]
            row = {"model": method, "attack": attack}
            row.update({k: values.get(k) for k in ("C", "RA", "ASafeR", "A", "U", "unknown_auroc", "unknown_f1", "official_score", "official_score_rounded", "n_known", "n_unknown", "n_folds")})
            row["clean_f1"] = values.get("clean_f1", values.get("clean_macro_f1_with_reject"))
            rows.append(row)
    with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result = {"protocol": {"version": CACHE_VERSION, "device": torch.cuda.get_device_name(0), "torch": torch.__version__, "data": "Train Session1; subject 5-fold; no Test labels/session", "threshold": "inner holdout 5th percentile, global score operating point", "full_none_global": "TaskSpecificModel full covariance Ridge score; none alignment; no target z-score; no Session rule", "adversarial_ridge": "existing raw-PGD five-fold result; clean+within-fit raw-PGD augmentation and Ridge refit", "attacks": [dict(name=n, snr_db=s, targeted=t, steps=10) for n,s,t in ATTACKS]}, "full_none_global": {"folds": folds, "attacks": full_aggregate}, "adversarial_ridge": {"source": str(raw_path), "attacks": raw_aggregate}, "metrics_csv": str(OUT / "metrics.csv")}
    (OUT / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# full_none_global vs raw-PGD 统一评估", "", "## 口径", "", "- 数据仅使用 Train Session1；五折按 subject 划分，阈值仅由 inner holdout 选择；未使用 Test 标签、Test Session 或 Session 规则。", "- `full_none_global`：完整协方差特征的 full Ridge 分支，none 对齐、全局 operating point。", "- `adversarial_ridge`：复用 `src/raw_eeg_robust/run.py` 已完成的 raw-PGD 五折结果：clean + 当前拟合折内 raw-PGD 增强并重拟合 Ridge。两者的攻击缓存、折划分和官方指标公式一致；raw-PGD 数值来源为已有真实结果缓存，未伪造或覆盖。", "", "## 五折加权汇总", "", "| 模型 | 攻击 | clean C | clean F1 | RA | ASafeR | A | U | unknown AUROC(AUC) | unknown F1 | 官方 Score |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        values = dict(r)
        for key in ("C", "clean_f1", "RA", "ASafeR", "A", "U", "unknown_auroc", "unknown_f1", "official_score"):
            values[key] = float(values[key]) if values[key] is not None else float("nan")
        lines.append("| {model} | {attack} | {C:.4f} | {clean_f1:.4f} | {RA:.4f} | {ASafeR:.4f} | {A:.4f} | {U:.4f} | {unknown_auroc:.4f} | {unknown_f1:.4f} | {official_score:.2f} |".format(**values))
    lines += ["", "`A = 0.7×RA + 0.3×ASafeR`; 官方 Score = `100×C^0.40×A^0.35×U^0.25`. clean F1 为含拒识的宏 F1；unknown F1 为未知检测 F1。", "", "## 运行", "", "`py -3.11 research/unified_full_none_vs_raw_pgd/run_unified.py`", "", "结果文件：`metrics.json`、`metrics.csv`、本报告。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "metrics": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
