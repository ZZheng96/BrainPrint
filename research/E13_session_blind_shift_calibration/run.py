"""严格嵌套五折：direct/base 与训练折内 OOF conditional calibration。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

from src.open_set_improvement.core import (
    COV_COLUMNS, TaskSpecificModel, confidence_values, resolve_cache,
)
from src.session_blind_candidate.shift_calibration import (
    fit_conditional_thresholds, score_diagnostics,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "session_blind_shift_calibration"


def holdout(metadata: pd.DataFrame, low: float, high: float | None = None) -> np.ndarray:
    maximum = metadata.groupby(["subject", "task"])["trial"].transform("max").to_numpy()
    trial = metadata["trial"].to_numpy()
    if high is None:
        return trial > low * maximum
    return (trial > low * maximum) & (trial <= high * maximum)


def evaluate(true: np.ndarray, pred: np.ndarray, known: np.ndarray, conf: np.ndarray) -> dict:
    detection_true = known.astype(int)
    return {
        "n_known": int(known.sum()), "n_unknown": int((~known).sum()),
        "C": float(np.mean(pred[known] == true[known])),
        "U": float(np.mean(pred[~known] == 0)),
        "unknown_auc": float(roc_auc_score(detection_true, conf)),
        "unknown_f1": float(f1_score(detection_true, pred != 0, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "known_acceptance": float(np.mean(pred[known] != 0)),
        "official_score_proxy": float(100 * np.mean(pred[known] == true[known]) ** 0.40 * np.mean(pred[~known] == 0) ** 0.25),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("本实验要求 CUDA GPU")
    device = torch.device("cuda")
    cache = resolve_cache()
    metadata = pd.read_csv(cache / "train_metadata.csv")
    subjects, tasks = metadata.subject.to_numpy(), metadata.task.to_numpy()
    x = np.asarray(np.load(cache / "train_features.npy", mmap_mode="r")[:, COV_COLUMNS])
    w = np.asarray(np.load(cache / "train_window2_features.npy", mmap_mode="r")[:, COV_COLUMNS]).reshape(len(metadata), 2, -1)
    forward = holdout(metadata, 0.8)
    inner = holdout(metadata, 0.6, 0.8)
    all_subjects = np.unique(subjects)
    outer_groups = np.array_split(np.random.default_rng(42).permutation(all_subjects), 5)
    fold_rows, fold_details = [], []

    for fold, unknown_subjects in enumerate(outer_groups, 1):
        known_subjects = np.setdiff1d(all_subjects, unknown_subjects)
        outer_known = np.isin(subjects, known_subjects)
        outer_fit = outer_known & ~forward
        inner_known_scores = {"direct": [], "base": []}
        inner_diag = []
        inner_groups = np.array_split(np.random.default_rng(100 + fold - 1).permutation(known_subjects), 2)
        for inner_unknown in inner_groups:
            enrolled = np.setdiff1d(known_subjects, inner_unknown)
            fit = np.isin(subjects, enrolled) & ~forward & ~inner
            eval_known = np.isin(subjects, enrolled) & inner
            eval_unknown = np.isin(subjects, inner_unknown) & ~forward
            eval_idx = np.flatnonzero(eval_known | eval_unknown)
            model = TaskSpecificModel(device).fit(x, w, subjects, tasks, fit)
            scored = model.predict_scores(x[eval_idx], w[eval_idx], tasks[eval_idx])
            n_known = int(eval_known[eval_idx].sum())
            direct, base = scored["full"], scored["base"]
            inner_known_scores["direct"].append(direct[:n_known])
            inner_known_scores["base"].append(base[:n_known])
            inner_diag.append(score_diagnostics(direct[:n_known], base[:n_known]))
            del model
            torch.cuda.empty_cache()

        direct_known = np.concatenate(inner_known_scores["direct"])
        base_known = np.concatenate(inner_known_scores["base"])
        direct_threshold = float(np.quantile(confidence_values(direct_known)["max_logit"], 0.05))
        base_threshold = float(np.quantile(confidence_values(base_known)["max_logit"], 0.05))
        diagnostics = {key: np.concatenate([d[key] for d in inner_diag]) for key in inner_diag[0]}
        calibration = fit_conditional_thresholds(base_known, diagnostics, "max_logit", n_bins=3, quantile=0.05)

        outer_eval_known = np.flatnonzero(outer_known & forward)
        outer_eval_unknown = np.flatnonzero(np.isin(subjects, unknown_subjects))
        eval_idx = np.concatenate([outer_eval_known, outer_eval_unknown])
        model = TaskSpecificModel(device).fit(x, w, subjects, tasks, outer_fit)
        scored = model.predict_scores(x[eval_idx], w[eval_idx], tasks[eval_idx])
        true = np.concatenate([subjects[outer_eval_known], np.zeros(len(outer_eval_unknown), dtype=int)])
        known = np.r_[np.ones(len(outer_eval_known), dtype=bool), np.zeros(len(outer_eval_unknown), dtype=bool)]
        detail = {"fold": fold, "unknown_subjects": unknown_subjects.tolist(), "methods": {}}
        row = {"fold": fold, "unknown_subjects": ",".join(map(str, unknown_subjects))}
        for method, values, threshold in (("direct", scored["full"], direct_threshold), ("base", scored["base"], base_threshold)):
            conf = confidence_values(values)["max_logit"]
            pred = model.classes[values.argmax(axis=1)].copy(); pred[conf < threshold] = 0
            metrics = evaluate(true, pred, known, conf)
            detail["methods"][method] = {"threshold": threshold, **metrics}
            row.update({f"{method}_{k}": v for k, v in metrics.items() if k not in ("n_known", "n_unknown")})
            row[f"{method}_threshold"] = threshold
        conf = confidence_values(scored["base"])["max_logit"]
        pred = model.classes[scored["base"].argmax(axis=1)].copy()
        pred[conf < calibration.threshold(score_diagnostics(scored["full"], scored["base"])["quality"])] = 0
        metrics = evaluate(true, pred, known, conf)
        detail["methods"]["conditional"] = {"thresholds": calibration.thresholds.tolist(), "edges": calibration.edges.tolist(), **metrics}
        row.update({f"conditional_{k}": v for k, v in metrics.items() if k not in ("n_known", "n_unknown")})
        fold_rows.append(row); fold_details.append(detail)
        del model; torch.cuda.empty_cache()
        print(f"fold={fold} complete", flush=True)

    frame = pd.DataFrame(fold_rows); frame.to_csv(OUT / "metrics.csv", index=False)
    summary = {}
    keys = ("C", "U", "unknown_auc", "unknown_f1", "macro_f1", "known_acceptance", "official_score_proxy")
    for method in ("direct", "base", "conditional"):
        summary[method] = {key: float(frame[f"{method}_{key}"].mean()) for key in keys}
    result = {"protocol": {"outer": "5-fold subject holdout; registered outer evaluation last 20% trials; held-out subjects unknown", "inner": "2-fold pseudo-unknown calibration; model fit only on inner training trials", "direct": "full covariance Ridge, max_logit 5% threshold from inner known predictions", "base": "equal full + two-window Ridge, max_logit 5% threshold from inner known predictions", "conditional": "base max_logit threshold stratified by score-consistency quality into 3 bins; thresholds from inner known OOF only", "gpu": torch.cuda.get_device_name(0), "no_test_labels": True, "official_formula_proxy": "100*C^0.40*U^0.25; RA/ASafeR unavailable in this clean/unknown experiment"}, "summary": summary, "folds": fold_details, "test_auc": None, "test_f1": None, "test_metrics_status": "不可计算：Test 无标签；本实验不生成 Test CSV"}
    (OUT / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Session-blind shift calibration：严格嵌套五折结果", "", "## 协议", "独立重跑模型并使用 CUDA。每个外层折的 direct/base 分数和 conditional 阈值只由外层训练折的内层预测拟合；外层评估标签不参与任何选择。conditional 为 base max_logit 按训练折质量信号三桶分层的 5% 分位阈值。", "", "## 五折宏平均", "| 方法 | C | U | unknown AUC | unknown F1 | macro-F1 | known acceptance | 官方公式代理分 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method, vals in summary.items():
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} |".format(method, vals["C"], vals["U"], vals["unknown_auc"], vals["unknown_f1"], vals["macro_f1"], vals["known_acceptance"], vals["official_score_proxy"]))
    lines += ["", "## Test 说明", "Test AUC/F1 不可计算：测试集没有公开标签。本轮未生成 Test CSV，也未改变正式提交文件。", "", "## 门槛判断", "本实验只覆盖 clean/unknown 维度，官方代理分缺少 RA/ASafeR，不能替代官方总分；conditional 未达到提交所需的完整验收证据，因此本轮不通过，不生成候选 CSV。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
