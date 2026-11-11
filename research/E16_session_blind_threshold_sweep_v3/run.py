"""Train-only threshold sensitivity; Test sessions are loaded after all CSVs exist."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from research.session_blind_calibrated_v2.run import load_features
from src.open_set_improvement.core import COV_COLUMNS, TaskSpecificModel, confidence_values, resolve_cache

SOURCE = ROOT / "research" / "session_blind_calibrated_v2"
OUTPUT = Path(__file__).resolve().parent
TARGETS = (0.99, 0.995, 0.999, 1.0)
RECOMMENDED = "accept_0p995"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def threshold_for(confidence, target):
    values, counts = np.unique(confidence, return_counts=True)
    acceptance = 1 - (np.cumsum(counts) - counts) / len(confidence)
    return float(values[np.argmin(np.abs(acceptance - target))])


def rejection_summary(prediction):
    return {"n": len(prediction), "rejected": int(np.sum(prediction == 0)),
            "rejection_rate": float(np.mean(prediction == 0))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    output = parser.parse_args().output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(p.name != "run.py" for p in output.iterdir()):
        raise FileExistsError("Output directory must contain only run.py; never overwrite artifacts")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(42)
    source_metrics = json.loads((SOURCE / "metrics.json").read_text(encoding="utf-8"))
    if source_metrics["protocol"]["cov_columns"] != COV_COLUMNS.tolist():
        raise ValueError("v2 covariance columns differ")
    if source_metrics["protocol"]["variant"] != "fused":
        raise ValueError("Expected v2 fused scores")
    score_name = source_metrics["candidate"]["score_name"]
    cache = resolve_cache()
    # Whitelist: no session column is loaded for fitting, inference or calibration.
    train = pd.read_csv(cache / "train_metadata.csv", usecols=["subject", "task"])
    test = pd.read_csv(cache / "test_metadata.csv", usecols=["sample", "task"])
    subjects, tasks = train["subject"].to_numpy(), train["task"].to_numpy()
    classes = np.unique(subjects)
    oof = np.load(SOURCE / "train_oof_scores.npy", allow_pickle=False)
    if oof.shape != (len(train), len(classes)) or not np.isfinite(oof).all():
        raise ValueError("Invalid v2 OOF shape or nonfinite scores")
    confidence = confidence_values(oof)[score_name]
    specifications = [("accept_" + str(t).replace(".", "p"), t, threshold_for(confidence, t),
                       "closest attainable acceptance, ties retained (v2 rule)") for t in TARGETS]
    specifications += [(name, None, float(np.quantile(confidence, q, method="linear")),
                        f"Train OOF quantile q={q}; numpy linear interpolation")
                       for name, q in (("q01", 0.01), ("q001", 0.001))]
    x, w = load_features(cache, "train", len(train))
    tx, tw = load_features(cache, "test", len(test))
    model = TaskSpecificModel(torch.device("cuda")).fit(
        x, w, subjects, tasks, np.ones(len(train), dtype=bool))
    if not np.array_equal(model.classes, classes):
        raise ValueError("Model class order differs from v2 OOF order")
    scores = model.predict_scores(tx, tw, test["task"].to_numpy())["fused"]
    test_confidence = confidence_values(scores)[score_name]
    if not np.isfinite(test_confidence).all():
        raise ValueError("Nonfinite Test confidence")
    identities = model.classes[scores.argmax(axis=1)]
    template = pd.read_csv(ROOT / "Brainprint_Competition" / "test.csv")
    if test["sample"].duplicated().any() or template["Sample"].duplicated().any():
        raise ValueError("Duplicate Sample keys")
    mapping = template["Sample"].map(pd.Series(np.arange(len(test)), index=test["sample"]))
    if mapping.isna().any() or len(template) != len(test):
        raise ValueError("Sample alignment is not bijective")
    rows = mapping.to_numpy(dtype=int)
    predictions, candidates = {}, []
    for name, target, threshold, rule in specifications:
        prediction = identities.copy()
        prediction[test_confidence < threshold] = 0
        predictions[name] = prediction
        oof_prediction = classes[oof.argmax(axis=1)].copy()
        oof_prediction[confidence < threshold] = 0
        candidate = template.copy()
        candidate["Label"] = prediction[rows]
        filename = f"test_candidate_{name}.csv"
        with (output / filename).open("x", encoding="utf-8", newline="") as handle:
            candidate.to_csv(handle, index=False)
        candidates.append({"name": name, "file": filename, "target_known_acceptance": target,
                           "threshold": threshold, "threshold_rule": rule,
                           "oof_known_acceptance": float(np.mean(confidence >= threshold)),
                           "oof_identity_macro_F1": float(f1_score(subjects, oof_prediction, average="macro", zero_division=0)),
                           "oof_detection_AUC": None, "test": rejection_summary(prediction),
                           "test_AUC": None, "test_F1": None})
    # All six candidate CSVs are frozen before session is first read.
    audit = pd.read_csv(cache / "test_metadata.csv", usecols=["sample", "session", "task"])
    audit = audit.set_index("sample").loc[test["sample"]].reset_index()
    for candidate in candidates:
        prediction = predictions[candidate["name"]]
        candidate["post_generation_audit"] = {}
        for columns, key in ((["session"], "session"), (["task"], "task"),
                             (["session", "task"], "session_task")):
            grouped = {}
            for group, indices in audit.groupby(columns, dropna=False).indices.items():
                label = "/".join(map(str, group)) if isinstance(group, tuple) else str(group)
                grouped[label] = rejection_summary(prediction[indices])
            candidate["post_generation_audit"][key] = grouped
    baseline = source_metrics["outer_unweighted_fold_mean"]
    result = {
        "protocol": {"source": str(SOURCE), "source_metrics_sha256": digest(SOURCE / "metrics.json"),
                     "source_oof_sha256": digest(SOURCE / "train_oof_scores.npy"),
                     "v2_protocol": source_metrics["protocol"], "score_name": score_name,
                     "variant": "fused", "cov_columns": COV_COLUMNS.tolist(),
                     "gpu": torch.cuda.get_device_name(0), "cache": str(cache),
                     "session_used_for_inference_or_threshold": False,
                     "inference_metadata_columns": ["task"], "sample_usage": "output alignment only",
                     "session_first_read": "after all six candidate CSVs written",
                     "accept_rule": "confidence >= threshold; rejection label 0",
                     "target_1_definition": "minimum OOF confidence, not negative infinity; Test may still be rejected"},
        "v2_outer_validation_reference": baseline,
        "v2_outer_folds_reference": [{"fold": f["outer_fold"], "outer": f["outer"]} for f in source_metrics["folds"]],
        "v2_candidate_reference": source_metrics["candidate"],
        "candidates": candidates,
        "recommendation": {"name": RECOMMENDED,
                           "basis": "Predeclared Train-only compromise: 99.5% known acceptance; not selected using Test/session audit",
                           "validated_as_optimal": False},
        "limitations": ["No Test ground truth: Test AUC/F1 unavailable; rejection is not U",
                        "Known-only OOF cannot estimate detection AUC or unknown rejection",
                        "v2 outer metrics are references, not validation of the six new thresholds",
                        "Full-fit versus OOF training-size shift remains; no session adaptation",
                        "Formal submission and existing experiments unchanged"]}
    lines = ["# Session-blind threshold sweep v3", "", "## 推荐与边界", "",
             "推荐候选：`test_candidate_accept_0p995.csv`。这是预先固定的 Train-only 折中（目标已知接受率99.5%），不是根据 Test 总拒识或 Session 审计选出的最优阈值。正式提交未修改。",
             "", "## v2 严格外层验证基准（沿用，非本轮阈值的验证结果）", "",
             f"- AUC：{baseline['unknown_auroc']:.6f}；身份 macro-F1：{baseline['macro_f1']:.6f}。",
             f"- 检测 macro-F1：{baseline['detection_macro_f1']:.6f}；未知 F1：{baseline['unknown_f1']:.6f}；U：{baseline['U']:.6f}。",
             "- v2 外层 subject 五折、已知末20% trial 留出；内层伪未知仅使用外层注册用户。此轮不重新训练外层，不将基准冒充新阈值成绩。",
             "", "## 推理与阈值协议", "",
             f"- 固定 v2 置信指标 `{score_name}`；复用 v2 五个连续 trial 块 OOF，不使用 Test 分布选择阈值。",
             "- Train/Test 原始 full/window 特征先取同一 COV_COLUMNS，再把相邻窗口重排为 (trial, 2, 1026)，直接复用 v2 load_features。全量 TaskSpecificModel(alpha=100) 在 CUDA 上计算 fused scores。",
             "- 推理前读取的 Train 元数据仅 subject/task，Test 仅 sample/task；模型接口只有 EEG 特征及 task，sample 只对齐提交顺序。确认推理不读取 session。",
             "- 所有六个候选 CSV 写出后才首次读取 session，Session1/Session2/task 仅作事后审计，不参与阈值、候选选择或后处理。",
             "- 目标阈值沿用 v2：取最接近目标的可达接受率，保留相同分数；q01=OOF 第1百分位，q001=第0.1百分位，采用 linear 插值。接受条件为 confidence >= threshold。",
             "- 目标1.0对应 OOF 最小置信度，不代表 Test 全接受。已知-only OOF 无未知标签，无法计算检测 AUC/U。",
             "", "## 阈值敏感性", "",
             "|候选|阈值|Train OOF known acceptance|OOF身份 macro-F1|Test拒识/总数|Test总拒识率|",
             "|---|---:|---:|---:|---:|---:|"]
    for c in candidates:
        t = c["test"]
        lines.append(f"|{c['name']}|{c['threshold']:.10g}|{c['oof_known_acceptance']:.6f}|{c['oof_identity_macro_F1']:.6f}|{t['rejected']}/{t['n']}|{t['rejection_rate']:.6%}|")
    lines += ["", "## Session1/Session2 与 task 事后拒识审计", "",
              "下表只是无标签分组拒识统计，不是准确率、AUC、F1 或未知拒识率 U；不从 Session 推断真实身份。", "",
              "|候选|审计维度|组|拒识/总数|拒识率|", "|---|---|---|---:|---:|"]
    for c in candidates:
        for dimension, groups in c["post_generation_audit"].items():
            for group, stats in groups.items():
                lines.append(f"|{c['name']}|{dimension}|{group}|{stats['rejected']}/{stats['n']}|{stats['rejection_rate']:.6%}|")
    lines += ["", "## 局限与复现", "", "Test 无真实标签，Test AUC/F1 均不可计算。降低阈值可能减少已知误拒，也可能增加未知误接受；此轮无法估计这种权衡的真实收益。不能把 Test 审计当验证指标。",
              "全量拟合与 OOF 折模型样本量不同，分数分布仍可能偏移；v2 外层基准不证明新候选更好。未重新评估对抗鲁棒性或跨 Session 泛化。",
              "", "运行：`py -3 research/E16_session_blind_threshold_sweep_v3/run.py --output-dir <新的空目录>`。依赖现有 numpy/pandas/scikit-learn/PyTorch CUDA；metrics.json 保留来源哈希、v2 外层各折及完整审计。现有输出一律不覆盖。"]
    with (output / "metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    with (output / "report.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output), "baseline": baseline, "candidates": candidates}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
