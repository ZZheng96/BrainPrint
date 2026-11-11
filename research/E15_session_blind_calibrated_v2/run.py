"""Session-blind nested calibration with freshly computed aligned-window OOF scores."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

from src.open_set_improvement.core import (
    COV_COLUMNS, SCORE_NAMES, TaskSpecificModel, confidence_values,
    open_set_report, resolve_cache,
)
from src.open_set_improvement.run_nested import forward_holdout, inner_holdout

OUTPUT = Path(__file__).resolve().parent
VARIANT = "fused"
TARGET = 0.98


def load_features(cache, split, n_rows):
    full = np.load(cache / f"{split}_features.npy", mmap_mode="r")
    windows = np.load(cache / f"{split}_window2_features.npy", mmap_mode="r")
    if full.ndim != 2 or windows.ndim != 2:
        raise ValueError("Expected two-dimensional original feature caches")
    if len(full) != n_rows or len(windows) != 2 * n_rows:
        raise ValueError("Window cache must contain two adjacent rows per metadata trial")
    # Select the identical original columns BEFORE grouping the two window rows.
    x = np.asarray(full[:, COV_COLUMNS])
    w = np.asarray(windows[:, COV_COLUMNS]).reshape(n_rows, 2, len(COV_COLUMNS))
    if not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError("Nonfinite EEG features")
    return x, w


def threshold_for(confidence):
    # Choose the attainable acceptance closest to 0.98, including tied scores.
    values, counts = np.unique(confidence, return_counts=True)
    acceptance = 1 - (np.cumsum(counts) - counts) / len(confidence)
    return float(values[np.argmin(np.abs(acceptance - TARGET))])


def detection_report(known, unknown, threshold):
    y = np.r_[np.ones(len(known), dtype=int), np.zeros(len(unknown), dtype=int)]
    scores = np.r_[known, unknown]
    accepted = scores >= threshold
    return {
        "threshold": threshold,
        "known_acceptance": float(np.mean(known >= threshold)),
        "U": float(np.mean(unknown < threshold)),
        "AUC": float(roc_auc_score(y, scores)),
        "detection_macro_F1": float(f1_score(y, accepted, average="macro", zero_division=0)),
        "unknown_F1": float(f1_score(y, accepted, pos_label=0, zero_division=0)),
        "n_known": len(known), "n_unknown": len(unknown),
    }


def calibrate(x, w, subjects, tasks, metadata, enrolled, seed, device):
    forward, inner = forward_holdout(metadata), inner_holdout(metadata)
    known = {name: [] for name in SCORE_NAMES}
    unknown = {name: [] for name in SCORE_NAMES}
    splits = []
    for pseudo in np.array_split(np.random.default_rng(seed).permutation(enrolled), 2):
        registered = np.isin(subjects, np.setdiff1d(enrolled, pseudo))
        fit = registered & ~forward & ~inner
        ki = np.flatnonzero(registered & inner)
        ui = np.flatnonzero(np.isin(subjects, pseudo) & ~forward)
        indices = np.r_[ki, ui]
        assert not fit[indices].any()
        model = TaskSpecificModel(device).fit(x, w, subjects, tasks, fit)
        confidence = confidence_values(model.predict_scores(x[indices], w[indices], tasks[indices])[VARIANT])
        for name in SCORE_NAMES:
            known[name].append(confidence[name][:len(ki)])
            unknown[name].append(confidence[name][len(ki):])
        splits.append({"pseudo_unknown_subjects": pseudo.tolist(), "n_fit": int(fit.sum()),
                       "n_known": len(ki), "n_unknown": len(ui)})
        del model
    candidates = []
    for name in SCORE_NAMES:
        kc, uc = np.concatenate(known[name]), np.concatenate(unknown[name])
        candidates.append({"score_name": name, **detection_report(kc, uc, threshold_for(kc))})
    selected = max(candidates, key=lambda r: (r["detection_macro_F1"], r["unknown_F1"], r["AUC"]))
    return {"selected": selected, "candidates": candidates, "splits": splits}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = ("metrics.json", "report.md", "test_candidate.csv", "train_oof_scores.npy")
    if any((output / name).exists() for name in names):
        raise FileExistsError("Output exists; choose a new --output-dir. Existing files are never overwritten.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    torch.manual_seed(42)
    device = torch.device("cuda")
    cache = resolve_cache()
    metadata = pd.read_csv(cache / "train_metadata.csv")
    subjects, tasks = metadata["subject"].to_numpy(), metadata["task"].to_numpy()
    x, w = load_features(cache, "train", len(metadata))
    all_subjects = np.unique(subjects)
    if len(all_subjects) < 10 or np.any(all_subjects == 0):
        raise ValueError("Expected registered training subjects with nonzero identity labels")
    forward = forward_holdout(metadata)
    folds = []
    for index, held in enumerate(np.array_split(np.random.default_rng(42).permutation(all_subjects), 5)):
        enrolled = np.setdiff1d(all_subjects, held)
        calibration = calibrate(x, w, subjects, tasks, metadata, enrolled, 100 + index, device)
        selection = calibration["selected"]
        fit = np.isin(subjects, enrolled) & ~forward
        ki = np.flatnonzero(np.isin(subjects, enrolled) & forward)
        ui = np.flatnonzero(np.isin(subjects, held))
        indices = np.r_[ki, ui]
        assert not fit[indices].any()
        model = TaskSpecificModel(device).fit(x, w, subjects, tasks, fit)
        scores = model.predict_scores(x[indices], w[indices], tasks[indices])[VARIANT]
        report, _ = open_set_report(subjects[ki], scores[:len(ki)], scores[len(ki):],
                                    model.classes, selection["score_name"], selection["threshold"])
        folds.append({"outer_fold": index + 1, "unknown_subjects": held.tolist(),
                      "registered_subjects": enrolled.tolist(), "inner": calibration, "outer": report})
        print(f"outer={index + 1}: {json.dumps(report)}", flush=True)
        del model
        torch.cuda.empty_cache()

    # A separate all-Train pseudo-unknown calibration selects the confidence rule;
    # the candidate threshold is then recalibrated ONLY on fresh known OOF scores.
    final_calibration = calibrate(x, w, subjects, tasks, metadata, all_subjects, 2026, device)
    score_name = final_calibration["selected"]["score_name"]
    block = np.full(len(metadata), -1, dtype=int)
    for indices in metadata.groupby(["subject", "task"], sort=False).indices.values():
        ordered = indices[np.argsort(metadata.iloc[indices]["trial"].to_numpy(), kind="stable")]
        for fold, part in enumerate(np.array_split(ordered, 5)):
            block[part] = fold
    oof = np.full((len(metadata), len(all_subjects)), np.nan, dtype=np.float32)
    for fold in range(5):
        valid = block == fold
        model = TaskSpecificModel(device).fit(x, w, subjects, tasks, ~valid)
        if not np.array_equal(model.classes, all_subjects):
            raise ValueError("OOF training fold missing a registered subject")
        oof[valid] = model.predict_scores(x[valid], w[valid], tasks[valid])[VARIANT]
        print(f"fresh aligned-window OOF fold={fold + 1}", flush=True)
        del model
    if not np.isfinite(oof).all():
        raise RuntimeError("Incomplete OOF scores")
    oof_conf = confidence_values(oof)[score_name]
    threshold = threshold_for(oof_conf)
    oof_pred = all_subjects[oof.argmax(axis=1)].copy()
    oof_pred[oof_conf < threshold] = 0
    model = TaskSpecificModel(device).fit(x, w, subjects, tasks, np.ones(len(metadata), dtype=bool))
    test_metadata = pd.read_csv(cache / "test_metadata.csv")
    tx, tw = load_features(cache, "test", len(test_metadata))
    scores = model.predict_scores(tx, tw, test_metadata["task"].to_numpy())[VARIANT]
    prediction = model.classes[scores.argmax(axis=1)].copy()
    prediction[confidence_values(scores)[score_name] < threshold] = 0
    submission = pd.read_csv(ROOT / "Brainprint_Competition" / "test.csv")
    if test_metadata["sample"].duplicated().any() or submission["Sample"].duplicated().any():
        raise ValueError("Duplicate sample keys")
    row_indices = submission["Sample"].map(pd.Series(np.arange(len(test_metadata)), index=test_metadata["sample"]))
    if row_indices.isna().any() or len(submission) != len(test_metadata):
        raise ValueError("Test sample mapping is not one-to-one")
    submission["Label"] = prediction[row_indices.to_numpy(dtype=int)]
    with (output / "test_candidate.csv").open("x", encoding="utf-8", newline="") as handle:
        submission.to_csv(handle, index=False)
    # Session is accessed only after the predictions and candidate have been frozen.
    audit = {}
    if "session" in test_metadata:
        for session, indices in test_metadata.groupby("session", dropna=False).indices.items():
            audit[str(session)] = {"n": len(indices), "rejected": int(np.sum(prediction[indices] == 0))}
    summary = {key: float(np.mean([fold["outer"][key] for fold in folds])) for key in
               ("unknown_auroc", "macro_f1", "unknown_f1", "detection_macro_f1", "C", "U", "known_acceptance")}
    result = {
        "protocol": {"outer_folds": 5, "seed": 42, "variant": VARIANT,
                     "model": "TaskSpecificModel(alpha=100); identical full/window/task fusion in inner, outer, OOF and Test",
                     "cov_columns": COV_COLUMNS.tolist(), "known_acceptance_target": TARGET,
                     "outer": "held-out subjects entirely unknown; enrolled last 20% trials held out",
                     "inner": "outer enrolled subjects only; 2 pseudo-unknown groups; known 60-80% trials; fit first 60%; unknown first 80%",
                     "final_oof": "fresh 5 contiguous trial blocks per subject/task; both windows always stay with parent trial",
                     "session_used_for_inference_or_threshold": False,
                     "gpu": torch.cuda.get_device_name(0), "cache": str(cache)},
        "folds": folds, "outer_unweighted_fold_mean": summary,
        "final_inner_calibration": final_calibration,
        "candidate": {"score_name": score_name, "threshold": threshold,
                      "threshold_source": "fresh current-model train OOF; no legacy score cache read",
                      "oof_known_acceptance": float(np.mean(oof_conf >= threshold)),
                      "oof_identity_macro_F1": float(f1_score(subjects, oof_pred, average="macro", zero_division=0)),
                      "oof_detection_AUC": None, "oof_detection_AUC_reason": "known-only OOF has no unknown class",
                      "test_rows": len(prediction), "test_rejected": int(np.sum(prediction == 0)),
                      "session_post_generation_audit": audit,
                      "test_AUC": None, "test_F1": None, "test_metrics_reason": "No Test ground truth used"},
    }
    with (output / "train_oof_scores.npy").open("xb") as handle:
        np.save(handle, oof)
    with (output / "metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    lines = ["# Session-blind calibrated v2 实验报告", "", "## 严格外层五折结果", "",
             "以下为折间等权平均，AUC 是已知/未知检测 ROC-AUC（已知为正类）；F1 同时区分身份与检测口径。",
             "", f"- AUC：{summary['unknown_auroc']:.6f}", f"- 身份 macro-F1（含拒识0）：{summary['macro_f1']:.6f}",
             f"- 检测 macro-F1：{summary['detection_macro_f1']:.6f}", f"- 未知 F1：{summary['unknown_f1']:.6f}",
             f"- U：{summary['U']:.6f}；C：{summary['C']:.6f}；known acceptance：{summary['known_acceptance']:.6f}",
             "", "|折|内层伪未知 U|内层 AUC|内层检测 F1|内层已知接受率|外层 AUC|外层身份 F1|外层 U|", "|---|---|---|---|---|---|---|---|"]
    for fold in folds:
        inner, outer = fold["inner"]["selected"], fold["outer"]
        lines.append(f"|{fold['outer_fold']}|{inner['U']:.6f}|{inner['AUC']:.6f}|{inner['detection_macro_F1']:.6f}|{inner['known_acceptance']:.6f}|{outer['unknown_auroc']:.6f}|{outer['macro_f1']:.6f}|{outer['U']:.6f}|")
    lines += ["", "## 阈值与候选", "", f"- 固定模型融合：{VARIANT}；Train/Test full 和 window 均先取 COV_COLUMNS，再将窗口重排为 (trial,2,1026)。",
              "- 外层 subject 严格五折；内层不接触外层未知 subject 或已知末20% trial。目标已知接受率0.98，按最接近的可达接受率选阈值；置信指标仅按内层检测F1选择。",
              f"- 最终置信指标：{score_name}；阈值：{threshold:.9g}。只来自本次重算、同模型同融合的五折 train OOF，不读取任何旧分数缓存。",
              f"- 新 OOF 已知接受率：{result['candidate']['oof_known_acceptance']:.6f}；身份 macro-F1：{result['candidate']['oof_identity_macro_F1']:.6f}。已知-only OOF 无法计算检测 AUC。",
              f"- Test 候选 {len(prediction)} 行，拒识 {result['candidate']['test_rejected']} 行。推理仅使用 EEG 特征和 task，sample 仅用于恢复提交顺序。",
              "- Session 只在候选写出后做计数审计，未参与训练、阈值、选择、后处理。", "", "## 局限", "",
              "内层结果是用于选择的校准结果，不是独立泛化估计。外层结果仅为 Train 内 subject/trial 留出，不代表跨 Session 或 Test 成绩；未进行对抗攻击评测。",
              "最终全量模型与 OOF 折模型训练样本量不同，阈值迁移仍有分布偏移风险。Test 无真实标签，Test AUC/F1 均未知。正式提交与历史实验未修改。"]
    with (output / "report.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output), "outer": summary, "candidate": result["candidate"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
