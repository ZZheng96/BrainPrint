"""Nested blocked calibration and one fixed robust target-reference update."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research.session_blind_candidate.generate_five import METHODS, sha256
from src.session_blind_candidate.run import load_inputs
from src.open_set_improvement.core import TaskSpecificModel, resolve_cache
from src.open_set_calibration.core import predict_calibration_scores, build_submission
from src.ea_riemannian_openset.core import (
    fit_recentered_model, SPDRecenter, matrix_power, matrix_log, matrix_exp,
    log_features_to_spd, riemannian_mean,
)

OUT = Path(__file__).resolve().parent
ROBUST = "full_riemannian_robust_once"
ALL_METHODS = (*METHODS, ROBUST)


def score(x, w, y, tasks, fit, tx, tw, tt, device):
    model = TaskSpecificModel(device, alpha=100.0).fit(x, w, y, tasks, fit)
    outputs = {}
    for alignment in ("none", "target_zscore"):
        current, _ = predict_calibration_scores(model, tx, tw, tt, alignment)
        outputs.update(current)
    del model
    model, source = fit_recentered_model(x, w, y, tasks, fit, "riemannian_recenter", device)
    target = SPDRecenter("riemannian_recenter", device).fit(tx, tt)
    aligned = target.transform(tx, tt)
    current, _ = predict_calibration_scores(model, aligned, tw, tt, "none")
    initial = current["full_none"]
    outputs[METHODS[-1]] = initial
    confidence = initial.max(axis=1)
    diag = {}
    for task in np.unique(tt):
        indices = np.flatnonzero(tt == task)
        selected = indices[confidence[indices] >= np.quantile(confidence[indices], 0.5)]
        cov = log_features_to_spd(tx[selected], device=device)
        old = target.references[str(task)]
        inv = matrix_power(old, -0.5)
        tangent = matrix_log(inv.unsqueeze(0) @ cov @ inv.unsqueeze(0))
        distance = tangent.square().sum(dim=(-1, -2, -3)).sqrt()
        keep = distance <= torch.quantile(distance, 0.9)
        reference = riemannian_mean(cov[keep])
        # Half-step on the affine-invariant geodesic limits reference drift.
        root = matrix_power(old, 0.5)
        target.references[str(task)] = root @ matrix_exp(0.5 * matrix_log(inv @ reference @ inv)) @ root
        diag[str(task)] = {"n_target": len(indices), "n_high_confidence": len(selected),
                           "n_reference": int(keep.sum().item())}
        del cov, tangent, distance
    aligned = target.transform(tx, tt)
    current, _ = predict_calibration_scores(model, aligned, tw, tt, "none")
    outputs[ROBUST] = current["full_none"]
    if any(not np.isfinite(v).all() for v in outputs.values()):
        raise ValueError("Nonfinite predictions")
    return outputs, diag


def thresholds(values, tasks, mode):
    confidence = values.max(axis=1)
    if mode == "global":
        return {str(t): float(np.quantile(confidence, 0.05)) for t in np.unique(tasks)}
    return {str(t): float(np.quantile(confidence[tasks == t], 0.05)) for t in np.unique(tasks)}


def predict(values, tasks, classes, cuts):
    pred = classes[values.argmax(axis=1)].astype(int)
    pred[values.max(axis=1) < np.array([cuts[str(t)] for t in tasks])] = 0
    return pred


def known_metrics(y, pred, raw, tasks):
    def summary(mask):
        return {"n": int(mask.sum()), "C": float(np.mean(pred[mask] == y[mask])),
                "F1": float(f1_score(y[mask], pred[mask], labels=np.arange(1, 21), average="macro", zero_division=0)),
                "reject_rate": float(np.mean(pred[mask] == 0)),
                "direct_C": float(np.mean(raw[mask] == y[mask])), "U": None, "AUC": None}
    result = summary(np.ones(len(y), dtype=bool))
    result["by_task"] = {str(t): summary(tasks == t) for t in np.unique(tasks)}
    return result


def main():
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    protected = sorted((ROOT / "commit/answer").glob("赞美诗_第*次.csv"))
    before = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    cache = resolve_cache()
    x, w, meta, tx, tw, tm = load_inputs(cache)
    y, tasks, tt = meta.subject.to_numpy(), meta.task.to_numpy(), tm.task.to_numpy()
    classes = np.unique(y)
    split_path = cache / "train_blocked_oof_predictions_cov_alpha100.csv"
    splits = pd.read_csv(split_path, usecols=["subject", "task", "fold"])
    if not splits[["subject", "task"]].equals(meta[["subject", "task"]]):
        raise ValueError("Split alignment failed")
    folds = splits.fold.to_numpy()
    assert set(np.unique(folds)) == set(range(5))
    oof = {m: np.full((len(y), len(classes)), np.nan, np.float32) for m in ALL_METHODS}
    predictions = {m + "_" + mode: np.zeros(len(y), int) for m in ALL_METHODS for mode in ("global", "task")}
    nested = {}
    diagnostics = {}
    for fold in range(5):
        validation = folds == fold
        calibration = folds == ((fold + 1) % 5)
        fit = ~(validation | calibration)
        print(f"Outer {fold}: inner blocked calibration (three fit folds)", flush=True)
        inner_scores, inner_diag = score(x, w, y, tasks, fit, x[calibration], w[calibration], tasks[calibration], device)
        print(f"Outer {fold}: refit four folds and score untouched outer fold", flush=True)
        outer_scores, outer_diag = score(x, w, y, tasks, ~validation, x[validation], w[validation], tasks[validation], device)
        nested[str(fold)] = {}
        diagnostics[str(fold)] = {"inner": inner_diag, "outer": outer_diag}
        for method in ALL_METHODS:
            oof[method][validation] = outer_scores[method]
            for mode in ("global", "task"):
                key = method + "_" + mode
                cuts = thresholds(inner_scores[method], tasks[calibration], mode)
                nested[str(fold)][key] = cuts
                predictions[key][validation] = predict(outer_scores[method], tasks[validation], classes, cuts)
    print("Full GPU training / unlabeled Test prediction", flush=True)
    test_scores, diagnostics["test"] = score(x, w, y, tasks, np.ones(len(y), bool), tx, tw, tt, device)
    template = pd.read_csv(ROOT / "Brainprint_Competition/test.csv", usecols=["Sample"])
    original_manifest_path = OUT.parent / "five_candidates/manifest.json"
    original = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    metrics = {"definition": "C=known identity accuracy including reject errors; F1=macro over registered labels 1..20; U/open-set AUC unavailable without unknown examples", "methods": {}}
    manifest = {
        "seed": 42, "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "command": "py -3.11 research/session_blind_candidate/improved_task_threshold/run.py",
        "metadata_columns_read": {"train": ["subject", "task"], "test": ["sample", "task"], "split": ["subject", "task", "fold"]},
        "forbidden_columns_read": [], "test_labels_used": False,
        "split_sha256": sha256(split_path), "n_train": len(y), "n_test": len(tx),
        "validation": "Five outer blocked folds. Inner calibration fold=(outer+1)%5; inner models fit other three folds. Outer refit uses four folds. No outer labels/features enter calibration models or thresholds. Unlabeled target reference is transductive within held-out batch only.",
        "deployment_threshold": "Per-method full blocked OOF 5th percentile, global or per task; no Test tuning",
        "robust_update": "One update: per-task upper 50% initial max-logit; retain 90% nearest to original reference by concatenated whitened log Frobenius distance; Riemannian mean; geodesic half-step shrinkage. Fixed a priori, no label-based tuning.",
        "nested_thresholds": nested, "reference_diagnostics": diagnostics,
        "original_manifest_sha256": sha256(original_manifest_path), "candidates": {},
    }
    records = pd.DataFrame({"subject": y, "task": tasks, "fold": folds})
    for method in ALL_METHODS:
        raw = classes[oof[method].argmax(axis=1)]
        for mode in ("global", "task"):
            key = method + "_" + mode
            pred = predictions[key]
            records[key] = pred
            cuts = thresholds(oof[method], tasks, mode)
            test_pred = predict(test_scores[method], tt, classes, cuts)
            submission = build_submission(template, tm, test_pred)
            assert len(submission) == 22251 and submission.Sample.is_unique
            assert submission.Sample.equals(template.Sample) and submission.Label.isin(range(21)).all()
            assert pd.api.types.is_integer_dtype(submission.Label)
            path = OUT / f"candidate_{key}.csv"
            submission.to_csv(path, index=False)
            test_summary = {"C": None, "U": None, "AUC": None, "F1": None,
                            "reject_rate": float(np.mean(test_pred == 0)), "n_reject": int(np.sum(test_pred == 0)),
                            "by_task_reject_rate": {str(t): float(np.mean(test_pred[tt == t] == 0)) for t in np.unique(tt)}}
            entry = {"oof": known_metrics(y, pred, raw, tasks), "test": test_summary}
            if method in METHODS:
                old_path = OUT.parent / "five_candidates" / original["candidates"][method]["file"]
                old = pd.read_csv(old_path, usecols=["Sample", "Label"])
                assert old.Sample.equals(submission.Sample)
                entry["original_test_reject_rate"] = original["candidates"][method]["reject_rate"]
                entry["changed_labels_vs_original"] = int(np.sum(old.Label.to_numpy() != submission.Label.to_numpy()))
            metrics["methods"][key] = entry
            manifest["candidates"][key] = {"file": path.name, "sha256": sha256(path), "thresholds": cuts,
                "n_rows": len(submission), "Sample_aligned": True, "Label_valid": True}
    for method in ALL_METHODS:
        a = metrics["methods"][method + "_global"]["oof"]
        b = metrics["methods"][method + "_task"]["oof"]
        metrics["methods"][method + "_task"]["delta_vs_global"] = {k: b[k] - a[k] for k in ("C", "F1", "reject_rate")}
    records.to_csv(OUT / "nested_blocked_oof_predictions.csv", index=False)
    np.savez_compressed(OUT / "blocked_oof_scores.npz", classes=classes, fold=folds, **oof)
    after = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    assert before == after
    manifest["formal_submissions_unchanged"] = True
    manifest["formal_submissions_sha256"] = after
    manifest["elapsed_seconds"] = time.time() - started
    manifest["script_sha256"] = sha256(Path(__file__))
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Task-specific 阈值与单次稳健目标参考实验", "", "## 设计与边界", "",
        "使用 CUDA；五个外层 blocked 折，内层固定留出一个完整 blocked 折校准（其余三折训练），外层模型用四折重拟合。全局/task 阈值均固定为校准置信度 5% 分位；最终 Test 阈值由完整训练 blocked-OOF 得分确定。未读取 session/trial/path/fs 或 Test 标签。", "",
        "高置信参考：每 task 取初始置信度前 50%，按白化切空间距离剔除最远 10%，几何均值与原参考作 0.5 测地线收缩，仅一次。该规则未根据验证结果选择。", "",
        "训练 OOF 只有已注册身份，C 为拒识计错的身份准确率，F1 为标签 1–20 宏平均（拒识作为相应真实类别 FN）；U、开集 AUC 不可计算。Test C/U/AUC/F1 均不可计算；Test 拒识率不等于 U。未模拟对抗，不能估计官方综合分。", "",
        "内层单折校准是严格独立的 blocked holdout，并非完整四折内层交叉验证；减少计算但阈值方差可能较大。目标批次对齐是无标签传导推理，不是单样本归纳推理。", "",
        "## 同口径嵌套 OOF 与 Test 拒识比较", "", "|方法|阈值|OOF C|OOF F1|OOF 拒识率|Test 拒识率|", "|---|---|---:|---:|---:|---:|"]
    for method in ALL_METHODS:
        for mode in ("global", "task"):
            entry = metrics["methods"][method + "_" + mode]
            a, b = entry["oof"], entry["test"]
            lines.append(f"|{method}|{mode}|{a['C']:.6f}|{a['F1']:.6f}|{a['reject_rate']:.6f}|{b['reject_rate']:.6f}|")
    lines += ["", "## 与原五候选比较", "", "|方法|原 Test 拒识率|新 task Test 拒识率|标签变化数|task-global OOF ΔC|ΔF1|", "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        e = metrics["methods"][method + "_task"]
        d = e["delta_vs_global"]
        lines.append(f"|{method}|{e['original_test_reject_rate']:.6f}|{e['test']['reject_rate']:.6f}|{e['changed_labels_vs_original']}|{d['C']:+.6f}|{d['F1']:+.6f}|")
    baseline = metrics["methods"][METHODS[-1] + "_task"]["oof"]
    robust = metrics["methods"][ROBUST + "_task"]["oof"]
    lines += ["", f"稳健参考相对原黎曼方法（均 task 阈值）：ΔC={robust['C'] - baseline['C']:+.6f}，ΔF1={robust['F1'] - baseline['F1']:+.6f}。",
        "", "以上只能判断已注册身份的训练 OOF 表现，不能证明 Test 或未知拒识改进，不按 Test 拒识率选优；不替换正式提交。按 task 的详细指标及阈值见 metrics.json、manifest.json。所有 12 份候选均为 22251 行，Sample 与模板对齐、Label 为 0–20 整数。正式提交 SHA256 保持不变。", "",
        "## 复现", "", "`py -3.11 research/session_blind_candidate/improved_task_threshold/run.py`", "", f"GPU: {manifest['device']}；耗时 {manifest['elapsed_seconds']:.1f} 秒。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
