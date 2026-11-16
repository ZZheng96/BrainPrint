"""E213: strict Train-only adaptive gate using E170 frozen scores."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CACHE = ROOT / "research" / "E170_three_way_fusion_ceiling" / "frozen_scores.npz"
TASKS = ["EC", "EO", "Imagery", "SSVEP"]


def auc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1; neg = ~pos
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def f1(y, pred):
    y = np.asarray(y, int); pred = np.asarray(pred, int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def features(scores):
    s = np.asarray(scores, np.float32)
    cols = [s]
    for i in range(s.shape[1]):
        for j in range(i + 1, s.shape[1]):
            cols.append(np.abs(s[:, i:i + 1] - s[:, j:j + 1]))
    cols += [s.std(1, keepdims=True), s.max(1, keepdims=True) - s.min(1, keepdims=True)]
    return np.concatenate(cols, 1).astype(np.float32)


def fit_logistic(x, y, reg, steps=800, lr=0.08):
    mu, sd = x.mean(0), x.std(0) + 1e-6
    z = (x - mu) / sd
    w = np.zeros(z.shape[1], np.float64); b = 0.0
    yy = y.astype(np.float64)
    for _ in range(steps):
        q = np.clip(1 / (1 + np.exp(-(z @ w + b))), 1e-6, 1 - 1e-6)
        g = q - yy
        w -= lr * ((z.T @ g) / len(z) + reg * w)
        b -= lr * g.mean()
    return w.astype(np.float32), np.float32(b), mu.astype(np.float32), sd.astype(np.float32)


def predict(x, model):
    w, b, mu, sd = model
    q = (x - mu) / sd
    return 1 / (1 + np.exp(-(q @ w + b)))


def main():
    z = np.load(CACHE)
    tags = ["a_E143_E65_PCA96_k12", "b_E167_best_auc", "c_E170_best_auc_3way_score", "d_E65_PCA64_k12"]
    train = np.column_stack([z[f"z_rf_{t}"] for t in tags])
    test = np.column_stack([z[f"z_te_{t}"] for t in tags])
    known = []
    unknown = []
    for t in tags:
        known.append(np.concatenate([z[f"loso_known_{t}_{i}"] for i in range(20)]))
        unknown.append(np.concatenate([z[f"loso_unknown_{t}_{i}"] for i in range(20)]))
    # LOSO fold boundaries are retained so thresholds and selection never use Test.
    known_folds = [np.column_stack([z[f"loso_known_{t}_{i}"] for t in tags]) for i in range(20)]
    unknown_folds = [np.column_stack([z[f"loso_unknown_{t}_{i}"] for t in tags]) for i in range(20)]
    x_unknown = features(np.column_stack(unknown))
    x_oof = features(train)
    x_test = features(test)
    y_fit = np.r_[np.ones(len(x_oof), int), np.zeros(len(x_unknown), int)]
    x_fit = np.vstack([x_oof, x_unknown])
    candidates = []
    for reg in [0.0001, 0.001, 0.01, 0.1, 1.0]:
        model = fit_logistic(x_fit, y_fit, reg)
        fold_known = [predict(features(a), model) for a in known_folds]
        fold_unknown = [predict(features(a), model) for a in unknown_folds]
        la = auc(np.r_[np.ones(sum(map(len, fold_known))), np.zeros(sum(map(len, fold_unknown)))],
                 np.r_[np.concatenate(fold_known), np.concatenate(fold_unknown)])
        candidates.append((la, reg, model, fold_known, fold_unknown))
    loso_auc, reg, model, fold_known, fold_unknown = max(candidates, key=lambda r: (r[0], -r[1]))
    p_oof = predict(x_oof, model); p_test = predict(x_test, model)
    p_unknown = np.concatenate(fold_unknown)
    y_train_gate = np.r_[np.ones(len(p_oof), int), np.zeros(len(p_unknown), int)]
    p_train_gate = np.r_[p_oof, p_unknown]
    train_auc = auc(y_train_gate, p_train_gate)
    train_f1 = f1(y_train_gate, p_train_gate >= 0.5)
    rows = []
    for acceptance in [0.80, 0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96]:
        threshold = float(np.quantile(p_oof, 1 - acceptance))
        loso_rej = float(np.mean([np.mean(pu < threshold) for pu in fold_unknown]))
        s1 = float(np.mean(p_test[z["is_s1"]] < threshold))
        s2 = float(np.mean(p_test[~z["is_s1"]] >= threshold))
        rows.append({"acceptance_rate": acceptance, "threshold": threshold,
                     "LOSO_rejection": loso_rej, "Test_S1_rejection": s1,
                     "Test_S2_acceptance": s2,
                     "W_audit": 0.25 * loso_rej + 0.25 * s1 + 0.5 * s2})
    ref = next(r for r in rows if r["acceptance_rate"] == 0.90)
    np.savez_compressed(OUT / "model.npz", weights=model[0], bias=model[1], feature_mean=model[2],
                        feature_scale=model[3], regularization=reg, train_oof_gate=p_oof,
                        test_gate=p_test)
    np.savez_compressed(OUT / "adaptive_scores.npz", train_oof_gate=p_oof, test_gate=p_test,
                        is_s1=z["is_s1"])
    results = {"experiment": "E213_e170_adaptive_cross_session_gate",
               "protocol": "E170 frozen_scores only; gate fit on Train OOF known plus Train LOSO unknown; model/reg selected by Train LOSO AUC; Test frozen audit only",
               "cache": str(CACHE), "features": ["4 E170 streams", "6 pairwise absolute disagreements", "stream std", "stream range"],
               "device": "numpy (cached-score experiment; no raw EEG extraction)",
               "selected_regularization": float(reg), "train_oof_auc": train_auc,
               "train_oof_f1": train_f1, "loso_auc": float(loso_auc),
               "reference_090": ref, "curve": rows,
               "test_official": {"AUC": "N/A(no labels)", "F1": "N/A(no labels)"},
               "test_diagnostic_auc": auc((~z["is_s1"]).astype(int), p_test),
               "model_file": "model.npz", "score_file": "adaptive_scores.npz"}
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# E213 E170 自适应跨会话 gate", "", "## 协议", "",
              "仅复用 E170 `frozen_scores.npz`，不重新提取原始 EEG。gate 输入为 E65、logbp、E162 及 E170 分数的 6 个两两绝对差、跨流标准差和极差。轻量 NumPy logistic gate 使用 Train OOF known 与 Train LOSO unknown 训练；正则强度只按 Train LOSO AUC 选择。Test 只在冻结模型后审计。", "",
              "## 结果", "", f"- Train OOF AUC：**{train_auc:.4f}**", f"- Train OOF F1（gate known/unknown，阈值 0.5）：**{train_f1:.4f}**", f"- LOSO AUC（训练选择依据）：**{loso_auc:.4f}**", f"- Test 官方 AUC/F1：**N/A**（Test 无标签；仅报告 session 诊断 AUC {results['test_diagnostic_auc']:.4f}）", "",
              "### 冻结参考工作点 acceptance=0.90", "", f"- LOSO 拒识：**{ref['LOSO_rejection']:.4f}**", f"- Test S1 拒识：**{ref['Test_S1_rejection']:.4f}**", f"- Test S2 接受：**{ref['Test_S2_acceptance']:.4f}**", f"- W 审计值（不用于选参/选点）：**{ref['W_audit']:.4f}**", "",
              "## 产物", "", "- `run_e213.py`：可复现实验脚本", "- `results.json`：完整曲线与指标", "- `model.npz`：冻结 gate 参数", "- `adaptive_scores.npz`：冻结 Train/Test gate 分数", "- `report.md`：本报告", "", "运行命令：`d:\\WORK\\BCIC2026\\BrainPrint\\.venv\\Scripts\\python.exe research\\E213_e170_adaptive_cross_session_gate\\run_e213.py`"]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"train_oof_auc": train_auc, "train_oof_f1": train_f1, "loso_auc": loso_auc, **ref}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
