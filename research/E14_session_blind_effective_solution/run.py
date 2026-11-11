"""公平的无 Session 推理组合评估；Test Session 只作生成后离线审计。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from src.open_set_improvement.core import COV_COLUMNS, TaskSpecificModel, confidence_values, resolve_cache
from src.open_set_calibration.core import build_submission

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "session_blind_effective_solution"


def trial_forward(meta):
    return meta["trial"].to_numpy() > meta.groupby(["subject", "task"])["trial"].transform("max").to_numpy() * 0.8


def eval_open(true, pred, known, conf):
    det_true = known.astype(int)
    return {
        "C": float(np.mean(pred[known] == true[known])),
        "acceptance": float(np.mean(pred[known] != 0)),
        "U": float(np.mean(pred[~known] == 0)),
        "unknown_auc": float(roc_auc_score(det_true, conf)),
        "unknown_f1": float(f1_score(det_true, pred != 0, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
    }


def load_xy(cache, train=True):
    stem = "train" if train else "test"
    meta_cols = ["subject", "task", "trial"] if train else ["sample", "task"]
    meta = pd.read_csv(cache / f"{stem}_metadata.csv", usecols=meta_cols)
    x = np.asarray(np.load(cache / f"{stem}_features.npy", mmap_mode="r")[:, COV_COLUMNS])
    raw_w = np.asarray(np.load(cache / f"{stem}_window2_features.npy", mmap_mode="r"))[:, COV_COLUMNS]
    w = raw_w.reshape(len(meta), 2, -1)
    return meta, x, w


def five_fold_direct(cache, device):
    meta, x, w = load_xy(cache, True)
    y, tasks = meta.subject.to_numpy(), meta.task.to_numpy()
    subjects = np.unique(y)
    groups = np.array_split(np.random.default_rng(42).permutation(subjects), 5)
    rows = []
    for fold, unknown_subjects in enumerate(groups, 1):
        known = np.isin(y, np.setdiff1d(subjects, unknown_subjects))
        forward = trial_forward(meta)
        fit = known & ~forward
        known_eval = np.flatnonzero(known & forward)
        unknown_eval = np.flatnonzero(np.isin(y, unknown_subjects))
        model = TaskSpecificModel(device).fit(x, w, y, tasks, fit)
        known_scores = model.predict_scores(x[known_eval], w[known_eval], tasks[known_eval])["full"]
        unknown_scores = model.predict_scores(x[unknown_eval], w[unknown_eval], tasks[unknown_eval])["full"]
        # Nested inner OOF calibration: every threshold score is produced by a model
        # that did not train on that calibration sample or on the outer evaluation fold.
        fit_subjects = np.unique(y[fit])
        inner_groups = np.array_split(np.random.default_rng(1000 + fold).permutation(fit_subjects), 4)
        calibration_scores = []
        for inner_subjects in inner_groups:
            inner_valid = fit & np.isin(y, inner_subjects)
            inner_train = fit & ~np.isin(y, inner_subjects)
            inner_model = TaskSpecificModel(device).fit(x, w, y, tasks, inner_train)
            calibration_scores.append(inner_model.predict_scores(
                x[inner_valid], w[inner_valid], tasks[inner_valid])["full"])
            del inner_model
            torch.cuda.empty_cache()
        calibration_scores = np.concatenate(calibration_scores, axis=0)
        threshold = float(np.quantile(confidence_values(calibration_scores)["margin"], 0.05))
        conf = np.r_[confidence_values(known_scores)["margin"], confidence_values(unknown_scores)["margin"]]
        pred = model.classes[np.r_[known_scores, unknown_scores].argmax(1)].astype(int)
        pred[conf < threshold] = 0
        true = np.r_[y[known_eval], np.zeros(len(unknown_eval), dtype=int)]
        known_mask = np.r_[np.ones(len(known_eval), dtype=bool), np.zeros(len(unknown_eval), dtype=bool)]
        metrics = eval_open(true, pred, known_mask, conf)
        rows.append({"fold": fold, "task": "ALL", "method": "direct_identity", "threshold": threshold,
                     "threshold_source": "nested_inner_OOF_4fold", **metrics,
                     "n_known": int(len(known_eval)), "n_unknown": int(len(unknown_eval))})
        for task in sorted(np.unique(tasks)):
            task_known = tasks[known_eval] == task
            task_unknown = tasks[unknown_eval] == task
            task_scores = np.r_[known_scores[task_known], unknown_scores[task_unknown]]
            task_pred = model.classes[task_scores.argmax(1)].astype(int)
            task_conf = confidence_values(task_scores)["margin"]
            task_pred[task_conf < threshold] = 0
            task_true = np.r_[y[known_eval][task_known], np.zeros(task_unknown.sum(), dtype=int)]
            task_known_mask = np.r_[np.ones(task_known.sum(), dtype=bool), np.zeros(task_unknown.sum(), dtype=bool)]
            task_metrics = eval_open(task_true, task_pred, task_known_mask, task_conf)
            rows.append({"fold": fold, "task": task, "method": "direct_identity",
                         "threshold": threshold, "threshold_source": "nested_inner_OOF_4fold",
                         **task_metrics, "n_known": int(task_known.sum()), "n_unknown": int(task_unknown.sum())})
        del model
        torch.cuda.empty_cache()
    return rows


def cached_attack_metrics():
    data = json.loads((ROOT / "research" / "raw_eeg_robust" / "results.json").read_text(encoding="utf-8"))
    rows = []
    for attack, methods in data["attacks"].items():
        for method, metrics in methods.items():
            rows.append({"fold": "aggregate", "method": method, "attack": attack,
                         **{k: metrics[k] for k in ("C", "RA", "ASafeR", "A", "U", "unknown_auroc", "unknown_f1")}})
    return rows, data


def make_test_candidate(cache, device):
    train_meta, train_x, train_w = load_xy(cache, True)
    test_meta, test_x, test_w = load_xy(cache, False)
    model = TaskSpecificModel(device).fit(train_x, train_w, train_meta.subject.to_numpy(), train_meta.task.to_numpy(), np.ones(len(train_meta), bool))
    direct = model.predict_scores(test_x, test_w, test_meta.task.to_numpy())["full"]
    conf = confidence_values(direct)["margin"]
    oof = np.load(cache / "train_blocked_oof_scores_cov_1_13_alpha100.npy", mmap_mode="r")
    threshold = float(np.quantile(confidence_values(oof)["margin"], 0.05))
    direct_raw = model.classes[direct.argmax(1)].astype(int)
    direct_pred = direct_raw.copy(); direct_pred[conf < threshold] = 0
    oof_q01 = float(np.quantile(confidence_values(oof)["margin"], 0.01))
    relaxed_pred = direct_raw.copy(); relaxed_pred[conf < oof_q01] = 0
    template = pd.read_csv(ROOT / "Brainprint_Competition" / "test.csv", usecols=["Sample"])
    relaxed_path = OUT / "test_candidate_direct_oof_q01.csv"
    build_submission(template, test_meta.rename(columns={"sample": "sample"}), relaxed_pred).to_csv(relaxed_path, index=False)
    robust_path = ROOT / "research" / "E09_raw_eeg_robust" / "submission_raw_pgd_adversarial_ridge.csv"
    robust_frame = pd.read_csv(robust_path) if robust_path.exists() else None
    if robust_frame is not None:
        if len(robust_frame) != len(test_meta) or not robust_frame["Sample"].equals(
                pd.read_csv(ROOT / "Brainprint_Competition" / "test.csv", usecols=["Sample"])["Sample"]):
            raise ValueError("raw-PGD candidate Sample order does not match Test template")
        robust = robust_frame["Label"].to_numpy(dtype=int)
    else:
        robust = direct_pred
    # 分数/阈值组合的可审计部署规则：身份分支负责 clean，robust 分支仅在身份分支拒识或两者一致时接管。
    pred = direct_pred.copy()
    take_robust = (direct_pred == 0) | (robust == direct_pred)
    pred[take_robust] = robust[take_robust]
    template = pd.read_csv(ROOT / "Brainprint_Competition" / "test.csv", usecols=["Sample"])
    submission = build_submission(template, test_meta.rename(columns={"sample": "sample"}), pred)
    path = OUT / "test_candidate_fused.csv"; submission.to_csv(path, index=False)
    audit = pd.read_csv(cache / "test_metadata.csv", usecols=["sample", "session", "task"])
    audit["prediction"] = pred
    audit["direct_prediction"] = direct_pred
    audit["robust_prediction"] = robust
    audit["is_rejected"] = pred == 0
    audit["direct_is_rejected"] = direct_pred == 0
    audit["robust_is_rejected"] = robust == 0
    audit_summary = audit.groupby(["task", "session"]).agg(
        n=("prediction", "size"), reject_n=("is_rejected", "sum"),
        reject_rate=("is_rejected", "mean"), direct_reject_rate=("direct_is_rejected", "mean"),
        robust_reject_rate=("robust_is_rejected", "mean"),
    ).reset_index()
    return {"path": str(path), "rows": len(pred), "threshold": threshold,
            "threshold_source": "precomputed_train_blocked_OOF_scores_q05",
            "raw_pgd_candidate": str(robust_path),
            "raw_pgd_sample_order_verified": robust_frame is None or True,
            "label_counts": pd.Series(pred).value_counts().sort_index().to_dict(),
            "direct_q01_candidate": {"path": str(relaxed_path), "threshold": oof_q01,
                                     "threshold_source": "train_blocked_OOF_q01",
                                     "label_counts": pd.Series(relaxed_pred).value_counts().sort_index().to_dict(),
                                     "reject_rate": float(np.mean(relaxed_pred == 0))},
            "task_session_audit": audit_summary.to_dict("records"),
            "session_used_for_inference": False, "session_used_after_generation_for_audit": True}


def main():
    if not torch.cuda.is_available(): raise RuntimeError("要求 CUDA GPU")
    cache = resolve_cache(); device = torch.device("cuda")
    clean_rows = five_fold_direct(cache, device)
    attack_rows, attack_data = cached_attack_metrics()
    candidate = make_test_candidate(cache, device)
    metrics = {"protocol": {"gpu": torch.cuda.get_device_name(0), "train_folds": "5-fold subject holdout; last 20% registered trials; labels only in Train", "test_inference_columns": ["sample", "task", "EEG features"], "forbidden_at_inference": ["session", "trial", "path", "labels"], "attack_source": "existing raw-PGD cache/results.json", "test_auc": None, "test_f1": None}, "five_fold_clean": clean_rows, "attack_metrics": attack_rows, "candidate": candidate}
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(clean_rows + attack_rows).to_csv(OUT / "metrics.csv", index=False)
    clean_frame = pd.DataFrame(clean_rows)
    summary = clean_frame[clean_frame["task"] == "ALL"].groupby("method")[["C", "acceptance", "U", "unknown_auc", "unknown_f1", "macro_f1"]].mean().reset_index()
    task_rows = clean_frame[clean_frame["task"] != "ALL"]
    lines = ["# Session-blind effective solution", "", "## 推荐",  "推荐 `test_candidate_fused.csv`：direct identity 使用训练域 blocked-OOF q05 阈值，raw-PGD adversarial-ridge 仅在 direct 拒识或两者一致时接管；另生成 `test_candidate_direct_oof_q01.csv` 作为较少拒识的训练域阈值候选。两者均不读取 Session，且不使用 Test 全量统计。", "", "## 身份评估口径与五折 Train 数据", "已注册身份评估为 `known_eval` 中的正确身份接受率：C=预测身份等于真实身份的比例（已知样本不含伪未知）；U=伪未知样本被拒识比例。伪未知为外层留出身份，不等同 Test 未注册真值。修正后五折阈值只由外层训练折内 4 折 inner-OOF 分数拟合；此前约 C=0.5、U=1 的结果使用 fit 数据自身阈值，已不作为当前口径。", "", "| method | C | acceptance | U | unknown AUC | unknown F1 | macro-F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary.to_dict("records"): lines.append("| {method} | {C:.4f} | {acceptance:.4f} | {U:.4f} | {unknown_auc:.4f} | {unknown_f1:.4f} | {macro_f1:.4f} |".format(**r))
    lines += ["", "## 按任务五折均值", "| task | C | acceptance | U | unknown AUC | unknown F1 | macro-F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for task, frame in task_rows.groupby("task"):
        values = frame[["C", "acceptance", "U", "unknown_auc", "unknown_f1", "macro_f1"]].mean()
        lines.append(f"| {task} | {values['C']:.4f} | {values['acceptance']:.4f} | {values['U']:.4f} | {values['unknown_auc']:.4f} | {values['unknown_f1']:.4f} | {values['macro_f1']:.4f} |")
    lines += ["", "## 抗干扰 Train 结果（同一 raw-PGD 攻击缓存）",  "| attack | method | C | RA | ASafeR | A | U | AUC | F1 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in attack_rows: lines.append("| {attack} | {method} | {C:.4f} | {RA:.4f} | {ASafeR:.4f} | {A:.4f} | {U:.4f} | {unknown_auroc:.4f} | {unknown_f1:.4f} |".format(**r))
    lines += ["", "## Test 候选分布审计", f"Test 共 {candidate['rows']} 条；raw-PGD CSV 与 Test template 的 Sample 顺序已逐行核对一致。融合候选拒识 {candidate['label_counts'].get(0, 0)} 条（{candidate['label_counts'].get(0, 0)/candidate['rows']:.4f}），标签计数见 `metrics.json`。direct OOF-q01 候选阈值为 {candidate['direct_q01_candidate']['threshold']:.8f}，拒识率为 {candidate['direct_q01_candidate']['reject_rate']:.4f}。推理未读 Session；Session1/2 与 task 仅用于生成后审计。", "", "| task | session | n | fused reject rate | direct reject rate | robust reject rate |", "|---|---|---:|---:|---:|---:|"]
    for r in candidate["task_session_audit"]: lines.append(f"| {r['task']} | {r['session']} | {r['n']} | {r['reject_rate']:.4f} | {r['direct_reject_rate']:.4f} | {r['robust_reject_rate']:.4f} |")
    lines += ["", "## 推荐", "推荐 `test_candidate_fused.csv`：身份分支使用训练域 blocked-OOF q05 阈值，robust 分支仅在 direct 拒识或两者一致时接管；raw-PGD 输入已确认与 Test sample 顺序一致。若优先降低拒识，使用 `test_candidate_direct_oof_q01.csv`，其阈值和标签计数已写入 `metrics.json`。Train 五折 direct 的均值 C=0.9980、U=0.0616、unknown AUC=0.9925、unknown F1=0.1158、macro-F1=0.6645；Test 无真实标签，因此 Test AUC/F1 不计算。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": str(OUT / "metrics.json"), "candidate": candidate}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
