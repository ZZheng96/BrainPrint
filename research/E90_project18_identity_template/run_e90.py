"""E90: Train-only identity template matching with nested subject/task/time splits."""
from __future__ import annotations
import argparse, csv, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CONFIG = OUT / "config.json"
RESULTS = OUT / "results.json"
SMOKE_RESULTS = OUT / "smoke_results.json"
REPORT = OUT / "report.md"
MODEL = OUT / "identity_templates.pt"
PRED = OUT / "sample_predictions.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def dump(path, obj):
    def conv(x):
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=conv) + "\n", encoding="utf-8")


def load_inputs(cfg):
    meta = pd.read_csv(ROOT / cfg["data"]["train_metadata"])
    if len(meta) != cfg["data"]["expected_trials"] or meta.subject.nunique() != 20:
        raise ValueError("Train metadata shape/subject count mismatch")
    if set(meta.subject.astype(int)) != set(range(1, 21)) or set(meta.session) != {"session1"}:
        raise ValueError("Train must contain subjects 1..20 and session1 only")
    if not meta.path.astype(str).str.startswith("Train/").all() or meta.path.duplicated().any():
        raise ValueError("Train-only path invariant failed")
    if set(meta.task) != set(cfg["data"]["tasks"]): raise ValueError("Task set mismatch")
    x = np.load(ROOT / cfg["data"]["feature_cache"], mmap_mode="r")
    if x.shape != (len(meta), 855) or not np.isfinite(x).all(): raise ValueError("Feature cache invalid")
    return meta, x


def masks(meta, fit_end=.60, query_end=.80):
    trial = meta.trial.to_numpy(float)
    maximum = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    frac = trial / np.maximum(maximum, 1.0)
    return {"fit": frac <= fit_end, "cal_known": (frac > fit_end) & (frac <= .70),
            "query": (frac > .70) & (frac <= query_end), "outer_query": frac > query_end,
            "early": frac <= query_end}


def outer_groups(subjects, seed, n):
    return [np.asarray(g, int) for g in np.array_split(np.random.default_rng(seed).permutation(subjects), n)]


def inner_groups(subjects, outer, n=2):
    return [np.asarray(g, int) for g in np.array_split(np.random.default_rng(100 + outer).permutation(subjects), n)]


def time_bin(meta, indices, bins=3):
    trial = meta.trial.to_numpy(float)
    maximum = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    frac = trial / np.maximum(maximum, 1.0)
    return np.minimum((frac * bins).astype(int), bins - 1)[indices]


def fit_templates(x, meta, indices, classes, stratified, bins=3):
    rows = []
    labels = []
    keys = []
    xb = np.asarray(x[indices], dtype=np.float32)
    xb = xb / np.maximum(np.linalg.norm(xb, axis=1, keepdims=True), 1e-12)
    subjects = meta.subject.to_numpy(int)[indices]
    tasks = meta.task.to_numpy(str)[indices]
    times = time_bin(meta, indices, bins)
    for subject in classes:
        selected = np.flatnonzero(subjects == subject)
        if not len(selected): raise ValueError(f"No fit samples for subject {subject}")
        if not stratified:
            vec = xb[selected].mean(0); key = (int(subject), "ALL", 0)
            rows.append(vec / max(np.linalg.norm(vec), 1e-12)); labels.append(subject); keys.append(key)
        else:
            for task in sorted(set(tasks[selected])):
                for tb in sorted(set(times[selected][tasks[selected] == task])):
                    local = selected[(tasks[selected] == task) & (times[selected] == tb)]
                    if len(local):
                        vec = xb[local].mean(0)
                        rows.append(vec / max(np.linalg.norm(vec), 1e-12)); labels.append(subject); keys.append((int(subject), str(task), int(tb)))
    return {"vectors": np.asarray(rows, np.float32), "labels": np.asarray(labels, int), "keys": keys, "classes": np.asarray(classes, int)}


def score_templates(model, x, indices, device, batch=2048):
    vectors = torch.as_tensor(model["vectors"], device=device)
    labels = model["labels"]
    out = np.empty((len(indices), len(model["classes"])), np.float32)
    lookup = {int(s): i for i, s in enumerate(model["classes"])}
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            sel = np.asarray(indices[start:start + batch], int)
            q = torch.as_tensor(np.asarray(x[sel], np.float32), device=device)
            q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
            raw = q @ vectors.T
            scores = torch.full((len(sel), len(model["classes"])), -1.0, device=device)
            for j, subject in enumerate(labels):
                scores[:, lookup[int(subject)]] = torch.maximum(scores[:, lookup[int(subject)]], raw[:, j])
            out[start:start + len(sel)] = scores.cpu().numpy()
    return out


def eer(known, unknown):
    y = np.r_[np.ones(len(known)), np.zeros(len(unknown))]
    fpr, tpr, _ = roc_curve(y, np.r_[known, unknown])
    return float(np.min(np.maximum(fpr, 1 - tpr)))


def evaluate(scores_known, true, scores_unknown, threshold, classes):
    kc = scores_known.max(1); uc = scores_unknown.max(1)
    known_direct = classes[scores_known.argmax(1)]
    unknown_direct = classes[scores_unknown.argmax(1)]
    kp = np.where(kc >= threshold, known_direct, 0)
    up = np.where(uc >= threshold, unknown_direct, 0)
    accepted = np.r_[kp != 0, up != 0]
    det = np.r_[np.ones(len(kp), int), np.zeros(len(up), int)]
    return {"N_known": int(len(kp)), "N_unknown": int(len(up)),
            "AUC": float(roc_auc_score(det, np.r_[kc, uc])),
            "F1": float(f1_score(true, kp, labels=classes, average="macro", zero_division=0)),
            "known_acceptance": float(np.mean(kp != 0)),
            "unknown_FAR": float(np.mean(up != 0)),
            "unknown_F1_pos_label_0": float(f1_score(det, accepted, pos_label=0, zero_division=0)),
            "EER": eer(kc, uc), "C": float(np.mean(kp == true)),
            "U": float(np.mean(up == 0)), "direct_C": float(np.mean(known_direct == true)),
            "threshold": float(threshold)}


def run(smoke=False):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    meta, x = load_inputs(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("E90 requires CUDA")
    subjects = meta.subject.to_numpy(int); all_subjects = np.arange(1, 21)
    m = masks(meta); groups = outer_groups(all_subjects, cfg["protocol"]["outer_seed"], cfg["protocol"]["outer_folds"])
    if smoke: groups = groups[:1]
    records, pred_rows, artifacts = [], [], {}
    for outer_no, outer_unknown in enumerate(groups, 1):
        known_subjects = np.setdiff1d(all_subjects, outer_unknown)
        split_groups = inner_groups(known_subjects, outer_no)
        if smoke: split_groups = split_groups[:1]
        inner_records = []
        for inner_no, inner_unknown in enumerate(split_groups, 1):
            enrolled = np.setdiff1d(known_subjects, inner_unknown)
            fit = np.flatnonzero(np.isin(subjects, enrolled) & m["fit"])
            cal_known = np.flatnonzero(np.isin(subjects, enrolled) & m["cal_known"])
            cal_unknown = np.flatnonzero(np.isin(subjects, inner_unknown) & (m["early"]))
            query_known = np.flatnonzero(np.isin(subjects, enrolled) & m["query"])
            query_unknown = np.flatnonzero(np.isin(subjects, inner_unknown) & m["query"])
            run_rec = {"outer_fold": outer_no, "inner_fold": inner_no, "outer_unknown": outer_unknown.tolist(),
                       "calibration_unknown": inner_unknown.tolist(), "enrolled": enrolled.tolist(),
                       "N_fit": int(len(fit)), "N_calibration_known": int(len(cal_known)),
                       "N_calibration_unknown": int(len(cal_unknown)), "variants": {}}
            for name, strat in (("single_centroid", False), ("task_time_multi_template", True)):
                model = fit_templates(x, meta, fit, enrolled, strat, cfg["protocol"]["template_time_bins"])
                cal_k = score_templates(model, x, cal_known, device, cfg["protocol"]["prediction_batch_size"])
                cal_u = score_templates(model, x, cal_unknown, device, cfg["protocol"]["prediction_batch_size"])
                threshold = float(np.quantile(cal_k.max(1), cfg["protocol"]["calibration_quantile"]))
                qk = score_templates(model, x, query_known, device, cfg["protocol"]["prediction_batch_size"])
                qu = score_templates(model, x, query_unknown, device, cfg["protocol"]["prediction_batch_size"])
                metrics = evaluate(qk, subjects[query_known], qu, threshold, enrolled)
                run_rec["variants"][name] = {"calibration_threshold": threshold, "metrics": metrics,
                                              "N_templates": int(len(model["labels"]))}
            inner_records.append(run_rec)
        outer_fit = np.flatnonzero(np.isin(subjects, known_subjects) & m["early"])
        outer_known = np.flatnonzero(np.isin(subjects, known_subjects) & m["outer_query"])
        outer_unk = np.flatnonzero(np.isin(subjects, outer_unknown))
        fold = {"outer_fold": outer_no, "outer_unknown": outer_unknown.tolist(), "registered": known_subjects.tolist(),
                "inner": inner_records, "N_outer_fit": int(len(outer_fit)), "N_outer_known": int(len(outer_known)),
                "N_outer_unknown": int(len(outer_unk)), "variants": {}}
        for name, strat in (("single_centroid", False), ("task_time_multi_template", True)):
            model = fit_templates(x, meta, outer_fit, known_subjects, strat, cfg["protocol"]["template_time_bins"])
            cal_known = np.flatnonzero(np.isin(subjects, known_subjects) & m["cal_known"])
            # Threshold is calibrated only from inner runs, never from outer query/unknown.
            thresholds = [r["variants"][name]["calibration_threshold"] for r in inner_records]
            threshold = float(np.mean(thresholds))
            ck = score_templates(model, x, cal_known, device, cfg["protocol"]["prediction_batch_size"])
            ok = score_templates(model, x, outer_known, device, cfg["protocol"]["prediction_batch_size"])
            ou = score_templates(model, x, outer_unk, device, cfg["protocol"]["prediction_batch_size"])
            metrics = evaluate(ok, subjects[outer_known], ou, threshold, known_subjects)
            fold["variants"][name] = {"calibration_threshold": threshold, "metrics": metrics,
                                       "N_templates": int(len(model["labels"]))}
            if not smoke:
                artifacts[f"outer_{outer_no}/{name}"] = {"classes": known_subjects, "vectors": model["vectors"],
                                                            "labels": model["labels"], "keys": model["keys"]}
                pred = np.where(ok.max(1) >= threshold, known_subjects[ok.argmax(1)], 0)
                for idx, value, score in zip(outer_known, pred, ok.max(1)):
                    pred_rows.append({"outer_fold": outer_no, "variant": name, "index": int(idx), "role": "known",
                                      "subject": int(subjects[idx]), "predicted": int(value), "score": float(score)})
                # Unknown prediction rows are included to make FAR auditable without Test access.
                pred = np.where(ou.max(1) >= threshold, known_subjects[ou.argmax(1)], 0)
                for idx, value, score in zip(outer_unk, pred, ou.max(1)):
                    pred_rows.append({"outer_fold": outer_no, "variant": name, "index": int(idx), "role": "unknown",
                                      "subject": int(subjects[idx]), "predicted": int(value), "score": float(score)})
        fold["inner_isolation"] = all(not np.intersect1d(np.flatnonzero(np.isin(subjects, np.setdiff1d(known_subjects, g)) & m["fit"]), np.flatnonzero(np.isin(subjects, g) & m["query"])).size for g in split_groups)
        records.append(fold)
        target = SMOKE_RESULTS if smoke else RESULTS
        dump(target, {"status": "running_smoke" if smoke else "running", "folds": records, "test": {"accessed": False, "metrics": "N/A"}})
    summary = {}
    inner_summary = {}
    for name in ("single_centroid", "task_time_multi_template"):
        rows = [f["variants"][name]["metrics"] for f in records]
        inner_rows = [r["variants"][name]["metrics"] for f in records for r in f["inner"]]
        summary[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k], (int, float)) and k not in ("N_known", "N_unknown")}
        summary[name]["N_known"] = int(sum(r["N_known"] for r in rows)); summary[name]["N_unknown"] = int(sum(r["N_unknown"] for r in rows))
        inner_summary[name] = {k: float(np.mean([r[k] for r in inner_rows])) for k in inner_rows[0] if isinstance(inner_rows[0][k], (int, float)) and k not in ("N_known", "N_unknown")}
        inner_summary[name]["N_known"] = int(sum(r["N_known"] for r in inner_rows)); inner_summary[name]["N_unknown"] = int(sum(r["N_unknown"] for r in inner_rows))
    result = {"status": "complete_smoke" if smoke else "complete", "experiment": cfg["experiment"],
              "mode": "smoke" if smoke else "full_5_outer_x_2_inner", "config_sha256": sha256(CONFIG),
              "data": {"metadata": cfg["data"]["train_metadata"], "feature_cache": cfg["data"]["feature_cache"],
                       "test_accessed": False, "test_metrics": {k: "N/A" for k in ("AUC", "F1", "C", "RA", "ASafeR", "U", "Score")}},
              "folds": records, "inner_summary": inner_summary, "summary": summary, "protocol_checks": {"train_only": True, "outer_subject_task_time_split": True,
              "independent_calibration": True, "test_accessed": False}}
    dump(SMOKE_RESULTS if smoke else RESULTS, result)
    if not smoke:
        torch.save({"experiment": cfg["experiment"], "config_sha256": sha256(CONFIG), "outer_models": artifacts}, MODEL)
        with PRED.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["outer_fold", "variant", "index", "role", "subject", "predicted", "score"]); writer.writeheader(); writer.writerows(pred_rows)
    write_report(result, smoke)
    return result


def write_report(result, smoke):
    lines = ["# E90 project18 身份模板（Train-only）", "", "## 结论与当前 AUC / F1", "",
             "仅使用 Train/session1 的 18 通道、E65 855 维相关 logSPD 特征；未读取项目 Test、Test 索引或 Test 标签。"]
    if smoke: lines += ["", "Smoke 仅执行 outer=1、inner=1，结果为流程检查，非正式证据。"]
    lines += ["", "| 模板 | AUC | F1 | known acceptance | unknown FAR | unknown-F1(pos_label=0) | EER | C | U |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in result["summary"].items():
        lines.append(f"| {name} | {row['AUC']:.6f} | {row['F1']:.6f} | {row['known_acceptance']:.6f} | {row['unknown_FAR']:.6f} | {row['unknown_F1_pos_label_0']:.6f} | {row['EER']:.6f} | {row['C']:.6f} | {row['U']:.6f} |")
    lines += ["", "## Inner calibration summary", "", "| 模板 | AUC | F1 | known acceptance | unknown FAR | unknown-F1(pos_label=0) | EER | C | U |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in result["inner_summary"].items():
        lines.append(f"| {name} | {row['AUC']:.6f} | {row['F1']:.6f} | {row['known_acceptance']:.6f} | {row['unknown_FAR']:.6f} | {row['unknown_F1_pos_label_0']:.6f} | {row['EER']:.6f} | {row['C']:.6f} | {row['U']:.6f} |")
    lines += ["", "## 协议", "",  "- outer：5 折留出被试（每折 4 人）；outer fit 为注册被试各 subject/task 的前 80%，outer known query 为 >80% 的末端时间块，outer unknown 为留出被试全部 Train trials。", "- inner：每个 outer 的 16 名已知被试再分 8/8；模型只在 enrolled 的 ≤60% 时间块拟合，calibration known 为 60–70%，calibration unknown 为留出 8 人 ≤80%；阈值仅由 inner calibration known q05 产生，outer query 不参与标定。", "- single centroid：每个身份一个归一化均值模板；task/time-stratified multi-template：每个身份按任务和时间分层建立多个归一化均值模板，查询取该身份模板最高 cosine。", "- 报告 unknown FAR、unknown-F1 使用 pos_label=0，EER 为 known/unknown confidence 等错误率交点；Test 指标全部 N/A。", "", "## 实际命令", "", "```powershell", ".venv/Scripts/python.exe research/E90_project18_identity_template/run_e90.py --smoke", ".venv/Scripts/python.exe research/E90_project18_identity_template/run_e90.py --full", "```", "", "模型/模板：`identity_templates.pt`；逐样本 Train 外层预测：`sample_predictions.csv`；结果：`results.json`。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    if args.smoke == args.full: ap.error("choose exactly one of --smoke or --full")
    result = run(args.smoke)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
