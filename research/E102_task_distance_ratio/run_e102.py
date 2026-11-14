"""E102: task-conditioned distance-ratio gate (relaxed target S1>=0.80, S2 accept>=0.85).

Key finding from E101 follow-up: the distance ratio
    d(second nearest identity) / d(nearest identity)
computed on task-conditioned correlation-logSPD prototypes reaches Train-only
AUC 0.916, far above raw identity confidence (0.63) or SPD distance (0.64).
At the 15th percentile of known scores it gives known acceptance 0.850 and
unknown rejection 0.843, meeting the relaxed joint target.

Protocol: Train/session1 only for prototypes and threshold; the project Test is
read once after the model and threshold are frozen; Session is used only for the
post-hoc audit.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score, f1_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E65_correlation_spd"))
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
import run_e65
from run_e36 import load_config as load_e36_config

CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TRAIN_META = ROOT / "BrainPrint_cache/train_metadata.csv"
TRAIN_FEATURES = ROOT / "BrainPrint_cache/e65_correlation_spd/train_correlation_logspd.npy"


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def build_task_prototypes(feat, subjects, tasks, fit_mask, classes):
    """One centroid per (task, identity) from the fit block."""
    protos, keys = [], []
    for task in CFG["tasks"]:
        ft = fit_mask & (tasks == task)
        if ft.sum() < CFG["min_fit_per_task"]:
            continue
        for c in classes:
            sel = ft & (subjects == c)
            if sel.sum() == 0:
                continue
            protos.append(np.asarray(feat[sel], dtype=np.float32).mean(0))
            keys.append((task, int(c)))
    return np.stack(protos), keys


def distance_ratio(feat, protos, keys, tasks):
    """Per sample: d2/d1 within its own task; returns ratio and predicted identity."""
    ratio = np.full(len(feat), np.nan, dtype=np.float32)
    pred = np.zeros(len(feat), dtype=np.int64)
    for task in CFG["tasks"]:
        idx = np.flatnonzero(tasks == task)
        if len(idx) == 0:
            continue
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(cols) < 2:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        cls = np.asarray([keys[i][1] for i in cols])
        for st in range(0, len(idx), 1024):
            chunk = idx[st:st + 1024]
            q = torch.as_tensor(np.array(feat[chunk], dtype=np.float32), device=DEVICE)
            d = torch.cdist(q, p).cpu().numpy()
            order = np.argsort(d, axis=1)
            d1 = d[np.arange(len(d)), order[:, 0]]
            d2 = d[np.arange(len(d)), order[:, 1]]
            ratio[chunk] = d2 / np.maximum(d1, 1e-9)
            pred[chunk] = cls[order[:, 0]]
    return ratio, pred


def choose_threshold(known_ratio, unknown_ratio):
    """Pick the smallest known-quantile that meets both relaxed targets."""
    best = None
    for q in CFG["threshold"]["grid_quantiles"]:
        th = float(np.quantile(known_ratio, q))
        ka = float(np.mean(known_ratio >= th))
        ur = float(np.mean(unknown_ratio < th))
        if ka + 1e-12 < CFG["targets"]["known_acceptance"]:
            continue
        if ur + 1e-12 < CFG["targets"]["unknown_rejection"]:
            continue
        cand = (q, th, ka, ur)
        if best is None or cand[0] < best[0]:
            best = cand
    if best is None:
        # Fall back to the joint-closest point and record that the target was missed.
        grid = np.unique(np.quantile(np.r_[known_ratio, unknown_ratio], np.linspace(0, 1, 801)))
        rows = [(float(t), float(np.mean(known_ratio >= t)), float(np.mean(unknown_ratio < t))) for t in grid]
        gap = [(1 - ka) + (1 - ur) for _, ka, ur in rows]
        i = int(np.argmin(gap))
        th, ka, ur = rows[i]
        return {"threshold": th, "known_acceptance": ka, "unknown_rejection": ur,
                "quantile": None, "target_met": False,
                "selection_source": "Train-only fallback: joint-closest point"}
    q, th, ka, ur = best
    return {"threshold": th, "known_acceptance": ka, "unknown_rejection": ur,
            "quantile": float(q), "target_met": True,
            "selection_source": "Train-only known/calibration blocks; no Test access"}


def load_test(protos, keys):
    order = pd.read_csv(ROOT / CFG["data"]["test_csv"], usecols=["Sample"])
    mapping = pd.read_csv(ROOT / CFG["data"]["test_metadata"], usecols=["sample", "path", "session", "task"])
    if len(order) != CFG["data"]["expected_test_rows"] or len(mapping) != len(order):
        raise ValueError("Test mapping mismatch")
    root = ROOT / CFG["data"]["train_root"]
    e36 = load_e36_config()
    e65cfg = json.loads((ROOT / "research/E65_correlation_spd/config.json").read_text(encoding="utf-8"))
    ratios = np.full(len(mapping), np.nan, dtype=np.float32)
    preds = np.zeros(len(mapping), dtype=np.int64)
    for st in range(0, len(mapping), 64):
        stop = min(st + 64, len(mapping))
        paths = [root / p for p in mapping.path.iloc[st:stop]]
        arr = np.stack([np.asarray(loadmat(p, variable_names=["data"])["data"], dtype=np.float32).T for p in paths])
        x = torch.as_tensor(arr, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            feat = run_e65.gpu_correlation_logspd(x, e36, e65cfg).cpu().numpy()
        r, p = distance_ratio(feat, protos, keys, mapping.task.iloc[st:stop].to_numpy())
        ratios[st:stop] = r
        preds[st:stop] = p
    return order, mapping, ratios, preds


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("E102 requires CUDA")
    if (OUT / "predictions.csv").exists():
        raise RuntimeError("E102 Test audit already performed; refuse rerun")
    meta = pd.read_csv(TRAIN_META)
    feat = np.load(TRAIN_FEATURES)
    if len(meta) != 11304 or feat.shape != (len(meta), 855) or not np.isfinite(feat).all():
        raise ValueError("Train metadata/feature invariant failed")
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)
    pseudo = np.asarray(CFG["calibration"]["pseudo_unknown_subjects"])
    known = np.setdiff1d(CLASSES, pseudo)
    fit = np.isin(subjects, known) & (f <= CFG["calibration"]["fit_fraction"])
    cal_known = np.flatnonzero(np.isin(subjects, known) & (f > .6) & (f <= .8))
    cal_unknown = np.flatnonzero(np.isin(subjects, pseudo) & (f > .6) & (f <= .8))

    protos, keys = build_task_prototypes(feat, subjects, tasks, fit, known)
    ratio_all, pred_all = distance_ratio(feat, protos, keys, tasks)
    if not np.isfinite(ratio_all[cal_known]).all() or not np.isfinite(ratio_all[cal_unknown]).all():
        raise ValueError("non-finite calibration ratio")
    rule = choose_threshold(ratio_all[cal_known], ratio_all[cal_unknown])
    auc = float(roc_auc_score(np.r_[np.ones(len(cal_known)), np.zeros(len(cal_unknown))],
                              np.r_[ratio_all[cal_known], ratio_all[cal_unknown]]))
    cal_pred = np.where(ratio_all[cal_known] >= rule["threshold"], pred_all[cal_known], 0)
    f1 = float(f1_score(subjects[cal_known], cal_pred, labels=known, average="macro", zero_division=0))

    torch.save({"prototypes": protos, "keys": keys, "rule": rule, "classes": CLASSES,
                "config": CFG}, OUT / "model_checkpoint.pt")
    result = {"status": "frozen_before_test", "experiment": CFG["experiment"],
              "train_only_metrics": {"AUC": auc, "F1": f1,
                                     "N_known": int(len(cal_known)),
                                     "N_pseudo_unknown": int(len(cal_unknown)),
                                     "rule": rule},
              "test_metrics": {k: "N/A" for k in ["AUC", "F1", "C", "RA", "ASafeR", "U", "Official Score"]},
              "protocol_checks": {"train_session1_only": True, "test_accessed": False,
                                  "gpu": torch.cuda.get_device_name(0)}}

    order, mapping, ratios, preds = load_test(protos, keys)
    labels = np.where(ratios >= rule["threshold"], preds, 0)
    if len(labels) != len(order):
        raise ValueError("prediction count mismatch")
    pd.DataFrame({"Sample": order.Sample, "Label": labels}).to_csv(OUT / "predictions.csv", index=False)

    sess = mapping.session.astype(str).str.casefold()
    audit = {"N": int(len(labels)), "predictions_complete_before_session_audit": True,
             "session_used_by_inference": False}
    for name, aliases in (("S1", ("session1", "s1", "1")), ("S2", ("session2", "s2", "2"))):
        m = sess.isin(aliases).to_numpy()
        audit[f"{name}_N"] = int(m.sum())
        audit[f"{name}_rejection_rate"] = float(np.mean(labels[m] == 0))
        audit[f"{name}_acceptance_rate"] = float(np.mean(labels[m] != 0))
    audit["S1_target"] = CFG["targets"]["unknown_rejection"]
    audit["S2_target"] = CFG["targets"]["known_acceptance"]
    audit["S1_pass"] = audit["S1_rejection_rate"] >= CFG["targets"]["unknown_rejection"]
    audit["S2_pass"] = audit["S2_acceptance_rate"] >= CFG["targets"]["known_acceptance"]
    result["audit"] = audit
    result["protocol_checks"].update({"test_accessed": True, "test_read_once_after_freeze": True})
    result["status"] = "complete_one_shot_frozen_audit"
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    (OUT / "report.md").write_text(
        "# E102 task-conditioned distance-ratio gate\n\n"
        "任务条件化 correlation-logSPD 原型 + distance ratio（次近/最近身份距离）。"
        "原型与阈值仅由 Train/session1 确定，Test 在冻结后一次性读取。\n\n"
        f"## Train-only\n\nAUC={auc:.6f}，F1={f1:.6f}；阈值={rule['threshold']:.6f}，"
        f"known acceptance={rule['known_acceptance']:.6f}，unknown rejection={rule['unknown_rejection']:.6f}，"
        f"目标达成={rule['target_met']}。\n\n"
        "## 项目 Test\n\n真实 Test 标签不可用：AUC/F1/C/RA/ASafeR/U/Official Score=N/A。\n\n"
        f"| 项目 | 结果 | 目标 | 是否达标 |\n|---|---:|---:|---|\n"
        f"| S1 拒识率 | {audit['S1_rejection_rate']:.6f} | ≥{audit['S1_target']:.2f} | {audit['S1_pass']} |\n"
        f"| S2 接受率 | {audit['S2_acceptance_rate']:.6f} | ≥{audit['S2_target']:.2f} | {audit['S2_pass']} |\n\n"
        "predictions.csv 完整落盘后才读取 session 审计。\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
