"""E120: domain adaptation on the correlation-logSPD features.

E116 and E119 together show that the Test S1-vs-S2 diagnostic AUC tops out near
0.68-0.70 across 14 score forms. So the gap is not in the score function. This
experiment tests the one remaining mechanical lever that changes the geometry
rather than the decision rule: domain adaptation on the SPD features.

Variants (all fitted on the Train fit block only, fraction <= 0.6, no Test data):
  D0  baseline           task-conditioned prototypes, max cosine
  D1  per-task z-score   standardise each task's features with Train mean/std
  D2  WCCN               whiten by within-class covariance (shrinkage 0.05)
  D3  WCCN + per-task z  both
  D4  PCA-64 + WCCN      reduce to 64 dims first, then whiten
  D5  D4 + cosine ratio  distance ratio on the reduced space
  D6  euclidean-align    per-task centre the features on the Train task mean,
                         then renormalise each trial to unit norm

Report: Test S1-vs-S2 diagnostic AUC and the best joint operating point.
Session is used only for measurement.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "research" / "E65_correlation_spd"))
sys.path.insert(0, str(ROOT / "research" / "E36_pure_eeg"))
import run_e65
from run_e36 import load_config as load_e36_config

CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = np.arange(1, 21, dtype=int)
TRAIN_META = ROOT / CFG["data"]["train_metadata"]
TRAIN_FEATURES = ROOT / CFG["data"]["train_features"]


def frac(meta):
    mx = meta.groupby(["subject", "task"]).trial.transform("max").to_numpy(float)
    return meta.trial.to_numpy(float) / np.maximum(mx, 1.0)


def build_prototypes(feat, subjects, tasks, fit_mask, classes):
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


def score_matrix(feat, protos, keys, tasks):
    n = len(feat)
    out = np.full((n, len(keys)), -np.inf, dtype=np.float32)
    for task in CFG["tasks"]:
        idx = np.flatnonzero(tasks == task)
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(idx) == 0 or len(cols) < 2:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        for st in range(0, len(idx), 1024):
            chunk = idx[st:st + 1024]
            q = torch.as_tensor(np.array(feat[chunk], dtype=np.float32), device=DEVICE)
            d = torch.cdist(q, p).cpu().numpy()
            out[np.ix_(chunk, cols)] = -d
    return out


def distance_ratio(feat, protos, keys, tasks):
    ratio = np.full(len(feat), np.nan, dtype=np.float32)
    for task in CFG["tasks"]:
        idx = np.flatnonzero(tasks == task)
        cols = [i for i, (t, _) in enumerate(keys) if t == task]
        if len(idx) == 0 or len(cols) < 2:
            continue
        p = torch.as_tensor(protos[cols], dtype=torch.float32, device=DEVICE)
        for st in range(0, len(idx), 1024):
            chunk = idx[st:st + 1024]
            q = torch.as_tensor(np.array(feat[chunk], dtype=np.float32), device=DEVICE)
            d = torch.cdist(q, p).cpu().numpy()
            order = np.argsort(d, axis=1)
            d1 = d[np.arange(len(d)), order[:, 0]]
            d2 = d[np.arange(len(d)), order[:, 1]]
            ratio[chunk] = d2 / np.maximum(d1, 1e-9)
    return ratio


def wccn_matrix(feat, subjects, fit_rows, shrinkage=0.05):
    """Within-class covariance whitening, fitted on the Train fit block."""
    dim = feat.shape[1]
    Sw = np.zeros((dim, dim), dtype=np.float64)
    total = 0
    for c in np.unique(subjects[fit_rows]):
        sel = fit_rows & (subjects == c)
        x = np.asarray(feat[sel], dtype=np.float64)
        if len(x) < 2:
            continue
        xc = x - x.mean(0)
        Sw += xc.T @ xc
        total += len(x)
    Sw /= max(total, 1)
    mu = float(np.trace(Sw) / dim)
    Sw = (1 - shrinkage) * Sw + shrinkage * mu * np.eye(dim)
    w, V = np.linalg.eigh(Sw)
    w = np.clip(w, 1e-6, None)
    return (V / np.sqrt(w)) @ V.T


def pca_matrix(feat, fit_rows, dim=64):
    x = np.asarray(feat[fit_rows], dtype=np.float64)
    mu = x.mean(0)
    xc = x - mu
    cov = xc.T @ xc / max(len(x) - 1, 1)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(-w)[:dim]
    return mu, V[:, order]


def task_standardize(train, test, train_tasks, test_tasks):
    """Per-task z-score using Train statistics."""
    tr = np.asarray(train, dtype=np.float32).copy()
    te = np.asarray(test, dtype=np.float32).copy()
    for task in CFG["tasks"]:
        m = train_tasks == task
        if m.sum() < 10:
            continue
        mu = tr[m].mean(0)
        sd = tr[m].std(0) + 1e-6
        tr[m] = (tr[m] - mu) / sd
        mt = test_tasks == task
        if mt.any():
            te[mt] = (te[mt] - mu) / sd
    return tr, te


def separability(score, is_s1):
    s1 = score[is_s1]
    s2 = score[~is_s1]
    auc = float(roc_auc_score(np.r_[np.zeros(len(s1)), np.ones(len(s2))], np.r_[s1, s2]))
    grid = np.unique(np.quantile(score, np.linspace(0, 1, 401)))
    ka = np.asarray([np.mean(s1 < t) for t in grid])
    ur = np.asarray([np.mean(s2 >= t) for t in grid])
    m80 = ka >= 0.80
    return {
        "diagnostic_AUC": auc,
        "best_S2_accept_at_S1_rej_0.80": float(ur[m80].max()) if m80.any() else 0.0,
        "best_joint_min": float(np.max(np.minimum(ka, ur))),
    }


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("E120 requires CUDA")
    meta = pd.read_csv(TRAIN_META)
    feat = np.load(TRAIN_FEATURES)
    if len(meta) != CFG["data"]["expected_trials"] or feat.shape != (len(meta), 855):
        raise ValueError("Train metadata/feature invariant failed")
    subjects = meta.subject.to_numpy(int)
    tasks = meta.task.to_numpy()
    f = frac(meta)
    fit_rows = f <= CFG["calibration"]["fit_fraction"]

    mapping = pd.read_csv(ROOT / CFG["data"]["test_metadata"],
                          usecols=["sample", "path", "session", "task"])
    is_s1 = mapping.session.astype(str).str.casefold().isin(["session1", "s1", "1"]).to_numpy()
    test_tasks = mapping.task.to_numpy()

    # ---- Test features
    root = ROOT / CFG["data"]["train_root"]
    e36 = load_e36_config()
    e65cfg = json.loads((ROOT / "research" / "E65_correlation_spd/config.json").read_text(encoding="utf-8"))
    parts = []
    for st in range(0, len(mapping), 64):
        stop = min(st + 64, len(mapping))
        paths = [root / p for p in mapping.path.iloc[st:stop]]
        arr = np.stack([np.asarray(loadmat(p, variable_names=["data"])["data"],
                                   dtype=np.float32).T for p in paths])
        x = torch.as_tensor(arr, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            parts.append(run_e65.gpu_correlation_logspd(x, e36, e65cfg).cpu().numpy())
        print(f"E120 test chunk {stop}/{len(mapping)}", flush=True)
    test_feat = np.concatenate(parts)

    results = {}

    def evaluate(name, train_feat, test_feat_use, ratio_mode=False):
        protos, keys = build_prototypes(train_feat, subjects, tasks, fit_rows, CLASSES)
        if ratio_mode:
            sc = distance_ratio(test_feat_use, protos, keys, test_tasks)
        else:
            sc = score_matrix(test_feat_use, protos, keys, test_tasks).max(1)
        results[name] = separability(np.asarray(sc, dtype=np.float64), is_s1)

    # D0 baseline
    evaluate("D0_baseline_max", feat, test_feat)
    evaluate("D0b_baseline_ratio", feat, test_feat, ratio_mode=True)

    # D1 per-task z-score
    tr_z, te_z = task_standardize(feat, test_feat, tasks, test_tasks)
    evaluate("D1_pertask_z_max", tr_z, te_z)

    # D2 WCCN
    W = wccn_matrix(feat, subjects, fit_rows)
    tr_w = np.asarray(feat, dtype=np.float64) @ W
    te_w = np.asarray(test_feat, dtype=np.float64) @ W
    evaluate("D2_wccn_max", tr_w.astype(np.float32), te_w.astype(np.float32))

    # D3 WCCN + per-task z
    tr_wz, te_wz = task_standardize(tr_w, te_w, tasks, test_tasks)
    evaluate("D3_wccn_pertask_z_max", tr_wz, te_wz)

    # D4 PCA-64 + WCCN
    mu_pca, V = pca_matrix(feat, fit_rows, CFG["pca_dim"])
    tr_p = ((np.asarray(feat, dtype=np.float64) - mu_pca) @ V).astype(np.float32)
    te_p = ((np.asarray(test_feat, dtype=np.float64) - mu_pca) @ V).astype(np.float32)
    Wp = wccn_matrix(tr_p, subjects, fit_rows)
    tr_pw = (tr_p.astype(np.float64) @ Wp).astype(np.float32)
    te_pw = (te_p.astype(np.float64) @ Wp).astype(np.float32)
    evaluate("D4_pca64_wccn_max", tr_pw, te_pw)
    evaluate("D5_pca64_wccn_ratio", tr_pw, te_pw, ratio_mode=True)

    # D6 per-task centring + per-trial renormalisation
    tr_c = np.asarray(feat, dtype=np.float32).copy()
    te_c = np.asarray(test_feat, dtype=np.float32).copy()
    for task in CFG["tasks"]:
        m = fit_rows & (tasks == task)
        if m.sum() < 10:
            continue
        mu = tr_c[m].mean(0)
        tr_c[tasks == task] -= mu
        mt = test_tasks == task
        if mt.any():
            te_c[mt] -= mu
    tr_c /= np.maximum(np.linalg.norm(tr_c, axis=1, keepdims=True), 1e-9)
    te_c /= np.maximum(np.linalg.norm(te_c, axis=1, keepdims=True), 1e-9)
    evaluate("D6_pertask_center_renorm_max", tr_c, te_c)

    result = {"experiment": CFG["experiment"], "diagnostic_only": True,
              "warning": "Session is used only to measure separability, never to build a score.",
              "N_S1": int(is_s1.sum()), "N_S2": int((~is_s1).sum()), "scores": results}
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    lines = ["# E120 Domain adaptation on correlation-logSPD (diagnostic only)", "",
             "E116/E119 已用 14 个分数证明 Test S1-vs-S2 AUC 上限约 0.68–0.70。"
             "本实验测试唯一剩下的机械杠杆：改变特征几何（而非决策规则）的域适应。"
             "所有变换只在 Train 拟合块上拟合，未用 Test 统计。", "",
             "| 变体 | 诊断 AUC | S1≥0.80 时 S2 接受 | 最佳均衡点 |",
             "|---|---:|---:|---:|"]
    for name in sorted(results):
        s = results[name]
        lines.append(f"| {name} | {s['diagnostic_AUC']:.4f} | "
                     f"{s['best_S2_accept_at_S1_rej_0.80']:.4f} | {s['best_joint_min']:.4f} |")
    lines += ["", "项目 Test AUC/F1/C/RA/ASafeR/U/Official Score = N/A（无真实标签）。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
